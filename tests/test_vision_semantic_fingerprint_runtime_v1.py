from __future__ import annotations

import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
SDK = importlib.import_module("vision_semantic_fingerprint_sdk_v1")
RUN = importlib.import_module("run_vision_semantic_fingerprint_v1")
BASE = importlib.import_module("run_vision_semantic_fingerprint_base_v1")
FIXTURE = importlib.import_module("test_vision_semantic_sdk_v1")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _module_path(module: Any) -> Path:
    value = module.__file__
    assert isinstance(value, str)
    return Path(value).resolve()


def _validated(tmp_path: Path) -> dict[str, Any]:
    output = tmp_path / "output"
    output.mkdir()
    return {
        "job_path": tmp_path / "job.json",
        "output_directory": output,
        "job_sha256": "a" * 64,
        "job": {"mode": "semantic_diagnostic_only", "source": {"files_lf": {}}},
    }


def test_sdk_seals_four_callback_fingerprints_before_base_report(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    validated = _validated(tmp_path)
    empty = FIXTURE._payload(empty=True)
    valid = FIXTURE._payload()

    def seal(report: dict[str, Any]) -> None:
        assert (validated["output_directory"] / SDK.SIDECAR_NAME).is_file()
        log.append("base_report_seal")

    result = SDK.run_one_view(
        validated,
        FIXTURE._App(log),
        seal_report=seal,
        dependencies=FIXTURE._dependencies(
            log,
            payloads=[empty, valid],
            callbacks_per_event=2,
        ),
    )
    sidecar = _read(validated["output_directory"] / SDK.SIDECAR_NAME)
    assert sidecar["callbacks_observed"] == sidecar["callbacks_preserved"] == 4
    assert sidecar["overflow"] is False
    assert sidecar["comparison"] == "equal"
    assert sidecar["used_for_decision"] is False
    assert sidecar["callbacks"][0]["fingerprint"] is None
    assert sidecar["callbacks"][0]["error_type"] == "builtins.ValueError"
    assert all(
        row["fingerprint"] is None for row in sidecar["callbacks"][:2]
    )
    assert all(
        row["fingerprint"]["whole_payload_sha256"] is not None
        for row in sidecar["callbacks"][2:]
    )
    assert [row["ordinal"] for row in sidecar["callbacks"]] == [0, 1, 2, 3]
    assert result["no_capture_approved"] is True
    assert log.index("writer_detach") < log.index("base_report_seal")
    assert log.index("base_report_seal") < log.index("render_product_destroy")


def test_sdk_reports_unequal_only_within_same_frame_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    validated = _validated(tmp_path)
    original = SDK.FINGERPRINT.fingerprint
    calls = 0

    def alternating(payload: Any) -> dict[str, Any]:
        nonlocal calls
        result = original(payload)
        calls += 1
        result["whole_payload_sha256"] = ("a" if calls % 2 else "b") * 64
        return result

    monkeypatch.setattr(SDK.FINGERPRINT, "fingerprint", alternating)
    SDK.run_one_view(
        validated,
        FIXTURE._App(log),
        seal_report=lambda _report: None,
        dependencies=FIXTURE._dependencies(
            log,
            payloads=[FIXTURE._payload(), FIXTURE._payload()],
            callbacks_per_event=2,
        ),
    )
    sidecar = _read(validated["output_directory"] / SDK.SIDECAR_NAME)
    assert sidecar["comparison"] == "unequal"
    assert sidecar["used_for_decision"] is False


def test_comparison_does_not_compare_different_frame_groups() -> None:
    fingerprint_a = SDK.FINGERPRINT.fingerprint(FIXTURE._payload())
    fingerprint_b = dict(fingerprint_a)
    fingerprint_b["whole_payload_sha256"] = "e" * 64
    rows = [
        {
            "comparison_group_sha256": "1" * 64,
            "fingerprint": fingerprint_a,
        },
        {
            "comparison_group_sha256": "2" * 64,
            "fingerprint": fingerprint_b,
        },
    ]
    assert SDK._comparison(rows) == "insufficient"


def _fingerprinted_row(payload: dict[str, Any], *, window: str) -> dict[str, Any]:
    fingerprint = SDK.FINGERPRINT.fingerprint(payload)
    return {
        "comparison_group_sha256": SDK._group_sha(
            window, SDK.DIAGNOSTIC._raw_frame_time(payload), fingerprint
        ),
        "fingerprint": fingerprint,
    }


def _render_product(payload: dict[str, Any]) -> str:
    return next(iter(payload["annotators"]["rgb"]))


@pytest.mark.parametrize(
    "mutation",
    [
        "system_time",
        "simulation_time",
        "trigger",
        "rgb",
        "camera",
    ],
)
def test_non_identity_payload_changes_are_unequal_in_same_group(mutation: str) -> None:
    first = FIXTURE._payload()
    second = deepcopy(first)
    rp = _render_product(second)
    if mutation == "system_time":
        second["annotators"]["IsaacReadTimes"][rp]["systemTime"] += 1.0
    elif mutation == "simulation_time":
        second["annotators"]["IsaacReadTimes"][rp]["simulationTime"] += 1.0
    elif mutation == "trigger":
        second["trigger_outputs"] = {"changed": True}
    elif mutation == "rgb":
        second["annotators"]["rgb"][rp]["data"][0, 0, 0] = 1
    else:
        second["annotators"]["camera"][rp] = "/World/ChangedCamera"
    rows = [
        _fingerprinted_row(first, window="event_1_running"),
        _fingerprinted_row(second, window="event_1_running"),
    ]
    assert rows[0]["comparison_group_sha256"] == rows[1]["comparison_group_sha256"]
    assert SDK._comparison(rows) == "unequal"


@pytest.mark.parametrize(
    "mutation",
    ["window", "render_product", "frame", "rational", "reference", "writer_swh", "isaac_swh"],
)
def test_identity_changes_are_not_compared(mutation: str) -> None:
    first = FIXTURE._payload()
    second = deepcopy(first)
    first_window = second_window = "event_1_running"
    rp = _render_product(second)
    if mutation == "window":
        second_window = "event_0_running"
    elif mutation == "render_product":
        for values in second["annotators"].values():
            values["RenderProduct_1"] = values.pop(rp)
    elif mutation == "frame":
        second["annotators"]["IsaacReadTimes"][rp]["frameNumber"] += 1
    elif mutation == "rational":
        second["annotators"]["IsaacReadTimes"][rp][
            "rationalTimeOfSimNumerator"
        ] += 1
    elif mutation == "reference":
        second["reference_time"] = (56, 30)
    elif mutation == "writer_swh":
        second["swhFrameNumber"] += 1
    else:
        second["annotators"]["IsaacReadTimes"][rp]["swhFrameNumber"] += 1
    rows = [
        _fingerprinted_row(first, window=first_window),
        _fingerprinted_row(second, window=second_window),
    ]
    assert rows[0]["comparison_group_sha256"] != rows[1]["comparison_group_sha256"]
    assert SDK._comparison(rows) == "insufficient"


def test_sdk_over_four_callbacks_fails_after_preserving_sidecar(tmp_path: Path) -> None:
    log: list[str] = []
    validated = _validated(tmp_path)
    with pytest.raises(FIXTURE.S.SDKDiagnosticFailure, match="semantic SDK diagnostic"):
        SDK.run_one_view(
            validated,
            FIXTURE._App(log),
            seal_report=lambda _report: pytest.fail("overflow는 base report를 봉인하지 않는다"),
            dependencies=FIXTURE._dependencies(
                log,
                callbacks_per_event=2,
                attach_payload=FIXTURE._payload(),
                detach_payload=FIXTURE._payload(),
            ),
        )
    sidecar = _read(validated["output_directory"] / SDK.SIDECAR_NAME)
    assert sidecar["callbacks_observed"] == 6
    assert sidecar["callbacks_preserved"] == 4
    assert sidecar["overflow"] is True


def test_valid_counterpart_without_hash_fails_closed_and_keeps_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    validated = _validated(tmp_path)

    def reject(_payload: Any) -> dict[str, Any]:
        raise TypeError("unsupported")

    monkeypatch.setattr(SDK.FINGERPRINT, "fingerprint", reject)
    with pytest.raises(FIXTURE.S.SDKDiagnosticFailure):
        SDK.run_one_view(
            validated,
            FIXTURE._App(log),
            seal_report=lambda _report: pytest.fail("hash가 없으면 봉인하지 않는다"),
            dependencies=FIXTURE._dependencies(
                log, payloads=[FIXTURE._payload(), FIXTURE._payload()]
            ),
        )
    sidecar = _read(validated["output_directory"] / SDK.SIDECAR_NAME)
    assert sidecar["callbacks_observed"] == 2
    assert all(row["fingerprint"] is None for row in sidecar["callbacks"])
    assert all(row["error_type"] == "builtins.TypeError" for row in sidecar["callbacks"])


def _source_validated(tmp_path: Path) -> dict[str, Any]:
    return {
        "job_path": tmp_path / "job.json",
        "job_sha256": "b" * 64,
        "output_directory": tmp_path / "output",
        "job": {
            "source": {
                "files_lf": {relative: "c" * 64 for relative in RUN.REQUIRED_SOURCE_FILES}
            }
        },
        "protocol": {
            "preregistration": {
                "observation_phrases": {
                    key: f"registered-{index}"
                    for index, key in enumerate(sorted(RUN.OBSERVATION_PHRASE_KEYS))
                }
            }
        },
    }


def test_source_contract_requires_all_new_and_bound_base_files(tmp_path: Path) -> None:
    validated = _source_validated(tmp_path)
    RUN._source_contract(validated)
    files = validated["job"]["source"]["files_lf"]
    del files["bench/vision_callback_fingerprint_v1.py"]
    with pytest.raises(ValueError, match="source closure"):
        RUN._source_contract(validated)


def test_source_contract_rejects_missing_or_duplicate_registered_phrase(
    tmp_path: Path,
) -> None:
    missing = _source_validated(tmp_path / "missing")
    del missing["protocol"]["preregistration"]["observation_phrases"][
        "fingerprint_unequal"
    ]
    with pytest.raises(ValueError, match="사전등록"):
        RUN._source_contract(missing)

    duplicate = _source_validated(tmp_path / "duplicate")
    phrases = duplicate["protocol"]["preregistration"]["observation_phrases"]
    phrases["fingerprint_equal"] = phrases["fingerprint_unequal"]
    with pytest.raises(ValueError, match="유일"):
        RUN._source_contract(duplicate)


def test_child_entry_adapter_changes_only_registered_entry(tmp_path: Path) -> None:
    captured: list[list[str]] = []
    adapter = RUN._ChildEntryAdapter(
        lambda command, **_kwargs: captured.append(command) or object()
    )
    command = [
        "/venv/bin/python",
        "-I",
        "-S",
        "-c",
        BASE.GATE.OWNED_WORKER_GATE,
        "/venv/bin/python",
        "-B",
        str(_module_path(BASE)),
        "--sdk-child",
        "--job",
        str((tmp_path / "job.json").resolve()),
    ]
    adapter(command, cwd="/isaaclab")
    changed = [
        index for index, (before, after) in enumerate(zip(command, captured[0], strict=True))
        if before != after
    ]
    assert changed == [7]
    assert captured[0][7] == str(_module_path(RUN))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda command: command.__setitem__(1, "-E"),
        lambda command: command.append("extra"),
        lambda command: command.__setitem__(4, "other-gate"),
        lambda command: command.__setitem__(5, "/other/python"),
        lambda command: command.__setitem__(10, "relative-job.json"),
    ],
)
def test_child_entry_adapter_rejects_any_other_command_change(
    mutate: Any, tmp_path: Path
) -> None:
    command = [
        "/venv/bin/python",
        "-I",
        "-S",
        "-c",
        BASE.GATE.OWNED_WORKER_GATE,
        "/venv/bin/python",
        "-B",
        str(_module_path(BASE)),
        "--sdk-child",
        "--job",
        str((tmp_path / "job.json").resolve()),
    ]
    mutate(command)
    with pytest.raises(ValueError, match="child entry 명령 계약"):
        RUN._ChildEntryAdapter(lambda *_args, **_kwargs: object())(command)


def test_stored_diagnostic_report_accepts_one_empty_then_three_valid_contract() -> None:
    summary = _read(
        ROOT / "evals/results/vision_semantic_sdk_dev_20261026r/summary.json"
    )
    stored = summary["observation"]["callbacks"]
    assert [row["classification"] for row in stored] == [
        "invalid", "valid", "valid", "valid"
    ]
    callbacks = [
        {
            "ordinal": row["ordinal"],
            "window": row["window"],
            "classification": row["classification"],
            "raw_frame_time": {
                    "swhFrameNumber": None,
                    "reference_time": None,
                    "isaac_read_times": {
                        "frameNumber": {
                            "type": "builtins.int",
                            "numeric": True,
                            "value": row["frame_number"],
                        }
                    },
            },
        }
        for row in stored
    ]
    envelope = {
        "diagnostic": {
            "callbacks_observed": 4,
            "callbacks_preserved": 4,
            "callbacks": callbacks,
            "observations": {"overflow": False},
        }
    }
    rows = []
    valid_fingerprint = SDK.FINGERPRINT.fingerprint(FIXTURE._payload())
    for index, callback in enumerate(callbacks):
        fingerprint = None if index == 0 else valid_fingerprint
        rows.append(
            {
                "ordinal": index,
                "window": callback["window"],
                "comparison_group_sha256": (
                    None
                    if fingerprint is None
                    else SDK._group_sha(
                        callback["window"], callback["raw_frame_time"], fingerprint
                    )
                ),
                "fingerprint": fingerprint,
                "error_type": "builtins.ValueError" if index == 0 else None,
            }
        )
    sidecar = {
        "schema": SDK.SIDECAR_SCHEMA,
        "job_sha256": "a" * 64,
        "callbacks_observed": 4,
        "callbacks_preserved": 4,
        "callback_limit": 4,
        "overflow": False,
        "callbacks": rows,
        "comparison": "equal",
        "used_for_decision": False,
        "sealed_before_base_report": True,
        "no_capture_approved": True,
    }
    SDK._validate_counterparts(sidecar, envelope)
    tampered = json.loads(json.dumps(sidecar))
    tampered["callbacks"][1]["fingerprint"]["components"].pop("rgb_data")
    with pytest.raises(ValueError, match="본문 계약"):
        SDK._validate_counterparts(tampered, envelope)

    tampered_comparison = json.loads(json.dumps(sidecar))
    tampered_comparison["comparison"] = "unequal"
    with pytest.raises(ValueError, match="비교 결과"):
        SDK._validate_counterparts(tampered_comparison, envelope)


def test_parent_writes_separate_fingerprint_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _source_validated(tmp_path)
    output = validated["output_directory"]
    output.mkdir()
    BASE.JOB.write_new(output / "result.json", {"base": True})
    BASE.JOB.write_new(tmp_path / "job.json", {"job": True})

    monkeypatch.setattr(BASE.JOB, "validate_job", lambda _path: validated)
    monkeypatch.setattr(RUN, "_source_contract", lambda _validated: None)
    monkeypatch.setattr(
        RUN,
        "_read_fingerprint_result",
        lambda _validated: {
            "comparison": "equal",
            "callbacks_observed": 4,
            "valid_fingerprints": 3,
            "used_for_decision": False,
        },
    )

    def base_run(_job: Path, *, popen: Any, **_kwargs: Any) -> dict[str, Any]:
        command = [
            "/venv/bin/python",
            "-I",
            "-S",
            "-c",
            BASE.GATE.OWNED_WORKER_GATE,
            "/venv/bin/python",
            "-B",
            str(_module_path(BASE)),
            "--sdk-child",
            "--job",
            str(tmp_path / "job.json"),
        ]
        BASE.JOB.write_new(
            tmp_path / "launch.json", {"command": command, "cwd": "/isaaclab"}
        )
        popen(command)
        BASE.JOB.write_new(tmp_path / "execution.json", {"completed": True})
        return {"completed": True}

    monkeypatch.setattr(BASE, "run_job", base_run)
    receipt = RUN.run_job(
        tmp_path / "job.json",
        popen=lambda _command, **_kwargs: object(),
        gpu_reader=list,
    )
    assert receipt["completed"] is True
    assert (tmp_path / "execution.json").is_file()
    assert (tmp_path / RUN.EXECUTION_NAME).is_file()
    assert receipt["child_entry_adapter"]["only_entry_replaced"] is True
    assert receipt["diagnostic"]["used_for_decision"] is False


@pytest.mark.skipif(RUN.os.name == "posix", reason="Windows 실제 실행 guard 회귀")
def test_default_launcher_cannot_bypass_existing_non_posix_guard(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="소유 Linux process group"):
        RUN.run_job(tmp_path / "job.json")


def test_fingerprint_postflight_cannot_exceed_base_wall_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _source_validated(tmp_path)
    validated["output_directory"].mkdir()
    BASE.JOB.write_new(tmp_path / "job.json", {"job": True})
    monkeypatch.setattr(BASE.JOB, "validate_job", lambda _path: validated)
    monkeypatch.setattr(RUN, "_source_contract", lambda _validated: None)

    def base_run(_job: Path, *, popen: Any, **_kwargs: Any) -> dict[str, Any]:
        command = [
            "/venv/bin/python", "-I", "-S", "-c", BASE.GATE.OWNED_WORKER_GATE,
            "/venv/bin/python", "-B", str(_module_path(BASE)), "--sdk-child",
            "--job", str((tmp_path / "job.json").resolve()),
        ]
        BASE.JOB.write_new(tmp_path / "launch.json", {"command": command, "cwd": "/lab"})
        popen(command)
        BASE.JOB.write_new(tmp_path / "execution.json", {"completed": True})
        return {"completed": True}

    monkeypatch.setattr(BASE, "run_job", base_run)
    readings = iter((0.0, 601.0, 601.0))
    monkeypatch.setattr(RUN.time, "monotonic", lambda: next(readings))
    receipt = RUN.run_job(
        tmp_path / "job.json",
        popen=lambda _command, **_kwargs: object(),
        gpu_reader=list,
    )
    assert receipt["completed"] is False
    assert receipt["error_type"] == "TimeoutError"
    assert receipt["wall_s"] == 601.0
