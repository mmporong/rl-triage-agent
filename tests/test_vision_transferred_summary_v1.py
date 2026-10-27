from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vision_transferred_summary_v1_test", ROOT / "evals/vision_transferred_summary_v1.py")
assert SPEC is not None and SPEC.loader is not None
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def test_real_archive_recomputes_four_unique_source_labels_when_available() -> None:
    archive = Path(
        "D:/rl-triage-artifacts/vision-goal-20261025/retrieved/20261026T082752839592Z"
    )
    if not archive.is_dir():
        pytest.skip("local 437-file source archive is unavailable")
    protocol = S.PROBE.validate_protocol(ROOT / "bench/protocols/vision_transferred_probe_v1.json")
    rows = [
        {"seed_group": group["group"], "source_role": S.ARM_ROLE[arm],
         "checkpoint_sha256": group["checkpoints"][arm]["sha256"]}
        for group in protocol["groups"] for arm in S.ARM_ROLE
    ]
    summary = S.read(ROOT / "evals/results/vision_source_prefix_dev_20261026x/summary.json")
    result = S.source_labels(archive, summary, rows)
    assert result["archive_verification"]["files_verified"] == 437
    assert [result["labels"][group][role]["label"]
            for group in S.METHOD.SEED_GROUPS for role in S.METHOD.ROLES] == [
                "healthy", "unhealthy", "healthy", "unhealthy"]
    assert result["fixed_eval_sha256"] == sorted([
        "92ed5b026480f366dfdfdee53b6b06327da6cdd3c166680b11d37c79996b736c",
        "e8832fbb682abe1db0a3f74c96a02f3d52f00acabb3e618074a34267e10b6274",
        "b1cad4a033edb3bc5785b786d98ac508e1e223935b4022f2cf631dc08cc78ef7",
        "1c36ec2f46ea953d06a443a560c15aeeb772f68147258ad0dc9f249ccabb2a1d",
    ])
    assert result["evaluation_raw_sha256"] == [
        "a106e84e97cf6a04281d21d2336b90e0ce2a8d4384c9c8729da9988ba8b10b42",
        "a5e4827f4e7c037ed845e9178dbd2c0ade23e36eb73c210b1dfa1a73614c58af",
    ]


def test_probe_support_uses_new_validate_pair_without_portable_repackaging(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol_path = tmp_path / "probe.json"
    _write(protocol_path, {"bound": True})
    checkpoints = {group: f"{index}" * 64
                   for index, group in enumerate(S.METHOD.SEED_GROUPS, start=1)}
    protocol = {
        "execution_root": "runtime/probes",
        "groups": [{"group": group, "checkpoints": {"F1": {"sha256": checkpoints[group]}}}
                   for group in S.METHOD.SEED_GROUPS],
    }
    monkeypatch.setattr(S.PROBE, "validate_protocol", lambda *_args, **_kwargs: dict(protocol))
    monkeypatch.setattr(S.PROBE, "verify_repository_binding", lambda *_args: None)
    monkeypatch.setattr(S.PROBE, "verify_source_inputs", lambda *_args: {"verified": True})
    monkeypatch.setattr(S.EXECUTION, "_portable_probe",
                        lambda *_args, **_kwargs: pytest.fail("legacy portable probe must not run"))
    called = []

    def validate(_root: Path, _protocol: dict, _sources: dict, stage: Path) -> dict:
        group = stage.name
        called.append(group)
        return {"schema": S.PROBE.RESULT_SCHEMA, "status": "complete", "group": group,
                "seed": S.GROUP_SEED[group], "probe": "P_reward",
                "outcomes": {"NONE": "normal", "F1": "abnormal"},
                "measurements": {}, "gpu_s": 3.0, "authorization": {}, "ledger": {},
                "descriptors": {}, "usable": True}

    monkeypatch.setattr(S.PROBE, "validate_pair", validate)
    for group in S.METHOD.SEED_GROUPS:
        stage = tmp_path / "runtime/probes" / group
        verified = validate(tmp_path, protocol, {}, stage)
        called.clear()
        _write(stage / "result.json", verified)
        _write(stage / "authorization.json", {
            "request": {"checkpoints": {"F1": checkpoints[group]}}})
        (stage / "approval.ledger.jsonl").write_text('{"event":"receipt"}\n', encoding="utf-8")
    result = S.probe_support(tmp_path, protocol_path, tmp_path)
    assert called == list(S.METHOD.SEED_GROUPS)
    assert result["status"] == "complete"
    assert all(row["approval_consumed"] is True and row["usable"] is True
               and row["outcome"] == "abnormal" for row in result["support"].values())


def test_selection_results_calls_transferred_boundaries_and_separates_validity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, locations = tmp_path / "selection.json", tmp_path / "locations.json"
    _write(protocol, {}); _write(locations, {})
    calls = []
    prepared = {"plan": {"fixture": True}, "protocol": {"rows": []}}
    monkeypatch.setattr(S.SELECTION, "prepare", lambda *_args, **_kwargs: calls.append("prepare") or prepared)
    monkeypatch.setattr(S.SELECTION, "inspect", lambda *_args, **_kwargs: calls.append("inspect") or {
        "status": "complete", "api": {"cap": 72, "remaining": 48}})
    monkeypatch.setattr(S.CALL, "_plan", lambda *_args: (tmp_path, tmp_path / "stage", {"rows": []}))
    rows = [{"model_status": "valid"} for _ in range(18)] + [
        {"model_status": "semantic_invalid"} for _ in range(4)] + [
        {"model_status": "infrastructure_failed"}, {"model_status": "unknown_pending"}]
    monkeypatch.setattr(S.EXECUTION, "project_results",
                        lambda *_args: calls.append("project_results") or rows)
    monkeypatch.setattr(S.EXECUTION, "_usage", lambda *_args: {"requests": 24, "tokens": {"sum": 12}})
    result = S.selection_results(protocol, "a" * 64, "b" * 40, locations,
                                 root=tmp_path, verify_raw_rgb=False)
    assert calls == ["prepare", "inspect", "project_results"]
    assert result["api_usage"]["requests"] == 24
    assert result["validity"] == {"denominator": 24, "terminal": 23, "valid": 18,
                                  "semantic_invalid": 4, "infrastructure_failed": 1,
                                  "unknown_pending": 1, "valid_rate": 0.75}


def test_missing_inputs_return_hold_without_inventing_results(tmp_path: Path) -> None:
    result = S.summarize(
        tmp_path / "selection.json", "a" * 64, "b" * 40,
        tmp_path / "locations.json", tmp_path / "archive", tmp_path / "probe.json",
        root=tmp_path, verify_raw_rgb=False,
    )
    assert result["status"] == "HOLD"
    assert result["reason"] == "inputs_insufficient"
    assert result["missing"] == ["selection_protocol", "input_locations", "source_archive", "probe_protocol"]
    assert result["comparison_summary"] is None
    assert result["probe_support"] is None
