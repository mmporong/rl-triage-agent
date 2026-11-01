# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import prepare_lineage_five_policy_v1 as M


def _write(path: Path, value: object) -> None:
    M._write_new(path, value)


def _fixture(tmp_path: Path) -> tuple[dict[str, Path], M.FixtureHooks]:
    index = tmp_path / "index.json"
    study = tmp_path / "study.json"
    batch = tmp_path / "same-play-result.json"
    template = tmp_path / "component-template.json"
    lab = tmp_path / "IsaacLab"
    protocol_dir = tmp_path / "public"
    private_root = tmp_path / "private"
    source_ids = [f"source-{number:02d}" for number in range(1, 21)]
    cases = []
    metadata_sources: dict[str, dict[str, Any]] = {}
    for number, source_id in enumerate(source_ids, 1):
        metadata_sources[f"job-{number:02d}"] = {"public_id": source_id}
    for number in range(1, 11):
        cases.append(
            {
                "case_id": f"case-{number:02d}",
                "source_job": f"job-{number * 2 - 1:02d}",
                "reference_job": f"job-{number * 2:02d}",
            }
        )
    public = {
        "sources": {
            source_id: {
                "checkpoint_sha256": f"{number:064x}",
                "telemetry_sha256": f"{number + 20:064x}",
            }
            for number, source_id in enumerate(source_ids, 1)
        },
        "case_ids": [row["case_id"] for row in cases],
    }
    metadata = {"sources": metadata_sources, "cases": cases}
    _write(index, {"fixture": True})
    _write(
        study,
        {
            "schema": "agentic_five_type_evaluation_design_v1",
            "case_plan": cases,
        },
    )
    pairs = {}
    for row in cases:
        pair = tmp_path / "pairs" / f"{row['case_id']}.json"
        _write(pair, {"case_id": row["case_id"]})
        pairs[row["case_id"]] = M._private_ref(pair)
    _write(
        batch,
        {
            "schema": "lineage_same_play_input_batch_result_v1",
            "status": "complete",
            "sdk_intents": 19,
            "api": 0,
            "source_artifacts": 20,
            "pairs": pairs,
            "parent_wall_s": 1.0,
            "diagnosis": False,
            "recovery": False,
        },
    )
    for name in M.HOST.BASE_PROBE.RUNTIME_PINS:
        runtime = lab / name
        runtime.parent.mkdir(parents=True, exist_ok=True)
        runtime.write_text("runtime\n", encoding="utf-8")
    launcher = lab / "_isaac_sim/python.bat"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("@echo off\n", encoding="utf-8")
    _write(
        template,
        {
            "schema": M.P1.PROTOCOL_SCHEMA,
            "source_sha256_lf": {
                "evals/lineage_probe_backend_v1.py": M._sha(
                    ROOT / "evals/lineage_probe_backend_v1.py", lf=True
                )
            },
            "runtime_source_sha256_lf": {
                name: "0" * 64 for name in M.HOST.BASE_PROBE.RUNTIME_PINS
            },
            "watchdogs_s": {"legacy": 480, "optimizer": 480},
            "isaac_python_relative": "_isaac_sim/python.bat",
            "scope": "fixture template",
        },
    )

    def pair_load(path: Path, _index: Path) -> dict[str, Any]:
        case_id = M._read(path)["case_id"]
        row = next(item for item in cases if item["case_id"] == case_id)
        bindings = {}
        for role, job_key in (("case", "source_job"), ("reference", "reference_job")):
            source_id = metadata_sources[row[job_key]]["public_id"]
            bindings[role] = public["sources"][source_id]
        return {
            "schema": M.DRIVER.SOURCE_SCHEMA,
            "case_id": case_id,
            "packet": {"input_bindings": bindings},
            "backend_packet": {"fixture": True},
            "source_binding": bindings,
            "providers": None,
            "visual_evidence": None,
            "qualification": {"label": "fixture"},
            "approval_granted": False,
            "api_authorized": False,
        }

    def bridge_collect(_index: Path, out: Path, *, source_id: str) -> dict[str, Any]:
        binding = public["sources"][source_id]
        value = {"source_id": source_id, "binding": binding}
        _write(out, value)
        return value

    def bridge_checked(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        return M._read(path), {"fixture": True}

    def protocol_value(
        _study: Path, sources: list[dict[str, Any]], *, extra_sources: tuple[str, ...]
    ) -> dict[str, Any]:
        assert len(sources) == 10
        return {
            "schema": M.DRIVER.SCHEMA,
            "case_order": [source["case_id"] for source in sources],
            "source_digests": {
                source["case_id"]: M.B.digest(source) for source in sources
            },
            "sources_lf": {
                name: M._sha(ROOT / name, lf=True) for name in extra_sources
            },
        }

    hooks = M.FixtureHooks(
        input_checked=lambda _path: (public, metadata),
        pair_load=pair_load,
        bridge_collect=bridge_collect,
        bridge_checked=bridge_checked,
        source_checked=lambda source: source,
        protocol_value=protocol_value,
        bridge_protocol=lambda component: {
            "schema": M.HOST.PROTOCOL_SCHEMA,
            "component": {
                "path": component.name,
                "sha256_lf": M._sha(component, lf=True),
            },
        },
        view_value=lambda: M._read(M.VIEW_PATH),
        view_verify=lambda _path, _anchor: M._read(M.VIEW_PATH),
        anchor_sources=lambda _anchor, _names: None,
    )
    paths = {
        "index": index,
        "study": study,
        "batch": batch,
        "template": template,
        "lab": lab,
        "protocol_dir": protocol_dir,
        "private_root": private_root,
    }
    return paths, hooks


def test_prepare_builds_twenty_bridges_ten_sources_and_public_path_safe_protocols(
    tmp_path: Path,
) -> None:
    paths, hooks = _fixture(tmp_path)
    result = M.prepare(
        paths["index"],
        paths["batch"],
        paths["study"],
        paths["template"],
        paths["lab"],
        source_anchor="a" * 40,
        tag="fixture_r3",
        private_root=paths["private_root"],
        fixture=hooks,
        protocol_dir=paths["protocol_dir"],
    )
    assert result["status"] == M.STATUS
    assert result["fixture"] is True and result["executable"] is False
    assert result["counts"] == {
        "sources": 20,
        "cases": 10,
        "probe_inputs": 20,
        "pair_inputs": 10,
    }
    assert len(list((paths["private_root"] / "bridges").glob("*.json"))) == 20
    assert len(list((paths["private_root"] / "sources").glob("*.json"))) == 10
    config = M._read(paths["private_root"] / "execution-config.json")
    assert list(config["cases"]) == [f"case-{number:02d}" for number in range(1, 11)]
    assert all(
        set(row["probe_inputs"]) == {"case", "reference"}
        for row in config["cases"].values()
    )
    component = M._read(paths["protocol_dir"] / "fixture_r3_probe_component.json")
    assert component["watchdogs_s"] == {"legacy": 480, "optimizer": 480}
    assert set(component["runtime_source_sha256_lf"]) == M.HOST.BASE_PROBE.RUNTIME_PINS
    protocol = M._read(paths["protocol_dir"] / "fixture_r3_five_policy.json")
    assert protocol["tool_protocols"]["view"]["path"] == M.VIEW.PROTOCOL_RELATIVE
    assert set(M.EXPORT.SOURCE_FILES) <= set(protocol["sources_lf"])
    assert set(M.MATRIX.SOURCE_FILES) <= set(protocol["sources_lf"])
    assert protocol["paired_cap"] == 150 and protocol["sdk_role_cap"] == 300
    assert protocol["wall_cap_s"] == 144000
    for path in paths["protocol_dir"].glob("*.json"):
        assert not M._public_has_private_path(M._read(path))


def test_prepare_requires_fresh_targets_before_consuming_fixture_inputs(
    tmp_path: Path,
) -> None:
    paths, hooks = _fixture(tmp_path)
    paths["private_root"].mkdir()
    called = False

    def forbidden(_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal called
        called = True
        raise AssertionError("freshness must fail first")

    bad = M.FixtureHooks(**{**hooks.__dict__, "input_checked": forbidden})
    with pytest.raises(ValueError, match="fresh"):
        M.prepare(
            paths["index"],
            paths["batch"],
            paths["study"],
            paths["template"],
            paths["lab"],
            source_anchor="a" * 40,
            tag="fixture_fresh",
            private_root=paths["private_root"],
            fixture=bad,
            protocol_dir=paths["protocol_dir"],
        )
    assert called is False


def test_invalid_actual_batch_contract_writes_nothing(tmp_path: Path) -> None:
    paths, hooks = _fixture(tmp_path)
    batch = M._read(paths["batch"])
    batch["sdk_intents"] = 18
    paths["batch"].unlink()
    _write(paths["batch"], batch)
    with pytest.raises(ValueError, match="complete20"):
        M.prepare(
            paths["index"],
            paths["batch"],
            paths["study"],
            paths["template"],
            paths["lab"],
            source_anchor="a" * 40,
            tag="fixture_invalid",
            private_root=paths["private_root"],
            fixture=hooks,
            protocol_dir=paths["protocol_dir"],
        )
    assert not paths["private_root"].exists()
    assert not paths["protocol_dir"].exists()


def test_existing_failure_wrapper_resolves_to_p1_template() -> None:
    path, value = M._resolve_component_template(
        ROOT / "bench/protocols/lineage_probe_failure_evidence_20261031_v3.json"
    )
    assert path.name == "lineage_paired_probes_20261031_v1.json"
    assert value["schema"] == M.P1.PROTOCOL_SCHEMA


def test_preflight_calls_runner_and_writes_nonexecution_receipt(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    config = tmp_path / "config.json"
    index = tmp_path / "index.json"
    for path in (protocol, config, index):
        _write(path, {"name": path.name})
    cases = [f"case-{number:02d}" for number in range(1, 11)]
    observed = []

    def runner(protocol_path: Path, anchor: str, config_path: Path):
        observed.append((protocol_path, anchor, config_path))
        return (
            {"case_order": cases},
            {"source_index": M._private_ref(index)},
            {case: {"source": {"case_id": case}, "inputs": {}} for case in cases},
        )

    receipt = tmp_path / "receipt.json"
    result = M.preflight(protocol, "b" * 40, config, receipt, runner_preflight=runner)
    assert observed == [(protocol, "b" * 40, config)]
    assert result["status"] == "registered_preflight_complete_not_executed"
    assert result["case_count"] == 10 and result["gpu"] == result["api"] == 0
    assert M._read(receipt) == result


def test_preflight_rejects_input_hold_without_receipt(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    config = tmp_path / "config.json"
    for path in (protocol, config):
        _write(path, {})
    cases = [f"case-{number:02d}" for number in range(1, 11)]

    def runner(_protocol: Path, _anchor: str, _config: Path):
        sources: dict[str, dict[str, Any]] = {
            case: {"source": {"case_id": case}} for case in cases
        }
        sources["case-01"] = {"source": None, "input_hold": {"decision": "HOLD"}}
        return {"case_order": cases}, {"source_index": {}}, sources

    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="available"):
        M.preflight(protocol, "c" * 40, config, receipt, runner_preflight=runner)
    assert not receipt.exists()
