# pyright: reportMissingImports=false
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import run_lineage_five_policy_v1 as M


def test_new_worker_entry_isolated_from_existing_parent(tmp_path):
    protocol = tmp_path / "protocol.json"
    previous = M.PREVIOUS.worker_command(
        tmp_path, tmp_path / "call", protocol, "a" * 40
    )
    actual = M.worker_command(tmp_path, tmp_path / "call", protocol, "a" * 40)
    assert (
        actual[:6]
        == previous[:6]
        == ["wsl.exe", "--distribution", "Ubuntu", "--exec", "bash", "-lc"]
    )
    assert "evals/lineage_five_policy_nim_transport_v1.py" in actual[6]
    assert "evals/local_pair_nim_transport_v1.py" in previous[6]
    assert M._WORKER.worker_command is M.worker_command
    assert M.PREVIOUS.worker_command is not M.worker_command
    assert (
        M._WORKER._worker_send.__code__.co_code
        == M.PREVIOUS._worker_send.__code__.co_code
    )


def setup(tmp, monkeypatch):
    (tmp / "not-live-protocol").write_text('{"fixture":true}', encoding="utf-8")
    config = tmp / "config.json"
    config.write_text("{}", encoding="utf-8")
    index = tmp / "source-index.json"
    index.write_text("{}", encoding="utf-8")
    cases = [f"case-{i:02d}" for i in range(1, 11)]
    protocol = {
        "execution_config_sha256": M.B.sha(config),
        "sources_lf": {},
        "wall_cap_s": 144000,
        "paired_cap": 150,
        "case_order": cases,
        "source_digests": {
            case: f"{number:064x}" for number, case in enumerate(cases, 1)
        },
        "arm_order": {
            case: list(M.DRIVER.ARMS[index % 5 :] + M.DRIVER.ARMS[: index % 5])
            for index, case in enumerate(cases)
        },
        "tool_protocols": {
            "paired_probe": {"path": "not-executed"},
            "view": {"path": "not-executed"},
        },
    }
    source = {
        case: {
            "source": {"backend_packet": {"fixture": True}},
            "inputs": {"case": tmp / "case", "reference": tmp / "reference"},
            "registered_input_roots": (),
        }
        for case in cases
    }
    prepared_config = {"lab": str(tmp), "source_index": M.B.ref(index)}
    monkeypatch.setattr(M, "preflight", lambda *_: (protocol, prepared_config, source))
    monkeypatch.setattr(
        M.LEASE, "lease_context", lambda: {"owner": M.LEASE.OWNER, "leaseId": "fixture"}
    )
    monkeypatch.setattr(M.NIM, "initialize", lambda *_args, **kwargs: kwargs)
    monkeypatch.setattr(
        M.NIM, "status", lambda *_: {"fixture": True, "reservations_charged": 0}
    )
    return protocol, config


def test_batch_keeps_all_fifty_rows_and_balances_order_without_external_execution(
    tmp_path, monkeypatch
):
    protocol, config = setup(tmp_path, monkeypatch)
    observed = []

    def arm(_source, **kwargs):
        observed.append((kwargs["output"].parent.name, kwargs["arm"]))
        result = {"end_state": "HOLD", "scoring": {"fixture": True}}
        M.B.write(kwargs["output"] / "result.json", result)
        return result

    monkeypatch.setattr(M.DRIVER, "run_arm", arm)
    result = M.execute(
        tmp_path / "not-live-protocol", "a" * 40, config, tmp_path / "run"
    )
    assert observed == [
        (case, policy)
        for case in protocol["case_order"]
        for policy in protocol["arm_order"][case]
    ]
    assert len(result["rows"]) == 50
    assert result["input_unavailable_rows"] == 0
    assert result["comparison_status"] == "eligible_for_registered_summary"
    assert result["paired_intents"] == result["sdk_roles_reserved"] == 0
    assert result["recovery"] is False


def test_pair_budget_reserves_before_dispatch_and_blocks_151st_request(
    tmp_path, monkeypatch
):
    _protocol, config = setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        M.HOST, "execute_pair", lambda *_args, **kwargs: calls.append(kwargs) or {}
    )

    def arm(_source, **kwargs):
        for _ in range(151):
            kwargs["execute_probe"](
                {"fixture": True}, {"fixture": True}, out=tmp_path / "not-a-gpu-job"
            )
        raise AssertionError("global pair cap must stop this fixture")

    monkeypatch.setattr(M.DRIVER, "run_arm", arm)
    with pytest.raises(ValueError, match="global cap"):
        M.execute(tmp_path / "not-live-protocol", "a" * 40, config, tmp_path / "run")
    assert len(calls) == 150
    assert len(list((tmp_path / "run/paired-intents").glob("*.json"))) == 150


def test_private_config_drift_is_rejected_before_any_policy_callback(
    tmp_path, monkeypatch
):
    _protocol, config = setup(tmp_path, monkeypatch)
    config.write_text('{"changed":true}', encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("changed config must not reach policy")

    monkeypatch.setattr(M.DRIVER, "run_arm", forbidden)
    with pytest.raises(ValueError, match="config changed"):
        M.execute(tmp_path / "not-live-protocol", "a" * 40, config, tmp_path / "run")


def test_global_missing_source_keeps_all_planned_rows_as_explicit_hold(
    tmp_path, monkeypatch
):
    protocol, config = setup(tmp_path, monkeypatch)
    _protocol, prepared_config, sources = M.preflight(None, None, None)
    gate = {
        "schema": M.DRIVER.GUARD.SCHEMA,
        "decision": "HOLD",
        "stage": "input",
        "reason": "registered_input_unavailable",
        "approval_granted": False,
        "dispatch_granted": False,
        "automatic_retry": False,
        "evidence": {},
    }
    sources["case-01"] = {
        "source": None,
        "inputs": {},
        "input_hold": gate,
        "registered_input_roots": (),
    }
    monkeypatch.setattr(M, "preflight", lambda *_: (protocol, prepared_config, sources))
    observed = []

    def arm(_source, **kwargs):
        observed.append(kwargs["run_id"])
        result = {"end_state": "HOLD", "scoring": {"fixture": True}}
        M.B.write(kwargs["output"] / "result.json", result)
        return result

    monkeypatch.setattr(M.DRIVER, "run_arm", arm)
    result = M.execute(
        tmp_path / "not-live-protocol", "a" * 40, config, tmp_path / "run"
    )
    missing = [row for row in result["rows"] if row["case_id"] == "case-01"]
    assert len(result["rows"]) == 50 and len(missing) == 5
    assert result["input_unavailable_rows"] == 5
    assert result["comparison_status"] == "inconclusive_input_unavailable"
    assert all(row["end_state"] == "HOLD" and row["scoring"] is None for row in missing)
    assert len(observed) == 45
    detail = M.B.read(tmp_path / "run/case-01/registry_next/result.json")
    assert detail["hold_reason"] == "registered_input_unavailable"
    assert detail["input_guard"]["dispatch_granted"] is False


def test_unbound_metadata_is_rejected_before_its_paths_are_parsed(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    M.B.write(metadata, {"sources": {}})
    index = tmp_path / "index.json"
    M.B.write(index, {"metadata_ref": {"file": metadata.name, "sha256": M.B.sha(metadata)}})
    inputs = {}
    for role in ("source", "case", "reference"):
        path = tmp_path / f"{role}.json"
        M.B.write(path, {})
        inputs[role] = M.B.ref(path)
    config = {"source_index": M.B.ref(index)}
    row = {"source": inputs["source"], "probe_inputs": {key: inputs[key] for key in ("case", "reference")}}
    assert metadata.resolve() in M._registered_case_roots(config, row)
    original_read, parsed = M.B.read, []
    def track(path):
        parsed.append(Path(path).resolve())
        return original_read(path)
    monkeypatch.setattr(M.B, "read", track)
    metadata.write_bytes(b"unregistered and invalid JSON")
    with pytest.raises(ValueError):
        M._registered_case_roots(config, row)
    assert metadata.resolve() not in parsed


def test_preflight_registered_metadata_permission_loss_holds_before_loading_cases(tmp_path, monkeypatch):
    import errno
    metadata = tmp_path / "metadata.json"
    M.B.write(metadata, {"sources": {}})
    index = tmp_path / "index.json"
    M.B.write(index, {"metadata_ref": {"file": metadata.name, "sha256": M.B.sha(metadata)}})
    inputs = {}
    for role in ("source", "case", "reference"):
        path = tmp_path / f"{role}.json"
        M.B.write(path, {})
        inputs[role] = M.B.ref(path)
    cases = [f"case-{i:02d}" for i in range(1, 11)]
    row = {"source": inputs["source"], "probe_inputs": {role: inputs[role] for role in ("case", "reference")}}
    config_path = tmp_path / "config.json"
    M.B.write(config_path, {"schema": "lineage_five_policy_execution_config_v1", "lab": str(tmp_path),
                            "source_index": M.B.ref(index), "cases": {case: row for case in cases}})
    tool = "bench/protocols/fixed_eval_v1.json"
    protocol = {"sources_lf": {name: "a" * 64 for name in M.SOURCE_FILES},
                "execution_config_sha256": M.B.sha(config_path), "case_order": cases,
                "paired_cap": 150, "sdk_role_cap": 300, "wall_cap_s": 144000,
                "tool_protocols": {key: {"path": tool, "sha256_lf": M.B.sha(ROOT / tool, lf=True)}
                                   for key in ("paired_probe", "view")}}
    monkeypatch.setattr(M.DRIVER, "verify_protocol", lambda *_: protocol)
    monkeypatch.setattr(M.HOST, "verify_protocol", lambda *_: None)
    monkeypatch.setattr(M.DRIVER.HOST.RV, "verify_protocol", lambda *_: None)
    monkeypatch.setattr(M.DRIVER, "checked_source", lambda *_: pytest.fail("no source load after registered permission loss"))
    original_sha = M.B.sha
    def locked(path, **kwargs):
        if Path(path).resolve() == metadata.resolve():
            raise PermissionError(errno.EACCES, "controlled input permission loss", str(metadata))
        return original_sha(path, **kwargs)
    monkeypatch.setattr(M.B, "sha", locked)
    _protocol, _config, sources = M.preflight(tmp_path / "not-live-protocol", "a" * 40, config_path)
    assert len(sources) == 10
    assert all(item["input_hold"]["decision"] == "HOLD" and item["source"] is None for item in sources.values())
    assert all(item["input_hold"]["approval_granted"] is False for item in sources.values())
