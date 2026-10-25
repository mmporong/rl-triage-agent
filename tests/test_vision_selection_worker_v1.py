"""영상 선택 비교 worker의 source·순서·재개 경계 CPU 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import model_decision_journal_v1 as J
import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as COMP
import vision_selection_contract_v1 as CONTRACT
import vision_selection_worker_v1 as W


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(CALL._canonical(value))


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True, encoding="utf-8", errors="strict",
    ).stdout.strip()


def _rows(method_sha: str, source_sha: str) -> list[dict]:
    outputs = []
    call_index = 0
    for slot_index, slot in enumerate(COMP.design(), 1):
        role_index = (slot_index - 1) // 2
        seed = 3100 if slot["seed_group"] == "source-01" else 3101
        shared = {
            "slot_id": slot["slot_id"], "seed_group": slot["seed_group"],
            "source_role": slot["source_role"], "window": slot["window"],
            "source_seed": seed, "source_lineage_id": f"lineage-{role_index}",
            "parent_lineage_id": None, "split": "comparison",
            "method_protocol_sha256": method_sha, "source_protocol_sha256": source_sha,
            "capsule_method_raw_sha256": {name: "1" * 64 for name in COMP.CAPSULE_METHODS},
            "checkpoint_sha256": f"{role_index + 1:064x}",
            "video_sha256": f"{role_index + 11:064x}",
            "trace_sha256": f"{slot_index + 21:064x}",
            "contract_sha256": f"{slot_index + 31:064x}",
            "snapshot_sha256": f"{slot_index + 41:064x}",
            "camera_pose_audit_sha256": f"{slot_index + 51:064x}",
        }
        for view in slot["view_order"]:
            call_index += 1
            outputs.append({
                **shared, "view_kind": view, "call_id": f"choice-{call_index:02d}",
                "capsule_sha256": f"{call_index + 100:064x}",
                "messages_sha256": f"{call_index + 200:064x}",
            })
    return outputs


def _provider(content: str, *, status: int = 200) -> dict:
    if status != 200:
        return {"status_code": status, "infra_error": "temporary"}
    return {
        "provider_model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "finish_reason": "stop", "content": content,
        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "wall_s": 0.01,
    }


@pytest.fixture
def case(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "fixture")

    for relative in sorted(W.REQUIRED_SOURCES):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / relative
        path.write_bytes(source.read_bytes() if source.is_file() else f"# {relative}\n".encode())
    dependencies = {
        "method-dependency.txt": b"method\n",
        "method-evidence.txt": b"method evidence\n",
        "source-dependency.txt": b"source\n",
        "runtime-dependency.txt": b"runtime\n",
        "package-dependency.txt": b"package\n",
        "recording-dependency.txt": b"recording\n",
        "execution-evidence.txt": b"execution evidence\n",
    }
    for name, raw in dependencies.items():
        (repo / name).write_bytes(raw)

    method_path = repo / "bench/protocols/vision_selection_comparison_v1.json"
    method = {
        "schema": "vision_selection_comparison_method_v1",
        "source_sha256_lf": {"method-dependency.txt": _sha_lf(repo / "method-dependency.txt")},
        "evidence_sha256_lf": {"method-evidence.txt": _sha_lf(repo / "method-evidence.txt")},
    }
    _write(method_path, method)
    source_path = repo / "bench/protocols/vision_brev_source_v1.json"
    source = {
        "schema": "vision_brev_source_protocol_v1",
        "source_sha256_lf": {"source-dependency.txt": _sha_lf(repo / "source-dependency.txt")},
        "runtime_source_sha256_lf": {"runtime-dependency.txt": _sha_lf(repo / "runtime-dependency.txt")},
        "python_package_source_sha256_lf": {"package-dependency.txt": _sha_lf(repo / "package-dependency.txt")},
        "recording": {"source_files_lf": {
            "recording-dependency.txt": _sha_lf(repo / "recording-dependency.txt"),
        }},
    }
    _write(source_path, source)

    frozen = repo / "inputs/frozen"
    bundle = {
        "schema": CONTRACT.SCHEMA, "view_kind": "text_only",
        "operational_base": {"case_checkpoint_sha256": "a" * 64},
        "selection_view": {
            "eligible_probes": ["P_reward"],
            "evidence_facts": {"telemetry:reward": {"value": 0.1}},
        },
    }
    _write(frozen / "bundle.json", bundle)
    _write(frozen / "messages.json", {"messages": [{"role": "user", "content": "choose"}]})
    _write(frozen / "capsule.json", {
        "schema": CALL.CAPSULE_SCHEMA,
        "bundle_sha256": J.file_sha(frozen / "bundle.json"),
        "messages_sha256": J.file_sha(frozen / "messages.json"),
        "method_sha256": {name: "1" * 64 for name in CALL._METHODS},
    })
    capsule = {"path": str((frozen / "capsule.json").resolve()),
               "sha256": J.file_sha(frozen / "capsule.json"), "status": "frozen"}

    execution_id = "vision-comparison-fixture"
    method_sha, source_sha = _sha_lf(method_path), _sha_lf(source_path)
    rows = _rows(method_sha, source_sha)
    sources = {name: _sha_lf(repo / name) for name in sorted(W.REQUIRED_SOURCES)}
    sources.update({
        "method-dependency.txt": _sha_lf(repo / "method-dependency.txt"),
        "source-dependency.txt": _sha_lf(repo / "source-dependency.txt"),
        "runtime-dependency.txt": _sha_lf(repo / "runtime-dependency.txt"),
        "package-dependency.txt": _sha_lf(repo / "package-dependency.txt"),
        "recording-dependency.txt": _sha_lf(repo / "recording-dependency.txt"),
        method_path.relative_to(repo).as_posix(): method_sha,
        source_path.relative_to(repo).as_posix(): source_sha,
    })
    evidence = {
        "method-evidence.txt": _sha_lf(repo / "method-evidence.txt"),
        "execution-evidence.txt": _sha_lf(repo / "execution-evidence.txt"),
    }
    locations = repo / "private/input-locations.json"
    _write(locations, {"fixture": True})
    protocol = {
        "schema": W.PROTOCOL_SCHEMA, "protocol_id": "vision-selection-worker-fixture-v1",
        "execution_id": execution_id, "phase": "comparison",
        "method_protocol": {"path": method_path.relative_to(repo).as_posix(), "sha256_lf": method_sha},
        "source_protocol": {"path": source_path.relative_to(repo).as_posix(), "sha256_lf": source_sha},
        "rows": rows, "source_sha256_lf": sources, "evidence_sha256_lf": evidence,
        "input_locations_sha256": J.file_sha(locations),
        "api": {"total_cap": 72, "logical_decisions": 24, "max_calls_per_decision": 3,
                "semantic_feedback": 1, "infra_retry": 1, "sdk_retries": 0},
    }
    protocol_path = repo / "bench/protocols/vision_selection_execution_v1.json"
    _write(protocol_path, protocol)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    anchor = _git(repo, "rev-parse", "HEAD")

    plan_rows = [{"call_id": row["call_id"],
                  "run_id": f"vision-selection-{execution_id}-{row['call_id']}",
                  "capsule": capsule} for row in rows]
    verified = {
        "schema": W.VERIFIED_INPUTS_SCHEMA, "execution_id": execution_id,
        "rows": plan_rows,
        "input_manifest_sha256": {row["call_id"]: f"{index + 301:064x}"
                                  for index, row in enumerate(rows)},
        "source_sha256_lf": sources, "evidence_sha256_lf": evidence,
        "input_locations_sha256": J.file_sha(locations),
    }
    calls: list[tuple[Path, dict, Path]] = []

    def verifier(root: Path, actual_protocol: dict, path: Path) -> dict:
        calls.append((root, actual_protocol, path))
        return verified

    return {"repo": repo, "protocol": protocol_path, "sha": _sha_lf(protocol_path),
            "anchor": anchor, "locations": locations, "verifier": verifier,
            "verifier_calls": calls, "capsule": capsule, "verified": verified}


def _prepare(case: dict) -> dict:
    return W.prepare(case["protocol"], case["sha"], case["anchor"], case["locations"],
                     root=case["repo"], input_verifier=case["verifier"])


def _linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(W.sys, "platform", "linux")
    monkeypatch.setattr(W, "_identity", lambda stage, call_id: {
        "pid": 1, "pgid": 1, "sid": 1, "start_ticks": "1",
        "stage": str(stage.resolve()), "call_id": call_id,
    })
    monkeypatch.setattr(W, "_owned", lambda *_: True)


def _advance(case: dict, monkeypatch: pytest.MonkeyPatch, call_id: str, provider) -> tuple[dict, int]:
    _linux(monkeypatch)
    monkeypatch.setattr(W, "_default_complete", provider)
    return W.advance(
        case["protocol"], case["sha"], case["anchor"], case["locations"], call_id,
        root=case["repo"], input_verifier=case["verifier"],
        provider_preflight=lambda: None,
    )


def test_verify_uses_actual_git_anchor_parent_maps_and_helper(case: dict) -> None:
    prepared = _prepare(case)
    assert prepared["source"]["source_anchor"] == case["anchor"]
    assert prepared["plan"]["status"] == "planned"
    assert len(prepared["protocol"]["rows"]) == 24
    assert case["verifier_calls"][-1][0] == case["repo"].resolve()
    assert _prepare(case)["plan"] == prepared["plan"]


def test_source_or_protocol_tampering_fails_before_helper_and_plan(case: dict) -> None:
    target = case["repo"] / "evals/vision_selection_call_v1.py"
    target.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="LF SHA|clean"):
        _prepare(case)
    assert case["verifier_calls"] == []
    assert not (case["repo"] / ".triage-runtime").exists()


def test_protocol_rejects_noncomparison_and_parent_schema(case: dict) -> None:
    value = json.loads(case["protocol"].read_text(encoding="utf-8"))
    value["phase"] = "e2e"
    changed = case["repo"] / "e2e.json"
    _write(changed, value)
    with pytest.raises(ValueError, match="phase"):
        W.verify_protocol(changed, _sha_lf(changed), root=case["repo"])

    method_path = case["repo"] / value["method_protocol"]["path"]
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["schema"] = "wrong"
    _write(method_path, method)
    value["phase"] = "comparison"
    value["method_protocol"]["sha256_lf"] = _sha_lf(method_path)
    value["rows"] = _rows(value["method_protocol"]["sha256_lf"], value["source_protocol"]["sha256_lf"])
    _write(changed, value)
    with pytest.raises(ValueError, match="method protocol schema"):
        W.verify_protocol(changed, _sha_lf(changed), root=case["repo"])

    method["schema"] = "vision_selection_comparison_method_v1"
    _write(method_path, method)
    value = json.loads(case["protocol"].read_text(encoding="utf-8"))
    value["rows"][0]["unregistered"] = True
    _write(changed, value)
    with pytest.raises(ValueError, match="INPUT_FIELDS"):
        W.verify_protocol(changed, _sha_lf(changed), root=case["repo"])

    value = json.loads(case["protocol"].read_text(encoding="utf-8"))
    source_path = case["repo"] / value["source_protocol"]["path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["runtime_source_sha256_lf"] = {"source-dependency.txt": "f" * 64}
    source["python_package_source_sha256_lf"] = {"source-dependency.txt": "e" * 64}
    _write(source_path, source)
    source_sha = _sha_lf(source_path)
    value["source_protocol"]["sha256_lf"] = source_sha
    value["source_sha256_lf"][value["source_protocol"]["path"]] = source_sha
    value["rows"] = _rows(value["method_protocol"]["sha256_lf"], source_sha)
    _write(changed, value)
    checked = W.verify_protocol(changed, _sha_lf(changed), root=case["repo"])
    assert checked["source_protocol"]["sha256_lf"] == source_sha


def test_order_then_terminal_replay_calls_provider_zero(case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    valid = json.dumps({"action": "continue", "probe": "P_reward", "fact_ids": ["telemetry:reward"]})
    called: list[int] = []
    with pytest.raises(ValueError, match="앞 selection"):
        _advance(case, monkeypatch, "choice-02", lambda *_: called.append(1) or _provider(valid))
    assert called == []
    first, code = _advance(case, monkeypatch, "choice-01", lambda *_: called.append(1) or _provider(valid))
    assert code == 0 and first["journal"]["status"] == "choice_ready" and called == [1]
    replay, code = _advance(
        case, monkeypatch, "choice-01",
        lambda *_: (_ for _ in ()).throw(AssertionError("terminal replay provider 호출")),
    )
    assert code == 0 and replay["journal"]["api"]["reserved"] == 1
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    cached, code = W.advance(
        case["protocol"], case["sha"], case["anchor"], case["locations"], "choice-01",
        root=case["repo"], input_verifier=case["verifier"],
    )
    assert code == 0 and cached["journal"]["api"]["reserved"] == 1
    receipts = sorted((case["repo"] / ".triage-runtime/vision-selection/vision-comparison-fixture/worker-receipts").glob("*.json"))
    assert len(receipts) == 3


def test_infra_feedback_caps_and_fake_walking_never_activate(case: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    valid = json.dumps({"action": "continue", "probe": "P_reward", "fact_ids": ["telemetry:reward"]})
    responses = iter([_provider('{"walking":true}'), _provider(valid)])
    result, code = _advance(case, monkeypatch, "choice-01", lambda *_: next(responses))
    assert code == 0 and result["journal"]["api"] == {"reserved": 2, "pending": 0, "remaining": 70}
    events = result["journal"]["engine_result"]["calls"]
    assert len(events) == 2
    assert sum(row.get("validation", {}).get("status") == "invalid" for row in events) == 1
    assert not (case["repo"] / "case/probes.ledger.jsonl").exists()

    frozen = CALL.prepared(case["capsule"])
    capped = iter([_provider("", status=500), _provider('{"walking":true}'), _provider(valid)])
    engine = CALL.engine(frozen, lambda *_: next(capped))
    assert engine["accepted"] is True and len(engine["calls"]) == 3
    assert sum(row["transport"] == "infra_error" for row in engine["calls"]) == 1


def test_pending_and_existing_lease_are_preserved_without_provider(case: dict,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    _linux(monkeypatch)
    prepared = _prepare(case)
    _, stage, manifest = CALL._plan(prepared["plan"])
    _, row, frozen = CALL._row(case["repo"], manifest, "choice-01")
    journal = CALL._journal(stage, manifest, row, frozen)
    J._write_new(stage / "api/reservations/0001.json", {
        "schema": "model_decision_reservation_v1", "index": 1,
        "manifest_sha256": journal.manifest_sha256, "binding_sha256": journal.binding_sha256,
        "run_id": row["run_id"], "ordinal": 1,
        "messages_sha256": J.digest(frozen["messages"]),
        "params_sha256": journal.params_sha256, "reserved_at": "2026-10-25T00:00:00+00:00",
    })
    called: list[int] = []
    monkeypatch.setattr(W, "_default_complete", lambda *_: called.append(1) or _provider("{}"))
    result, code = W.advance(
        case["protocol"], case["sha"], case["anchor"], case["locations"], "choice-01",
        root=case["repo"], input_verifier=case["verifier"], provider_preflight=lambda: None,
    )
    assert code == 4 and result["journal"]["status"] == "blocked_pending" and called == []

    identity = stage / "worker.identity.json"
    _write(identity, {"foreign": True})
    blocked, code = W.advance(
        case["protocol"], case["sha"], case["anchor"], case["locations"], "choice-01",
        root=case["repo"], input_verifier=case["verifier"], provider_preflight=lambda: None,
    )
    assert code == 4 and blocked["provider_calls"] == 0 and identity.exists() and called == []


def test_nonlinux_and_missing_key_stop_before_reservation(case: dict,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(W.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="WSL Linux"):
        W.advance(case["protocol"], case["sha"], case["anchor"], case["locations"], "choice-01",
                  root=case["repo"], input_verifier=case["verifier"])
    assert called == [] and not (case["repo"] / ".triage-runtime").exists()

    _linux(monkeypatch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        monkeypatch.setattr(W, "_default_complete", lambda *_: called.append(1) or _provider("{}"))
        W.advance(case["protocol"], case["sha"], case["anchor"], case["locations"], "choice-01",
                  root=case["repo"], input_verifier=case["verifier"])
    stage = case["repo"] / ".triage-runtime/vision-selection/vision-comparison-fixture"
    assert called == [] and not (stage / ".active").exists() and not (stage / "api").exists()

    monkeypatch.setenv("NVIDIA_API_KEY", "present-but-not-printed")
    monkeypatch.setattr(W.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="openai package"):
        W.advance(case["protocol"], case["sha"], case["anchor"], case["locations"], "choice-01",
                  root=case["repo"], input_verifier=case["verifier"])
    assert called == [] and not (stage / ".active").exists() and not (stage / "api").exists()


@pytest.mark.skipif(os.name != "posix", reason="Linux symlink 경계를 검사한다")
def test_receipt_directory_symlink_is_rejected_before_provider(
    case: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _linux(monkeypatch)
    prepared = _prepare(case)
    stage = Path(prepared["plan"]["path"]).parent
    outside = tmp_path / "outside"
    outside.mkdir()
    (stage / "worker-receipts").symlink_to(outside, target_is_directory=True)
    called: list[int] = []
    monkeypatch.setattr(W, "_default_complete", lambda *_: called.append(1) or _provider("{}"))
    with pytest.raises(J.JournalIntegrityError, match="canonical child"):
        W.advance(
            case["protocol"], case["sha"], case["anchor"], case["locations"], "choice-01",
            root=case["repo"], input_verifier=case["verifier"], provider_preflight=lambda: None,
        )
    assert called == [] and list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="Linux dirfd 경계를 검사한다")
def test_receipt_parent_swap_after_check_cannot_escape_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = (tmp_path / "stage").resolve()
    folder = stage / "worker-receipts"
    outside = (tmp_path / "outside").resolve()
    folder.mkdir(parents=True)
    outside.mkdir()
    real_open = W.os.open
    swapped = False

    def swap_after_check(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and path == "worker-receipts" and dir_fd is not None:
            swapped = True
            folder.rename(stage / "held-receipts")
            folder.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(W.os, "open", swap_after_check)
    with pytest.raises(J.JournalIntegrityError, match="쓰기 중 변경"):
        W._write_receipt_exclusive(stage, folder / "0001.json", {"status": "test"})
    assert list(outside.iterdir()) == []
    assert list((stage / "held-receipts").iterdir()) == []
    assert not (folder / "0001.json").exists()


def test_identity_creation_race_is_preserved_as_blocked(
    case: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _linux(monkeypatch)
    real_write = W._write_exclusive

    def race(path: Path, value: dict) -> None:
        if path.name == "worker.identity.json":
            raise FileExistsError(path)
        real_write(path, value)

    monkeypatch.setattr(W, "_write_exclusive", race)
    result, code = W.advance(
        case["protocol"], case["sha"], case["anchor"], case["locations"], "choice-01",
        root=case["repo"], input_verifier=case["verifier"], provider_preflight=lambda: None,
    )
    assert code == 4 and result["reason"] == "active_or_stale_worker_preserved"
