from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import ppo_resumable_model_v1 as R  # pyright: ignore[reportMissingImports]

NOW = datetime(2026, 10, 25, tzinfo=timezone.utc)  # noqa: UP017


def packet() -> dict:
    return {
        "schema": R.B.C.PACKET_SCHEMA,
        "facts": {"telemetry:error": 1.2, "telemetry:alive": 20},
        "observed": [],
        "budget_left": R.B.C.MAX_PROBES,
        "execution_budget_s": 1200,
    }


def binding(value: dict | None = None) -> dict:
    value = packet() if value is None else value
    snapshot = R.B.build_snapshot(value)
    return {
        "run_id": "ppo-inverse-ratio-s2200",
        "source_anchor": "a" * 40,
        "protocol_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "snapshot_sha256": R.J.digest(snapshot),
        "initial_messages_sha256": R.J.digest(R.B.messages(snapshot)),
    }


def choice(probe: str = "P_reward", action: str = "continue") -> dict:
    return {"action": action, "probe": probe if action == "continue" else None,
            "fact_ids": ["telemetry:error"]}


def response(body: object, *, status: int | None = None) -> dict:
    if status is not None:
        return {"error_type": "APIStatusError", "infra_error": "APIStatusError",
                "status_code": status, "wall_s": 0.1}
    return {"response_id": "r", "provider_model": R.MODEL, "finish_reason": "stop",
            "content": json.dumps(body), "reasoning_content": None,
            "input_tokens": 2, "output_tokens": 3, "total_tokens": 5, "wall_s": 0.1}


def test_500_pauses_then_new_process_resumes_and_terminal_replays_without_http(tmp_path: Path):
    value, linked = packet(), binding()
    first = R.run(tmp_path, value, linked, complete=lambda *_: response({}, status=500), now_utc=lambda: NOW)
    assert first["status"] == "paused_infra" and first["api"]["reserved"] == 1

    packet_path, binding_path = tmp_path / "packet.fixture.json", tmp_path / "binding.fixture.json"
    packet_path.write_text(json.dumps(value), encoding="utf-8")
    binding_path.write_text(json.dumps(linked), encoding="utf-8")
    script = r'''
import json,sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/"evals"))
import ppo_resumable_model_v1 as R
packet=json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
binding=json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
mode=sys.argv[5]
def complete(*_):
 if mode=="forbidden": raise AssertionError("HTTP called")
 return {"response_id":"r2","provider_model":R.MODEL,"finish_reason":"stop",
  "content":json.dumps({"action":"continue","probe":"P_value","fact_ids":["telemetry:error"]}),
  "input_tokens":2,"output_tokens":3,"total_tokens":5,"wall_s":0.1}
result=R.run(Path(sys.argv[2]),packet,binding,complete=complete,
 now_utc=lambda:datetime(2026,10,25,0,0,31,tzinfo=timezone.utc))
print(json.dumps(result,sort_keys=True))
'''
    command = [sys.executable, "-c", script, str(ROOT), str(tmp_path), str(packet_path), str(binding_path)]
    resumed = subprocess.run([*command, "success"], cwd=ROOT, check=True, capture_output=True,
                             text=True, encoding="utf-8")
    result = json.loads(resumed.stdout)
    assert result["status"] == "choice_ready" and result["api"]["reserved"] == 2

    for _ in range(3):
        replay = subprocess.run([*command, "forbidden"], cwd=ROOT, check=True, capture_output=True,
                                text=True, encoding="utf-8")
        assert json.loads(replay.stdout) == result
    assert len(list((tmp_path / "api/reservations").glob("*.json"))) == 2


def test_snapshot_and_initial_messages_binding_mismatch_stop_before_journal(tmp_path: Path):
    value = packet()
    for key in ("snapshot_sha256", "initial_messages_sha256"):
        changed = binding(value)
        changed[key] = "d" * 64
        with pytest.raises(ValueError, match=key):
            R.run(tmp_path / key, value, changed, complete=lambda *_: pytest.fail("provider called"))
        assert not (tmp_path / key / "decision").exists()


def test_terminal_model_hold_is_cached(tmp_path: Path):
    value, linked = packet(), binding()
    first = R.run(tmp_path, value, linked, complete=lambda *_: response(choice(action="hold")),
                  now_utc=lambda: NOW)
    assert first["status"] == "terminal_hold"
    assert first["engine_result"]["status"] == "terminal_hold"
    replay = R.run(tmp_path, value, linked, complete=lambda *_: pytest.fail("provider called"),
                   now_utc=lambda: NOW + timedelta(minutes=1))
    assert replay == first


def test_second_500_becomes_cached_terminal_hold_after_delay(tmp_path: Path):
    value, linked = packet(), binding()
    assert R.run(tmp_path, value, linked, complete=lambda *_: response({}, status=500),
                 now_utc=lambda: NOW)["status"] == "paused_infra"
    second = R.run(tmp_path, value, linked, complete=lambda *_: response({}, status=500),
                   now_utc=lambda: NOW + timedelta(seconds=31))
    assert second["status"] == "terminal_hold"
    assert second["engine_result"]["status"] == "terminal_hold"
    assert second["engine_result"]["reason"] == "transport_infra_error"
    assert len(list((tmp_path / "api/reservations").glob("*.json"))) == 2


def test_existing_journal_params_mismatch_is_rejected(tmp_path: Path):
    value, linked = packet(), binding()
    R.run(tmp_path, value, linked, complete=lambda *_: response(choice()), now_utc=lambda: NOW)
    manifest = tmp_path / "decision/manifest.json"
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    stored["params"]["temperature"] = 1
    manifest.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(R.J.JournalIntegrityError, match="manifest"):
        R.run(tmp_path, value, linked, complete=lambda *_: pytest.fail("provider called"), now_utc=lambda: NOW)


def test_cli_preserves_immutable_inputs_and_appends_statuses(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    packet_path, binding_path = source / "packet.json", source / "binding.json"
    value, linked = packet(), binding()
    packet_path.write_text(json.dumps(value), encoding="utf-8")
    binding_path.write_text(json.dumps(linked), encoding="utf-8")
    folder = tmp_path / "run"
    first = R.cli(packet_path, binding_path, folder, complete=lambda *_: response(choice()),
                  now_utc=lambda: NOW)
    second = R.cli(packet_path, binding_path, folder, complete=lambda *_: pytest.fail("provider called"),
                   now_utc=lambda: NOW + timedelta(minutes=1))
    assert second == first
    assert sorted(path.name for path in (folder / "statuses").glob("*.json")) == ["0001.json", "0002.json"]
    assert R._read_object(folder / "input_packet.json") == value
    assert R._read_object(folder / "input_binding.json") == linked

    changed = {**value, "facts": {**value["facts"], "telemetry:error": 9.9}}
    packet_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(R.J.JournalIntegrityError, match="input_packet"):
        R.cli(packet_path, binding_path, folder, complete=lambda *_: pytest.fail("provider called"))
    assert len(list((folder / "statuses").glob("*.json"))) == 2


def test_verify_source_accepts_bound_clean_repo_and_rejects_dirty_or_protocol_mismatch(tmp_path: Path):
    repo = tmp_path / "repo"
    protocol_path = repo / "bench/protocols/ppo_resumable_demo_v1.json"
    source_path, evidence_path = repo / "evals/source.py", repo / "evidence/result.json"
    for path, content in ((source_path, "value = 1\n"), (evidence_path, '{"ok":true}\n')):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "source_sha256_lf": {"evals/source.py": R._sha_lf(source_path)},
        "evidence_sha256_lf": {"evidence/result.json": R._sha_lf(evidence_path)},
        "runtime_source_sha256_lf": {"outside/isaac.py": "f" * 64},
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8", newline="\n")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=repo, check=True)
    subprocess.run(["git", "add", "bench/protocols/ppo_resumable_demo_v1.json",
                    "evals/source.py", "evidence/result.json"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    anchor = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            capture_output=True, text=True, encoding="utf-8").stdout.strip()
    linked = {**binding(), "source_anchor": anchor, "protocol_sha256": R._sha_lf(protocol_path)}
    verified = R.verify_source(linked, root=repo, protocol_path=protocol_path)
    assert verified["verified_source_files"] == 2
    assert verified["runtime_source_verification"] == "excluded_controller_only"

    source_path.write_text("value = 2\n", encoding="utf-8", newline="\n")
    protocol["source_sha256_lf"]["evals/source.py"] = R._sha_lf(source_path)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8", newline="\n")
    dirty_binding = {**linked, "protocol_sha256": R._sha_lf(protocol_path)}
    with pytest.raises(ValueError, match="clean"):
        R.verify_source(dirty_binding, root=repo, protocol_path=protocol_path)
    source_path.write_text("value = 1\n", encoding="utf-8", newline="\n")
    protocol["source_sha256_lf"]["evals/source.py"] = R._sha_lf(source_path)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="protocol SHA"):
        R.verify_source({**linked, "protocol_sha256": "0" * 64}, root=repo, protocol_path=protocol_path)


def test_injected_complete_skips_production_source_preflight(tmp_path: Path):
    result = R.run(tmp_path, packet(), binding(), complete=lambda *_: response(choice()), now_utc=lambda: NOW)
    assert result["status"] == "choice_ready"


def test_default_provider_is_blocked_before_call_when_source_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    def reject_source(_binding: dict) -> dict:
        raise ValueError("source preflight rejected")

    monkeypatch.setattr(R, "verify_source", reject_source)
    monkeypatch.setattr(R, "_default_complete", lambda *_: pytest.fail("provider called"))
    with pytest.raises(ValueError, match="source preflight rejected"):
        R.run(tmp_path, packet(), binding(), now_utc=lambda: NOW)
