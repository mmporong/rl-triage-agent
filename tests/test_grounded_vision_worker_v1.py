from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evals import grounded_vision_observer_v1 as G
from evals import grounded_vision_worker_v1 as W
from evals import model_decision_journal_v1 as J
from evals import vision_evidence_contract_v1 as V


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contract(root: Path, name: str) -> Path:
    source = root / name
    source.mkdir(parents=True)
    video = source / "source.mp4"
    checkpoint = source / "policy.pt"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "testsrc=duration=6:size=64x64:rate=2", "-c:v", "mpeg4", "-y", str(video),
        ],
        check=True,
        capture_output=True,
    )
    checkpoint.write_bytes(("checkpoint-" + name).encode())
    run_id = f"play-{name}"
    frames = []
    for index in range(1, 7):
        frames.append({
            "timestamp_s": index - 0.5,
            "play_run_id": run_id, "video_sha256": _sha(video.read_bytes()),
        })
    spec = {
        "schema": V.PREPARE_SCHEMA,
        "play_run_id": run_id,
        "source_lineage_id": f"lineage-{name}",
        "parent_source_lineage_id": None,
        "video": {"path": str(video), "logical_id": f"{name}/source.mp4", "play_run_id": run_id},
        "checkpoint": {
            "path": str(checkpoint), "logical_id": f"{name}/policy.pt", "play_run_id": run_id,
        },
        "frames": frames,
        "measurement": None,
        "play": {
            "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "camera": "follow_camera",
            "command_name": "base_velocity", "command_frame": "body", "motion_frame": "world",
            "settings_sha256": "a" * 64,
        },
    }
    spec_path = source / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return V.prepare(spec_path, root / f"sealed-{name}")


def _run_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True, encoding="utf-8",
    ).stdout.strip()


def _repo(tmp_path: Path, contracts: list[Path]) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "fixture@example.invalid")
    _run_git(repo, "config", "user.name", "Fixture")
    source = repo / "source.py"
    evidence = repo / "evidence.json"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    evidence.write_text('{"evidence":true}\n', encoding="utf-8")
    runtime_sources = {}
    for name in sorted(W.REQUIRED_SOURCE_FILES):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# fixture for {name}\n", encoding="utf-8")
        runtime_sources[name] = W._sha_lf(path)
    inputs = []
    for index, contract_path in enumerate(contracts, 1):
        contract = V.verify(contract_path)
        inputs.append({
            "logical_id": f"clip-{index:02d}",
            "contract_sha256": V.sha256(contract_path),
            "checkpoint_sha256": contract["provenance"]["checkpoint"]["sha256"],
        })
    protocol = {
        "schema": W.PROTOCOL_SCHEMA,
        "protocol_id": "fixture-grounded-vision-v1",
        "model": G.MODEL,
        "provider_aliases": sorted(G.ACCEPTED_PROVIDER_MODELS),
        "api": {
            "total_cap": 6, "dialogue_cap": 3, "infra_retry": 1,
            "semantic_feedback": 1, "sdk_retries": 0,
        },
        "development_inputs": inputs,
        "fixed_order": [row["logical_id"] for row in inputs],
        "registration": {"purpose": "CPU fixture", "prediction": "registered"},
        "source_sha256_lf": {"source.py": W._sha_lf(source), **runtime_sources},
        "evidence_sha256_lf": {"evidence.json": W._sha_lf(evidence)},
    }
    protocol_path = repo / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    _run_git(repo, "add", "source.py", "evidence.json", "protocol.json", "evals")
    _run_git(repo, "commit", "-m", "fixture")
    return repo, protocol_path, _run_git(repo, "rev-parse", "HEAD")


def _binding(contract: Path, protocol: Path, anchor: str, run_id: str | None = None) -> dict:
    value = json.loads(protocol.read_text(encoding="utf-8"))
    contract_sha = V.sha256(contract)
    logical_id = next(row["logical_id"] for row in value["development_inputs"]
                      if row["contract_sha256"] == contract_sha)
    return G.binding_template(
        contract,
        run_id=run_id or f"v02-dev-{logical_id}",
        source_anchor=anchor,
        protocol_sha256=W._sha_lf(protocol),
    )


def _provider(contract: Path, content: str | None = None) -> dict:
    frames = {row["frame_id"]: row["timestamp_s"] for row in V.verify(contract)["frames"]}
    body = content or json.dumps({
        "walking": "yes", "in_place": "no", "falling": "no",
        "evidence": {
            "walking": [{"frame_id": "f01", "timestamp_s": frames["f01"]}],
            "in_place": [{"frame_id": "f02", "timestamp_s": frames["f02"]}],
            "falling": [{"frame_id": "f03", "timestamp_s": frames["f03"]}],
        },
    })
    return {
        "provider_model": "private/nvidia/nemotron-3.5-super-120b-a12b",
        "finish_reason": "stop", "content": body,
        "input_tokens": 100, "output_tokens": 40, "total_tokens": 140, "wall_s": 0.1,
    }


def _write_binding(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _paths(repo: Path, protocol: Path, contract: Path) -> dict[str, Path]:
    return W._runtime_paths(repo, json.loads(protocol.read_text(encoding="utf-8")), contract)


def test_source_verification_rejects_mutation_and_changed_anchor_blob(tmp_path: Path) -> None:
    contract = _contract(tmp_path, "one")
    repo, protocol, anchor = _repo(tmp_path, [contract])
    binding = _binding(contract, protocol, anchor)
    checked = W.verify_source(binding, protocol, contract, root=repo)
    assert checked["verified_source_files"] == len(W.REQUIRED_SOURCE_FILES) + 1
    assert checked["verified_evidence_files"] == 1

    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="LF SHA|clean"):
        W.verify_source(binding, protocol, contract, root=repo)
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _run_git(repo, "add", "source.py")
    _run_git(repo, "commit", "--allow-empty", "-m", "later")
    assert W.verify_source(binding, protocol, contract, root=repo)["source_anchor"] == anchor


def test_execute_writes_state_receipt_and_rejects_parameter_change(tmp_path: Path, monkeypatch) -> None:
    contract = _contract(tmp_path, "one")
    repo, protocol, anchor = _repo(tmp_path, [contract])
    binding = _binding(contract, protocol, anchor)
    binding_path = _write_binding(tmp_path / "binding.json", binding)
    monkeypatch.setattr(W.sys, "platform", "linux")
    paths = _paths(repo, protocol, contract)
    result, code = W.execute(
        contract, binding_path, protocol, paths["folder"], paths["api"],
        paths["receipts"], complete=lambda *_: _provider(contract), root=repo,
    )
    assert result["status"] == "observation_ready" and code == 0
    receipt = json.loads((paths["receipts"] / "0001.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "observation_ready" and receipt["observation_valid"] is True
    assert receipt["api"] == {"reserved": 1, "pending": 0, "remaining": 5}

    monkeypatch.setattr(
        W.OBSERVER,
        "REQUEST_PARAMS",
        {**W.OBSERVER.REQUEST_PARAMS, "max_tokens": 2048},
    )
    with pytest.raises(W.OBSERVER.JOURNAL.JournalIntegrityError, match="manifest"):
        W.execute(
            contract, binding_path, protocol, paths["folder"], paths["api"],
            paths["receipts"], complete=lambda *_: _provider(contract), root=repo,
        )


def test_distinct_contract_decisions_share_one_six_call_ledger(tmp_path: Path, monkeypatch) -> None:
    first = _contract(tmp_path, "one")
    second = _contract(tmp_path, "two")
    repo, protocol, anchor = _repo(tmp_path, [first, second])
    monkeypatch.setattr(W.sys, "platform", "linux")
    for index, contract in enumerate((first, second), 1):
        binding = _binding(contract, protocol, anchor)
        binding_path = _write_binding(tmp_path / f"binding-{index}.json", binding)
        paths = _paths(repo, protocol, contract)
        result, code = W.execute(
            contract, binding_path, protocol, paths["folder"],
            paths["api"], paths["receipts"],
            complete=lambda *_, _contract=contract: _provider(_contract), root=repo,
        )
        assert result["status"] == "observation_ready" and code == 0
    api_root = _paths(repo, protocol, first)["api"]
    audit = J.DecisionJournal(
        tmp_path / "unused-decision", api_root,
        _binding(first, protocol, anchor, "audit-only"), G.REQUEST_PARAMS, api_cap=6,
    )._audit_api()
    assert audit["reserved"] == 2 and audit["remaining"] == 4

    third_binding = _binding(first, protocol, anchor)
    third_binding_path = _write_binding(tmp_path / "binding-wrong.json", third_binding)
    with pytest.raises(ValueError, match="runtime"):
        W.execute(
            first, third_binding_path, protocol, _paths(repo, protocol, first)["folder"],
            tmp_path / "wrong-api", _paths(repo, protocol, first)["receipts"],
            complete=lambda *_: _provider(first), root=repo,
        )


def test_pending_reservation_blocks_worker_without_provider_call(tmp_path: Path, monkeypatch) -> None:
    contract = _contract(tmp_path, "one")
    repo, protocol, anchor = _repo(tmp_path, [contract])
    binding = _binding(contract, protocol, anchor)
    binding_path = _write_binding(tmp_path / "binding.json", binding)
    paths = _paths(repo, protocol, contract)
    api_root = paths["api"]
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    W._prepare_stage(paths["stage"], W._stage_manifest(protocol_value, protocol, binding))
    journal = J.DecisionJournal(
        paths["decision"], api_root, binding, G.REQUEST_PARAMS, api_cap=6,
    )
    J._write_new(api_root / "reservations/0001.json", {
        "schema": "model_decision_reservation_v1", "index": 1,
        "manifest_sha256": journal.manifest_sha256,
        "binding_sha256": journal.binding_sha256,
        "run_id": binding["run_id"], "ordinal": 1,
        "messages_sha256": J.digest(G.request_view(contract)["messages"]),
        "params_sha256": journal.params_sha256,
        "reserved_at": "2026-10-25T00:00:00+00:00",
    })
    monkeypatch.setattr(W.sys, "platform", "linux")
    called = False

    def forbidden(*_args):
        nonlocal called
        called = True
        return _provider(contract)

    result, code = W.execute(
        contract, binding_path, protocol, paths["folder"], api_root,
        paths["receipts"], complete=forbidden, root=repo,
    )
    assert result["status"] == "blocked_pending" and code == 4
    assert called is False
    receipt = json.loads((paths["receipts"] / "0001.json").read_text(encoding="utf-8"))
    assert receipt["observation_valid"] is False


def test_worker_rejects_windows_and_protocol_cannot_raise_api_cap(tmp_path: Path, monkeypatch) -> None:
    contract = _contract(tmp_path, "one")
    repo, protocol, anchor = _repo(tmp_path, [contract])
    binding = _binding(contract, protocol, anchor)
    binding_path = _write_binding(tmp_path / "binding.json", binding)
    monkeypatch.setattr(W.sys, "platform", "win32")
    paths = _paths(repo, protocol, contract)
    with pytest.raises(RuntimeError, match="WSL Linux"):
        W.execute(
            contract, binding_path, protocol, paths["folder"], paths["api"],
            paths["receipts"], complete=lambda *_: _provider(contract), root=repo,
        )
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    missing = json.loads(json.dumps(protocol_value))
    missing["source_sha256_lf"].pop("evals/grounded_vision_worker_v1.py")
    protocol.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime 필수"):
        W.verify_protocol(protocol, contract, binding)

    protocol_value["api"]["total_cap"] = 7
    protocol.write_text(json.dumps(protocol_value), encoding="utf-8")
    with pytest.raises(ValueError, match="모델·API"):
        W.verify_protocol(protocol, contract, binding)


def test_stage_is_canonical_ordered_and_binds_run_id(tmp_path: Path, monkeypatch) -> None:
    first = _contract(tmp_path, "one")
    second = _contract(tmp_path, "two")
    repo, protocol, anchor = _repo(tmp_path, [first, second])
    monkeypatch.setattr(W.sys, "platform", "linux")
    second_binding = _binding(second, protocol, anchor)
    second_binding_path = _write_binding(tmp_path / "binding-second.json", second_binding)
    second_paths = _paths(repo, protocol, second)
    with pytest.raises(ValueError, match="fixed_order"):
        W.execute(
            second, second_binding_path, protocol, second_paths["folder"], second_paths["api"],
            second_paths["receipts"], complete=lambda *_: _provider(second), root=repo,
        )

    first_binding = _binding(first, protocol, anchor)
    first_binding_path = _write_binding(tmp_path / "binding-first.json", first_binding)
    first_paths = _paths(repo, protocol, first)
    result, _ = W.execute(
        first, first_binding_path, protocol, first_paths["folder"], first_paths["api"],
        first_paths["receipts"], complete=lambda *_: _provider(first), root=repo,
    )
    assert result["status"] == "observation_ready"
    manifest = json.loads((first_paths["stage"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol_sha256_lf"] == W._sha_lf(protocol)
    assert manifest["source_anchor"] == anchor
    assert manifest["fixed_order"] == ["clip-01", "clip-02"]

    terminal_path = first_paths["decision"] / "terminal.json"
    terminal_raw = terminal_path.read_bytes()
    terminal = json.loads(terminal_raw)
    terminal["manifest_sha256"] = "f" * 64
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(W.OBSERVER.JOURNAL.JournalIntegrityError, match="terminal 봉인"):
        W.execute(
            second, second_binding_path, protocol, second_paths["folder"], second_paths["api"],
            second_paths["receipts"], complete=lambda *_: _provider(second), root=repo,
        )
    terminal_path.write_bytes(terminal_raw)

    event_path = first_paths["decision"] / "events" / "0001.json"
    event_raw = event_path.read_bytes()
    event_path.unlink()
    with pytest.raises(W.OBSERVER.JOURNAL.JournalIntegrityError, match="events 봉인"):
        W.execute(
            second, second_binding_path, protocol, second_paths["folder"], second_paths["api"],
            second_paths["receipts"], complete=lambda *_: _provider(second), root=repo,
        )
    event_path.write_bytes(event_raw)

    response_path = first_paths["api"] / "responses" / "0001.json"
    response_raw = response_path.read_bytes()
    response_path.unlink()
    with pytest.raises(W.OBSERVER.JOURNAL.JournalIntegrityError, match="API responses 봉인"):
        W.execute(
            second, second_binding_path, protocol, second_paths["folder"], second_paths["api"],
            second_paths["receipts"], complete=lambda *_: _provider(second), root=repo,
        )
    response_path.write_bytes(response_raw)

    second_result, _ = W.execute(
        second, second_binding_path, protocol, second_paths["folder"], second_paths["api"],
        second_paths["receipts"], complete=lambda *_: _provider(second), root=repo,
    )
    assert second_result["status"] == "observation_ready"
    replayed, _ = W.execute(
        first, first_binding_path, protocol, first_paths["folder"], first_paths["api"],
        first_paths["receipts"],
        complete=lambda *_: (_ for _ in ()).throw(AssertionError("terminal cache에서 provider 호출")),
        root=repo,
    )
    assert replayed["status"] == "observation_ready"
    assert sorted(path.name for path in first_paths["receipts"].iterdir()) == ["0001.json", "0002.json"]

    changed_binding = _binding(first, protocol, anchor, "changed-run-id")
    changed_path = _write_binding(tmp_path / "binding-changed.json", changed_binding)
    with pytest.raises(ValueError, match="canonical"):
        W.execute(
            first, changed_path, protocol, first_paths["folder"], first_paths["api"],
            first_paths["receipts"], complete=lambda *_: _provider(first), root=repo,
        )

    with pytest.raises(ValueError, match="canonical runtime"):
        W.execute(
            first, first_binding_path, protocol, repo / "sibling" / "decision",
            first_paths["api"], first_paths["receipts"],
            complete=lambda *_: _provider(first), root=repo,
        )


def test_paused_input_and_fake_terminal_do_not_open_next_input(tmp_path: Path, monkeypatch) -> None:
    first = _contract(tmp_path, "one")
    second = _contract(tmp_path, "two")
    repo, protocol, anchor = _repo(tmp_path, [first, second])
    first_binding = _binding(first, protocol, anchor)
    second_binding = _binding(second, protocol, anchor)
    first_binding_path = _write_binding(tmp_path / "binding-first.json", first_binding)
    second_binding_path = _write_binding(tmp_path / "binding-second.json", second_binding)
    first_paths = _paths(repo, protocol, first)
    second_paths = _paths(repo, protocol, second)
    monkeypatch.setattr(W.sys, "platform", "linux")
    monkeypatch.setattr(W.OBSERVER, "run", lambda *_args, **_kwargs: {
        "status": "paused_infra", "engine_result": None,
        "api": {"reserved": 1, "pending": 0, "remaining": 5},
    })
    result, _ = W.execute(
        first, first_binding_path, protocol, first_paths["folder"], first_paths["api"],
        first_paths["receipts"], complete=lambda *_: _provider(first), root=repo,
    )
    assert result["status"] == "paused_infra"
    assert not (first_paths["stage"] / "completions" / "clip-01.json").exists()
    if os.name == "posix":
        outside = tmp_path / "outside-completions"
        outside.mkdir()
        os.symlink(outside, first_paths["completions"], target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            W.execute(
                second, second_binding_path, protocol, second_paths["folder"], second_paths["api"],
                second_paths["receipts"], complete=lambda *_: _provider(second), root=repo,
            )
        first_paths["completions"].unlink()
    fake_terminal = first_paths["decision"] / "terminal.json"
    J._write_new(fake_terminal, {
        "schema": "model_decision_terminal_v1", "manifest_sha256": "f" * 64,
        "status": "choice_ready", "engine_result": {}, "calls_sha256": "f" * 64,
        "completed_calls": 1,
    })
    with pytest.raises(ValueError, match="완료 receipt"):
        W.execute(
            second, second_binding_path, protocol, second_paths["folder"], second_paths["api"],
            second_paths["receipts"], complete=lambda *_: _provider(second), root=repo,
        )


def test_receipt_directory_appends_for_terminal_cache_and_paused_resume(
    tmp_path: Path, monkeypatch,
) -> None:
    contract = _contract(tmp_path, "one")
    repo, protocol, anchor = _repo(tmp_path, [contract])
    binding = _binding(contract, protocol, anchor)
    binding_path = _write_binding(tmp_path / "binding.json", binding)
    paths = _paths(repo, protocol, contract)
    monkeypatch.setattr(W.sys, "platform", "linux")

    for _ in range(2):
        result, _ = W.execute(
            contract, binding_path, protocol, paths["folder"], paths["api"],
            paths["receipts"], complete=lambda *_: _provider(contract), root=repo,
        )
        assert result["status"] == "observation_ready"
    assert sorted(path.name for path in paths["receipts"].iterdir()) == ["0001.json", "0002.json"]

    other_fixture = tmp_path / "other-fixture"
    other_fixture.mkdir()
    other = _contract(other_fixture, "other")
    other_repo, other_protocol, other_anchor = _repo(other_fixture, [other])
    other_binding = _binding(other, other_protocol, other_anchor)
    other_binding_path = _write_binding(tmp_path / "binding-other.json", other_binding)
    other_paths = _paths(other_repo, other_protocol, other)
    original_run = W.OBSERVER.run
    advances = 0

    def paused_then_real(*args, **kwargs):
        nonlocal advances
        advances += 1
        if advances == 1:
            return {
                "status": "paused_infra", "engine_result": None,
                "api": {"reserved": 0, "pending": 0, "remaining": 6},
            }
        return original_run(*args, **kwargs)

    monkeypatch.setattr(W.OBSERVER, "run", paused_then_real)
    for expected in ("paused_infra", "observation_ready"):
        result, _ = W.execute(
            other, other_binding_path, other_protocol, other_paths["folder"], other_paths["api"],
            other_paths["receipts"], complete=lambda *_: _provider(other), root=other_repo,
        )
        assert result["status"] == expected
    assert sorted(path.name for path in other_paths["receipts"].iterdir()) == ["0001.json", "0002.json"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process group 계약")
def test_worker_requires_real_private_process_group() -> None:
    command = [
        sys.executable, "-c",
        "import json; from evals import grounded_vision_worker_v1 as w; print(json.dumps(w._process_group()))",
    ]
    private = subprocess.run(command, check=True, capture_output=True, text=True, start_new_session=True)
    pid, pgid, sid = json.loads(private.stdout)
    assert pid == pgid == sid

    inherited = subprocess.run(command, check=False, capture_output=True, text=True)
    assert inherited.returncode != 0
    assert "전용 process group" in inherited.stderr


def test_live_identity_rejects_stale_start_ticks(tmp_path: Path, monkeypatch) -> None:
    identity_path = tmp_path / "identity.json"
    identity = {"pid": 123, "start_ticks": "10", "pgid": 123, "sid": 123}
    J._write_new(identity_path, identity)
    monkeypatch.setattr(W, "_start_ticks", lambda _pid: "11")
    monkeypatch.setattr(W.os, "getpgid", lambda _pid: 123, raising=False)
    monkeypatch.setattr(W.os, "getsid", lambda _pid: 123, raising=False)
    with pytest.raises(W.OBSERVER.JOURNAL.JournalIntegrityError, match="stale"):
        W._verify_live_identity(identity_path, identity)


def test_main_initializes_stage_before_identity_and_runs_same_cli_again(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    contract = _contract(tmp_path, "one")
    repo, protocol, anchor = _repo(tmp_path, [contract])
    binding = _binding(contract, protocol, anchor)
    binding_path = _write_binding(tmp_path / "binding.json", binding)
    paths = _paths(repo, protocol, contract)
    provider_calls = []
    monkeypatch.setattr(W, "ROOT", repo)
    monkeypatch.setattr(W.sys, "platform", "linux")
    monkeypatch.setattr(
        W, "_default_complete",
        lambda *_: (provider_calls.append(1), _provider(contract))[1],
    )
    monkeypatch.setattr(W, "_process_group", lambda: (123, 123, 123))
    monkeypatch.setattr(W, "_start_ticks", lambda _pid: "10")
    monkeypatch.setattr(W.os, "getpgid", lambda _pid: 123, raising=False)
    monkeypatch.setattr(W.os, "getsid", lambda _pid: 123, raising=False)
    argv = [
        "--contract", str(contract), "--binding", str(binding_path),
        "--protocol", str(protocol), "--folder", str(paths["folder"]),
        "--api-root", str(paths["api"]), "--receipt", str(paths["receipts"]),
        "--pid-file", str(paths["pid"]),
    ]
    assert W.main(argv) == 0
    assert W.main(argv) == 0
    assert (paths["stage"] / "manifest.json").is_file()
    assert not paths["pid"].exists()
    assert sorted(path.name for path in paths["receipts"].iterdir()) == ["0001.json", "0002.json"]
    outputs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["status"] for row in outputs] == ["observation_ready", "observation_ready"]
    assert provider_calls == [1]


@pytest.mark.skipif(sys.platform != "linux", reason="WSL child setsid CLI 회귀")
def test_real_linux_child_main_replays_without_second_provider_call(tmp_path: Path) -> None:
    contract = _contract(tmp_path, "one")
    repo, protocol, anchor = _repo(tmp_path, [contract])
    binding = _binding(contract, protocol, anchor)
    binding_path = _write_binding(tmp_path / "binding.json", binding)
    paths = _paths(repo, protocol, contract)
    payload = tmp_path / "provider.json"
    payload.write_text(json.dumps(_provider(contract)), encoding="utf-8")
    counter = tmp_path / "provider-count.txt"
    child = tmp_path / "child.py"
    child.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "source_root, root, payload, counter, *args = sys.argv[1:]\n"
        "sys.path.insert(0, source_root)\n"
        "from evals import grounded_vision_worker_v1 as W\n"
        "W.ROOT = Path(root)\n"
        "def provider(*_):\n"
        "    path = Path(counter)\n"
        "    count = int(path.read_text()) if path.exists() else 0\n"
        "    path.write_text(str(count + 1))\n"
        "    return json.loads(Path(payload).read_text())\n"
        "W._default_complete = provider\n"
        "raise SystemExit(W.main(args))\n",
        encoding="utf-8",
    )
    argv = [
        str(Path(__file__).resolve().parents[1]), str(repo), str(payload), str(counter),
        "--contract", str(contract), "--binding", str(binding_path),
        "--protocol", str(protocol), "--folder", str(paths["folder"]),
        "--api-root", str(paths["api"]), "--receipt", str(paths["receipts"]),
        "--pid-file", str(paths["pid"]),
    ]
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, str(child), *argv], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="strict", start_new_session=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert '"status": "observation_ready"' in completed.stdout
    assert counter.read_text(encoding="utf-8") == "1"
    assert sorted(path.name for path in paths["receipts"].iterdir()) == ["0001.json", "0002.json"]
    assert not paths["pid"].exists()
