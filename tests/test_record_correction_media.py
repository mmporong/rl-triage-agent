"""교정 정책 미디어 기록기의 CPU 전용 게이트 회귀."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))


def _module():
    spec = importlib.util.spec_from_file_location(
        "record_correction_media_under_test", ROOT / "bench/record_correction_media.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(module, status="no_recovery"):
    clips = []
    for clip_id, role, checkpoint in zip(module.CLIP_IDS,
                                         ("original_failed_source", "corrected", "fault_sham"),
                                         ("model_299.pt", "model_598.pt", "model_598.pt")):
        clips.append({"id": clip_id, "role": role, "run_basename": f"run_{clip_id}",
                      "checkpoint": checkpoint, "checkpoint_sha256": clip_id[0] * 64,
                      "params_env_sha256": "e" * 64, "params_agent_sha256": "a" * 64,
                      "overrides": module.FIXED["overrides"]})
    return {"tag": "source_tag", "status": status, "wording": f"wording:{status}",
            "source_commit_sha": "c" * 40,
            "source_commit_scope": "experiment_execution_and_verdict_code_only",
            "publisher_sources": {"evals/publish_training_correction.py": "d" * 64},
            "source_protocol_sha256_lf": "3" * 64,
            "fixed_eval_sha256_lf": "f" * 64,
            "public_sources": [{"path": "evals/results/source_tag/result.json", "sha256": "1" * 64}],
            "private_sources": [{"id": "result", "sha256": "2" * 64}],
            "clips": clips}


def _protocol(module):
    source = _source(module)
    return {**module.FIXED, "source_tag": source["tag"], "result_status": source["status"],
            "result_wording": source["wording"], "source_commit_sha": source["source_commit_sha"],
            "source_commit_scope": source["source_commit_scope"],
            "publisher_sources": source["publisher_sources"],
            "source_protocol_sha256_lf": source["source_protocol_sha256_lf"],
            "fixed_eval_sha256_lf": source["fixed_eval_sha256_lf"],
            "code_sha256_lf": {name: "1" * 64 for name in module.CODE},
            "play_py_sha256": "2" * 64, "isaaclab_commit": module.H.FIXED["isaaclab_commit"],
            "public_sources": source["public_sources"], "private_sources": source["private_sources"],
            "clips": source["clips"]}


def test_prepare_writes_status_preserving_three_clip_protocol(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for name in module.CODE:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    play = tmp_path / module.PLAY
    play.parent.mkdir(parents=True, exist_ok=True)
    play.write_text("play", encoding="utf-8")
    source = _source(module, "no_recovery")
    monkeypatch.setattr(module, "_verified_source", lambda *_args: source)

    result = module.prepare("source_tag", tmp_path)
    protocol = json.loads((tmp_path / result["protocol"]).read_text(encoding="utf-8"))

    assert result["status"] == "prepared_commit_before_recording"
    assert result["protocol"] == "bench/protocols/training_correction_media_source_tag.json"
    assert protocol["result_status"] == "no_recovery"
    assert protocol["model_selected_correction"] is False
    assert protocol["source_commit_scope"] == "experiment_execution_and_verdict_code_only"
    assert protocol["publisher_sources"]
    assert [clip["id"] for clip in protocol["clips"]] == list(module.CLIP_IDS)
    assert protocol["video_contract"] == {"width": 1280, "height": 720, "frame_rate": "50/1",
                                            "frame_count": 299, "duration_s": 5.98,
                                            "duration_tolerance_s": 0.02}
    rendered = json.dumps(protocol)
    assert "bench/private" not in rendered and str(tmp_path) not in rendered


def test_verified_source_rejects_invalid_or_partial_public_result(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    public = tmp_path / "evals/results/source_tag"
    public.mkdir(parents=True)
    (public / "result.json").write_text(json.dumps({"case": "case_01", "numeric": {
        "status": "invalid", "model_selected_correction": False}}), encoding="utf-8")
    monkeypatch.setattr(module.Q, "replay", lambda _path: {"status": "invalid"})

    with pytest.raises(ValueError, match="invalid, partial"):
        module._verified_source("source_tag", tmp_path)


def test_record_checks_commit_before_gpu_or_launcher(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(module, "verify", lambda *_args: (_ for _ in ()).throw(ValueError("must be committed")))
    monkeypatch.setattr(module.M, "query_gpu", lambda: pytest.fail("GPU queried before commit gate"))
    monkeypatch.setattr(module.M, "_record_clip", lambda *_args: pytest.fail("launcher entered"))

    with pytest.raises(ValueError, match="committed"):
        module.record("source_tag", tmp_path)


@pytest.mark.parametrize("mutation", ["checkpoint_escape", "override_change", "role_change"])
def test_tampered_clip_contract_is_rejected_before_gpu(monkeypatch, tmp_path, mutation):
    module = _module()
    protocol = _protocol(module)
    if mutation == "checkpoint_escape":
        protocol["clips"][0]["checkpoint"] = "../../model_299.pt"
    elif mutation == "override_change":
        protocol["clips"][1]["overrides"] = ["env.viewer.eye=[0,0,0]"]
    else:
        protocol["clips"][2]["role"] = "corrected"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(module, "protocol_path", lambda _tag: path)
    monkeypatch.setattr(module.M, "query_gpu", lambda: pytest.fail("GPU queried after protocol tamper"))

    with pytest.raises(ValueError, match="clip contract changed"):
        module.record("source_tag", tmp_path)


def test_git_blob_hash_reads_bound_commit_instead_of_current_worktree():
    module = _module()
    commit = module.P.S._git_head(ROOT)
    relative = module.H.FIXED_EVAL.relative_to(ROOT).as_posix()

    assert module._git_blob_lf(commit, relative) == module.H.sha(module.H.FIXED_EVAL, True)


def test_private_source_bindings_are_opaque_ids_without_paths(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(module.H, "PRIVATE", tmp_path / "bench/private/training_correction")
    paths = module._private_paths("source_tag")
    for logical_id, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(logical_id, encoding="utf-8")

    rows = module._private_artifacts("source_tag")

    assert len(rows) == len(paths)
    assert all(set(row) == {"id", "sha256"} for row in rows)
    assert "bench/private" not in json.dumps(rows)
    module._check_private_artifacts("source_tag", rows)


def test_public_source_binding_cannot_point_at_private_folder(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    public = tmp_path / "evals/results/source_tag"
    public.mkdir(parents=True)
    (public / "artifact_manifest.json").write_text(json.dumps({"files": {"result.json": "1" * 64}}),
                                                    encoding="utf-8")
    rows = [{"path": "bench/private/training_correction/source_tag/result.json",
             "sha256": "1" * 64}]

    with pytest.raises(ValueError, match="public source binding is malformed"):
        module._check_public_artifacts("source_tag", rows)


@pytest.mark.parametrize("changed_group", ["source_files", "sdk_sources", "rsl_sources"])
def test_training_runtime_source_change_is_rejected(monkeypatch, tmp_path, changed_group):
    module = _module()
    source_protocol = {"source_files": {"project.py": "d" * 64},
                       "sdk_sources": {"sdk.py": "a" * 64},
                       "rsl_sources": {"rsl.py": "b" * 64},
                       "resume_inputs": {"runtime_versions": {}},
                       "correction_source_contract": {"runtime": {}}}
    current = {key: dict(source_protocol[key]) for key in ("source_files", "sdk_sources", "rsl_sources")}
    changed_file = next(iter(current[changed_group]))
    current[changed_group][changed_file] = "c" * 64
    monkeypatch.setattr(module.H, "source_maps", lambda _root: current)

    with pytest.raises(ValueError, match=f"{changed_group} changed"):
        module._check_training_runtime_sources(source_protocol, tmp_path)


def test_training_runtime_distribution_change_is_rejected(monkeypatch, tmp_path):
    module = _module()
    versions = {name: "1.0" for name in
                ("torch", "rsl-rl-lib", "isaaclab", "isaaclab_tasks", "isaaclab_rl")}
    runtime = {"implementation": "CPython", "version": [3, 10, 15], "cache_tag": "cpython-310"}
    source_protocol = {"source_files": {}, "sdk_sources": {}, "rsl_sources": {},
                       "resume_inputs": {"runtime_versions": versions},
                       "correction_source_contract": {"runtime": runtime}}
    monkeypatch.setattr(module.H, "source_maps", lambda _root: {
        "source_files": {}, "sdk_sources": {}, "rsl_sources": {}})
    monkeypatch.setattr(module, "_runtime_fingerprint", lambda _root: {
        "runtime": runtime, "distributions": {**versions, "torch": "2.0"}})

    with pytest.raises(ValueError, match="distributions changed"):
        module._check_training_runtime_sources(source_protocol, tmp_path)


@pytest.mark.parametrize("field,value", [("frame_count", 89), ("duration_s", 1.0), ("duration_s", float("nan"))])
def test_incomplete_gif_is_rejected(field, value):
    module = _module()
    probe = {"width": 960, "height": 180, "frame_count": 90, "duration_s": 6.0}
    module._validate_gif(probe)
    with pytest.raises(RuntimeError, match="frame and duration"):
        module._validate_gif({**probe, field: value})


def test_gpu_busy_records_nothing(monkeypatch, tmp_path):
    module = _module()
    protocol = {**module.FIXED, "result_status": "no_recovery"}
    monkeypatch.setattr(module, "verify", lambda *_args: (protocol, [], "c" * 40))
    monkeypatch.setattr(module.M, "query_gpu", lambda: {"busy": True})
    monkeypatch.setattr(module.M, "_record_clip", lambda *_args: pytest.fail("launcher entered"))

    assert module.record("source_tag", tmp_path) == {"status": "gpu_busy_no_launch", "clips_recorded": 0}


def test_cli_rejection_does_not_print_private_path(monkeypatch, tmp_path, capsys):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    private = tmp_path / "bench/private/training_correction/source_tag/result.json"
    monkeypatch.setattr(module, "prepare", lambda *_args: (_ for _ in ()).throw(
        ValueError(f"source binding changed: {private}")))

    assert module.main(["prepare", "source_tag", "--isaaclab", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    public_error = json.loads(captured.err)
    assert public_error == {"status": "rejected", "reason_code": "artifact_binding_mismatch",
                            "stage": "prepare", "error_type": "ValueError",
                            "rejection_id": public_error["rejection_id"]}
    assert len(public_error["rejection_id"]) == 32
    assert str(private) not in captured.err and "bench/private" not in captured.err
    records = list((tmp_path / "bench/private/correction_media_rejections/source_tag").glob("*.json"))
    assert len(records) == 1
    rejection = json.loads(records[0].read_text(encoding="utf-8"))
    assert rejection["rejection_id"] == public_error["rejection_id"]
    assert rejection["reason_code"] == "artifact_binding_mismatch"
    assert str(private) in rejection["detail"] and "traceback" in rejection


def test_record_publishes_three_clips_and_neutral_comparison_manifest(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    protocol_file = module.protocol_path("source_tag")
    protocol_file.parent.mkdir(parents=True)
    protocol_file.write_text("{}", encoding="utf-8")
    protocol = {**module.FIXED, "source_tag": "source_tag", "result_status": "no_recovery",
                "result_wording": "회복하지 못했다", "source_commit_sha": "s" * 40,
                "source_commit_scope": "experiment_execution_and_verdict_code_only",
                "publisher_sources": {"evals/publish_training_correction.py": "d" * 64},
                "code_sha256_lf": {}, "play_py_sha256": "p" * 64,
                "isaaclab_commit": "i" * 40, "public_sources": [],
                "private_sources": [{"id": "result", "sha256": "2" * 64}],
                "fixed_eval_sha256_lf": "f" * 64}
    inputs = []
    for clip_id in module.CLIP_IDS:
        clip = {"id": clip_id, "role": clip_id, "checkpoint_sha256": "c" * 64,
                "params_env_sha256": "e" * 64, "params_agent_sha256": "a" * 64}
        inputs.append({"clip": clip, "checkpoint": tmp_path / "unused", "params_dir": tmp_path})
    monkeypatch.setattr(module, "verify", lambda *_args: (protocol, inputs, "r" * 40))
    monkeypatch.setattr(module.M, "query_gpu", lambda: {"busy": False, "processes": []})

    def fake_record(_protocol, item, _isaaclab, private, _timeout):
        video = private / item["clip"]["id"] / "video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(item["clip"]["id"].encode())
        return {"clip": item["clip"], "video": video, "probe": module.FIXED["video_contract"],
                "own_pid_cleanup": True, "residual_processes_observed": 0}

    def fake_compose(public):
        mp4, gif = public / "comparison.mp4", public / "comparison.gif"
        mp4.write_bytes(b"mp4")
        gif.write_bytes(b"gif")
        return mp4, gif

    monkeypatch.setattr(module.M, "_record_clip", fake_record)
    monkeypatch.setattr(module, "_compose", fake_compose)
    def fake_probe(path):
        if path.suffix == ".gif":
            return {"width": 960, "height": 180, "frame_rate": "15/1", "frame_count": 90,
                    "duration_s": 6.0}
        return {**module.FIXED["video_contract"], "width": 1920, "height": 360}

    monkeypatch.setattr(module.M, "_probe_video", fake_probe)

    result = module.record("source_tag", tmp_path)
    manifest = json.loads((tmp_path / "docs/media/go2-correction-source_tag/source_manifest.json").read_text(encoding="utf-8"))

    assert result["status"] == "recorded" and result["clips"] == 3
    assert result["folder"] == "docs/media/go2-correction-source_tag"
    assert manifest["source_result_status"] == "no_recovery"
    assert manifest["gif_is_recovery_claim"] is False
    assert [row["id"] for row in manifest["clips"]] == list(module.CLIP_IDS)
    assert {row["frame_info"]["frame_count"] for row in manifest["clips"]} == {299}
    rendered = json.dumps(manifest)
    assert ("pid" not in rendered and "stdout" not in rendered and str(tmp_path) not in rendered
            and "bench/private" not in rendered)
