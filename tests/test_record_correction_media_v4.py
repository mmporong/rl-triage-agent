"""v4 회복 방식 미디어 기록기의 CPU 전용 게이트 회귀. Isaac 재생·ffmpeg는 실행하지 않는다."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals", ROOT / "tests"):
    sys.path.insert(0, str(directory))

import record_correction_media_v4 as MEDIA  # noqa: E402
import test_publish_training_correction_v4 as PT  # noqa: E402

H = MEDIA.H
TAG = "reward_v4_s2026"


def _clips():
    return [{"id": clip_id, "role": role, "run_basename": f"run_{clip_id}", "checkpoint": checkpoint,
             "checkpoint_sha256": "c" * 64, "params_env_sha256": "e" * 64, "params_agent_sha256": "a" * 64,
             "overrides": MEDIA.FIXED["overrides"]}
            for clip_id, (role, checkpoint) in MEDIA.CLIP_CONTRACT.items()]


def _protocol():
    labels = {"resume_corrected": "not_recovered", "resume_corrected_std_reset": "recovered",
              "restart_corrected": "recovered"}
    return {**copy.deepcopy(MEDIA.FIXED), "source_tag": TAG, "seed": 2026, "result_status": "complete",
            "result_labels": labels, "result_wording": H.FIXED["result_wording"]["seed_summary"],
            "source_commit_sha": "c" * 40, "source_commit_scope": MEDIA.Q.SOURCE_COMMIT_SCOPE,
            "publisher_sources": {"evals/publish_training_correction_v4.py": "d" * 64},
            "source_protocol_sha256_lf": "3" * 64, "fixed_eval_sha256_lf": "f" * 64,
            "code_sha256_lf": {name: "1" * 64 for name in MEDIA.CODE}, "play_py_sha256": "2" * 64,
            "isaaclab_commit": H.FIXED["isaaclab_commit"],
            "public_sources": [{"path": f"evals/results/{TAG}/result.json", "sha256": "1" * 64}],
            "private_sources": [{"id": "result", "sha256": "2" * 64}], "clips": _clips()}


def test_fixed_contract_orders_six_clips_in_a_two_by_three_grid():
    assert MEDIA.CLIP_IDS == ("before_original", "after_resume_corrected", "after_restart_corrected",
                              "after_std_reset", "reference_clean_resume", "reference_original_none")
    assert {role for role, _ in MEDIA.CLIP_CONTRACT.values()} == {
        "original_fault_source", "original_clean_source", *H.ARMS}
    assert MEDIA.CLIP_CONTRACT["after_restart_corrected"] == ("restart_corrected", "model_299.pt")
    assert MEDIA.FIXED["grid"] == {"columns": 3, "rows": 2, "cell_width": 640, "cell_height": 360}
    assert MEDIA.FIXED["comparison_size"] == [1920, 720] and MEDIA.FIXED["gif_size"] == [960, 360]
    assert MEDIA.FIXED["model_selected_correction"] is False and MEDIA.FIXED["model_calls"] == 0
    assert {"evals/publish_training_correction_v4.py", "evals/run_training_correction_v4.py",
            "bench/record_correction_media_v4.py"} <= MEDIA.CODE
    assert json.loads(json.dumps(MEDIA.FIXED)) == MEDIA.FIXED
    MEDIA._validate_protocol_shape(_protocol(), TAG)


@pytest.mark.parametrize("mutation", [
    "clip_order", "checkpoint_escape", "role_change", "restart_checkpoint", "seed", "status", "label", "override",
])
def test_protocol_shape_rejects_tampering(mutation):
    protocol = _protocol()
    if mutation == "clip_order":
        protocol["clips"][1], protocol["clips"][2] = protocol["clips"][2], protocol["clips"][1]
    elif mutation == "checkpoint_escape":
        protocol["clips"][0]["run_basename"] = "../escape"
    elif mutation == "role_change":
        protocol["clips"][4]["role"] = "resume_corrected"
    elif mutation == "restart_checkpoint":
        protocol["clips"][2]["checkpoint"] = "model_598.pt"
    elif mutation == "seed":
        protocol["seed"] = 2027
    elif mutation == "status":
        protocol["result_status"] = "invalid"
    elif mutation == "label":
        protocol["result_labels"]["restart_corrected"] = "restoration_observed_single_run"
    else:
        protocol["clips"][3]["overrides"] = ["env.viewer.eye=[0,0,0]"]
    with pytest.raises(ValueError, match="clip order|clip contract|source identity"):
        MEDIA._validate_protocol_shape(protocol, TAG)


def _lineage():
    common = {"tag": TAG, "seed": 2026, "source_commit_sha": "c" * 40, "protocol_sha256_lf": "p" * 64}
    labels = _protocol()["result_labels"]
    return {"tag": TAG, "seed": 2026, "source_commit": "c" * 40, "source_protocol_sha": "p" * 64,
            "status": "complete", "labels": labels, "request": dict(common),
            "public_result": {**common, "numeric": {"status": "complete", "labels": dict(labels)}},
            "private_result": {**common, "status": "complete", "labels": dict(labels)}}


@pytest.mark.parametrize(("document", "field"), [
    ("request", "seed"), ("request", "source_commit_sha"), ("public_result", "tag"),
    ("public_result", "protocol_sha256_lf"), ("private_result", "status"), ("private_result", "labels"),
])
def test_result_lineage_rejects_cross_execution_fields(document, field):
    MEDIA._validate_result_lineage(**_lineage())
    lineage = _lineage()
    lineage[document][field] = "changed"
    with pytest.raises(ValueError, match="lineage differs"):
        MEDIA._validate_result_lineage(**lineage)


def _media_fixture(monkeypatch, tmp_path, **kwargs):
    private, results, isaaclab, tag = PT._fixture(monkeypatch, tmp_path, **kwargs)
    PT.PUB.export(tag, isaaclab)
    monkeypatch.setattr(MEDIA, "ROOT", tmp_path)
    originals = {}
    for fault, name in ((H.FAULT, "h02_s2026"), ("NONE", "baseline_p0c_s2026")):
        run = isaaclab / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-09_00-00-00_{name}"
        (run / "params").mkdir(parents=True)
        (run / "model_299.pt").write_bytes(name.encode())
        (run / "params/env.yaml").write_text("env", encoding="utf-8")
        (run / "params/agent.yaml").write_text("agent", encoding="utf-8")
        originals[fault] = {"checkpoint": run / "model_299.pt", "environment": run / "params/env.yaml",
                            "agent": run / "params/agent.yaml"}
    monkeypatch.setattr(H, "source_paths", lambda root, seed, fault: originals[fault])
    monkeypatch.setattr(MEDIA.M, "_check_hash", lambda path, expected, label: None)
    for name in MEDIA.CODE:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    (isaaclab / MEDIA.PLAY).parent.mkdir(parents=True, exist_ok=True)
    (isaaclab / MEDIA.PLAY).write_text("play", encoding="utf-8")
    return private, isaaclab, tag


def test_prepare_binds_exported_seed_bundle_into_six_clip_protocol(monkeypatch, tmp_path):
    _, isaaclab, tag = _media_fixture(monkeypatch, tmp_path)
    result = MEDIA.prepare(tag, isaaclab)
    assert result == {"status": "prepared_commit_before_recording",
                      "protocol": f"bench/protocols/training_correction_media_v4_{tag}.json"}
    protocol = json.loads((tmp_path / result["protocol"]).read_text(encoding="utf-8"))
    MEDIA._validate_protocol_shape(protocol, tag)
    assert protocol["seed"] == 2026 and protocol["result_status"] == "complete"
    assert protocol["result_labels"]["restart_corrected"] == "recovered"
    assert protocol["result_wording"] == H.FIXED["result_wording"]["seed_summary"]
    assert [clip["run_basename"] for clip in protocol["clips"]] == [
        "2026-10-09_00-00-00_h02_s2026", "2026-10-13_00-00-00_tag_resume_corrected",
        "2026-10-13_00-00-00_tag_restart_corrected", "2026-10-13_00-00-00_tag_resume_corrected_std_reset",
        "2026-10-13_00-00-00_tag_clean_resume_reference", "2026-10-09_00-00-00_baseline_p0c_s2026"]
    assert len(protocol["private_sources"]) == 3 + 4 * len(MEDIA.PRIVATE_FILES)
    assert all(row["path"].startswith(f"evals/results/{tag}/") for row in protocol["public_sources"])
    rendered = json.dumps(protocol)
    assert str(tmp_path) not in rendered and "bench/private" not in rendered
    MEDIA._check_public_artifacts(tag, protocol["public_sources"])
    MEDIA._check_private_artifacts(tag, protocol["private_sources"])


def test_prepare_refuses_partially_executed_seed(monkeypatch, tmp_path):
    _, isaaclab, tag = _media_fixture(monkeypatch, tmp_path, failed_arm="restart_corrected")
    with pytest.raises(ValueError, match="partially executed"):
        MEDIA.prepare(tag, isaaclab)
    assert not MEDIA.protocol_path(tag).exists()


def test_resolve_clips_reads_seed_originals_and_arm_receipts(monkeypatch, tmp_path):
    private, isaaclab, tag = _media_fixture(monkeypatch, tmp_path)
    MEDIA.prepare(tag, isaaclab)
    protocol = json.loads(MEDIA.protocol_path(tag).read_text(encoding="utf-8"))
    resolved = MEDIA._resolve_clips(protocol, isaaclab)
    assert [item["checkpoint"].name for item in resolved] == [
        "model_299.pt", "model_598.pt", "model_299.pt", "model_598.pt", "model_598.pt", "model_299.pt"]
    assert resolved[2]["checkpoint"].parent.name.endswith("_restart_corrected")
    assert resolved[5]["params_dir"].parent.name.endswith("baseline_p0c_s2026")


def test_record_checks_commit_before_gpu_or_launcher(monkeypatch, tmp_path):
    monkeypatch.setattr(MEDIA, "verify", lambda *_args: (_ for _ in ()).throw(ValueError("must be committed")))
    monkeypatch.setattr(MEDIA.M, "query_gpu", lambda: pytest.fail("GPU queried before commit gate"))
    with pytest.raises(ValueError, match="committed"):
        MEDIA.record(TAG, tmp_path)


def test_gpu_busy_records_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(MEDIA, "verify", lambda *_args: (_protocol(), [], "r" * 40))
    monkeypatch.setattr(MEDIA.M, "query_gpu", lambda: {"busy": True})
    monkeypatch.setattr(MEDIA.M, "_record_clip", lambda *_args: pytest.fail("launcher entered"))
    assert MEDIA.record(TAG, tmp_path) == {"status": "gpu_busy_no_launch", "clips_recorded": 0}


def test_compose_stacks_six_titled_cells_into_two_rows(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(MEDIA.V3M, "_run_ffmpeg", lambda args: calls.append(args))
    mp4, gif = MEDIA._compose(tmp_path)
    assert (mp4.name, gif.name) == ("comparison.mp4", "comparison.gif")
    inputs = [calls[0][index + 1] for index, value in enumerate(calls[0]) if value == "-i"]
    assert [Path(path).stem for path in inputs] == list(MEDIA.CLIP_IDS)
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert "[c0][c1][c2]hstack=inputs=3[top]" in graph and "[top][bottom]vstack=inputs=2[v]" in graph
    assert all(f"text='{title}'" in graph for title in MEDIA.CLIP_TITLES.values())
    assert "scale=960:-2" in calls[1][calls[1].index("-vf") + 1]


@pytest.mark.parametrize("field,value", [("height", 180), ("frame_count", 89), ("duration_s", float("nan"))])
def test_gif_contract_rejects_wrong_size_or_length(field, value):
    probe = {"width": 960, "height": 360, "frame_count": 90, "duration_s": 6.0}
    MEDIA._validate_gif(probe)
    with pytest.raises(RuntimeError, match="size, frame, and duration"):
        MEDIA._validate_gif({**probe, field: value})


def test_record_publishes_six_clips_and_neutral_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(MEDIA, "ROOT", tmp_path)
    protocol_file = MEDIA.protocol_path(TAG)
    protocol_file.parent.mkdir(parents=True)
    protocol_file.write_text("{}", encoding="utf-8")
    protocol = _protocol()
    inputs = [{"clip": clip, "checkpoint": tmp_path / "unused", "params_dir": tmp_path} for clip in protocol["clips"]]
    monkeypatch.setattr(MEDIA, "verify", lambda *_args: (protocol, inputs, "r" * 40))
    monkeypatch.setattr(MEDIA.M, "query_gpu", lambda: {"busy": False, "processes": []})

    def fake_record(_protocol, item, _isaaclab, private, _timeout):
        video = private / item["clip"]["id"] / "video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(item["clip"]["id"].encode())
        return {"clip": item["clip"], "video": video, "probe": MEDIA.FIXED["video_contract"],
                "own_pid_cleanup": True, "residual_processes_observed": 0}

    def fake_compose(public):
        mp4, gif = public / "comparison.mp4", public / "comparison.gif"
        mp4.write_bytes(b"mp4")
        gif.write_bytes(b"gif")
        return mp4, gif

    def fake_probe(path):
        if path.suffix == ".gif":
            return {"width": 960, "height": 360, "frame_rate": "15/1", "frame_count": 90, "duration_s": 6.0}
        return {**MEDIA.FIXED["video_contract"], "width": 1920, "height": 720}

    monkeypatch.setattr(MEDIA.M, "_record_clip", fake_record)
    monkeypatch.setattr(MEDIA, "_compose", fake_compose)
    monkeypatch.setattr(MEDIA.M, "_probe_video", fake_probe)
    result = MEDIA.record(TAG, tmp_path)
    manifest = json.loads((tmp_path / f"docs/media/go2-correction-{TAG}/source_manifest.json").read_text(encoding="utf-8"))
    assert result == {"status": "recorded", "source_result_status": "complete",
                      "folder": f"docs/media/go2-correction-{TAG}", "clips": 6}
    assert [row["id"] for row in manifest["clips"]] == list(MEDIA.CLIP_IDS)
    assert manifest["source_seed"] == 2026 and manifest["gif_is_recovery_claim"] is False
    assert manifest["source_result_labels"]["resume_corrected"] == "not_recovered"
    assert manifest["layout"]["order"] == list(MEDIA.CLIP_IDS)
    rendered = json.dumps(manifest)
    assert "pid" not in rendered and str(tmp_path) not in rendered and "bench/private" not in rendered


def test_cli_rejection_does_not_print_private_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(MEDIA, "ROOT", tmp_path)
    secret = tmp_path / "bench/private/training_correction_v4" / TAG / "result.json"
    monkeypatch.setattr(MEDIA, "prepare", lambda *_args: (_ for _ in ()).throw(
        ValueError(f"source binding changed: {secret}")))
    assert MEDIA.main(["prepare", TAG, "--isaaclab", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert captured.out == "" and error["reason_code"] == "artifact_binding_mismatch"
    assert str(secret) not in captured.err and "bench/private" not in captured.err
    records = list((tmp_path / "bench/private/correction_media_rejections" / TAG).glob("*.json"))
    assert len(records) == 1 and str(secret) in json.loads(records[0].read_text(encoding="utf-8"))["detail"]
