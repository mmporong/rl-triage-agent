"""E4 종단 시연 비교 영상 기록기의 CPU 전용 게이트 회귀. Isaac 재생은 하지 않고, ffmpeg 합성은 합성 영상으로만 확인한다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import build_replay_page as PAGE  # noqa: E402
import record_e2e_demo_media as R  # noqa: E402

E, DR, M = R.E, R.DR, R.M
TAG = "e2e_reward_s2029_test"
COMMIT = "c" * 40
PROTOCOL_SHA = "9" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _run(isaac: Path, name: str) -> Path:
    run = isaac / "logs/rsl_rl/unitree_go2_flat" / f"2026-10-13_00-00-00_{name}"
    (run / "params").mkdir(parents=True)
    (run / "model_299.pt").write_bytes(f"policy {name}".encode())
    (run / "model_0.pt").write_bytes(f"first {name}".encode())
    (run / "params/env.yaml").write_text(f"env {name}", encoding="utf-8")
    (run / "params/agent.yaml").write_text(f"agent {name}", encoding="utf-8")
    return run


def _files(run: Path) -> dict:
    return {key: {"path": str((run / rel).resolve()), "sha256": _sha(run / rel)} for key, rel in (
        ("checkpoint", "model_299.pt"), ("first_checkpoint", "model_0.pt"),
        ("environment", "params/env.yaml"), ("agent", "params/agent.yaml"))}


@pytest.fixture(params=["v1", "v2"])
def demo(request, monkeypatch, tmp_path):
    """시연 한 번의 공개·비공개 산출물을 그 판의 실제 단계 형식대로 임시 저장소에 만든다."""
    if request.param == "v2" and not (ROOT / "evals/e2e_reward_demo_v2.py").is_file():
        pytest.skip("v2 demo module is not in this checkout")
    schema = R.VERSIONS[request.param]
    mods = R._modules(schema)
    D, DRV = mods.E, mods.DR
    for module, name, value in ((R, "REPO", tmp_path), (D, "ROOT", tmp_path),
                                (D, "PRIVATE", tmp_path / "bench/private/e2e_reward_demo"),
                                (DRV, "PRIVATE", tmp_path / "bench/private/e2e_reward_demo"),
                                (D, "RESULTS", tmp_path / "evals/results")):
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(R, "protocol_path", lambda tag: tmp_path / R.protocol_relative(tag))
    isaac = tmp_path / "IsaacLab"
    (isaac / R.PLAY).parent.mkdir(parents=True)
    (isaac / R.PLAY).write_text("play", encoding="utf-8")
    case, clean = D.CASES["F1"], D.CASES["NONE"]
    runs = {"F1": _run(isaac, case), "NONE": _run(isaac, clean), "fix": _run(isaac, f"{DRV.fix_tag(TAG)}_{DRV.ARM}")}
    folder, fix = tmp_path / "bench/private/e2e_reward_demo" / TAG, tmp_path / "bench/private/e2e_reward_demo" / TAG / "fix"
    public = tmp_path / "evals/results" / TAG
    for name in (clean, case):
        _json(public / "runs" / f"{name}.json", {"name": name})
        (tmp_path / "bench/runs_p0c").mkdir(parents=True, exist_ok=True)
        (tmp_path / f"bench/runs_p0c/{name}.telemetry.json").write_text(name, encoding="utf-8")
    for relative in R.FIX_FILES:
        _json(folder / relative, {"file": relative})
    fix_files = _files(runs["fix"])
    receipt = {"checkpoint": fix_files["checkpoint"],
               "configs": {"environment": fix_files["environment"], "agent": fix_files["agent"]}}
    _json(fix / "train/receipt.json", {"completed": True, **receipt})
    fix_result = {"schema": "e2e_restart_correction_result", "tag": DRV.fix_tag(TAG), "case": case,
                  "arm": DRV.ARM, "restart_seed": D.RESTART_SEED, "reference_seed": D.SEED,
                  "correction_request_id": "rid1", "verdict": {"label": "healthy"}, "label": "recovered",
                  "standstill_ratio": 0.131, "costs": {"train_parent_wall_s": 650.0, "eval_parent_wall_s": 61.0},
                  "training_receipt_sha256": _sha(fix / "train/receipt.json"),
                  "fixed_report_sha256": _sha(fix / "eval/report.json"),
                  "reference": f"evals/results/{TAG}/runs/{clean}.json", "source_commit_sha": COMMIT}
    _json(fix / "result.json", fix_result)
    proposal = {"accepted": True, "decision": "propose", "agrees_with_rule": True, "reasons": [],
                "unmet_checklist": [], "rule_visible_to_model": True}
    extra = {
        "train_NONE": {"artifacts": {**_files(runs["NONE"]), "telemetry": {
            "path": f"bench/runs_p0c/{clean}.telemetry.json",
            "sha256_lf": _sha(tmp_path / f"bench/runs_p0c/{clean}.telemetry.json")}}},
        "train_F1": {"artifacts": {**_files(runs["F1"]), "telemetry": {
            "path": f"bench/runs_p0c/{case}.telemetry.json",
            "sha256_lf": _sha(tmp_path / f"bench/runs_p0c/{case}.telemetry.json")}}},
        "evaluate": {"reports": {"NONE": {"path": f"evals/results/{TAG}/runs/{clean}.json"}}},
        "correction": {"request_id": "rid1", "model_proposal": proposal,
                       "receipt": {"event": "receipt", "verdict": "recovered"}, "result": fix_result},
    }
    previous, step_sha = None, {}
    for name in D.STEPS:
        path = _json(D.step_path(TAG, name), {"step": name, "previous_sha256": previous, "source_commit_sha": COMMIT,
                                               "protocol_sha256_lf": PROTOCOL_SHA, **extra.get(name, {})})
        previous = step_sha[name] = _sha(path)
    result = {"schema": schema, "tag": TAG, "seed": D.SEED, "case": case,
              "outcome": "recovered", "demo_success": True, "success_conditions": {"restart_recovered": True},
              "unmet_conditions": [], "wording": "결과 문구.", "common_wording": "공통 문구.",
              "steps_sha256": step_sha, "terminal": None, "protocol_sha256_lf": PROTOCOL_SHA,
              "source_commit_sha": [COMMIT], "approver_identity_attested": False,
              "correction": {"request_id": "rid1", "model_proposal": proposal, "call": {"input_tokens": 1},
                             "receipt": {"event": "receipt", "verdict": "recovered"},
                             "result": {k: fix_result[k] for k in ("label", "verdict", "standstill_ratio", "costs")}}}
    _json(public / "demo_result.json", result)
    _json(folder / "result.json", result)
    executions = []
    monkeypatch.setattr(R, "_verify_demo_execution", lambda demo_ns, tag, root, commit: (
        executions.append((demo_ns.version, commit)) or ({"fixed_eval_sha256_lf": "f" * 64}, receipt)))
    return {"isaac": isaac, "runs": runs, "folder": folder, "public": public, "result": result,
            "executions": executions, "tmp": tmp_path, "version": request.param, "mods": mods, "E": D}


# --- 고정 계약 ---
def test_fixed_contract_keeps_play_seed_and_neutral_wording():
    assert R.CLIP_IDS == ("before_fault", "after_restart_fix", "reference_clean")
    assert not set(R.FIXED) & R.GENERATED_KEYS and R.FIXED["seed"] == 42
    assert R.FIXED["recording_model_calls"] == 0 and "model_selected_correction" not in R.FIXED
    assert R.FIXED["comparison_size"] == [1920, 400] and R.FIXED["gif_size"] == [960, 200]
    assert "고정 평가 수치" in R.FIXED["verdict_source"] and "고정 평가 수치" in R.INTERPRETATION
    # 승인은 사용자가 위임한 운영자 승인을 포함한다. 공개 문구에서 사람 승인으로 단정하지 않는다.
    assert "사람" not in R.INTERPRETATION + R.FIXED["verdict_source"] and "운영자 승인" in R.V1_APPROVAL
    assert "fixed-eval numbers" in R.CAPTION and all(R.OVERLAY_TEXT.fullmatch(t) for t in R.CLIP_TITLES.values())
    for text in (R.FIXED["verdict_source"], R.INTERPRETATION, R.CAPTION):
        assert PAGE.claim_violations(text) == [] and PAGE.path_violations(text) == []
    assert json.loads(json.dumps(R.FIXED)) == R.FIXED


# --- prepare ---
def test_build_protocol_binds_three_clips_and_public_counts_without_private_paths(demo):
    protocol = R.build_protocol(TAG, demo["isaac"])
    assert [(c["id"], c["role"], c["checkpoint"]) for c in protocol["clips"]] == [
        ("before_fault", "demo_fault_source", "model_299.pt"),
        ("after_restart_fix", "approved_restart_correction", "model_299.pt"),
        ("reference_clean", "demo_clean_source", "model_299.pt")]
    assert [c["run_basename"] for c in protocol["clips"]] == [demo["runs"][k].name for k in ("F1", "fix", "NONE")]
    D = demo["E"]
    assert (protocol["outcome"], protocol["demo_version"], protocol["demo_seed"], protocol["restart_seed"], protocol["seed"]) == (
        "recovered", demo["version"], D.SEED, D.RESTART_SEED, 42)
    assert set(protocol["code_sha256_lf"]) == R.code_for(demo["version"])
    assert protocol["correction_proposal"] == {"decision": "propose", "accepted": True, "agrees_with_rule": True,
                                               "unmet_checklist_count": 0, "operator_approved": True}
    assert protocol["correction_result"] == {"label": "recovered", "standstill_ratio": 0.131}
    assert [row["path"] for row in protocol["public_sources"]] == [
        f"evals/results/{TAG}/demo_result.json", f"evals/results/{TAG}/runs/{D.CASES['NONE']}.json",
        f"evals/results/{TAG}/runs/{D.CASES['F1']}.json"]
    assert [row["id"] for row in protocol["private_sources"]] == list(R.private_files(demo["mods"]))
    assert demo["executions"] == [(demo["version"], COMMIT)]
    rendered = json.dumps(protocol, ensure_ascii=False)
    assert str(demo["tmp"]) not in rendered and "bench/private" not in rendered and "IsaacLab" not in rendered


def _mutate(demo, mutation):
    folder, public = demo["folder"], demo["public"]
    if mutation == "terminal":
        _json(folder / "terminal.json", {"outcome": "invalid"})
    elif mutation == "public_only":
        _json(public / "demo_result.json", {**demo["result"], "wording": "다른 문구."})
    elif mutation == "not_judged":
        for path in (public / "demo_result.json", folder / "result.json"):
            _json(path, {**demo["result"], "outcome": "correction_unknown"})
    elif mutation == "fix_label":
        fix = json.loads((folder / "fix/result.json").read_text(encoding="utf-8"))
        _json(folder / "fix/result.json", {**fix, "label": "not_recovered"})
    elif mutation == "receipt_changed":
        receipt = json.loads((folder / "fix/train/receipt.json").read_text(encoding="utf-8"))
        _json(folder / "fix/train/receipt.json", {**receipt, "completed": False})
    elif mutation == "step_edited":
        step = demo["E"].step_path(TAG, "alarm")
        _json(step, {**json.loads(step.read_text(encoding="utf-8")), "results": {"F1": {"warned": False}}})
    elif mutation == "checkpoint_changed":
        (demo["runs"]["F1"] / "model_299.pt").write_bytes(b"changed")
    elif mutation == "private_missing":
        (folder / "fix/eval/jobs.json").unlink()
    elif mutation == "chain_broken":
        # 결과의 단계 SHA는 맞춰 두고 사슬만 끊는다(SHA 대조와 별개로 사슬 검사가 잡아야 한다).
        step = demo["E"].step_path(TAG, "alarm")
        _json(step, {**json.loads(step.read_text(encoding="utf-8")), "previous_sha256": "0" * 64})
        result = {**demo["result"], "steps_sha256": {**demo["result"]["steps_sha256"], "alarm": _sha(step)}}
        for path in (public / "demo_result.json", folder / "result.json"):
            _json(path, result)


@pytest.mark.parametrize(("mutation", "message"), [
    ("terminal", "ended early"), ("public_only", "differ"), ("not_judged", "did not reach"),
    ("fix_label", "lineage differs"), ("receipt_changed", "not bound"), ("step_edited", "hashes differ"),
    ("checkpoint_changed", "changed after training"), ("private_missing", "incomplete"),
    ("chain_broken", "chain is broken"),
])
def test_build_protocol_refuses_unverified_or_unfinished_demo(demo, mutation, message):
    _mutate(demo, mutation)
    with pytest.raises((ValueError, RuntimeError), match=message):
        R.build_protocol(TAG, demo["isaac"])
    assert not R.protocol_path(TAG).exists()


def test_after_clip_must_be_the_final_restart_checkpoint_in_its_run(demo, monkeypatch):
    run = demo["runs"]["fix"]
    (run / "model_100.pt").write_bytes(b"early")
    files = _files(run)
    early = {"checkpoint": {"path": str(run / "model_100.pt"), "sha256": _sha(run / "model_100.pt")},
             "configs": {"environment": files["environment"], "agent": files["agent"]}}
    monkeypatch.setattr(R, "_verify_demo_execution", lambda *_args: ({"fixed_eval_sha256_lf": "f" * 64}, early))
    with pytest.raises(ValueError, match="clip contract"):
        R.build_protocol(TAG, demo["isaac"])


def test_clip_rejects_a_non_final_checkpoint_or_foreign_params(demo):
    run = demo["runs"]["fix"]
    (run / "model_100.pt").write_bytes(b"early")
    files = _files(run)
    _, (checkpoint, params) = R._clip("after_restart_fix", files["checkpoint"], files["environment"], files["agent"])
    assert (checkpoint.name, params) == ("model_299.pt", (run / "params").resolve())
    early = {"path": str(run / "model_100.pt"), "sha256": _sha(run / "model_100.pt")}
    with pytest.raises(ValueError, match="checkpoint or params location"):
        R._clip("after_restart_fix", early, files["environment"], files["agent"])
    foreign = _files(demo["runs"]["NONE"])
    with pytest.raises(ValueError, match="checkpoint or params location"):
        R._clip("after_restart_fix", files["checkpoint"], foreign["environment"], files["agent"])


def test_demo_source_commit_swaps_and_restores_the_commit_gate():
    original = E.P
    with R._demo_source_commit(E, "a" * 40):
        assert isinstance(E.P, R.Q4._FrozenCommit) and E.P._source_commit == "a" * 40
    assert E.P is original
    with pytest.raises(RuntimeError):
        with R._demo_source_commit(E, "b" * 40):
            raise RuntimeError("inside")
    assert E.P is original
    with pytest.raises(ValueError, match="malformed"):
        with R._demo_source_commit(E, "HEAD"):
            pass


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_demo_execution_is_verified_at_the_demo_commit_for_each_version(version, tmp_path, monkeypatch):
    calls = []
    fake_p = object()

    def verify_protocol(*args):
        calls.append((args, type(module.P).__name__))
        return {"fixed_eval_sha256_lf": "f" * 64}, COMMIT
    module = SimpleNamespace(P=fake_p, verify_protocol=verify_protocol, read=lambda path: {"versions": {"torch": "x"}},
                             step_record=lambda tag, name: {"reports": {"NONE": {"path": "r.json"}}})
    restart = SimpleNamespace(fix_folder=lambda tag: tmp_path, execution_valid=lambda row: True,
                              verify_training=lambda folder, execution, protocol, versions: {"checkpoint": "ok"})
    demo_ns = SimpleNamespace(version=version, E=module, DR=restart)
    monkeypatch.setattr(R, "REPO", tmp_path)
    assert R._verify_demo_execution(demo_ns, TAG, tmp_path, COMMIT)[1] == {"checkpoint": "ok"}
    expected = ((tmp_path,), "_FrozenCommit") if version == "v1" else ((tmp_path, COMMIT), "object")
    assert calls == [expected] and module.P is fake_p


def test_demo_version_comes_from_the_public_result_schema(demo):
    assert R.demo_for(TAG).version == demo["version"]
    _json(demo["public"] / "demo_result.json", {**demo["result"], "schema": "e2e_reward_demo_result_v9"})
    with pytest.raises(ValueError, match="registered demo version"):
        R.demo_for(TAG)
    (demo["public"] / "demo_result.json").unlink()
    with pytest.raises(ValueError, match="summarize"):
        R.demo_for(TAG)


# --- prepare → verify → record 진입(실제 FIXED) ---
def test_prepare_verify_record_round_trip_with_real_fixed_contract(demo, monkeypatch):
    assert R.prepare(TAG, demo["isaac"]) == {"status": "prepared_commit_before_recording",
                                             "protocol": f"bench/protocols/e2e_demo_media_{TAG}.json"}
    protocol = json.loads(R.protocol_path(TAG).read_text(encoding="utf-8"))
    assert protocol["seed"] == R.FIXED["seed"] == 42 and protocol["demo_seed"] == demo["E"].SEED
    committed = []
    monkeypatch.setattr(R.shutil, "which", lambda tool: tool)
    monkeypatch.setattr(R.P.S, "_git_head", lambda _root: demo["E"].FIXED["isaaclab_commit"])
    monkeypatch.setattr(R.P.S, "_git_tracked_clean", lambda _root: True)
    monkeypatch.setattr(R.P, "verify_committed", lambda _root, paths: committed.append(paths) or "r" * 40)
    verified, inputs, commit = R.verify(TAG, demo["isaac"])
    assert (verified["seed"], commit) == (42, "r" * 40) and R.protocol_relative(TAG) in committed[0]
    assert [(i["clip"]["id"], i["checkpoint"].parent.name, i["params_dir"].name) for i in inputs] == [
        ("before_fault", demo["runs"]["F1"].name, "params"), ("after_restart_fix", demo["runs"]["fix"].name, "params"),
        ("reference_clean", demo["runs"]["NONE"].name, "params")]
    monkeypatch.setattr(R.M, "query_gpu", lambda: {"busy": True})
    assert R.record(TAG, demo["isaac"]) == {"status": "gpu_busy_no_launch", "clips_recorded": 0}
    R.protocol_path(TAG).write_text(json.dumps({**protocol, "seed": demo["E"].SEED}), encoding="utf-8")
    with pytest.raises(ValueError, match="recording contract changed"):
        R.verify(TAG, demo["isaac"])
    R.protocol_path(TAG).write_text(json.dumps(protocol), encoding="utf-8")
    _json(demo["public"] / "runs" / f"{demo['E'].CASES['F1']}.json", {"name": "edited"})
    with pytest.raises(ValueError, match="recording source binding changed"):
        R.verify(TAG, demo["isaac"])


def test_record_publishes_three_clips_with_neutral_manifest(demo, monkeypatch):
    R.prepare(TAG, demo["isaac"])
    protocol = json.loads(R.protocol_path(TAG).read_text(encoding="utf-8"))
    inputs = [{"clip": clip, "checkpoint": demo["tmp"] / "unused", "params_dir": demo["tmp"]} for clip in protocol["clips"]]
    monkeypatch.setattr(R, "ROOT", demo["tmp"])
    monkeypatch.setattr(R, "verify", lambda *_args: (protocol, inputs, "r" * 40))
    monkeypatch.setattr(R.M, "query_gpu", lambda: {"busy": False, "processes": []})

    def fake_record(_protocol, item, _isaaclab, private, _timeout):
        assert _protocol["seed"] == 42
        video = private / item["clip"]["id"] / "video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(item["clip"]["id"].encode())
        return {"clip": item["clip"], "video": video, "probe": R.FIXED["video_contract"],
                "own_pid_cleanup": True, "residual_processes_observed": 0}

    def fake_compose(folder):
        (folder / "comparison.mp4").write_bytes(b"mp4")
        (folder / "comparison.gif").write_bytes(b"gif")
        return folder / "comparison.mp4", folder / "comparison.gif"

    def fake_probe(path):
        if path.suffix == ".gif":
            return {"width": 960, "height": 200, "frame_rate": "15/1", "frame_count": 90, "duration_s": 6.0}
        return {**R.FIXED["video_contract"], "width": 1920, "height": 400}

    monkeypatch.setattr(R.M, "_record_clip", fake_record)
    monkeypatch.setattr(R, "_compose", fake_compose)
    monkeypatch.setattr(R.M, "_probe_video", fake_probe)
    assert R.record(TAG, demo["isaac"]) == {"status": "recorded", "outcome": "recovered",
                                            "folder": f"docs/media/go2-e2e-{TAG}", "clips": 3}
    manifest = json.loads((demo["tmp"] / f"docs/media/go2-e2e-{TAG}/source_manifest.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in manifest["clips"]] == list(R.CLIP_IDS)
    assert manifest["video_is_recovery_claim"] is False and manifest["after_checkpoint_equals_reference"] is False
    assert manifest["interpretation"] == R.INTERPRETATION and manifest["playback"]["seed"] == 42
    assert manifest["approval"] == getattr(demo["E"], "APPROVAL", R.V1_APPROVAL) and "신원 인증 아님" in manifest["approval"]
    rendered = json.dumps(manifest, ensure_ascii=False)
    assert '"pid"' not in rendered and str(demo["tmp"]) not in rendered and "bench/private" not in rendered


def test_cli_rejection_keeps_detail_private(demo, monkeypatch, capsys):
    monkeypatch.setattr(R, "ROOT", demo["tmp"])
    assert R.main(["record", TAG, "--isaaclab", str(demo["isaac"])]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "rejected" and "bench/private" not in json.dumps(error)
    assert len(list((demo["tmp"] / "bench/private/e2e_demo_media_rejections" / TAG).glob("*.json"))) == 1


# --- 합성 ---
def test_filter_graph_uses_plain_overlay_text_and_pads_a_caption_band():
    graph = R.filter_graph()
    assert "[c0][c1][c2]hstack=inputs=3[row]" in graph and "pad=1920:400:0:0:color=black" in graph
    assert all(f"text='{title}'" in graph for title in R.CLIP_TITLES.values()) and f"text='{R.CAPTION}'" in graph


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg is required")
def test_compose_meets_the_mp4_and_gif_contract_on_synthetic_clips(tmp_path):
    contract = R.FIXED["video_contract"]
    for index, clip_id in enumerate(R.CLIP_IDS):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        f"testsrc=size={contract['width']}x{contract['height']}:rate=50:duration=10",
                        "-frames:v", str(contract["frame_count"]), "-vf", f"hue=h={index * 90}", "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(tmp_path / f"{clip_id}.mp4")], check=True)
    mp4, gif = R._compose(tmp_path)
    R._validate_outputs(R.M._probe_video(mp4), R.M._probe_video(gif), contract)
