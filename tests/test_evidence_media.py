import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evidence_media", ROOT / "docs" / "demo" / "build_evidence_media.py")
assert SPEC is not None and SPEC.loader is not None
media = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = media
SPEC.loader.exec_module(media)

V1_RECORDER_SOURCE_COMMIT = "d35f88adb1ec40c8748d3a7188bdb87b74fc6a32"


@pytest.fixture(scope="module")
def verified_evidence():
    return media.verify_evidence(ROOT, media.DEFAULT_SNAPSHOT, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL)


@pytest.fixture(scope="module")
def recovery_repo(tmp_path_factory):
    temp_root = tmp_path_factory.mktemp("recovery-repo")
    protocol_relative = Path("bench/protocols/evidence_media_c01_v1.json")
    recorder_relative = Path("bench/record_evidence_media.py")
    for relative in (protocol_relative, recorder_relative, *(Path(path) for path in media.RECOVERY_PUBLIC_SOURCES)):
        target = temp_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == recorder_relative:
            historical = subprocess.run(
                ["git", "show", f"{V1_RECORDER_SOURCE_COMMIT}:{relative.as_posix()}"],
                cwd=ROOT, check=True, capture_output=True,
            ).stdout
            target.write_bytes(historical)
        else:
            shutil.copyfile(ROOT / relative, target)

    protocol = media.load_json(temp_root / protocol_relative)
    output = temp_root / "docs" / "media" / "synthetic-recovery-fixture"
    output.mkdir(parents=True)
    clips = []
    colors = {"before": "#315f9b", "true_rollback": "#26906a", "benign_rollback": "#bd7a2f"}
    for canonical in protocol["clips"]:
        clip_id = canonical["id"]
        video = output / f"{clip_id}.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "lavfi",
            "-i", f"color=c={colors[clip_id]}:s=1280x720:r=50", "-frames:v", "299",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(video),
        ], check=True)
        probe = media._probe_recovery_video(video)
        clips.append({
            "id": clip_id,
            "run_basename": canonical["run_basename"],
            "checkpoint_sha256": canonical["checkpoint_sha256"],
            "params_env_sha256": canonical["params_env_sha256"],
            "params_agent_sha256": canonical["params_agent_sha256"],
            "task": protocol["task"],
            "flags": {
                "seed": protocol["seed"], "num_envs": protocol["num_envs"],
                "video_length": protocol["video_length"], "kit_args": protocol["kit_args"],
                "overrides": canonical["overrides"],
            },
            "file": video.name, "size_bytes": video.stat().st_size,
            "clip_sha256": media.hashlib.sha256(video.read_bytes()).hexdigest(),
            "ffprobe": probe, "execution_exit": 0, "own_pid_cleanup": True,
        })
    manifest = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"],
        "scope": "old_checkpoint_playback", "interpretation": protocol["interpretation"],
        "same_case_as_active": False, "retraining_shown": False,
        "disallowed_claims": protocol["interpretation"]["disallowed_claims"],
        "clips": clips, "source_bindings": protocol["public_sources"],
        "gpu_admission": protocol["gpu_admission"],
        "gpu_observations_before_clips": [{
            "clip_id": clip["id"], "admission_policy": "desktop_shared_recording_v1", "busy": False,
            "gpu_exclusive": False, "performance_measurement": False, "gpu_index": 0,
            "compute_process_count": 0, "memory_used_mib": 1300, "utilization_percent": 15,
            "c_plus_g_contexts_present": True, "observed_at_utc": "2026-10-11T06:30:00+00:00",
        } for clip in clips],
        "implementation_bindings": {
            "protocol": {"path": protocol_relative.as_posix(), "sha256_lf": media.RECOVERY_PROTOCOL_SHA256_LF},
            "recorder": protocol["recorder_source"], "play_py_sha256": protocol["play_py_sha256"],
        },
    }
    manifest["gpu_observations_after_clips"] = [dict(row) for row in manifest["gpu_observations_before_clips"]]
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (output / "status.json").write_text(json.dumps({"state": "complete", "clips": 3}), encoding="utf-8")
    (output / "overlay.ppm").write_bytes(b"P6\n1280 720\n255\n" + b"\xff\xff\xff" * (1280 * 720))
    return temp_root, output / "source_manifest.json"


def test_frozen_public_snapshot_bindings_are_valid():
    evidence = media.verify_evidence(ROOT, media.DEFAULT_SNAPSHOT, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL)
    assert evidence.audit["valid"] is True
    assert evidence.selected["case"] == "h01_s2027"
    assert evidence.primary_count == 0
    assert evidence.selected["result"]["status"] == "hold"


def test_source_manifest_tamper_is_rejected(tmp_path):
    snapshot = json.loads(media.DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["sources"][0]["sha256_lf"] = "0" * 64
    tampered = tmp_path / "snapshot.json"
    tampered.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(media.EvidenceError, match="source LF SHA-256 불일치"):
        media.verify_evidence(ROOT, tampered, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL)


def test_receipt_digest_tamper_is_rejected():
    evidence = media.verify_evidence(ROOT, media.DEFAULT_SNAPSHOT, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL)
    receipt = dict(evidence.selected["result"]["observed"][0])
    receipt["receipt_digest"] = "0" * 64
    result = {**evidence.selected["result"], "observed": [receipt]}
    with pytest.raises(media.EvidenceError, match="receipt digest 불일치"):
        media._verify_receipts(ROOT, evidence.snapshot["tag"], evidence.selected["case"], result, evidence.source_hashes)


def test_wrong_existing_commit_cannot_claim_frozen_sources():
    wrong_commit = "43fa1da2b9e9a6c02aec0023de091051d9f47146"
    with pytest.raises(media.EvidenceError, match="동결 커밋 code source 불일치"):
        media.verify_evidence(ROOT, media.DEFAULT_SNAPSHOT, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL, wrong_commit)


def test_receipt_cannot_substitute_an_unrelated_registered_source_hash():
    evidence = media.verify_evidence(ROOT, media.DEFAULT_SNAPSHOT, media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL)
    receipt = dict(evidence.selected["result"]["observed"][0])
    receipt["report_sha256_lf"] = evidence.source_hashes["bench/protocols/active_diagnosis_v2.json"]
    result = {**evidence.selected["result"], "observed": [receipt]}
    with pytest.raises(media.EvidenceError, match="report/execution 결속 불일치"):
        media._verify_receipts(ROOT, evidence.snapshot["tag"], evidence.selected["case"], result, evidence.source_hashes)


def test_public_artifact_tamper_is_rejected(tmp_path):
    artifact = tmp_path / "card.png"
    artifact.write_bytes(b"verified card")
    digest = media.hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"artifacts": {
        "card.png": {"sha256": digest, "bytes": artifact.stat().st_size}
    }}), encoding="utf-8")
    media.verify_artifact_manifest(tmp_path, {"card.png"})
    artifact.write_bytes(b"tampered card")
    with pytest.raises(media.EvidenceError, match="artifact hash/size 불일치"):
        media.verify_artifact_manifest(tmp_path, {"card.png"})


def test_first_verified_primary_wins_over_registered_fallback():
    snapshot = {"registered_order": ["first", "second"], "cases": [
        {"case": "first", "state": "verified", "proof": False},
        {"case": "second", "state": "verified", "proof": True},
    ]}
    audit = {"cases": [
        {"case": "first", "valid": True, "proof_changed": False},
        {"case": "second", "valid": True, "proof_changed": True},
    ]}
    assert media.select_scene(snapshot, audit)["case"] == "second"


def test_no_primary_uses_first_registered_case_only():
    snapshot = {"registered_order": ["registered-first", "later"], "cases": [
        {"case": "later", "state": "verified", "proof": False},
        {"case": "registered-first", "state": "verified", "proof": False},
    ]}
    audit = {"cases": [
        {"case": "registered-first", "valid": True, "proof_changed": False},
        {"case": "later", "valid": True, "proof_changed": False},
    ]}
    assert media.select_scene(snapshot, audit)["case"] == "registered-first"


def test_verified_recovery_manifest_accepts_exact_three_clip_contract(recovery_repo):
    temp_root, manifest_path = recovery_repo
    verified = media.verify_recovery_media(temp_root, manifest_path)
    assert set(verified.clips) == {"before", "true_rollback", "benign_rollback"}
    assert all(verified.clips[clip]["ffprobe"]["frame_count"] == 299 for clip in verified.clips)


def test_current_default_v3_profile_binds_current_recorder_and_original_v1_protocol():
    recorder_spec = importlib.util.spec_from_file_location(
        "record_evidence_media", ROOT / "bench/record_evidence_media.py")
    assert recorder_spec is not None and recorder_spec.loader is not None
    recorder_module = importlib.util.module_from_spec(recorder_spec)
    recorder_spec.loader.exec_module(recorder_module)
    current = media.load_json(ROOT / "bench/protocols/evidence_media_c01_v3.json")
    superseded = media.load_json(ROOT / "bench/protocols/evidence_media_c01_v2.json")
    original = ROOT / "bench/protocols/evidence_media_c01_v1.json"
    recorder = ROOT / current["recorder_source"]["path"]
    assert recorder_module.DEFAULT_PROTOCOL == ROOT / "bench/protocols/evidence_media_c01_v3.json"
    assert current["recorder_source"]["sha256_lf"] == media.sha256_lf(recorder)
    # v2·v3은 같은 C01 녹화 계약의 소스 결속 갱신이며 둘 다 원래 v1 프로토콜을 가리킨다.
    assert current["previous_protocol_sha256_lf"] == superseded["previous_protocol_sha256_lf"] == media.sha256_lf(original)
    assert {key for key in current if current[key] != superseded.get(key)} == {"recorder_source"}
    assert current["recorder_source"] != superseded["recorder_source"]


def test_recovery_option_changes_only_the_two_separate_c01_cards(verified_evidence, recovery_repo):
    temp_root, manifest_path = recovery_repo
    recovery = media.verify_recovery_media(temp_root, manifest_path)
    without_recovery = media.card_spec(verified_evidence)["cards"]
    with_recovery = media.card_spec(verified_evidence, recovery)["cards"]

    assert "layout" not in without_recovery[6]
    assert "아직 없습니다" in without_recovery[6]["body"][-1]
    assert with_recovery[6]["layout"] == with_recovery[7]["layout"] == "reference_overlay"
    assert with_recovery[6]["left_label"] == "원복 전 기존 체크포인트"
    assert with_recovery[6]["right_label"] == "원인 변경 원복 체크포인트"
    assert "20초" in with_recovery[6]["footer"] and "5.98초" in with_recovery[6]["footer"]
    assert "1초" in with_recovery[7]["footer"] and "미결정" in with_recovery[7]["badge"]
    assert "판단이 달라진 사례 0/8" in with_recovery[8]["body"][0]


def test_recovery_chapters_fit_single_source_playback_and_gif_boundaries():
    sequence = media._sequence(True)
    assert sequence[-2:] == [("recovery", 5.9), ("control", 5.9)]
    assert sum(duration for _, duration in sequence[:6]) == 43
    assert sum(duration for _, duration in sequence[:7]) == 48.9
    assert sum(duration for _, duration in sequence) == 54.8
    recovery_windows = media._gif_windows(True)
    assert recovery_windows == [(6, 9), (15, 18), (22, 25), (29, 32), (43, 46), (48.9, 51.9)]
    assert sum(end - start for start, end in recovery_windows) == 18
    recovery_filter = media._gif_filter(recovery_windows)
    assert "[clip0][clip1][clip2][clip3][clip4][clip5]concat=n=6" in recovery_filter


def test_default_gif_keeps_five_scenes_and_fifteen_seconds():
    windows = media._gif_windows(False)
    assert windows == [(6, 9), (15, 18), (22, 25), (29, 32), (43, 46)]
    assert sum(end - start for start, end in windows) == 15
    assert "[clip0][clip1][clip2][clip3][clip4]concat=n=5" in media._gif_filter(windows)


@pytest.mark.parametrize("rejected_input", ["baseline_outside", "failure_outside", "missing"])
def test_render_rejects_untrusted_reference_video_before_output_creation(verified_evidence, tmp_path, monkeypatch, rejected_input):
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"private")
    baseline = outside if rejected_input == "baseline_outside" else media.DEFAULT_BASELINE_VIDEO
    failure = outside if rejected_input == "failure_outside" else media.DEFAULT_FAILURE_VIDEO
    if rejected_input == "missing":
        baseline = ROOT / "docs" / "media" / "missing-reference.mp4"
    output = tmp_path / "output"
    monkeypatch.setattr(media, "run", lambda command: pytest.fail(f"검증 전에 명령이 실행됨: {command}"))

    with pytest.raises(media.EvidenceError):
        media.render(
            verified_evidence, output, baseline, failure, media.DEFAULT_SNAPSHOT,
            media.DEFAULT_BRIDGE, media.DEFAULT_RELABEL,
        )
    assert not output.exists()


def test_render_rejects_reference_video_reparse_ancestor_before_output_creation(verified_evidence, tmp_path, monkeypatch):
    fake_root = tmp_path / "repo"
    media_root = fake_root / "docs" / "media"
    real_dir = media_root / "real-inputs"
    reference_dir = media_root / "linked-inputs"
    real_dir.mkdir(parents=True)
    (real_dir / "baseline.mp4").write_bytes(b"baseline")
    (real_dir / "failure.mp4").write_bytes(b"failure")
    if os.name == "nt":
        link_literal = str(reference_dir).replace("'", "''")
        target_literal = str(real_dir).replace("'", "''")
        subprocess.run([
            "powershell", "-NoProfile", "-Command",
            f"New-Item -ItemType Junction -Path '{link_literal}' -Target '{target_literal}' | Out-Null",
        ], check=True)
    else:
        os.symlink(real_dir, reference_dir, target_is_directory=True)
    baseline = reference_dir / "baseline.mp4"
    failure = reference_dir / "failure.mp4"
    output = tmp_path / "output"
    monkeypatch.setattr(media, "ROOT", fake_root)
    monkeypatch.setattr(media, "run", lambda command: pytest.fail(f"검증 전에 명령이 실행됨: {command}"))

    with pytest.raises(media.EvidenceError, match="symlink/reparse"):
        media.render(
            verified_evidence, output, baseline, failure, fake_root / "docs" / "demo" / "snapshot.json",
            fake_root / "bridge.json", fake_root / "relabel.json",
        )
    assert not output.exists()


def test_cli_snapshot_path_rejects_repo_private_and_outside_inputs(tmp_path):
    outside = tmp_path / "snapshot.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(media.EvidenceError, match="저장소 밖 source 경로"):
        media._snapshot_path(ROOT, outside)
    with pytest.raises(media.EvidenceError, match="docs/demo 밖"):
        media._snapshot_path(ROOT, ROOT / "bench" / "private" / "snapshot.json")


def test_recovery_pair_command_does_not_loop_source_clips(recovery_repo, tmp_path, monkeypatch):
    _, manifest_path = recovery_repo
    source_dir = manifest_path.parent
    commands = []
    monkeypatch.setattr(media, "run", commands.append)
    media._ffmpeg_pair(
        source_dir / "before.mp4", source_dir / "true_rollback.mp4", source_dir / "overlay.ppm",
        media.RECOVERY_DISPLAY_DURATION_S, tmp_path / "pair.mp4", loop=False,
    )
    assert "-stream_loop" not in commands[0]
    assert commands[0][commands[0].index("-t") + 1] == "5.9"


def test_recovery_pair_renders_one_chapter_below_source_duration(recovery_repo, tmp_path):
    _, manifest_path = recovery_repo
    source_dir = manifest_path.parent
    target = tmp_path / "pair.mp4"
    media._ffmpeg_pair(
        source_dir / "before.mp4", source_dir / "true_rollback.mp4", source_dir / "overlay.ppm",
        media.RECOVERY_DISPLAY_DURATION_S, target, loop=False,
    )
    probe = media._probe_media(target)
    assert float(probe["format"]["duration"]) == pytest.approx(5.9, abs=0.001)
    assert probe["streams"][0]["width"] == 1280
    assert probe["streams"][0]["height"] == 720


@pytest.mark.parametrize("attack", [
    "missing_clip", "partial_status", "wrong_clipset", "video_tamper",
    "source_binding", "embedded_probe", "cleanup", "unsafe_path", "protocol_tamper",
    "recorder_tamper",
    "manifest_symlink", "video_symlink", "parent_symlink", "gpu_exclusive", "gpu_busy", "gpu_memory",
    "gpu_timestamp", "gpu_contexts", "gpu_after_busy",
])
def test_recovery_manifest_rejects_partial_or_tampered_capture(recovery_repo, tmp_path, monkeypatch, attack):
    fixture_root, fixture_manifest = recovery_repo
    attacked_root = tmp_path / "repo"
    shutil.copytree(fixture_root, attacked_root)
    manifest_path = attacked_root / fixture_manifest.relative_to(fixture_root)
    manifest = media.load_json(manifest_path)

    if attack == "missing_clip":
        (manifest_path.parent / "before.mp4").unlink()
    elif attack == "partial_status":
        (manifest_path.parent / "status.json").write_text(json.dumps({"state": "recording", "clips": 2}), encoding="utf-8")
    elif attack == "wrong_clipset":
        manifest["clips"][0]["id"] = "unexpected"
    elif attack == "video_tamper":
        with (manifest_path.parent / "true_rollback.mp4").open("ab") as stream:
            stream.write(b"tamper")
    elif attack == "source_binding":
        manifest["source_bindings"][0]["sha256_lf"] = "0" * 64
    elif attack == "embedded_probe":
        manifest["clips"][0]["ffprobe"]["frame_count"] = 300
    elif attack == "cleanup":
        manifest["clips"][0]["own_pid_cleanup"] = False
    elif attack == "unsafe_path":
        manifest["clips"][0]["file"] = "../../../before.mp4"
    elif attack == "protocol_tamper":
        protocol_path = attacked_root / "bench" / "protocols" / "evidence_media_c01_v1.json"
        protocol = media.load_json(protocol_path)
        protocol["video_length"] = 301
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    elif attack == "recorder_tamper":
        recorder_path = attacked_root / "bench" / "record_evidence_media.py"
        recorder_path.write_bytes(recorder_path.read_bytes() + b"\n# tampered\n")
    elif attack == "gpu_exclusive":
        manifest["gpu_admission"]["gpu_exclusive"] = True
    elif attack == "gpu_busy":
        manifest["gpu_observations_before_clips"][0]["busy"] = True
    elif attack == "gpu_memory":
        manifest["gpu_observations_before_clips"][0]["memory_used_mib"] = 4096
    elif attack == "gpu_timestamp":
        manifest["gpu_observations_before_clips"][0]["observed_at_utc"] = "2026-10-11T06:30:00"
    elif attack == "gpu_contexts":
        del manifest["gpu_observations_before_clips"][0]["c_plus_g_contexts_present"]
    elif attack == "gpu_after_busy":
        manifest["gpu_observations_after_clips"][0]["busy"] = True
    elif attack == "manifest_symlink":
        original = media._is_reparse
        monkeypatch.setattr(media, "_is_reparse", lambda path: path == manifest_path or original(path))
    elif attack == "video_symlink":
        video = manifest_path.parent / "before.mp4"
        original = media._is_reparse
        monkeypatch.setattr(media, "_is_reparse", lambda path: path == video or original(path))
    elif attack == "parent_symlink":
        linked_dir = manifest_path.parent
        real_dir = linked_dir.with_name("real-capture")
        linked_dir.replace(real_dir)
        if os.name == "nt":
            link_literal = str(linked_dir).replace("'", "''")
            target_literal = str(real_dir).replace("'", "''")
            subprocess.run([
                "powershell", "-NoProfile", "-Command",
                f"New-Item -ItemType Junction -Path '{link_literal}' -Target '{target_literal}' | Out-Null",
            ], check=True)
        else:
            os.symlink(real_dir, linked_dir, target_is_directory=True)
        manifest_path = linked_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(media.EvidenceError):
        media.verify_recovery_media(attacked_root, manifest_path)
