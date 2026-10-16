from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "bench"))

import record_scope_transfer_media as S

ISAACLAB = Path.home() / "IsaacLab"


def recording_inputs_available() -> bool:
    if not (ISAACLAB / "scripts/reinforcement_learning/rsl_rl/play.py").is_file():
        return False
    if any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")):
        return False
    if any(not path.is_file() for path in S.PRIVATE_SOURCE_PATHS.values()):
        return False
    for protocol in protocols():
        if any(not (ISAACLAB / source["relative_path"]).is_file() for source in protocol["play_task_sources"]):
            return False
        for clip in protocol["clips"]:
            candidates = list((ISAACLAB / "logs" / "rsl_rl").glob(f"*/{clip['run_basename']}"))
            if not any(
                (folder / clip["checkpoint"]).is_file()
                and (folder / "params/env.yaml").is_file()
                and (folder / "params/agent.yaml").is_file()
                for folder in candidates
            ):
                return False
    return True


def protocols() -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in S.PROTOCOLS.values()]


HAS_INPUTS = recording_inputs_available()


def test_protocols_share_exact_play_and_camera_settings():
    u1, anymal = protocols()
    for key, value in S.FIXED.items():
        assert u1[key] == value
        assert anymal[key] == value
    assert u1["task"] != anymal["task"]
    assert tuple(clip["id"] for clip in u1["clips"]) == S.CLIP_IDS
    assert tuple(clip["id"] for clip in anymal["clips"]) == S.CLIP_IDS


def test_input_gate_does_not_convert_prepare_regression_into_skip(monkeypatch):
    if not HAS_INPUTS:
        pytest.skip("complete recording fixture is unavailable")
    monkeypatch.setattr(S, "prepare", lambda case, isaaclab: (_ for _ in ()).throw(ValueError("bad digest")))
    assert recording_inputs_available() is True


def test_public_protocols_hide_private_paths_and_disclose_result_informed_design():
    windows_user_root = "C:" + "\\Users\\"
    for path, protocol in zip(S.PROTOCOLS.values(), protocols(), strict=True):
        text = path.read_text(encoding="utf-8")
        assert windows_user_root not in text
        assert "bench/private" not in text
        assert all(set(item) == {"logical_id", "sha256"} for item in protocol["private_sources"])
        assert "결과를 본 뒤" in protocol["source_result_disclosure"]
        assert protocol["media_claim"] == "visual_playback_only_fixed_eval_labels_preserved"


@pytest.mark.skipif(not HAS_INPUTS, reason="recording inputs are not available")
@pytest.mark.parametrize("case", ["u1", "anymal"])
def test_prepare_binds_checkpoints_configs_and_results_without_gpu(case: str, monkeypatch):
    def forbidden_gpu():
        raise AssertionError("prepare must not query the GPU")

    monkeypatch.setattr(S.M, "query_gpu", forbidden_gpu)
    result = S.prepare(case, ISAACLAB)
    assert result["status"] == "prepared"
    assert result["gpu_calls"] == 0 and result["model_calls"] == 0
    assert len(result["clips"]) == 3


def test_source_digest_mutation_is_rejected(monkeypatch):
    protocol = protocols()[0]
    original = S.sha

    def changed(path: Path, lf: bool = False):
        if path == ROOT / protocol["public_sources"][0]["path"]:
            return "0" * 64
        return original(path, lf)

    monkeypatch.setattr(S, "sha", changed)
    with pytest.raises(ValueError, match="public source differs"):
        S._check_public_sources(protocol)


def test_clip_override_must_match_shared_play_settings():
    protocol = protocols()[0]
    protocol["clips"][0]["overrides"] = ["env.viewer.eye=[0.0,0.0,0.0]"]
    with pytest.raises(ValueError, match="override"):
        S._validate_protocol(protocol, "u1")


@pytest.mark.parametrize(
    ("section", "mutation", "message"),
    [
        ("public_sources", lambda rows: rows.append(dict(rows[0])), "public source inventory"),
        ("private_sources", lambda rows: rows.append({"logical_id": "unexpected", "sha256": "0" * 64}), "private source inventory"),
        ("clips", lambda rows: rows[0].update(role="healthy"), "clip identity"),
        ("play_task_sources", lambda rows: rows[0].update(role="environment"), "Play task source inventory"),
    ],
)
def test_case_inventory_is_exact(section, mutation, message):
    protocol = protocols()[0]
    mutation(protocol[section])
    with pytest.raises(ValueError, match=message):
        S._validate_protocol(protocol, "u1")


@pytest.mark.parametrize(
    ("section", "extra", "message"),
    [
        ("public_sources", {"unexpected": True}, "public source fields"),
        ("private_sources", {"unexpected": True}, "private source fields"),
        ("clips", {"unexpected": True}, "clip fields"),
        ("play_task_sources", {"unexpected": True}, "Play task source fields"),
    ],
)
def test_inventory_rows_reject_unregistered_fields(section, extra, message):
    protocol = protocols()[0]
    protocol[section][0].update(extra)
    with pytest.raises(ValueError, match=message):
        S._validate_protocol(protocol, "u1")


def test_unknown_or_mismatched_fixed_eval_label_is_rejected(monkeypatch):
    protocol = protocols()[0]
    original = S.read

    def changed(path: Path):
        value = original(path)
        if path.name == "summary.json":
            value["corrected_u1_restart"]["label"] = "unknown"
        return value

    monkeypatch.setattr(S, "read", changed)
    with pytest.raises(ValueError, match="policy binding"):
        S._validate_result_bindings(protocol, "u1", {})


def test_checkpoint_cannot_be_connected_to_another_policy(monkeypatch):
    protocol = protocols()[1]
    original = S.read
    corrected = protocol["clips"][1]
    private = {
        "anymal_corrected_eval": {
            "name": "restart_corrected",
            "checkpoint": {"sha256": corrected["checkpoint_sha256"]},
        }
    }

    def changed(path: Path):
        value = original(path)
        if path.name == "anymal_c_h02_s2047.json":
            value["checkpoint"]["sha256"] = protocol["clips"][2]["checkpoint_sha256"]
        return value

    monkeypatch.setattr(S, "read", changed)
    monkeypatch.setattr(S.GR, "_behavior", lambda report: {})
    monkeypatch.setattr(S.RC, "compare", lambda corrected_behavior, none_behavior: {})
    monkeypatch.setattr(S.RC, "verdict", lambda comparison, bands: {"label": "healthy"})
    with pytest.raises(ValueError, match="policy binding"):
        S._validate_result_bindings(protocol, "anymal", private)


def test_anymal_corrected_policy_must_rejudge_as_healthy(monkeypatch):
    protocol = protocols()[1]
    corrected = protocol["clips"][1]
    private = {
        "anymal_corrected_eval": {
            "name": "restart_corrected",
            "checkpoint": {"sha256": corrected["checkpoint_sha256"]},
        }
    }
    monkeypatch.setattr(S.GR, "_behavior", lambda report: {})
    monkeypatch.setattr(S.RC, "compare", lambda corrected_behavior, none_behavior: {})
    monkeypatch.setattr(S.RC, "verdict", lambda comparison, bands: {"label": "unknown"})
    with pytest.raises(ValueError, match="policy binding"):
        S._validate_result_bindings(protocol, "anymal", private)


def test_invalid_source_stops_before_gpu_query(monkeypatch):
    calls = 0

    def gpu():
        nonlocal calls
        calls += 1
        return {"busy": False}

    monkeypatch.setattr(S, "_check_code", lambda protocol: (_ for _ in ()).throw(ValueError("changed")))
    monkeypatch.setattr(S.M, "query_gpu", gpu)
    with pytest.raises(ValueError, match="changed"):
        S.record("u1", ISAACLAB, "must_not_launch", 900)
    assert calls == 0


def test_gpu_busy_record_returns_nonzero_for_sequential_runner(monkeypatch, capsys):
    monkeypatch.setattr(S, "record", lambda case, isaaclab, tag, timeout_s: {"status": "gpu_busy_no_launch", "clips_recorded": 0})
    exit_code = S.main(["record", "u1", "--tag", "busy_case", "--isaaclab", str(ISAACLAB)])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert output == {"status": "gpu_busy_no_launch", "clips_recorded": 0}


def test_existing_public_or_private_output_is_never_overwritten(tmp_path: Path):
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    with pytest.raises(FileExistsError):
        S.ensure_outputs_absent(public, private)
    public.rmdir()
    private.mkdir()
    with pytest.raises(FileExistsError):
        S.ensure_outputs_absent(public, private)


def test_manifest_contract_keeps_fixed_eval_labels_separate_from_video_claim():
    for protocol in protocols():
        assert [clip["fixed_eval_label"] for clip in protocol["clips"]] == ["unhealthy", "healthy", "healthy"]
        assert "영상만으로 회복을 판정했다" in protocol["claims_not_supported"]
        assert protocol["model_calls"] == 0
