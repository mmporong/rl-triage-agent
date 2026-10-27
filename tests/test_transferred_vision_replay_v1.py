import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import build_transferred_vision_replay_v1 as R


def _write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_bytes(value)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root(tmp_path: Path) -> Path:
    target = tmp_path / R.COMPARISON
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / R.COMPARISON, target)
    return tmp_path


def _pending() -> dict:
    return {
        "schema": R.SCHEMA,
        "status": "pending",
        "measurement_scope": "registered_f1_two_seed_development_e2e",
        "fault_novelty": "registered_f1_not_unseen_fault",
        "selection_outcome": "observation_only",
        "selection_change_count": 0,
        "used_for_decision": False,
        "seeds": [],
    }


def _complete(root: Path) -> dict:
    rows = []
    comparison = R.validate_comparison(root)
    for seed in R.SEEDS:
        prefix = f"evals/results/e2e-s{seed}"
        fixed = Path(prefix) / "fixed-eval.json"
        fault = Path(prefix) / "fault.mp4"
        corrected = Path(prefix) / "corrected.mp4"
        fault_sha = _write(root / fault, f"fault-{seed}".encode())
        corrected_sha = _write(root / corrected, f"corrected-{seed}".encode())
        frames = []
        for phase in ("fault", "corrected"):
            path = Path(prefix) / f"{phase}.png"
            frames.append({"phase": phase, "path": path.as_posix(),
                           "sha256": _write(root / path, f"frame-{phase}-{seed}".encode()),
                           "frame_index": 25, "pts_s": 0.5})
        source_checkpoint = comparison["summary"]["source_labels"][f"source-0{1 if seed == 2220 else 2}"]["registered-f1"]["checkpoint_sha256"]
        corrected_checkpoint = hashlib.sha256(f"corrected-{seed}".encode()).hexdigest()
        fixed_sha = _write(root / fixed, {
            "source_fault_checkpoint_sha256": source_checkpoint,
            "corrected_checkpoint_sha256": corrected_checkpoint,
            "metrics": {"lin_vel_rmse_mps": 0.14, "fall_rate": 0.0},
        })
        rows.append({
            "seed": seed,
            "status": "recovered",
            "source_fault_checkpoint_sha256": source_checkpoint,
            "corrected_checkpoint_sha256": corrected_checkpoint,
            "selection": {"probe": "P_reward", "model_status": "valid", "used_for_decision": True},
            "probe": {"name": "P_reward", "outcome": "abnormal"},
            "correction": {"id": "restore_body_frame_reward", "recovery_mode": "restart", "registered": True},
            "fixed_eval": {"outcome": "recovered", "lin_vel_rmse_mps": 0.14, "fall_rate": 0.0,
                           "artifact": {"path": fixed.as_posix(), "sha256": fixed_sha}},
            "videos": {
                "fault": {"path": fault.as_posix(), "sha256": fault_sha},
                "corrected": {"path": corrected.as_posix(), "sha256": corrected_sha},
            },
            "frame_evidence": frames,
            "public_result_refs": [{"path": fixed.as_posix(), "sha256": fixed_sha}],
        })
    return {**_pending(), "status": "complete", "seeds": rows}


def test_actual_comparison_is_cross_validated():
    result = R.validate_comparison(ROOT)
    assert result["summary"]["validity"]["valid"] == 24
    assert len(result["choices"]) == 24
    assert {seed: row["outcomes"] for seed, row in result["pairs"].items()} == {
        2220: {"F1": "abnormal", "NONE": "normal"},
        2221: {"F1": "abnormal", "NONE": "normal"},
    }


def test_changed_comparison_terminal_binding_is_rejected(tmp_path):
    root = _root(tmp_path)
    path = root / R.COMPARISON / "model-selections.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["terminal_sha256"] = "0" * 64
    _write(path, rows)
    with pytest.raises(ValueError, match="선택 원문 해시"):
        R.validate_comparison(root)


def test_pending_renders_no_invented_recovery_media_or_frames(tmp_path):
    root = _root(tmp_path)
    manifest = Path("pending.json")
    _write(root / manifest, _pending())
    rendered = R.build(root, manifest, Path("out/replay.html"))
    assert rendered.count("E2E 결과 대기") == 2
    assert "<video" not in rendered and "<img" not in rendered
    assert "2/2 recovered" not in rendered
    assert "영상에 따른 선택 변화</b><span>0회" in rendered
    assert "후속 영상 관찰은 회복 판정에 사용하지 않음" in rendered


def test_complete_manifest_binds_two_seed_media_frames_and_metrics(tmp_path):
    root = _root(tmp_path)
    manifest = _complete(root)
    verified = R.validate_e2e(manifest, root)
    assert verified["status"] == "complete" and len(verified["seeds"]) == 2
    path = Path("e2e.json")
    _write(root / path, manifest)
    rendered = R.build(root, path, Path("replay/replay.html"))
    assert rendered.count("<video") == 4
    assert rendered.count("<img") == 4
    assert rendered.count("등록 보상 복원 · restart") == 2
    assert rendered.count("RMSE 0.140 m/s · 낙상률 0.000") == 2


@pytest.mark.parametrize("mutation", ["scope", "selection", "probe", "correction", "metric", "video", "frame", "ref"])
def test_complete_manifest_rejects_changed_or_unbound_evidence(tmp_path, mutation):
    root = _root(tmp_path)
    manifest = _complete(root)
    changed = copy.deepcopy(manifest)
    row = changed["seeds"][0]
    if mutation == "scope":
        changed["fault_novelty"] = "unseen"
    elif mutation == "selection":
        row["selection"]["probe"] = "P_value"
    elif mutation == "probe":
        row["probe"]["outcome"] = "normal"
    elif mutation == "correction":
        row["correction"]["recovery_mode"] = "resume"
    elif mutation == "metric":
        row["fixed_eval"]["fall_rate"] = 0.1
    elif mutation == "video":
        row["videos"]["fault"]["sha256"] = "0" * 64
    elif mutation == "frame":
        row["frame_evidence"] = row["frame_evidence"][:1]
    else:
        row["public_result_refs"] = []
    with pytest.raises(ValueError):
        R.validate_e2e(changed, root)


def test_fixed_1920x1080_layout_exposes_runtime_overflow_measurement(tmp_path):
    root = _root(tmp_path)
    manifest = Path("pending.json")
    _write(root / manifest, _pending())
    rendered = R.build(root, manifest, Path("out/replay.html"))
    assert "width:1920px;height:1080px;overflow:hidden" in rendered
    assert "overflowX:Math.max(0,s.scrollWidth-1920)" in rendered
    assert "overflowY:Math.max(0,s.scrollHeight-1080)" in rendered
    assert "v.dataset.overflow=" in rendered


def test_cli_write_once_then_check(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path)
    _write(root / "pending.json", _pending())
    monkeypatch.setattr(R, "ROOT", root)
    argv = ["--e2e-manifest", "pending.json", "--output", "out/replay.html"]
    assert R.main(argv) == 0
    assert R.main([*argv, "--check"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "current"
    with pytest.raises(FileExistsError):
        R.main(argv)


def test_rendered_page_has_no_private_or_user_path(tmp_path):
    root = _root(tmp_path)
    manifest = Path("pending.json")
    _write(root / manifest, _pending())
    rendered = R.build(root, manifest, Path("out/replay.html"))
    assert "bench/" + "private" not in rendered
    assert "LIM" + "MM" not in rendered
    assert ":\\" not in rendered
    assert "직" + "접" not in rendered
