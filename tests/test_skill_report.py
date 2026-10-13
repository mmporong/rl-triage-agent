"""공식 디버깅 스킬 체크리스트 보고서: 원문 결합, 체크포인트 차원, 근거 누락 처리, 실제 사례 회귀."""
from __future__ import annotations

from collections import OrderedDict
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import types
import zipfile

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import build_replay_page as PAGE  # noqa: E402
import skill_report as CLI  # noqa: E402
from rl_triage import skill_report as SR  # noqa: E402
from rl_triage.triage_tools import _TolerantLoader  # noqa: E402

ENV = yaml.load((ROOT / "bench/reference/params/p0c_seed2027/env.yaml").read_text(encoding="utf-8"), Loader=_TolerantLoader)


def _skill_copy(root: Path) -> Path:
    target = root / SR.SKILL_DIR
    shutil.copytree(ROOT / SR.SKILL_DIR, target)
    return target


def _fake_checkpoint(monkeypatch, path: Path, shapes: dict[str, tuple[int, ...]], iteration: int = 299) -> None:
    """torch 없이 torch 저장 형식(zip 안 data.pkl)과 같은 전역 참조를 가진 파일을 만든다."""
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(*args):
        raise AssertionError("not used while writing")
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    FloatStorage.__module__, FloatStorage.__qualname__ = "torch", "FloatStorage"
    torch.FloatStorage, utils._rebuild_tensor_v2 = FloatStorage, _rebuild_tensor_v2
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._utils", utils)

    class Storage:
        def __init__(self, numel):
            self.numel = numel

    class Tensor:
        def __init__(self, shape):
            self.shape = shape

        def __reduce__(self):
            numel = 1
            for value in self.shape:
                numel *= value
            return _rebuild_tensor_v2, (Storage(numel), 0, self.shape, (1,) * len(self.shape), False, OrderedDict())

    class Writer(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                return ("storage", FloatStorage, "0", "cpu", obj.numel)
            return None

    buffer = io.BytesIO()
    Writer(buffer, protocol=2).dump({"iter": iteration,
                                     "model_state_dict": OrderedDict((name, Tensor(shape)) for name, shape in shapes.items()),
                                     "optimizer_state_dict": {"state": {}, "param_groups": []}})
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{path.stem}/data.pkl", buffer.getvalue())


GO2_SHAPES = {"std": (12,), "actor.0.weight": (128, 48), "actor.0.bias": (128,),
              "actor.6.weight": (12, 128), "critic.0.weight": (128, 48)}


def test_load_skill_binds_pinned_text_and_counts():
    skill = SR.load_skill(ROOT)
    assert (len(skill["workflow"]), len(skill["validation"]), len(skill["decision_rules"])) == (9, 6, 5)
    assert skill["workflow"][4].startswith("Compare reward curves against task metrics")
    assert skill["commit_sha"] == "5c3ca250b400b9038a03d46494c1812c6b48dab4"


def test_load_skill_rejects_changed_skill_text(tmp_path):
    folder = _skill_copy(tmp_path)
    (folder / "SKILL.md").write_text((folder / "SKILL.md").read_text(encoding="utf-8") + "\nextra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned skill file changed"):
        SR.load_skill(tmp_path)


def test_checkpoint_shapes_reads_tensor_shapes_without_torch(monkeypatch, tmp_path):
    path = tmp_path / "model_299.pt"
    _fake_checkpoint(monkeypatch, path, GO2_SHAPES)
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.delitem(sys.modules, "torch._utils")
    shapes = SR.checkpoint_shapes(path)
    assert shapes["iteration"] == 299 and shapes["tensors"]["actor.0.weight"] == [128, 48]
    compat = SR.checkpoint_compatibility(shapes, ENV)
    assert compat["compatible"] is True and compat["config_policy_obs_dim"] == 48 and compat["action_dim"] == 12


def test_checkpoint_dimension_mismatch_is_reported(monkeypatch, tmp_path):
    path = tmp_path / "model_299.pt"
    _fake_checkpoint(monkeypatch, path, {**GO2_SHAPES, "actor.0.weight": (128, 45)})
    compat = SR.checkpoint_compatibility(SR.checkpoint_shapes(path), ENV)
    assert compat["compatible"] is False and compat["actor_obs_dim"] == 45


def test_checkpoint_reader_rejects_unlisted_globals(tmp_path):
    path = tmp_path / "model_299.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model_299/data.pkl", pickle.dumps({"iter": 1, "model_state_dict": os.getcwd}, protocol=2))
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        SR.checkpoint_shapes(path)


@pytest.mark.parametrize("mutation", ["unknown_term", "subset_action"])
def test_config_dims_refuses_terms_outside_the_table(mutation):
    env = json.loads(json.dumps(ENV, default=str))
    if mutation == "unknown_term":
        env["observations"]["policy"]["height_scan"] = {"func": "isaaclab.envs.mdp.observations:height_scan"}
        match = "outside the registered dimension table"
    else:
        env["actions"]["joint_pos"]["joint_names"] = [".*_hip_joint"]
        match = "joint count is not derivable"
    with pytest.raises(ValueError, match=match):
        SR.config_dims(env, 12)


def test_missing_inputs_stay_unmet_without_invented_evidence(tmp_path):
    _skill_copy(tmp_path)
    for relative in ("bench/runs_p0c/h02_s2027.telemetry.json", "evals/results/p0c_fixed_eval_20261009/runs/h02_s2027.json"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    inputs = SR.CaseInputs(case="h02_s2027", telemetry="bench/runs_p0c/h02_s2027.telemetry.json",
                           fixed_report="evals/results/p0c_fixed_eval_20261009/runs/h02_s2027.json",
                           loop_state="evals/loops/h02_s2027/state.json", env_config=None, checkpoint="logs/x/model_299.pt")
    report = SR.build_report(inputs, tmp_path, tmp_path / "IsaacLab")
    status = {item["id"]: item["status"] for section in report["items"].values() for item in section}
    assert status["W5"] == "met" and status["W3"] == "met"
    assert {status[key] for key in ("W2", "W7", "V2", "V3", "W9")} == {"unmet"}
    assert [row["path"] for row in report["evidence"]] == [
        "bench/runs_p0c/h02_s2027.telemetry.json", "evals/results/p0c_fixed_eval_20261009/runs/h02_s2027.json"]
    assert set(report["missing_inputs"]) == {"evals/loops/h02_s2027/state.json", "isaaclab:logs/x/model_299.pt"}
    assert all(item["evidence"] == [] for item in report["items"]["validation"] if item["id"] in ("V2", "V3"))


def _text_fields(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in ("path", "sha256", "text"):
                yield from _text_fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from _text_fields(child)
    elif isinstance(value, str):
        yield value


def test_real_h02_s2027_report_binds_existing_records_and_is_publishable():
    isaaclab = Path.home() / "IsaacLab"
    isaaclab = isaaclab if isaaclab.is_dir() else None
    report = SR.build_report(CLI.inputs_for("h02_s2027", isaaclab), ROOT, isaaclab)
    status = {item["id"]: item for section in report["items"].values() for item in section}
    assert status["V2"]["status"] == "met" and "reward" in status["V2"]["finding"]
    assert status["W1"]["status"] == "partial" and status["W8"]["status"] == "not_applicable"
    assert status["D2"]["applies"] is True and status["D1"]["applies"] is False
    assert "evals/loops/h02_s2027/correction_ledger.jsonl" in report["missing_inputs"]
    if isaaclab and (report["checkpoint_compatibility"] is not None):
        assert status["W7"]["status"] == "met" and report["checkpoint_compatibility"]["compatible"] is True
    rendered = json.dumps(report, ensure_ascii=False) + SR.render_markdown(report)
    assert not PAGE.path_violations(rendered.replace("isaaclab:logs/", "logs/"))
    assert not [problem for text in _text_fields(report) for problem in PAGE.claim_violations(text)]
