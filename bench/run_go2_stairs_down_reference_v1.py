"""기존30초 관찰·인코더를 그대로 사용하고 하행 명령만0.75m/s로 바꾼다."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str, expected: str) -> ModuleType:
    path = ROOT / relative
    if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
        raise ValueError("동결된 하행 관찰·인코더 source가 다르다")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load("reference_stairs_base", "bench/run_go2_stairs_down_long_v1.py",
              "5c1f8a9417a3f29faeca546fdda495f92649429eb7e1573d8cff6e5f7719d0e9")
_BASE.__dict__["CLIP_ID"] = "quadruped-stairs-down-step-0p05-cmd-0p75-30s"
_ENCODER = _load("reference_stairs_encoder", "bench/run_go2_stairs_down_long_v2.py",
                 "47f6eb84f28e14f50bcd786b6d626e509118f9cdce3dcf64328a0d9460d662cc")
_ENCODER.__dict__["V1"] = _BASE
sha, read = _BASE.sha, _BASE.read
verify_record = _BASE.verify_record
CHECKPOINT_SHA = _BASE.CHECKPOINT_SHA
install_exact_frame_encoder = _ENCODER.install_exact_frame_encoder
ENCODER_SCHEMA = _ENCODER.ENCODER_SCHEMA


def make_spec(previous: dict) -> dict:
    value = _BASE.make_spec(previous)
    value["capture"]["command_x_m_s"] = 0.75
    return value


def load_base(repo: Path | None = None) -> ModuleType:
    base = _BASE.load_base(repo)
    base.__dict__["COMMAND_X_M_S"] = 0.75
    return base


def compare_prefix(old_trace: Path, new_trace: Path) -> dict:
    """명령이 다른 두 조건은 초기 상태만 비교하고 같은 궤적이라고 주장하지 않는다."""
    with old_trace.open(encoding="utf-8") as stream:
        old = json.loads(next(stream))
    with new_trace.open(encoding="utf-8") as stream:
        new = json.loads(next(stream))
    return {"rows_compared": 1, "same_initial_state": old == {key: new.get(key) for key in old},
            "old_trace_sha256": sha(old_trace), "new_trace_sha256": sha(new_trace),
            "command_differs": True, "trajectory_equality_claimed": False}
