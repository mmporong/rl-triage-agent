"""로컬 F1 fresh restart 공개 실행기의 공통 순수 함수."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

SOURCE_FILES = (
    "bench/local_f1_restart_base_v2.py",
    "bench/local_f1_restart_prepare_v2.py",
    "bench/local_f1_restart_terminal_v2.py",
    "bench/local_f1_restart_child_v2.py",
    "bench/local_f1_restart_worker_v2.py",
    "bench/finalize_local_f1_restart_v2.py",
    "bench/protocols/local_f1_restart_s3221_v2.json",
    "bench/brev_recovery_worker_v2.py",
    "bench/brev_recovery_worker_v3.py",
    "bench/residual_training_v1.py",
    "bench/hidden_faults.py",
    "bench/optimizer_faults.py",
    "bench/r1_repair_training.py",
    "bench/residual_faults_v1.py",
    "bench/residual_probe_v1.py",
    "bench/extract_tb.py",
    "evals/fixed_eval.py",
    "evals/optimizer_observability.py",
    "bench/protocols/fixed_eval_v1.json",
    "src/rl_triage/correction_ledger.py",
    "src/rl_triage/correction_registry.py",
    "src/rl_triage/grounded_recovery.py",
    "src/rl_triage/probe_loop.py",
    "src/rl_triage/grounded_recovery.py",
    "src/rl_triage/recovery.py",
)


def repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / ".git").exists() and (parent / "bench/brev_recovery_worker_v3.py").is_file():
            return parent
    raise FileNotFoundError("rl-triage 저장소 root를 찾지 못했다")


ROOT = repo_root(Path(__file__))


def sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def binding(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"regular file이 아니다: {path}")
    return {"path": str(path.resolve()), "sha256": sha(path)}


def source_map(root: Path | None = None) -> dict[str, str]:
    root = ROOT if root is None else root
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source closure 누락: {missing}")
    return {name: sha(root / name, lf=True) for name in SOURCE_FILES}


def git_blob_map(root: Path, anchor: str, names: tuple[str, ...] = SOURCE_FILES) -> dict[str, str]:
    if subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{anchor}^{{commit}}"], check=False).returncode:
        raise ValueError("등록 anchor가 commit이 아니다")
    result = {}
    for name in names:
        raw = subprocess.run(["git", "-C", str(root), "show", f"{anchor}:{name}"],
                             capture_output=True, check=True).stdout
        result[name] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    return result


def judge(none: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    from rl_triage import grounded_recovery as GR

    n, c = none["metrics"], corrected["metrics"]
    numbers = [n.get(key) for key in ("lin_vel_rmse_mps", "yaw_rate_rmse_radps", "mean_survival_s", "fall_rate")]
    numbers += [c.get(key) for key in ("lin_vel_rmse_mps", "yaw_rate_rmse_radps", "mean_survival_s", "fall_rate")]
    standstill = none.get("standstill", {}).get("lin_vel_rmse_mps")
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in (*numbers, standstill)) or n["lin_vel_rmse_mps"] <= 0
            or n["yaw_rate_rmse_radps"] <= 0 or n["mean_survival_s"] <= 0 or standstill <= 0):
        return {"outcome": "unknown", "reference_invalid": True, "reason": "nonfinite_or_zero_reference",
                "ratios": None}
    behavior = GR._behavior(none)
    if not (behavior["err_xy"] < GR.WALK_FRACTION * standstill
            and behavior["fall_frac"] <= GR.REF_FALL_MAX):
        return {"outcome": "unknown", "reference_invalid": True, "reason": "none_reference_unhealthy",
                "ratios": None}
    ratios = {"err_xy_ratio": c["lin_vel_rmse_mps"] / n["lin_vel_rmse_mps"],
              "err_yaw_ratio": c["yaw_rate_rmse_radps"] / n["yaw_rate_rmse_radps"],
              "survival_ratio": c["mean_survival_s"] / n["mean_survival_s"],
              "fall_frac": c["fall_rate"]}
    recovered = (ratios["err_xy_ratio"] <= 1.1513 and ratios["err_yaw_ratio"] <= 1.1719
                 and ratios["survival_ratio"] >= 0.9 and ratios["fall_frac"] <= 0.05)
    return {"outcome": "recovered" if recovered else "not_recovered", "reference_invalid": False,
            "ratios": ratios}
