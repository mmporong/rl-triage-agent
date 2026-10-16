"""R1 실행 backend v1의 소유 작업 폴더 계약을 보완하는 최소 adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import r1_repair_execution as V1
import run_upstream_r1_smoke as R1

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "bench/private/codex_support_20261015/task21/runs"

TASK = V1.TASK
FIXED_EVAL_PROTOCOL = V1.FIXED_EVAL_PROTOCOL
sha = V1.sha
write_new = V1.write_new


def preflight(isaaclab: Path) -> dict[str, Any]:
    """프로세스를 시작하지 않고 Isaac Lab 실행·runner 경로의 존재만 확인한다."""
    root = Path(isaaclab).resolve()
    python = root / "_isaac_sim/python.bat"
    train_entry = root / "scripts/reinforcement_learning/rsl_rl/train.py"
    log_root = root / "logs/rsl_rl/unitree_go2_flat"
    if not root.is_dir():
        raise FileNotFoundError("Isaac Lab root가 없다")
    if not python.is_file():
        raise FileNotFoundError("Isaac Sim Python launcher가 없다")
    if not train_entry.is_file():
        raise FileNotFoundError("Isaac Lab RSL-RL train.py가 없다")
    if not log_root.is_dir():
        raise FileNotFoundError("Isaac Lab RSL-RL log root가 없다")
    runner_files = {name: (root / relative).resolve() for name, relative in R1.SOURCE_FILES.items()}
    missing = sorted(name for name, path in runner_files.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"필수 runner file이 없다: {missing}")
    private = PRIVATE_ROOT.resolve()
    if ROOT.resolve() not in private.parents:
        raise ValueError("private root가 저장소 밖을 가리킨다")
    return {
        "isaaclab_root": str(root),
        "python": str(python.resolve()),
        "train_entry": str(train_entry.resolve()),
        "log_root": str(log_root.resolve()),
        "runner_files": {name: str(path) for name, path in sorted(runner_files.items())},
        "private_root": str(private),
        "processes_started": 0,
        "gpu_queries": 0,
    }


def run_prepared_owned(folder: Path, args: list[str], cwd: Path, watchdog_s: float,
                       owned: Callable[..., Mapping[str, Any]] | None = None) -> Mapping[str, Any]:
    """실행 전제 검증 뒤 새 작업 폴더를 만들고 v1 runner 또는 주입 runner를 호출한다."""
    work = Path(folder)
    working_directory = Path(cwd)
    if not working_directory.is_dir():
        raise FileNotFoundError("owned 실행 cwd가 없다")
    if not isinstance(args, list) or not args:
        raise ValueError("owned 실행 argv가 비었다")
    executable = Path(args[0])
    if not executable.is_file():
        raise FileNotFoundError("owned 실행 파일이 없다")
    work.mkdir(parents=False, exist_ok=False)
    runner = owned or V1.OWNED.run_owned
    return runner(work, args, working_directory, watchdog_s)


def _prepared_owned(owned: Callable[..., Mapping[str, Any]] | None):
    def execute(folder: Path, args: list[str], cwd: Path, watchdog_s: float):
        return run_prepared_owned(folder, args, cwd, watchdog_s, owned)
    return execute


def train(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
          binding: Mapping[str, Any], owned: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    return V1.train(stage, isaaclab, arm, seed, run_name, binding, _prepared_owned(owned))


def inspect(stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
            binding: Mapping[str, Any], owned: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    return V1.inspect(stage, isaaclab, arm, seed, run_name, binding, _prepared_owned(owned))


def fixed_evaluate(stage: Path, isaaclab: Path,
                   artifacts: Mapping[str, Any] | Sequence[Mapping[str, Any]], run_name: str,
                   binding: Mapping[str, Any], owned=None) -> dict[str, Any]:
    return V1.fixed_evaluate(stage, isaaclab, artifacts, run_name, binding, _prepared_owned(owned))


def legacy_probe(stage: Path, isaaclab: Path,
                 artifacts: Mapping[str, Any] | Sequence[Mapping[str, Any]], run_name: str,
                 binding: Mapping[str, Any], owned=None, *, probes=None) -> dict[str, Any]:
    return V1.legacy_probe(stage, isaaclab, artifacts, run_name, binding,
                           _prepared_owned(owned), probes=probes)
