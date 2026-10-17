"""ANYmal-C flat과 U1 보상 함수군 교정의 CPU 적용성·NONE 무변경 의미를 확인한다."""
from __future__ import annotations

import argparse
import hashlib
import ast
import json
import os
from pathlib import Path
import sys
from types import FunctionType, ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import reward_family_correction as FAMILY  # noqa: E402
import upstream_faults as FAULTS  # noqa: E402

TASK = "Isaac-Velocity-Flat-Anymal-C-v0"
EXPECTED_WEIGHTS = {
    "track_lin_vel_xy_exp": 1.0,
    "track_ang_vel_z_exp": 0.5,
    "lin_vel_z_l2": -2.0,
    "ang_vel_xy_l2": -0.05,
}


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _task_terms(isaaclab: Path) -> tuple[dict, dict, dict]:
    """공통 reward 선언과 ANYmal 상속 경로를 AST로 읽어 앱 없이 실제 설정 결합을 확인한다."""
    velocity = isaaclab / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
    rough = isaaclab / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/rough_env_cfg.py"
    flat = isaaclab / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/flat_env_cfg.py"
    tree = ast.parse(velocity.read_text(encoding="utf-8"))
    rewards = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg")
    terms = {}
    for node in rewards.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in FAMILY.FUNCTIONS or not isinstance(node.value, ast.Call):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.value.keywords}
        func = keywords.get("func")
        weight = keywords.get("weight")
        if weight is None:
            raise ValueError("등록 보상 항의 weight가 없다")
        terms[name] = {
            "exists": True,
            "weight": ast.literal_eval(weight),
            "function_module": FAMILY.MODULE,
            "function_name": func.attr if isinstance(func, ast.Attribute) else None,
        }
    inherited_sources = {"velocity": velocity, "rough": rough, "flat": flat}
    rough_tree = ast.parse(rough.read_text(encoding="utf-8"))
    flat_tree = ast.parse(flat.read_text(encoding="utf-8"))

    def class_base(tree: ast.Module, name: str) -> list[str]:
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        return [ast.unparse(base) for base in cls.bases]

    def reward_override(tree: ast.Module, name: str) -> bool:
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            for target in targets:
                if (isinstance(target, ast.Attribute) and target.attr in {"weight", "func"}
                        and isinstance(target.value, ast.Attribute) and target.value.attr == name
                        and isinstance(target.value.value, ast.Attribute) and target.value.value.attr == "rewards"):
                    return True
        return False

    overrides = {name: reward_override(rough_tree, name) or reward_override(flat_tree, name)
                 for name in FAMILY.FUNCTIONS}
    inheritance = {
        "AnymalCRoughEnvCfg": class_base(rough_tree, "AnymalCRoughEnvCfg"),
        "AnymalCFlatEnvCfg": class_base(flat_tree, "AnymalCFlatEnvCfg"),
    }
    return terms, {name: terms[name]["weight"] for name in FAMILY.FUNCTIONS}, {
        "sources": {key: {"path": path.relative_to(isaaclab).as_posix(), "sha256_lf": sha_lf(path)}
                    for key, path in inherited_sources.items()},
        "registered_targets_overridden_in_anymal": overrides,
        "inheritance": inheritance,
    }


def _isolated_reward_module(isaaclab: Path, contract: dict) -> tuple[ModuleType, dict[str, object]]:
    """설치 source의 최상위 함수 code를 별도 모듈에 구성해 교정기의 실제 지문·변경 계약을 실행한다."""
    source = isaaclab / FAMILY.SOURCE
    compiled = compile(source.read_bytes(), FAMILY.SOURCE, "exec", dont_inherit=True)
    module = ModuleType(FAMILY.MODULE)
    module.__file__ = str(source.resolve())
    pristine = {}
    for name in FAMILY.FUNCTIONS:
        code = FAMILY._top_level_code(compiled, name)
        function = FunctionType(code, {"__name__": FAMILY.MODULE}, name)
        function.__module__ = FAMILY.MODULE
        setattr(module, name, function)
        pristine[name] = code
    return module, pristine


def inspect(isaaclab: Path) -> dict:
    if Path.cwd().resolve() != isaaclab.resolve():
        raise ValueError("Isaac Lab root에서 의미 검사를 실행해야 한다")
    contract = FAMILY.source_contract(isaaclab)
    module, pristine = _isolated_reward_module(isaaclab, contract)
    terms, weights, task_binding = _task_terms(isaaclab)
    applicable = all(
        terms[name] == {
            "exists": True,
            "weight": EXPECTED_WEIGHTS[name],
            "function_module": FAMILY.MODULE,
            "function_name": name,
        }
        for name in FAMILY.FUNCTIONS
    )
    original = sys.modules.get(FAMILY.MODULE)
    sys.modules[FAMILY.MODULE] = module
    snapshot = FAMILY.capture(isaaclab, contract)
    none = FAMILY.apply(snapshot, FAMILY.CORRECTION_ID)
    none_noop = (
        none["before_fault"] == none["after_fault"] == none["after_correction"]
        and none["changed_targets"] == []
        and none["restored_targets"] == []
        and none["unrelated_changed_after_fault"] == []
        and none["unrelated_changed_after_correction"] == []
    )
    fault_snapshot = FAMILY.capture(isaaclab, contract)
    injected_before = {name: FAMILY.BASE.fingerprint(getattr(module, name)) for name in FAMILY.FUNCTIONS}
    FAULTS.before_env("U1")
    injected = {name: FAMILY.BASE.fingerprint(getattr(module, name)) for name in FAMILY.FUNCTIONS}
    fault_changes_all = all(injected[name] != injected_before[name] for name in FAMILY.FUNCTIONS)
    restoration = FAMILY.apply(fault_snapshot, FAMILY.CORRECTION_ID)
    restored = {name: FAMILY.BASE.fingerprint(getattr(module, name)) for name in FAMILY.FUNCTIONS}
    if original is None:
        sys.modules.pop(FAMILY.MODULE, None)
    else:
        sys.modules[FAMILY.MODULE] = original
    restore_exact = restored == contract["functions"]
    restore_scope_clean = (
        restoration["restored_targets"] == list(FAMILY.FUNCTIONS)
        and restoration["unrelated_changed_after_fault"] == []
        and restoration["unrelated_changed_after_correction"] == []
    )
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("CPU 의미 검사에서 CUDA가 초기화됐다")
    checks = {
        "all_targets_exist_and_active": applicable,
        "weights_match_registered_task": weights == EXPECTED_WEIGHTS,
        "anymal_inherits_common_rewards_without_registered_overrides": (
            not any(task_binding["registered_targets_overridden_in_anymal"].values())
            and task_binding["inheritance"] == {
                "AnymalCRoughEnvCfg": ["LocomotionVelocityRoughEnvCfg"],
                "AnymalCFlatEnvCfg": ["AnymalCRoughEnvCfg"],
            }
        ),
        "none_correction_is_noop": none_noop,
        "u1_changes_all_registered_targets": fault_changes_all,
        "restore_returns_exact_source_fingerprints": restore_exact,
        "restore_changes_only_registered_family": restore_scope_clean,
        "cuda_not_initialized": True,
    }
    if not all(checks.values()):
        raise ValueError(f"조합 의미 검사 미통과: {[name for name, ok in checks.items() if not ok]}")
    return {
        "schema": "combination_semantics_result_v2",
        "status": "applicable",
        "robot": "ANYmal-C",
        "task": TASK,
        "fault": "U1",
        "correction_id": FAMILY.CORRECTION_ID,
        "active_targets": list(FAMILY.FUNCTIONS),
        "task_terms": terms,
        "task_binding": task_binding,
        "checks": checks,
        "source_contract": contract,
        "fault_fingerprints": {"before": injected_before, "after": injected, "restored": restored},
        "restoration_readback": restoration,
        "none_noop": none,
        "runtime": {"python": list(sys.version_info[:3]), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "source_sha256_lf": {
            "bench/combination_semantics_v1.py": sha_lf(Path(__file__)),
            "bench/reward_family_correction.py": sha_lf(ROOT / "bench/reward_family_correction.py"),
            "bench/upstream_faults.py": sha_lf(ROOT / "bench/upstream_faults.py"),
        },
        "api_calls": 0,
        "gpu_calls": 0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--isaaclab", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = inspect(args.isaaclab)
    if args.output is not None:
        write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
