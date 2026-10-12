"""실제 Isaac Lab 소스에서 추출한 세 함수를 CPU에서 결함·복원 순서로 확인한다.

이 검사는 App, PhysX, 환경 또는 전체 SDK 모듈을 실행하지 않는다. 소스 전체를
compile한 뒤 등록 함수의 CodeType만 꺼내 최소 namespace에서 호출하는
``extracted_upstream_function_cpu_check``이다.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Any

import correction_adapter as A


ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS = {
    "F1": "restore_body_frame_reward",
    "F2": "restore_motor_torque_speed",
    "F6": "restore_configured_timeout",
}
INPUTS = {
    "F1": {
        "command_xy": [1.0, 0.0], "body_linear_velocity_xy": [1.0, 0.0],
        "world_linear_velocity_xy": [0.0, 1.0], "std": 0.5,
    },
    "F2": {
        "cases": [
            {"name": "positive_speed", "effort": 23.5, "joint_velocity": 6.0},
            {"name": "zero_speed_positive_limit", "effort": 23.5, "joint_velocity": 0.0},
            {"name": "zero_speed_negative_limit", "effort": -23.5, "joint_velocity": 0.0},
        ],
        "velocity_limit": 30.0, "saturation_effort": 23.5, "effort_limit": 23.5,
    },
    "F6": {"episode_length_step": 50, "maximum_episode_length": 1000},
}
GO2_SOURCE = "source/isaaclab_assets/isaaclab_assets/robots/unitree.py"


def _load_file(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compiled_code(isaaclab_root: Path, correction_id: str):
    spec = A.TARGETS[correction_id]
    path = (isaaclab_root / spec["source"]).resolve()
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    compiled = compile(raw, spec["source"], "exec", dont_inherit=True)
    return A._find_code(compiled, spec["qualname"]), path


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _install_extracted_modules(isaaclab_root: Path, torch) -> dict[str, ModuleType]:
    modules: dict[str, ModuleType] = {}
    for correction_id, spec in A.TARGETS.items():
        code, path = _compiled_code(isaaclab_root, correction_id)
        module = modules.get(spec["module"])
        if module is None:
            module = ModuleType(spec["module"])
            module.__file__ = str(path)
            modules[spec["module"]] = module
        function = FunctionType(code, {"torch": torch}, code.co_name)
        parts = spec["qualname"].split(".")
        if len(parts) == 1:
            setattr(module, parts[0], function)
        elif parts == ["DCMotor", "_clip_effort"]:
            module.DCMotor = type("DCMotor", (), {"_clip_effort": function})
        else:
            raise RuntimeError(f"unsupported extracted target {spec['qualname']!r}")

    packages = {
        "isaaclab": _package("isaaclab"),
        "isaaclab.envs": _package("isaaclab.envs"),
        "isaaclab.envs.mdp": _package("isaaclab.envs.mdp"),
        "isaaclab.actuators": _package("isaaclab.actuators"),
    }
    for name, package in packages.items():
        sys.modules[name] = package
    for name, module in modules.items():
        sys.modules[name] = module
        parent_name, child_name = name.rsplit(".", 1)
        setattr(sys.modules[parent_name], child_name, module)
    sys.modules["isaaclab"].envs = sys.modules["isaaclab.envs"]
    sys.modules["isaaclab"].actuators = sys.modules["isaaclab.actuators"]
    sys.modules["isaaclab.envs"].mdp = sys.modules["isaaclab.envs.mdp"]
    return modules


def _behavior(fault: str, modules: dict[str, ModuleType], torch) -> list[Any]:
    if fault == "F1":
        row = INPUTS[fault]
        asset = SimpleNamespace(data=SimpleNamespace(
            root_lin_vel_b=torch.tensor([row["body_linear_velocity_xy"]], dtype=torch.float64),
            root_lin_vel_w=torch.tensor([row["world_linear_velocity_xy"]], dtype=torch.float64),
        ))
        command = torch.tensor([row["command_xy"]], dtype=torch.float64)
        env = SimpleNamespace(
            scene={"robot": asset},
            command_manager=SimpleNamespace(get_command=lambda _name: command),
        )
        function = modules[A.TARGETS[CORRECTIONS[fault]]["module"]].track_lin_vel_xy_exp
        value = function(env, row["std"], "base_velocity", SimpleNamespace(name="robot"))
        return [float(item) for item in value.tolist()]
    if fault == "F2":
        row = INPUTS[fault]
        motor_class = modules[A.TARGETS[CORRECTIONS[fault]]["module"]].DCMotor
        motor = object.__new__(motor_class)
        motor._joint_vel = torch.tensor(
            [case["joint_velocity"] for case in row["cases"]], dtype=torch.float64,
        )
        motor._vel_at_effort_lim = torch.tensor(row["velocity_limit"], dtype=torch.float64)
        motor.velocity_limit = torch.tensor(row["velocity_limit"], dtype=torch.float64)
        motor._saturation_effort = torch.tensor(row["saturation_effort"], dtype=torch.float64)
        motor.effort_limit = torch.tensor(row["effort_limit"], dtype=torch.float64)
        effort = torch.tensor([case["effort"] for case in row["cases"]], dtype=torch.float64)
        value = motor_class._clip_effort(motor, effort)
        return [float(item) for item in value.tolist()]
    if fault == "F6":
        row = INPUTS[fault]
        env = SimpleNamespace(
            episode_length_buf=torch.tensor([row["episode_length_step"]]),
            max_episode_length=row["maximum_episode_length"],
        )
        function = modules[A.TARGETS[CORRECTIONS[fault]]["module"]].time_out
        return [bool(item) for item in function(env).tolist()]
    raise ValueError(f"unsupported fault {fault!r}")


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _go2_motor_binding(isaaclab_root: Path) -> dict[str, Any]:
    """GO2_CFG의 base_legs DCMotorCfg 수치를 소스 AST에서 읽어 입력값과 결합한다."""
    path = (isaaclab_root / GO2_SOURCE).resolve()
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    tree = ast.parse(raw, filename=GO2_SOURCE)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "UNITREE_GO2_CFG"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("UNITREE_GO2_CFG must appear exactly once in the asset source")
    motors = [node for node in ast.walk(assignments[0].value) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "DCMotorCfg"]
    if len(motors) != 1:
        raise ValueError("UNITREE_GO2_CFG must contain exactly one DCMotorCfg")
    keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in motors[0].keywords
                if keyword.arg in {"effort_limit", "saturation_effort", "velocity_limit"}}
    expected = {
        "effort_limit": INPUTS["F2"]["effort_limit"],
        "saturation_effort": INPUTS["F2"]["saturation_effort"],
        "velocity_limit": INPUTS["F2"]["velocity_limit"],
    }
    if keywords != expected:
        raise ValueError("GO2 motor constants differ from the component-check inputs")
    return {"source": GO2_SOURCE, "source_sha256": hashlib.sha256(raw).hexdigest(), "values": keywords}


def run_check(isaaclab_root: Path, fault: str, correction_id: str) -> dict[str, Any]:
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU component check")
    contract = A.source_contract(isaaclab_root)
    input_binding = _go2_motor_binding(isaaclab_root) if fault == "F2" else None
    modules = _install_extracted_modules(isaaclab_root, torch)
    snapshot = A.capture(isaaclab_root, contract)
    clean = _behavior(fault, modules, torch)
    hidden_faults = _load_file("hidden_faults_cpu_check", ROOT / "bench" / "hidden_faults.py")
    hidden_faults._before_env(fault)
    faulty = _behavior(fault, modules, torch)
    readback = A.apply(snapshot, correction_id)
    corrected = _behavior(fault, modules, torch)
    matching = correction_id == CORRECTIONS[fault]
    checks = {
        "clean_differs_from_fault": clean != faulty,
        "matching_restore_returns_clean": (corrected == clean) if matching else True,
        "matching_restore_changes_fault": (corrected != faulty) if matching else True,
        "wrong_target_retains_fault_behavior": (corrected == faulty) if not matching else True,
        "wrong_target_digest_change_count_zero": (len(readback["changed_targets"]) == 0) if not matching else True,
        "matching_readback_restores_selected": (
            readback["changed_targets"] == [correction_id]
            and readback["restored_targets"] == [correction_id]
        ) if matching else True,
    }
    cuda_initialized = torch.cuda.is_initialized()
    checks["cuda_remained_uninitialized"] = not cuda_initialized
    return {
        "schema": "correction_cpu_check_v1",
        "verification_type": "extracted_upstream_function_cpu_check",
        "scope": {
            "isaac_app_started": False, "physx_started": False,
            "full_sdk_modules_imported": False, "extracted_target_functions_executed": True,
            "performance_measurement": False, "gpu_used": False,
        },
        "fault": fault,
        "correction_id": correction_id,
        "matching_correction": matching,
        "inputs": INPUTS[fault],
        "input_binding": input_binding,
        "outputs": {"clean": clean, "fault": faulty, "corrected": corrected},
        "output_digests": {"clean": _digest(clean), "fault": _digest(faulty), "corrected": _digest(corrected)},
        "source_contract": contract,
        "raw_source_hashes": {
            key: row["source_sha256"] for key, row in contract["targets"].items()
        },
        "runtime_fingerprint": contract["runtime"],
        "torch": {"version": torch.__version__, "cuda_initialized": cuda_initialized},
        "correction_readback": readback,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab", type=Path, required=True)
    parser.add_argument("--fault", choices=sorted(CORRECTIONS))
    parser.add_argument("--correction", choices=sorted(A.TARGETS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contract-output", type=Path)
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    isaaclab_root = args.isaaclab.resolve()
    if args.contract_output is not None:
        if args.fault is not None or args.correction is not None or args.output is not None:
            parser.error("--contract-output cannot be combined with scenario arguments")
        _write_new(args.contract_output, A.source_contract(isaaclab_root))
        return 0
    if args.fault is None or args.correction is None or args.output is None:
        parser.error("scenario mode requires --fault, --correction, and --output")
    result = run_check(isaaclab_root, args.fault, args.correction)
    _write_new(args.output, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
