"""ANYmal-C LSTM 액추에이터의 F2 의미를 보강된 CPU 정적 검사로 확인한다."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "bench/protocols/anymal_lstm_motor_semantics_v2.json"


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_paths(isaaclab: Path) -> dict[str, Path]:
    return {
        "validator": Path(__file__).resolve(),
        "actuator_net": isaaclab / "source/isaaclab/isaaclab/actuators/actuator_net.py",
        "actuator_pd": isaaclab / "source/isaaclab/isaaclab/actuators/actuator_pd.py",
        "anymal_asset": isaaclab / "source/isaaclab_assets/isaaclab_assets/robots/anymal.py",
        "fault": ROOT / "bench/hidden_faults.py",
    }


def sdk_python_files(isaaclab: Path) -> list[Path]:
    root = isaaclab / "source/isaaclab/isaaclab"
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def sdk_manifest_sha256(isaaclab: Path) -> str:
    root = isaaclab / "source/isaaclab/isaaclab"
    digest = hashlib.sha256()
    for path in sdk_python_files(isaaclab):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha_lf(path)))
        digest.update(b"\n")
    return digest.hexdigest()


def class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    rows = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(rows) != 1:
        raise ValueError(f"class {name} definition count differs")
    return rows[0]


def function_node(nodes, name: str) -> ast.FunctionDef:
    rows = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(rows) != 1:
        raise ValueError(f"function {name} definition count differs")
    return rows[0]


def references_self_attribute(node: ast.AST, attribute: str) -> bool:
    return any(
        isinstance(child, ast.Attribute)
        and child.attr == attribute
        and isinstance(child.value, ast.Name)
        and child.value.id == "self"
        for child in ast.walk(node)
    )


def self_attribute_written(function: ast.FunctionDef, attribute: str) -> bool:
    """Detect direct, slice/subscript, augmented, and in-place copy writes."""
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            if any(references_self_attribute(target, attribute) for target in node.targets):
                return True
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if references_self_attribute(node.target, attribute):
                return True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"copy_", "fill_", "zero_"} and references_self_attribute(node.func.value, attribute):
                return True
    return False


def self_attribute_initialized_with_zeros(function: ast.FunctionDef, attribute: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(references_self_attribute(target, attribute) for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            if isinstance(value.func.value, ast.Name) and value.func.value.id == "torch":
                if value.func.attr in {"zeros", "zeros_like"}:
                    return True
    return False


def calls_self_method(function: ast.FunctionDef, method: str, argument_attribute: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not (node.func.attr == method and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"):
            continue
        if len(node.args) == 1 and isinstance(node.args[0], ast.Attribute):
            arg = node.args[0]
            if isinstance(arg.value, ast.Name) and arg.value.id == "self" and arg.attr == argument_attribute:
                return True
    return False


def assigns_name_into_self_buffer(function: ast.FunctionDef, buffer: str, name: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(references_self_attribute(target, buffer) for target in targets):
            continue
        if any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node.value)):
            return True
    return False


def literal_config(tree: ast.Module) -> dict:
    assignment = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "ANYDRIVE_3_LSTM_ACTUATOR_CFG" for target in node.targets)
        ),
        None,
    )
    if assignment is None or not isinstance(assignment.value, ast.Call):
        raise ValueError("ANYDRIVE_3_LSTM_ACTUATOR_CFG assignment differs")
    values = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in assignment.value.keywords
        if keyword.arg in {"saturation_effort", "effort_limit", "velocity_limit"}
    }
    if set(values) != {"saturation_effort", "effort_limit", "velocity_limit"}:
        raise ValueError("ANYdrive motor limit configuration differs")
    return values


def anymal_c_uses_lstm(tree: ast.Module) -> bool:
    assignment = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "ANYMAL_C_CFG" for target in node.targets)
        ),
        None,
    )
    if assignment is None or not isinstance(assignment.value, ast.Call):
        return False
    actuators = next((keyword.value for keyword in assignment.value.keywords if keyword.arg == "actuators"), None)
    if not isinstance(actuators, ast.Dict) or len(actuators.keys) != 1:
        return False
    return (
        isinstance(actuators.keys[0], ast.Constant)
        and actuators.keys[0].value == "legs"
        and isinstance(actuators.values[0], ast.Name)
        and actuators.values[0].id == "ANYDRIVE_3_LSTM_ACTUATOR_CFG"
    )


def compile_function(node: ast.FunctionDef, torch):
    copy = ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=node.body,
        decorator_list=[],
        returns=node.returns,
        type_comment=node.type_comment,
    )
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy], type_ignores=[])), "<ast-contract>", "exec"), namespace)
    return namespace[node.name]


def extract_fault_clip(tree: ast.Module) -> ast.FunctionDef:
    before = function_node(tree.body, "_before_env")
    rows = [node for node in ast.walk(before) if isinstance(node, ast.FunctionDef) and node.name == "_clip_effort"]
    if len(rows) != 1:
        raise ValueError("F2 nested _clip_effort definition count differs")
    return rows[0]


def enclosing_names(tree: ast.Module, target: ast.AST) -> tuple[str | None, str | None]:
    owner_class = None
    owner_function = None

    def visit(node: ast.AST, classes: list[str], functions: list[str]) -> bool:
        nonlocal owner_class, owner_function
        if node is target:
            owner_class = classes[-1] if classes else None
            owner_function = functions[-1] if functions else None
            return True
        next_classes = classes + [node.name] if isinstance(node, ast.ClassDef) else classes
        next_functions = functions + [node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else functions
        return any(visit(child, next_classes, next_functions) for child in ast.iter_child_nodes(node))

    visit(tree, [], [])
    return owner_class, owner_function


def sdk_joint_vel_accesses(isaaclab: Path) -> list[dict]:
    root = isaaclab / "source/isaaclab/isaaclab"
    rows = []
    for path in sdk_python_files(isaaclab):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr != "_joint_vel":
                continue
            owner_class, owner_function = enclosing_names(tree, node)
            base = ast.unparse(node.value)
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "line": node.lineno,
                    "class": owner_class,
                    "function": owner_function,
                    "base": base,
                    "context": type(node.ctx).__name__,
                }
            )
    return sorted(rows, key=lambda row: (row["path"], row["line"], row["base"]))


class MotorState:
    pass


def numeric_grid(protocol: dict, normal, fault, torch) -> dict:
    cfg = protocol["configuration"]
    velocities = torch.tensor(protocol["grid"]["joint_velocity"], dtype=torch.float64)
    efforts = torch.tensor(protocol["grid"]["requested_effort"], dtype=torch.float64)
    velocity_grid, effort_grid = torch.meshgrid(velocities, efforts, indexing="ij")
    state = MotorState()
    state._joint_vel = velocity_grid.clone()
    state._zeros_effort = torch.zeros_like(velocity_grid)
    state._saturation_effort = cfg["saturation_effort"]
    state.effort_limit = cfg["effort_limit"]
    state.velocity_limit = cfg["velocity_limit"]
    state._vel_at_effort_lim = state.velocity_limit * (1.0 + state.effort_limit / state._saturation_effort)
    normal_output = normal(state, effort_grid.clone())
    state._joint_vel = velocity_grid.clone()
    fault_output = fault(state, effort_grid.clone())
    state._joint_vel = velocity_grid.clone()
    restored_output = normal(state, effort_grid.clone())
    zero = velocities == 0
    nonzero = velocities != 0
    return {
        "zero_velocity_equal": bool(torch.equal(normal_output[zero], fault_output[zero])),
        "nonzero_difference_count": int(torch.count_nonzero(normal_output[nonzero] != fault_output[nonzero]).item()),
        "restore_equal": bool(torch.equal(normal_output, restored_output)),
        "max_abs_difference": float(torch.max(torch.abs(normal_output - fault_output)).item()),
    }


def decide_outcome(structural: dict, grid: dict) -> str:
    required_true = (
        "inherits_dc_motor",
        "calls_inherited_clip",
        "dc_init_zeros_clip_joint_velocity",
        "dc_compute_writes_clip_joint_velocity",
        "network_input_uses_joint_velocity",
        "anymal_c_legs_use_anydrive_3_lstm",
    )
    if not all(structural.get(name) is True for name in required_true):
        raise ValueError("required structural contract differs")
    if structural.get("sdk_nonself_joint_velocity_access_count") != 0:
        raise ValueError("unexpected external _joint_vel access exists")
    counterfactual = (
        grid.get("zero_velocity_equal") is True
        and isinstance(grid.get("nonzero_difference_count"), int)
        and grid["nonzero_difference_count"] > 0
        and grid.get("restore_equal") is True
        and isinstance(grid.get("max_abs_difference"), (int, float))
        and grid["max_abs_difference"] > 0
    )
    if not counterfactual:
        raise ValueError("counterfactual numeric contract differs")
    return "applicable" if structural.get("lstm_compute_writes_clip_joint_velocity") is True else "not_applicable"


def inspect(protocol: dict, isaaclab: Path, protocol_path: Path) -> dict:
    if protocol.get("schema") != "anymal_lstm_motor_semantics_protocol_v2":
        raise ValueError("protocol identity differs")
    paths = source_paths(isaaclab)
    for name, path in paths.items():
        if not path.is_file() or sha_lf(path) != protocol["sources"][name]["sha256_lf"]:
            raise ValueError(f"{name} source differs")
    if sdk_manifest_sha256(isaaclab) != protocol["sdk_scope"]["python_manifest_sha256"]:
        raise ValueError("Isaac Lab SDK Python manifest differs")

    net_tree = ast.parse(paths["actuator_net"].read_text(encoding="utf-8"))
    pd_tree = ast.parse(paths["actuator_pd"].read_text(encoding="utf-8"))
    asset_tree = ast.parse(paths["anymal_asset"].read_text(encoding="utf-8"))
    fault_tree = ast.parse(paths["fault"].read_text(encoding="utf-8"))
    lstm_compute = function_node(class_node(net_tree, "ActuatorNetLSTM").body, "compute")
    dc = class_node(pd_tree, "DCMotor")
    dc_init = function_node(dc.body, "__init__")
    dc_compute = function_node(dc.body, "compute")
    normal_node = function_node(dc.body, "_clip_effort")
    accesses = sdk_joint_vel_accesses(isaaclab)
    nonself = [row for row in accesses if row["base"] != "self"]

    structural = {
        "inherits_dc_motor": any(isinstance(base, ast.Name) and base.id == "DCMotor" for base in class_node(net_tree, "ActuatorNetLSTM").bases),
        "calls_inherited_clip": calls_self_method(lstm_compute, "_clip_effort", "computed_effort"),
        "dc_init_zeros_clip_joint_velocity": self_attribute_initialized_with_zeros(dc_init, "_joint_vel"),
        "dc_compute_writes_clip_joint_velocity": self_attribute_written(dc_compute, "_joint_vel"),
        "lstm_compute_writes_clip_joint_velocity": self_attribute_written(lstm_compute, "_joint_vel"),
        "network_input_uses_joint_velocity": assigns_name_into_self_buffer(lstm_compute, "sea_input", "joint_vel"),
        "anymal_c_legs_use_anydrive_3_lstm": anymal_c_uses_lstm(asset_tree),
        "sdk_nonself_joint_velocity_access_count": len(nonself),
    }
    cfg = literal_config(asset_tree)
    expected_cfg = {key: protocol["configuration"][key] for key in ("saturation_effort", "effort_limit", "velocity_limit")}
    if cfg != expected_cfg:
        raise ValueError("ANYdrive configuration differs from protocol")
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("CUDA was initialized during CPU semantics check")
    grid = numeric_grid(protocol, compile_function(normal_node, torch), compile_function(extract_fault_clip(fault_tree), torch), torch)
    outcome = decide_outcome(structural, grid)
    applicable = outcome == "applicable"
    expected = protocol["predictions"]["semantic_gate"]
    return {
        "schema": "anymal_lstm_motor_semantics_result_v2",
        "outcome": outcome,
        "prediction": expected,
        "prediction_match": outcome == expected,
        "validator_revision": protocol["validator_revision"],
        "structural": structural,
        "sdk_scope": {
            "python_file_count": len(sdk_python_files(isaaclab)),
            "python_manifest_sha256": sdk_manifest_sha256(isaaclab),
            "joint_velocity_accesses": accesses,
        },
        "configuration": cfg,
        "counterfactual_grid": grid,
        "effective_path": {
            "initial_clip_joint_velocity": "DCMotor.__init__ initializes _joint_vel with torch zeros",
            "dc_compute_update": "present",
            "lstm_compute_update": "absent" if not structural["lstm_compute_writes_clip_joint_velocity"] else "present",
            "external_actuator_instance_update": "absent" if not nonself else "present",
        },
        "limitations": protocol["hidden_constraints"],
        "followup": protocol["followup_if_not_applicable"] if not applicable else None,
        "source_sha256_lf": {name: sha_lf(path) for name, path in paths.items()},
        "protocol_sha256_lf": sha_lf(protocol_path),
        "api_calls": 0,
        "gpu_calls": 0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not args.check and args.output is None:
        parser.error("--check 또는 --output이 필요하다")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = inspect(protocol, args.isaaclab, args.protocol)
    if args.output is not None:
        write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["prediction_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
