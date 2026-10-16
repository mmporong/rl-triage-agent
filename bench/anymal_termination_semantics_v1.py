"""ANYmal-C flat에서 F6 조기 종료와 등록 교정의 CPU 의미를 검증한다."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import FunctionType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "bench/protocols/anymal_termination_semantics_v1.json"
TASK_ID = "Isaac-Velocity-Flat-Anymal-C-v0"
CORRECTION_ID = "restore_configured_timeout"
FAULT_ID = "F6"


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_paths(isaaclab: Path) -> dict[str, Path]:
    task_root = isaaclab / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    return {
        "validator": Path(__file__).resolve(),
        "test": ROOT / "tests/test_anymal_termination_semantics_v1.py",
        "fault": ROOT / "bench/hidden_faults.py",
        "correction_adapter": ROOT / "bench/correction_adapter.py",
        "registration": task_root / "config/anymal_c/__init__.py",
        "flat_config": task_root / "config/anymal_c/flat_env_cfg.py",
        "rough_config": task_root / "config/anymal_c/rough_env_cfg.py",
        "velocity_config": task_root / "velocity_env_cfg.py",
        "manager_env": isaaclab / "source/isaaclab/isaaclab/envs/manager_based_env.py",
        "manager_rl_env": isaaclab / "source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
        "termination": isaaclab / "source/isaaclab/isaaclab/envs/mdp/terminations.py",
    }


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    rows = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(rows) != 1:
        raise ValueError(f"class {name} definition count differs")
    return rows[0]


def _function(nodes: list[ast.stmt], name: str) -> ast.FunctionDef:
    rows = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(rows) != 1:
        raise ValueError(f"function {name} definition count differs")
    return rows[0]


def _self_assignment(function: ast.FunctionDef, attribute: str) -> Any:
    rows = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            and target.attr == attribute
        ):
            rows.append(ast.literal_eval(node.value))
    if len(rows) != 1:
        raise ValueError(f"self.{attribute} literal assignment count differs")
    return rows[0]


def _sim_dt(function: ast.FunctionDef) -> float:
    rows = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "dt"
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "sim"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "self"
        ):
            rows.append(ast.literal_eval(node.value))
    if len(rows) != 1:
        raise ValueError("self.sim.dt literal assignment count differs")
    return float(rows[0])


def _base_name(node: ast.ClassDef) -> str:
    if len(node.bases) != 1 or not isinstance(node.bases[0], ast.Name):
        raise ValueError(f"class {node.name} inheritance differs")
    return node.bases[0].id


def _calls_super_post_init(node: ast.ClassDef) -> bool:
    function = _function(node.body, "__post_init__")
    for call in (child for child in ast.walk(function) if isinstance(child, ast.Call)):
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "__post_init__":
            continue
        owner = call.func.value
        if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id == "super":
            return True
    return False


def _registration_entry(tree: ast.Module) -> str:
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "register":
            continue
        keywords = {item.arg: item.value for item in call.keywords if item.arg is not None}
        task_id = keywords.get("id")
        if task_id is None or ast.literal_eval(task_id) != TASK_ID:
            continue
        kwargs = keywords.get("kwargs")
        if not isinstance(kwargs, ast.Dict):
            raise ValueError("ANYmal-C registration kwargs differs")
        for key, value in zip(kwargs.keys, kwargs.values):
            if isinstance(key, ast.Constant) and key.value == "env_cfg_entry_point":
                if not isinstance(value, ast.JoinedStr):
                    raise ValueError("ANYmal-C env cfg entry point differs")
                return ast.unparse(value)
    raise ValueError("ANYmal-C flat task registration is absent")


def _termination_binding(tree: ast.Module) -> bool:
    node = _class(tree, "TerminationsCfg")
    rows = [
        statement
        for statement in node.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "time_out" for target in statement.targets)
    ]
    if len(rows) != 1 or not isinstance(rows[0].value, ast.Call):
        return False
    call = rows[0].value
    if not (isinstance(call.func, ast.Name) and call.func.id == "DoneTerm"):
        return False
    keywords = {item.arg: item.value for item in call.keywords if item.arg is not None}
    func = keywords.get("func")
    timeout_flag = keywords.get("time_out")
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "mdp"
        and func.attr == "time_out"
        and timeout_flag is not None
        and ast.literal_eval(timeout_flag) is True
    )


def _has_return_expression(function: ast.FunctionDef, expected: str) -> bool:
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    return len(returns) == 1 and returns[0].value is not None and ast.dump(returns[0].value, include_attributes=False) == ast.dump(
        ast.parse(expected, mode="eval").body, include_attributes=False
    )


def inspect_static(isaaclab: Path) -> dict[str, Any]:
    paths = source_paths(isaaclab)
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"{name} source is absent")
    registration = _tree(paths["registration"])
    flat = _class(_tree(paths["flat_config"]), "AnymalCFlatEnvCfg")
    rough = _class(_tree(paths["rough_config"]), "AnymalCRoughEnvCfg")
    velocity_tree = _tree(paths["velocity_config"])
    velocity = _class(velocity_tree, "LocomotionVelocityRoughEnvCfg")
    post = _function(velocity.body, "__post_init__")
    manager_env = _class(_tree(paths["manager_env"]), "ManagerBasedEnv")
    step_dt = _function(manager_env.body, "step_dt")
    manager_rl = _class(_tree(paths["manager_rl_env"]), "ManagerBasedRLEnv")
    max_length = _function(manager_rl.body, "max_episode_length")
    termination = _function(_tree(paths["termination"]).body, "time_out")
    fault = _function(_tree(paths["fault"]).body, "_time_out_early")

    episode_length_s = float(_self_assignment(post, "episode_length_s"))
    decimation = int(_self_assignment(post, "decimation"))
    physics_dt = _sim_dt(post)
    control_dt = physics_dt * decimation
    max_episode_length = math.ceil(episode_length_s / control_dt)
    fault_ratio = next(
        (
            float(node.value)
            for node in ast.walk(fault)
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value == 0.05
        ),
        None,
    )
    if fault_ratio is None:
        raise ValueError("F6 timeout ratio differs")
    fault_step = int(max_episode_length * fault_ratio)
    structural = {
        "task_registration_entry": _registration_entry(registration),
        "flat_base": _base_name(flat),
        "rough_base": _base_name(rough),
        "flat_calls_super_post_init": _calls_super_post_init(flat),
        "rough_calls_super_post_init": _calls_super_post_init(rough),
        "termination_uses_mdp_time_out": _termination_binding(velocity_tree),
        "step_dt_formula_matches": _has_return_expression(step_dt, "self.cfg.sim.dt * self.cfg.decimation"),
        "max_episode_length_formula_matches": _has_return_expression(
            max_length, "math.ceil(self.max_episode_length_s / self.step_dt)"
        ),
        "upstream_timeout_formula_matches": _has_return_expression(
            termination, "env.episode_length_buf >= env.max_episode_length"
        ),
        "fault_timeout_formula_matches": _has_return_expression(
            fault, "env.episode_length_buf >= int(env.max_episode_length * 0.05)"
        ),
    }
    if structural != {
        "task_registration_entry": "f'{__name__}.flat_env_cfg:AnymalCFlatEnvCfg'",
        "flat_base": "AnymalCRoughEnvCfg",
        "rough_base": "LocomotionVelocityRoughEnvCfg",
        "flat_calls_super_post_init": True,
        "rough_calls_super_post_init": True,
        "termination_uses_mdp_time_out": True,
        "step_dt_formula_matches": True,
        "max_episode_length_formula_matches": True,
        "upstream_timeout_formula_matches": True,
        "fault_timeout_formula_matches": True,
    }:
        raise ValueError(f"ANYmal-C timeout structural contract differs: {structural!r}")
    return {
        "task": TASK_ID,
        "structural": structural,
        "episode_length_s": episode_length_s,
        "physics_dt_s": physics_dt,
        "decimation": decimation,
        "control_dt_s": control_dt,
        "configured_max_episode_length": max_episode_length,
        "fault_ratio": fault_ratio,
        "fault_timeout_step": fault_step,
        "fault_timeout_s": fault_step * control_dt,
    }


def build_protocol(isaaclab: Path, *, correction_contract: dict | None = None) -> dict[str, Any]:
    static = inspect_static(isaaclab)
    paths = source_paths(isaaclab)
    sys.path.insert(0, str(ROOT / "bench"))
    import correction_adapter

    return {
        "schema": "anymal_termination_semantics_protocol_v1",
        "status": "preregistered",
        "question": "ANYmal-C flat이 등록된 F6 조기 종료의 표적을 실제로 사용하며 등록 교정이 설정된 종료 시점을 복원하는가",
        "scope": "CPU source-and-tensor semantics check; no Isaac Kit, PhysX, environment construction, policy, training, GPU, or API",
        "task": TASK_ID,
        "fault": {"id": FAULT_ID, "ratio": static["fault_ratio"]},
        "correction": {"id": CORRECTION_ID, "target": "isaaclab.envs.mdp.terminations:time_out"},
        "configuration": {key: static[key] for key in (
            "episode_length_s", "physics_dt_s", "decimation", "control_dt_s",
            "configured_max_episode_length", "fault_timeout_step", "fault_timeout_s",
        )},
        "structural": static["structural"],
        "probe_steps": [static["fault_timeout_step"] - 1, static["fault_timeout_step"], static["configured_max_episode_length"] - 1, static["configured_max_episode_length"]],
        "prediction": {
            "semantic_gate": "applicable",
            "baseline": [False, False, False, True],
            "fault": [False, True, True, True],
            "restored": [False, False, False, True],
        },
        "gate": {
            "applicable": "task registration·inheritance·termination binding이 맞고 실제 upstream 함수에서 F6가 조기 종료를 만들며 restore_configured_timeout 뒤 최초 출력과 코드 지문이 복원될 때",
            "not_applicable": "결합은 유효하지만 ANYmal-C 종료 경로가 표적을 사용하지 않거나 F6가 출력 차이를 만들지 않을 때",
            "invalid": "파일 SHA·구조·등록 교정·CPU 실행·복원 계약 중 하나를 검증할 수 없을 때",
        },
        "sources": {
            name: {
                "path": (path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else "<isaaclab>/" + path.relative_to(isaaclab).as_posix()),
                "sha256_lf": sha_lf(path),
            }
            for name, path in paths.items()
        },
        "correction_source_contract": (correction_adapter.source_contract(isaaclab)
                                       if correction_contract is None else correction_contract),
        "limitations": {
            "behavior": "학습 실패와 고정 평가 영향은 이 CPU 의미 검사에서 측정하지 않는다",
            "runtime": "환경을 생성하지 않고 upstream 종료 함수에 CPU tensor 상태를 넣는다",
            "time_interpretation": "fault_timeout_s는 설정에서 계산한 control step 경계이며 물리 실행 시간 측정값이 아니다",
        },
        "claims_not_supported": ["ANYmal-C 학습 실패", "정책 회복", "물리 시뮬레이션 결과", "일반화", "절감", "모델 우월성"],
        "api_calls": 0,
        "gpu_calls": 0,
    }


def validate_protocol(protocol: dict[str, Any], isaaclab: Path) -> dict[str, Any]:
    if protocol.get("schema") != "anymal_termination_semantics_protocol_v1":
        raise ValueError("protocol identity differs")
    if protocol.get("status") != "preregistered" or protocol.get("task") != TASK_ID:
        raise ValueError("protocol registration differs")
    if protocol.get("fault") != {"id": FAULT_ID, "ratio": 0.05}:
        raise ValueError("fault registration differs")
    if protocol.get("correction") != {"id": CORRECTION_ID, "target": "isaaclab.envs.mdp.terminations:time_out"}:
        raise ValueError("correction registration differs")
    paths = source_paths(isaaclab)
    rows = protocol.get("sources")
    if not isinstance(rows, dict) or set(rows) != set(paths):
        raise ValueError("protocol source inventory differs")
    for name, path in paths.items():
        if not isinstance(rows[name], dict) or rows[name].get("sha256_lf") != sha_lf(path):
            raise ValueError(f"{name} source differs")
    static = inspect_static(isaaclab)
    expected_configuration = {key: static[key] for key in protocol["configuration"]}
    if protocol["configuration"] != expected_configuration:
        raise ValueError("ANYmal-C timeout configuration differs")
    expected_steps = [static["fault_timeout_step"] - 1, static["fault_timeout_step"], static["configured_max_episode_length"] - 1, static["configured_max_episode_length"]]
    if protocol.get("probe_steps") != expected_steps:
        raise ValueError("probe step registration differs")
    expected = build_protocol(isaaclab, correction_contract=protocol.get("correction_source_contract"))
    if protocol != expected:
        raise ValueError("registered CPU protocol contract differs")
    return static


def smoke_binding(report: dict[str, Any]) -> dict[str, Any]:
    """CPU 결과를 GPU smoke 사전등록에 넣을 결정적 요약으로 좁힌다."""
    if not isinstance(report, dict) or report.get("schema") != "anymal_termination_semantics_result_v1":
        raise ValueError("CPU semantics report identity differs")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_sha = sha_lf(PROTOCOL)
    if report.get("protocol_sha256_lf") != protocol_sha:
        raise ValueError("CPU semantics report protocol binding differs")
    configuration = protocol.get("configuration")
    if report.get("configuration") != configuration:
        raise ValueError("CPU semantics report configuration differs")
    if report.get("structural") != protocol["structural"]:
        raise ValueError("CPU semantics structural binding differs")
    expected_source = {name: row["sha256_lf"] for name, row in protocol["sources"].items()}
    if report.get("source_sha256_lf") != expected_source:
        raise ValueError("CPU semantics report source binding differs")
    prediction = protocol["prediction"]
    probe = report.get("probe")
    if not isinstance(probe, dict) or probe.get("steps") != protocol["probe_steps"]:
        raise ValueError("CPU semantics report probe inventory differs")
    outputs = {key: probe.get(key) for key in ("baseline", "fault", "restored")}
    if any(outputs[key] != prediction[key] for key in outputs):
        raise ValueError("CPU semantics report outputs differ")
    correction = report.get("correction_readback")
    if not isinstance(correction, dict):
        raise ValueError("CPU semantics correction readback is absent")
    restored = (
        correction.get("selected_correction") == CORRECTION_ID
        and correction.get("registered_mode") == "code"
        and correction.get("changed_targets") == [CORRECTION_ID]
        and correction.get("restored_targets") == [CORRECTION_ID]
        and correction.get("clean_fingerprint") == correction.get("restored_fingerprint")
        and correction.get("fault_fingerprint") != correction.get("clean_fingerprint")
        and correction.get("clean_fingerprint") == protocol["correction_source_contract"]["targets"][CORRECTION_ID]["clean_fingerprint"]
        and all(isinstance(correction.get(key), str) and len(correction[key]) == 64
                and all(c in "0123456789abcdef" for c in correction[key])
                for key in ("clean_fingerprint", "fault_fingerprint", "restored_fingerprint"))
    )
    if not restored:
        raise ValueError("CPU semantics correction readback differs")
    if report.get("outcome") != "applicable" or report.get("prediction") != "applicable" or report.get("prediction_match") is not True:
        raise ValueError("CPU semantics outcome differs")
    if report.get("api_calls") != 0 or report.get("gpu_calls") != 0:
        raise ValueError("CPU semantics resource counters differ")
    return {
        "schema": "anymal_termination_semantics_smoke_binding_v1",
        "applicable": True,
        "all_checks_passed": True,
        "cpu_protocol_sha256_lf": protocol_sha,
        "configuration": configuration,
        "structural": protocol["structural"],
        "probe": {"steps": probe["steps"], **outputs},
        "correction": {"id": CORRECTION_ID, "restored": True},
        "source_sha256_lf": expected_source,
    }


class _Env:
    max_episode_length: int
    episode_length_buf: Any


def execute(protocol: dict[str, Any], isaaclab: Path, protocol_path: Path) -> dict[str, Any]:
    static = validate_protocol(protocol, isaaclab)
    sys.path.insert(0, str(ROOT / "bench"))
    import correction_adapter
    import hidden_faults
    import torch

    if torch.cuda.is_initialized():
        raise ValueError("CUDA was initialized before the CPU semantics check")
    contract_rows = correction_adapter._validate_contract(isaaclab, protocol["correction_source_contract"])
    target_spec = correction_adapter.TARGETS.get(CORRECTION_ID)
    if target_spec is None or target_spec != {
        "module": "isaaclab.envs.mdp.terminations",
        "qualname": "time_out",
        "source": "source/isaaclab/isaaclab/envs/mdp/terminations.py",
        "mode": "code",
    }:
        raise ValueError("registered timeout correction target differs")
    target_row = contract_rows[CORRECTION_ID]
    source = (isaaclab / target_spec["source"]).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    module_code = compile(source, target_spec["source"], "exec", dont_inherit=True)
    clean_code = correction_adapter._find_code(module_code, target_spec["qualname"])
    if correction_adapter.fingerprint(clean_code) != target_row["clean_fingerprint"]:
        raise ValueError("registered timeout correction fingerprint differs")
    time_out = FunctionType(clean_code, {"torch": torch}, "time_out")
    clean_fingerprint = correction_adapter.fingerprint(time_out)
    env = _Env()
    env.max_episode_length = static["configured_max_episode_length"]
    env.episode_length_buf = torch.tensor(protocol["probe_steps"], dtype=torch.int64, device="cpu")
    baseline = time_out(env).tolist()
    try:
        hidden_faults._swap_code(time_out, hidden_faults._time_out_early)
        fault_fingerprint = correction_adapter.fingerprint(time_out)
        fault = time_out(env).tolist()
        if target_spec["mode"] != "code":
            raise ValueError("registered timeout correction mode differs")
        time_out.__code__ = clean_code
        restored_fingerprint = correction_adapter.fingerprint(time_out)
        restored = time_out(env).tolist()
    finally:
        time_out.__code__ = clean_code
    expected = protocol["prediction"]
    outputs = {"baseline": baseline, "fault": fault, "restored": restored}
    if any(outputs[key] != expected[key] for key in outputs):
        outcome = "not_applicable"
    elif fault_fingerprint == clean_fingerprint or restored_fingerprint != clean_fingerprint:
        raise ValueError("registered correction did not restore the clean time_out fingerprint")
    else:
        outcome = "applicable"
    if torch.cuda.is_initialized():
        raise ValueError("CUDA was initialized during the CPU semantics check")
    return {
        "schema": "anymal_termination_semantics_result_v1",
        "outcome": outcome,
        "prediction": expected["semantic_gate"],
        "prediction_match": outcome == expected["semantic_gate"],
        "configuration": protocol["configuration"],
        "probe": {"steps": protocol["probe_steps"], **outputs},
        "structural": static["structural"],
        "correction_readback": {
            "selected_correction": CORRECTION_ID,
            "registered_mode": target_spec["mode"],
            "changed_targets": [CORRECTION_ID],
            "restored_targets": [CORRECTION_ID],
            "clean_fingerprint": clean_fingerprint,
            "fault_fingerprint": fault_fingerprint,
            "restored_fingerprint": restored_fingerprint,
        },
        "source_sha256_lf": {name: sha_lf(path) for name, path in source_paths(isaaclab).items()},
        "protocol_sha256_lf": sha_lf(protocol_path),
        "limitations": protocol["limitations"],
        "api_calls": 0,
        "gpu_calls": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        write_new(args.protocol, build_protocol(args.isaaclab.resolve()))
        print(args.protocol)
        return 0
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.command == "check":
        static = validate_protocol(protocol, args.isaaclab.resolve())
        print(json.dumps(static, ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        parser.error("run에는 --output이 필요하다")
    result = execute(protocol, args.isaaclab.resolve(), args.protocol.resolve())
    write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["prediction_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
