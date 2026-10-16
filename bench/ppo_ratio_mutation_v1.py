"""등록된 inverse-ratio 개발 오류를 PPO 인스턴스에만 적용하고 복원한다."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from types import MethodType
from typing import Any


MUTATION = "inverse_importance_ratio_v1"


def ast_sha(tree: ast.AST) -> str:
    # Python 3.12가 추가한 빈 type_params는 3.10 Isaac 런타임과 같은 의미다.
    def canonical(node):
        if isinstance(node, ast.AST):
            return {"node": type(node).__name__, "fields": [
                [name, canonical(value)] for name, value in ast.iter_fields(node)
                if not (name == "type_params" and value == [])]}
        if isinstance(node, list):
            return [canonical(value) for value in node]
        return node
    return hashlib.sha256(json.dumps(canonical(tree), sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def mutation_tree(source: str):
    """등록한 ratio assignment 하나만 바꾼 AST와 결합값을 만든다."""
    tree = ast.parse(textwrap.dedent(source))
    before = ast_sha(tree)
    expected = ast.parse(
        "torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))",
        mode="eval").body
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == "ratio"]
    if len(matches) != 1 or ast_sha(matches[0].value) != ast_sha(expected):
        raise ValueError("등록된 PPO ratio 식 하나와 일치해야 한다")
    call = matches[0].value
    assert isinstance(call, ast.Call) and isinstance(call.args[0], ast.BinOp)
    delta = call.args[0]
    delta.left, delta.right = delta.right, delta.left
    inverse = ast.parse(
        "torch.exp(torch.squeeze(old_actions_log_prob_batch) - actions_log_prob_batch)",
        mode="eval").body
    if ast_sha(call) != ast_sha(inverse):
        raise ValueError("교환한 ratio 식이 등록한 inverse 식과 다르다")
    ast.fix_missing_locations(tree)
    after = ast_sha(tree)
    return tree, {"mutation": MUTATION, "changed_ratio_assignments": len(matches),
                  "original_ast_sha256": before, "mutated_ast_sha256": after}


def mutated_update(original):
    """ratio의 차감 피연산자 한 쌍을 교환한다. 설치된 파일은 쓰지 않는다."""
    tree, evidence = mutation_tree(inspect.getsource(original))
    namespace = dict(original.__globals__)
    exec(compile(tree, "<registered_ppo_ratio_mutation_v1>", "exec"), namespace)
    changed = namespace[original.__name__]
    return changed, evidence


def install(ppo: Any, arm: str):
    if arm not in {"NONE", "INVERSE_RATIO", "RESTORED"}:
        raise ValueError("미등록 계산 오류 arm")
    had = "update" in ppo.__dict__
    previous = ppo.__dict__.get("update")
    original = ppo.update.__func__
    changed, evidence = mutated_update(original)
    def restore():
        if had:
            ppo.update = previous
        else:
            ppo.__dict__.pop("update", None)

    if arm == "RESTORED":
        ppo.update = MethodType(changed, ppo)
        if ppo.update.__func__ is not changed:
            raise ValueError("변이 메서드 설치 확인 실패")
        restore()
        if ppo.update.__func__ is not original:
            raise ValueError("원래 update 메서드 복원 확인 실패")
    else:
        ppo.update = MethodType(changed if arm == "INVERSE_RATIO" else original, ppo)
    readback = {**evidence, "arm": arm, "active_mutation": arm == "INVERSE_RATIO",
                "restored_original": arm == "RESTORED",
                "restoration_before_training": arm == "RESTORED",
                "installed_ast_sha256": evidence[
                    "mutated_ast_sha256" if arm == "INVERSE_RATIO" else "original_ast_sha256"]}
    return restore, readback


def state_digest(value: Any) -> str:
    """직렬화 시각·장치명 없이 tensor 값과 구조를 결합한다."""
    def material(item):
        if hasattr(item, "detach") and hasattr(item, "numpy"):
            tensor = item.detach().cpu().contiguous()
            return {"tensor": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                    "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        if isinstance(item, dict):
            return {"mapping": [[type(key).__name__, str(key), material(child)]
                                for key, child in sorted(item.items(), key=lambda row: (type(row[0]).__name__, str(row[0]))) ]}
        if isinstance(item, (list, tuple)):
            return {"sequence": [material(child) for child in item], "type": type(item).__name__}
        if item is None or type(item) in (str, bool, int, float):
            return item
        raise TypeError(f"상태 결합 미지원 자료형: {type(item).__name__}")
    return hashlib.sha256(json.dumps(material(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def learning_state(ppo: Any, *, cuda: bool) -> dict:
    import torch
    return {"policy_sha256": state_digest(ppo.policy.state_dict()),
            "optimizer_sha256": state_digest(ppo.optimizer.state_dict()),
            "cpu_rng_sha256": state_digest(torch.get_rng_state()),
            "cuda_rng_sha256": state_digest(torch.cuda.get_rng_state_all()) if cuda else None,
            "learning_rate": ppo.learning_rate}
