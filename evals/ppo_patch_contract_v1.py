"""Restricted contract for PPO algorithm-kwargs adapter proposals.

This module validates a deliberately small Python AST language.  It is not a
general-purpose Python sandbox and must not be described or used as one.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass
from typing import Final

ENTRYPOINT: Final = "build_algorithm_kwargs"
MAX_SOURCE_BYTES: Final = 8_192
MAX_AST_NODES: Final = 256
MAX_AST_DEPTH: Final = 20
EPOCH_KEY: Final = "num_learning_epochs"
PROPOSAL_KEYS: Final = frozenset(
    {"action", "source_sha256", "replacement_source", "evidence_ids", "rationale"}
)


class PatchContractError(ValueError):
    """Raised when source or a proposal violates the restricted contract."""


@dataclass(frozen=True)
class _Origin:
    kind: str
    key: str | None = None


_CONFIG_DICT = _Origin("config_dict")
_LOCAL_DICT = _Origin("local_dict")
_CONSTANT = _Origin("constant")


def _normalise_lf(source: str) -> str:
    if not isinstance(source, str):
        raise PatchContractError("source must be a string")
    return source.replace("\r\n", "\n").replace("\r", "\n")


def source_sha256(source: str) -> str:
    """Return the SHA-256 of UTF-8 source after LF normalisation."""

    return hashlib.sha256(_normalise_lf(source).encode("utf-8")).hexdigest()


def _is_dunder(value: str) -> bool:
    return value.startswith("__") or value.endswith("__")


def _constant_key(node: ast.expr) -> str:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        raise PatchContractError("dictionary keys must be constant strings")
    if _is_dunder(node.value):
        raise PatchContractError("dunder keys are not allowed")
    return node.value


def _literal(node: ast.expr) -> _Origin:
    if not isinstance(node, ast.Constant) or not isinstance(
        node.value, (str, int, float, bool, type(None))
    ):
        raise PatchContractError("only scalar literals are allowed")
    return _CONSTANT


def _tree_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_tree_depth(child) for child in children)


class _FunctionValidator:
    def __init__(self) -> None:
        self.locals: dict[str, _Origin] = {}
        self.local_dicts: set[str] = set()

    def _name(self, node: ast.Name) -> _Origin:
        if _is_dunder(node.id):
            raise PatchContractError("dunder names are not allowed")
        if node.id == "config":
            return _CONFIG_DICT
        if node.id not in self.locals:
            raise PatchContractError(f"unknown local name: {node.id}")
        return self.locals[node.id]

    def _call(self, node: ast.Call) -> _Origin:
        if node.keywords:
            raise PatchContractError("keyword arguments are not allowed")

        if isinstance(node.func, ast.Name):
            if node.func.id != "dict" or len(node.args) != 1:
                raise PatchContractError("only dict(config) is allowed")
            if not isinstance(node.args[0], ast.Name) or node.args[0].id != "config":
                raise PatchContractError("dict() must copy config")
            return _CONFIG_DICT

        if not isinstance(node.func, ast.Attribute):
            raise PatchContractError("dynamic calls are not allowed")
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "config":
            raise PatchContractError("method calls are limited to config")
        if _is_dunder(node.func.attr):
            raise PatchContractError("dunder attributes are not allowed")

        if node.func.attr == "copy" and not node.args:
            return _CONFIG_DICT
        if node.func.attr == "get" and len(node.args) in {1, 2}:
            key = _constant_key(node.args[0])
            if len(node.args) == 2:
                _literal(node.args[1])
            return _Origin("config_value", key)
        raise PatchContractError("only config.copy() and config.get() are allowed")

    def expression(self, node: ast.expr) -> _Origin:
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.Constant):
            return _literal(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise PatchContractError("dictionary unpacking is not allowed")
            for key, value in zip(node.keys, node.values, strict=True):
                assert key is not None
                _constant_key(key)
                self.expression(value)
            return _LOCAL_DICT
        if isinstance(node, ast.Subscript):
            if not isinstance(node.ctx, ast.Load):
                raise PatchContractError("subscript expression must be a lookup")
            if not isinstance(node.value, ast.Name):
                raise PatchContractError("lookups require config or a local dictionary")
            container = self._name(node.value)
            if container.kind not in {"config_dict", "local_dict"}:
                raise PatchContractError("lookups require a local dictionary")
            return _Origin("config_value", _constant_key(node.slice))
        raise PatchContractError(f"expression is not allowed: {type(node).__name__}")

    def assignment(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise PatchContractError("chained assignment is not allowed")
        target = node.targets[0]
        origin = self.expression(node.value)

        if isinstance(target, ast.Name):
            if target.id in {"config", "dict"} or _is_dunder(target.id):
                raise PatchContractError("config and dict cannot be assigned or shadowed")
            if target.id in self.locals:
                raise PatchContractError("local variables are single-assignment")
            self.locals[target.id] = origin
            if origin.kind in {"config_dict", "local_dict"}:
                self.local_dicts.add(target.id)
            return

        if not isinstance(target, ast.Subscript):
            raise PatchContractError("assignment target is not allowed")
        if not isinstance(target.ctx, ast.Store):
            raise PatchContractError("dictionary field assignment must store a value")
        if not isinstance(target.value, ast.Name) or target.value.id not in self.local_dicts:
            raise PatchContractError("only a local dictionary may be updated")
        if _constant_key(target.slice) != EPOCH_KEY:
            raise PatchContractError(f"only {EPOCH_KEY!r} may be updated")

    def validate(self, function: ast.FunctionDef) -> None:
        arguments = function.args
        if (
            len(arguments.posonlyargs) != 0
            or len(arguments.args) != 1
            or arguments.args[0].arg != "config"
            or arguments.vararg is not None
            or arguments.kwonlyargs
            or arguments.kwarg is not None
            or arguments.defaults
            or arguments.kw_defaults
        ):
            raise PatchContractError("entrypoint must take exactly one argument named config")
        if (
            function.decorator_list
            or function.returns is not None
            or function.type_comment is not None
            or arguments.args[0].annotation is not None
        ):
            raise PatchContractError("decorators, annotations, and type comments are not allowed")

        body = list(function.body)
        if body and isinstance(body[0], ast.Expr):
            if not isinstance(body[0].value, ast.Constant) or not isinstance(
                body[0].value.value, str
            ):
                raise PatchContractError("only a string docstring expression is allowed")
            body.pop(0)
        if not body or not isinstance(body[-1], ast.Return):
            raise PatchContractError("entrypoint must end with return")

        for statement in body[:-1]:
            if not isinstance(statement, ast.Assign):
                raise PatchContractError(
                    f"statement is not allowed: {type(statement).__name__}"
                )
            self.assignment(statement)

        result = body[-1].value
        if not isinstance(result, ast.Name) or result.id not in self.local_dicts:
            raise PatchContractError("entrypoint must return a local dictionary")


def validate_source(source: str) -> None:
    """Validate source against the restricted adapter AST language."""

    normalised = _normalise_lf(source)
    if len(normalised.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise PatchContractError("source exceeds 8192 UTF-8 bytes")
    if "\x00" in normalised:
        raise PatchContractError("source contains a NUL byte")
    try:
        tree = ast.parse(normalised, mode="exec")
    except (SyntaxError, ValueError) as exc:
        raise PatchContractError("source is not valid Python") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise PatchContractError("source exceeds the AST node limit")
    if _tree_depth(tree) > MAX_AST_DEPTH:
        raise PatchContractError("source exceeds the AST depth limit")

    module_body = list(tree.body)
    if module_body and isinstance(module_body[0], ast.Expr):
        if not isinstance(module_body[0].value, ast.Constant) or not isinstance(
            module_body[0].value.value, str
        ):
            raise PatchContractError("only a module docstring expression is allowed")
        module_body.pop(0)
    if len(module_body) != 1 or not isinstance(module_body[0], ast.FunctionDef):
        raise PatchContractError("module must contain exactly the entrypoint function")
    function = module_body[0]
    if function.name != ENTRYPOINT:
        raise PatchContractError(f"entrypoint must be named {ENTRYPOINT}")
    _FunctionValidator().validate(function)


def apply_adapter(source: str, config: dict) -> dict:
    """Execute a validated adapter and enforce result/original invariants."""

    validate_source(source)
    if type(config) is not dict:
        raise PatchContractError("config must be a plain dictionary")
    try:
        before = copy.deepcopy(config)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise PatchContractError("config must be deepcopy-compatible") from exc

    namespace: dict[str, object] = {"__builtins__": {"dict": dict}}
    try:
        code = compile(_normalise_lf(source), "<ppo-patch-adapter>", "exec")
        exec(code, namespace, namespace)  # noqa: S102 - validated restricted AST only
        result = namespace[ENTRYPOINT](config)  # type: ignore[operator]
    except Exception as exc:
        raise PatchContractError("adapter execution failed") from exc
    if type(result) is not dict:
        raise PatchContractError("adapter must return a plain dictionary")
    if config != before:
        raise PatchContractError("adapter mutated the input config")
    return result


def validate_proposal(proposal: dict, context: dict) -> dict:
    """Validate proposal shape and evidence binding, not patch correctness."""

    if type(proposal) is not dict or set(proposal) != PROPOSAL_KEYS:
        raise PatchContractError("proposal keys do not match the contract")
    if type(context) is not dict:
        raise PatchContractError("context must be a dictionary")
    source = context.get("source")
    declared_source_sha = context.get("source_sha256")
    facts = context.get("facts")
    if not isinstance(source, str) or not isinstance(declared_source_sha, str):
        raise PatchContractError("context source binding is missing")
    if type(facts) is not dict:
        raise PatchContractError("context facts must be a dictionary")
    actual_source_sha = source_sha256(source)
    if declared_source_sha != actual_source_sha:
        raise PatchContractError("context source SHA does not match its source")
    if proposal["source_sha256"] != actual_source_sha:
        raise PatchContractError("proposal source SHA does not match context")

    evidence_ids = proposal["evidence_ids"]
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise PatchContractError("at least one evidence ID is required")
    if any(not isinstance(item, str) or not item for item in evidence_ids):
        raise PatchContractError("evidence IDs must be non-empty strings")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise PatchContractError("evidence IDs must not be duplicated")
    if any(item not in facts for item in evidence_ids):
        raise PatchContractError("proposal cites an unknown evidence ID")

    rationale = proposal["rationale"]
    if not isinstance(rationale, str) or not rationale or len(rationale) > 1_000:
        raise PatchContractError("rationale must contain 1 to 1000 characters")

    action = proposal["action"]
    replacement = proposal["replacement_source"]
    if action == "hold":
        if replacement is not None:
            raise PatchContractError("hold proposals must not include replacement source")
    elif action == "patch":
        if not isinstance(replacement, str):
            raise PatchContractError("patch proposals require replacement source")
        if source_sha256(replacement) == actual_source_sha:
            raise PatchContractError("patch replacement must differ from current source")
        validate_source(replacement)
    else:
        raise PatchContractError("action must be patch or hold")
    return dict(proposal)

