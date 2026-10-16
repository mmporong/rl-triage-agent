"""Isaac Lab train.py의 마지막 SimulationApp.close 전에 완료 증거를 봉인한다."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import types
from typing import Callable

FAILURE = "training_failure.json"
EVIDENCE = "training_evidence.json"
RECORDS = "records.jsonl"
CLOSE_READY = "close_ready.json"
CLOSE_SCHEMA = "ppo_gradient_close_ready_v4"


def durable_new(path: Path, value: object) -> None:
    """JSON을 새 파일에만 기록하고 지원되는 호스트에서 디렉터리도 동기화한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is not None:
        descriptor = os.open(path.parent, os.O_RDONLY | directory_flag)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def record_failure(output: Path, exc: BaseException, stage: str) -> dict:
    """최초 실패만 보존한다. 오류 문자열과 traceback은 UTF-8 JSON으로 남긴다."""
    if not isinstance(exc, BaseException) or not isinstance(stage, str) or not stage:
        raise TypeError("실패 예외와 stage가 필요하다")
    output = Path(output)
    path = output / FAILURE
    value = {"status": "invalid", "type": type(exc).__name__, "message": str(exc),
             "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
             "stage": stage}
    try:
        durable_new(path, value)
        return value
    except FileExistsError:
        return _read(path)


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    values = [test.left, *test.comparators]
    return (len(values) == 2
            and any(isinstance(value, ast.Name) and value.id == "__name__" for value in values)
            and any(isinstance(value, ast.Constant) and value.value == "__main__" for value in values))


def _is_simulation_close(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and not node.args and not node.keywords
            and isinstance(node.func, ast.Attribute) and node.func.attr == "close"
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "simulation_app")


class _CloseReplacement(ast.NodeTransformer):
    def __init__(self, target: ast.Call):
        self.target = target
        self.replaced = 0

    def visit_Call(self, node: ast.Call):  # noqa: N802 - ast.NodeTransformer API
        if node is self.target:
            self.replaced += 1
            return ast.copy_location(ast.Call(func=ast.Name(id="_guarded_close", ctx=ast.Load()),
                                              args=[ast.Name(id="simulation_app", ctx=ast.Load())],
                                              keywords=[]), node)
        return self.generic_visit(node)


def _guarded_tree(path: Path) -> ast.Module:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    guards = [node for node in tree.body if isinstance(node, ast.If) and _is_main_guard(node)]
    closes = [node for node in ast.walk(tree) if _is_simulation_close(node)]
    if len(guards) != 1 or len(closes) != 1:
        raise ValueError("train.py에는 유일한 __main__ guard와 simulation_app.close() 한 번이 필요하다")
    guard = guards[0]
    terminal = guard.body[-1] if guard.body else None
    if (not isinstance(terminal, ast.Expr) or terminal.value is not closes[0]
            or not _is_simulation_close(terminal.value)):
        raise ValueError("simulation_app.close()는 __main__ guard의 마지막 문장이어야 한다")
    close_call = closes[0]
    if not isinstance(close_call, ast.Call):
        raise AssertionError("close 노드가 호출식이 아니다")
    replacement = _CloseReplacement(close_call)
    transformed = replacement.visit(tree)
    if replacement.replaced != 1:
        raise AssertionError("terminal close 치환 수가 다르다")
    return ast.fix_missing_locations(transformed)


def _ready_for_close(output: Path) -> None:
    marker_path = Path(output) / CLOSE_READY
    evidence_path = Path(output) / EVIDENCE
    records_path = Path(output) / RECORDS
    if not marker_path.is_file() or not evidence_path.is_file() or not records_path.is_file():
        raise RuntimeError("before_close가 완료 증거와 close marker를 모두 기록하지 않았다")
    marker = _read(marker_path)
    if marker.get("schema") != CLOSE_SCHEMA:
        raise ValueError("close marker schema가 다르다")


def _print_failure(exc: BaseException) -> None:
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()


def run_upstream(upstream: Path, output: Path,
                 before_close: Callable[[object], None]) -> dict:
    """설치 train.py를 수정하지 않고 마지막 close 호출만 완료 callback으로 감싼다."""
    upstream, output = Path(upstream).resolve(), Path(output)
    stage = "upstream_ast"
    callback_completed = False
    close_completed = False
    try:
        if not upstream.is_file() or not callable(before_close):
            raise ValueError("설치 train.py와 before_close callback이 필요하다")
        tree = _guarded_tree(upstream)
        code = compile(tree, str(upstream), "exec", dont_inherit=True)
        stage = "upstream_script"

        def guarded_close(app: object) -> None:
            nonlocal stage, callback_completed, close_completed
            stage = "before_close"
            before_close(app)
            _ready_for_close(output)
            callback_completed = True
            stage = "simulation_app_close"
            try:
                close = getattr(app, "close", None)
                if not callable(close):
                    raise TypeError("simulation_app.close가 호출 가능해야 한다")
                close()
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
                close_completed = True
                return
            close_completed = True

        main_module = types.ModuleType("__main__")
        main_module.__dict__.update({"__file__": str(upstream), "__package__": None,
                                     "__cached__": None, "_guarded_close": guarded_close})
        inserted = not sys.path or Path(sys.path[0]).resolve() != upstream.parent
        if inserted:
            sys.path.insert(0, str(upstream.parent))
        previous_main = sys.modules.get("__main__")
        sys.modules["__main__"] = main_module
        try:
            exec(code, main_module.__dict__)
        finally:
            if previous_main is None:
                sys.modules.pop("__main__", None)
            else:
                sys.modules["__main__"] = previous_main
            if inserted and sys.path and sys.path[0] == str(upstream.parent):
                sys.path.pop(0)
        if not callback_completed or not close_completed:
            stage = "terminal_close_missing"
            raise RuntimeError("upstream이 완료 callback과 terminal close를 실행하지 않았다")
        return {"status": "complete", "callback_completed": True, "close_completed": True,
                "upstream": str(upstream)}
    except BaseException as exc:
        record_failure(output, exc, stage)
        _print_failure(exc)
        raise


def validate_completion(output: Path, manifest_sha: str) -> None:
    """학습 결과는 실패 기록이 없고 evidence·records·close marker가 모두 맞아야 한다."""
    output = Path(output)
    failure = output / FAILURE
    if failure.exists():
        raise ValueError("학습 실패 기록이 있어 완료로 승인할 수 없다")
    if (not isinstance(manifest_sha, str) or len(manifest_sha) != 64
            or any(character not in "0123456789abcdef" for character in manifest_sha)):
        raise ValueError("manifest SHA가 유효하지 않다")
    evidence_path, records_path = output / EVIDENCE, output / RECORDS
    marker_path = output / CLOSE_READY
    if not evidence_path.is_file() or not records_path.is_file() or not marker_path.is_file():
        raise ValueError("학습 완료 evidence·records·close marker가 없다")
    evidence, marker = _read(evidence_path), _read(marker_path)
    updates = evidence.get("updates")
    marker_updates = marker.get("updates")
    if (marker.get("schema") != CLOSE_SCHEMA or marker.get("manifest_sha256") != manifest_sha
            or evidence.get("manifest_sha256") != manifest_sha
            or type(updates) is not int or updates <= 0
            or type(marker_updates) is not int or marker_updates <= 0 or marker_updates != updates
            or marker.get("evidence_sha256_lf") != _sha_lf(evidence_path)
            or marker.get("records_sha256_lf") != _sha_lf(records_path)):
        raise ValueError("학습 evidence와 close marker 결합이 다르다")
