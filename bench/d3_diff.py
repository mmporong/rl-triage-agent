"""D3 변경 diff 렌더러와 단서 canary.

변경마다 Isaac Lab v2.1.1 소스 파일의 unified diff를 만든다. 코드 변경은 대상 함수 범위 안, 설정 변경은 값이
정의된 클래스·대입 범위 안에서 bench/d3_changes.py의 편집(old → new)을 적용한다. 원본의 주석·서식은 그대로 둔다.

렌더링이 실행과 어긋나지 않게 두 가지를 확인하고, 어긋나면 렌더링하지 않는다.
- 코드: 편집 뒤 함수의 AST가 실제 교체 본문의 AST와 같다(이름·docstring·인자 주석·기본값·지역 변수 주석은 정규화).
- 설정: 편집 줄에서 바뀐 숫자 쌍이 카탈로그의 (기준값, 새 값)과 같다.
canary는 diff에 변경 id·결함/무해/정답 관련 단어·교체 함수 이름·추가 줄 주석이 없는지 본다. 모델에 줄 diff는
canary를 통과한 것만 쓴다. Isaac Sim 없이 CPU에서 돈다.
"""
from __future__ import annotations

import ast
import copy
import difflib
import inspect
import io
from pathlib import Path
import re
import sys
import textwrap
import tokenize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d3_changes as D  # noqa: E402

CLUE_WORDS = ("fault", "harm", "benign", "decoy", "bug", "inject", "upstream", "unilab", "motphys", "hidden",
              "triage", "answer", "noqa", "d3", "결함", "무해", "해로운", "미끼", "정답", "범인")


def read_source(isaaclab: Path, relative: str) -> str:
    return (Path(isaaclab) / relative).read_text(encoding="utf-8").replace("\r\n", "\n")


def _node(tree: ast.Module, qualname: str):
    """'Class.method'·'Outer.Inner'·'NAME'(최상위 대입)을 찾는다. 없거나 여럿이면 ValueError."""
    scope, found = tree.body, None
    for part in qualname.split("."):
        matches = [n for n in scope if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
                   or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == part for t in n.targets))]
        if len(matches) != 1:
            raise ValueError(f"{qualname}: {part}를 하나로 찾지 못했다({len(matches)}개)")
        found = matches[0]
        scope = getattr(found, "body", [])
    return found


def _span(text: str, node) -> tuple[int, int]:
    """노드가 차지하는 줄 전체의 문자 범위(장식자 포함)."""
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return starts[first - 1], starts[node.end_lineno]


def _apply(text: str, span: tuple[int, int], edits) -> str:
    start, end = span
    segment = text[start:end]
    for old, new in edits:
        if segment.count(old) != 1:
            raise ValueError(f"편집 원문이 대상 범위에 정확히 한 번 있지 않다({segment.count(old)}회): {old!r}")
        segment = segment.replace(old, new)
    return text[:start] + segment + text[end:]


def _edit_groups(cid: str) -> dict[str, list[tuple[str, list]]]:
    """파일 → [(범위 이름, 편집)]."""
    change = D.CHANGES[cid]
    groups: dict[str, list] = {}
    if change["kind"] == "code":
        for target in D.code_targets(cid):
            groups.setdefault(target["file"], []).append((target["qualname"], target["edits"]))
    else:
        groups[change["file"]] = [(change["scope"], change["edits"])]
    return groups


def edited_sources(cid: str, isaaclab: Path) -> dict[str, tuple[str, str]]:
    """파일 → (원본, 편집본). 같은 파일의 여러 범위는 뒤에서부터 적용해 앞 범위의 위치를 지킨다."""
    out = {}
    for relative, scopes in _edit_groups(cid).items():
        before = read_source(isaaclab, relative)
        tree = ast.parse(before)
        spans = sorted(((_span(before, _node(tree, scope)), edits) for scope, edits in scopes), reverse=True)
        after = before
        for span, edits in spans:
            after = _apply(after, span, edits)
        out[relative] = (before, after)
    return out


class _Locals(ast.NodeTransformer):
    """함수 안 지역 변수 주석(x: T = v)은 실행되지 않으므로 대입(x = v)과 같게 본다."""

    def visit_AnnAssign(self, node):
        if node.value is None or not isinstance(node.target, ast.Name):
            return node
        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)


def normalized(function: ast.FunctionDef) -> str:
    function = copy.deepcopy(function)
    function.name, function.decorator_list, function.returns = "_", [], None
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    function.body = body
    arguments = function.args
    for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]:
        if arg is not None:
            arg.annotation = None
    arguments.defaults, arguments.kw_defaults = [], [None] * len(arguments.kwonlyargs)
    return ast.dump(_Locals().visit(function), include_attributes=False)


def replacement_ast(replacement) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(inspect.getsource(replacement))).body[0]


def _signature(function: ast.FunctionDef) -> tuple:
    return (ast.dump(function.args), [ast.dump(d) for d in function.decorator_list],
            ast.dump(function.returns) if function.returns else None)


def mangled_names(function: ast.FunctionDef) -> list[str]:
    """클래스 안에서만 이름이 바뀌는(_Class__x) 식별자. 모듈 수준 교체 본문에서는 바뀌지 않아 의미가 달라진다."""
    names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(function) if isinstance(n, ast.Attribute)}
    return sorted(n for n in names if n.startswith("__") and not n.endswith("__"))


def check_code(cid: str, sources: dict[str, tuple[str, str]]) -> None:
    """__code__ 교체는 본문만 바꾸고 원본의 서명(기본값·주석)·장식자를 그대로 둔다. 그 밖의 편집은 실행되지 않으므로 거부한다."""
    for target in D.code_targets(cid):
        before, after = (ast.parse(text) for text in sources[target["file"]])
        rendered, original = _node(after, target["qualname"]), _node(before, target["qualname"])
        replacement = replacement_ast(target["replacement"])
        if _signature(rendered) != _signature(original):
            raise ValueError(f"{cid} {target['qualname']}: 서명·장식자 편집은 __code__ 교체로 실행되지 않는다")
        if "." in target["qualname"] and mangled_names(replacement):
            raise ValueError(f"{cid} {target['qualname']}: 메서드 교체 본문에 이름이 바뀌는 식별자가 있다")
        if normalized(rendered) != normalized(replacement):
            raise ValueError(f"{cid} {target['qualname']}: 렌더링한 본문이 실제 교체 본문과 다르다")


def numbers(text: str) -> list[float]:
    """줄 조각의 숫자 리터럴(단항 -는 앞이 연산자·여는 괄호일 때 부호로 붙인다)."""
    out, previous = [], []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text + "\n").readline):
            if token.type == tokenize.NUMBER:
                negative = (len(previous) >= 1 and previous[-1].string == "-"
                            and (len(previous) < 2 or previous[-2].type == tokenize.OP
                                 and previous[-2].string not in (")", "]", "}")))
                out.append(-float(token.string) if negative else float(token.string))
            if token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                previous.append(token)
    except tokenize.TokenError:
        pass  # 닫히지 않은 괄호의 줄 조각
    return out


def changed_numbers(cid: str) -> list[tuple[float, float]]:
    pairs = []
    for old, new in D.CHANGES[cid]["edits"]:
        before, after = numbers(old), numbers(new)
        if len(before) != len(after):
            raise ValueError(f"{cid}: 편집 전후 숫자 개수가 다르다")
        pairs += [(a, b) for a, b in zip(before, after) if a != b]
    return sorted(pairs)


def field_value(text: str, field: str) -> float | None:
    """'field=값'·'field = 값'·'"field": 값' 꼴의 숫자. 없으면 None."""
    match = re.search(rf'(?<![A-Za-z0-9_]){re.escape(field)}"?\s*[=:]\s*(-?\d[\d.eE+-]*)', text)
    return float(match.group(1)) if match else None


def check_config(cid: str) -> None:
    """바뀐 숫자 전체가 override와 같고, override 키마다 그 필드 이름의 편집 하나가 기준값→새 값으로 바뀐다."""
    change = D.CHANGES[cid]
    if changed_numbers(cid) != sorted(change["overrides"].values()):
        raise ValueError(f"{cid}: 소스 편집의 숫자 변화가 override 기준값·새 값과 다르다")
    for key, pair in change["overrides"].items():
        field = key.rsplit(".", 1)[1]
        edits = [(old, new) for old, new in change["edits"] if field_value(old, field) is not None]
        if len(edits) != 1 or (field_value(edits[0][0], field), field_value(edits[0][1], field)) != pair:
            raise ValueError(f"{cid}: override {key}의 필드가 소스 편집 하나에서 기준값→새 값으로 바뀌지 않는다")


def render(cid: str, isaaclab: Path, context: int = 3) -> str:
    """변경 하나의 unified diff. 실행과의 대조를 통과해야 돌려준다."""
    sources = edited_sources(cid, isaaclab)
    if D.CHANGES[cid]["kind"] == "code":
        check_code(cid, sources)
    else:
        check_config(cid)
    chunks = []
    for relative, (before, after) in sources.items():
        if before == after:
            raise ValueError(f"{cid}: {relative}에 바뀐 줄이 없다")
        chunks.append("".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                                   fromfile=f"a/{relative}", tofile=f"b/{relative}", n=context)))
    return "".join(chunks)


def clue_patterns() -> dict[str, re.Pattern]:
    patterns = {f"id:{cid}": re.compile(rf"(?<![A-Za-z0-9_]){cid}(?![A-Za-z0-9_])") for cid in D.CHANGES}
    patterns.update({f"word:{word}": re.compile(rf"(?<![A-Za-z]){re.escape(word)}", re.IGNORECASE) for word in CLUE_WORDS})
    names = {target["replacement"].__name__ for cid in D.CHANGES for target in D.code_targets(cid)}
    patterns.update({f"name:{name}": re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])") for name in names})
    return patterns


def canary(diff: str) -> list[str]:
    """diff 안의 단서. 빈 목록이면 통과다. 추가 줄의 주석은 원본에 없던 설명이므로 단서로 본다."""
    found = [label for label, pattern in clue_patterns().items() if pattern.search(diff)]
    added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if any("#" in line for line in added):
        found.append("added_comment")
    return found
