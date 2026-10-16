"""등록된 실행을 시작하기 전에 로컬 실험 기록의 seed 사용 여부를 읽기 검사한다.

검사 범위는 저장소 telemetry/실행 기록과 지정 Isaac Lab 로그다. 계획·프로토콜은
실행으로 세지 않는다. 다른 장비나 보관하지 않은 실행의 미사용을 보증하지 않는다.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_RECORD_NAMES = {
    "receipt.json", "result.json", "aggregate.json", "training_attestation.json",
    "execution.json", "train_NONE.json", "train_FAULT.json", "launch.json",
    "request.json", "failure.json",
}
PLANNING_NAMES = {"prereg.json", "protocol.json", "plan.json", "manifest.json", "approval.json"}
SEED_KEY = re.compile(r"(?:^|_)seeds?$")


class SeedReuseError(ValueError):
    """요청한 seed가 실행 기록에 이미 존재한다."""


def _named_seeds(value: str, seeds: set[int]) -> set[int]:
    return {seed for seed in seeds if re.search(rf"(?:^|[_/\\-])s{seed}(?=$|[_./\\-])", value)}


def _record_seeds(value: Any, seeds: set[int], key: str = "") -> set[int]:
    if isinstance(value, dict):
        found: set[int] = set()
        for name, child in value.items():
            found.update(_record_seeds(child, seeds, name))
        return found
    if isinstance(value, list):
        found = set()
        for index, item in enumerate(value):
            if item == "--seed" and index + 1 < len(value):
                token = str(value[index + 1])
                if token.isdecimal() and int(token) in seeds:
                    found.add(int(token))
        for child in value:
            found.update(_record_seeds(child, seeds, key))
        return found
    if SEED_KEY.search(key) and type(value) is int and value in seeds:
        return {value}
    if isinstance(value, str):
        found = _named_seeds(value, seeds)
        for match in re.finditer(r"(?:^|\s)--seed(?:=|\s+)(\d+)(?=$|\s)", value):
            if int(match[1]) in seeds:
                found.add(int(match[1]))
        if SEED_KEY.search(key) and value.isdecimal() and int(value) in seeds:
            found.add(int(value))
        return found
    return set()


def _files(root: Path):
    def fail(error):
        raise ValueError(f"seed 기록을 열거할 수 없음: {error.filename}") from error

    for directory, subdirs, filenames in os.walk(root, followlinks=False, onerror=fail):
        base = Path(directory)
        for name in [*subdirs, *filenames]:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"seed 검사 범위의 symlink는 허용하지 않음: {path.name}")
        yield base, filenames


def assert_fresh_seeds(seeds: Iterable[int], isaaclab: Path, *, repo: Path = ROOT) -> dict[str, Any]:
    values = list(seeds)
    if not values or any(type(seed) is not int or seed < 0 for seed in values) or len(set(values)) != len(values):
        raise ValueError("seed는 중복 없는 음이 아닌 정수 목록이어야 한다")
    requested = set(values)
    repo, isaaclab = Path(repo).resolve(), Path(isaaclab).resolve()
    log_root = isaaclab / "logs"
    if not (repo / "bench").is_dir() or not (repo / "evals").is_dir() or not log_root.is_dir():
        raise ValueError("seed 검사에 필요한 bench/evals/Isaac Lab logs 경로가 없음")
    candidates: set[Path] = set()
    for directory, filenames in _files(repo / "bench"):
        relative = directory.relative_to(repo)
        private = relative.parts[:2] == ("bench", "private")
        for name in filenames:
            if (name.endswith(".telemetry.json") or (private and
                    (name.endswith(".meta.json") or name in PRIVATE_RECORD_NAMES
                     or (directory.name == "steps" and re.fullmatch(r"\d+_.+\.json", name))))):
                candidates.add(directory / name)
    checked_roots = ["<repo>/bench/**/*.telemetry.json", "<repo>/bench/private/execution_records",
                     "<repo>/evals/results", "<repo>/evals/loops", "<isaaclab>/logs"]
    for relative in ("evals/results", "evals/loops"):
        root = repo / relative
        if root.exists():
            for directory, filenames in _files(root):
                candidates.update(directory / name for name in filenames
                                  if name.endswith(".json") and name not in PLANNING_NAMES)
    evidence = []
    for path in sorted(candidates):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"seed 기록 읽기 실패: {path.relative_to(repo).as_posix()}") from error
        used = _named_seeds(path.relative_to(repo).as_posix(), requested) | _record_seeds(record, requested)
        for seed in sorted(used):
            evidence.append({"seed": seed, "path": "<repo>/" + path.relative_to(repo).as_posix()})
    checked = len(candidates)
    for directory, filenames in _files(log_root):
        used = _named_seeds(directory.relative_to(log_root).as_posix(), requested)
        for name in filenames:
            used.update(_named_seeds(name, requested))
            if name in {"env.yaml", "agent.yaml"}:
                path = directory / name
                try:
                    text = path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeError) as error:
                    raise ValueError(f"seed 설정 읽기 실패: {name}") from error
                checked += 1
                for seed in requested:
                    if re.search(rf"(?m)^seed:\s*['\"]?{seed}['\"]?\s*$", text):
                        used.add(seed)
        for seed in sorted(used):
            evidence.append({"seed": seed, "path": "<isaaclab>/logs/" + directory.relative_to(log_root).as_posix()})
    if evidence:
        raise SeedReuseError("이미 기록된 seed: " + json.dumps(evidence, ensure_ascii=False))
    return {
        "schema": "preregistered_seed_freshness_v1", "status": "fresh", "seeds": values,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "checked_roots": checked_roots,
        "checked_file_count": checked, "evidence": [],
        "scope": "현재 저장소의 실행 기록과 지정 Isaac Lab 로그에 한정한다",
    }
