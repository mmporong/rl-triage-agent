"""Parent8을 저장소 밖 cwd에서도 불러오는 독립 CLI 진입점."""
from __future__ import annotations

import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if all((candidate / name).is_dir() for name in ("bench", "evals", "src")):
            return candidate
    raise RuntimeError("repository root unavailable")


ROOT = _repo_root()
for location in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    value = str(location)
    if value not in sys.path:
        sys.path.insert(0, value)

import run_agentic_brev_probe_parent_v8 as V8

SCHEMA = V8.SCHEMA
SOURCE = "bench/run_agentic_brev_probe_parent_v9.py"
TEST_SOURCE = "tests/test_run_agentic_brev_probe_parent_v9.py"
SOURCES = V8.SOURCES | {SOURCE, TEST_SOURCE}

V1 = V8.V1
C = V8.C
E = V8.E
fresh_selection = V8.fresh_selection
parent9_scope = V8.parent8_scope


def prepare(*args, **kwargs):
    return V8.prepare(*args, **kwargs)


def lookup(*args, **kwargs):
    return V8.lookup(*args, **kwargs)


def run(*args, **kwargs):
    return V8.run(*args, **kwargs)


def verify_authority(*args, **kwargs):
    return V8.verify_authority(*args, **kwargs)


def _verify_completion(*args, **kwargs):
    return V8._verify_completion(*args, **kwargs)


def main() -> None:
    V8.main()


def __getattr__(name: str):
    return getattr(V8, name)


if __name__ == "__main__":
    main()
