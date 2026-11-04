"""U2 post runner에 등록된 bounded GPU 재관측만 결합한다."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _find_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "AGENTS.md").is_file() and (
            parent / "evals/registered_desktop_gpu_retry_v1.py"
        ).is_file():
            return parent
    raise RuntimeError("repository root is unavailable")


ROOT = _find_repo()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals import registered_desktop_gpu_retry_v1 as RETRY

CORE: Any = importlib.import_module(
    "evals.registered_post_actuator_confirmation_v5_core"
)

RUNNER_REL = "evals/run_registered_post_actuator_confirmation_v5.py"
CORE_REL = "evals/registered_post_actuator_confirmation_v5_core.py"
RETRY_RELATIONS = (
    "evals/registered_desktop_gpu_retry_v1.py",
    "tests/test_registered_desktop_gpu_retry_v1.py",
    "bench/protocols/registered_desktop_gpu_retry_v1.json",
)
SOURCE_RELATIONS = tuple(
    dict.fromkeys((RUNNER_REL, CORE_REL, *CORE.SOURCE_RELATIONS, *RETRY_RELATIONS))
)
_LOCK = threading.RLock()

# Public constants consumed by the registration finalizer and existing tests.
OUTER_SCHEMA = CORE.OUTER_SCHEMA
AUTHORITY_SCHEMA = CORE.AUTHORITY_SCHEMA
CHILD_REL = CORE.CHILD_REL
INNER_REL = CORE.INNER_REL
POST = CORE.POST
PRODUCER = CORE.PRODUCER
RUN_ROOT = CORE.RUN_ROOT


class _GpuNamespaceProxy:
    """CORE 전용 query override. shared module 자체는 변경하지 않는다."""

    def __init__(self, base: Any) -> None:
        self._base = base._base if isinstance(base, _GpuNamespaceProxy) else base

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def query_registered_desktop_gpu_shared(self) -> dict[str, Any]:
        return RETRY.query_registered_desktop_gpu_retry()


@contextmanager
def retry_observation_scope() -> Iterator[None]:
    """Base predicate를 유지하고 등록된 두 process-churn 관측만 재조회한다."""
    with _LOCK:
        old_relations = CORE.SOURCE_RELATIONS
        old_gpu = CORE.GPU
        CORE.SOURCE_RELATIONS = SOURCE_RELATIONS
        CORE.GPU = _GpuNamespaceProxy(old_gpu)
        try:
            yield
        finally:
            CORE.GPU = old_gpu
            CORE.SOURCE_RELATIONS = old_relations


def validate_outer(value: object, *, repo_root: Path = CORE.ROOT) -> dict[str, Any]:
    with retry_observation_scope():
        return CORE.validate_outer(value, repo_root=repo_root)


def _authority(
    value: object, source_path: Path, outer_path: Path, outer: Mapping[str, Any]
) -> dict[str, Any]:
    with retry_observation_scope():
        return CORE._authority(value, source_path, outer_path, outer)


def _git_anchor(
    anchor: str,
    outer: Mapping[str, Any],
    outer_path: Path,
    *,
    repo_root: Path = CORE.ROOT,
) -> None:
    with retry_observation_scope():
        CORE._git_anchor(anchor, outer, outer_path, repo_root=repo_root)


def _source(value: object) -> dict[str, Any]:
    return CORE._source(value)


def _primary_budget(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return CORE._primary_budget(*args, **kwargs)


def run(
    source_path: Path,
    outer_path: Path,
    authority_path: Path,
    isaaclab: Path,
    tag: str,
    out_dir: Path,
) -> dict[str, Any]:
    with retry_observation_scope():
        return CORE.run(source_path, outer_path, authority_path, isaaclab, tag, out_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--source-inputs", required=True, type=Path)
    parser.add_argument("--outer-protocol", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(
            args.source_inputs,
            args.outer_protocol,
            args.authority,
            args.isaaclab,
            args.tag,
            args.out_dir,
        )
        print(
            json.dumps(
                {"status": result["status"], "sdk_launches": result["sdk_launches"]},
                sort_keys=True,
            )
        )
        return 0 if result["status"] == "complete" else 3
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(
            json.dumps(
                {"status": "rejected", "error_type": type(error).__name__},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
