"""agentic play job v2: 고정 HOME cache alias의 canonical 실제 경로를 lease root로 쓰는 얇은 adapter.

v1(``agentic_play_job_v1``)의 계약과 검증을 그대로 쓴다. ``canonical_lease_scope()`` 안에서만 process-global
값 세 개를 바꾸고, 바깥으로 나가면 원래 값으로 되돌린다.

- ``JOB1.LEASE_ROOT``: ``Path.home()/.cache/rl-triage/cloud-gpu-leases``(alias)를 resolve한 canonical 경로.
  자원별 lock 파일 이름·heartbeat·snapshot 위치 규칙은 v1과 같다(자원별 mutex를 나누지 않는다).
- ``JOB1.REQUIRED_SOURCE_FILES``: v1 closure에 v2 모듈·테스트와 capture unit v1을 더한 v2 closure.
- ``RUN1._child_launcher``: RUN5 entry를 RUN1로 바꾼 자식 명령을 다시 RUN2 entry로 바꾼다. SDK·AABA 자식도
  RUN2에서 같은 scope로 들어가 같은 canonical root로 parent lease를 검증한다.

v1의 symlink 조상·owner·start tick·heartbeat·protocol·만료·dev/inode·live 증거 검사는 약화하지 않는다.
alias 자체만 resolve하므로 canonical 경로 조상에 링크가 남으면 v1 검사가 그대로 거부한다.
"""

from __future__ import annotations

import functools
import importlib
import os
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import agentic_play_job_v1 as JOB1

ALIAS_PARTS = (".cache", "rl-triage", "cloud-gpu-leases")
if tuple(Path(JOB1.LEASE_ROOT).parts[-3:]) != ALIAS_PARTS:
    raise ImportError("v1 lease alias 정의가 v2 가정과 다르다")
V2_FILES = frozenset({
    "bench/agentic_play_job_v2.py",
    "bench/run_agentic_play_record_v2.py",
    "bench/run_agentic_capture_unit_v2.py",
    "tests/test_agentic_play_job_v2.py",
    "tests/test_run_agentic_play_record_v2.py",
    "tests/test_agentic_capture_unit_v2.py",
})
REQUIRED_SOURCE_FILES = frozenset(
    set(JOB1.REQUIRED_SOURCE_FILES) | V2_FILES
    | {"bench/run_agentic_capture_unit_v1.py", "tests/test_agentic_capture_unit_v1.py"}
)
RUN2_ENTRY = Path(__file__).resolve().with_name("run_agentic_play_record_v2.py")
OBSERVATION_SCHEMA = "agentic_lease_root_observation_v1"
_LOCK = threading.RLock()
_STATE: dict[str, Any] = {"depth": 0, "root": None, "saved": None, "run1": None}


def lease_alias() -> Path:
    """v1과 같은 고정 alias. env·run별 경로를 받지 않는다."""
    return Path.home().joinpath(*ALIAS_PARTS)


def canonical_lease_root() -> Path:
    """alias를 실제 경로로 푼다(아직 없는 꼬리는 그대로 붙는다)."""
    root = lease_alias().resolve(strict=False)
    if not root.is_absolute():
        raise JOB1.LeaseError("canonical lease root가 절대 경로가 아니다")
    return root


def _retargeting_launcher(original: Callable[..., Any], run1_file: str) -> Callable[..., Any]:
    old = str(Path(run1_file).resolve())
    new = str(RUN2_ENTRY)

    def launcher(popen: Callable[..., Any], job_path: Path, *, before_launch: Callable[[], Any] | None = None) -> Any:
        def retarget(command: Any, **kwargs: Any) -> Any:
            values = list(command)
            if values.count(old) != 1:
                raise ValueError("v2 entry로 바꿀 RUN1 entry가 정확히 하나가 아니다")
            values[values.index(old)] = new
            return popen(values, **kwargs)

        return original(retarget, job_path, before_launch=before_launch)

    return launcher


@contextmanager
def canonical_lease_scope() -> Iterator[Path]:
    """process-global 교체 scope. 같은 thread에서 재진입하면 처음 정한 root를 그대로 쓴다."""
    with _LOCK:
        if _STATE["depth"] == 0:
            if not RUN2_ENTRY.is_file():
                raise JOB1.LeaseError("v2 자식 entry가 없다")
            root = canonical_lease_root()
            run1: Any = importlib.import_module("run_agentic_play_record_v1")  # RUN5 연쇄 import는 scope 진입 때만
            saved = (JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, run1._child_launcher)
            JOB1.LEASE_ROOT = root
            JOB1.REQUIRED_SOURCE_FILES = REQUIRED_SOURCE_FILES
            run1._child_launcher = _retargeting_launcher(saved[2], run1.__file__)
            _STATE.update(root=root, saved=saved, run1=run1)
        _STATE["depth"] += 1
        try:
            yield _STATE["root"]
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, _STATE["run1"]._child_launcher = _STATE["saved"]
                _STATE.update(root=None, saved=None, run1=None)


def scope_active_root() -> Path | None:
    return _STATE["root"]


def _component(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False, "is_symlink": False, "is_reparse_point": False,
                "link_target": None, "is_dir": False}
    reparse = bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    target = None
    if stat.S_ISLNK(info.st_mode) or reparse:
        try:
            target = os.readlink(path)
        except OSError:
            target = None
    return {"path": str(path), "exists": True, "is_symlink": stat.S_ISLNK(info.st_mode),
            "is_reparse_point": reparse, "link_target": target, "is_dir": path.is_dir()}


def observe_lease_root() -> dict[str, Any]:
    """GPU 입장 전에 남길 alias·canonical 관측. 판정·보정 없이 원 상태만 기록한다."""
    home = Path.home()
    alias = lease_alias()
    canonical = scope_active_root() or canonical_lease_root()
    components = [_component(home)]
    cursor = home
    for part in ALIAS_PARTS:
        cursor = cursor / part
        components.append(_component(cursor))
    identity = None
    if canonical.exists():
        info = canonical.stat()
        identity = {"device": info.st_dev, "inode": info.st_ino}
    try:
        JOB1.COMMON._check_ancestors(canonical, "canonical lease root")
        link_free = True
    except ValueError:
        link_free = False
    return {
        "schema": OBSERVATION_SCHEMA,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - Python 3.10
        "home": str(home), "alias": str(alias), "alias_parts": list(ALIAS_PARTS), "components": components,
        "canonical": str(canonical), "canonical_exists": canonical.exists(), "canonical_identity": identity,
        "canonical_ancestors_link_free": link_free, "resolved_by": "Path.resolve(strict=False)",
        "alias_is_canonical": str(alias.absolute()) == str(canonical),
    }


def _scoped(name: str) -> Callable[..., Any]:
    function = getattr(JOB1, name)

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with canonical_lease_scope():
            return function(*args, **kwargs)

    return wrapper


# lease root 또는 source closure를 읽는 v1 API는 모두 scope 안에서 실행한다(서명은 v1과 같다).
build_protocol = _scoped("build_protocol")
validate_protocol = _scoped("validate_protocol")
validate_resource_proof = _scoped("validate_resource_proof")
build_resource_proof = _scoped("build_resource_proof")
validate_job = _scoped("validate_job")
create_job = _scoped("create_job")
lease_paths = _scoped("lease_paths")
verify_lease = _scoped("verify_lease")


@contextmanager
def resource_lease(protocol_path: Path, **kwargs: Any) -> Iterator[dict[str, str]]:
    """v1 resource_lease와 같은 서명. lease가 살아 있는 동안 scope를 유지한다."""
    with canonical_lease_scope(), JOB1.resource_lease(protocol_path, **kwargs) as reference:
        yield reference


def __getattr__(name: str) -> Any:
    """나머지 v1 이름(상수·helper·모듈)은 그대로 넘긴다. LEASE_ROOT만 canonical 값을 돌려준다."""
    if name == "LEASE_ROOT":
        return scope_active_root() or canonical_lease_root()
    try:
        return getattr(JOB1, name)
    except AttributeError as error:
        raise AttributeError(f"agentic_play_job_v2에 {name}이 없다") from error
