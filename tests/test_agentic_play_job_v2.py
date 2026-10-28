from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_fault_playback_v1 as PLAYBACK  # pyright: ignore[reportMissingImports]
import agentic_play_job_v1 as JOB1  # pyright: ignore[reportMissingImports]
import agentic_play_job_v2 as JOB2  # pyright: ignore[reportMissingImports]
import run_agentic_play_record_v1 as RUN1  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as FIX  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

ORIGINAL = (JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, RUN1._child_launcher)


@dataclass
class LinkedHome:
    """CPU fixture: HOME/.cache가 다른 실제 디렉터리를 가리키는 링크(Windows junction, POSIX symlink)."""

    home: Path
    alias: Path
    canonical: Path
    link_kind: str
    real_cache: Path


def make_dir_link(link: Path, target: Path) -> str:
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True,
                                check=False)
        if result.returncode != 0:
            pytest.skip("이 Windows 계정은 junction을 만들 수 없다")
        return "junction"
    link.symlink_to(target, target_is_directory=True)
    return "symlink"


def remove_dir_link(link: Path) -> None:
    if os.name == "nt":
        os.rmdir(link)  # junction만 지우고 대상은 남긴다
    else:
        link.unlink()


def linked_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LinkedHome:
    home = tmp_path / "home"
    home.mkdir()
    real_cache = tmp_path / "real-cache"
    kind = make_dir_link(home / ".cache", real_cache)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    alias = home / ".cache" / "rl-triage" / "cloud-gpu-leases"
    canonical = (real_cache / "rl-triage" / "cloud-gpu-leases").resolve()
    assert Path.home() == home and JOB2.lease_alias() == alias
    return LinkedHome(home, alias, canonical, kind, real_cache)


def hold_scope() -> Path:
    stack = ExitStack()
    FIX._OPEN.append(stack)
    return stack.enter_context(JOB2.canonical_lease_scope())


def build_world_v2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, lease: bool = True,
                   hold: bool = True) -> tuple[Any, LinkedHome]:
    """v1 fixture 조립을 v2 scope 안에서 한다(v2 closure·canonical root). LEASE_ROOT는 monkeypatch하지 않는다."""
    layout = linked_home(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    stack = ExitStack()
    if hold:
        FIX._OPEN.append(stack)
    stack.enter_context(JOB2.canonical_lease_scope())
    repo, commit, sources = FIX._repo(tmp_path)  # scope 안이라 v2 closure 파일까지 커밋된다
    monkeypatch.setattr(PLAYBACK, "ROOT", repo.resolve())
    monkeypatch.setattr(JOB1.COMMON, "_python_executable", lambda path: Path(path))
    lab, runtime, rows = FIX._lab(tmp_path)
    protocol = JOB2.build_protocol(
        protocol_id="agentic-play-20261028c", registered_at_utc=(now - timedelta(minutes=1)).isoformat(),
        source_commit=commit, source_sha256_lf=sources, runtime_source_sha256_lf=runtime,
        recording_target=FIX.TARGET, compute_quote_usd_per_h="1.74", batch_budget_usd="2.00",
        deadline_utc=(now + timedelta(hours=2)).isoformat(), checkpoints=rows,
        faults={FIX.F2_ID: "F2", FIX.NONE_ID: "NONE"},
        pairs=[{"pair_id": "go2-s2033", "seed": 2033, "case": FIX.F2_ID, "reference": FIX.NONE_ID}],
    )
    protocol_path = FIX.commit_protocol(repo, "agentic_play_fixture.json", protocol)
    reference: dict[str, str] = {}
    lease_stack = ExitStack()
    if lease:
        FIX._OPEN.append(lease_stack)
        reference = lease_stack.enter_context(JOB2.resource_lease(protocol_path, repo=repo))
    proof = FIX.write_proof(tmp_path, protocol_path, observed=now, lease=reference or FIX.OMIT)
    if not hold:
        stack.close()
    world = FIX.World(tmp_path, now, repo, commit, sources, lab, runtime, rows, protocol_path, protocol, proof,
                      reference, lease_stack)
    return world, layout


def _protocol(world: Any, path: Path | None = None) -> tuple[dict[str, Any], str]:
    target: Path = world.protocol_path if path is None else path
    return JOB2.validate_protocol(target, repo=world.repo, now=world.now), JOB1._sha(target, lf=True)


# --- scope -----------------------------------------------------------------------------------


def test_v2_exposes_v1_api_names() -> None:
    for name in ("resource_lease", "verify_lease", "validate_resource_proof", "lease_paths", "build_resource_proof",
                 "validate_protocol", "build_protocol", "validate_job", "create_job", "write_new", "_read", "_sha",
                 "_utc", "COMMON", "REMOTE", "OWNED", "RECORD", "LeaseError", "TECHNICAL_LIMITS",
                 "sidecar_directory_for", "motor_catalog", "batch_layout", "MODE", "LEASE_ROOT"):
        assert hasattr(JOB2, name), name
    with pytest.raises(AttributeError):
        _ = JOB2.not_a_v1_name


def test_scope_switches_three_globals_and_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = linked_home(tmp_path, monkeypatch)
    assert (JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, RUN1._child_launcher) == ORIGINAL
    with JOB2.canonical_lease_scope() as root:
        assert root == layout.canonical and JOB1.LEASE_ROOT == layout.canonical
        assert JOB1.REQUIRED_SOURCE_FILES == JOB2.REQUIRED_SOURCE_FILES and JOB2.V2_FILES <= JOB2.REQUIRED_SOURCE_FILES
        assert RUN1._child_launcher is not ORIGINAL[2]
        assert JOB2.lease_paths("vm-a")["lock"] == layout.canonical / "vm-a.lock"
        assert JOB2.LEASE_ROOT == layout.canonical
        with JOB2.canonical_lease_scope() as nested:  # 재진입은 처음 정한 root를 그대로 쓴다
            assert nested is root
    assert (JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, RUN1._child_launcher) == ORIGINAL
    with pytest.raises(RuntimeError, match="boom"), JOB2.canonical_lease_scope():
        raise RuntimeError("boom")
    assert (JOB1.LEASE_ROOT, JOB1.REQUIRED_SOURCE_FILES, RUN1._child_launcher) == ORIGINAL


def test_retargeting_launcher_requires_exactly_one_run1_entry() -> None:
    seen: list[list[str]] = []

    def original(popen: Any, job_path: Path, *, before_launch: Any = None) -> Any:
        return popen(["py", "-B", str(Path(RUN1.__file__).resolve()), "--sdk-child"])

    launcher = JOB2._retargeting_launcher(original, RUN1.__file__)
    launcher(lambda command, **_kwargs: seen.append(command), Path("job.json"))
    assert seen == [["py", "-B", str(JOB2.RUN2_ENTRY), "--sdk-child"]]

    def wrong(popen: Any, job_path: Path, *, before_launch: Any = None) -> Any:
        return popen(["py", "-B", "other.py"])

    with pytest.raises(ValueError, match="RUN1 entry"):
        JOB2._retargeting_launcher(wrong, RUN1.__file__)(lambda *_a, **_k: None, Path("job.json"))


# --- alias·canonical lease -----------------------------------------------------------------------


def test_v1_alias_root_reproduces_link_failure_and_v2_uses_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, layout = build_world_v2(tmp_path, monkeypatch, lease=False, hold=False)
    monkeypatch.setattr(JOB1, "LEASE_ROOT", layout.alias)  # v1이 VM에서 쓴 alias 그대로
    with (pytest.raises(ValueError, match="lease root 조상에 symlink/reparse"),
          JOB1.resource_lease(world.protocol_path, repo=world.repo)):
        pass
    with JOB2.resource_lease(world.protocol_path, repo=world.repo) as reference:
        snapshot = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
        assert snapshot["lock_path"] == str(layout.canonical / f"{FIX.TARGET['resource_id']}.lock")
        assert Path(reference["path"]).parent == layout.canonical / "snapshots"
        protocol, sha = _protocol(world)
        assert JOB2.verify_lease(reference, protocol=protocol, protocol_sha=sha)["resource_id"] == FIX.TARGET["resource_id"]
    assert JOB1.LEASE_ROOT == layout.alias  # scope 밖 원래 값 복원


def test_alias_and_canonical_contenders_share_one_lock_inode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, layout = build_world_v2(tmp_path, monkeypatch)
    rid = FIX.TARGET["resource_id"]
    assert os.path.samefile(layout.alias / f"{rid}.lock", layout.canonical / f"{rid}.lock")
    with pytest.raises(JOB1.LeaseError, match="이미"), JOB2.resource_lease(world.protocol_path, repo=world.repo):
        pass
    with pytest.raises(JOB1.OWNED.B.BackendError), JOB1.OWNED._gpu_lock(layout.alias / f"{rid}.lock"):
        pass  # alias 경로로 열어도 같은 inode라 잠겨 있다
    other = FIX.other_protocol(world, "parallel-l40s")
    with JOB2.resource_lease(other, repo=world.repo) as reference:  # 다른 자원은 병렬 허용
        protocol, sha = _protocol(world, other)
        assert JOB2.verify_lease(reference, protocol=protocol, protocol_sha=sha)["resource_id"] == "parallel-l40s"


def test_existing_lease_rejections_are_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, _layout = build_world_v2(tmp_path, monkeypatch)
    protocol, sha = _protocol(world)
    with pytest.raises(JOB1.LeaseError, match="SHA"):
        JOB2.verify_lease(dict(world.lease, sha256="0" * 64), protocol=protocol, protocol_sha=sha)
    later = datetime.now(timezone.utc) + timedelta(seconds=JOB1.HEARTBEAT_MAX_AGE_S + 5)  # noqa: UP017
    with pytest.raises(JOB1.LeaseError, match="heartbeat"):
        JOB2.verify_lease(world.lease, protocol=protocol, protocol_sha=sha, now=later)
    world.lease_stack.close()
    with pytest.raises(JOB1.LeaseError, match="heartbeat"):  # inactive
        JOB2.verify_lease(world.lease, protocol=protocol, protocol_sha=sha)


def _child_verify(world: Any, layout: LinkedHome, *, module: str) -> subprocess.CompletedProcess[str]:
    script = "\n".join([
        "import json, sys",
        "from pathlib import Path",
        "sys.dont_write_bytecode = True",
        f"[sys.path.insert(0, p) for p in {[str(ROOT / name) for name in ('src', 'evals', 'bench')]!r}]",
        "import agentic_fault_playback_v1 as PLAYBACK",
        f"import {module} as J",
        f"PLAYBACK.ROOT = Path({str(world.repo.resolve())!r})",
        f"path = Path({str(world.protocol_path)!r})",
        f"protocol = J.validate_protocol(path, repo=Path({str(world.repo)!r}))",
        f"info = J.verify_lease({world.lease!r}, protocol=protocol, protocol_sha=J._sha(path, lf=True))",
        "print(json.dumps({'root': str(J.LEASE_ROOT), 'lock_proof': info['lock_proof']}))",
    ])
    env = {**os.environ, "HOME": str(layout.home), "USERPROFILE": str(layout.home), "PYTHONUTF8": "1"}
    return subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, encoding="utf-8",
                          env=env, timeout=180, check=False)


def test_child_process_with_v2_verifies_parent_lease_on_same_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, layout = build_world_v2(tmp_path, monkeypatch)
    v2 = _child_verify(world, layout, module="agentic_play_job_v2")
    assert v2.returncode == 0, v2.stderr
    assert json.loads(v2.stdout.strip().splitlines()[-1])["root"] == str(layout.canonical)
    v1 = _child_verify(world, layout, module="agentic_play_job_v1")  # alias root로는 같은 lease를 검증할 수 없다
    assert v1.returncode != 0 and "LeaseError" in v1.stderr


def test_alias_repointed_after_lease_is_rejected_by_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, layout = build_world_v2(tmp_path, monkeypatch)
    remove_dir_link(layout.home / ".cache")
    make_dir_link(layout.home / ".cache", tmp_path / "other-cache")
    moved = _child_verify(world, layout, module="agentic_play_job_v2")
    assert moved.returncode != 0 and "LeaseError" in moved.stderr


# --- observation·source closure -----------------------------------------------------------------


def test_observation_records_alias_components_and_canonical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = linked_home(tmp_path, monkeypatch)
    with JOB2.canonical_lease_scope():
        observation = JOB2.observe_lease_root()
    assert observation["schema"] == JOB2.OBSERVATION_SCHEMA
    assert observation["alias"] == str(layout.alias) and observation["canonical"] == str(layout.canonical)
    assert observation["alias_is_canonical"] is False and observation["canonical_ancestors_link_free"] is True
    cache = observation["components"][1]
    assert cache["path"] == str(layout.home / ".cache")
    assert (cache["is_reparse_point"] if layout.link_kind == "junction" else cache["is_symlink"]) is True
    assert cache["link_target"] is not None and "real-cache" in cache["link_target"]


def test_v2_source_closure_is_required_and_bound_to_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, _layout = build_world_v2(tmp_path, monkeypatch, lease=False)
    args = FIX._build_args(world)
    for missing in sorted(JOB2.V2_FILES | {"bench/run_agentic_capture_unit_v1.py"}):
        sources = {key: value for key, value in world.sources.items() if key != missing}
        with pytest.raises(ValueError, match="source closure"):
            JOB2.build_protocol(**{**args, "source_sha256_lf": sources})
    JOB2.validate_protocol(world.protocol_path, repo=world.repo, now=world.now)
    FIX._write(world.repo / "bench/run_agentic_play_record_v2.py", "# changed after commit\n")
    with pytest.raises(ValueError):
        JOB2.validate_protocol(world.protocol_path, repo=world.repo, now=world.now)
