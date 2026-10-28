"""JOB3: F1 scope 안에서만 F1·정상 pair가 v1 계약을 통과하고, checkpoint는 historical base catalog 행과 같아야 한다."""
from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_f1_playback_v1 as PB  # pyright: ignore[reportMissingImports]
import agentic_fault_playback_v1 as V1  # pyright: ignore[reportMissingImports]
import agentic_play_job_v2 as JOB2  # pyright: ignore[reportMissingImports]
import agentic_play_job_v3 as JOB3  # pyright: ignore[reportMissingImports]
import record_agentic_state_sidecar_v1 as SIDECAR  # pyright: ignore[reportMissingImports]
import run_vision_transferred_reuse_v1 as REUSE  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v1 as FIX  # pyright: ignore[reportMissingImports]
import test_agentic_play_job_v2 as T2  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v1 import _close_leases  # noqa: F401  # pyright: ignore[reportMissingImports]

JOB1 = JOB2.JOB1
F1_ID, NONE_ID = "source-s2221-f1", "source-s2221-none"
PAIR = {"pair_id": "source-s2221", "seed": 2221, "case": F1_ID, "reference": NONE_ID}
WORLD_EXTRA = JOB3.SOURCE_FILES | {
    "bench/run_agentic_play_record_v5.py", "tests/test_run_agentic_play_record_v5.py",
    "bench/run_agentic_capture_unit_v7.py", "tests/test_agentic_capture_unit_v7.py",
}
BASE_GLOBALS = {name: getattr(JOB1, name) for name in JOB3._BASE}


@contextmanager
def f1_fresh() -> Iterator[None]:
    """같은 프로세스에서 새 자식의 import 직후 상태(F1 scope 없음)를 흉내 낸다."""
    state = dict(JOB3._STATE)
    saved = {name: getattr(JOB1, name) for name in JOB3._BASE}
    JOB3._STATE.update(depth=0)
    for name, value in JOB3._BASE.items():
        setattr(JOB1, name, value)
    try:
        yield
    finally:
        JOB3._STATE.update(state)
        for name, value in saved.items():
            setattr(JOB1, name, value)


def _repo_f1(root: Path) -> tuple[Path, str, dict[str, str]]:
    repo = root / "repo"
    repo.mkdir(parents=True)
    for args in (("init", "-q"), ("config", "user.email", "fixture@example.com"), ("config", "user.name", "fixture"),
                 ("config", "core.autocrlf", "false")):
        FIX._git(repo, *args)
    files = set(JOB1.REQUIRED_SOURCE_FILES) | WORLD_EXTRA
    for relative in sorted(files):
        if relative == SIDECAR.SOURCE_PATH:  # sidecar worker는 실제 파일 SHA 결합을 확인한다
            FIX._write(repo / relative, (ROOT / relative).read_bytes().replace(b"\r\n", b"\n"))
        else:
            FIX._write(repo / relative, f"# fixture {relative}\n")
    FIX._git(repo, "add", "-A")
    FIX._git(repo, "commit", "-q", "-m", "fixture")
    return repo, FIX._git(repo, "rev-parse", "HEAD"), {name: JOB1._sha(repo / name, lf=True) for name in files}


def _lab_f1(root: Path) -> tuple[Path, dict[str, str], list[dict[str, Any]]]:
    lab = root / "isaaclab"
    runtime = {name: JOB1._sha(FIX._write(lab / name, f"# runtime {name}\n"), lf=True)
               for name in sorted(JOB1.REQUIRED_RUNTIME_FILES)}
    rows = []
    for identity, run in ((F1_ID, "2026-10-26_08-21-05_vision_s2221_f1"),
                          (NONE_ID, "2026-10-26_08-17-51_vision_s2221_none")):
        path = FIX._write(lab / "logs" / "rsl_rl" / "unitree_go2_flat" / run / "model_299.pt", identity.encode())
        rows.append({"logical_id": identity, "isaaclab_relative": path.relative_to(lab).as_posix(),
                     "sha256": JOB1._sha(path)})
    return lab, runtime, rows


def build_world_f1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, T2.LinkedHome]:
    """CPU fixture. checkpoint는 가짜 파일이라 historical catalog 대신 fixture 행으로 lineage를 대신한다."""
    layout = T2.linked_home(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    stack = ExitStack()
    FIX._OPEN.append(stack)
    stack.enter_context(JOB2.canonical_lease_scope())
    stack.enter_context(JOB3.f1_play_scope())
    repo, commit, sources = _repo_f1(tmp_path)
    monkeypatch.setattr(V1, "ROOT", repo.resolve())
    monkeypatch.setattr(PB, "_hidden_category", lambda: "reward")
    monkeypatch.setattr(JOB1.COMMON, "_python_executable", lambda path: Path(path))
    lab, runtime, rows = _lab_f1(tmp_path)
    monkeypatch.setattr(JOB3, "historical_rows", lambda repo=None: {row["logical_id"]: dict(row) for row in rows})
    protocol = JOB3.build_protocol(
        protocol_id="agentic-f1-render-20261028a", registered_at_utc=(now - timedelta(minutes=1)).isoformat(),
        source_commit=commit, source_sha256_lf=sources, runtime_source_sha256_lf=runtime,
        recording_target=FIX.TARGET, compute_quote_usd_per_h="1.74", batch_budget_usd="2.00",
        deadline_utc=(now + timedelta(hours=2)).isoformat(), checkpoints=rows,
        faults={F1_ID: "F1", NONE_ID: "NONE"}, pairs=[dict(PAIR)],
    )
    protocol_path = FIX.commit_protocol(repo, "agentic_f1_fixture.json", protocol)
    lease_stack = ExitStack()
    FIX._OPEN.append(lease_stack)
    reference = lease_stack.enter_context(JOB3.resource_lease(protocol_path, repo=repo))
    proof = FIX.write_proof(tmp_path, protocol_path, observed=now, lease=reference)
    world = FIX.World(tmp_path, now, repo, commit, sources, lab, runtime, rows, protocol_path, protocol, proof,
                      reference, lease_stack)
    return world, layout


# --- scope ------------------------------------------------------------------------------------


def test_scope_swaps_seven_globals_nests_and_restores() -> None:
    assert BASE_GLOBALS["PROTOCOL_SCHEMA"] == "agentic_play_protocol_v1" and BASE_GLOBALS["PLAYBACK"] is V1
    with JOB3.f1_play_scope():
        with JOB3.f1_play_scope():
            assert JOB1.PROTOCOL_SCHEMA == JOB3.PROTOCOL_SCHEMA and JOB1.JOB_SCHEMA == JOB3.JOB_SCHEMA
            assert JOB1.FAULT_IDS == ("NONE", "F1") and JOB1.PLAYBACK is PB
            assert JOB1.INTERPRETATION is JOB3.INTERPRETATION and JOB1._pairs is JOB3._pairs
            assert JOB1._registrations is JOB3._registrations and JOB3._STATE["depth"] == 2
        assert JOB3.scope_active()
    assert {name: getattr(JOB1, name) for name in JOB3._BASE} == BASE_GLOBALS and not JOB3.scope_active()


def test_scope_refuses_foreign_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JOB1, "_pairs", lambda value, registrations: [])
    with pytest.raises(RuntimeError, match="이미 교체"), JOB3.f1_play_scope():
        pass
    assert JOB1.PROTOCOL_SCHEMA == BASE_GLOBALS["PROTOCOL_SCHEMA"] and JOB3._STATE["depth"] == 0


# --- historical catalog·protocol 본문 ------------------------------------------------------------


def test_historical_rows_are_base_catalog_and_match_reuse_constants(tmp_path: Path) -> None:
    rows = JOB3.historical_rows(ROOT)
    assert set(rows) == {"source-s2220-none", "source-s2220-f1", F1_ID, NONE_ID}
    for identity, value in REUSE.CHECKPOINTS.items():
        assert rows[identity] == {"logical_id": identity, **value}
    copy = tmp_path / JOB3.HISTORICAL_CATALOG["path"]
    copy.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / JOB3.HISTORICAL_CATALOG["path"], copy)
    copy.write_bytes(copy.read_bytes().replace(b"model_299.pt", b"model_298.pt", 1))
    with pytest.raises(ValueError, match="SHA"):
        JOB3.historical_rows(tmp_path)


def _real_args(**changes: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    rows = JOB3.historical_rows(ROOT)
    with JOB2.canonical_lease_scope():
        required = set(JOB1.REQUIRED_SOURCE_FILES)
    sources = dict.fromkeys(sorted(required | WORLD_EXTRA), "0" * 64)
    sources["bench/hidden_faults.py"] = JOB1._sha(ROOT / "bench/hidden_faults.py", lf=True)
    args = {
        "protocol_id": "agentic-f1-render-s2221-draft", "registered_at_utc": (now - timedelta(minutes=1)).isoformat(),
        "source_commit": "0" * 40, "source_sha256_lf": sources,
        "runtime_source_sha256_lf": dict.fromkeys(sorted(JOB1.REQUIRED_RUNTIME_FILES), "0" * 64),
        "recording_target": FIX.TARGET, "compute_quote_usd_per_h": "2.18064", "batch_budget_usd": "2.00",
        "deadline_utc": (now + timedelta(hours=2)).isoformat(), "checkpoints": [rows[F1_ID], rows[NONE_ID]],
        "faults": {F1_ID: "F1", NONE_ID: "NONE"}, "pairs": [dict(PAIR)],
    }
    args.update(changes)
    return args


def test_build_with_real_historical_catalog_rows() -> None:
    value = JOB3.build_protocol(**_real_args())
    assert value["schema"] == JOB3.PROTOCOL_SCHEMA and value["interpretation"] == JOB3.INTERPRETATION
    assert value["playback_registrations"][F1_ID]["schema"] == PB.SCHEMA
    assert value["playback_registrations"][F1_ID]["fault_id"] == "F1"
    assert value["playback_registrations"][NONE_ID] == V1.registration(NONE_ID, REUSE.CHECKPOINTS[NONE_ID]["sha256"], "NONE")
    assert value["preregistration"]["prediction"] == JOB3.PREDICTION and value["pairs"] == [PAIR]
    assert {name: getattr(JOB1, name) for name in JOB3._BASE} == BASE_GLOBALS


def _case(case: str) -> dict[str, Any]:
    rows = JOB3.historical_rows(ROOT)
    if case == "f2-fault":
        return _real_args(faults={F1_ID: "F2", NONE_ID: "NONE"})
    if case == "reversed":
        return _real_args(pairs=[dict(PAIR, case=NONE_ID, reference=F1_ID)])
    if case == "seed":
        return _real_args(pairs=[dict(PAIR, seed=2220)])
    if case == "tampered-sha":
        return _real_args(checkpoints=[dict(rows[F1_ID], sha256="e" * 64), rows[NONE_ID]])
    if case == "renamed":
        renamed = dict(rows[F1_ID], logical_id="go2-s2221-f1")
        return _real_args(checkpoints=[renamed, rows[NONE_ID]], faults={"go2-s2221-f1": "F1", NONE_ID: "NONE"},
                          pairs=[dict(PAIR, case="go2-s2221-f1")])
    mixed = rows["source-s2220-f1"]
    return _real_args(checkpoints=[mixed, rows[NONE_ID]], faults={"source-s2220-f1": "F1", NONE_ID: "NONE"},
                      pairs=[dict(PAIR, case="source-s2220-f1")])


@pytest.mark.parametrize(("case", "message"), [
    ("f2-fault", "NONE·F1"), ("reversed", "F1 case"), ("seed", "seed"), ("tampered-sha", "historical"),
    ("renamed", "historical"), ("s2220-mix", "seed")])
def test_build_rejects_f2_reversal_seed_and_lineage_errors(case: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        JOB3.build_protocol(**_case(case))
    assert {name: getattr(JOB1, name) for name in JOB3._BASE} == BASE_GLOBALS


def test_existing_contract_rejects_f1_and_v3_body_outside_scope() -> None:
    with pytest.raises(ValueError, match="등록 playback"):  # v1 playback.registration이 F1을 먼저 거부한다
        JOB2.build_protocol(**_real_args())
    value = JOB3.build_protocol(**_real_args())
    with JOB2.canonical_lease_scope(), pytest.raises(ValueError, match="기본 계약"):
        JOB1._check_body(value, now=datetime.now(timezone.utc), require_current=False)  # noqa: UP017


def test_registered_f2_protocol_passes_outside_and_fails_inside_scope() -> None:
    path = ROOT / "bench/protocols/agentic_motor_render_preflight_s2033_20261028_v11.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    registered = JOB1._utc(value["registered_at_utc"], "registered")
    with JOB2.canonical_lease_scope():
        assert JOB1._check_body(value, now=registered, require_current=False)["pairs"][0]["case"] == "motor-s2033-f2"
        with JOB3.f1_play_scope(), pytest.raises(ValueError, match="기본 계약"):
            JOB1._check_body(value, now=registered, require_current=False)


# --- git fixture world: lease·입장·job ----------------------------------------------------------


def test_world_jobs_bind_f1_pair_and_validate_only_inside_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, _layout = build_world_f1(tmp_path, monkeypatch)
    jobs = {}
    for identity in (NONE_ID, F1_ID):
        jobs[identity] = JOB3.create_job(world.protocol_path, world.proof, JOB1._sha(world.proof), world.lab,
                                         Path(sys.executable), tmp_path / identity, identity, repo=world.repo)
        validated = JOB3.validate_job(jobs[identity])
        assert validated["job"]["schema"] == JOB3.JOB_SCHEMA and validated["pair"] == PAIR
        assert validated["playback_registration"]["fault_id"] == ("F1" if identity == F1_ID else "NONE")
    with f1_fresh(), pytest.raises(ValueError, match="schema"):
        JOB2.validate_job(jobs[F1_ID])
