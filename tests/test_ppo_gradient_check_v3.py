"""실제 Isaac Python 3.10 child에서 lease 파싱과 학습 진입을 검증한다."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import ppo_gradient_check_v3 as Q

STAMP = "2026-10-16T23:02:45.4686492+00:00"
NOW = datetime(2026, 10, 16, 23, 2, 46, tzinfo=timezone.utc)


def lease_fixture():
    return {"owner": Q.LEASE_OWNER, "state": "running", "childPid": 10, "wrapperPid": 20,
        "completedUtc": None, "acquiredUtc": "2026-10-16T23:02:25.0673901Z",
        "heartbeatUtc": STAMP, "leaseId": "7656aeb6-e588-4e65-93e8-7aaa18b65674"}


CHAIN = [{"pid": 10, "parent": 20, "name": "python.exe"},
         {"pid": 20, "parent": 30, "name": "powershell.exe"}]


def test_valid_lease_retains_original_metadata():
    metadata = lease_fixture()
    result = Q.validate_lease(metadata, CHAIN, 10, NOW)
    assert result["heartbeatUtc"] == STAMP
    assert result["acquiredUtc"] == metadata["acquiredUtc"]
    assert metadata == lease_fixture()


@pytest.mark.parametrize("age,valid", [(0, True), (60, True), (60.000001, False), (-0.000001, False)])
def test_heartbeat_age_window_is_unchanged(age, valid):
    now = Q.parse_lease_timestamp(STAMP) + timedelta(seconds=age)
    if valid:
        assert Q.validate_lease(lease_fixture(), CHAIN, 10, now)
    else:
        with pytest.raises(ValueError):
            Q.validate_lease(lease_fixture(), CHAIN, 10, now)


@pytest.mark.parametrize("field,value", [
    ("owner", "other-owner"), ("state", "completed"), ("childPid", 11), ("wrapperPid", 30),
    ("completedUtc", STAMP), ("leaseId", "bad"), ("heartbeatUtc", "bad"),
    ("acquiredUtc", "bad"), ("acquiredUtc", "2026-10-16T23:02:46.0000000Z"),
])
def test_bad_lease_is_not_admitted(field, value):
    metadata = lease_fixture()
    metadata[field] = value
    with pytest.raises(ValueError):
        Q.validate_lease(metadata, CHAIN, 10, NOW)


@pytest.mark.parametrize("change", ["no_child", "no_wrapper", "wrong_parent", "wrong_binary"])
def test_ancestor_constraints_are_preserved(change):
    chain = deepcopy(CHAIN)
    if change == "no_child":
        chain.pop(0)
    elif change == "no_wrapper":
        chain.pop()
    elif change == "wrong_parent":
        chain[0]["parent"] = 99
    else:
        chain[1]["name"] = "unowned.exe"
    with pytest.raises(ValueError):
        Q.validate_lease(lease_fixture(), chain, 10, NOW)


def test_naive_validation_clock_is_rejected():
    with pytest.raises(ValueError):
        Q.validate_lease(lease_fixture(), CHAIN, 10, NOW.replace(tzinfo=None))


def test_v3_only_changes_infrastructure_and_seed_cohort():
    old, new = Q.V2.fixed_protocol(), Q.fixed_protocol()
    assert {key for key in old if old[key] != new[key]} == {"schema", "seeds", "runs", "supersedes"}
    assert set(new) - set(old) == {"lease_timestamp_contract"}
    assert new["supersedes"]["previous_invalid"] == old["supersedes"]
    assert set(Q.SEEDS).isdisjoint(Q.V2.SEEDS)
    assert {"evals/lease_timestamps.py", "tests/test_lease_timestamps.py"} <= Q.CODE


# child는 실제 3.10 런타임에서 실행한다. 공용 lease 파일·조상 조회·lock은
# fixture로 대체하되 parser/validate_lease/lease_context/verify_training은 실행한다.
CPU_CHILD = r'''
import json, os, pathlib, sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
assert sys.version_info[:2] == (3, 10), sys.version
assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
root, output = map(pathlib.Path, sys.argv[1:3])
sys.path[:0] = [str(root / "evals"), str(root / "bench")]
import ppo_gradient_check_v3 as Q
import ppo_gradient_training_v3 as T
from lease_timestamps import parse_lease_timestamp
stamp = "2026-10-16T23:02:45.4686492+00:00"
try:
    datetime.fromisoformat(stamp)
except ValueError:
    legacy_rejected = True
else:
    raise AssertionError("등록된 Python 3.10 재현 조건이 다르다")
parsed = parse_lease_timestamp(stamp)
assert parsed == datetime(2026, 10, 16, 23, 2, 45, 468649, timezone.utc)
for fraction in ("", ".4", ".46", ".468", ".4686", ".46864", ".468649", ".4686492"):
    assert parse_lease_timestamp("2026-10-16T23:02:45" + fraction + "Z").tzinfo is timezone.utc
metadata = {"owner": Q.LEASE_OWNER, "state": "running", "childPid": 10, "wrapperPid": 20,
    "completedUtc": None, "acquiredUtc": "2026-10-16T23:02:25.0673901Z",
    "heartbeatUtc": stamp, "leaseId": "7656aeb6-e588-4e65-93e8-7aaa18b65674"}
chain = [{"pid": os.getpid(), "parent": 10, "name": "python.exe"},
    {"pid": 10, "parent": 20, "name": "python.exe"},
    {"pid": 20, "parent": 30, "name": "powershell.exe"}]
class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 16, 23, 2, 46, tzinfo=timezone.utc)
for age in (60.000001, -0.000001):
    try:
        Q.validate_lease(metadata, chain, 10, parsed + timedelta(seconds=age))
    except ValueError:
        pass
    else:
        raise AssertionError("stale/future lease 허용")
lab = pathlib.Path.home() / "IsaacLab"
protocol, anchor = Q.verify(lab, committed=False)
scratch = output / "fixture"
spec = Q.specs()[0]
folder = scratch / "runs/ppo_gradient_v3_20261017" / spec["id"]
folder.mkdir(parents=True)
fixed = Q.manifest(spec, protocol, anchor)
arguments = Q.training_args(lab, folder, spec, folder.parent.name)
launch = {"args": arguments, "cwd": str(lab.resolve())}
Q.write_new(folder / "launch_intent.json", launch)
Q.write_new(folder / "launch.json", launch)
Q.write_new(folder.parent / "plan.json", {"lease": metadata})
real_read = Q.read
def fixture_read(path):
    return metadata if path.name == "gpu-lease.json" else real_read(path)
with patch.object(Q, "PRIVATE", scratch), patch.object(Q, "verify", return_value=(protocol, anchor)), \
     patch.object(Q, "read", side_effect=fixture_read), patch.object(Q, "datetime", Clock), \
     patch.object(Q.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(chain))), \
     patch.object(Q, "assert_exclusive_lock") as lock:
    context = Q.lease_context(10)
    assert context["heartbeatUtc"] == stamp and context["callerPid"] == os.getpid()
    assert T.verify_training(fixed, folder, arguments[arguments.index("--") + 1:], lab) == spec
    assert lock.call_count == 2
    assert all(call.args == (pathlib.Path.home() / "CodexSignals/gpu-lease.lock",) for call in lock.call_args_list)
assert "gymnasium" not in sys.modules and "isaacsim" not in sys.modules
Q.write_new(output / "cpu_evidence.json", {"python": sys.version, "legacy_7_digit_rejected": legacy_rejected,
    "parsed_utc": parsed.isoformat(), "lease_context": "passed", "training_preflight": "passed",
    "stale_future": "rejected", "source_check": "passed", "gpu_started": False,
    "fixture_boundaries": ["lease IO and ancestry", "clock", "lock", "committed gate after real source check"]})
print("Python 3.10 lease·학습 진입 CPU 회귀 PASS", flush=True)
'''


@pytest.mark.skipif(os.name != "nt", reason="실제 Windows Job + Isaac Python 3.10 회귀")
def test_actual_owned_isaac310_parses_7_digits_and_validates_training(tmp_path, monkeypatch):
    launcher = Path.home() / "IsaacLab/_isaac_sim/python.bat"
    if not launcher.is_file():
        pytest.skip("Isaac Python 3.10 launcher가 없는 호스트")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    child = tmp_path / "lease cpu child.py"
    child.write_text(CPU_CHILD, encoding="utf-8")
    folder = tmp_path / "owned"
    folder.mkdir()
    execution = Q.execute_owned(folder, [str(launcher), "-B", str(child), str(ROOT), str(folder)], folder, 30)
    Q.OLD.verify_execution(execution)
    assert execution["cleanup_verified"] and execution["natural_exit"] and execution["exit_code"] == 0
    assert Q.read(folder / "launch.json") == Q.read(folder / "launch_intent.json")
    evidence = Q.read(folder / "cpu_evidence.json")
    assert evidence["legacy_7_digit_rejected"] is True
    assert evidence["training_preflight"] == evidence["lease_context"] == "passed"
    assert evidence["gpu_started"] is False
    assert not list(folder.rglob("training_evidence.json"))
    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        Q.execute_owned(folder, [str(launcher), "-B", str(child)], folder, 30)
    assert before == {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
