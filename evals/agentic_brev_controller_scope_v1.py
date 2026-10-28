"""등록된 POSIX 검사 원문을 기존 controller에 연결하는 단일 소유 context.

기존 controller·판정·승인 원장은 보존한다. 이 모듈은 실행 권한을 발급하지 않으며,
전용 프로세스의 같은 thread 안에서 controller와 조사 runtime을 사용해야 한다.
"""
from __future__ import annotations

import threading
from copy import deepcopy
from pathlib import Path

import agentic_brev_probe_evidence_v1 as E
import selection_comparison_controller_v1 as C

SOURCE_FILES = frozenset({
    "evals/agentic_brev_controller_scope_v1.py",
    "tests/test_agentic_brev_controller_scope_v1.py",
    "evals/agentic_brev_probe_evidence_v1.py",
    "tests/test_agentic_brev_probe_evidence_v1.py",
    "evals/selection_comparison_controller_v1.py",
    "evals/brev_recovery_backend_v4.py",
    "evals/lease_timestamps.py",
    "evals/model_decision_journal_v1.py",
    "evals/vision_remote_gpu_v1.py",
    "evals/optimizer_observability.py",
    "bench/residual_training_v1.py",
    "evals/run_agentic_development_v2.py",
    "evals/run_agentic_development_v1.py",
    "evals/vision_transferred_e2e_v2.py",
    "evals/agentic_investigation_evidence_v1.py",
})
_LOCK = threading.RLock()
_ACTIVE = None
_HOOKS = ("GPU_CONTRACTS", "_probe_measurement", "_validate_evidence_plan", "_execution_plan")
_BASE = {key: getattr(C, key) for key in _HOOKS}
EXPECTED_KEYS = {"probe", "side", "checkpoint_sha256", "seed", "run_name", "stage", "operation_id"}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def validate_source_registration(registration: dict) -> None:
    """기존 readonly object reader로 필수 소스와 공개 등록 blob을 확인한다."""
    from run_agentic_development_v2 import verify_sources

    require(set(registration) == {"protocol", "anchor", "repo"}, "source registration fields")
    ref = registration["protocol"]
    require(set(ref) == {"path", "sha256"}, "source protocol reference")
    path = Path(ref["path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and C.digest_file(path) == ref["sha256"], "source protocol SHA")
    protocol = E.loads(path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == "agentic_investigation_runtime_v1_protocol"
            and protocol.get("controller_extensions") == [E.GPU_CONTRACT]
            and SOURCE_FILES <= protocol.get("sources_lf", {}).keys(), "controller source closure")
    verify_sources(protocol, registration["anchor"], repo=Path(registration["repo"]), protocol_path=path)


class BrevProbeScope:
    """호스트가 봉인한 완료 원문만 등록한다. 생성·등록 자체는 실행 승인이 아니다."""

    def __init__(self, *, allow_fixture: bool = False, source_registration: dict | None = None):
        require(type(allow_fixture) is bool, "fixture mode must be explicit bool")
        require(allow_fixture or source_registration is not None, "actual scope requires source registration")
        if source_registration is not None:
            validate_source_registration(source_registration)
        self.allow_fixture = allow_fixture
        self.source_registration = deepcopy(source_registration)
        self.entries: dict[str, dict] = {}
        self.owner: int | None = None

    def register(self, reference: dict, *, expected: dict, roots: list,
                 request_sha256: str, roots_sha256: str) -> None:
        """expected는 승인된 host intent에서, 두 digest는 외부 완료 봉인에서 받는다."""
        if self.owner is not None:
            self._owner()
        require(set(reference) == {"path", "sha256"} and set(expected) == EXPECTED_KEYS,
                "registered evidence/intent fields")
        path = Path(reference["path"])
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.stat().st_nlink == 1 and C.digest_file(path) == reference["sha256"],
                "registered evidence file SHA")
        value = E.loads(path.read_text(encoding="utf-8"))
        entry = {"reference": deepcopy(reference), "expected": deepcopy(expected),
                 "roots": deepcopy(roots), "request_sha256": request_sha256,
                 "roots_sha256": roots_sha256}
        self._check(value, entry)
        key = E.digest(value)
        require(key not in self.entries, "duplicate registered evidence")
        self.entries[key] = entry

    def _owner(self) -> None:
        require(_ACTIVE is self and self.owner == threading.get_ident(), "controller scope owner thread")

    def _check(self, value: dict, entry: dict) -> dict:
        require(value.get("schema") == E.SCHEMA, "POSIX evidence schema")
        checked = E.verify(value, entry["roots"], expected_request_sha256=entry["request_sha256"],
            expected_roots_sha256=entry["roots_sha256"], allow_fixture=self.allow_fixture)
        require(checked["evidence_kind"] != E.ACTUAL or self.source_registration is not None,
                "actual completion requires source registration")
        observed = {key: checked[key] for key in EXPECTED_KEYS - {"operation_id"}}
        observed["operation_id"] = checked["request"]["operation_id"]
        require(observed == entry["expected"], "completed evidence differs from approved host intent")
        return checked

    def _verified(self, value: dict) -> tuple[dict, dict]:
        self._owner()
        key = E.digest(value)
        require(key in self.entries, "unregistered POSIX completion")
        entry = self.entries[key]
        path = Path(entry["reference"]["path"])
        require(C.digest_file(path) == entry["reference"]["sha256"]
                and E.loads(path.read_text(encoding="utf-8")) == value, "completion record changed")
        return self._check(value, entry), entry

    def _measurement(self, value: dict, evidence_kind: str) -> dict:
        if value.get("schema") != E.SCHEMA:
            return _BASE["_probe_measurement"](value, evidence_kind)
        checked, _ = self._verified(value)
        expected = C.ACTUAL_EVIDENCE if checked["evidence_kind"] == E.ACTUAL else C.FIXTURE_EVIDENCE
        require(evidence_kind == expected, "CPU fixture and actual evidence kinds cannot cross")
        return deepcopy(checked["measurement"])

    def _plan(self, value: dict, plan: dict) -> None:
        if value.get("schema") != E.SCHEMA:
            return _BASE["_validate_evidence_plan"](value, plan)
        checked, entry = self._verified(value)
        resolver = E.Resolver(entry["roots"], entry["roots_sha256"])
        # 원래 POSIX stage를 고치지 않는다. 등록 root-map의 job 파일 부모와 계획을 대조한다.
        local_stage = resolver.file(checked["request"]["files"]["job"]).parent
        require(Path(plan["stage"]).resolve() == local_stage.resolve()
                and plan["gpu_contract"] == E.GPU_CONTRACT
                and all(plan[key] == checked[key] for key in (
                    "checkpoint_sha256", "seed", "run_name")), "POSIX execution plan binding")

    def _execution_plan(self, value: dict, spec: dict) -> dict:
        if (value.get("gpu_contract") == E.GPU_CONTRACT
                or value.get("reference", {}).get("gpu_contract") == E.GPU_CONTRACT):
            self._owner()  # 승인·소비보다 앞서 새 계약의 thread 소유를 확인한다.
            if self.source_registration is None:
                raise ValueError("probe authorization requires source registration")
            validate_source_registration(self.source_registration)
        return _BASE["_execution_plan"](value, spec)

    def __enter__(self) -> BrevProbeScope:  # noqa: PYI034 - Python 3.10, Self는 3.11부터 제공
        global _ACTIVE
        _LOCK.acquire()
        try:
            require(_ACTIVE is None and self.owner is None, "nested controller scope")
            require(all(getattr(C, key) is value for key, value in _BASE.items()),
                    "controller extension baseline changed")
            if self.source_registration is not None:
                validate_source_registration(self.source_registration)
            self.owner = threading.get_ident()
            _ACTIVE = self
            C.GPU_CONTRACTS = {*_BASE["GPU_CONTRACTS"], E.GPU_CONTRACT}
            C._probe_measurement = self._measurement
            C._validate_evidence_plan = self._plan
            C._execution_plan = self._execution_plan
            return self
        except BaseException:
            _LOCK.release()
            raise

    def __exit__(self, _kind, _value, _traceback) -> None:
        global _ACTIVE
        self._owner()
        try:
            for key, value in _BASE.items():
                setattr(C, key, value)
        finally:
            _ACTIVE, self.owner = None, None
            _LOCK.release()
