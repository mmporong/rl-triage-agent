"""D4 실행기에서 공용 owned-process 호출의 누적 벽시계 예산을 fail-closed로 관리한다."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import ppo_ratio_repair_execution_v3 as GPU_BACKEND
import residual_execution_v1 as RESIDUAL
import selection_comparison_gpu_v1 as GPU_VERIFY

OwnedRunner = Callable[[Path, list[str], Path, float], Mapping[str, Any]]
Clock = Callable[[], float]

OWNED_PARENT_WALL_CAP_S = 16_200.0
OVERALL_ELAPSED_CAP_S = 18_000.0
SCHEMA = "selection_comparison_owned_budget_v1"
SEAL_SCHEMA = "selection_comparison_stage_seal_v1"
GPU_CONTRACT = "apps_bounded_v1"
_STAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_AUDIT = re.compile(r"(?P<index>\d{4})_(?P<stage>[A-Za-z0-9_.-]+)\.(?P<kind>reservation|execution|completed|failed)\.json")


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name}은 유한한 숫자여야 한다")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} 범위가 잘못됐다")
    return result


class BudgetedOwned:
    """등록 stage별 owned runner 호출을 누적 예산과 append-only audit로 감싼다."""

    def __init__(
        self,
        audit_root: Path,
        *,
        runner: OwnedRunner | None = None,
        clock: Clock = time.monotonic,
        owned_parent_wall_cap_s: float = OWNED_PARENT_WALL_CAP_S,
        overall_elapsed_cap_s: float = OVERALL_ELAPSED_CAP_S,
    ) -> None:
        self.audit_root = Path(audit_root).resolve()
        self.runner = runner or RESIDUAL.V1.OWNED.run_owned
        self.clock = clock
        self.owned_parent_wall_cap_s = _number(
            owned_parent_wall_cap_s, "owned parent-wall 상한", positive=True
        )
        self.overall_elapsed_cap_s = _number(overall_elapsed_cap_s, "overall elapsed 상한", positive=True)
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self._batch_path = self.audit_root / "batch.json"
        self._batch_failure_path = self.audit_root / "batch_failure.json"
        now = _number(self.clock(), "clock")
        expected = {
            "schema": SCHEMA,
            "owned_parent_wall_cap_s": self.owned_parent_wall_cap_s,
            "overall_elapsed_cap_s": self.overall_elapsed_cap_s,
        }
        if not self._batch_path.exists():
            _write_new(self._batch_path, {**expected, "batch_started_clock_s": now})
        batch = _read(self._batch_path)
        if not isinstance(batch, dict) or any(batch.get(key) != value for key, value in expected.items()):
            raise ValueError("owned budget batch 결합값이 다르다")
        self._batch_started = _number(batch.get("batch_started_clock_s"), "batch 시작 clock")
        if now < self._batch_started:
            raise RuntimeError("batch clock 연속성을 확인하지 못했다")
        self._consumed, self._next_index, self._blocked = self._restore()
        if self._batch_failure_path.is_file():
            failure = _read(self._batch_failure_path)
            if not isinstance(failure, dict) or failure.get("schema") != SCHEMA or not isinstance(failure.get("reason"), str):
                raise ValueError("owned batch failure audit 형식이 다르다")
            self._blocked = self._blocked or "batch_failure:" + failure["reason"]

    def _restore(self) -> tuple[float, int, str | None]:
        groups: dict[tuple[int, str], set[str]] = {}
        for path in self.audit_root.iterdir():
            if path in {self._batch_path, self._batch_failure_path}:
                continue
            if not path.is_file() or path.is_symlink():
                raise RuntimeError("owned audit inventory에 허용되지 않은 항목이 있다")
            match = _AUDIT.fullmatch(path.name)
            if not match:
                raise RuntimeError("owned audit 파일 이름이 계약과 다르다")
            key = int(match["index"]), match["stage"]
            groups.setdefault(key, set()).add(match["kind"])
        consumed = 0.0
        blocked = None
        indices = []
        for (index, stage), kinds in sorted(groups.items()):
            indices.append(index)
            if "reservation" not in kinds or not ({"completed", "failed"} & kinds):
                blocked = blocked or f"pending:{index:04d}:{stage}"
                continue
            if {"completed", "failed"} <= kinds or kinds - {"reservation", "execution", "completed", "failed"}:
                raise RuntimeError("owned audit terminal 상태가 유일하지 않다")
            terminal = "completed" if "completed" in kinds else "failed"
            record = _read(self.audit_root / f"{index:04d}_{stage}.{terminal}.json")
            reservation = self.audit_root / f"{index:04d}_{stage}.reservation.json"
            reserved = _read(reservation)
            if (
                not isinstance(reserved, dict)
                or reserved.get("schema") != SCHEMA
                or reserved.get("index") != index
                or reserved.get("stage") != stage
            ):
                raise ValueError("owned audit reservation 결합값이 다르다")
            if (
                not isinstance(record, dict)
                or record.get("schema") != SCHEMA
                or record.get("index") != index
                or record.get("stage") != stage
            ):
                raise ValueError("owned audit terminal 결합값이 다르다")
            if record.get("reservation_sha256") != _sha(reservation):
                raise ValueError("owned audit reservation SHA가 다르다")
            execution = self.audit_root / f"{index:04d}_{stage}.execution.json"
            if terminal == "completed" and "execution" not in kinds:
                raise ValueError("완료 audit에 execution receipt가 없다")
            if "execution" in kinds and record.get("execution_sha256") != _sha(execution):
                raise ValueError("owned audit execution SHA가 다르다")
            charged = record.get("charged_owned_wall_s")
            if charged is None and terminal == "failed":
                blocked = blocked or f"unquantified:{index:04d}:{stage}"
            else:
                consumed += _number(charged, "charged owned wall")
            if terminal == "failed":
                blocked = blocked or f"failed:{index:04d}:{stage}"
        if indices and sorted(indices) != list(range(1, max(indices) + 1)):
            raise RuntimeError("owned audit sequence에 공백이 있다")
        if consumed > self.owned_parent_wall_cap_s:
            blocked = blocked or "owned_parent_wall_cap_exceeded"
        return consumed, (max(indices) + 1 if indices else 1), blocked

    @property
    def consumed_owned_parent_wall_s(self) -> float:
        return self._consumed

    def remaining(self) -> dict[str, float]:
        now = _number(self.clock(), "clock")
        elapsed = now - self._batch_started
        if elapsed < 0:
            raise RuntimeError("batch clock이 뒤로 이동했다")
        return {
            "owned_parent_wall_s": max(0.0, self.owned_parent_wall_cap_s - self._consumed),
            "overall_elapsed_s": max(0.0, self.overall_elapsed_cap_s - elapsed),
        }

    def stop(self, reason: str) -> None:
        """상위 backend 검증 실패를 영속적인 batch 중단으로 기록한다."""
        if not isinstance(reason, str) or not reason or len(reason) > 160:
            raise ValueError("batch 중단 사유 형식이 잘못됐다")
        if not self._batch_failure_path.exists():
            _write_new(
                self._batch_failure_path,
                {
                    "schema": SCHEMA,
                    "reason": reason,
                    "consumed_owned_parent_wall_s": self._consumed,
                    "remaining": self.remaining(),
                },
            )
        self._blocked = "batch_failure:" + reason

    def for_stage(self, stage: str) -> OwnedRunner:
        """residual_execution_v1의 owned callback 자리에 전달할 네 인자 함수를 만든다."""
        self._stage(stage)

        def owned(folder: Path, args: list[str], cwd: Path, watchdog_s: float) -> Mapping[str, Any]:
            return self.run(stage, folder, args, cwd, watchdog_s)

        return owned

    @staticmethod
    def _stage(stage: str) -> str:
        if not isinstance(stage, str) or not _STAGE.fullmatch(stage):
            raise ValueError("owned stage 이름이 잘못됐다")
        return stage

    def _paths(self, stage: str) -> dict[str, Path]:
        stem = f"{self._next_index:04d}_{stage}"
        return {kind: self.audit_root / f"{stem}.{kind}.json" for kind in (
            "reservation", "execution", "completed", "failed"
        )}

    def run(
        self,
        stage: str,
        folder: Path,
        args: Sequence[str],
        cwd: Path,
        requested_watchdog_s: float,
    ) -> Mapping[str, Any]:
        stage = self._stage(stage)
        if self._blocked:
            raise RuntimeError(f"owned batch가 fail-closed 상태다: {self._blocked}")
        work, working_directory = Path(folder).resolve(), Path(cwd).resolve()
        if not work.is_dir():
            raise FileNotFoundError("_prepared_owned가 만든 실행 folder가 없다")
        if not working_directory.is_dir():
            raise FileNotFoundError("owned 실행 cwd가 없다")
        argv = [str(item) for item in args]
        if not argv or any(not item for item in argv):
            raise ValueError("owned 실행 argv가 비었다")
        requested = _number(requested_watchdog_s, "요청 watchdog", positive=True)
        remaining = self.remaining()
        effective = min(requested, remaining["owned_parent_wall_s"], remaining["overall_elapsed_s"])
        if effective <= 0:
            self._blocked = "budget_exhausted_before_launch"
            raise RuntimeError("owned 실행 전 예산이 소진됐다")
        paths = self._paths(stage)
        reservation = {
            "schema": SCHEMA,
            "index": self._next_index,
            "stage": stage,
            "folder": str(work),
            "cwd": str(working_directory),
            "args": argv,
            "requested_watchdog_s": requested,
            "effective_watchdog_s": effective,
            "consumed_owned_parent_wall_s_before": self._consumed,
            "remaining_before": remaining,
        }
        _write_new(paths["reservation"], reservation)
        reservation_sha = _sha(paths["reservation"])
        started = _number(self.clock(), "실행 시작 clock")
        try:
            raw = self.runner(work, argv, working_directory, effective)
            observed = _number(self.clock(), "실행 종료 clock") - started
            if observed < 0:
                raise RuntimeError("owned runner clock이 뒤로 이동했다")
            execution = dict(raw) if isinstance(raw, Mapping) else {"invalid_return_type": type(raw).__name__}
            _write_new(paths["execution"], execution)
            charged = self._receipt_wall(execution, observed)
            terminal = {
                "schema": SCHEMA,
                "index": self._next_index,
                "stage": stage,
                "reservation_sha256": reservation_sha,
                "execution_sha256": _sha(paths["execution"]),
                "wrapper_observed_wall_s": observed,
                "charged_owned_wall_s": charged,
            }
            if not RESIDUAL.V1._finished(execution):
                reason = self._failure_reason(execution)
                _write_new(paths["failed"], {**terminal, "status": "failed", "reason": reason})
                self._consumed += charged
                self._blocked = f"failed:{self._next_index:04d}:{stage}:{reason}"
                self._next_index += 1
                raise RuntimeError(f"owned 실행이 완결되지 않았다: {reason}")
            elapsed_after = _number(self.clock(), "완료 clock") - self._batch_started
            limit_reason = None
            if self._consumed + charged > self.owned_parent_wall_cap_s:
                limit_reason = "owned_parent_wall_cap_exceeded"
            elif elapsed_after > self.overall_elapsed_cap_s:
                limit_reason = "overall_elapsed_cap_exceeded"
            if limit_reason:
                _write_new(paths["failed"], {**terminal, "status": "failed", "reason": limit_reason})
                self._consumed += charged
                self._blocked = limit_reason
                self._next_index += 1
                raise RuntimeError(f"owned 실행이 설계 예산을 넘었다: {limit_reason}")
            _write_new(paths["completed"], {**terminal, "status": "completed"})
            self._consumed += charged
            self._next_index += 1
            return raw
        except Exception as error:
            if paths["completed"].exists() or paths["failed"].exists():
                raise
            ended = _number(self.clock(), "실패 clock")
            observed = max(0.0, ended - started)
            charged = observed if observed > 0 else None
            failure = {
                "schema": SCHEMA,
                "index": self._next_index,
                "stage": stage,
                "reservation_sha256": reservation_sha,
                "status": "failed",
                "reason": "runner_exception",
                "error_type": type(error).__name__,
                "wrapper_observed_wall_s": observed,
                "charged_owned_wall_s": charged,
            }
            if paths["execution"].exists():
                failure["execution_sha256"] = _sha(paths["execution"])
            _write_new(paths["failed"], failure)
            if charged is not None:
                self._consumed += charged
            self._blocked = f"failed:{self._next_index:04d}:{stage}:runner_exception"
            self._next_index += 1
            raise

    @staticmethod
    def _receipt_wall(receipt: Mapping[str, Any], observed: float) -> float:
        values = []
        for key in ("wall_s", "parent_wall_s", "elapsed_s"):
            if key in receipt:
                values.append(_number(receipt[key], f"execution {key}"))
        if not values:
            raise ValueError("owned execution receipt에 확인 가능한 벽시계 소비가 없다")
        return max([observed, *values])

    @staticmethod
    def _failure_reason(receipt: Mapping[str, Any]) -> str:
        if receipt.get("timed_out") is True:
            return "watchdog_timeout"
        if receipt.get("cleanup_verified") is not True:
            return "cleanup_not_verified"
        return "execution_not_finished"


def _shared_frozen_backend() -> None:
    if GPU_BACKEND.FROZEN is not RESIDUAL.V1 or RESIDUAL.V2.V1 is not RESIDUAL.V1:
        raise RuntimeError("GPU admission과 residual backend가 같은 동결 모듈을 공유하지 않는다")


def _execution_names(kind: str) -> tuple[str, ...]:
    values = {
        "train": ("execution.json", "checkpoint_execution.json", "extract_execution.json"),
        "inspect": ("execution.json", "checkpoint_execution.json"),
        "fixed_evaluate": ("execution.json",),
        "legacy_probe": ("execution.json",),
    }
    try:
        return values[kind]
    except KeyError as error:
        raise ValueError("등록되지 않은 D4 backend stage 종류다") from error


def _build_stage_seal(stage: Path, kind: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """저장된 execution·GPU 원본에서 stage seal을 순수하게 재구성한다."""
    _shared_frozen_backend()
    folder = Path(stage).resolve()
    if not folder.is_dir() or folder.is_symlink() or not isinstance(raw, Mapping):
        raise ValueError("봉인할 backend stage 또는 반환값 형식이 잘못됐다")
    expected_names = _execution_names(kind)
    expected_paths = {folder / name for name in expected_names}
    present_paths = set(folder.glob("*execution.json"))
    if present_paths != expected_paths or any(path.is_symlink() or not path.is_file() for path in expected_paths):
        raise ValueError("stage execution receipt inventory가 다르다")
    execution_hashes = {}
    for name in expected_names:
        path = folder / name
        stored = _read(path)
        key = name.removesuffix(".json")
        if not isinstance(stored, dict) or raw.get(key) != stored or not RESIDUAL.V1._finished(stored):
            raise ValueError(f"{name} 원시 receipt가 self-report와 다르거나 완결되지 않았다")
        execution_hashes[name] = _sha(path)
    gpu_paths = {folder / "gpu_before.json", folder / "gpu_after.json"}
    if set(folder.glob("gpu_*.json")) != gpu_paths or any(
        path.is_symlink() or not path.is_file() for path in gpu_paths
    ):
        raise ValueError("stage GPU receipt inventory가 다르다")
    gpu_receipts = {}
    for path in sorted(gpu_paths):
        value = _read(path)
        if not isinstance(value, dict):
            raise TypeError("GPU receipt 형식이 다르다")
        status = GPU_VERIFY.verify_gpu_receipt(value)
        gpu_receipts[path.name] = {"sha256": _sha(path), "status": status}
    return {
        "schema": SEAL_SCHEMA,
        "kind": kind,
        "stage": str(folder),
        "gpu_contract": GPU_CONTRACT,
        "execution_receipts": execution_hashes,
        "gpu_receipts": gpu_receipts,
    }


def seal_stage(stage: Path, kind: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """backend 호출 직후 stage seal을 새 파일로 봉인한다."""
    seal = _build_stage_seal(stage, kind, raw)
    _write_new(Path(stage).resolve() / "stage_seal.json", seal)
    return seal


def verify_stage_seal(stage: Path, kind: str, raw: Mapping[str, Any], seal: Mapping[str, Any]) -> dict[str, Any]:
    """stage seal과 원시 receipt를 쓰기 없이 재계산해 검증한다."""
    folder = Path(stage).resolve()
    seal_path = folder / "stage_seal.json"
    if seal_path.is_symlink() or not seal_path.is_file() or not isinstance(seal, Mapping):
        raise ValueError("stage seal 파일 또는 반환값 형식이 잘못됐다")
    stored = _read(seal_path)
    expected = _build_stage_seal(folder, kind, raw)
    if stored != dict(seal) or stored != expected:
        raise ValueError("stage seal과 원시 receipt 재계산값이 다르다")
    return expected


class SelectionComparisonBackend:
    """residual backend 호출을 apps-bounded GPU context와 stage seal에 결합한다."""

    def __init__(self, budget: BudgetedOwned, *, gpu_contract: str = GPU_CONTRACT) -> None:
        if not isinstance(budget, BudgetedOwned):
            raise TypeError("BudgetedOwned 인스턴스가 필요하다")
        if gpu_contract != GPU_CONTRACT:
            raise ValueError("등록되지 않은 GPU contract다")
        _shared_frozen_backend()
        self.budget = budget
        self.gpu_contract = gpu_contract

    @staticmethod
    def _stage_id(kind: str, stage: Path) -> str:
        name = Path(stage).name
        value = f"{kind}.{name}"
        if not _STAGE.fullmatch(value):
            raise ValueError("D4 backend stage 이름이 audit 형식과 다르다")
        return value

    def _call(self, kind: str, stage: Path, function: Callable[..., dict[str, Any]], *args, **kwargs) -> dict[str, Any]:
        _shared_frozen_backend()
        owned = self.budget.for_stage(self._stage_id(kind, stage))
        try:
            with GPU_BACKEND.gpu_admission():
                raw = function(stage, *args, owned=owned, **kwargs)
            seal = seal_stage(stage, kind, raw)
            return {**raw, "stage_seal": seal}
        except Exception as error:
            self.budget.stop(f"{kind}:{type(error).__name__}")
            raise

    def train(self, stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
              binding: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("train", stage, RESIDUAL.train, isaaclab, arm, seed, run_name, binding)

    def inspect(self, stage: Path, isaaclab: Path, arm: str, seed: int, run_name: str,
                binding: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("inspect", stage, RESIDUAL.inspect, isaaclab, arm, seed, run_name, binding)

    def fixed_evaluate(self, stage: Path, isaaclab: Path, artifacts: Any, run_name: str,
                       binding: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("fixed_evaluate", stage, RESIDUAL.fixed_evaluate,
                          isaaclab, artifacts, run_name, binding)

    def legacy_probe(self, stage: Path, isaaclab: Path, artifacts: Any, run_name: str,
                     binding: Mapping[str, Any], *, probes=None) -> dict[str, Any]:
        return self._call("legacy_probe", stage, RESIDUAL.legacy_probe,
                          isaaclab, artifacts, run_name, binding, probes=probes)
