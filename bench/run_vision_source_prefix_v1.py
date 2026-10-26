"""등록된 source batch의 학습 4건과 평가 2건만 직렬 실행한다.

호출자는 루트 호스트의 단일 controller여야 한다. 이 모듈은 기존 source
protocol과 원장을 재사용하지만 legacy record action에는 진입하지 않는다.
여섯 action이 모두 검증되면 checkpoint 등록에 쓸 handoff를 write-once로
봉인하고 후속 prepared record protocol 등록을 기다린다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import run_vision_brev_batch_v1 as BASE
import vision_brev_render_backend_v5 as BACKEND  # pyright: ignore[reportMissingImports]

HANDOFF_SCHEMA = "vision_source_prefix_handoff_v1"
CONTROLLER_CONTRACT = "root_host_single_serial_controller_v1"
SEEDS = (2210, 2211)
TRAIN_COUNT = 4
EVALUATION_COUNT = 2
PREFIX_COUNT = TRAIN_COUNT + EVALUATION_COUNT
EXPECTED_TRAIN_IDS = {
    f"vision-s{seed}-{arm.lower()}" for seed in SEEDS for arm in ("NONE", "F1")
}


class SourcePrefixBatch(BASE.SourceBatch):
    """기존 batch 실행을 사용하되 record dispatch를 마지막 방어선에서 거부한다."""

    def _execute(
        self, action: Mapping[str, Any], completed: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        if action.get("kind") == "record":
            raise RuntimeError("source-prefix controller는 legacy record를 실행하지 않는다")
        return super()._execute(action, completed)


class SourcePrefixController:
    """한 호스트 controller가 한 번에 source action 하나만 진행하는 경계."""

    def __init__(self, batch: SourcePrefixBatch):
        self.batch = batch
        self._train_rows, self._record_rows = self._validate_registered_prefix()
        self.handoff_path = self.batch.runtime / "source-prefix-handoff.json"

    def _validate_registered_prefix(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        plan = self.batch.protocol["training_plan"]
        actions = self.batch.protocol["source_action_order"]
        plan_by_id = {row["job_id"]: row for row in plan}
        if sorted({row["seed"] for row in plan}) != list(SEEDS):
            raise ValueError("source-prefix seed는 2210·2211로 고정된다")
        prefix = actions[:PREFIX_COUNT]
        if [row["kind"] for row in prefix] != ["train"] * TRAIN_COUNT + [
            "evaluate"
        ] * EVALUATION_COUNT:
            raise ValueError("source-prefix는 학습4→평가2 순서여야 한다")
        trains = prefix[:TRAIN_COUNT]
        train_rows: dict[str, dict[str, Any]] = {}
        observed = set()
        for action in trains:
            row = plan_by_id[action["job_id"]]
            identity = (row["seed"], row["arm"])
            if row["arm"] not in {"NONE", "F1"} or identity in observed:
                raise ValueError("source-prefix 학습 job이 seed별 NONE/F1과 다르다")
            observed.add(identity)
            train_rows[action["job_id"]] = row
        expected = {(seed, arm) for seed in SEEDS for arm in ("NONE", "F1")}
        if observed != expected or len(train_rows) != TRAIN_COUNT:
            raise ValueError("source-prefix 학습 job은 seed별 NONE/F1 네 건이어야 한다")
        if set(train_rows) != EXPECTED_TRAIN_IDS:
            raise ValueError("source-prefix 학습 job ID가 등록된 2210·2211 NONE/F1과 다르다")
        evaluations = prefix[TRAIN_COUNT:]
        if [row["seed"] for row in evaluations] != list(SEEDS):
            raise ValueError("source-prefix 평가는 seed 2210→2211 순서여야 한다")
        for action in evaluations:
            seed = action["seed"]
            expected_ids = {
                train_id
                for train_id, row in train_rows.items()
                if row["seed"] == seed
            }
            if {action["none_job_id"], action["f1_job_id"]} != expected_ids:
                raise ValueError("source-prefix 평가가 같은 seed의 NONE/F1과 다르다")
        records = actions[PREFIX_COUNT:]
        record_rows = {row["source_job_id"]: row for row in records}
        logical_ids = {row["checkpoint_logical_id"] for row in records}
        if (
            len(records) != TRAIN_COUNT
            or set(record_rows) != set(train_rows)
            or len(logical_ids) != TRAIN_COUNT
        ):
            raise ValueError("checkpoint logical ID가 네 source 학습과 일대일이 아니다")
        return train_rows, record_rows

    def _inspect(self) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        completed: list[dict[str, Any]] = []
        actions = self.batch.protocol["source_action_order"]
        current: dict[str, Any] | None = None
        missing_seen = False
        for index, action in enumerate(actions[PREFIX_COUNT:], start=PREFIX_COUNT):
            if self.batch._action_folder(index, action).exists():
                raise ValueError("legacy record action 상태가 source-prefix에 존재한다")
        for index, action in enumerate(actions[:PREFIX_COUNT]):
            folder = self.batch._action_folder(index, action)
            receipt = folder / "receipt.json"
            if folder.is_symlink():
                raise ValueError("source-prefix action folder가 symlink다")
            if receipt.exists() or receipt.is_symlink():
                if receipt.is_symlink() or not receipt.is_file():
                    raise ValueError("source-prefix receipt가 regular file이 아니다")
                if missing_seen:
                    raise ValueError("source-prefix 완료 receipt가 연속되지 않는다")
                completed.append(self.batch._raw(index))
                continue
            if folder.exists():
                if missing_seen:
                    raise ValueError("source-prefix 미완료 action 상태가 연속되지 않는다")
                current = action
                missing_seen = True
                continue
            if current is None:
                current = action
            missing_seen = True
        return completed, current

    def _receipt_rows(self) -> list[dict[str, Any]]:
        rows = []
        for index, action in enumerate(
            self.batch.protocol["source_action_order"][:PREFIX_COUNT]
        ):
            receipt = self.batch._action_folder(index, action) / "receipt.json"
            rows.append(
                {
                    "ordinal": index,
                    "action_id": action["action_id"],
                    "kind": action["kind"],
                    "sha256": BASE.sha256(receipt),
                }
            )
        return rows

    def _handoff(self, completed: Sequence[dict[str, Any]]) -> dict[str, Any]:
        by_job = {row["job_id"]: row for row in completed if row["kind"] == "train"}
        checkpoints = []
        lab = self.batch.lab.resolve()
        for job_id, row in self._train_rows.items():
            artifact = by_job[job_id]["artifact"]
            checkpoint = artifact["checkpoint"]
            checkpoint_path = Path(checkpoint["path"])
            resolved = checkpoint_path.resolve(strict=True)
            if checkpoint_path.is_symlink() or not resolved.is_file():
                raise ValueError("source checkpoint가 regular file이 아니다")
            try:
                relative = resolved.relative_to(lab).as_posix()
            except ValueError as error:
                raise ValueError("source checkpoint가 Isaac Lab 경로를 벗어났다") from error
            if BASE.sha256(resolved) != checkpoint["sha256"]:
                raise ValueError("source checkpoint SHA가 학습 artifact와 다르다")
            checkpoints.append(
                {
                    "source_job_id": job_id,
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "logical_id": self._record_rows[job_id]["checkpoint_logical_id"],
                    "isaaclab_relative": relative,
                    "sha256": checkpoint["sha256"],
                }
            )
        evaluations = [
            {
                "action_id": row["action_id"],
                "seed": row["seed"],
                "source_gate": row["source_gate"],
            }
            for row in completed
            if row["kind"] == "evaluate"
        ]
        body = {
            "schema": HANDOFF_SCHEMA,
            "status": "await_checkpoint_registration",
            "controller_contract": CONTROLLER_CONTRACT,
            "protocol": {
                "protocol_id": self.batch.protocol["protocol_id"],
                "sha256_lf": self.batch.protocol_sha,
                "source_anchor": self.batch.anchor,
            },
            "resource": {
                "proof_sha256": self.batch.resource_sha,
                "resource_id": self.batch.resource["resource_id"],
                "gpu_uuid": self.batch.resource["gpu_uuid"],
                "lease_id": self.batch.resource["lease_id"],
            },
            "prefix": {
                "train_count": TRAIN_COUNT,
                "evaluation_count": EVALUATION_COUNT,
                "completed": PREFIX_COUNT,
                "action_ids": [
                    row["action_id"]
                    for row in self.batch.protocol["source_action_order"][:PREFIX_COUNT]
                ],
            },
            "action_receipts": self._receipt_rows(),
            "checkpoints": checkpoints,
            "evaluations": evaluations,
        }
        if self.handoff_path.exists():
            if self.handoff_path.is_symlink() or not self.handoff_path.is_file():
                raise ValueError("source-prefix handoff가 regular file이 아니다")
            if BASE.read(self.handoff_path) != body:
                raise ValueError("기존 source-prefix handoff가 현재 원문과 다르다")
        else:
            BASE.write_new(self.handoff_path, body)
        return {
            **body,
            "handoff": {
                "path": str(self.handoff_path),
                "sha256": BASE.sha256(self.handoff_path),
            },
        }

    def verify(self) -> dict[str, Any]:
        verified = self.batch.verify()
        completed, current = self._inspect()
        return {
            "status": "verified",
            "controller_contract": CONTROLLER_CONTRACT,
            "base": verified,
            "completed": len(completed),
            "current_action": None if current is None else current["action_id"],
        }

    def advance(self) -> dict[str, Any]:
        self.batch.verify()
        completed, current = self._inspect()
        if len(completed) == PREFIX_COUNT:
            return self._handoff(completed)
        if current is None:
            raise ValueError("source-prefix 현재 action을 찾을 수 없다")
        folder = self.batch._action_folder(len(completed), current)
        if folder.exists():
            return {
                "status": "HOLD",
                "reason": "source_prefix_action_pending_or_failed",
                "controller_contract": CONTROLLER_CONTRACT,
                "completed": len(completed),
                "current_action": current["action_id"],
            }
        result = self.batch.advance()
        if result.get("status") == "HOLD":
            return {
                **result,
                "controller_contract": CONTROLLER_CONTRACT,
                "current_action": current["action_id"],
            }
        completed_after, current_after = self._inspect()
        if len(completed_after) == PREFIX_COUNT:
            return self._handoff(completed_after)
        return {
            **result,
            "controller_contract": CONTROLLER_CONTRACT,
            "current_action": None if current_after is None else current_after["action_id"],
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "advance"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--source-anchor", required=True)
    parser.add_argument("--resource-proof", required=True, type=Path)
    parser.add_argument("--resource-sha256", required=True)
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    backend_factory: Callable[..., Any] = BACKEND.AtomicVisionBrevBackend.from_runtime,
) -> int:
    args = _parser().parse_args(argv)
    batch = SourcePrefixBatch(
        args.repo.resolve(strict=True),
        args.protocol,
        args.protocol_sha256_lf,
        args.source_anchor,
        args.resource_proof,
        args.resource_sha256,
        args.isaaclab,
        args.python,
        backend_factory=backend_factory,
    )
    controller = SourcePrefixController(batch)
    result = controller.verify() if args.command == "verify" else controller.advance()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result["status"] == "HOLD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
