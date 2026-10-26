"""새 seed 2220·2221 source prefix를 worker-v3 backend로 실행한다."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import run_vision_source_prefix_v1 as V1
import vision_brev_training_backend_v2 as BACKEND  # pyright: ignore[reportMissingImports]

SEEDS = (2220, 2221)
EXPECTED_TRAIN_IDS = {
    f"vision-s{seed}-{arm}" for seed in SEEDS for arm in ("none", "f1")
}
SourcePrefixBatch = V1.SourcePrefixBatch


class SourcePrefixController(V1.SourcePrefixController):
    """v1 상태 전이를 보존하고 새 seed pair만 승인한다."""

    def _validate_registered_prefix(
        self,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        plan = self.batch.protocol["training_plan"]
        actions = self.batch.protocol["source_action_order"]
        plan_by_id = {row["job_id"]: row for row in plan}
        if sorted({row["seed"] for row in plan}) != list(SEEDS):
            raise ValueError("source-prefix seed는 2220·2221로 고정된다")
        prefix = actions[: V1.PREFIX_COUNT]
        if [row["kind"] for row in prefix] != ["train"] * V1.TRAIN_COUNT + [
            "evaluate"
        ] * V1.EVALUATION_COUNT:
            raise ValueError("source-prefix는 학습4→평가2 순서여야 한다")
        train_rows: dict[str, dict[str, Any]] = {}
        observed: set[tuple[int, str]] = set()
        for action in prefix[: V1.TRAIN_COUNT]:
            row = plan_by_id[action["job_id"]]
            identity = (row["seed"], row["arm"])
            if row["arm"] not in {"NONE", "F1"} or identity in observed:
                raise ValueError("source-prefix 학습 job이 seed별 NONE/F1과 다르다")
            observed.add(identity)
            train_rows[action["job_id"]] = row
        expected = {(seed, arm) for seed in SEEDS for arm in ("NONE", "F1")}
        if observed != expected or set(train_rows) != EXPECTED_TRAIN_IDS:
            raise ValueError("source-prefix 학습 job은 새 seed별 NONE/F1 네 건이어야 한다")
        evaluations = prefix[V1.TRAIN_COUNT :]
        if [row["seed"] for row in evaluations] != list(SEEDS):
            raise ValueError("source-prefix 평가는 seed 2220→2221 순서여야 한다")
        for action in evaluations:
            expected_ids = {
                job_id
                for job_id, row in train_rows.items()
                if row["seed"] == action["seed"]
            }
            if {action["none_job_id"], action["f1_job_id"]} != expected_ids:
                raise ValueError("source-prefix 평가가 같은 seed의 NONE/F1과 다르다")
        records = actions[V1.PREFIX_COUNT :]
        record_rows = {row["source_job_id"]: row for row in records}
        logical_ids = {row["checkpoint_logical_id"] for row in records}
        if (
            len(records) != V1.TRAIN_COUNT
            or set(record_rows) != set(train_rows)
            or len(logical_ids) != V1.TRAIN_COUNT
        ):
            raise ValueError("checkpoint logical ID가 네 source 학습과 일대일이 아니다")
        return train_rows, record_rows


def main(
    argv: list[str] | None = None,
    *,
    backend_factory: Callable[..., Any] = BACKEND.VisionBrevTrainingBackendV2.from_runtime,
) -> int:
    args = V1._parser().parse_args(argv)
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
