from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

S = importlib.import_module("run_vision_source_prefix_v2")


class FakeBatch:
    def __init__(self, root: Path, seeds: tuple[int, int] = S.SEEDS):
        self.runtime = root / "runtime"
        self.lab = root / "lab"
        self.runtime.mkdir()
        self.lab.mkdir()
        self.protocol_sha = "1" * 64
        self.anchor = "2" * 40
        self.resource_sha = "3" * 64
        self.resource = {"resource_id": "owned", "gpu_uuid": "GPU-one", "lease_id": "lease"}
        plan = []
        actions: list[dict[str, Any]] = []
        for seed in seeds:
            for arm in ("NONE", "F1", "CORRECTED"):
                job_id = f"vision-s{seed}-{arm.lower()}"
                plan.append({
                    "job_id": job_id,
                    "seed": seed,
                    "arm": arm,
                    "run_name": f"vision_s{seed}_{arm.lower()}",
                    "source_job_id": f"vision-s{seed}-f1" if arm == "CORRECTED" else None,
                })
            for arm in ("none", "f1"):
                actions.append({"action_id": f"train-{seed}-{arm}", "kind": "train", "job_id": f"vision-s{seed}-{arm}"})
        for seed in seeds:
            actions.append({
                "action_id": f"eval-{seed}",
                "kind": "evaluate",
                "seed": seed,
                "none_job_id": f"vision-s{seed}-none",
                "f1_job_id": f"vision-s{seed}-f1",
                "run_name": f"vision_eval_s{seed}",
            })
        for seed in seeds:
            for arm in ("none", "f1"):
                actions.append({
                    "action_id": f"record-{seed}-{arm}",
                    "kind": "record",
                    "job_id": f"record-{seed}-{arm}",
                    "source_job_id": f"vision-s{seed}-{arm}",
                    "checkpoint_logical_id": f"source-s{seed}-{arm}",
                })
        self.protocol = {"protocol_id": "fixture", "training_plan": plan, "source_action_order": actions}

    def _action_folder(self, index: int, action: dict[str, Any]) -> Path:
        return self.runtime / "actions" / f"{index:02d}_{action['action_id']}"


def test_controller_accepts_only_the_new_seed_pair(tmp_path: Path) -> None:
    controller = S.SourcePrefixController(FakeBatch(tmp_path))  # type: ignore[arg-type]
    assert set(controller._train_rows) == S.EXPECTED_TRAIN_IDS


def test_controller_rejects_the_consumed_seed_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2220·2221"):
        S.SourcePrefixController(FakeBatch(tmp_path, (2210, 2211)))  # type: ignore[arg-type]


def test_default_main_backend_is_the_worker_v3_backend() -> None:
    assert S.main.__kwdefaults__ == {
        "backend_factory": S.BACKEND.VisionBrevTrainingBackendV2.from_runtime
    }
