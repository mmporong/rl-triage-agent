# pyright: reportMissingImports=false
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evals import recorded_ppo_gradient_probe_v1 as R

RUNS = R.ROOT / "evals/results/ppo_gradient_v5_20261017/runs"


def _consume(tmp_path: Path, name: str) -> tuple[dict, Path, Path]:
    plan = tmp_path / f"{name}-plan.json"
    receipt = tmp_path / f"{name}-receipt.json"
    R.prepare(RUNS / name, request_id=name, out=plan)
    value = R.execute(plan, out=receipt)
    assert R.checked_receipt(plan, receipt) == value
    return value, plan, receipt


def test_actual_development_records_keep_count_and_gradient_independent(tmp_path):
    healthy, _, _ = _consume(tmp_path, "none_s2144_on")
    faulty, _, _ = _consume(tmp_path, "inverse_ratio_s2144_on")

    assert healthy["count_status"] == healthy["outcome"] == "normal"
    assert faulty["count_status"] == "normal"
    assert faulty["outcome"] == "abnormal"
    assert faulty["gradient_primitive"]["first_mismatch_update"] == 0
    assert faulty["scope"] == R.SCOPE
    assert faulty["evidence_kind"] == "development_fixture"


def _copy_run(tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(RUNS / name, target)
    return target


def _rewrite_close(folder: Path) -> None:
    path = folder / "close_ready.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["records_sha256_lf"] = R._lf_sha(folder / "records.jsonl")
    value["evidence_sha256_lf"] = R._lf_sha(folder / "training_evidence.json")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_incomplete_records_are_unknown_not_normal(tmp_path):
    folder = _copy_run(tmp_path, "none_s2144_on")
    lines = (folder / "records.jsonl").read_text(encoding="utf-8").splitlines()
    (folder / "records.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
    _rewrite_close(folder)
    plan = tmp_path / "plan.json"
    receipt = tmp_path / "receipt.json"
    R.prepare(folder, request_id="incomplete", out=plan)
    result = R.execute(plan, out=receipt)
    assert result["count_status"] == result["outcome"] == "unknown"


def test_raw_tamper_after_prepare_is_rejected(tmp_path):
    folder = _copy_run(tmp_path, "none_s2144_on")
    plan = tmp_path / "plan.json"
    R.prepare(folder, request_id="tamper", out=plan)
    with (folder / "records.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="close-ready|현재 원문"):
        R.execute(plan, out=tmp_path / "receipt.json")


def test_cross_run_close_ready_and_missing_raw_are_rejected(tmp_path):
    folder = _copy_run(tmp_path, "none_s2144_on")
    shutil.copy2(RUNS / "inverse_ratio_s2144_on/close_ready.json",
                 folder / "close_ready.json")
    with pytest.raises(ValueError, match="close-ready"):
        R.prepare(folder, request_id="cross", out=tmp_path / "cross.json")

    (folder / "close_ready.json").unlink()
    with pytest.raises(ValueError, match="모두 필요"):
        R.prepare(folder, request_id="missing", out=tmp_path / "missing.json")


def test_nonfinite_record_is_rejected(tmp_path):
    folder = _copy_run(tmp_path, "none_s2144_on")
    lines = (folder / "records.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["observed"]["optimizer_steps"] = float("nan")
    lines[0] = json.dumps(row, allow_nan=True)
    (folder / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rewrite_close(folder)
    with pytest.raises(ValueError, match="비유한"):
        R.prepare(folder, request_id="nonfinite", out=tmp_path / "plan.json")


def test_receipt_tamper_is_rejected(tmp_path):
    _, plan, receipt = _consume(tmp_path, "none_s2144_on")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["outcome"] = "abnormal"
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt"):
        R.checked_receipt(plan, receipt)
