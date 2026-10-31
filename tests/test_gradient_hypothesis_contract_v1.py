# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import gradient_hypothesis_contract_v1 as CONTRACT
import recorded_ppo_gradient_probe_v1 as RECORDED

RAW = ROOT / "evals/results/ppo_gradient_v5_20261017/runs"


@pytest.fixture()
def receipts(tmp_path: Path) -> dict[str, tuple[Path, Path]]:
    result = {}
    for name, source in (
        ("case", "inverse_ratio_s2144_on"),
        ("reference", "none_s2144_on"),
        ("normal_case", "none_s2144_on"),
    ):
        folder = tmp_path / name
        plan, receipt = folder / "plan.json", folder / "receipt.json"
        RECORDED.prepare(RAW / source, request_id=f"gradient-contract-{name}", out=plan)
        RECORDED.execute(plan, out=receipt)
        result[name] = (plan, receipt)
    return result


def binding(receipts: dict[str, tuple[Path, Path]], case: str = "case") -> dict:
    return {
        "schema": CONTRACT.PAIR_SCHEMA,
        "subtype_schema": CONTRACT.SUBTYPE_SCHEMA,
        "case": {"role": "case", "plan": receipts[case][0], "receipt": receipts[case][1]},
        "reference": {
            "role": "reference",
            "plan": receipts["reference"][0],
            "receipt": receipts["reference"][1],
        },
    }


def paths(receipts: dict[str, tuple[Path, Path]], case: str = "case") -> tuple[Path, ...]:
    return (*receipts[case], *receipts["reference"])


def test_count_normal_preserves_surrogate_then_gradient_abnormal_confirms(receipts: dict) -> None:
    state = CONTRACT.initialize(binding(receipts), 2)
    after_count = CONTRACT.advance_count(state, *paths(receipts))
    assert "optimizer_update_count" not in after_count["hypotheses"]
    assert "optimizer_surrogate" in after_count["hypotheses"]
    assert after_count["status"] == "open"
    final = CONTRACT.advance_gradient(after_count, *paths(receipts))
    assert final["status"] == "confirmed"
    assert final["conclusion"] == "optimizer_surrogate"
    assert final["hypotheses"] == ["optimizer_surrogate"]
    assert final["evidence_kind"] == "development_fixture"


def test_normal_gradient_removes_only_surrogate(receipts: dict) -> None:
    state = CONTRACT.initialize(binding(receipts, "normal_case"), 2)
    result = CONTRACT.advance_gradient(state, *paths(receipts, "normal_case"))
    assert result["status"] == "open"
    assert "optimizer_surrogate" not in result["hypotheses"]
    assert "optimizer_update_count" in result["hypotheses"]
    assert len(result["hypotheses"]) == 7


def test_legacy_seven_subtype_binding_is_rejected(receipts: dict) -> None:
    value = binding(receipts)
    value["subtype_schema"] = "r1_7_subtype_v1"
    with pytest.raises(ValueError, match="암묵적 변환"):
        CONTRACT.initialize(value, 2)


def test_tampered_original_plan_is_rejected(receipts: dict, tmp_path: Path) -> None:
    state = CONTRACT.initialize(binding(receipts), 2)
    plan = receipts["case"][0]
    value = json.loads(plan.read_text(encoding="utf-8"))
    value["request_id"] = "tampered"
    plan.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA 불일치"):
        CONTRACT.advance_count(state, *paths(receipts))


def test_unknown_case_holds_without_elimination(receipts: dict, tmp_path: Path) -> None:
    source = tmp_path / "unknown-source"
    shutil.copytree(RAW / "none_s2144_on", source)
    records = source / "records.jsonl"
    lines = records.read_text(encoding="utf-8").splitlines()
    records.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    close_path = source / "close_ready.json"
    close = json.loads(close_path.read_text(encoding="utf-8"))
    close["records_sha256_lf"] = hashlib.sha256(records.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    close_path.write_text(json.dumps(close), encoding="utf-8")
    plan, receipt = tmp_path / "unknown-plan.json", tmp_path / "unknown-receipt.json"
    RECORDED.prepare(source, request_id="incomplete", out=plan)
    RECORDED.execute(plan, out=receipt)
    pair = binding(receipts)
    pair["case"] = {"role": "case", "plan": plan, "receipt": receipt}
    state = CONTRACT.initialize(pair, 2)
    result = CONTRACT.advance_gradient(state, plan, receipt, *receipts["reference"])
    assert result["status"] == "hold"
    assert result["observed"]["P_ppo_gradient"] == "unknown"
    assert "optimizer_surrogate" in result["hypotheses"]


def test_non_normal_reference_becomes_unknown_hold(receipts: dict) -> None:
    faulty_reference = {
        "schema": CONTRACT.PAIR_SCHEMA,
        "subtype_schema": CONTRACT.SUBTYPE_SCHEMA,
        "case": {"role": "case", "plan": receipts["normal_case"][0], "receipt": receipts["normal_case"][1]},
        "reference": {"role": "reference", "plan": receipts["case"][0], "receipt": receipts["case"][1]},
    }
    state = CONTRACT.initialize(faulty_reference, 2)
    supplied = (*receipts["normal_case"], *receipts["case"])
    result = CONTRACT.advance_gradient(state, *supplied)
    assert result["status"] == "hold"
    assert result["observed"]["P_ppo_gradient"] == "unknown"
    assert "optimizer_surrogate" in result["hypotheses"]


def test_role_swap_and_public_information_leak_are_blocked(receipts: dict) -> None:
    state = CONTRACT.initialize(binding(receipts), 2)
    swapped = (*receipts["reference"], *receipts["case"])
    with pytest.raises(ValueError, match="role/path 불일치"):
        CONTRACT.advance_gradient(state, *swapped)

    context = CONTRACT.public_context(state)
    encoded = json.dumps(context, ensure_ascii=False, sort_keys=True)
    for forbidden in ("inverse_ratio_s2144_on", "none_s2144_on", "run_id", "seed", "source_binding", "path"):
        assert forbidden not in encoded
    assert context["public_categories"] == [
        "reward", "actuator", "exploration", "optimizer", "physics", "termination"
    ]
    assert context["function_permissions"] == {
        "approval": False, "execution": False, "correction": False, "recovery": False
    }
