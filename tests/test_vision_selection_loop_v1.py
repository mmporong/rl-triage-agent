"""V03 영상 선택 bridge와 기존 host request 경계 회귀."""
# pyright: reportMissingImports=false
from __future__ import annotations

import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "evals"), str(ROOT / "src")]
import selection_comparison_controller_v1 as C
import test_vision_selection_contract_v1 as FIXTURE
import vision_selection_contract_v1 as S
import vision_selection_loop_v1 as L


def _content(bundle: dict, fact_id: str | None = None) -> str:
    return FIXTURE.valid_content(bundle, fact_id)


def test_all_views_project_to_the_same_host_request_without_approval(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    specs = []
    for kind in ("text_only", "video_only", "text_and_video"):
        bundle = S.prepare(snapshot, contract, binding, kind)
        fact = "vision:frame:f01" if kind != "text_only" else None
        path = tmp_path / f"{kind}.bridge.json"
        sealed = L.seal_bridge(path, snapshot, contract, binding, kind, _content(bundle, fact))
        assert sealed["status"] == "choice_ready"
        result = L.host_request_spec(sealed, snapshot, contract, binding)
        assert result["status"] == "request_ready"
        assert result["approval_granted"] is False
        specs.append(result["spec"])
    assert specs[0] == specs[1] == specs[2]
    assert specs[0]["approval_granted"] is False
    assert specs[0]["args"] == C.RC.registry()["probes"][specs[0]["probe"]]["args"]


def test_invalid_model_output_holds_without_rule_fallback_or_host_request(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    path = tmp_path / "invalid.bridge.json"
    with patch.object(C, "choose", side_effect=AssertionError("규칙 대체")), \
            patch.object(C, "request_spec", side_effect=AssertionError("host request")):
        sealed = L.seal_bridge(
            path, snapshot, contract, binding, "video_only",
            json.dumps({"action": "continue", "probe": "P_fake", "fact_ids": ["vision:frame:f01"]}),
        )
        assert sealed["status"] == "HOLD"
        result = L.host_request_spec(sealed, snapshot, contract, binding)
    assert result == {"status": "HOLD", "reason": "model_invalid", "spec": None,
                      "approval_granted": False, "rule_fallback_used": False}
    record = L.verify_bridge(sealed, snapshot, contract, binding)
    assert record["rule_fallback_used"] is False and record["projected_choice"]["action"] == "hold"


def test_valid_model_hold_preserves_evidence_without_host_request(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    for kind in ("text_only", "video_only", "text_and_video"):
        bundle = S.prepare(snapshot, contract, binding, kind)
        fact = next(iter(bundle["selection_view"]["evidence_facts"]))
        content = json.dumps({"action": "hold", "probe": None, "fact_ids": [fact]})
        path = tmp_path / f"{kind}.hold.json"
        with patch.object(C, "choose", side_effect=AssertionError("규칙 대체")), \
                patch.object(C, "request_spec", side_effect=AssertionError("host request")):
            sealed = L.seal_bridge(path, snapshot, contract, binding, kind, content)
            assert sealed["status"] == "HOLD"
            result = L.host_request_spec(sealed, snapshot, contract, binding)
        assert result == {"status": "HOLD", "reason": "model_hold", "spec": None,
                          "approval_granted": False, "rule_fallback_used": False}
        record = L.verify_bridge(sealed, snapshot, contract, binding)
        assert record["validation_code"] is None
        assert record["projected_choice"]["model_status"] == "valid"
        assert record["projected_choice"]["selected_by"] == "model"
        assert record["projected_choice"]["fact_ids"] == [fact]
    assert not (tmp_path / "case/probes.ledger.jsonl").exists()


def test_bridge_seals_visual_base_messages_choice_and_v01_sha(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    bundle = S.prepare(snapshot, contract, binding, "video_only")
    path = tmp_path / "bridge.json"
    sealed = L.seal_bridge(path, snapshot, contract, binding, "video_only",
                           _content(bundle, "vision:frame:f01"))
    original = json.loads(path.read_text(encoding="utf-8"))
    for field in ("visual_view_digest", "operational_base_digest", "messages_digest",
                  "bundle_digest", "v01_contract_sha256", "projected_choice"):
        changed = deepcopy(original)
        changed[field] = "0" * 64 if field != "projected_choice" else {**changed[field], "probe": "P_fake"}
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="bridge"):
            L.verify_bridge(sealed, snapshot, contract, binding)
        path.write_text(json.dumps(original), encoding="utf-8")


def test_context_or_prior_receipt_transition_makes_bridge_stale(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    bundle = S.prepare(snapshot, contract, binding, "video_only")
    path = tmp_path / "bridge.json"
    sealed = L.seal_bridge(path, snapshot, contract, binding, "video_only",
                           _content(bundle, "vision:frame:f01"))
    checkpoint_sha = snapshot["inputs"]["packet"]["input_bindings"]["case"]["checkpoint_sha256"]
    spec = C.request_spec(snapshot, L.verify_bridge(
        sealed, snapshot, contract, binding,
    )["projected_choice"])
    ledger = C.S.M.RegistryLedger(tmp_path / "case/probes.ledger.jsonl", C.RC.registry())
    ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    with pytest.raises(ValueError, match="bridge|selection|pending|오래된 snapshot"):
        L.verify_bridge(sealed, snapshot, contract, binding)
    other = FIXTURE.selection_snapshot(tmp_path, checkpoint_sha, arm="other")
    with pytest.raises(ValueError, match="bridge|selection"):
        L.verify_bridge(sealed, other, contract, binding)


def test_bridge_rejects_numeric_zero_substituted_for_boolean(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    bundle = S.prepare(snapshot, contract, binding, "video_only")
    path = tmp_path / "typed.bridge.json"
    sealed = L.seal_bridge(path, snapshot, contract, binding, "video_only",
                           _content(bundle, "vision:frame:f01"))
    original = json.loads(path.read_text(encoding="utf-8"))
    for field in ("approval_granted", "rule_fallback_used"):
        changed = deepcopy(original)
        changed[field] = 0
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="bridge"):
            L.verify_bridge(sealed, snapshot, contract, binding)


def test_original_reference_rejects_self_consistent_rewrite_and_path_only(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    bundle = S.prepare(snapshot, contract, binding, "video_only")
    eligible = bundle["selection_view"]["eligible_probes"]
    assert len(eligible) >= 2
    original_path = tmp_path / "original.bridge.json"
    original = L.seal_bridge(
        original_path, snapshot, contract, binding, "video_only",
        json.dumps({"action": "continue", "probe": eligible[0], "fact_ids": ["vision:frame:f01"]}),
    )
    alternate_path = tmp_path / "alternate.bridge.json"
    L.seal_bridge(
        alternate_path, snapshot, contract, binding, "video_only",
        json.dumps({"action": "continue", "probe": eligible[1], "fact_ids": ["vision:frame:f01"]}),
    )
    shutil.copyfile(alternate_path, original_path)
    with pytest.raises(ValueError, match="SHA"):
        L.verify_bridge(original, snapshot, contract, binding)
    with pytest.raises(TypeError, match="reference"):
        L.verify_bridge(original_path, snapshot, contract, binding)  # type: ignore[arg-type]

    whitespace_path = tmp_path / "whitespace.bridge.json"
    whitespace_ref = L.seal_bridge(
        whitespace_path, snapshot, contract, binding, "video_only", _content(bundle, "vision:frame:f01"),
    )
    replacement_path = tmp_path / "whitespace-replacement.bridge.json"
    L.seal_bridge(
        replacement_path, snapshot, contract, binding, "video_only",
        _content(bundle, "vision:frame:f01") + " ",
    )
    shutil.copyfile(replacement_path, whitespace_path)
    with pytest.raises(ValueError, match="SHA"):
        L.verify_bridge(whitespace_ref, snapshot, contract, binding)


def test_strict_reader_rejects_duplicate_key_nan_and_hardlink(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    for name, payload in (
        ("duplicate", '{"schema":"first","schema":"second"}'),
        ("nan", '{"schema":"vision_selection_bridge_receipt_v1","value":NaN}'),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(payload, encoding="utf-8")
        reference = {"path": str(path.resolve()), "sha256": C.digest_file(path), "status": "choice_ready"}
        with pytest.raises(ValueError, match="중복|constant"):
            L.verify_bridge(reference, snapshot, contract, binding)

    bundle = S.prepare(snapshot, contract, binding, "video_only")
    path = tmp_path / "linked.bridge.json"
    reference = L.seal_bridge(
        path, snapshot, contract, binding, "video_only", _content(bundle, "vision:frame:f01"),
    )
    os.link(path, tmp_path / "second-name.bridge.json")
    with pytest.raises(ValueError, match="regular file"):
        L.verify_bridge(reference, snapshot, contract, binding)


def test_bridge_reference_rejects_leaf_and_parent_symlink_paths(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    bundle = S.prepare(snapshot, contract, binding, "video_only")
    path = tmp_path / "symlink-policy" / "bridge.json"
    reference = L.seal_bridge(
        path, snapshot, contract, binding, "video_only", _content(bundle, "vision:frame:f01"),
    )
    original_is_symlink = Path.is_symlink
    for blocked in (path, path.parent):
        def pretend_symlink(candidate: Path, blocked_path: Path = blocked) -> bool:
            return candidate == blocked_path or original_is_symlink(candidate)

        with patch.object(Path, "is_symlink", pretend_symlink), \
                pytest.raises(ValueError, match="symlink"):
            L.verify_bridge(reference, snapshot, contract, binding)


def test_bridge_only_prepares_request_and_cannot_confirm_or_repair(tmp_path: Path) -> None:
    contract, binding, snapshot = FIXTURE.setup_case(tmp_path)
    bundle = S.prepare(snapshot, contract, binding, "text_and_video")
    path = tmp_path / "bridge.json"
    before = C.L.digest(C.V.checked(snapshot))
    sealed = L.seal_bridge(path, snapshot, contract, binding, "text_and_video",
                           _content(bundle, "vision:frame:f02"))
    result = L.host_request_spec(sealed, snapshot, contract, binding)
    after = C.L.digest(C.V.checked(snapshot))
    assert result["status"] == "request_ready" and result["spec"]["approval_granted"] is False
    assert before == after and C.V.checked(snapshot)["view"]["status"] == "open"
    assert not (tmp_path / "case/probes.ledger.jsonl").exists()
    assert not hasattr(L, "authorize_repair") and not hasattr(L, "seal_probe")
