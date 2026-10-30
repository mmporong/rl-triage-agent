"""Nemotron 3.5 영상 가치 사례 생성기의 결합·거부·보류 회귀. 모델·GPU·네트워크를 쓰지 않는다."""
from __future__ import annotations

import json
import shutil
import socket
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import nemotron35_value_cases_v1 as G

F2A, F2B, F1A, F1B = "go2-flat-f2-s2033", "go2-flat-f2-s2046", "go2-flat-f1-s2034", "go2-flat-f1-s2080"


@pytest.fixture(scope="module")
def protocol() -> dict:
    return G.load_protocol()


@pytest.fixture(scope="module")
def manifest(protocol: dict) -> dict:
    return G.build(protocol)[0]


def _entry(protocol: dict, lineage_id: str) -> dict:
    return next(row for row in protocol["lineages"] if row["lineage_id"] == lineage_id)


def _swap(protocol: dict, lineage_id: str, update) -> dict:
    value = deepcopy(protocol)
    update(_entry(value, lineage_id), value)
    return value


def _rows(manifest: dict) -> dict:
    return {row["lineage_id"]: row for row in manifest["lineages"]}


def _choice(row: dict, arm: str, probe: str | None = None, status: str = "valid",
            facts: list[str] | None = None) -> dict:
    value = {"status": status, "capsule_sha256": row["arms"][arm]["capsule_sha256"]}
    if status == "valid":
        value.update(action="continue" if probe else "hold", probe=probe,
                     fact_ids=facts or ["telemetry:case:Train/mean_reward"])
    return value


def _choices(manifest: dict, table: dict) -> dict:
    return {"schema": G.CHOICES_SCHEMA, "manifest_digest": manifest["manifest_digest"],
            "measurement_kind": "synthetic_cpu", "choices": table}


def test_prepare_check_keeps_model_arms_pending_and_frames_blocked(tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    prepared = G.prepare(G.PROTOCOL, out)
    assert G.check(G.PROTOCOL, out / "manifest.json") == prepared
    assert prepared["outcome"] == "inconclusive" and prepared["external_calls"] == 0
    assert prepared["comparison"]["B_vs_C_image_contribution"] == "inconclusive_frames_blocked"
    assert not any(prepared["claims"].values())
    rows = _rows(prepared)
    assert set(rows) == {F2A, F2B, F1A, F1B}
    for row in rows.values():
        arms = row["arms"]
        assert arms["A"]["status"] == arms["B"]["status"] == "pending_api"
        assert arms["A"]["capsule_sha256"] == arms["B"]["capsule_sha256"] == row["capsule"]["sha256"]
        assert arms["C"]["status"] == "blocked_new_collection_required" and arms["C"]["capsule_sha256"] is None
        assert row["historical_text_decision"]["counted_as_arm_A"] is False
        assert len(row["fact_ids"]) == 49
    assert len({row["numbers_digest"] for row in rows.values()}) == 4
    assert {probe: item["outcome"] for probe, item in rows[F2A]["receipts"].items()} == {
        "P_reward": "normal", "P_noise": "normal", "P_torque": "abnormal", "P_value": "normal"}
    assert rows[F1A]["receipts"]["P_reward"]["outcome"] == "abnormal"
    assert rows[F2A]["historical_text_decision"]["probe"] == "P_reward"
    rules = rows[F2A]["arms"]["D"]["rules"]
    assert rules["registry_next"]["first_probe"] == "P_value"
    assert [step["probe"] for step in rules["registry_next"]["steps"]] == ["P_value", "P_physics"]
    assert rules["registry_next"]["status"] == rules["fixed_eligible"]["status"] == "pending_receipt"
    assert rules["fixed_eligible"]["first_probe"] == "P_episode"
    messages = json.loads((out / rows[F2A]["capsule"]["file"]).read_text(encoding="utf-8"))
    assert messages[0]["role"] == "system" and "vision:frame" not in messages[1]["content"]
    (out / rows[F2A]["capsule"]["file"]).write_bytes(b"[]")
    with pytest.raises(ValueError, match="capsule"):
        G.check(G.PROTOCOL, out / "manifest.json")


def test_label_and_lineage_swaps_are_rejected(protocol: dict) -> None:
    label = _swap(protocol, F2A, lambda e, p: e.update(demo_result=_entry(p, F2B)["demo_result"]))
    with pytest.raises(ValueError, match="판정 파일"):
        G.build(label)
    mechanism = _swap(protocol, F1A, lambda e, p: e.update(registered_mechanism="actuator", confirming_probe="P_torque"))
    with pytest.raises(ValueError, match="등록 원인"):
        G.build(mechanism)
    specificity = _swap(protocol, F2A, lambda e, p: e.update(specificity_artifact=_entry(p, F2B)["specificity_artifact"]))
    with pytest.raises(ValueError, match="checkpoint"):
        G.build(specificity)
    duplicate = deepcopy(protocol)
    duplicate["lineages"][1] = {**deepcopy(duplicate["lineages"][0]), "lineage_id": "copy-of-f2-s2033"}
    with pytest.raises(ValueError, match="같은 checkpoint"):
        G.build(duplicate)


def test_telemetry_swap_and_tampered_manifest_are_rejected(tmp_path: Path, protocol: dict) -> None:
    swapped = _swap(protocol, F2A, lambda e, p: e["telemetry"].update(case=_entry(p, F2B)["telemetry"]["case"]))
    with pytest.raises(ValueError, match="학습 기록 실행 이름"):
        G.build(swapped)
    out = tmp_path / "prepared"
    G.prepare(G.PROTOCOL, out)
    value = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    value["lineages"][0]["numbers_digest"] = value["lineages"][1]["numbers_digest"]
    (out / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        G.check(G.PROTOCOL, out / "manifest.json")


def test_frame_swap_is_rejected_and_stale_or_unverified_render_is_not_accepted(protocol: dict, manifest: dict) -> None:
    rows = _rows(manifest)
    decisions = {item["media_id"]: (item["accepted"], item["reason"]) for item in rows[F2A]["media"] + rows[F2B]["media"]}
    assert decisions.pop("v11-normal-s2033") == (True, "render_sync_verified")
    assert set(decisions.values()) == {(False, "render_sync_unverified_model_use_deferred")}
    assert rows[F2A]["arms"]["C"]["reasons"] == ["render_verified_case_frames_missing",
                                                 "v01_same_play_frame_contract_missing"]
    assert "render_verified_reference_frames_missing" in rows[F2B]["arms"]["C"]["reasons"]

    def other_lineage_video(entry: dict, value: dict) -> None:
        entry["media"][0]["video"] = _entry(value, F2B)["media"][0]["video"]

    with pytest.raises(ValueError, match="checkpoint"):
        G.build(_swap(protocol, F2A, other_lineage_video))

    def borrowed_receipt(entry: dict, _value: dict) -> None:
        media = {item["media_id"]: item for item in entry["media"]}
        media["v9-motor-s2033-none"]["render_evidence"] = deepcopy(media["v11-normal-s2033"]["render_evidence"])

    with pytest.raises(ValueError, match="다른 영상"):
        G.build(_swap(protocol, F2A, borrowed_receipt))
    item = next(row for row in _entry(protocol, F2A)["media"] if row["media_id"] == "v11-normal-s2033")
    accepted = next(row for row in rows[F2A]["media"] if row["media_id"] == "v11-normal-s2033")
    stale = G.media_decision(protocol, G.ROOT, item, rows[F2A]["checkpoints"], {accepted["video_sha256"]})
    assert (stale["accepted"], stale["reason"]) == (False, "stale_render_observed")


def test_unbound_or_tampered_inputs_are_rejected(tmp_path: Path, protocol: dict) -> None:
    for rel in protocol["bindings_sha256"]:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    assert G.build(protocol, tmp_path)[0]["manifest_digest"] == G.build(protocol)[0]["manifest_digest"]
    rel = _entry(protocol, F2A)["telemetry"]["case"]
    (tmp_path / rel).write_bytes((tmp_path / rel).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA"):
        G.build(protocol, tmp_path)
    unbound = deepcopy(protocol)
    del unbound["bindings_sha256"][_entry(protocol, F1A)["telemetry"]["reference"]]
    with pytest.raises(ValueError, match="결합되지 않은"):
        G.build(unbound)
    executed = deepcopy(protocol)
    executed["status"] = "registered"
    with pytest.raises(ValueError, match="DRAFT"):
        G.build(executed)


def test_score_counts_only_receipt_backed_confirmations(protocol: dict, manifest: dict) -> None:
    rows = _rows(manifest)
    table = {
        F2A: {"A": _choice(rows[F2A], "A", "P_reward"), "B": _choice(rows[F2A], "B", "P_torque")},
        F2B: {"A": _choice(rows[F2B], "A", "P_reward"), "B": _choice(rows[F2B], "B", "P_torque")},
        F1A: {"A": _choice(rows[F1A], "A", "P_reward"), "B": _choice(rows[F1A], "B", "P_reward")},
        F1B: {"A": _choice(rows[F1B], "A", "P_reward"), "B": _choice(rows[F1B], "B", "P_reward")},
    }
    result = G.score(manifest, _choices(manifest, table), protocol)
    assert result["axes"]["model_change"]["outcome"] == "development_observation_only"
    assert result["axes"]["image_contribution"]["outcome"] == "inconclusive"
    assert result["selection_change_alone_counted"] is False and not any(result["claims"].values())
    better = result["results"][F2A]["B"]["followup"]
    assert (better["status"], better["conclusion"], better["probes_used"]) == ("confirmed", "actuator", 2)
    assert [step["probe"] for step in better["steps"]] == ["P_torque", "P_noise"]
    first = result["results"][F2A]["A"]["followup"]
    assert first["status"] == "pending_receipt" and first["steps"][-1] == {"probe": "P_physics", "outcome": None}
    assert result["wording"]["image_contribution"] == protocol["outcome_wording"]["image_contribution"]["inconclusive"]

    table[F1B]["B"] = _choice(rows[F1B], "B", status="semantic_invalid")
    invalid = G.score(manifest, _choices(manifest, table), protocol)
    assert invalid["results"][F1B]["B"]["hold"] is True and invalid["results"][F1B]["B"]["followup"] is None
    assert invalid["axes"]["model_change"]["outcome"] == "regression"
    table[F1B]["A"] = _choice(rows[F1B], "A", status="model_unavailable")
    assert G.score(manifest, _choices(manifest, table), protocol)["axes"]["model_change"]["outcome"] == "inconclusive"


def test_score_rejects_blocked_frames_swapped_capsules_and_unseen_facts(protocol: dict, manifest: dict) -> None:
    rows = _rows(manifest)
    blocked = {F2A: {"C": {"status": "valid", "capsule_sha256": None, "action": "continue", "probe": "P_torque",
                           "fact_ids": ["vision:frame:f01"]}}}
    with pytest.raises(ValueError, match="준비되지 않아"):
        G.score(manifest, _choices(manifest, blocked), protocol)
    swapped = {F2A: {"B": {**_choice(rows[F2A], "B", "P_torque"), "capsule_sha256": rows[F2B]["capsule"]["sha256"]}}}
    with pytest.raises(ValueError, match="다른 계보"):
        G.score(manifest, _choices(manifest, swapped), protocol)
    unseen = {F2A: {"B": _choice(rows[F2A], "B", "P_torque", facts=["vision:frame:f01"])}}
    with pytest.raises(ValueError, match="보이지 않은"):
        G.score(manifest, _choices(manifest, unseen), protocol)
    stale = {**_choices(manifest, {}), "manifest_digest": "0" * 64}
    with pytest.raises(ValueError, match="다른 manifest"):
        G.score(manifest, stale, protocol)


def test_cpu_paths_make_no_external_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    def blocked(*_args, **_kwargs):
        raise AssertionError("외부 연결 시도")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    assert G.main(["--prepare", "--output", str(tmp_path / "out")]) == 0
    assert G.main(["--check", "--manifest", str(tmp_path / "out" / "manifest.json")]) == 0
    assert G.main(["--prepare", "--output", str(tmp_path / "out")]) == 1
    assert "rejected" in capsys.readouterr().out
    source = (ROOT / "evals/nemotron35_value_cases_v1.py").read_text(encoding="utf-8")
    for name in ("requests", "httpx", "openai", "urllib.request", "socket", "subprocess"):
        assert f"import {name}" not in source


def test_seen_telemetry_ranges_are_descriptive_only() -> None:
    value = G.describe()
    ranges = value["f2_ranges_disjoint_from_f1_and_f5"]["Episode_Reward/dof_acc_l2"]
    assert ranges["F2"][2] == 6 and ranges["F1"][2] == 7 and ranges["F5"][2] == 3
    assert ranges["F2"][0] > max(ranges["F1"][1], ranges["F5"][1])
    assert value["threshold_created"] is False and value["used_for_selection"] is False


def test_protocol_is_unexecuted_draft_with_fixed_wording(protocol: dict) -> None:
    assert protocol["status"] == "DRAFT_not_executed" and protocol["executed"] is False
    assert protocol["pre_result_statement"]["image_selection_value"] == "not_established"
    outcomes = {"supported", "development_observation_only", "not_supported", "regression", "inconclusive"}
    assert all(set(protocol["outcome_wording"][axis]) == outcomes for axis in ("model_change", "image_contribution"))
    assert not any(protocol["claims"].values())
    banned = ("Team " + "Mollet", chr(0xC9C1) + chr(0xC811))  # 공개 문구 금지어를 이 파일에도 쓰지 않는다.
    for path in (G.PROTOCOL, ROOT / "docs/NEMOTRON35-VALUE-CASES.md"):
        text = path.read_text(encoding="utf-8")
        assert not any(word in text for word in banned)
