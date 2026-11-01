from __future__ import annotations

import json
import subprocess
import threading
from copy import deepcopy
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]
from test_agentic_investigation_evidence_v1 import (
    B,
    E,
    case,  # noqa: F401
    ref,
    response,
    write,
)

from evals import agentic_frame_quality_guard_v1 as Q


def policy(tmp_path: Path, threshold: float = 10.0) -> dict:
    path = tmp_path / "quality-policy.json"
    write(path, {"schema": Q.POLICY_SCHEMA, "metric": Q.METRIC, "unit": Q.UNIT,
                 "min_laplacian_variance": threshold})
    return {"path": str(path), "sha256": E.V01.sha256(path)}


def adapter(data: dict, policy_ref: dict) -> Q.QualityEvidenceAdapter:
    return Q.QualityEvidenceAdapter(
        data["index"], data["roots"], data["registration"],
        expected_registration_sha256=E.index_sha256(data["registration"]),
        policy_ref=policy_ref)


def rebind_frames(data: dict, raw: bytes | None) -> None:
    contract_path = E._ref(data["entry"]["contract"], data["roots"])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for frame in contract["frames"][:2]:
        target = contract_path.parent / frame["path"]
        if raw is None:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 "color=c=gray:s=64x64", "-frames:v", "1", "-y", str(target)],
                check=True, capture_output=True)
        else:
            target.write_bytes(raw)
        frame["sha256"] = E.V01.sha256(target)
        frame["bytes"] = target.stat().st_size
    provenance = contract["provenance"]
    contract["contract_id"] = E.V01.digest({
        "run": provenance["play_run_id"],
        "lineage": {"id": provenance["source_lineage_id"],
                    "parent_id": provenance["parent_source_lineage_id"]},
        "video": provenance["video"]["sha256"],
        "checkpoint": provenance["checkpoint"]["sha256"],
        "frames": [{"timestamp_s": row["timestamp_s"], "sha256": row["sha256"]}
                   for row in contract["frames"]],
        "measurement": provenance["measurement"]["sha256"],
        "play": provenance["play"],
        "frame_extraction": provenance["frame_extraction"],
    })[:24]
    write(contract_path, contract)
    entry = deepcopy(data["entry"])
    entry["contract"] = ref(data["root"], contract_path)
    index = E.build_index({"schema": E.SPEC_SCHEMA, "entries": [entry]}, data["roots"])
    registration = B.register(
        index, data["roots"], data["snapshot"], expected_index_sha256=E.index_sha256(index),
        source_lineage_id=data["registration"]["session"]["source_lineage_id"],
        aliases=data["registration"]["aliases"])
    data.update(index=index, registration=registration, entry=entry)


def inspect(bridge: Q.QualityEvidenceAdapter, data: dict, visual: dict | None = None) -> tuple:
    loop = bridge.start(data["snapshot"], goal="영상 품질 경계", allow_fixture=True)
    if visual is None:
        loop.propose(response("inspect_frames", "evidence:clip-01"))
    receipt = bridge.inspect_frames(loop, "evidence:clip-01", visual_context=visual)
    return loop, receipt


def test_clear_registered_frames_are_sent_only_after_actual_quality_check(case: dict) -> None:  # noqa: F811
    bridge = adapter(case, policy(case["root"]))
    loop, receipt = inspect(bridge, case)
    assert receipt["visual_outcome"] == "available" and receipt["image_count"] == 2
    assert all(row["metric"]["value"] >= 10.0 and row["outcome"] == "available"
               for row in receipt["quality_receipts"])
    message = bridge.messages(loop, visual_context=receipt["visual_context"])
    assert message["image_count"] == 2
    assert sum(part["type"] == "image_url" for part in message["messages"][1]["content"]) == 2


def test_blurred_alias_is_unknown_and_images_are_quarantined_but_text_remains(case: dict) -> None:  # noqa: F811
    rebind_frames(case, None)
    bridge = adapter(case, policy(case["root"]))
    loop = bridge.start(case["snapshot"], goal="영상 품질 경계", allow_fixture=True)
    loop.propose(response("read_evidence", "evidence:training-01"))
    training = bridge.read_evidence(loop, "evidence:training-01")
    loop.propose(response("inspect_frames", "evidence:clip-01"))
    receipt = bridge.inspect_frames(loop, "evidence:clip-01",
                                    visual_context=training["visual_context"])
    assert receipt["visual_outcome"] == "unknown" and receipt["image_count"] == 0
    assert all(row["outcome"] == "quality_below_registered_floor"
               for row in receipt["quality_receipts"])
    message = bridge.messages(loop, visual_context=receipt["visual_context"])
    assert message["image_count"] == 0
    assert not any(part["type"] == "image_url" for part in message["messages"][1]["content"])
    assert "telemetry:case:" in message["messages"][1]["content"][0]["text"]


def test_decode_failure_metric_is_not_converted_to_a_numeric_quality_value() -> None:
    with pytest.raises(ValueError, match="decode unavailable"):
        Q.laplacian_variance(b"registered-sha-but-not-an-image")


def test_official_venv_import_has_no_pillow_dependency() -> None:
    assert "PIL" not in Q.sys.modules
    assert Q.METRIC.startswith("ffmpeg_gray_u8_")


def test_registered_sha_tamper_raises_before_quality_outcome(case: dict) -> None:  # noqa: F811
    bridge = adapter(case, policy(case["root"]))
    image = case["root"] / "contract/artifacts/frames/frame_01.jpg"
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA"):
        bridge.start(case["snapshot"], goal="영상 품질 경계", allow_fixture=True)


def test_policy_and_visual_receipts_are_lineage_bound_and_resume_identically(case: dict) -> None:  # noqa: F811
    policy_ref = policy(case["root"])
    first = adapter(case, policy_ref)
    loop, receipt = inspect(first, case)
    second = adapter(case, policy_ref)
    resumed = second.messages(loop, visual_context=receipt["visual_context"])
    assert resumed["visual_context"] == receipt["visual_context"]
    assert resumed["image_count"] == receipt["image_count"]
    changed = deepcopy(receipt["visual_context"])
    changed["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="quality visual context"):
        second.messages(loop, visual_context=changed)
    Path(policy_ref["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="policy SHA"):
        adapter(case, policy_ref)


def test_receipt_contains_only_registered_quality_evidence_fields(case: dict) -> None:  # noqa: F811
    bridge = adapter(case, policy(case["root"]))
    _, receipt = inspect(bridge, case)
    expected = {"schema", "policy_sha256", "registration_sha256", "index_sha256",
                "frame_content_sha256", "metric", "outcome"}
    assert all(set(row) == expected for row in receipt["quality_receipts"])
    lowered = json.dumps(receipt["quality_receipts"], ensure_ascii=False).lower()
    assert all(token not in lowered for token in ("behavior", "fault", "healthy", "recovered"))


def test_adapter_scope_is_exact_spec_current_thread_only_and_restores(case: dict) -> None:  # noqa: F811
    bridge = adapter(case, policy(case["root"]))
    spec = Q.quality_evidence_spec(bridge)
    original = Q.BASE._adapter
    other: list[type] = []
    with Q.adapter_scope(spec):
        assert isinstance(Q.BASE._adapter(spec), Q.QualityEvidenceAdapter)
        changed = deepcopy(spec)
        changed["registration_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="new registration required"):
            Q.BASE._adapter(changed)
        assert Q.BASE._adapter(None) is None

        def consume() -> None:
            other.append(type(Q.BASE._adapter(spec)))

        thread = threading.Thread(target=consume)
        thread.start()
        thread.join()
        assert other == [B.EvidenceAdapter]
    assert Q.BASE._adapter is original


def test_two_concurrent_scopes_capture_and_restore_inside_lock(case: dict) -> None:  # noqa: F811
    spec = Q.quality_evidence_spec(adapter(case, policy(case["root"])))
    original = Q.BASE._adapter
    first_entered = threading.Event()
    second_attempting = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    failures: list[BaseException] = []

    def first() -> None:
        try:
            with Q.adapter_scope(spec):
                first_entered.set()
                assert second_attempting.wait(5)
                assert release_first.wait(5)
        except BaseException as error:  # noqa: BLE001 - thread assertion을 부모로 전달한다.
            failures.append(error)

    def second() -> None:
        try:
            assert first_entered.wait(5)
            second_attempting.set()
            with Q.adapter_scope(spec):
                second_entered.set()
                assert isinstance(Q.BASE._adapter(spec), Q.QualityEvidenceAdapter)
        except BaseException as error:  # noqa: BLE001 - thread assertion을 부모로 전달한다.
            failures.append(error)

    one, two = threading.Thread(target=first), threading.Thread(target=second)
    one.start()
    two.start()
    assert first_entered.wait(5) and second_attempting.wait(5)
    assert not second_entered.wait(0.1)
    release_first.set()
    one.join(5)
    two.join(5)
    assert not one.is_alive() and not two.is_alive() and not failures
    assert second_entered.is_set() and Q.BASE._adapter is original


def test_new_epoch_checkpoint_requires_a_new_registration(case: dict) -> None:  # noqa: F811
    bridge = adapter(case, policy(case["root"]))
    changed = deepcopy(case["snapshot"])
    changed["inputs"]["packet"]["input_bindings"]["case"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        bridge.start(changed, goal="새 epoch", allow_fixture=True)


@pytest.mark.parametrize("raw", ["true", "NaN", "Infinity", "-Infinity"])
def test_policy_floor_rejects_bool_nan_and_infinity(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "bad-policy.json"
    path.write_text(
        '{"schema":"agentic_frame_quality_policy_v1",'
        f'"metric":"{Q.METRIC}",'
        '"unit":"intensity_squared","min_laplacian_variance":' + raw + "}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quality policy contract"):
        Q._read_ref({"path": str(path), "sha256": E.V01.sha256(path)})
