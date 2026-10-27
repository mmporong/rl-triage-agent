from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import vision_archived_native_observer_v1 as A


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _contract(checkpoint: str = "c" * 64) -> dict:
    frames = [
        {"frame_id": f"f{index:02d}", "sha256": f"{index:x}" * 64}
        for index in range(1, 7)
    ]
    commands = [
        {"source_frame_index": index * 10, "source_frame_pts_s": index + 0.5}
        for index in range(1, 7)
    ]
    return {
        "provenance": {
            "video": {"sha256": "a" * 64},
            "checkpoint": {"sha256": checkpoint},
            "measurement": {"sha256": "b" * 64},
            "frame_extraction": {"commands": commands},
        },
        "frames": frames,
        "synchronized_samples": [{"timestamp_s": index + 0.5} for index in range(1, 7)],
    }


def _protocol_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict]:
    archive = tmp_path / "archive"
    contract_path = archive / "tree/repo/native/source-01/contract.json"
    receipt_path = archive / "tree/repo/native/source-01/receipt.json"
    _write(contract_path, {"fixture": True})
    contract_sha = A.file_sha(contract_path)
    source_ref = {
        "path": "/remote/native/source-01/contract.json",
        "sha256": contract_sha,
    }
    receipt = {
        "schema": "vision_corrected_native_record_receipt_v1",
        "status": "verified_native_record",
        "seed": 2220,
        "checkpoint_sha256": "c" * 64,
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
        "v01_contracts": [{"window": 1, "verified": True, "contract": source_ref}],
    }
    _write(receipt_path, receipt)
    manifest = {
        "schema": "vision_transferred_e2e_close_manifest_v1",
        "archive_repo_prefix": "repo",
        "source_cache_rearchived": False,
        "files": [
            {
                "source_path": source_ref["path"],
                "archive_path": "repo/native/source-01/contract.json",
                "sha256": contract_sha,
            }
        ],
    }
    manifest_path = archive / "retrieved/e2e-close-manifest.json"
    _write(manifest_path, manifest)
    checked_contract = _contract()
    monkeypatch.setattr(A.V01, "verify", lambda path: checked_contract)
    monkeypatch.setattr(
        A.OBSERVER,
        "request_view",
        lambda path: {"messages": [{"role": "user", "content": "same"}]},
    )
    evidence = A.evidence_fingerprint(checked_contract)
    row = {
        "group": "source-01",
        "seed": 2220,
        "label": "known_healthy_corrected_policy",
        "checkpoint_sha256": "c" * 64,
        "native_receipt": {
            "path": receipt_path.relative_to(archive).as_posix(),
            "sha256": A.file_sha(receipt_path),
        },
        "contract": {
            "path": contract_path.relative_to(archive).as_posix(),
            "sha256": contract_sha,
        },
        "window": 1,
        "evidence": evidence,
        "request_messages_sha256": A.JOURNAL.digest(
            [{"role": "user", "content": "same"}]
        ),
        "original_request_messages_sha256": A.JOURNAL.digest(
            [{"role": "user", "content": "same"}]
        ),
        "provenance_transition": "absolute_remote_reference_to_verified_archive_member_only",
        "used_for_decision": False,
    }
    row2 = json.loads(json.dumps(row))
    row2.update(group="source-02", seed=2221)
    receipt2_path = archive / "tree/repo/native/source-02/receipt.json"
    contract2_path = archive / "tree/repo/native/source-02/contract.json"
    _write(contract2_path, {"fixture": True})
    source_ref2 = {
        "path": "/remote/native/source-02/contract.json",
        "sha256": A.file_sha(contract2_path),
    }
    receipt2 = {
        **receipt,
        "seed": 2221,
        "v01_contracts": [{"window": 1, "verified": True, "contract": source_ref2}],
    }
    _write(receipt2_path, receipt2)
    manifest["files"].append(
        {
            "source_path": source_ref2["path"],
            "archive_path": "repo/native/source-02/contract.json",
            "sha256": source_ref2["sha256"],
        }
    )
    _write(manifest_path, manifest)
    row2["native_receipt"] = {
        "path": receipt2_path.relative_to(archive).as_posix(),
        "sha256": A.file_sha(receipt2_path),
    }
    row2["contract"] = {
        "path": contract2_path.relative_to(archive).as_posix(),
        "sha256": source_ref2["sha256"],
    }
    protocol = {
        "schema": A.SCHEMA,
        "protocol_id": "fixture-v1",
        "registered_before_api": True,
        "development_scope": "known_healthy_corrected_policy_auxiliary_observation",
        "source_commit": "d" * 40,
        "source_sha256_lf": {"evals/unit.py": "e" * 64},
        "archive": {
            "manifest": {
                "path": manifest_path.relative_to(archive).as_posix(),
                "sha256": A.file_sha(manifest_path),
            },
            "archive_repo_prefix": "repo",
        },
        "model": {"name": A.OBSERVER.MODEL, "params": A.OBSERVER.REQUEST_PARAMS},
        "limits": {"new_api_requests": 6, "cases": 2, "max_calls_per_case": 3},
        "prior_execution": {
            "initial_model_api_requests": 2,
            "remote_post_observation_api_requests": 0,
            "combined_api_requests_after_upper_bound": 8,
            "original_e2e_api_cap": 32,
        },
        "rows": [row, row2],
        "preregistration": {
            "prediction": "두 known-healthy 교정 정책의 실제 native 프레임에서 walking=yes, falling=no 관찰을 예측한다.",
            "valid": "grounded 계약이 유효하면 보조 관찰로 기록하고 회복 판정에는 사용하지 않는다.",
            "hold": "전송·형식·근거 검증이 끝나지 않으면 observed_HOLD로 기록하며 유효 관찰로 바꾸지 않는다.",
            "label_separation": "known-healthy 개발 관찰이며 새 회복률·일반화 근거로 사용하지 않는다.",
        },
    }
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, protocol)
    return protocol_path, archive, protocol


def test_relocated_contract_binds_remote_member_and_same_model_view(
    tmp_path, monkeypatch
):
    protocol_path, archive, _ = _protocol_fixture(tmp_path, monkeypatch)
    checked = A.validate_protocol(protocol_path, archive, verify_source=False)
    assert [row["group"] for row in checked["rows"]] == ["source-01", "source-02"]
    assert all(
        row["request_messages_sha256"] == row["original_request_messages_sha256"]
        for row in checked["rows"]
    )


def test_frame_state_or_checkpoint_tamper_is_rejected(tmp_path, monkeypatch):
    _protocol_path, archive, protocol = _protocol_fixture(tmp_path, monkeypatch)
    protocol["rows"][0]["evidence"]["state_digest"] = "0" * 64
    _write(tmp_path / "tampered.json", protocol)
    with pytest.raises(ValueError, match="frame/video/state/checkpoint"):
        A.validate_protocol(tmp_path / "tampered.json", archive, verify_source=False)


def test_archive_path_escape_is_rejected(tmp_path, monkeypatch):
    _protocol_path, archive, protocol = _protocol_fixture(tmp_path, monkeypatch)
    protocol["rows"][0]["contract"]["path"] = "../outside.json"
    _write(tmp_path / "escape.json", protocol)
    with pytest.raises(ValueError, match="탈출"):
        A.validate_protocol(tmp_path / "escape.json", archive, verify_source=False)


def test_prepare_and_advance_keep_shared_cap_and_auxiliary_boundary(
    tmp_path, monkeypatch
):
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, {"fixture": True})
    contract = tmp_path / "contract.json"
    _write(contract, {"fixture": True})
    protocol = {
        "source_commit": "a" * 40,
        "archive": {"manifest": {"sha256": "b" * 64}},
    }
    checked = {
        "sha256": A.file_sha(protocol_path),
        "value": protocol,
        "rows": [
            {
                "group": "source-01",
                "seed": 2220,
                "label": "known_healthy_corrected_policy",
                "contract": {"sha256": A.file_sha(contract)},
                "contract_path": contract,
                "request_messages_sha256": "c" * 64,
            }
        ],
    }
    monkeypatch.setattr(A, "validate_protocol", lambda *_args, **_kwargs: checked)
    monkeypatch.setattr(
        A.OBSERVER, "binding_template", lambda *_args, **_kwargs: {"binding": True}
    )
    calls = []

    def run(folder, contract_path, binding, complete, *, api_root, api_cap):
        calls.append((folder, contract_path, binding, api_root, api_cap, complete))
        return {
            "status": "observation_ready",
            "engine_result": {"status": "observation_ready"},
            "api": {},
        }

    monkeypatch.setattr(A.OBSERVER, "run", run)
    output = tmp_path / "out"
    assert A.prepare(protocol_path, tmp_path, output)["api_cap"] == 6
    result = A.advance(
        protocol_path, tmp_path, output, "source-01", complete=lambda *_: {}
    )
    assert result["status"] == "observation_ready"
    assert result["used_for_decision"] is False
    assert calls[0][3] == output / "shared-api" and calls[0][4] == 6
