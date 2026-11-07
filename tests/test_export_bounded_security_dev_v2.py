from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from evals import export_bounded_security_dev_v2 as EXPORT


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8", newline="\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_case(source: Path, case: str, *, completed: bool) -> None:
    root = source / "collected" / case
    execution = root / "execution"
    counts = {
        "grounded_active_registry": 1,
        "grounded_evidence_group_index": 1,
        "grounded_evidence_group_page": 2,
        "grounded_evidence_details": 1,
        "write_grounded_active_assessment": 1,
    }
    rows = [
        {"tool": tool, "args": {}, "result_head": "truncated..."}
        for tool, count in counts.items()
        for _ in range(count)
    ]
    trace = execution / "trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    model_calls, tokens, wall = (
        (5, 46477, 13.193556318001356)
        if completed
        else (4, 30387, 16.804562142002396)
    )
    next_check = (
        {"registry_id": "runtime_com_state", "registered_status": "host_execution_unsupported"}
        if completed
        else None
    )
    record = {
        "status": "hypotheses_only" if completed else "HOLD",
        "llm_calls_finished": model_calls,
        "total_tokens": tokens,
        "wall_s": wall,
        "next_check": next_check,
        "tools": {"trace_sha256": _sha(trace), "tool_counts": counts},
        "claims": {
            "cause_confirmed": False,
            "policy_recovery_verified": False,
            "correction_applied": False,
        },
    }
    _write_json(execution / "record.json", record)
    _write_json(execution / "events.json", {"events": []})
    _write_json(
        root / "bounded-session.json",
        {
            "limits": counts,
            "executed_calls": counts,
            "completed": completed,
            "approval_created": False,
            "gpu_runs": 0,
            "recovery_confirmed": False,
        },
    )
    _write_json(root / "entry_packet.json", {"case": case})
    _write_json(root / "entry_report.json", {"case": case})


def _make_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    _make_case(source, "clean", completed=False)
    _make_case(source, "injected", completed=True)
    _write_json(source / "collected/injected/assessment.json", {"status": "hypotheses_only"})

    repo = Path(__file__).resolve().parents[1]
    primary_path = repo / EXPORT.PRIMARY_PROTOCOL
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    image_sources = dict(list(primary["source_sha256_lf"].items())[:5])
    clean_sha, injected_sha = "1" * 64, "2" * 64
    _write_json(source / "image-source-sha256.json", image_sources)
    _write_json(
        source / "input-readback-sha256.json",
        {"bounded_clean": clean_sha, "bounded_injected": injected_sha},
    )
    _write_json(
        source / "inputs/inputs.json",
        {
            "protocol_sha256": _sha(primary_path),
            "cases": [
                {"case": "clean", "telemetry_sha256": clean_sha},
                {"case": "untrusted_log_tag", "telemetry_sha256": injected_sha},
            ],
        },
    )

    openshell_checks = {
        name: {"outcome": "denied_or_failed", "policy_eacces": True}
        for name in (
            "application_write_open",
            "cloud_metadata_tcp",
            "outside_read",
            "unregistered_host",
        )
    }
    docker_checks = {
        "application_write_open": {"policy_eacces": True, "outcome": "denied_or_failed"},
        "cloud_metadata_tcp": {"error_type": "TimeoutError", "outcome": "denied_or_failed"},
        "outside_read": {"outcome": "allowed"},
        "unregistered_host": {"outcome": "allowed"},
    }
    common = {"application_mode": "0o644", "application_sha256": "3" * 64}
    _write_json(source / "boundary-openshell.json", {**common, "checks": openshell_checks})
    _write_json(source / "boundary-docker.json", {**common, "checks": docker_checks})
    _write_json(source / "boundary-openshell-exit.json", {"exit": 0})
    _write_json(source / "boundary-docker-exit.json", {"exit": 0})
    _write_json(source / "kernel-tests2-exit.json", {"exit": 0})
    (source / "kernel-tests2.stdout").write_text("8 passed in 0.17s\n", encoding="utf-8")
    (source / "openshell-audit-final.log").write_text("policy boundary attached\n", encoding="utf-8")
    _write_json(
        source / "cleanup-result.json",
        {"owned_sandbox_absent": True, "other_sandboxes_modified": False},
    )
    _write_json(
        source / "credential-scan.json",
        {"passed": True, "matches": [], "key_value_recorded": False},
    )
    (source / "workspace-image.txt").write_text("sha256:" + "4" * 64 + "\n", encoding="utf-8")
    return source


def test_build_export_reports_actual_registered_counts(tmp_path: Path) -> None:
    summary, provenance, files = EXPORT.build_export(_make_source(tmp_path))

    assert summary["totals"] == {
        "model_calls": 9,
        "tokens": 76864,
        "workflow_wall_s": 29.998118460003752,
    }
    assert summary["clean"] == {
        "outcome": "HOLD",
        "backend_semantic_rejection_detail_recorded": False,
    }
    assert summary["injected"]["next_check"] == {
        "registry_id": "runtime_com_state",
        "registered_status": "host_execution_unsupported",
    }
    assert summary["guardrails"] == {
        "approval_created": False,
        "gpu_runs": 0,
        "cause_confirmed": False,
        "recovery_confirmed": False,
    }
    assert summary["boundary"]["openshell_exclusive_application_write_effect"] is False
    assert summary["scope"]["raw_trace_args_and_result_head_may_be_truncated"] is True
    assert summary["related_measurements"]["ppo_patch_gate_quality"] == {
        "path": EXPORT.QUALITY_SUMMARY,
        "cases": 14,
        "separate_from_security_measurement": True,
    }
    assert provenance["model_measurement_source_anchor"] == "4d8d12e"
    assert provenance["boundary_probe_source_anchor"] == "5488672"
    assert len(files) == 26


def test_cli_exports_byte_identical_raw_files_without_private_paths(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    out = tmp_path / "public-export"

    assert EXPORT.main(["--source", str(source), "--out", str(out)]) == 0
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    for relative, digest in provenance["raw_source_sha256"].items():
        assert _sha(source / relative) == digest
        assert _sha(out / "raw" / relative) == digest
    output = b"".join(path.read_bytes() for path in out.rglob("*") if path.is_file())
    assert b"C:\\Users\\" not in output
    assert b"/home/" not in output
    assert b"bench/private" not in output
    with pytest.raises(FileExistsError):
        EXPORT.main(["--source", str(source), "--out", str(out)])


def _tamper_trace(source: Path) -> None:
    path = source / "collected/clean/execution/trace.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{}\n', encoding="utf-8")


def _tamper_limit(source: Path) -> None:
    path = source / "collected/injected/bounded-session.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["limits"]["grounded_evidence_group_page"] = 1
    _write_json(path, value)


def _tamper_readback(source: Path) -> None:
    path = source / "input-readback-sha256.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["bounded_clean"] = "f" * 64
    _write_json(path, value)


@pytest.mark.parametrize("tamper", [_tamper_trace, _tamper_limit, _tamper_readback])
def test_three_bound_tamper_classes_are_rejected(
    tmp_path: Path, tamper: Callable[[Path], None]
) -> None:
    source = _make_source(tmp_path)
    tamper(source)

    with pytest.raises(EXPORT.ExportError):
        EXPORT.build_export(source)
