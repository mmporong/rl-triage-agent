from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from evals import summarize_balanced_recovery_continuation_v1 as summary


def _plan() -> dict:
    methods = list(summary.E.controller.METHODS)
    cases = [
        {"case_id": case_id, "rotation": methods[index:] + methods[:index]}
        for index, case_id in enumerate(summary.E.controller.SUPPORTED_CASES)
    ]
    return {
        "schema": summary.PLAN_SCHEMA,
        "cases": cases,
        "limits": {
            "max_recoveries": 16,
            "gpu_sdk_launch_cap": 16,
            "cpu_checkpoint_launch_cap": 8,
            "cpu_telemetry_launch_cap": 8,
            "total_owned_launch_cap": 32,
            "max_parent_wall_s_per_recovery": 5000,
            "api_cap": 0,
        },
        "continuation": {
            "profile_id": summary.PROFILE,
            "no_model_requery": True,
            "prior_whole_study_sdk_launches": 49,
            "cumulative_gpu_sdk_launch_cap": 65,
        },
    }


def _manifest(plan: dict) -> dict:
    arms = []
    effect_keys = set(summary.E.controller.V17_EFFECT_KEYS)
    training_key = summary.E.controller.V17_TRAINING_KEY
    for case in plan["cases"]:
        for method in case["rotation"]:
            key = (case["case_id"], method)
            continuation = None
            if key in effect_keys:
                continuation = {"kind": "completed_effect"}
            elif key == training_key:
                continuation = {"kind": "completed_training"}
            arms.append({
                "case_id": key[0],
                "method": key[1],
                "continuation_import": continuation,
                "recovery_payload": None if key in effect_keys or key == summary.E.EXPECTED_HOLD else {},
            })
    return {"arms": arms}


def _status(manifest: dict, plan: dict) -> dict:
    return {
        "completed": True,
        "halted": False,
        "completed_arms": 16,
        "api_calls": 0,
        "limits": plan["limits"],
        "arms": [
            {"case_id": row["case_id"], "method": row["method"], "resource_cost": None}
            for row in manifest["arms"]
        ],
    }


@pytest.mark.parametrize("mutation,match", [
    ("profile", "not complete"),
    ("limits", "not complete"),
    ("hold", "HOLD"),
    ("imports", "imports"),
])
def test_validate_uses_v11_continuation_and_exact_status_contract(
    mutation: str, match: str,
) -> None:
    plan = _plan()
    manifest = _manifest(plan)
    status = _status(manifest, plan)
    if mutation == "profile":
        plan["continuation"]["profile_id"] = "other"
    elif mutation == "limits":
        status["limits"] = {**plan["limits"], "gpu_sdk_launch_cap": 17}
    elif mutation == "hold":
        next(row for row in manifest["arms"]
             if row["recovery_payload"] == {} and row["continuation_import"] is None)[
            "recovery_payload"
        ] = None
    else:
        next(row for row in manifest["arms"]
             if (row.get("continuation_import") or {}).get("kind") == "completed_effect")[
                 "continuation_import"
             ] = None
    with pytest.raises(summary.E.ExportError, match=match):
        summary._validate(manifest, plan, status)


def _timing(keys: list[tuple[str, str]]) -> dict:
    return {
        "transitions": [
            {"case_id": case_id, "method": method}
            for case_id, method in keys
            for _ in range(2)
        ]
    }


def test_join_order_requires_overlap_at_interrupted_arm_and_excludes_hold_from_timing() -> None:
    plan = _plan()
    expected = summary.E._expected_order(plan)
    current = [key for key in expected[7:] if key != summary.E.EXPECTED_HOLD]
    assert summary._join_order(plan, _timing(expected[:8]), _timing(current)) == expected
    with pytest.raises(summary.E.ExportError, match="interrupted arm"):
        summary._join_order(plan, _timing(expected[:7]), _timing(current))
    with pytest.raises(summary.E.ExportError, match="interrupted arm"):
        summary._join_order(plan, _timing(expected[:8]), _timing(current[1:]))


def _resource_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict[str, dict]]:
    plan = _plan()
    manifest = _manifest(plan)
    status = _status(manifest, plan)
    bodies: dict[str, dict] = {}
    verified: dict[str, dict] = {}
    counter = 0

    def ref(prefix: str, body: dict) -> dict:
        nonlocal counter
        counter += 1
        sha = f"{counter:064x}"
        value = {"path": f"{prefix}-{counter}.json", "sha256": sha}
        bodies[sha] = body
        return value

    def raw(kind: str, sdk: int, cpu: int, telemetry: int) -> dict:
        sampler = ref("sampler", {
            "schema": "periodic_device_sampling_stop_v1",
            "scope": "whole_device",
            "started_utc": "2026-11-05T00:00:00+00:00",
            "ended_utc": "2026-11-05T00:00:10+00:00",
            "device_summary": {"GPU": {"memory_peak": 1}},
            "energy": {"devices": {"GPU": {"energy_wh": 1.0}}},
            "process_gpu_seconds": None,
            "provider_gpu_seconds": None,
        })
        value = {
            "request_id": kind,
            "effect_key": kind,
            "simulator_sdk_launches": sdk,
            "cpu_checkpoint_launches": cpu,
            "cpu_telemetry_launches": telemetry,
            "total_owned_children": sdk + cpu + telemetry,
            "phase_wall_s": 10.0,
            "owned_gpu_s": 8.0,
            "lease_await_s": None,
            "sampler_ref": sampler,
            "raw_receipt_ref": None,
        }
        resource = ref("resource", value)
        verified[resource["sha256"]] = value
        return resource

    historical_effects = iter(raw(f"effect-{index}", 2, 0, 0) for index in range(7))
    prior_training = raw("prior-training", 1, 0, 0)
    current_refs = [raw("fixed-only", 1, 1, 1)] + [
        raw(f"fresh-{index}", 2, 1, 1) for index in range(7)
    ]
    current = iter(current_refs)
    for arm, source in zip(status["arms"], manifest["arms"], strict=True):
        imported = (source.get("continuation_import") or {}).get("kind")
        if imported == "completed_effect":
            arm["resource_cost"] = {
                "schema": "registered_recovery_imported_resource_v1",
                "case_id": arm["case_id"], "method": arm["method"],
                "plan_sha256": "a" * 64, "status": "complete",
                "execution_origin": "verified_prior_effect",
                "prior_effect_proof_ref": {"path": "proof", "sha256": "e" * 64},
                "prior_resource_ref": next(historical_effects),
                "prior_admission_ref": {"path": "admission", "sha256": "f" * 64},
                "historical_recovery_folder": "historical",
                "new_dispatch_costs": {}, "historical_effect_costs": {},
                "receipt_ref": {"path": "projection", "sha256": "d" * 64},
            }
        elif (arm["case_id"], arm["method"]) == summary.E.EXPECTED_HOLD:
            arm["resource_cost"] = None
        else:
            resource = next(current)
            if imported == "completed_training":
                receipt = ref("receipt", {
                    "training_reuse": {
                        "new_training_launched": False,
                        "prior_dispatch_resource_ref": prior_training,
                    }
                })
                verified[resource["sha256"]]["raw_receipt_ref"] = receipt
            arm["resource_cost"] = {
                "schema": "registered_recovery_resource_receipt_v1",
                "recovery_resource_ref": resource,
            }

    monkeypatch.setattr(summary, "_read", lambda value, _label: (
        Path(value["path"]), bodies[value["sha256"]]
    ))
    monkeypatch.setattr(summary.E.resources, "verify", lambda value, **_kwargs: (
        verified[value["sha256"]]
    ))
    return status, verified


def test_resources_count_historical_and_current_once_and_mark_overlapping_energy_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, _ = _resource_fixture(monkeypatch)
    arms, totals, refs = summary._resources(status)
    assert len(arms) == 16
    assert len(refs) == 32
    assert totals["historical_completed_effects"] == 7
    assert totals["historical_completed_training"] == 1
    assert totals["current_dispatches"] == 8
    assert totals["historical_sdk_launches"] == totals["new_sdk_launches"] == 15
    assert totals["cpu_checkpoint_launches"] == totals["cpu_telemetry_launches"] == 8
    assert totals["total_owned_children"] == 31
    assert totals["device_energy_wh"] is None
    assert totals["device_energy_status"] == "unknown_overlap"
    assert summary._validate_accounting(_plan(), totals) == 64


@pytest.mark.parametrize("field,value", [
    ("new_sdk_launches", 14),
    ("historical_sdk_launches", 14),
    ("cpu_checkpoint_launches", 7),
    ("cpu_telemetry_launches", 7),
    ("total_owned_children", 33),
])
def test_accounting_rejects_incomplete_or_over_cap_counts(field: str, value: int) -> None:
    totals = {
        "new_sdk_launches": 15, "historical_sdk_launches": 15,
        "cpu_checkpoint_launches": 8, "cpu_telemetry_launches": 8,
        "total_owned_children": 31, "historical_completed_effects": 7,
        "historical_completed_training": 1, "current_dispatches": 8,
    }
    totals[field] = value
    with pytest.raises(summary.E.ExportError, match="accounting"):
        summary._validate_accounting(_plan(), totals)


def test_duplicate_resource_and_forbidden_tokenfactory_path_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, _ = _resource_fixture(monkeypatch)
    imported = [row for row in status["arms"]
                if (row.get("resource_cost") or {}).get("schema")
                == "registered_recovery_imported_resource_v1"]
    imported[1]["resource_cost"]["prior_resource_ref"] = deepcopy(
        imported[0]["resource_cost"]["prior_resource_ref"]
    )
    with pytest.raises(summary.E.ExportError, match="counted twice"):
        summary._resources(status)
    with pytest.raises(summary.E.ExportError, match="forbidden source subtree"):
        summary._guard(Path("bench/private/ri_codex2_tokenfactory_live_160/result.json"))


def test_base_imported_resource_routes_v11_to_continuation_v5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evals import registered_recovery_continuation_v5 as continuation_v5

    proof = {"path": "proof.json", "sha256": "a" * 64}
    prior_resource = {"path": "resource.json", "sha256": "b" * 64}
    prior_admission = {"path": "admission.json", "sha256": "c" * 64}
    scope = {
        "kind": "completed_effect", "case_id": "case-05", "method": "fixed_sequence",
        "proof_sha256": proof["sha256"], "prior_plan_sha256": "d" * 64,
        "diagnostic_manifest_sha256": "e" * 64,
        "source_checkpoint_sha256": "f" * 64,
    }
    row = {
        "prior_effect_proof_ref": proof, "prior_resource_ref": prior_resource,
        "prior_admission_ref": prior_admission, "historical_recovery_folder": "old",
        "receipt_ref": {"path": "projection.json", "sha256": "1" * 64},
    }
    calls = []

    def verify(proof_ref, *, expected_import, repo):
        calls.append((proof_ref, expected_import, repo))
        return {
            "prior_resource_ref": prior_resource, "prior_admission_ref": prior_admission,
            "recovery_folder": "old",
        }

    monkeypatch.setattr(continuation_v5, "verify_completed_effect", verify)
    monkeypatch.setattr(summary.E.base, "_resource", lambda *_args, **_kwargs: ({
        "current_dispatch_simulator_owned_wall_s": 2.0,
        "current_dispatch_sdk_launches": 2,
    }, []))
    monkeypatch.setattr(summary.E.base, "_generic_ref", lambda ref, logical, private: {
        "logical_id": logical, "sha256": ref["sha256"],
    })
    costs, _ = summary.E.base._imported_resource(
        row, "case-05:fixed_sequence", [], scope,
        summary.PLAN_SCHEMA,
    )
    assert len(calls) == 1
    assert calls[0][1] is scope
    assert costs["execution_origin"] == "verified_prior_effect"
    assert costs["reused_completed_dispatch_sdk_launches"] == 2
