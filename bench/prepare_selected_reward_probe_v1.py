"""새 프레임 모델 선택을 s2221 F1 단일 P_reward 실행에 결합한다.

이 모듈은 승인·lease·resource proof를 만들지 않는다. Root가 만든 단일 probe plan과
authorization을 기존 production unit으로 다시 검증한 뒤에만 실행 명령을 출력한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "AGENTS.md").is_file() and (parent / "bench").is_dir())
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import agentic_investigation_contract_v2 as A2  # noqa: E402 # pyright: ignore[reportMissingImports]
import run_agentic_brev_probe_unit_v1 as UNIT  # noqa: E402

from rl_triage import probe_loop  # noqa: E402

SCHEMA = "selected_reward_probe_preregistration_v1"
CASE = {
    "side": "case",
    "arm": "F1",
    "seed": 2221,
    "probe": "P_reward",
    "checkpoint_sha256": "1d73d523c97c5c89c988febfbd3d5e533e90681cd0e6132d6ffe13888fa95731",
    "telemetry_sha256": "07ad9b4b3981eafd3a4ed84aba4a3ef04f54b94ddbc7be5b0b189d55b2e8ff6e",
}
REFERENCE_CHECKPOINT = "b346a0c3e3ad6410ce93dcf73e6165ed69e2d6ab95b2c5398943b3a8874cf54d"
REQUIRED_ACTUAL_KEYS = {
    "home_observation", "actual_metadata", "actual_environment", "create_intent",
    "created_binding", "create_receipt", "initial_inventory", "actual_host_resource",
    "canonical_lease_observation", "host_resource",
}
TARGET = {
    "resource_id": "d3cxwsmgd",
    "resource_name": "rl-triage-l40s-renderer-20261029h",
    "instance_type": "gpu-l40s-d.1gpu-16vcpu-96gb",
    "provider": "nebius",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_ref(value: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    if set(value) != {"path", "sha256"}:
        raise ValueError("immutable reference fields")
    path = Path(value["path"]).resolve(strict=True)
    if _sha(path) != value["sha256"]:
        raise ValueError(f"immutable reference changed: {path.name}")
    return path, _read(path)


def _contains(value: Any, key: str, expected: Any) -> bool:
    if isinstance(value, dict):
        if value.get(key) == expected:
            return True
        return any(_contains(item, key, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains(item, key, expected) for item in value)
    return False


def validate_actual_refs(value: dict[str, str]) -> dict[str, Any]:
    _, refs = _checked_ref(value)
    if set(refs) != REQUIRED_ACTUAL_KEYS:
        raise ValueError(f"actual ref keys mismatch: {sorted(set(refs) ^ REQUIRED_ACTUAL_KEYS)}")
    observations: list[dict[str, Any]] = []
    for name, ref in refs.items():
        _, observed = _checked_ref(ref)
        observations.append(observed)
        if name in {"actual_host_resource", "host_resource"}:
            rendered = json.dumps(observed, ensure_ascii=False)
            if any(item not in rendered for item in TARGET.values()):
                raise ValueError(f"H resource identity missing: {name}")
    if not any(_contains(row, "cuda_available", True) and _contains(row, "cuda_device_count", 1)
               for row in observations):
        raise ValueError("fresh H CUDA true/1 observation required")
    return {"keys": sorted(refs), "cuda_available": True, "cuda_device_count": 1}


def validate_selection(config: dict[str, Any]) -> dict[str, Any]:
    _, prepared = _checked_ref(config["prepared_linux"])
    _, source_prepared = _checked_ref(config["source_prepared"])
    _, projection = _checked_ref(config["prepared_projection"])
    _, result = _checked_ref(config["actual_call_result"])
    context = A2.checked(source_prepared["context"])
    if not (projection.get("status") == "PASS"
            and projection.get("projected_prepared_sha256") == config["prepared_linux"]["sha256"]
            and projection.get("source_prepared_sha256") == config["source_prepared"]["sha256"]
            and projection.get("same_model_messages") is True
            and projection.get("same_model_view") is True
            and projection.get("same_prior_history") is True
            and prepared["context"]["model_view"] == context["model_view"]):
        raise ValueError("prepared Linux projection binding differs")
    decision = result["engine_result"]["review"]["review"]["decision"]
    review = A2.assess(json.dumps(decision, ensure_ascii=False), context)
    choice = A2.probe_choice(context, review)
    recorded = result["engine_result"]["review"]["controller_choice"]
    if choice != recorded or choice.get("probe") != CASE["probe"] or choice.get("selected_by") != "model":
        raise ValueError("registered model choice differs from P_reward")
    fact_ids = choice.get("fact_ids", [])
    if len(fact_ids) != 6 or any(not item.startswith("telemetry:") for item in fact_ids):
        raise ValueError("selection must cite exactly six telemetry facts")
    api = result.get("api", {})
    if api != {"reserved": 10, "pending": 0, "remaining": 2}:
        raise ValueError("API ledger state differs from 10/12 pending0")
    if result.get("status") != "choice_ready" or result["engine_result"].get("approval_granted") is not False:
        raise ValueError("model choice is not valid and unapproved")
    return {"choice": choice, "api": api, "context_digest": prepared["context"]["context_digest"],
            "source_context_checked": True, "linux_projection_checked": True}


def validate_history(config: dict[str, Any]) -> dict[str, Any]:
    _, history = _checked_ref(config["r19_verification"])
    none = history.get("none_probe", {})
    f1 = history.get("f1", {})
    if not (history.get("status") == "PASS" and none.get("completed") is True
            and none.get("exit_code") == 0 and none.get("checkpoint_sha256") == REFERENCE_CHECKPOINT
            and f1 == {"parent_intent_present": True, "execution_present": False, "worker_present": False}):
        raise ValueError("R19 historical NONE/F1 state differs")
    differences = history["root_cause_reproduction"]["only_differences"]
    paths = {row["path"] for row in differences}
    if paths != {"$.environment.cuda_available", "$.environment.cuda_device_count"}:
        raise ValueError("R19 seal-only mismatch changed")
    return {
        "none": "historical_actual_exit0_unsealed_not_new_receipt",
        "none_gpu_rerun": False,
        "f1_execution_present": False,
        "historical_archive_sha256": history["archive"]["sha256"],
    }


def validate_preregistration(config: dict[str, Any]) -> dict[str, Any]:
    _, protocol = _checked_ref(config["protocol"])
    if protocol.get("schema") != SCHEMA:
        raise ValueError("selected probe protocol schema")
    if protocol.get("case") != CASE or protocol.get("expected_outcome") != "abnormal":
        raise ValueError("case source or prediction changed")
    if protocol.get("normal_or_unknown") != "HOLD_no_cause_confirmation_no_correction":
        raise ValueError("normal/unknown rule changed")
    if protocol.get("reference_policy") != "reuse_historical_unsealed_result_without_gpu_rerun":
        raise ValueError("NONE no-rerun rule changed")
    if protocol.get("second_model_call") != "only_after_valid_sealed_probe_receipt":
        raise ValueError("second model call gate changed")
    if protocol.get("time5", {}).get("sha256_lf") != config["time5_sha256_lf"]:
        raise ValueError("time5 binding changed")
    threshold = probe_loop.THRESHOLDS["P_reward"]
    if threshold != ("track_lin_vel_xy_exp_rel_error", "above", 0.05):
        raise ValueError("frozen P_reward threshold changed")
    if UNIT.INITIAL["case"] != {"arm": "F1", "checkpoint_sha256": CASE["checkpoint_sha256"],
                                "telemetry_sha256": CASE["telemetry_sha256"]}:
        raise ValueError("production unit case source changed")
    return {"threshold": list(threshold), "case_only_supported": True}


def prepare(config_path: Path, output: Path, *, allow_fixture: bool = False,
            fixture_home: Path | None = None, fixture_now: Any = None) -> dict[str, Any]:
    config = _read(config_path.resolve(strict=True))
    static = {
        "selection": validate_selection(config),
        "history": validate_history(config),
        "preregistration": validate_preregistration(config),
    }
    output.mkdir(parents=True, exist_ok=False)
    if "actual_refs" not in config:
        result = {
            "schema": "selected_reward_probe_prepare_receipt_v1",
            "status": "HOLD_missing_fresh_h_actual_refs",
            "required_actual_ref_keys": sorted(REQUIRED_ACTUAL_KEYS),
            "required_dynamic_inputs": [
                "fresh_H_actual_refs", "public_unit_protocol_anchor", "canonical_live_lease_snapshot",
                "capture_parent", "resource_proof", "selection_probe_authorization_and_ledger_prefix",
                "case_source_root_map", "single_unit_plan",
            ],
            "static": static,
            "gpu_operations": 0,
            "api_calls": 0,
            "authority_issued": False,
        }
    else:
        actual = validate_actual_refs(config["actual_refs"])
        if "plan" not in config or "authorization" not in config:
            result = {
                "schema": "selected_reward_probe_prepare_receipt_v1",
                "status": "HOLD_missing_root_plan_or_authorization",
                "actual": actual,
                "static": static,
                "gpu_operations": 0,
                "api_calls": 0,
                "authority_issued": False,
            }
        else:
            plan_path, _ = _checked_ref(config["plan"])
            auth_path, _ = _checked_ref(config["authorization"])
            unit = UNIT.prepare(
                {"path": str(plan_path), "sha256": config["plan"]["sha256"]},
                {"path": str(auth_path), "sha256": config["authorization"]["sha256"]},
                allow_fixture=allow_fixture, home=fixture_home, now=fixture_now,
            )
            identity = unit["plan"]["identity"]
            if identity != {"side": "case", "arm": "F1", "seed": 2221,
                            "probe": "P_reward", "run_name": identity["run_name"]}:
                raise ValueError("production unit is not case-only P_reward")
            result = {
                "schema": "selected_reward_probe_prepare_receipt_v1",
                "status": "ready_for_root_execute",
                "actual": actual,
                "static": static,
                "unit_status": unit["status"],
                "plan": config["plan"],
                "authorization": config["authorization"],
                "execute_argv": [
                    "<registered-python>", "-B", "bench/run_agentic_brev_probe_unit_v1.py",
                    "--plan", "<plan-path>", "--plan-sha256", config["plan"]["sha256"],
                    "--authorization", "<authorization-path>",
                    "--authorization-sha256", config["authorization"]["sha256"], "--execute",
                ],
                "gpu_operations": 0,
                "api_calls": 0,
                "authority_issued": False,
            }
    target = output / "receipt.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.config, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "ready_for_root_execute" else 3


if __name__ == "__main__":
    raise SystemExit(main())
