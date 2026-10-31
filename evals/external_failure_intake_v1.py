"""Deterministic intake for selected public, user-reported RL training failures.

This module does not diagnose a root cause, choose a repair, or claim recovery.
It converts verified self-reported evidence into a small selector-facing fact card
and keeps later maintainer replies outside that input boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class ExternalIntakeError(ValueError):
    """Raised when an external-report manifest is not admissible."""


SCHEMA = "external_failure_source_manifest_v1"
OUTPUT_SCHEMA = "external_failure_intake_result_v1"
FORBIDDEN_SELECTOR_KEYS = frozenset(
    {
        "later_maintainer_diagnosis",
        "maintainer_diagnosis",
        "maintainer_answer",
        "resolution",
        "accepted_answer",
        "comments",
    }
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _span_fact(text: str, span: str) -> dict[str, str]:
    if text.count(span) != 1:
        raise ExternalIntakeError("등록 source span이 없거나 고유하지 않습니다")
    return {"text_sha256": _sha256_text(span), "text": span}


REPORTS: dict[str, dict[str, Any]] = {
    "rsl_rl_33_actor_std_nan": {
        "source_url": "https://github.com/leggedrobotics/rsl_rl/issues/33",
        "retrieved_on": "2026-11-01",
        "original_body_sha256": "43bedbc1648f0dc23455a66ae20d33bb64037aca18def7e95a62fef0fc53c849",
        "stage": "training_update_failure",
        "applicability_scope": "nonfinite_observation_and_optimizer_update_check_only",
        "checks": (
            {
                "check_id": "external_nonfinite_actor_std",
                "spans": (
                    "RuntimeError: normal expects all elements of std >= 0.0",
                    "env_id: 1111, row: tensor([nan, nan, nan",
                ),
                "fact_id": "external.actor_std_nonfinite_reported",
            },
        ),
    },
    "isaaclab_1847_rnd_cfg_keyerror": {
        "source_url": "https://github.com/isaac-sim/IsaacLab/issues/1847",
        "retrieved_on": "2026-11-01",
        "original_body_sha256": "8b1fc9cd00bbcef9d64c86f1b2b7ea7278c980e81ea24fbac7f2d10c0d8f7941",
        "stage": "runner_initialization_failure",
        "applicability_scope": "configuration_schema_inventory_check_only",
        "checks": (
            {
                "check_id": "external_config_schema_missing_key",
                "spans": (
                    'if self.alg_cfg["rnd_cfg"] is not None:',
                    "KeyError: 'rnd_cfg'",
                ),
                "fact_id": "external.runner_config_key_missing_reported",
            },
        ),
    },
    "isaaclab_2513_remote_asset_startup_failure": {
        "source_url": "https://github.com/isaac-sim/IsaacLab/issues/2513",
        "retrieved_on": "2026-11-01",
        "original_body_sha256": "19af897fceb59685f68be204489dd8304c0b014a6b31702c97612f9d6d76c6db",
        "stage": "scene_asset_readiness_failure_before_ppo",
        "applicability_scope": "asset_readiness_check_only",
        "checks": (
            {
                "check_id": "external_asset_readiness_failure",
                "spans": (
                    "Could not open asset @http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd@",
                    "AttributeError: 'NoneType' object has no attribute 'GetPath'",
                ),
                "fact_id": "external.required_scene_asset_unavailable_reported",
            },
        ),
    },
}

MINIMUM_ARTIFACTS = [
    "checkpoint with SHA-256",
    "TensorBoard event export or equivalent scalar telemetry",
    "resolved task and agent configuration",
    "environment/version/seed record",
    "fixed evaluation contract and matching normal reference",
]


def _find_forbidden_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_SELECTOR_KEYS:
                return key
            found = _find_forbidden_key(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_forbidden_key(nested)
            if found is not None:
                return found
    return None


def _require_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ExternalIntakeError(f"필수 문자열이 없습니다: {key}")
    return value


def _validate_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if manifest.get("schema") != SCHEMA:
        raise ExternalIntakeError("지원하지 않는 source manifest schema입니다")
    forbidden = _find_forbidden_key(manifest)
    if forbidden is not None:
        raise ExternalIntakeError(f"후속 답변/진단은 selector input에 넣을 수 없습니다: {forbidden}")

    report_id = _require_string(manifest, "report_id")
    registration = REPORTS.get(report_id)
    if registration is None:
        raise ExternalIntakeError("등록되지 않은 외부 원문입니다")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ExternalIntakeError("source metadata가 없습니다")
    source_url = _require_string(source, "url")
    if source_url != registration["source_url"] or not source_url.startswith("https://github.com/"):
        raise ExternalIntakeError("안전하지 않거나 등록값과 다른 source URL입니다")
    if _require_string(source, "retrieved_on") != registration["retrieved_on"]:
        raise ExternalIntakeError("조회 날짜가 등록값과 다릅니다")
    if _require_string(source, "original_body_sha256") != registration["original_body_sha256"]:
        raise ExternalIntakeError("원문 SHA-256이 등록값과 다릅니다")

    self_reported = manifest.get("self_reported")
    if not isinstance(self_reported, dict):
        raise ExternalIntakeError("self_reported가 없습니다")
    text = _require_string(self_reported, "initial_failure_text")
    if _require_string(self_reported, "initial_failure_text_sha256") != _sha256_text(text):
        raise ExternalIntakeError("self_reported text SHA-256이 일치하지 않습니다")

    artifacts = manifest.get("actual_measured_artifacts")
    if not isinstance(artifacts, list):
        raise ExternalIntakeError("actual_measured_artifacts는 list여야 합니다")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ExternalIntakeError("actual measured artifact 형식이 아닙니다")
        if any(key in artifact for key in ("path", "file_url", "local_path")):
            raise ExternalIntakeError("unsafe artifact path는 허용하지 않습니다")
        _require_string(artifact, "kind")
        _require_string(artifact, "sha256")
    return registration, text


def intake_external_failure(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate a registered public report and return selector-safe facts.

    The result is always HOLD because these three public reports contain no
    caller-owned checkpoint, telemetry export, resolved config, and fixed eval.
    """

    registration, text = _validate_manifest(manifest)
    selector_facts: list[dict[str, Any]] = []
    deterministic_checks: list[dict[str, Any]] = []
    for check in registration["checks"]:
        spans = [_span_fact(text, span) for span in check["spans"]]
        selector_facts.append(
            {
                "fact_id": check["fact_id"],
                "value": True,
                "provenance": "self_reported_public_source",
                "source_span_sha256": [item["text_sha256"] for item in spans],
            }
        )
        deterministic_checks.append(
            {
                "check_id": check["check_id"],
                "outcome": "reported_abnormal",
                "matched_source_spans": spans,
            }
        )

    return {
        "schema": OUTPUT_SCHEMA,
        "report_id": manifest["report_id"],
        "model_diagnosis": False,
        "status": "HOLD",
        "execution_stage": registration["stage"],
        "applicability_scope": registration["applicability_scope"],
        "selector_input": {
            "facts": selector_facts,
            "actual_measured_artifacts": [],
            "later_maintainer_diagnosis_included": False,
        },
        "deterministic_checks": deterministic_checks,
        "required_minimum_artifacts": MINIMUM_ARTIFACTS,
        "hold_reason": "external self-report에는 checkpoint, telemetry, resolved config, environment record, fixed evaluation이 함께 없으므로 repair와 recovery를 허용하지 않습니다",
        "prohibited_actions": ["repair", "resume_training", "recovery_claim"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalIntakeError(f"manifest를 읽을 수 없습니다: {error}") from error
    if not isinstance(value, dict):
        raise ExternalIntakeError("manifest root는 object여야 합니다")
    return value


def _write_once_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ExternalIntakeError("기존 intake 결과를 덮어쓸 수 없습니다")
    if path.suffix != ".json" or ".." in path.parts:
        raise ExternalIntakeError("unsafe output path입니다")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = intake_external_failure(_read_json(args.manifest))
        _write_once_json(args.out, result)
    except ExternalIntakeError as error:
        print(f"HOLD: {error}")
        return 2
    print(json.dumps({"report_id": result["report_id"], "status": result["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
