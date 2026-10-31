from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "evals" / "external_failure_intake_v1.py"
SPEC = importlib.util.spec_from_file_location("external_failure_intake_v1", MODULE_PATH)
assert SPEC and SPEC.loader
INTAKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INTAKE)


TEXTS = {
    "rsl_rl_33_actor_std_nan": """num_env is 3600
RuntimeError: normal expects all elements of std >= 0.0
env_id: 1111, row: tensor([nan, nan, nan, nan])
""",
    "isaaclab_1847_rnd_cfg_keyerror": """if self.alg_cfg[\"rnd_cfg\"] is not None:
KeyError: 'rnd_cfg'
""",
    "isaaclab_2513_remote_asset_startup_failure": """Could not open asset @http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Grid/default_environment.usd@
AttributeError: 'NoneType' object has no attribute 'GetPath'
""",
}


def manifest_for(report_id: str) -> dict[str, object]:
    registration = INTAKE.REPORTS[report_id]
    text = TEXTS[report_id]
    return {
        "schema": INTAKE.SCHEMA,
        "report_id": report_id,
        "source": {
            "url": registration["source_url"],
            "retrieved_on": registration["retrieved_on"],
            "original_body_sha256": registration["original_body_sha256"],
        },
        "self_reported": {
            "initial_failure_text": text,
            "initial_failure_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        "actual_measured_artifacts": [],
    }


@pytest.mark.parametrize("report_id", sorted(TEXTS))
def test_registered_public_reports_produce_hold_only(report_id: str) -> None:
    result = INTAKE.intake_external_failure(manifest_for(report_id))

    assert result["status"] == "HOLD"
    assert result["model_diagnosis"] is False
    assert result["selector_input"]["later_maintainer_diagnosis_included"] is False
    assert result["selector_input"]["actual_measured_artifacts"] == []
    assert result["prohibited_actions"] == ["repair", "resume_training", "recovery_claim"]
    assert len(result["deterministic_checks"]) == 1


@pytest.mark.parametrize("report_id", sorted(TEXTS))
def test_unique_registered_source_span_is_required(report_id: str) -> None:
    manifest = manifest_for(report_id)
    text = manifest["self_reported"]["initial_failure_text"]
    manifest["self_reported"]["initial_failure_text"] = text.replace("Error", "error", 1)
    manifest["self_reported"]["initial_failure_text_sha256"] = hashlib.sha256(
        manifest["self_reported"]["initial_failure_text"].encode("utf-8")
    ).hexdigest()

    with pytest.raises(INTAKE.ExternalIntakeError, match="source span"):
        INTAKE.intake_external_failure(manifest)


def test_rejects_original_source_sha_tamper() -> None:
    manifest = manifest_for("rsl_rl_33_actor_std_nan")
    manifest["source"]["original_body_sha256"] = "0" * 64

    with pytest.raises(INTAKE.ExternalIntakeError, match="SHA-256"):
        INTAKE.intake_external_failure(manifest)


@pytest.mark.parametrize("key", ["later_maintainer_diagnosis", "comments", "resolution"])
def test_rejects_later_answer_leakage(key: str) -> None:
    manifest = manifest_for("isaaclab_1847_rnd_cfg_keyerror")
    manifest[key] = {"label": "must not be consumed"}

    with pytest.raises(INTAKE.ExternalIntakeError, match="후속 답변"):
        INTAKE.intake_external_failure(manifest)


def test_rejects_unregistered_source_and_unsafe_artifact_path() -> None:
    manifest = manifest_for("isaaclab_2513_remote_asset_startup_failure")
    manifest["source"]["url"] = "file:///not-an-official-report"
    with pytest.raises(INTAKE.ExternalIntakeError, match="source URL"):
        INTAKE.intake_external_failure(manifest)

    manifest = manifest_for("isaaclab_2513_remote_asset_startup_failure")
    manifest["actual_measured_artifacts"] = [{"kind": "checkpoint", "sha256": "a" * 64, "path": "../unsafe"}]
    with pytest.raises(INTAKE.ExternalIntakeError, match="unsafe artifact path"):
        INTAKE.intake_external_failure(manifest)


def test_cli_writes_once_and_refuses_overwrite(tmp_path: Path) -> None:
    manifest_path = tmp_path / "source.json"
    result_path = tmp_path / "result.json"
    manifest_path.write_text(json.dumps(manifest_for("rsl_rl_33_actor_std_nan")), encoding="utf-8")

    assert INTAKE.main(["--manifest", str(manifest_path), "--out", str(result_path)]) == 0
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "HOLD"
    assert INTAKE.main(["--manifest", str(manifest_path), "--out", str(result_path)]) == 2
