"""PPO ratio repair C4 결과를 격리한 v2 검증 경계와 GPU receipt로 공개 변환한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals", ROOT / "bench"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import export_ppo_ratio_repair_v2 as V2
import export_ppo_ratio_repair_v3 as V3
import ppo_ratio_repair_demo_v4 as DEMO

TAG = "ppo_ratio_repair_s2165_20261023"
RUN = ROOT / "bench/private/claude_reward_track_20261013/ppo_ratio_repair/runs" / TAG
PROTOCOL = ROOT / "bench/protocols/ppo_ratio_repair_demo_v4.json"
OUTPUT = ROOT / "evals/results" / TAG
AUDIT = ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261023/c_v4/export/audit"
EXPECTED_COMMIT = "ae63bfeec9401635da4b195bd504202a226e7e77"
PUBLIC_FILES = set(V2.PUBLIC_FILES)
EXPECTED_ARMS = {
    "NONE": (2165, "training/none", f"{TAG}_none", None),
    DEMO.FAULT: (2165, "training/inverse_ratio", f"{TAG}_inverse_ratio", None),
    "CORRECTED": (3165, "correction_training", f"{TAG}_corrected", "fault"),
}

sha256, read, serialized = V2.sha256, V2.read, V2.serialized


def _isolated_v2() -> ModuleType:
    """v2 exporter를 별도 module로 읽어 V2/V3 전역을 바꾸지 않는다."""
    path = ROOT / "evals/export_ppo_ratio_repair_v2.py"
    spec = importlib.util.spec_from_file_location("_ppo_ratio_repair_export_v2_for_v4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("v2 exporter module을 격리해 읽을 수 없다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configured_base(run: Path, protocol_path: Path) -> ModuleType:
    base = _isolated_v2()
    for name, value in {
        "TAG": TAG,
        "RUN": Path(run),
        "PROTOCOL": Path(protocol_path),
        "OUTPUT": OUTPUT,
        "AUDIT": AUDIT,
        "DEMO": DEMO,
        "CONTRACT": DEMO.C,
        "MODEL_CALL": DEMO.MODEL_CALL,
        "EXPECTED_ARMS": EXPECTED_ARMS,
    }.items():
        setattr(base, name, value)
    original_read = base.read
    result_path = (Path(run) / "result.json").resolve()
    protocol_resolved = Path(protocol_path).resolve()

    def compatible_read(path: Path) -> dict:
        value = original_read(path)
        resolved = Path(path).resolve()
        if resolved == result_path and value.get("schema") == "ppo_ratio_repair_demo_result_v4":
            return {**value, "schema": "ppo_ratio_repair_demo_result_v2"}
        if resolved == protocol_resolved and value.get("schema") == "ppo_ratio_repair_demo_protocol_v4":
            return {**value, "schema": "ppo_ratio_repair_demo_protocol_v2"}
        return value

    cast(Any, base).read = compatible_read
    return base


def _exact_v4_identity(result: dict, protocol: dict, protocol_path: Path) -> None:
    protocol_sha = sha256(protocol_path, True)
    if (
        protocol.get("schema") != "ppo_ratio_repair_demo_protocol_v4"
        or protocol.get("seed") != 2165
        or protocol.get("restart_seed") != 3165
        or protocol.get("model", {}).get("total_http_request_cap") != 4
        or protocol.get("model", {}).get("infra_dialogue_attempts") != 1
        or protocol.get("model", {}).get("feedback_retries") != 1
        or protocol.get("c4_retry", {}).get("previous_commit")
        != "03c04a5c750dd8a0434a7d9dbfa550158e3da1eb"
        or result.get("schema") != "ppo_ratio_repair_demo_result_v4"
        or result.get("tag") != TAG
        or result.get("seed") != 2165
        or result.get("restart_seed") != 3165
        or result.get("experiment_commit") != EXPECTED_COMMIT
        or result.get("protocol_sha256_lf") != protocol_sha
        or result.get("design_coupling") != protocol.get("design_coupling")
        or result.get("outcome") not in protocol.get("outcome_wording", {})
        or result.get("wording") != protocol["outcome_wording"][result["outcome"]]
    ):
        raise ValueError("C4 result/protocol/commit identity differs")


def _v4_payloads(payloads: dict[str, dict], gpu_hashes: dict[str, str],
                 gpu_counts: dict[str, int], result_sha: str, protocol_sha: str) -> dict[str, dict]:
    schemas = {
        "summary.json": "public_ppo_ratio_repair_summary_v4",
        "evaluations.json": "public_ppo_ratio_repair_evaluations_v4",
        "model_evidence.json": "public_ppo_ratio_repair_model_evidence_v4",
        "probe_correction.json": "public_ppo_ratio_repair_probe_correction_v4",
        "provenance.json": "public_ppo_ratio_repair_provenance_v4",
    }
    converted = {name: {**value, "schema": schemas[name]} for name, value in payloads.items()}
    converted["summary.json"]["gpu_admission_evidence"] = {
        "validated_receipts": sum(gpu_counts.values()),
        **gpu_counts,
        "used_for_performance_claim": False,
    }
    provenance = converted["provenance.json"]
    provenance["export_contract"] = "v4_allowlisted_fields_only_no_private_or_user_paths"
    provenance["source_result"] = {
        "schema": "ppo_ratio_repair_demo_result_v4",
        "sha256": result_sha,
    }
    provenance["source_protocol"] = {
        "schema": "ppo_ratio_repair_demo_protocol_v4",
        "sha256_lf": protocol_sha,
    }
    provenance["source_artifact_sha256"] = {
        **provenance["source_artifact_sha256"],
        **gpu_hashes,
    }
    V2.verify_public_payloads(converted)
    return converted


def verify_and_build(run: Path = RUN, protocol_path: Path = PROTOCOL) -> dict[str, dict]:
    run, protocol_path = Path(run), Path(protocol_path)
    result_path = run / "result.json"
    result_sha_before = sha256(result_path)
    protocol_sha_before = sha256(protocol_path)
    original_result, protocol = read(result_path), read(protocol_path)
    _exact_v4_identity(original_result, protocol, protocol_path)
    gpu_hashes, gpu_counts = V3._verify_gpu_receipts(run, original_result)

    base = _configured_base(run, protocol_path)
    payloads = base.verify_and_build(run, protocol_path)

    if (
        sha256(result_path) != result_sha_before
        or sha256(protocol_path) != protocol_sha_before
        or read(result_path).get("schema") != "ppo_ratio_repair_demo_result_v4"
        or read(protocol_path).get("schema") != "ppo_ratio_repair_demo_protocol_v4"
    ):
        raise ValueError("C4 source result/protocol bytes or schema changed during verification")
    model = payloads["model_evidence.json"]
    saved = model.get("api_requests_in_saved_records")
    if type(saved) is not int or saved < 0 or saved > 4:
        raise ValueError("C4 saved API request count exceeds the registered cap")
    for kind, part in original_result.get("model", {}).items():
        attempts = part.get("attempts") if isinstance(part, dict) else None
        if (
            kind not in {"diagnosis", "correction"}
            or not isinstance(attempts, list)
            or len(attempts) != 1
            or attempts[0].get("record") != f"{kind}.feedback1.json"
        ):
            raise ValueError("C4 model kind must use one registered dialogue")
    return _v4_payloads(
        payloads, gpu_hashes, gpu_counts, result_sha_before, sha256(protocol_path, True)
    )


def write_new(payloads: dict[str, dict], output: Path = OUTPUT, audit: Path = AUDIT) -> None:
    V2.verify_public_payloads(payloads)
    if output.exists() or audit.exists():
        raise FileExistsError("public output or private export audit already exists")
    output.mkdir(parents=True, exist_ok=False)
    for relative, payload in sorted(payloads.items()):
        (output / relative).write_bytes(serialized(payload))
    public_hashes = {relative: sha256(output / relative) for relative in sorted(payloads)}
    audit.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "ppo_ratio_repair_public_export_receipt_v4",
        "tag": TAG,
        "source_artifact_sha256": payloads["provenance.json"]["source_artifact_sha256"],
        "public_artifact_sha256": public_hashes,
    }
    (audit / "receipt.json").write_bytes(serialized(receipt))


def verify_export(output: Path = OUTPUT, run: Path = RUN, protocol_path: Path = PROTOCOL,
                  audit: Path = AUDIT) -> dict[str, str]:
    expected = verify_and_build(run, protocol_path)
    entries = list(output.iterdir()) if output.is_dir() else []
    if (
        {path.name for path in entries} != PUBLIC_FILES
        or any(not path.is_file() or path.is_symlink() for path in entries)
        or any(path.is_dir() for path in output.rglob("*"))
    ):
        raise ValueError("exported public artifact inventory differs")
    for relative, payload in expected.items():
        if (output / relative).read_bytes() != serialized(payload):
            raise ValueError(f"exported public artifact differs: {relative}")
    public_hashes = {relative: sha256(output / relative) for relative in sorted(expected)}
    receipt = {
        "schema": "ppo_ratio_repair_public_export_receipt_v4",
        "tag": TAG,
        "source_artifact_sha256": expected["provenance.json"]["source_artifact_sha256"],
        "public_artifact_sha256": public_hashes,
    }
    receipt_path = audit / "receipt.json"
    audit_entries = list(audit.iterdir()) if audit.is_dir() else []
    if (
        [path.name for path in audit_entries] != ["receipt.json"]
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.read_bytes() != serialized(receipt)
    ):
        raise ValueError("private export audit differs")
    return public_hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "export", "verify"))
    args = parser.parse_args()
    if args.command == "check":
        value = {"status": "pass", "public_files": sorted(verify_and_build())}
    elif args.command == "export":
        payloads = verify_and_build()
        write_new(payloads)
        value = {"status": "exported", "public_files": sorted(payloads)}
    else:
        value = {"status": "pass", "public_sha256": verify_export()}
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
