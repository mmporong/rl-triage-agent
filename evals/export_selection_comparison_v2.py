"""D4 v2 선택 정책 비교 실행을 v1 검증 경계로 격리해 공개 변환한다."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

FLOW = importlib.import_module("selection_comparison_flow_v2")

TAG = "selection_comparison_s2171_20261023"
RUN = ROOT / "bench/private/claude_reward_track_20261013/selection_comparison/runs" / TAG
PROTOCOL = ROOT / "bench/protocols/selection_comparison_demo_v2.json"
OUTPUT = ROOT / "evals/results" / TAG
AUDIT = ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261023/d4_v2/export/audit"
EXPECTED_ANCHOR = "64fa1241be04ac51d173c18bc7845e4d666102f4"
EXPECTED_PROTOCOL_SHA = "97171edd754f49985c513999a368ade9b5321169a1458368f2dcff7345f9c6b2"
PUBLIC_FILES = {"summary.json", "policies.json", "model_evidence.json", "costs.json", "provenance.json"}


def _isolated_v1() -> ModuleType:
    path = ROOT / "evals/export_selection_comparison_v1.py"
    spec = importlib.util.spec_from_file_location("_selection_comparison_export_v1_for_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("v1 exporter를 격리해 읽을 수 없다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _isolated_v1()
read, serialized, sha256, require = BASE.read, BASE.serialized, BASE.sha256, BASE.require
API, CONTROL, EXECUTION = BASE.API, BASE.CONTROL, BASE.EXECUTION


def _identity(run: Path, protocol_path: Path) -> tuple[dict, dict, dict]:
    run, protocol_path = Path(run).resolve(), Path(protocol_path).resolve()
    result, protocol, begin = read(run / "result.json"), read(protocol_path), read(run / "begin.json")
    require(sha256(protocol_path, True) == EXPECTED_PROTOCOL_SHA, "D4 v2 protocol SHA 불일치")
    retry = protocol.get("infrastructure_retry", {})
    require(protocol.get("schema") == "selection_comparison_demo_protocol_v2"
            and protocol.get("seed") == 2171 and protocol.get("api", {}).get("http_cap") == 60
            and protocol.get("api", {}).get("slot_plan", {}).get("slots") is not None
            and retry.get("prior_seed") == 2170 and retry.get("prior_status") == "invalid"
            and retry.get("consumed") == {"training_runs": 3, "model_calls": 2},
            "D4 v2 protocol identity 불일치")
    require(result.get("schema") == "selection_comparison_result_v2"
            and result.get("tag") == TAG and result.get("seed") == 2171
            and result.get("status") in BASE.STATUSES, "D4 v2 result identity 불일치")
    require(begin == {"tag": TAG, "seed": 2171, "anchor": EXPECTED_ANCHOR,
                      "protocol_sha256_lf": EXPECTED_PROTOCOL_SHA}, "D4 v2 begin 결합 불일치")
    freshness_path = BASE._inside(run, result.get("preflight", {}).get("seed_freshness_path", ""))
    require(freshness_path == run / "seed_freshness.json"
            and sha256(freshness_path) == result["preflight"].get("seed_freshness_sha256"),
            "D4 v2 seed freshness 결합 불일치")
    freshness = read(freshness_path)
    require(freshness.get("status") == "fresh" and freshness.get("seeds") == [2171],
            "D4 v2 seed freshness 내용 불일치")
    return result, protocol, begin


def _configured_base(run: Path, protocol_path: Path) -> ModuleType:
    base = _isolated_v1()
    for name, value in {
        "TAG": TAG, "RUN": Path(run), "PROTOCOL": Path(protocol_path), "OUTPUT": OUTPUT,
        "AUDIT": AUDIT, "EXPECTED_ANCHOR": EXPECTED_ANCHOR,
        "EXPECTED_PROTOCOL_SHA": EXPECTED_PROTOCOL_SHA, "FLOW": FLOW,
    }.items():
        setattr(base, name, value)
    cast(Any, base)._identity = _identity
    return base


def _v2_payloads(payloads: dict[str, dict]) -> dict[str, dict]:
    schemas = {
        "summary.json": "public_selection_comparison_summary_v2",
        "policies.json": "public_selection_comparison_policies_v2",
        "model_evidence.json": "public_selection_comparison_model_evidence_v2",
        "costs.json": "public_selection_comparison_costs_v2",
        "provenance.json": "public_selection_comparison_provenance_v2",
    }
    converted = {name: {**deepcopy(value), "schema": schemas[name]} for name, value in payloads.items()}
    converted["summary.json"]["seed"] = 2171
    converted["provenance.json"]["export_contract"] = (
        "v2_isolated_v1_validation_allowlisted_fields_only_no_private_or_user_paths"
    )
    BASE._public_safe(converted)
    return converted


def _verified_usage(run: Path, protocol: dict) -> dict:
    root = Path(run).resolve() / "model"
    fields = ("input_tokens", "output_tokens", "total_tokens")
    if not root.exists():
        return {"verified_request_count": 0,
                "usage_known_subtotal": {name: None for name in fields},
                "usage_unknown_request_count": {name: 0 for name in fields}}
    plan = API.validate_plan(read(root / "plan.json"))
    require(plan == protocol["api"]["slot_plan"], "D4 v2 usage model plan 불일치")
    audit = API.audit_api(root, plan)
    known: dict[str, list[int]] = {name: [] for name in fields}
    unknown = {name: 0 for name in fields}
    for row in audit["rows"]:
        response = row["response"]
        result = response.get("result", {}) if isinstance(response, dict) else {}
        for name in fields:
            value = result.get(name)
            if type(value) is int and value >= 0:
                known[name].append(value)
            else:
                unknown[name] += 1
    return {"verified_request_count": audit["reserved"],
            "usage_known_subtotal": {name: sum(values) if values else None
                                      for name, values in known.items()},
            "usage_unknown_request_count": unknown}


def verify_and_build(run: Path = RUN, protocol_path: Path = PROTOCOL) -> dict[str, dict]:
    run, protocol_path = Path(run).resolve(), Path(protocol_path).resolve()
    before = {path: sha256(path) for path in (run / "result.json", protocol_path)}
    base = _configured_base(run, protocol_path)
    payloads = base.verify_and_build(run, protocol_path)
    require(before == {path: sha256(path) for path in before}, "D4 v2 검증 중 원본 변경")
    converted = _v2_payloads(payloads)
    usage = _verified_usage(run, read(protocol_path))
    converted["model_evidence.json"].update(usage)
    BASE._public_safe(converted)
    provenance = converted["provenance.json"]
    require(provenance["source_result"]["schema"] == "selection_comparison_result_v2"
            and provenance["source_protocol"] == {
                "schema": "selection_comparison_demo_protocol_v2", "sha256_lf": EXPECTED_PROTOCOL_SHA},
            "D4 v2 공개 provenance schema 결합")
    return converted


def write_new(payloads: dict[str, dict], output: Path = OUTPUT, audit: Path = AUDIT) -> None:
    BASE._public_safe(payloads)
    if output.exists() or audit.exists():
        raise FileExistsError("public output or private export audit already exists")
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in sorted(payloads.items()):
        (output / name).write_bytes(serialized(payload))
    audit.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "selection_comparison_public_export_receipt_v2", "tag": TAG,
               "source_result_sha256": payloads["provenance.json"]["source_result"]["sha256"],
               "public_sha256": {name: sha256(output / name) for name in sorted(payloads)}}
    (audit / "receipt.json").write_bytes(serialized(receipt))


def verify_export(output: Path = OUTPUT, run: Path = RUN, protocol_path: Path = PROTOCOL,
                  audit: Path = AUDIT) -> dict[str, str]:
    expected = verify_and_build(run, protocol_path)
    require(output.is_dir() and {path.name for path in output.iterdir()} == PUBLIC_FILES
            and all(path.is_file() and not path.is_symlink() for path in output.iterdir()),
            "D4 v2 공개 output inventory")
    for name, payload in expected.items():
        require((output / name).read_bytes() == serialized(payload), f"D4 v2 공개 artifact 불일치: {name}")
    hashes = {name: sha256(output / name) for name in sorted(expected)}
    receipt = {"schema": "selection_comparison_public_export_receipt_v2", "tag": TAG,
               "source_result_sha256": expected["provenance.json"]["source_result"]["sha256"],
               "public_sha256": hashes}
    require(audit.is_dir() and [path.name for path in audit.iterdir()] == ["receipt.json"]
            and (audit / "receipt.json").read_bytes() == serialized(receipt),
            "D4 v2 private export receipt 불일치")
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "export", "verify"))
    args = parser.parse_args()
    if args.command == "check":
        result = {"status": "pass", "public_files": sorted(verify_and_build())}
    elif args.command == "export":
        payloads = verify_and_build()
        write_new(payloads)
        result = {"status": "exported", "public_files": sorted(payloads)}
    else:
        result = {"status": "pass", "public_sha256": verify_export()}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# v1 회귀가 검증하는 순수 경계를 v2 테스트에서도 같은 이름으로 노출한다.
_stage_evidence = BASE._stage_evidence
_owned_audit = BASE._owned_audit
_source_evidence = BASE._source_evidence
_reference_evidence = BASE._reference_evidence
_ledger_and_policy_evidence = BASE._ledger_and_policy_evidence
_model_evidence = BASE._model_evidence
