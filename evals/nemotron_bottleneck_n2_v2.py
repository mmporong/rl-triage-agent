"""B1: 저장된 N2 첫 호출 입력과 기존 검증기를 재사용한다."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import nemotron_upgrade_compare_v1 as V1
import feedback_model_call as F
import n2_pass_k as N2

SOURCE_FILES = V1.SOURCE_FILES | {"evals/nemotron_bottleneck_n2_v2.py"}


def build_cases():
    cases = []
    for row in V1.packet_rows():
        data = N2.packet_path(row["set_name"], row["kind"]).read_bytes()
        messages, binding = F.first_messages(row["kind"], data)
        cases.append({"id": row["packet_id"], "bottleneck": "B1", "kind": row["kind"],
                      "messages": messages, "request_params": V1.REQUEST_COMMON,
                      "binding": binding, "packet": json.loads(data)})
    return cases


def public_case(case):
    return {key: case[key] for key in ("id", "bottleneck", "kind", "request_params", "binding")}


def validate_response(case, content):
    verdict = F.validate(case["kind"], content, case["packet"])
    valid = verdict["status"] == "valid"
    decision = verdict.get("decision")
    return {"valid": valid, "errors": verdict["reasons"], "codes": verdict["codes"],
            "decision_key": list(N2.decision_key(case["kind"], decision)) if valid else None,
            "agrees_with_rule": decision.get("agrees_with_rule") if valid and case["kind"] == "correction" else None}


def historical_summary():
    path = ROOT / "evals/results/n2_pass_k_v1/summary.json"
    result = V1.read(path)
    singles = [kind["single"] for case in result["sets"].values() for kind in case["kinds"].values()]
    return {"logical_source": "n2_pass_k_v1", "sha256_lf": V1.sha_lf(path),
            "requests": result["budget"]["requests_used"], "invalid": sum(row["counts"]["invalid"] for row in singles),
            "scope": "피드백 재요청 전의 저장된 개발 측정",
            "source_schema": result.get("schema")}
