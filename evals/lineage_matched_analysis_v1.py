"""고정 lineage 10-case × 3-policy 공개 결과의 matched-pair 분석기."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_comparison_batch_v1 as BATCH

SCHEMA = "lineage_matched_analysis_v1"
ROW_SCHEMA = "lineage_comparison_public_row_v1"
OBSERVATION_SCHEMA = "lineage_comparison_public_observation_v1"
POLICY_PAIRS = (
    ("model_adaptive", "fixed_eligible"),
    ("model_adaptive", "registry_next"),
    ("fixed_eligible", "registry_next"),
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str):
        raise TypeError(f"{label} 상대경로 형식")
    require(relative != "" and "\\" not in relative, f"{label} 상대경로 형식")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts and bool(pure.parts),
            f"{label} path traversal")
    resolved_root = root.resolve(strict=True)
    candidate = root / Path(*pure.parts)
    require(candidate.exists(), f"{label} 파일이 없다")
    target = candidate.resolve(strict=True)
    require(target == resolved_root or resolved_root in target.parents,
            f"{label} symlink가 export root 밖으로 이탈했다")
    require(target.is_file(), f"{label} 파일이 아니다")
    return target


def _ref(root: Path, item: object, label: str) -> Path:
    require(isinstance(item, dict) and set(item) >= {"path", "sha256"},
            f"{label} ref 형식")
    ref = cast(dict[str, Any], item)
    target = _path(root, ref["path"], label)
    require(isinstance(ref["sha256"], str) and sha(target) == ref["sha256"],
            f"{label} SHA 불일치")
    if "size_bytes" in ref:
        require(ref["size_bytes"] == target.stat().st_size, f"{label} size 불일치")
    return target


def _nim(export_root: Path, entries: object) -> dict[str, Any]:
    if not isinstance(entries, list):
        raise TypeError("NIM raw manifest 형식")
    require(bool(entries), "NIM raw manifest가 없다")
    paths: set[str] = set()
    values: dict[str, dict[str, Any]] = {}
    schema_by_prefix = {
        "nim/session.json": "lineage_nim_transport_session_v1",
        "nim/closed.json": "lineage_nim_transport_closed_v1",
        "nim/worker/start.json": "lineage_nim_transport_worker_start_v1",
        "nim/worker/result.json": "lineage_nim_transport_worker_result_v1",
        "nim/requests": "lineage_nim_transport_request_v1",
        "nim/dispatches": "lineage_nim_transport_dispatch_v1",
        "nim/responses": "lineage_nim_transport_response_v1",
    }
    for index, item in enumerate(entries):
        require(isinstance(item, dict), "NIM ref 행 형식")
        relative = cast(dict[str, Any], item).get("path")
        if not isinstance(relative, str):
            raise TypeError("NIM ref path 형식")
        require(relative not in paths, "NIM ref path 중복")
        paths.add(relative)
        target = _ref(export_root, item, f"NIM[{index}]")
        value = read(target)
        values[relative] = value
        expected = next((schema for prefix, schema in schema_by_prefix.items()
                         if relative == prefix or relative.startswith(prefix + "/")), None)
        if expected is not None:
            require(value.get("schema") == expected, f"{relative} schema 불일치")
        elif relative.startswith("nim/response-sha/"):
            require(set(value) == {"request_id", "response_sha256"},
                    f"{relative} response SHA sidecar 형식")
        else:
            raise ValueError(f"등록되지 않은 NIM raw 경로: {relative}")
    actual = {path.relative_to(export_root).as_posix()
              for path in (export_root / "nim").rglob("*") if path.is_file()}
    require(paths == actual, "NIM manifest와 실제 파일 집합이 다르다")

    requests = {value["request_id"]: (path, value) for path, value in values.items()
                if path.startswith("nim/requests/")}
    dispatches = {value["request_id"]: (path, value) for path, value in values.items()
                  if path.startswith("nim/dispatches/")}
    responses = {value["request_id"]: (path, value) for path, value in values.items()
                 if path.startswith("nim/responses/")}
    sidecars = {value["request_id"]: (path, value) for path, value in values.items()
                if path.startswith("nim/response-sha/")}
    require(set(requests) == set(dispatches) == set(responses) == set(sidecars),
            "NIM request/dispatch/response/sidecar ID 집합 불일치")
    for request_id, (request_path, request) in requests.items():
        dispatch_path, dispatch = dispatches[request_id]
        response_path, response = responses[request_id]
        _sidecar_path, sidecar = sidecars[request_id]
        require(dispatch.get("request_sha256") == sha(export_root / request_path)
                and response.get("request_sha256") == sha(export_root / request_path)
                and response.get("dispatch_sha256") == sha(export_root / dispatch_path)
                and sidecar == {"request_id": request_id,
                                "response_sha256": sha(export_root / response_path)},
                f"NIM {request_id} raw 결합 불일치")
        require(isinstance(request.get("slot_id"), str)
                and request["slot_id"].startswith("case-"), "NIM slot ID 형식")
        require(type(request.get("ordinal")) is int and request["ordinal"] >= 1
                and request.get("request_id", "").startswith(
                    f"{request.get('slot_id')}-{request['ordinal']}-"),
                f"NIM {request_id} case/ordinal/slot 결합 불일치")
    return {"request_count": len(requests), "paths": sorted(paths),
            "requests": [value for _path_name, value in requests.values()]}


def _source_jobs(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require(source.get("schema") == "lineage_source_fixed_development_public_v1"
            and source.get("status") == "complete", "source public summary가 complete가 아니다")
    jobs = source.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("source job 형식")
    require(len(jobs) == 18, "source job은 18개여야 한다")
    by_id: dict[str, dict[str, Any]] = {}
    for item in jobs:
        require(isinstance(item, dict), "source job 형식")
        job = cast(dict[str, Any], item)
        job_id = job.get("job_id")
        if not isinstance(job_id, str):
            raise TypeError("source job ID 형식")
        require(job_id not in by_id, "source job ID 중복")
        require(type(job.get("seed")) is int and job.get("max_iterations") == 300
                and job.get("num_envs") == 4096, "source job seed/training binding")
        checkpoint = job.get("checkpoint")
        require(isinstance(checkpoint, dict)
                and isinstance(checkpoint.get("sha256"), str)
                and isinstance(job.get("telemetry_sha256"), str), "source job CP/telemetry SHA")
        by_id[job_id] = job
    return by_id


def _technical(observation: dict[str, Any]) -> str:
    if observation.get("status") == "complete":
        return "complete"
    if observation.get("evidence_scope") == "parent_exception_without_probe_receipt":
        return "parent_without_receipt"
    execution = observation.get("execution")
    if (observation.get("status") == "unknown"
            and observation.get("error_type") == "PermissionError"
            and observation.get("evidence_scope") == "probe_receipt"
            and isinstance(execution, dict) and execution.get("exit_code") == 0
            and execution.get("natural_exit") is True
            and execution.get("cleanup_verified") is True
            and execution.get("timed_out") is False
            and observation.get("measurement") is None):
        return "post_run_validation_unknown"
    error = observation.get("error_type")
    return "unknown:" + (error if isinstance(error, str) and error else "unspecified")


def _model_failure(row: dict[str, Any]) -> bool:
    validity = row.get("model_validity")
    if not isinstance(validity, list):
        raise TypeError("model_validity 형식")
    return any(item not in {"valid", "not_used"} for item in validity)


def _row(
    export_root: Path, row: dict[str, Any], plan: dict[str, Any], case: dict[str, Any],
    source_jobs: dict[str, dict[str, Any]], observation_paths: set[str],
) -> dict[str, Any]:
    require(row.get("schema") == ROW_SCHEMA, "public row schema 불일치")
    require(row.get("order") == plan["order"] and row.get("case_id") == plan["case_id"]
            and row.get("policy") == plan["policy"] and row.get("seed") == case["seed"]
            and row.get("arm") == case["arm"], "row latin/case/seed/arm 결합 불일치")
    steps, probe_order = row.get("steps"), row.get("probe_order")
    if not isinstance(steps, list) or not isinstance(probe_order, list):
        raise TypeError("row step/probe 형식")
    require(type(row.get("decision_count")) is int and type(row.get("requests")) is int
            and row["decision_count"] == len(steps) == len(probe_order),
            "row decision/step/probe 수 불일치")
    require(0 <= row["requests"] <= row["decision_count"] <= 3,
            "row decision/request budget 불일치")
    if row["policy"] == "model_adaptive":
        require(row["requests"] == row["decision_count"],
                "model row request/decision 수 불일치")
    else:
        require(row["requests"] == 0, "비-model policy에 NIM request가 있다")
    expected_binding = {}
    for role, job_key in (("case", "source_job"), ("reference", "reference_job")):
        job = source_jobs[case[job_key]]
        require(job["seed"] == case["seed"]
                and (job["arm"] == case["arm"] if role == "case" else job["arm"] == "NONE"),
                f"{case['case_id']} source job seed/arm 불일치")
        expected_binding[role] = {
            "checkpoint_sha256": job["checkpoint"]["sha256"],
            "job_id": job["job_id"], "seed": job["seed"],
            "telemetry_sha256": job["telemetry_sha256"],
        }
    require(row.get("source_binding") == expected_binding, "row source CP/telemetry binding 불일치")
    expected_classification = BATCH._classification(case, row)
    require(row.get("classification") == expected_classification, "row classification 재계산 불일치")

    technical: list[str] = []
    registry = BATCH.E.C.S.C1.registry()
    candidates = list(registry["mechanisms"])
    observed: dict[str, str] = {}
    for index, step in enumerate(steps, 1):
        require(not any(outcome == "unknown" for outcome in observed.values()),
                "unknown 관측 뒤 후속 step이 있다")
        require(BATCH.E.C.S.M._status(registry, candidates, observed, 4 - index) == "open"
                and BATCH.E.C.V.eligible(registry, candidates, observed, step.get("probe")),
                "종료·중복·무의미 probe step")
        require(step.get("decision") == index and step.get("probe") == row["probe_order"][index - 1]
                and step.get("outcome") in {"normal", "abnormal", "unknown"},
                "step decision/probe/outcome 결합 불일치")
        observations = step.get("observations")
        require(isinstance(observations, dict) and set(observations) == {"case", "reference"},
                "step paired observation role 불일치")
        pair: dict[str, dict[str, Any]] = {}
        for role in ("case", "reference"):
            ref = observations[role]
            require(isinstance(ref, dict) and ref.get("path") not in observation_paths,
                    "observation ref 중복")
            path = _ref(export_root, ref, f"row {row['order']} observation {role}")
            observation_paths.add(ref["path"])
            observation = read(path)
            pair[role] = observation
            require(observation.get("schema") == OBSERVATION_SCHEMA
                    and observation.get("order") == row["order"]
                    and observation.get("decision") == index
                    and observation.get("role") == role,
                    "observation order/decision/role 결합 불일치")
            category = _technical(observation)
            if category == "parent_without_receipt":
                require(observation.get("status") == "unknown"
                        and observation.get("probe") is None
                        and observation.get("measurement") is None
                        and observation.get("execution") == {}
                        and observation.get("raw_refs") == {},
                        "parent exception은 receipt 없는 unknown stub이어야 한다")
            else:
                require(observation.get("probe") == step["probe"], "observation probe 불일치")
            if observation.get("status") == "complete":
                execution = observation.get("execution")
                measurement = observation.get("measurement")
                require(observation.get("evidence_scope") == "probe_receipt"
                        and isinstance(execution, dict)
                        and execution.get("exit_code") == 0
                        and execution.get("natural_exit") is True
                        and execution.get("cleanup_verified") is True
                        and execution.get("timed_out") is False
                        and isinstance(measurement, dict) and bool(measurement),
                        "complete observation의 성공 receipt/measurement가 유효하지 않다")
            else:
                require(observation.get("status") == "unknown"
                        and observation.get("measurement") is None,
                        "unknown observation 원문 상태 불일치")
            technical.append(category)
        if all(observation["status"] == "complete" for observation in pair.values()):
            recomputed = BATCH.E.C.classify_raw(
                step["probe"], pair["case"]["measurement"], pair["reference"]["measurement"])
        else:
            recomputed = "unknown"
        require(step["outcome"] == recomputed, "step outcome과 raw paired measurement가 다르다")
        candidates, _dropped = BATCH.E.C.S.M.update(
            registry, candidates, step["probe"], recomputed)
        observed[step["probe"]] = recomputed
    if case["arm"] == "NONE":
        require(row["decision_count"] == 0 and row["steps"] == []
                and row["probe_order"] == [] and row.get("end_state") == "no_intervention"
                and row.get("hold_reason") == "host_anomaly_gate",
                "정상 case는 host gate 무개입이어야 한다")
    require(row.get("remaining_candidates") == candidates,
            "row remaining_candidates가 frozen registry 재생과 다르다")
    replay_status = BATCH.E.C.S.M._status(registry, candidates, observed, 3 - len(steps))
    if any(outcome == "unknown" for outcome in observed.values()):
        expected_terminal = ("HOLD", "probe_outcome_unknown", None)
    elif replay_status == "confirmed":
        expected_terminal = ("confirmed", None, candidates[0])
    elif case["arm"] == "NONE":
        expected_terminal = ("no_intervention", "host_anomaly_gate", None)
    else:
        expected_terminal = ("HOLD", "decision_limit", None)
    require((row.get("end_state"), row.get("hold_reason"), row.get("confirmed_subtype"))
            == expected_terminal, "row terminal state/confirmation이 registry 재생과 다르다")
    model_failure = _model_failure(row)
    has_unknown = any(step.get("outcome") == "unknown" for step in row["steps"])
    if expected_classification["correct_confirmation"]:
        score_state = "success"
    elif has_unknown and not model_failure:
        score_state = "unknown"
    else:
        score_state = "known_zero"
    incomplete = sorted({item for item in technical if item != "complete"})
    wall = row.get("wall")
    if not isinstance(wall, dict):
        raise TypeError("row wall 형식")
    return {
        "order": row["order"], "policy": row["policy"], "end_state": row["end_state"],
        "hold_reason": row["hold_reason"], "confirmed_subtype": row["confirmed_subtype"],
        "record_status": "complete", "technical_complete": not incomplete,
        "technical_details": incomplete, "model_failure": model_failure,
        "primary": {"score": 1 if score_state == "success" else 0,
                    "state": score_state,
                    "correct_component_confirmation": expected_classification["correct_confirmation"],
                    "full_cause_confirmed": expected_classification["full_cause_confirmed"],
                    "compound_partial": expected_classification["compound_partial"],
                    "false_confirmation": expected_classification["false_confirmation"]},
        "probe_order": deepcopy(row["probe_order"]), "logical_probes": row["decision_count"],
        "timing": {"probe_parent_s": wall.get("probe_parent_known_s"),
                   "probe_child_s": wall.get("probe_child_known_s"),
                   "probe_child_unknown_roles": deepcopy(wall.get("probe_child_unknown_roles")),
                   "model_s": wall.get("model_wall_s")},
    }


def exact_sign_p(wins: int, losses: int) -> float:
    require(type(wins) is int and type(losses) is int and wins >= 0 and losses >= 0,
            "sign test win/loss는 0 이상 정수여야 한다")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def _binary(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    wins = losses = ties = 0
    for case in rows:
        left_score = case["policies"][left]["primary"]["score"]
        right_score = case["policies"][right]["primary"]["score"]
        wins += left_score > right_score
        losses += left_score < right_score
        ties += left_score == right_score
    denominator = len(rows)
    return {"wins": wins, "losses": losses, "ties": ties, "denominator": denominator,
            "success_difference_count": wins - losses,
            "success_rate_difference": (wins - losses) / denominator if denominator else None,
            "exact_two_sided_sign_p": exact_sign_p(wins, losses)}


def _bounds(cases: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_success = sum(row["policies"][left]["primary"]["score"] for row in cases)
    right_success = sum(row["policies"][right]["primary"]["score"] for row in cases)
    left_unknown = sum(row["policies"][left]["primary"]["state"] == "unknown" for row in cases)
    right_unknown = sum(row["policies"][right]["primary"]["state"] == "unknown" for row in cases)
    denominator = len(cases)
    return {
        "denominator": denominator, "left_known_successes": left_success,
        "right_known_successes": right_success, "left_unknown": left_unknown,
        "right_unknown": right_unknown,
        "minimum_difference": (left_success - right_success - right_unknown) / denominator,
        "maximum_difference": (left_success + left_unknown - right_success) / denominator,
        "interpretation": "unknown을 0 또는 1로 둘 때의 식별 가능 범위이며 일반화 신뢰구간이 아니다",
    }


def _matched_cost(cases: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    eligible = []
    for case in cases:
        left_row, right_row = case["policies"][left], case["policies"][right]
        same = (left_row["technical_complete"] and right_row["technical_complete"]
                and left_row["primary"]["score"] == right_row["primary"]["score"] == 1
                and left_row["confirmed_subtype"] == right_row["confirmed_subtype"])
        if same:
            eligible.append((case["case_id"], left_row, right_row))
    timing = {}
    for name in ("probe_parent_s", "probe_child_s", "model_s"):
        known, missing = [], []
        for case_id, left_row, right_row in eligible:
            left_value, right_value = left_row["timing"][name], right_row["timing"][name]
            if (isinstance(left_value, (int, float)) and not isinstance(left_value, bool)
                    and isinstance(right_value, (int, float)) and not isinstance(right_value, bool)):
                known.append({"case_id": case_id, "left_minus_right": left_value - right_value})
            else:
                missing.append(case_id)
        timing[name] = {"known_differences": known, "missing_case_ids": missing,
                        "mean_difference": (sum(row["left_minus_right"] for row in known) / len(known)
                                            if known else None)}
    probe_deltas = [{"case_id": case_id,
                     "left": left_row["logical_probes"], "right": right_row["logical_probes"],
                     "left_minus_right": left_row["logical_probes"] - right_row["logical_probes"]}
                    for case_id, left_row, right_row in eligible]
    return {"case_ids": [row[0] for row in eligible], "logical_probe_differences": probe_deltas,
            "timing_differences": timing}


def analyze(export_root: Path, source_summary_path: Path, batch_protocol_path: Path) -> dict[str, Any]:
    export_root = Path(export_root).resolve(strict=True)
    source_summary_path = Path(source_summary_path).resolve(strict=True)
    batch_protocol_path = Path(batch_protocol_path).resolve(strict=True)
    manifest_path, summary_path = export_root / "manifest.json", export_root / "summary.json"
    manifest, original_summary = read(manifest_path), read(summary_path)
    source, protocol = read(source_summary_path), read(batch_protocol_path)
    require(manifest.get("schema") == "lineage_comparison_public_export_v1"
            and manifest.get("status") == "complete" and manifest.get("rows") == 30,
            "public export manifest가 complete 30행이 아니다")
    require(original_summary.get("schema") == "lineage_comparison_public_summary_v1"
            and original_summary.get("status") == "complete", "original summary schema/status")
    comparison = original_summary.get("comparison")
    if not isinstance(comparison, dict):
        raise TypeError("original summary comparison 형식")
    require(comparison.get("rows") == 30
            and comparison.get("denominator_case_policy_pairs") == 30,
            "original summary 30행 분모 불일치")
    require(_ref(export_root, manifest.get("summary"), "original summary") == summary_path,
            "manifest summary 경로 불일치")
    require(manifest.get("source_public_summary_sha256") == sha(source_summary_path),
            "source public summary SHA 불일치")
    require(manifest.get("batch_protocol_sha256") == sha(batch_protocol_path)
            and manifest.get("batch_protocol") == batch_protocol_path.relative_to(ROOT).as_posix(),
            "batch protocol path/SHA 불일치")
    require(protocol.get("schema") == BATCH.SCHEMA
            and protocol.get("policies") == list(BATCH.POLICIES)
            and protocol.get("budget", {}).get("max_decisions_per_policy_case") == 3
            and protocol.get("budget", {}).get("execution_request_budget_s") == 1200,
            "batch protocol policy/budget 불일치")
    cases = protocol.get("cases")
    latin = protocol.get("latin_execution_plan")
    if not isinstance(cases, list) or not isinstance(latin, list):
        raise TypeError("batch protocol cases/latin 형식")
    require(len(cases) == 10 and len(latin) == 30,
            "batch protocol은 10 cases × 3 latin 행이어야 한다")
    require([row["order"] for row in latin] == list(range(1, 31)), "latin order 불연속")
    case_by_id = {row["case_id"]: row for row in cases}
    require(len(case_by_id) == 10, "case ID 중복")
    arm_expected = BATCH.validate_arm_expected()
    correction_registry = BATCH.E.C.RC.correction_registry()
    source_jobs = _source_jobs(source)
    nim = _nim(export_root, manifest.get("nim_raw_manifest"))

    row_manifest = manifest.get("rows_manifest")
    if not isinstance(row_manifest, list):
        raise TypeError("row manifest 형식")
    require(len(row_manifest) == 30, "row manifest 30개 필요")
    row_paths: set[str] = set()
    observation_paths: set[str] = set()
    table: dict[str, dict[str, dict[str, Any]]] = {case_id: {} for case_id in case_by_id}
    for index, (item, plan) in enumerate(zip(row_manifest, latin, strict=True), 1):
        if not isinstance(item, dict):
            raise TypeError("row manifest entry 형식")
        require(item.get("path") == f"rows/{index:02d}.json"
                and item["path"] not in row_paths, "row manifest path/order/중복 불일치")
        row_paths.add(item["path"])
        row = read(_ref(export_root, item, f"row {index}"))
        projected = _row(export_root, row, plan, case_by_id[plan["case_id"]],
                         source_jobs, observation_paths)
        require(projected["policy"] not in table[plan["case_id"]], "case-policy row 중복")
        table[plan["case_id"]][projected["policy"]] = projected
    actual_rows = {
        path.relative_to(export_root).as_posix()
        for path in (export_root / "rows").rglob("*") if path.is_file()
    }
    require(row_paths == actual_rows, "row manifest와 실제 row 파일 집합이 다르다")
    require(sum(row["requests"] for row in (read(export_root / f"rows/{i:02}.json")
                                             for i in range(1, 31))) == nim["request_count"],
            "row model requests와 NIM request 수 불일치")
    model_rows = {case_id: rows["model_adaptive"] for case_id, rows in table.items()}
    expected_slots = {
        (case_id, f"{case_id}.model.{decision}")
        for case_id, row in model_rows.items()
        for decision in range(1, row["logical_probes"] + 1)
    }
    actual_slots = {(row.get("case_id"), row.get("slot_id"))
                    for row in nim["requests"]}
    require(actual_slots == expected_slots, "NIM request와 model row case/decision 슬롯 불일치")
    actual_observations = {
        path.relative_to(export_root).as_posix()
        for path in (export_root / "observations").rglob("*") if path.is_file()
    }
    require(observation_paths == actual_observations,
            "row observation ref와 실제 observation 파일 집합이 다르다")
    require(all(set(rows) == set(BATCH.POLICIES) for rows in table.values()),
            "case별 세 policy가 완전하지 않다")

    public_cases = []
    for case in cases:
        rows = table[case["case_id"]]
        bindings = [read(export_root / f"rows/{rows[policy]['order']:02}.json")
                    for policy in BATCH.POLICIES]
        require(len({json.dumps(row["source_binding"], sort_keys=True) for row in bindings}) == 1
                and len({row["view_digest"] for row in bindings}) == 1,
                f"{case['case_id']} 세 정책 source_binding/view_digest 불일치")
        expected = sorted(arm_expected[case["arm"]])
        public_cases.append({
            "case_id": case["case_id"], "seed": case["seed"], "arm": case["arm"],
            "expected_subtypes": expected,
            "correction_supported_subtypes": [name for name in expected if name in correction_registry],
            "out_of_correction_support": bool(expected) and not any(
                name in correction_registry for name in expected),
            "normal_host_no_intervention": case["arm"] == "NONE",
            "policies": {policy: rows[policy] for policy in BATCH.POLICIES},
        })

    by_policy = {}
    for policy in BATCH.POLICIES:
        rows = [case["policies"][policy] for case in public_cases]
        by_policy[policy] = {
            "denominator_cases": 10,
            "correct_component_confirmations": sum(row["primary"]["correct_component_confirmation"] for row in rows),
            "full_cause_confirmed": sum(row["primary"]["full_cause_confirmed"] for row in rows),
            "compound_partial": sum(row["primary"]["compound_partial"] for row in rows),
            "false_confirmations": sum(row["primary"]["false_confirmation"] for row in rows),
            "unknown_primary_rows": sum(row["primary"]["state"] == "unknown" for row in rows),
            "model_failures_known_zero": sum(row["model_failure"] for row in rows),
            "normal_host_no_intervention_cases": sum(case["arm"] == "NONE" for case in public_cases),
        }
        original_by_policy = comparison.get("by_policy")
        if not isinstance(original_by_policy, dict):
            raise TypeError("original summary by_policy 형식")
        if not isinstance(original_by_policy.get(policy), dict):
            raise TypeError(f"original summary {policy} 형식")
        original = original_by_policy[policy]
        require(original.get("denominator_cases") == 10
                and original.get("correct_confirmations")
                == by_policy[policy]["correct_component_confirmations"]
                and original.get("compound_partial") == by_policy[policy]["compound_partial"]
                and original.get("false_confirmations") == by_policy[policy]["false_confirmations"]
                and original.get("unknown") == by_policy[policy]["unknown_primary_rows"],
                f"original summary {policy}와 30행 재계산이 다르다")

    unknown_by_arm = {}
    for arm in sorted({case["arm"] for case in public_cases}):
        arm_cases = [case for case in public_cases if case["arm"] == arm]
        unknown_by_arm[arm] = {
            policy: {
                "case_ids": [case["case_id"] for case in arm_cases
                             if case["policies"][policy]["primary"]["state"] == "unknown"],
                "technical_details": sorted({
                    detail for case in arm_cases
                    for detail in case["policies"][policy]["technical_details"]
                }),
            }
            for policy in BATCH.POLICIES
        }

    pairs = {}
    for left, right in POLICY_PAIRS:
        key = f"{left}_vs_{right}"
        paired = [case for case in public_cases
                  if case["policies"][left]["technical_complete"]
                  and case["policies"][right]["technical_complete"]]
        excluded = [{"case_id": case["case_id"],
                     "reasons": {policy: case["policies"][policy]["technical_details"]
                                 for policy in (left, right)
                                 if not case["policies"][policy]["technical_complete"]}}
                    for case in public_cases if case not in paired]
        pairs[key] = {
            "left": left, "right": right,
            "intention_to_diagnose_all_10": _binary(public_cases, left, right),
            "technical_complete_paired_subset": {
                "case_ids": [case["case_id"] for case in paired],
                "excluded": excluded, "denominator": len(paired),
                "binary": _binary(paired, left, right),
            },
            "unknown_identification_bounds_all_10": _bounds(public_cases, left, right),
            "same_success_subtype_costs": _matched_cost(public_cases, left, right),
        }

    f1 = [case for case in public_cases if case["arm"] == "F1"]
    f1_probe = _matched_cost(f1, "model_adaptive", "fixed_eligible")
    return {
        "schema": SCHEMA, "status": "complete",
        "inputs": {"manifest_sha256": sha(manifest_path),
                   "original_summary_sha256": sha(summary_path),
                   "source_summary_sha256": sha(source_summary_path),
                   "batch_protocol_sha256": sha(batch_protocol_path),
                   "rows": 30, "observations": len(observation_paths),
                   "nim_files": len(nim["paths"])},
        "cases": public_cases, "by_policy": by_policy, "unknown_by_arm": unknown_by_arm,
        "matched_pairs": pairs,
        "posthoc_f1_probe_comparison": {
            "posthoc": True, "case_ids": [case["case_id"] for case in f1],
            "model_vs_fixed": f1_probe,
        },
        "scope": {"intention_to_diagnose_denominator": 10,
                  "normal_host_gate_counted_as_llm_accuracy": False,
                  "out_of_support_unknown_counted_as_success": False,
                  "rows_deleted_or_replaced": 0, "posthoc_analysis": True,
                  "overall_policy_superiority_claim": False,
                  "generalization_claim": False, "gpu_savings_claim": False,
                  "recovery_claim": False},
    }


def write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    require(not path.exists(), "analysis output은 fresh write-once 파일이어야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--batch-protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_new(args.out, analyze(args.export_root, args.source_summary, args.batch_protocol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
