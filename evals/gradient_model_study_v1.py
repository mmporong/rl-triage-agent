"""저장 PPO gradient 기록에서 Nemotron option→receipt 연결을 개발 측정한다.

이미 공개된 seed 2144~2146 기록만 재사용한다. 모델은 검사를 고르고, 확정은
등록된 CPU receipt와 host approval이 만든다. 학습·GPU·교정·회복 권한은 없다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import gradient_hypothesis_contract_v1 as GH
import gradient_registry_action_loop_v1 as LOOP
import nemotron_reasoning_options_study_v1 as MODE
import recorded_ppo_gradient_probe_v1 as REC

PLAN_SCHEMA = "gradient_model_study_plan_v1"
RESULT_SCHEMA = "gradient_model_study_result_v1"
INTENT_SCHEMA = "gradient_model_study_call_intent_v1"
CLAIM_SCHEMA = "gradient_model_study_call_claim_v1"
CALL_SCHEMA = "gradient_model_study_call_v1"
MODEL_MODE = "enabled_low"
API_CAP = 8
TIMEOUT_S = 360
RAW_FILES = REC.RAW_FILES
RUNS_ROOT = Path("evals/results/ppo_gradient_v5_20261017/runs")
CASES = (
    ("g01", "none_s2144_on", "none_s2144_on"),
    ("g02", "inverse_ratio_s2144_on", "none_s2144_on"),
    ("g03", "none_s2145_on", "none_s2144_on"),
    ("g04", "none_s2146_on", "none_s2144_on"),
)
SOURCE_FILES = tuple(dict.fromkeys((
    "evals/gradient_model_study_v1.py",
    "tests/test_gradient_model_study_v1.py",
    "evals/gradient_registry_action_options_v1.py",
    "evals/gradient_registry_action_loop_v1.py",
    "tests/test_gradient_registry_action_loop_v1.py",
    "evals/gradient_hypothesis_contract_v1.py",
    "tests/test_gradient_hypothesis_contract_v1.py",
    "evals/recorded_ppo_gradient_probe_v1.py",
    "tests/test_recorded_ppo_gradient_probe_v1.py",
    "evals/ppo_gradient_invariant_v1.py",
    "tests/test_ppo_gradient_invariant_v1.py",
    "evals/optimizer_observability.py",
    "tests/test_optimizer_observability.py",
    *MODE.SOURCE_FILES,
    *MODE.CALL.SOURCE_FILES,
)))


def require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON 객체가 아니다: {Path(path).name}")
    return value


def _write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value))


def _public_ref(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    require(resolved.is_file() and resolved.is_relative_to(ROOT), "공개 입력 경로가 아니다")
    return {"path": resolved.relative_to(ROOT).as_posix(), "sha256": file_sha(resolved)}


def _case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, case_name, reference_name in CASES:
        endpoints: dict[str, Any] = {}
        for role, name in (("case", case_name), ("reference", reference_name)):
            folder = (ROOT / RUNS_ROOT / name).resolve()
            require(folder.is_dir(), f"{case_id} {role} 입력 폴더가 없다")
            endpoints[role] = {
                "logical_name": name,
                "raw": {item: _public_ref(folder / item) for item in RAW_FILES},
            }
        rows.append({"case_id": case_id, **endpoints})
    return rows


def _mode_params() -> dict[str, Any]:
    request = MODE.build_request([{"role": "user", "content": "<registered>"}], MODEL_MODE)
    return {key: deepcopy(value) for key, value in request.items() if key != "messages"}


def prepare_plan(source_anchor: str) -> dict[str, Any]:
    """결과 전 고정할 네 사례·enabled-low·최대 16회 계획을 만든다."""
    require(re.fullmatch(r"[0-9a-f]{40}", source_anchor) is not None, "anchor 형식")
    for relative in SOURCE_FILES:
        require((ROOT / relative).is_file(), f"source closure 파일이 없다: {relative}")
    return {
        "schema": PLAN_SCHEMA,
        "status": "registered_only_after_commit",
        "source_anchor": source_anchor,
        "purpose": "post_exposure_gradient_option_receipt_integration_development",
        "model": MODE.MODEL,
        "accepted_provider_models": sorted(MODE.CALL.ACCEPTED_PROVIDER_MODELS),
        "mode": MODEL_MODE,
        "request_params": _mode_params(),
        "cases": _case_rows(),
        "bounds": {"cases": 4, "model_decisions_per_case": 2,
                   "actual_calls_per_decision": 1, "format_feedback_calls": 0,
                   "api_requests": API_CAP,
                   "timeout_s": TIMEOUT_S, "gpu_runs": 0},
        "baselines": {
            "count_only": "count_normal_leaves_other_hypotheses_unresolved",
            "count_plus_gradient": "same_recorded_pair_cpu_receipts",
            "other_hypotheses_after_count_normal": "not_evaluated_not_normal",
        },
        "analysis": {
            "development_inputs_already_observed": True,
            "primary_score": "model_selection_and_receipt_integration_not_accuracy_advantage",
            "prediction": "the model may select the registered gradient probe; confirmation is not guaranteed",
            "candidate_rule": {
                "all_four_rows_complete": True,
                "g02_confirmed_subtype": GH.SURROGATE,
                "normal_rows_wrong_confirmations": 0,
            },
            "reporting": {
                "rule_pass": "confirmed_development_integration_candidate",
                "rule_fail": "not_confirmed",
                "provider_or_incomplete": "indeterminate",
            },
        },
        "claims": {
            "unseen": False, "new_observation": False, "new_training": False,
            "gpu_execution": False, "repair": False, "recovery": False,
            "improvement": False, "llm_superiority": False,
        },
        "source_sha256_lf": {name: lf_sha(ROOT / name) for name in SOURCE_FILES},
    }


def verify_plan(plan_path: Path) -> dict[str, Any]:
    value = _read(plan_path)
    require(value.get("schema") == PLAN_SCHEMA, "gradient model plan schema 불일치")
    require(value == prepare_plan(value.get("source_anchor", "")),
            "plan이 현재 source/raw 입력과 다르다")
    return value


def _git(*args: str) -> bytes:
    run = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    require(run.returncode == 0, f"git {' '.join(args)} 실패")
    return run.stdout


def verify_registered(plan_path: Path, anchor: str) -> dict[str, Any]:
    path = Path(plan_path).resolve()
    require(path.is_relative_to(ROOT / "bench" / "protocols") and path.suffix == ".json",
            "plan은 공개 protocol JSON이어야 한다")
    plan = verify_plan(path)
    source_anchor = plan["source_anchor"]
    require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "protocol anchor 형식")
    _git("cat-file", "-e", f"{source_anchor}^{{commit}}")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", source_anchor, anchor)
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    relative = path.relative_to(ROOT).as_posix()
    require(_git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
            == path.read_bytes().replace(b"\r\n", b"\n"), "anchor plan blob 불일치")
    for source, expected in plan["source_sha256_lf"].items():
        require(hashlib.sha256(_git("show", f"{source_anchor}:{source}").replace(
            b"\r\n", b"\n")).hexdigest() == expected,
            f"source anchor blob 불일치: {source}")
        require(hashlib.sha256(_git("show", f"{anchor}:{source}").replace(b"\r\n", b"\n")).hexdigest()
                == expected, f"anchor source blob 불일치: {source}")
        require(lf_sha(ROOT / source) == expected, f"현재 source 변경: {source}")
    return plan


def _require_wsl() -> None:
    require(sys.platform.startswith("linux") and "microsoft" in Path("/proc/version").read_text(
        encoding="utf-8", errors="ignore").lower(), "실제 실행은 WSL 단일 Python에서만 허용한다")


def _prepare_pair(row: dict[str, Any], out: Path) -> dict[str, Any]:
    endpoints: dict[str, dict[str, str]] = {}
    for role in ("case", "reference"):
        folder = ROOT / Path(row[role]["raw"]["manifest.json"]["path"]).parent
        plan_path = out / role / "plan.json"
        receipt_path = out / role / "receipt.json"
        REC.prepare(folder, request_id=f"{row['case_id']}.{role}", out=plan_path)
        REC.execute(plan_path, out=receipt_path)
        receipt = REC.checked_receipt(plan_path, receipt_path)
        endpoints[role] = {"role": role, "plan": str(plan_path), "receipt": str(receipt_path)}
        require(receipt["source_binding"]["raw_source_sha256"]
                == {name: row[role]["raw"][name]["sha256"] for name in RAW_FILES},
                f"{row['case_id']} {role} raw SHA 불일치")
    return GH.initialize({"schema": GH.PAIR_SCHEMA, "subtype_schema": GH.SUBTYPE_SCHEMA,
                          "case": endpoints["case"], "reference": endpoints["reference"]}, 2)


def _complete_factory(call_root: Path, plan: dict[str, Any], verify_context: Callable[[], object],
                      complete: Callable[[dict[str, Any]], dict[str, Any]], counter: dict[str, int]
                      ) -> Callable[[str, dict], Callable]:
    def factory(slot: str, state: dict) -> Callable:
        require(re.fullmatch(r"g0[1-4]\.gradient_model\.d[12]", slot) is not None,
                "등록되지 않은 model slot")

        def invoke(messages: list[dict[str, str]], supplied_params: dict[str, Any]) -> dict[str, Any]:
            require(supplied_params == MODE.CALL.F.request_params(MODE.CALL.MODEL),
                    "option dialogue 기본 params 불일치")
            counter["calls"] += 1
            require(counter["calls"] <= plan["bounds"]["api_requests"], "API cap 초과")
            request = MODE.build_request(messages, MODEL_MODE)
            require({key: value for key, value in request.items() if key != "messages"}
                    == plan["request_params"], "enabled-low params 불일치")
            attempt = call_root / f"{counter['calls']:02d}-{slot}"
            intent = {"schema": INTENT_SCHEMA, "index": counter["calls"], "slot": slot,
                      "state_digest": state["state_digest"], "messages_sha256": digest(messages),
                      "request": request, "protocol_sha256": counter["protocol_sha256"]}
            _write_new(attempt / "intent.json", intent)
            _write_new(attempt / "claim.json", {
                "schema": CLAIM_SCHEMA, "index": counter["calls"], "slot": slot,
                "intent_sha256": file_sha(attempt / "intent.json"), "exclusive": True,
            })
            verify_context()
            started = time.perf_counter()
            result = complete(deepcopy(request))
            wall_s = time.perf_counter() - started
            verify_context()
            require(isinstance(result, dict), "provider result 형식")
            record = {"schema": CALL_SCHEMA, "index": counter["calls"], "slot": slot,
                      "intent": {"path": str(attempt / "intent.json"),
                                 "sha256": file_sha(attempt / "intent.json")},
                      "claim": {"path": str(attempt / "claim.json"),
                                "sha256": file_sha(attempt / "claim.json")},
                      "response": deepcopy(result), "wall_s": wall_s,
                      "automatic_retry": False}
            _write_new(attempt / "result.json", record)
            return deepcopy(result)
        return invoke
    return factory


def _baseline(state: dict[str, Any]) -> dict[str, Any]:
    source = state["source_pair_binding"]
    paths = tuple(Path(source[role][kind]["path"])
                  for role in ("case", "reference") for kind in ("plan", "receipt"))
    after_count = GH.advance_count(state, *paths)
    after_both = (GH.advance_gradient(after_count, *paths)
                  if after_count["status"] == "open" else after_count)
    return {
        "count_only": {"outcome": "unresolved" if after_count["status"] != "confirmed"
                       else "confirmed", "confirmed_subtype": after_count["conclusion"],
                       "other_hypotheses_normal": False},
        "count_plus_gradient": {"outcome": ("confirmed" if after_both["status"] == "confirmed"
                                             else "unresolved"),
                                "confirmed_subtype": after_both["conclusion"]},
    }


def _aggregate(rows: list[dict[str, Any]]) -> str:
    if (len(rows) != 4 or any(row["status"] != "complete" for row in rows)
            or any(not row["model"]["all_model_calls_valid"] for row in rows)):
        return "indeterminate"
    g02 = next(row for row in rows if row["case_id"] == "g02")
    normals = [row for row in rows if row["case_id"] != "g02"]
    passed = (g02["model"]["decision_outcome"] == "confirmed"
              and g02["model"]["confirmed_subtype"] == GH.SURROGATE
              and all(row["model"]["decision_outcome"] != "confirmed" for row in normals))
    return "confirmed_development_integration_candidate" if passed else "not_confirmed"


def execute(plan_path: Path, anchor: str, out: Path, *,
            complete: Callable[[dict[str, Any]], dict[str, Any]] = MODE.openai_complete,
            verify_registered_fn: Callable[[Path, str], dict[str, Any]] = verify_registered,
            require_platform: Callable[[], None] = _require_wsl) -> dict[str, Any]:
    """등록 plan의 네 행을 한 번씩 소비한다. provider 오류에는 재시도하지 않는다."""
    require_platform()
    plan = verify_registered_fn(Path(plan_path).resolve(), anchor)
    out = Path(out).resolve()
    require(not out.exists(), "실행 out은 fresh 경로여야 한다")
    out.mkdir(parents=True, exist_ok=False)
    counter: dict[str, Any] = {"calls": 0, "protocol_sha256": file_sha(plan_path)}
    rows: list[dict[str, Any]] = []

    def verify_context() -> object:
        require(verify_registered_fn(Path(plan_path).resolve(), anchor) == plan,
                "실행 중 plan/source/raw 변경")
        return True

    factory = _complete_factory(out / "calls", plan, verify_context, complete, counter)
    try:
        for row in plan["cases"]:
            state = _prepare_pair(row, out / "sources" / row["case_id"])
            baseline = _baseline(state)
            epoch = LOOP.run_epoch(
                state, run_id=f"{row['case_id']}.gradient_model",
                out=out / "epochs" / row["case_id"], verify_context=verify_context,
                complete_factory=factory, consume_pair=LOOP.consume_immutable_pair,
            )
            rows.append({"case_id": row["case_id"], "status": "complete",
                         "model": {"decision_outcome": epoch["decision_outcome"],
                                   "confirmed_subtype": epoch["confirmed_subtype"],
                                   "model_actions": epoch["model_actions"],
                                   "recorded_pair_consumptions": epoch["recorded_pair_consumptions"],
                                   "all_model_calls_valid": all(
                                       step["model_valid"] for step in epoch["steps"])},
                         "rules": baseline,
                         "epoch": {"path": str(out / "epochs" / row["case_id"] / "result.json"),
                                   "sha256": file_sha(out / "epochs" / row["case_id"] / "result.json")}})
        outcome = _aggregate(rows)
        result = {"schema": RESULT_SCHEMA, "status": "complete", "outcome": outcome,
                  "api_requests": counter["calls"], "rows": rows,
                  "claims": deepcopy(plan["claims"]),
                  "scope": {"model_selects_probe": True, "host_confirms_from_receipt": True,
                            "reasoning_body_public": False, "automatic_retry": False}}
        _write_new(out / "result.json", result)
        return deepcopy(result)
    except BaseException as error:
        _write_new(out / "failure.json", {"schema": RESULT_SCHEMA, "status": "invalid",
                                            "error_type": type(error).__name__,
                                            "api_requests": counter["calls"],
                                            "automatic_retry": False})
        raise


def _cli() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-plan")
    prepare.add_argument("--source-anchor", required=True)
    prepare.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify-plan")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--anchor", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--anchor", required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-plan":
        _write_new(args.out, prepare_plan(args.source_anchor))
    elif args.command == "verify-plan":
        verify_registered(args.plan, args.anchor)
    else:
        execute(args.plan, args.anchor, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = ["CASES", "PLAN_SCHEMA", "RESULT_SCHEMA", "SOURCE_FILES", "execute",
           "prepare_plan", "verify_plan", "verify_registered"]
