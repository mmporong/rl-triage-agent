"""B1~B4 개발 비교 상위 사전등록·실행·재검증. GPU와 Git 쓰기는 수행하지 않는다."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import feedback_model_call as F
import nemotron_bottleneck_ledger_v2 as L
import nemotron_bottleneck_n2_v2 as B1
import nemotron_bottleneck_d3_v2 as B2
import nemotron_bottleneck_nat_v2 as B3
import nemotron_upgrade_compare_v1 as V1

PROTOCOL = ROOT / "bench/protocols/nemotron_bottleneck_compare_v2.json"
RUN_ROOT = ROOT / "bench/private/codex_support_20261015/task18/v2/run"
TAG = "nemotron_four_bottlenecks_20261016_v2"
STAGES = ("B1", "B2", "B3", "B4")
SLUGS = {L.BASELINE: "old", L.CANDIDATE: "new"}

FIXED = {
    "schema": "nemotron_four_bottlenecks_protocol_v2", "tag": TAG,
    "v1_disposition": "v1(4288540dfdfdf81c9994979dcc404fd2bc890fda)은 실행 없이 대체됨.",
    "scope": "기존 모델의 병목과 입력을 본 뒤 설계한 개발 측정. 새 seed 성능 평가가 아니다.",
    "models": {"old": L.BASELINE, "new": L.CANDIDATE},
    "api_cap_including_infrastructure_retries_and_feasibility": L.API_CAP,
    "prior_discovery_requests": L.PRIOR_DISCOVERY_REQUESTS,
    "stage_caps": L.STAGE_CAPS,
    "stage_order": list(STAGES), "concurrency": 1,
    "predictions": {
        "B1": "새 모델이 출력 계약 위반을 줄일 수 있으나 단순 보류 증가만으로 통과시키지 않는다.",
        "B2": "새 모델도 줄 번호 근거를 틀릴 수 있다. 원인 1위와 인용 유효성을 분리한다.",
        "B3": "같은 NAT 스키마에서 인자 오류가 줄 수 있으나 2개 입력의 재현 여부를 먼저 확인한다.",
        "B4": "이미지 입력은 받을 것으로 예상한다. 짧은 화면의 관찰과 정책 고정 평가가 일치하지 않을 수 있다.",
    },
    "comparisons": {
        "B1": {"inputs": 11, "repeats": 3, "models": 2, "base_requests": 66,
               "contract": "기존 N2 첫 요청·검증기. 피드백 재요청 없음.",
               "solved": "전체 완료; new invalid<=1/33이고 old보다 적음; 종류별 valid와 mean pass3>=old; mean pass3>=0.8; valid continue>=old; 교정 규칙 불일치<=old",
               "historical": "기존 첫 호출 무효 6/55; 이번 old 33회와 구분"},
        "B2": {"inputs": 8, "repeats": 3, "models": 2, "base_requests": 48,
               "contract": "기존 D3 줄 번호 계약. hunk 계약으로 바꾸지 않음.",
               "solved": "전체 완료; new invalid<=1/24이고 old보다 적음; new 인용 오류=0이고 old보다 적음; 모든 파싱 원인1위와 유효 원인1위 수가 각각 old 이상",
               "historical": "기존 old 계약 무효 8/24와 이번 모델 비교를 구분"},
        "B3": {"inputs": 2, "repeats": 1, "models": 2, "max_requests_before_errors": 48,
               "contract": "s2030 교정 v2, h02_s2027 진단 v3가 재사용한 v2 설정·도구. 모델 ID만 변경.",
               "solved": "4 workflow와 HTTP 계측 완료; 양 모델의 첫 요청은 model 외 동일; old 인자 오류가 하나 이상 재현; new 인자 오류 0, new 제출 2/2 유효, 두 모델 모두 도구 호출 지원 확인",
               "retries": "workflow 재시도 없음. NAT에 원래 있던 제출 전 점검은 같은 계약으로 유지. SDK 재시도 0. HTTP 실패를 기록하고 그 workflow 종료.",
               "unsupported": "새 모델이 도구 입력을 거부하거나 계측이 불완전하면 판정 불가"},
        "B4": {"inputs": 18, "normal": 7, "failed": 11, "repeats": 1,
               "model": L.CANDIDATE, "preflight_max_requests": 3, "graph_requests": 2,
               "contract": "평가 밖 클립의 고정 프레임으로 가능성 확인, 성공 뒤 고정 18개 정책 프레임을 관찰. 회복 결정에는 미사용.",
               "solved": "이미지 가능성 확인 성공; 18/18 유효 관찰; walking이 정상정책 yes·실패정책 no인 대응 >=16/18, 정상 >=6/7 및 실패 >=10/11; unclear는 불일치",
               "capability": "입력 수용 성공은 별도 수치. 고정 평가 라벨은 클립 행동의 정답이 아니며 명령 추종 성능을 화면으로 판정하지 않음.",
               "historical": "기존 모델은 텍스트 입력 모델. Cosmos 호스팅은 과거 404. 동일 프레임 old 추론은 비교하지 않음.",
               "preflight_invalid": "응답을 받았으나 관찰 JSON 계약이 무효면 미해결, 지원·인프라·예산 문제면 판정 불가. 의미 재시도 없음.",
               "graph": "학습 곡선 그림 2회의 형식·경향 읽기 가능성만 기록하며 B4 해결 조건에 미포함"},
    },
    "text_order": "repeat 1..3, fixed case order, model order alternates by (repeat+case_index)%2",
    "retry_rule": "B1/B2 및 B4 평가·그래프는 429/5xx/연결/시간초과/빈 응답/잘림에만 같은 요청 최대1회. B4 가능성은 최대3회 중 인프라일 때만 재시도. 의미 무효는 재시도하지 않음.",
    "budget_rule": "이전에 수행한 /v1/models GET 1회와 단계 상한 합계199를 합쳐200. 다른 단계의 남은 몫을 옮기지 않으며 예약이 응답보다 먼저 기록된다. 중복 label·미완료 예약은 재전송 금지.",
    "source_binding": "실험 커밋이 HEAD의 조상이고 결합 파일 blob이 동일. 입력·메시지·프레임·정답·과거 기록 SHA를 고정.",
    "metrics": ["모델별·입력별 계약 유효/무효", "반복 결정 분포와 두 모델의 유효 결정 일치",
                "D3 인용 오류와 원인1위", "NAT 도구 인자·제출·전송 상태 분리", "영상 정책 라벨 대응표",
                "모든 시도의 토큰(알려진 합계·누락 수), 요청 수, 벽시계 시간"],
    "status_rules": {"solved": "해당 병목의 모든 사전등록 조건을 충족",
                     "unresolved": "비교가 완료됐으나 상한·퇴행 방지 조건을 충족하지 못함",
                     "inconclusive": "호출/계측/입력 미완료 또는 old 병목 미재현. 개선이 없다고 단정하지 않음"},
    "wording": {"solved": "이 개발 입력에서 사전등록한 병목 해결 기준을 충족했다.",
                "unresolved": "이 개발 입력에서 사전등록한 병목 해결 기준을 충족하지 못했다.",
                "inconclusive": "등록 비교를 완료하지 못했거나 기존 병목이 재현되지 않아 해결 여부를 판정하지 않았다.",
                "common": "이미 본 입력을 반복한 개발 측정이다. 모델의 전반적 성능이나 다른 실패에 관한 근거로 확대하지 않는다."},
}


def adapters():
    import nemotron_bottleneck_vision_v2 as B4
    return {"B1": B1, "B2": B2, "B3": B3, "B4": B4}


def source_files():
    paths = {"evals/nemotron_bottleneck_compare_v2.py", "evals/nemotron_bottleneck_ledger_v2.py",
             "tests/test_nemotron_bottleneck_compare_v2.py", "tests/test_nemotron_bottleneck_ledger_v2.py"}
    for adapter in adapters().values():
        paths |= set(adapter.SOURCE_FILES)
    return paths


def source_hash(path):
    data = Path(path).read_bytes()
    if Path(path).suffix.lower() not in {".mp4", ".jpg", ".png", ".gif"}:
        data = V1.lf_bytes(data)
    return V1.sha_bytes(data)


def dynamic():
    modules = adapters()
    inputs = {stage: [module.public_case(c) for c in module.build_cases()] for stage, module in modules.items()}
    inputs["B4_preflight"] = [modules["B4"].public_case(c) for c in modules["B4"].build_preflight_cases()]
    inputs["B4_graph"] = [modules["B4"].public_case(c) for c in modules["B4"].build_graph_cases()]
    return {"source_files": {name: source_hash(ROOT / name) for name in sorted(source_files())},
            "inputs": inputs, "historical": {"B1": B1.historical_summary(), "B2": B2.historical_summary()},
            "prior_models_get_record_sha256": V1.sha_raw(ROOT / "bench/private/codex_support_20261015/task18/models_inventory.json")}


def prepare():
    value = {**FIXED, **dynamic()}
    serialized = json.dumps(value, ensure_ascii=False)
    # source_files에는 공개 저장소 내부의 상대 경로만 있다.
    if re.search(r"[A-Za-z]:[/\\]|/home/|bench/private", serialized):
        raise ValueError("공개 프로토콜에 사용자 또는 비공개 경로가 있다")
    L.write_new(PROTOCOL, value)
    return {"status": "prepared", "protocol": PROTOCOL.relative_to(ROOT).as_posix()}


def verify(commit=None):
    protocol = L.read(PROTOCOL)
    if protocol != {**FIXED, **dynamic()}:
        raise ValueError("사전등록의 고정값·소스·입력 결합이 다르다")
    if commit is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("커밋 SHA 형식")
        if V1.git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode:
            raise ValueError("실험 커밋이 HEAD의 조상이 아니다")
        for name in sorted(source_files() | {PROTOCOL.relative_to(ROOT).as_posix()}):
            blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{name}"], capture_output=True)
            data = blob.stdout if Path(name).suffix.lower() in {".mp4", ".jpg", ".png", ".gif"} else V1.lf_bytes(blob.stdout)
            if blob.returncode or V1.sha_bytes(data) != source_hash(ROOT / name):
                raise ValueError(f"결합 파일 blob이 다르다: {name}")
            if V1.git("diff", "--quiet", "HEAD", "--", name, check=False).returncode:
                raise ValueError(f"결합 파일이 미커밋 상태다: {name}")
    return protocol


def plan(commit):
    verify(commit)
    result = {"schema": "nemotron_four_bottlenecks_plan_v2", "tag": TAG, "experiment_commit": commit,
              "protocol_sha256_lf": V1.sha_lf(PROTOCOL), "created_at": L.now()}
    L.write_new(RUN_ROOT / "plan.json", result)
    return result


def load_plan():
    value = L.read(RUN_ROOT / "plan.json")
    if value.get("tag") != TAG or value.get("protocol_sha256_lf") != V1.sha_lf(PROTOCOL):
        raise ValueError("실행 계획 결합이 다르다")
    verify(value["experiment_commit"])
    return value


def ledger():
    value = load_plan()
    return L.Ledger(RUN_ROOT / "ledger", {"experiment_commit": value["experiment_commit"],
                                         "protocol_sha256_lf": value["protocol_sha256_lf"], "tag": TAG})


@contextmanager
def run_lock():
    path = RUN_ROOT / "run.lock"
    L.write_new(path, {"pid": os.getpid(), "created_at": L.now()})
    try:
        yield
    finally:
        path.unlink()


def entries(stage, cases):
    if stage in ("B1", "B2"):
        for repeat in range(1, 4):
            for index, case in enumerate(cases):
                first = (repeat + index) % 2
                for model in (L.MODELS[first], L.MODELS[1-first]):
                    yield {"id": case["id"], "repeat": repeat, "model": model, "kind": case["kind"]}
    elif stage == "B3":
        for index, case in enumerate(cases):
            for model in (L.MODELS[index % 2], L.MODELS[1-index % 2]):
                yield {"id": case["id"], "repeat": 1, "model": model, "kind": case["kind"]}
    else:
        for case in cases:
            yield {"id": case["id"], "repeat": 1, "model": L.CANDIDATE, "kind": case["kind"]}


def entry_id(entry):
    return f"{entry['id']}_r{entry['repeat']}_{SLUGS[entry['model']]}"


def text_result(book, stage, entry, case, adapter, attempts=2, complete=F.openai_complete, allow_send=True):
    label = entry_id(entry)
    params = {"model": entry["model"], **case["request_params"]}
    request = {"messages": case["messages"], **params}
    raw, state = None, None
    for attempt in range(1, attempts + 1):
        attempt_label = f"{label}_a{attempt}"
        prior = [r for r in book.audit() if r["reservation"]["stage"] == stage and r["reservation"]["label"] == attempt_label]
        if prior:
            record = prior[0]
            if record["reservation"]["request_sha256"] != L.digest(request) or record["response"] is None:
                raise L.BudgetStop("이전 요청 결합 불일치 또는 미완료 예약")
            raw = record["response"]["result"]
        else:
            if not allow_send:
                return {"entry": entry, "status": "budget_or_orphan", "verdict": None}
            try:
                raw = book.call(stage, attempt_label, request, lambda: complete(case["messages"], params))
            except L.BudgetStop:
                return {"entry": entry, "status": "budget_or_orphan", "verdict": None}
        state = L.transport_status(raw, entry["model"])
        if state != "infra_error":
            break
    verdict = adapter.validate_response(case, raw.get("content")) if state == "ok" else None
    return {"entry": entry, "status": ("valid" if verdict["valid"] else "invalid") if verdict else state,
            "verdict": verdict}


def text_stage(stage, module, cases, book):
    results = []
    by_id = {c["id"]: c for c in cases}
    for entry in entries(stage, cases):
        result = text_result(book, stage, entry, by_id[entry["id"]], module)
        results.append(result)
        print(json.dumps({"stage": stage, "id": entry_id(entry), "status": result["status"],
                          "requests": len(book.audit())}, ensure_ascii=False), flush=True)
        if result["status"] in ("call_error", "budget_or_orphan"):
            break
    return results


def nat_stage(cases, book):
    results = []
    by_id = {c["id"]: c for c in cases}
    for entry in entries("B3", cases):
        folder = RUN_ROOT / "nat" / entry_id(entry)
        if (folder / "result.json").exists():
            result = L.read(folder / "result.json")
            for name, sha in result["artifact_sha256"].items():
                if V1.sha_raw(folder / name) != sha:
                    raise ValueError("NAT 산출물 SHA 변경")
        elif folder.exists():
            result = {"valid": False, "error_type": "InterruptedWorkflow", "id": entry["id"], "model": entry["model"]}
        else:
            result = B3.run_case(by_id[entry["id"]], entry["model"], folder, book)
        results.append({"entry": entry, "result": result})
        print(json.dumps({"stage": "B3", "id": entry_id(entry), "valid": result["valid"],
                          "requests": len(book.audit())}), flush=True)
    return results


def vision_stage(module, book):
    preflight = module.build_preflight_cases()[0]
    entry = next(entries("B4", [preflight]))
    feasibility = text_result(book, "B4", entry, preflight, module, attempts=3)
    if feasibility["status"] != "valid":
        return {"feasibility": feasibility, "observations": [], "graphs": []}
    observations = text_stage("B4", module, module.build_cases(), book)
    graphs = text_stage("B4", module, module.build_graph_cases(), book)
    return {"feasibility": feasibility, "observations": observations, "graphs": graphs}


def run(stage):
    book = ledger()
    with run_lock():
        for previous in STAGES[:STAGES.index(stage)]:
            if not (RUN_ROOT / f"{previous}.json").exists():
                raise ValueError("앞 단계 결과를 먼저 완료해야 한다")
        path = RUN_ROOT / f"{stage}.json"
        if path.exists():
            raise FileExistsError("완료한 단계는 재실행하지 않는다")
        module = adapters()[stage]
        result = (vision_stage(module, book) if stage == "B4" else
                  nat_stage(module.build_cases(), book) if stage == "B3" else
                  text_stage(stage, module, module.build_cases(), book))
        L.write_new(path, result)
    return {"stage": stage, "requests_reserved": len(book.audit()), "output": str(path)}


def b1_metrics(rows):
    result = {}
    for model in L.MODELS:
        subset = [r for r in rows if r["entry"]["model"] == model]
        kinds = {}
        for kind in ("diagnosis", "correction"):
            kind_rows = [r for r in subset if r["entry"]["kind"] == kind]
            groups = {}
            for row in kind_rows:
                groups.setdefault(row["entry"]["id"], []).append(row)
            passes = [V1.pass_hat([tuple(r["verdict"]["decision_key"]) if r["status"] == "valid" else None
                                  for r in group], 3) if len(group) == 3 else None for group in groups.values()]
            kinds[kind] = {"valid": sum(r["status"] == "valid" for r in kind_rows),
                           "continue": sum(r["status"] == "valid" and r["verdict"]["decision_key"][0] == "continue" for r in kind_rows),
                           "rule_disagreements": sum(r["status"] == "valid" and r["verdict"].get("agrees_with_rule") is False for r in kind_rows),
                           "mean_pass3": sum(passes) / len(passes) if passes and all(v is not None for v in passes) else None}
        result[model] = {"counts": dict(Counter(r["status"] for r in subset)), "by_kind": kinds}
    return result


def judge_text(stage, rows):
    expected = 66 if stage == "B1" else 48
    complete = len(rows) == expected and all(r["status"] in ("valid", "invalid") for r in rows)
    metrics = b1_metrics(rows) if stage == "B1" else {
        model: {"counts": dict(Counter(r["status"] for r in rows if r["entry"]["model"] == model)),
                **{key: sum(r.get("verdict") is not None and r["verdict"].get(key) is True
                            for r in rows if r["entry"]["model"] == model)
                   for key in ("evidence_citation_error", "true_cause_top1", "valid_true_cause_top1")}}
        for model in L.MODELS}
    old, new = metrics[L.BASELINE], metrics[L.CANDIDATE]
    conditions = {"complete": complete, "new_invalid_at_most_one": new["counts"].get("invalid", 0) <= 1,
                  "invalid_lower": new["counts"].get("invalid", 0) < old["counts"].get("invalid", 0)}
    if stage == "B1":
        conditions.update(kind_valid_not_lower=all(new["by_kind"][k]["valid"] >= old["by_kind"][k]["valid"] for k in ("diagnosis", "correction")),
                          continue_not_lower=new["by_kind"]["diagnosis"]["continue"] >= old["by_kind"]["diagnosis"]["continue"],
                          correction_disagreements_not_higher=new["by_kind"]["correction"]["rule_disagreements"] <= old["by_kind"]["correction"]["rule_disagreements"],
                          pass3_guard=all(new["by_kind"][k]["mean_pass3"] is not None and old["by_kind"][k]["mean_pass3"] is not None
                                          and new["by_kind"][k]["mean_pass3"] >= max(.8, old["by_kind"][k]["mean_pass3"])
                                          for k in ("diagnosis", "correction")))
        reproduced = old["counts"].get("invalid", 0) > 0
    else:
        conditions.update(zero_citation_error=new["evidence_citation_error"] == 0,
                          citation_errors_lower=new["evidence_citation_error"] < old["evidence_citation_error"],
                          all_parsed_top1_not_lower=new["true_cause_top1"] >= old["true_cause_top1"],
                          valid_top1_not_lower=new["valid_true_cause_top1"] >= old["valid_true_cause_top1"])
        reproduced = old["evidence_citation_error"] > 0
    status = "inconclusive" if not complete or not reproduced else "solved" if all(conditions.values()) else "unresolved"
    paired, agree = 0, 0
    groups, by_input = {}, {}
    for row in rows:
        entry = row["entry"]
        groups.setdefault((entry["id"], entry["repeat"]), []).append(row)
        group = by_input.setdefault(entry["id"], {}).setdefault(entry["model"], {"statuses": {}, "decisions": {}})
        group["statuses"][row["status"]] = group["statuses"].get(row["status"], 0) + 1
        if row["status"] == "valid":
            key = json.dumps(row["verdict"]["decision_key"], ensure_ascii=False)
            group["decisions"][key] = group["decisions"].get(key, 0) + 1
    for pair in groups.values():
        if len(pair) == 2 and all(row["status"] == "valid" for row in pair):
            paired += 1
            agree += pair[0]["verdict"]["decision_key"] == pair[1]["verdict"]["decision_key"]
    return {"status": status, "conditions": conditions, "models": metrics, "by_input": by_input,
            "cross_model_decision_agreement": {"matching": agree, "both_valid": paired,
                                                "fraction": agree / paired if paired else None}}


def summarize():
    book = ledger()
    reports = {}
    for stage in STAGES:
        path = RUN_ROOT / f"{stage}.json"
        if not path.exists():
            reports[stage] = {"status": "inconclusive", "reason": "not_run"}
            continue
        rows = L.read(path)
        if stage in ("B1", "B2"):
            module = adapters()[stage]
            review_text_rows(book, stage, rows, module.build_cases(), module)
            reports[stage] = judge_text(stage, rows)
        elif stage == "B3":
            cases = {c["id"]: c for c in B3.build_cases()}
            expected = list(entries("B3", list(cases.values())))
            if [row["entry"] for row in rows] != expected:
                raise ValueError("NAT 실행 순서 또는 입력이 다르다")
            for row in rows:
                folder = RUN_ROOT / "nat" / entry_id(row["entry"])
                if not (folder / "result.json").is_file():
                    raise ValueError("NAT 원본 산출물이 없어 판정하지 않는다")
                reviewed = B3.review_case(cases[row["entry"]["id"]], row["entry"]["model"], folder, book)
                if reviewed != row["result"]:
                    raise ValueError("NAT 기록 재검증 불일치")
            reports[stage] = judge_nat(rows, book.audit())
        else:
            module = adapters()[stage]
            review_text_rows(book, stage, [rows["feasibility"]], module.build_preflight_cases(), module, attempts=3)
            review_text_rows(book, stage, rows["observations"], module.build_cases(), module)
            review_text_rows(book, stage, rows["graphs"], module.build_graph_cases(), module)
            reports[stage] = judge_vision(rows)
    audited = book.audit()
    result = {"schema": "nemotron_four_bottlenecks_result_v2", "protocol_sha256_lf": V1.sha_lf(PROTOCOL),
              "experiment_commit": load_plan()["experiment_commit"], "requests_reserved": len(audited),
              "requests_total_including_discovery": len(audited) + L.PRIOR_DISCOVERY_REQUESTS,
              "orphaned": sum(r["response"] is None for r in audited), "bottlenecks": reports,
              "cost_by_stage_model": {stage: {model: L.costs([r for r in audited if r["reservation"]["stage"] == stage
                                                            and r["reservation"]["model"] == model]) for model in L.MODELS} for stage in STAGES}}
    return result


def review_text_rows(book, stage, rows, cases, module, attempts=2):
    expected = list(entries(stage, cases))
    if len(rows) > len(expected) or [r["entry"] for r in rows] != expected[:len(rows)]:
        raise ValueError("실행 결과의 입력·모델·순서가 사전등록과 다르다")
    by_id = {case["id"]: case for case in cases}
    for row in rows:
        fresh = text_result(book, stage, row["entry"], by_id[row["entry"]["id"]], module,
                            allow_send=False, attempts=attempts)
        if fresh != row:
            raise ValueError("저장 판정과 원시 응답 재검증이 다르다")


def judge_nat(rows, audited):
    models = {}
    for model in L.MODELS:
        selected = [r["result"] for r in rows if r["entry"]["model"] == model]
        calls = [r for r in audited if r["reservation"]["stage"] == "B3" and r["reservation"]["model"] == model]
        models[model] = {"workflows": len(selected), "valid": sum(r["valid"] for r in selected),
                         "tool_errors": sum(r.get("tool_arg_issue_count", 0) for r in selected),
                         "tool_support": any(r["response"] and r["response"]["result"].get("tool_calls") for r in calls),
                         "transport_complete": all(r["response"] and not r["response"]["result"].get("error_type") for r in calls)
                             and all(r.get("trace_complete") is True and r.get("transport_complete") is True
                                     and r.get("metered_complete") is True and
                                     r.get("error_type") not in {"InterruptedWorkflow", "BudgetStop", "TimeoutError"}
                                     for r in selected)}
    pairs = {}
    for row in rows:
        pairs.setdefault(row["entry"]["id"], []).append(row["result"].get("first_request", {}).get("request_without_model_sha256"))
    same = len(pairs) == 2 and all(len(values) == 2 and values[0] is not None and values[0] == values[1] for values in pairs.values())
    complete = len(rows) == 4 and same and all(m["transport_complete"] and m["tool_support"] for m in models.values())
    old, new = models[L.BASELINE], models[L.CANDIDATE]
    conditions = {"complete": complete, "same_initial_request": same, "old_error_reproduced": old["tool_errors"] > 0,
                  "new_no_tool_error": new["tool_errors"] == 0, "new_valid_two": new["valid"] == 2}
    status = "inconclusive" if not complete or not conditions["old_error_reproduced"] else "solved" if all(conditions.values()) else "unresolved"
    return {"status": status, "conditions": conditions, "models": models}


def judge_vision(rows):
    observations = rows["observations"]
    # adapter가 라벨 대응을 계산한다. 화면의 실제 정답으로 취급하지 않는다.
    valid = [r["verdict"] for r in observations if r["status"] == "valid"]
    groups = {label: [v for v in valid if v.get("policy_label") == label] for label in ("normal", "failed")}
    aligned = {label: sum(v.get("label_alignment") is True for v in group) for label, group in groups.items()}
    complete = len(observations) == 18 and all(r["status"] in ("valid", "invalid") for r in observations)
    conditions = {"image_input_accepted": rows["feasibility"]["status"] == "valid", "complete": complete,
                  "all_18_valid": len(valid) == 18, "normal_at_least_6": aligned["normal"] >= 6,
                  "failed_at_least_10": aligned["failed"] >= 10, "total_at_least_16": sum(aligned.values()) >= 16}
    status = ("unresolved" if rows["feasibility"]["status"] == "invalid" else
              "inconclusive" if not complete else "solved" if all(conditions.values()) else "unresolved")
    return {"status": status, "conditions": conditions, "valid": len(valid), "alignment": aligned,
            "observations": observations, "graphs": rows["graphs"], "auxiliary_only": True}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("verify")
    p = sub.add_parser("plan")
    p.add_argument("--commit", required=True)
    p = sub.add_parser("run")
    p.add_argument("--stage", choices=STAGES, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "verify":
        result = {"status": "verified", "schema": verify()["schema"]}
    elif args.command == "plan":
        result = plan(args.commit)
    elif args.command == "run":
        result = run(args.stage)
    else:
        result = summarize()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
