"""요청 endpoint와 관측 반환 ID를 구분하는 네 병목 비교. v2 기록·예산을 계승한다."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))
import nemotron_bottleneck_compare_v2 as C
import nemotron_bottleneck_ledger_v2 as L
import nemotron_endpoint_ledger_v3 as P

PROTOCOL = ROOT / "bench/protocols/nemotron_endpoint_compare_v3.json"
RUN_ROOT = ROOT / "bench/private/codex_support_20261015/task18/v3/run"
PARENT_COMMIT = "c7fb5668c94b4d0e16202fea664107bd822a0f07"
TAG = "nemotron_endpoint_four_bottlenecks_20261016_v3"
PREFLIGHT = ROOT / "bench/private/codex_support_20261015/task18/v2/image_identity_preflight/result.json"

FIXED = {
    "schema": "nemotron_endpoint_comparison_protocol_v3", "tag": TAG,
    "parent_commit": PARENT_COMMIT,
    "v2_disposition": "v2 첫 생성 응답은 요청 ID와 반환 ID가 달라 call_error로 보존했다. 계약 채점에 넣지 않으며 같은 응답을 소급 채점하지 않는다.",
    "design_timing": "v2의 첫 응답과 추가 합성 이미지 1회 확인 뒤 반환 ID 계약을 마련했다. 새 평가 호출 전에 커밋한다. 기존 결과를 본 개발 측정이다.",
    "comparison_unit": "NVIDIA 공식 API의 요청 모델 경로. 반환 모델 ID 원문을 함께 기록하며 제공자 내부 가중치의 동일성을 확정하지 않는다.",
    "response_identity": {"requested_old": L.BASELINE, "requested_new": L.CANDIDATE,
                          "candidate_observed_response_id": "private/nvidia/nemotron-3.5-super-120b-a12b",
                          "official_alias_mapping_confirmed": False,
                          "policy": "새 요청에는 요청 ID와 관측 반환 ID 두 개만 허용. old 요청에는 old ID만 허용. 다른 ID는 call_error. 원문 원장을 수정하지 않는다."},
    "budget": {"overall_cap": 200, "prior_models_get": 1, "prior_v2_generation": 1,
               "additional_synthetic_image": 1, "remaining_generation_max": 197,
               "shared_stage_caps": L.STAGE_CAPS,
               "rule": "동일 v2 원장을 이어 사용하고 v3:: prefix로 구분. 기존 B1 1회·B4 1회도 단계 예산에서 계속 차감. 예산 초기화 없음."},
    "image_feasibility": {"synthetic_objects": 3, "matched": 3, "evaluation_clip": False,
                          "meaning": "작은 합성 이미지의 형태·색·위치 관찰 성공. 실제 정책 영상의 관찰 성능 근거는 아님.",
                          "B4_enabled": True},
    "comparison_rules": C.FIXED,
    "adoption": "B1~B4 결과는 요청 경로의 개발 비교로 한정한다. 반환 ID의 공식 대응은 미확정이며 기존 기본 모델은 자동 변경하지 않는다.",
}


def parent_book():
    plan = L.read(C.RUN_ROOT / "plan.json")
    if (plan.get("experiment_commit") != PARENT_COMMIT or plan.get("tag") != C.TAG
            or plan.get("protocol_sha256_lf") != C.V1.sha_lf(C.PROTOCOL)):
        raise ValueError("이전 원장의 실험 결합이 다르다")
    return L.Ledger(C.RUN_ROOT / "ledger", {"experiment_commit": plan["experiment_commit"],
                    "protocol_sha256_lf": plan["protocol_sha256_lf"], "tag": plan["tag"]})


def source_files():
    return C.source_files() | {
        "bench/protocols/nemotron_bottleneck_compare_v2.json",
        "evals/nemotron_endpoint_compare_v3.py", "evals/nemotron_endpoint_ledger_v3.py",
        "tests/test_nemotron_endpoint_compare_v3.py", "tests/test_nemotron_endpoint_ledger_v3.py"}


def allowed_reservations():
    allowed = set()
    for stage, module in C.adapters().items():
        groups = [(module.build_cases(), 2)]
        if stage == "B4":
            groups += [(module.build_preflight_cases(), 3), (module.build_graph_cases(), 2)]
        for cases, attempts in groups:
            for entry in C.entries(stage, cases):
                labels = ([f"{entry['id']}_{entry['model'].split('/')[-1]}_http{i}" for i in range(1, 13)]
                          if stage == "B3" else [f"{C.entry_id(entry)}_a{i}" for i in range(1, attempts+1)])
                allowed.update((stage, P.PHASE_PREFIX + label, entry["model"]) for label in labels)
    return allowed


def validate_parent_rows(rows):
    prefix = rows[:2]
    if (len(prefix) != 2 or [r["reservation"]["stage"] for r in prefix] != ["B1", "B4"]
            or prefix[1]["reservation"]["label"] != "supplemental_image_identity_check_v1"
            or any(r["response"] is None for r in prefix)):
        raise ValueError("이전 요청 두 개의 원장 결합이 다르다")
    allowed = allowed_reservations()
    for row in rows[2:]:
        reservation = row["reservation"]
        if (reservation["stage"], reservation["label"], reservation["model"]) not in allowed:
            raise ValueError("v3 등록 일정 밖의 요청이 원장에 있다")
    return {"parent_prefix_sha256": L.digest(prefix), "parent_generation_count_at_design": 2,
            "parent_tail_reservation_sha256": L.digest(prefix[-1]["reservation"])}


def dynamic():
    inherited = C.dynamic()
    rows = parent_book().audit()
    parent_snapshot = validate_parent_rows(rows)
    feasibility = L.read(PREFLIGHT)
    if feasibility.get("matches_three_objects") is not True or feasibility.get("total_requests_including_models_get") != 3:
        raise ValueError("이미지 가능성 확인 조건이 다르다")
    return {"source_files": {name: C.source_hash(ROOT / name) for name in sorted(source_files())},
            "inputs": inherited["inputs"], "historical": inherited["historical"],
            **parent_snapshot, "synthetic_image_result_sha256": C.V1.sha_raw(PREFLIGHT),
            "synthetic_image_definition_sha256": C.V1.sha_raw(PREFLIGHT.with_name("definition.json")),
            "synthetic_image_sha256": C.V1.sha_raw(PREFLIGHT.with_name("synthetic.png"))}


def prepare():
    protocol = {**FIXED, **dynamic()}
    if re.search(r"[A-Za-z]:[/\\]|/home/|bench/private", json.dumps(protocol, ensure_ascii=False)):
        raise ValueError("공개 프로토콜 경로 검사 실패")
    L.write_new(PROTOCOL, protocol)
    return {"status": "prepared", "generation_remaining": 197}


def verify(commit=None):
    C.verify()  # 부모 판의 소스·입력·의미도 유지한다.
    if L.read(PROTOCOL) != {**FIXED, **dynamic()}:
        raise ValueError("v3 소스·입력·기존 소비 결합이 다르다")
    if commit is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or C.V1.git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode:
            raise ValueError("실험 커밋이 HEAD 조상이 아니다")
        paths = source_files() | {PROTOCOL.relative_to(ROOT).as_posix()}
        tracked = set(C.V1.git("ls-tree", "-r", "--name-only", commit).stdout.splitlines())
        if not paths <= tracked:
            raise ValueError("실험 커밋에 결합 파일이 없다")
        # 한 번의 git 비교로 commit/index/worktree의 모든 결합 blob을 검사한다.
        if C.V1.git("diff", "--quiet", commit, "--", *sorted(paths), check=False).returncode:
            raise ValueError("실험 커밋과 현재 결합 파일 blob이 다르다")
    return {"status": "verified", "parent_requests": 3, "remaining_max": 197}


def plan(commit):
    with run_lock():
        verify(commit)
        if len(parent_book().audit()) != 2:
            raise ValueError("v3 실행 계획 전에 추가 요청이 존재한다")
        value = {"schema": "nemotron_endpoint_comparison_plan_v3", "tag": TAG, "experiment_commit": commit,
                 "protocol_sha256_lf": C.V1.sha_lf(PROTOCOL), "parent_prefix_sha256": dynamic()["parent_prefix_sha256"],
                 "created_at": L.now()}
        L.write_new(RUN_ROOT / "plan.json", value)
    return value


def load_plan():
    value = L.read(RUN_ROOT / "plan.json")
    if value.get("tag") != TAG or value.get("protocol_sha256_lf") != C.V1.sha_lf(PROTOCOL):
        raise ValueError("v3 실행 계획 결합 불일치")
    verify(value["experiment_commit"])
    if value.get("parent_prefix_sha256") != dynamic()["parent_prefix_sha256"]:
        raise ValueError("v3 실행 계획의 부모 원장 결합 불일치")
    return value


def phase_book():
    load_plan()
    return P.PhaseLedger(parent_book())


@contextmanager
def run_lock():
    # v2와 v3가 같은 요청 원장을 쓰므로 실행 lock도 부모의 같은 파일을 사용한다.
    with C.run_lock():
        yield


def run(stage):
    with run_lock():
        book = phase_book()
        for earlier in C.STAGES[:C.STAGES.index(stage)]:
            if not (RUN_ROOT / f"{earlier}.json").is_file():
                raise ValueError("이전 단계가 완료되지 않았다")
        path = RUN_ROOT / f"{stage}.json"
        if path.exists():
            raise FileExistsError("완료 결과는 덮어쓰지 않는다")
        module = C.adapters()[stage]
        if stage in ("B1", "B2"):
            result = C.text_stage(stage, module, module.build_cases(), book)
        elif stage == "B4":
            result = C.vision_stage(module, book)
        else:
            result = []
            cases = {c["id"]: c for c in module.build_cases()}
            for entry in C.entries(stage, list(cases.values())):
                folder = RUN_ROOT / "nat" / C.entry_id(entry)
                if (folder / "result.json").exists():
                    value = module.review_case(cases[entry["id"]], entry["model"], folder, book)
                elif folder.exists():
                    raise ValueError("NAT 미완료 실행은 재전송하지 않는다")
                else:
                    value = module.run_case(cases[entry["id"]], entry["model"], folder, book)
                result.append({"entry": entry, "result": value})
                print(json.dumps({"stage": stage, "id": C.entry_id(entry), "valid": value["valid"],
                                  "global_requests": len(parent_book().audit())+1}), flush=True)
        L.write_new(path, result)
        counts = {"stage": stage, "global_requests_including_get": len(parent_book().audit())+1,
                  "phase_generation_requests": len(book.audit())}
    return counts


def summarize():
    with run_lock():
        return summarize_locked()


def summarize_locked():
    book, reports = phase_book(), {}
    for stage in C.STAGES:
        path = RUN_ROOT / f"{stage}.json"
        if not path.is_file():
            reports[stage] = {"status": "inconclusive", "reason": "not_run"}
            continue
        rows = L.read(path)
        module = C.adapters()[stage]
        if stage in ("B1", "B2"):
            C.review_text_rows(book, stage, rows, module.build_cases(), module)
            reports[stage] = C.judge_text(stage, rows)
        elif stage == "B3":
            cases = {c["id"]: c for c in module.build_cases()}
            if [r["entry"] for r in rows] != list(C.entries(stage, list(cases.values()))):
                raise ValueError("NAT 실행 순서 불일치")
            for row in rows:
                folder = RUN_ROOT / "nat" / C.entry_id(row["entry"])
                if not (folder / "result.json").is_file():
                    raise ValueError("NAT 원본 산출물이 없다")
                reviewed = module.review_case(cases[row["entry"]["id"]], row["entry"]["model"], folder, book)
                if reviewed != row["result"]:
                    raise ValueError("NAT 원시 재검증 불일치")
            reports[stage] = C.judge_nat(rows, book.audit())
        else:
            C.review_text_rows(book, stage, [rows["feasibility"]], module.build_preflight_cases(), module, attempts=3)
            C.review_text_rows(book, stage, rows["observations"], module.build_cases(), module)
            C.review_text_rows(book, stage, rows["graphs"], module.build_graph_cases(), module)
            reports[stage] = C.judge_vision(rows)
    raw_rows = parent_book().audit()
    phase_raw = [row for row in raw_rows if row["reservation"]["label"].startswith(P.PHASE_PREFIX)]
    providers = {}
    for row in phase_raw:
        if row["response"]:
            name = row["response"]["result"].get("provider_model")
            providers[str(name)] = providers.get(str(name), 0) + 1
    return {"schema": "nemotron_endpoint_comparison_result_v3", "experiment_commit": load_plan()["experiment_commit"],
            "protocol_sha256_lf": C.V1.sha_lf(PROTOCOL), "comparison_unit": FIXED["comparison_unit"],
            "official_alias_mapping_confirmed": False, "requested_models": {"old": L.BASELINE, "new": L.CANDIDATE},
            "provider_model_raw_counts": providers, "bottlenecks": reports,
            "raw_global_ledger_sha256": L.digest(raw_rows), "raw_phase_ledger_sha256": L.digest(phase_raw),
            "parent_prefix_sha256": L.digest(raw_rows[:2]),
            "terminal_reservation_sha256": L.digest(raw_rows[-1]["reservation"]),
            "global_requests_including_get": len(raw_rows)+1, "phase_generation_requests": len(phase_raw),
            "prior_generation_requests": len(raw_rows)-len(phase_raw),
            "cost_by_stage_model": {stage: {model: L.costs([r for r in phase_raw if r["reservation"]["stage"] == stage
                and r["reservation"]["model"] == model]) for model in L.MODELS} for stage in C.STAGES},
            "prior_cost": L.costs([r for r in raw_rows if r not in phase_raw])}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("verify")
    p = sub.add_parser("plan")
    p.add_argument("--commit", required=True)
    p = sub.add_parser("run")
    p.add_argument("--stage", choices=C.STAGES, required=True)
    sub.add_parser("summarize")
    args = parser.parse_args()
    result = (prepare() if args.command == "prepare" else verify() if args.command == "verify" else
              plan(args.commit) if args.command == "plan" else run(args.stage) if args.command == "run" else summarize())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
