"""E4b: E4 종단 시연의 고정 packet으로 NeMo Agent Toolkit 결정과 단일 호출 결정을 비교하는 사전등록·요약기.

GPU와 모델을 쓰지 않는다. NAT 실행은 WSL에서 evals/nat_decision_call.py로 하고, 이 파일은 입력 결합·실행 명령·
기록 재검증·지표만 맡는다. 단일 호출은 정답이 아니므로 일치율은 같은 입력에서의 일치 기록일 뿐이며, 정확도·
우위·절감을 주장하지 않는다. 다음 seed 시연에 NAT 결정을 쓸지는 실행 전에 정한 유효성 조건으로만 판단한다.

  python evals/e4b_compare.py prepare|verify
  python evals/e4b_compare.py plan --tag <E4 tag>        # 입력 SHA 결합과 WSL 실행 명령
  python evals/e4b_compare.py summarize --tag <E4 tag>
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

PROTOCOL = ROOT / "bench/protocols/e4b_nat_compare_v1.json"
PRIVATE = ROOT / "bench/private/e4b"
RESULTS = ROOT / "evals/results"
E4_PRIVATE = ROOT / "bench/private/e2e_reward_demo"
KINDS = ("diagnosis", "correction")
RUNS_PER_KIND = 5
MAX_ATTEMPTS_PER_RUN = 2  # 인프라 오류일 때만 같은 run 번호로 한 번 더
MODEL = "nvidia/nemotron-3-super-120b-a12b"
TAG_RE = r"[a-z0-9][a-z0-9_-]{0,35}"
CODE = {
    "src/rl_triage/nat_demo_tools.py", "evals/nat_decision_call.py", "evals/e4b_compare.py",
    "configs/nat_demo_diagnosis.yml", "configs/nat_demo_correction.yml",
    "src/rl_triage/active_diagnosis.py", "src/rl_triage/probe_loop.py", "evals/correction_model_call.py",
    "src/rl_triage/correction_registry.py", "bench/reference/nvidia_skills/isaaclab-debugging-rl-training/SKILL.md",
}
ADOPTION = {"min_non_infra_runs": 4, "min_valid_runs": 4}
FIXED = {
    "schema": "e4b_nat_compare_protocol_v1",
    "question": "E4 종단 시연과 같은 고정 입력에서 NAT 도구 호출 워크플로가 계약상 유효한 결정을 얼마나 내는가, 단일 호출 결정과 얼마나 같은가",
    "scope": "fixed_packet_repeated_nat_decision_development_measurement_not_accuracy_superiority_or_savings",
    "model": MODEL, "nat_version": "1.9.0", "temperature": 0,
    "inputs": {
        "diagnosis_packet": "E4 packet 단계가 결합한 bench/private/e2e_reward_demo/<tag>/model/diagnosis.packet.json",
        "diagnosis_single_call": "E4 start-loop가 마지막으로 받은 diagnosis.call<n>.json(같은 packet SHA)",
        "correction_packet": "E4 correct.py evidence 출력 bench/private/e2e_reward_demo/<tag>/model/correction.packet.json",
        "correction_single_call": "E4 correction.call1.json(같은 packet digest)",
        "not_reached": "E4가 그 packet을 만들기 전에 끝났으면 그 종류의 비교는 not_reached로 기록하고 대체 입력을 쓰지 않는다",
    },
    "runs_per_kind": RUNS_PER_KIND,
    "attempts": "run 번호마다 첫 시도. 인프라 오류(제공자 연결·시간 초과·429·5xx·빈 응답)일 때만 같은 run 번호로 1회 더",
    "status_definitions": {
        "valid": ("제출 도구가 계약 검증을 통과한 결정 하나를 저장했고 worker가 같은 검증기로 다시 통과시켰다. "
                  "제출 뒤 마지막 응답 단계의 오류는 post_submission_error로 따로 기록하고 결정을 버리지 않는다"),
        "invalid": "제출 거절(두 번째 제출 포함, 제출 파일이 있어도 거절 기록이 있으면 무효), 미제출, 반복 한도, 그 밖의 의미 오류",
        "infra_error": "두 번째 시도까지 인프라 오류. 첫 시도만 인프라 오류이고 두 번째 시도가 없으면 미완료로 두고 판정하지 않는다",
    },
    "comparison_conditions": {
        "same": ["모델·temperature 0·reasoning_effort none(extra_body)", "같은 packet", "같은 판단 계약 프롬프트와 검증기"],
        "different": ["NAT는 사실·카탈로그·등록 표·체크리스트를 도구로 조회하는 여러 턴, 단일 호출은 packet 전체를 한 번에 받는다",
                      "NAT 시스템 프롬프트는 계약 프롬프트 뒤에 도구 사용 규칙을 붙인다",
                      "NAT 기록은 결정 제출 도구의 저장 파일에서, 단일 호출은 응답 본문에서 결정을 읽는다"],
    },
    "metrics": ["유효·무효·인프라 실행 수", "단일 호출 결정과의 일치(진단: action·1위·상위 3 집합·probe, 교정: 결정·교정·회복 방식·미충족 체크리스트)",
                "도구 호출 순서와 수", "LLM 호출 수", "입력·출력 토큰", "wall 시간", "단일 호출의 같은 비용 항목(나란히 기록)"],
    "adoption_rule": {**ADOPTION,
                      "rule": "다음 seed 시연에 NAT 결정을 쓰려면 두 종류 모두 인프라 오류를 뺀 실행이 4 이상이고 그중 유효 결정이 4 이상이어야 한다. 일치율은 채택 조건에 쓰지 않는다(단일 호출이 정답이 아니다)"},
    "wording": {
        "adopt": "같은 고정 입력에서 NAT 도구 호출 워크플로가 사전등록 조건만큼 유효한 결정을 냈다. 다음 seed 시연의 결정 단계에 쓸 수 있다. 정확도·우위의 근거가 아니다.",
        "do_not_adopt": "같은 고정 입력에서 NAT 도구 호출 워크플로의 유효 결정이 사전등록 조건에 못 미쳤다. 다음 seed 시연은 단일 호출 결정을 유지한다.",
        "not_reached": "E4가 이 결정의 입력을 만들기 전에 끝나 비교하지 않았다.",
        "common": ("E4 한 사례의 고정 입력을 반복 호출한 개발 측정이다. NAT는 근거를 도구로 나눠 조회하고 단일 호출은 한 번에 받으므로 "
                   "일치율·비용 차이에는 이 제공 방식 차이가 섞여 있다. 일반화·정확도·절감을 주장하지 않는다."),
    },
    "forbidden_claims": ["NAT가 더 정확하다", "모든 결정이 NAT로 이뤄졌다", "토큰·시간 절감", "일반화"],
}
DYNAMIC = {"source_files"}


def sha_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_raw(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


# --- 프로토콜 ---
def dynamic_fields() -> dict:
    return {"source_files": {name: sha_lf(ROOT / name) for name in sorted(CODE)}}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("protocol fields differ")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("fixed comparison contract changed")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("bound sources changed")


def prepare() -> dict:
    write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields()})
    return {"status": "prepared_commit_before_runs", "path": rel(PROTOCOL)}


def verify_protocol() -> tuple[dict, str]:
    import run_runtime_readback_smoke as P
    protocol = read(PROTOCOL)
    check_protocol(protocol, dynamic_fields())
    return protocol, P.verify_committed(ROOT, CODE | {rel(PROTOCOL)})


# --- 입력 결합 ---
def folder_for(tag: str) -> Path:
    if not isinstance(tag, str) or not re.fullmatch(TAG_RE, tag):
        raise ValueError("invalid E4 tag")
    return PRIVATE / tag


def _latest_diagnosis_call(model_dir: Path) -> Path | None:
    calls = sorted((p for p in model_dir.glob("diagnosis.call*.json")
                    if re.fullmatch(r"diagnosis\.call\d+\.json", p.name)), key=lambda p: int(p.name[14:-5]))
    return calls[-1] if calls else None


def bind_inputs(tag: str) -> dict:
    """E4 산출물에서 두 종류의 packet과 단일 호출 기록을 찾아 SHA를 대조한다. 없으면 not_reached."""
    import correction_model_call as CM
    model_dir = E4_PRIVATE / tag / "model"
    out = {"tag": tag}
    packet = model_dir / "diagnosis.packet.json"
    call = _latest_diagnosis_call(model_dir) if model_dir.exists() else None
    if packet.is_file() and call is not None:
        record = read(call)
        if record.get("schema") != "active_model_call_v1" or record.get("packet_sha256") != sha_raw(packet):
            raise ValueError("진단 단일 호출 기록이 E4 진단 packet과 다르다")
        out["diagnosis"] = {"packet": rel(packet), "packet_sha256": sha_raw(packet),
                            "single_call": rel(call), "single_call_sha256": sha_raw(call)}
    else:
        out["diagnosis"] = {"not_reached": True}
    packet, call = model_dir / "correction.packet.json", model_dir / "correction.call1.json"
    if packet.is_file() and call.is_file():
        record = read(call)
        if (record.get("schema") != CM.RECORD_SCHEMA
                or record.get("packet_digest") != CM.packet_digest(read(packet))):
            raise ValueError("교정 단일 호출 기록이 E4 교정 packet과 다르다")
        out["correction"] = {"packet": rel(packet), "packet_sha256": sha_raw(packet),
                             "single_call": rel(call), "single_call_sha256": sha_raw(call)}
    else:
        out["correction"] = {"not_reached": True}
    return out


def run_path(tag: str, kind: str, run: int, attempt: int) -> Path:
    return folder_for(tag) / kind / f"run{run}_a{attempt}.json"


def wsl(path: Path) -> str:
    p = Path(path).resolve().as_posix()
    return f"/mnt/{p[0].lower()}/{p[3:]}"


def plan(tag: str) -> dict:
    protocol, commit = verify_protocol()
    inputs_path = folder_for(tag) / "inputs.json"
    if inputs_path.exists():
        inputs = read(inputs_path)
        if inputs != {**bind_inputs(tag), "protocol_sha256_lf": inputs["protocol_sha256_lf"],
                       "source_commit_sha": inputs["source_commit_sha"]}:
            raise ValueError("E4 입력이 결합 뒤 바뀌었다")
    else:
        inputs = {**bind_inputs(tag), "protocol_sha256_lf": sha_lf(PROTOCOL), "source_commit_sha": commit}
        write_new(inputs_path, inputs)
    commands = []
    for kind in KINDS:
        if inputs[kind].get("not_reached"):
            continue
        packet = ROOT / inputs[kind]["packet"]
        for run in range(1, RUNS_PER_KIND + 1):
            commands.append(
                "wsl -d Ubuntu -- bash -lc '. \"$HOME/.config/nvidia/env\"; exec \"$HOME/.venvs/rl-triage/bin/python\" "
                f"{wsl(ROOT / 'evals/nat_decision_call.py')} --kind {kind} --packet {wsl(packet)} "
                f"--output {wsl(run_path(tag, kind, run, 1))}'")
    return {"inputs": inputs, "commands": commands,
            "retry_rule": "인프라 오류 기록(status=infra_error)이면 같은 명령의 출력만 run<i>_a2.json으로 바꿔 1회 더 실행"}


# --- 기록 재검증과 지표 ---
def single_call_decision(kind: str, inputs: dict) -> dict | None:
    """단일 호출 기록을 같은 계약으로 다시 검증한 결정. 무효면 None."""
    record = read(ROOT / inputs[kind]["single_call"])
    packet = read(ROOT / inputs[kind]["packet"])
    if kind == "diagnosis":
        from rl_triage import active_diagnosis as D
        text = record.get("content") or ""
        if text.startswith("```json") and text.rstrip().endswith("```"):
            text = text[7:text.rfind("```")].strip()
        try:
            return D.validate_decision(json.loads(text), packet)
        except (ValueError, TypeError):
            return None
    import correction_model_call as CM
    proposal, error = CM.parse_content(record.get("content"))
    if error:
        return None
    verdict = CM.validate_model_proposal(proposal, packet)
    return {"proposal": proposal, **verdict} if verdict["accepted"] else None


def single_call_costs(kind: str, inputs: dict) -> dict:
    record = read(ROOT / inputs[kind]["single_call"])
    return {k: record.get(k) for k in ("input_tokens", "output_tokens", "execution_wall_s")}


def agreement(kind: str, nat: dict, single: dict | None) -> dict | None:
    if single is None:
        return None
    if kind == "diagnosis":
        return {"action": nat["action"] == single["action"], "top1": nat["ranking"][0] == single["ranking"][0],
                "top3_set": set(nat["ranking"][:3]) == set(single["ranking"][:3]), "probe": nat["probe"] == single["probe"]}
    return {"decision": nat["decision"] == single["decision"],
            "correction_id": nat["proposal"]["correction_id"] == single["proposal"]["correction_id"],
            "recovery_mode": nat["proposal"]["recovery_mode"] == single["proposal"]["recovery_mode"],
            "unmet_checklist": nat["unmet_checklist"] == single["unmet_checklist"]}


def expected_prompt_sha(kind: str) -> str:
    import yaml
    import nat_decision_call as W
    rules = yaml.safe_load(W.CONFIGS[kind].read_text(encoding="utf-8"))["workflow"]["system_prompt"]
    return W.text_sha(W.system_prompt(kind, rules))


def verify_record(path: Path, record: dict, kind: str, inputs: dict, protocol: dict) -> None:
    """NAT 기록의 결합값과, 사건·도구 기록·궤적·거절 파일에서 다시 계산한 값을 대조한다."""
    import nat_decision_call as W
    expected = {"schema": W.RECORD_SCHEMA, "kind": kind, "model": MODEL, "temperature": 0, "nat_version": "1.9.0",
                "semantic_retries": 0, "packet_sha256": inputs[kind]["packet_sha256"],
                "config_sha256_lf": protocol["source_files"][f"configs/nat_demo_{kind}.yml"],
                "tools_module_sha256_lf": protocol["source_files"]["src/rl_triage/nat_demo_tools.py"],
                "system_prompt_sha256": expected_prompt_sha(kind),
                "user_message_sha256": W.text_sha(W.user_message(kind, read(ROOT / inputs[kind]["packet"])))}
    if any(record.get(k) != v for k, v in expected.items()) or record.get("status") not in ("valid", "invalid", "infra_error"):
        raise ValueError(f"{path.name}: NAT 기록이 등록 입력·설정·프롬프트와 다르다")
    events = path.with_suffix(".events.json")
    if record.get("events_sha256") != sha_raw(events):
        raise ValueError(f"{path.name}: 사건 기록이 바뀌었다")
    summary = W.summarize_events(read(events)["events"])
    if any(record.get(k) != v for k, v in summary.items()):
        raise ValueError(f"{path.name}: 기록의 호출 수·토큰·도구 순서가 사건 파일과 다르다")
    if record.get("tools_trace_sha256") != sha_raw(path.with_suffix(".tools.jsonl")):
        raise ValueError(f"{path.name}: 도구 호출 기록이 바뀌었다")
    if record.get("atif_sha256") is not None and record["atif_sha256"] != sha_raw(path.with_suffix(".atif.json")):
        raise ValueError(f"{path.name}: ATIF 궤적이 바뀌었다")
    rejected = path.with_suffix(".submission.rejected.json")
    if rejected.exists() != ("rejection_sha256" in record) or (rejected.exists() and record["rejection_sha256"] != sha_raw(rejected)):
        raise ValueError(f"{path.name}: 거절 기록 결합이 맞지 않는다")
    if record["status"] == "valid" and rejected.exists():
        raise ValueError(f"{path.name}: 거절 기록이 있는데 유효로 기록됐다")


def load_run(tag: str, kind: str, run: int, inputs: dict, protocol: dict) -> dict:
    """run 번호의 최종 시도. 첫 시도가 인프라 오류일 때만 두 번째 시도를 본다. 재시도가 남았으면 missing."""
    import nat_decision_call as W
    attempts = [run_path(tag, kind, run, a) for a in range(1, MAX_ATTEMPTS_PER_RUN + 1)]
    records = [(i + 1, p, read(p)) for i, p in enumerate(attempts) if p.is_file()]
    if [a for a, _, _ in records] != list(range(1, len(records) + 1)):
        raise ValueError(f"{kind} run{run}: 시도 번호가 이어지지 않는다")
    for attempt, path, record in records:
        verify_record(path, record, kind, inputs, protocol)
        if attempt < len(records) and record["status"] != "infra_error":
            raise ValueError(f"{kind} run{run}: 인프라 오류가 아닌 시도 뒤에 재시도 기록이 있다")
    if not records:
        return {"run": run, "status": "missing"}
    attempt, path, record = records[-1]
    if record["status"] == "infra_error" and attempt < MAX_ATTEMPTS_PER_RUN:
        return {"run": run, "status": "missing", "pending": "infra_retry"}
    if record["status"] == "valid":
        submission = path.with_suffix(".submission.json")
        if sha_raw(submission) != record["submission_sha256"]:
            raise ValueError(f"{path.name}: 제출 파일이 바뀌었다")
        if W.validate_submission(kind, read(ROOT / inputs[kind]["packet"]), read(submission)) != record["decision"]:
            raise ValueError(f"{path.name}: 제출이 다시 검증되지 않는다")
    return {"run": run, "attempt": attempt, "status": record["status"], "decision": record["decision"],
            "error_type": record.get("error_type"), "tool_sequence": record.get("tool_sequence"),
            "llm_calls": record.get("llm_calls"), "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"), "wall_s": record.get("wall_s"),
            "atif_sha256": record.get("atif_sha256")}


def _stats(values: list) -> dict | None:
    numbers = [v for v in values if isinstance(v, (int, float))]
    if not numbers:
        return None
    return {"n": len(numbers), "median": statistics.median(numbers), "min": min(numbers), "max": max(numbers)}


def kind_summary(tag: str, kind: str, inputs: dict, protocol: dict) -> dict:
    if inputs[kind].get("not_reached"):
        return {"status": "not_reached", "wording": FIXED["wording"]["not_reached"]}
    runs = [load_run(tag, kind, run, inputs, protocol) for run in range(1, RUNS_PER_KIND + 1)]
    single = single_call_decision(kind, inputs)
    valid = [r for r in runs if r["status"] == "valid"]
    counts = {status: sum(1 for r in runs if r["status"] == status) for status in ("valid", "invalid", "infra_error", "missing")}
    matches = [agreement(kind, r["decision"], single) for r in valid]
    agree = None if single is None or not matches else {key: sum(1 for m in matches if m[key]) for key in matches[0]}
    sequences = [" > ".join(r["tool_sequence"] or []) for r in runs if r["status"] != "missing"]
    return {"counts": counts, "non_infra_runs": counts["valid"] + counts["invalid"],
            "single_call_valid": single is not None, "agreement_with_single_call": agree,
            "agreement_denominator": len(valid) if single is not None else 0,
            "tool_sequences": {seq: sequences.count(seq) for seq in sorted(set(sequences))},
            **{key: _stats([r.get(key) for r in runs]) for key in ("llm_calls", "input_tokens", "output_tokens", "wall_s")},
            "single_call_costs": single_call_costs(kind, inputs), "runs": runs,
            "complete": counts["missing"] == 0}


def adoption(summaries: dict) -> dict:
    reached = {k: s for k, s in summaries.items() if s.get("status") != "not_reached"}
    if not reached or any(not s["complete"] for s in reached.values()):
        return {"decision": None, "reason": "실행이 모두 끝나지 않았다"}
    if len(reached) < len(KINDS):
        return {"decision": "do_not_adopt", "reason": "두 종류를 모두 비교하지 못했다"}
    ok = all(s["non_infra_runs"] >= ADOPTION["min_non_infra_runs"] and s["counts"]["valid"] >= ADOPTION["min_valid_runs"]
             for s in reached.values())
    return {"decision": "adopt" if ok else "do_not_adopt", "rule": FIXED["adoption_rule"]["rule"]}


def summarize(tag: str) -> dict:
    protocol, commit = verify_protocol()
    inputs = read(folder_for(tag) / "inputs.json")
    if inputs["protocol_sha256_lf"] != sha_lf(PROTOCOL):
        raise ValueError("입력 결합 뒤 프로토콜이 바뀌었다")
    for kind in KINDS:
        if not inputs[kind].get("not_reached"):
            for key in ("packet", "single_call"):
                if sha_raw(ROOT / inputs[kind][key]) != inputs[kind][f"{key}_sha256"]:
                    raise ValueError(f"{kind} {key}가 결합 뒤 바뀌었다")
    summaries = {kind: kind_summary(tag, kind, inputs, protocol) for kind in KINDS}
    decision = adoption(summaries)
    result = {"schema": "e4b_nat_compare_result_v1", "tag": tag, "protocol_sha256_lf": sha_lf(PROTOCOL),
              "source_commit_sha": commit, "kinds": summaries, "adoption": decision,
              "wording": None if decision["decision"] is None else FIXED["wording"][decision["decision"]],
              "common_wording": FIXED["wording"]["common"]}
    if decision["decision"] is not None:
        write_new(RESULTS / f"e4b_{tag}" / "summary.json", result)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4b NAT 결정 비교")
    ap.add_argument("command", choices=("prepare", "verify", "plan", "summarize"))
    ap.add_argument("--tag")
    args = ap.parse_args(argv)
    if args.command in ("plan", "summarize") and not args.tag:
        ap.error(f"{args.command}에는 --tag가 필요하다")
    commands = {"prepare": prepare, "verify": lambda: {"status": "pass", "source_commit_sha": verify_protocol()[1]},
                "plan": lambda: plan(args.tag), "summarize": lambda: summarize(args.tag)}
    try:
        out = commands[args.command]()
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"거부: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
