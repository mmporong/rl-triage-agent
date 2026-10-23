"""공통뷰의 선택 호출·불변 원장 사본·20회 API 원장. 실행 승인은 부여하지 않는다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import residual_model_call_v1 as MC  # noqa: E402
import selection_comparison_view_v1 as V  # noqa: E402

F, MODEL = MC.F, V.S.MODEL
PROTOCOL = ROOT / "bench/protocols/selection_comparison_call_v1.json"
CONDITIONS = ("production", "exposure_full", "exposure_masked")
SOURCE_FILES = tuple(sorted(set(V.SOURCE_FILES) | {
    "bench/protocols/selection_comparison_view_v1.json", "evals/selection_comparison_call_v1.py",
    "tests/test_selection_comparison_call_v1.py", "evals/residual_model_call_v1.py",
    "evals/r1_repair_model_call.py", "evals/feedback_model_call.py",
}))
FEEDBACK = {**V.S.FEEDBACK, "S_PROBE_INELIGIBLE": "Select a probe from eligible_probes, or hold."}
require = V.require


def raw_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capsule_bytes(value: dict) -> bytes:
    return V.canonical(value) + b"\n"


def verify_sources(expected: str, anchor: str | None = None) -> dict:
    raw = PROTOCOL.read_bytes().replace(b"\r\n", b"\n")
    require(raw_sha(raw) == expected, "caller protocol SHA 불일치")
    value = json.loads(raw)
    require(value.get("schema") == "selection_comparison_call_protocol_v1"
            and value.get("measurement_kind") == "synthetic_cpu_contract_only"
            and value.get("model") == MODEL and value.get("params") == F.request_params(MODEL)
            and value.get("api_cap") == MC.API_CAP == 20
            and value.get("request_limit_per_logical_call") == 3
            and value.get("semantic_feedback_limit") == 1 and value.get("infra_retry_limit") == 1
            and value.get("conditions") == list(CONDITIONS)
            and set(value.get("source_sha256_lf", {})) == set(SOURCE_FILES), "caller protocol 범위 불일치")
    for path, sha in value["source_sha256_lf"].items():
        require(V.file_digest(ROOT / path) == sha, "caller source 변경")
    old = V.verify_protocol(value["source_sha256_lf"]["bench/protocols/selection_comparison_view_v1.json"])
    require(set(old["public_base_fact_ids"]) == V.BASE_FACT_IDS, "공개 fact ID 의미 결합 불일치")
    if anchor is not None:
        require(bool(re.fullmatch("[0-9a-f]{40}", anchor)), "실험 anchor 형식")
        require(subprocess.run(["git", "merge-base", "--is-ancestor", anchor, "HEAD"],
                               cwd=ROOT, capture_output=True).returncode == 0, "실험 커밋이 현재 HEAD의 조상이 아님")
        bound = {**value["source_sha256_lf"], PROTOCOL.relative_to(ROOT).as_posix(): expected}
        for path, sha in bound.items():
            for version in (f"{anchor}:{path}", f":{path}"):
                data = subprocess.run(["git", "show", version], cwd=ROOT, capture_output=True)
                require(data.returncode == 0 and raw_sha(data.stdout.replace(b"\r\n", b"\n")) == sha,
                        "source commit/blob/index 불일치")
    return value


def inside(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative) and "\\" not in relative and ":" not in relative
            and all(part not in ("", ".", "..") for part in relative.split("/")), "상대 경로 형식")
    child, root = (Path(root) / relative).resolve(), Path(root).resolve()
    require(child != root and child.is_relative_to(root), "batch 밖 경로")
    return child


def initialize(root: Path, run_id: str, protocol_sha: str, *, fixture: bool,
               experiment_sha: str | None = None, anchor: str | None = None) -> dict:
    verify_sources(protocol_sha, anchor)
    require(type(fixture) is bool and bool(re.fullmatch("[a-z0-9_]{1,80}", run_id)), "session 식별자")
    require(fixture or (V.S._valid_sha(experiment_sha) and anchor is not None), "실제 실행 사전등록 결합 없음")
    owner = {"schema": "selection_call_session_v1", "run_id": run_id, "caller_protocol_sha": protocol_sha,
             "fixture": fixture, "experiment_sha": experiment_sha, "anchor": anchor, "api_cap": 20}
    path = Path(root) / "session.json"
    if path.exists():
        require(MC.read(path) == owner, "다른 batch/protocol의 API 원장")
    else:
        MC.write_new(path, owner)
    return owner


def session(root: Path) -> dict:
    owner = MC.read(Path(root) / "session.json")
    require(set(owner) == {"schema", "run_id", "caller_protocol_sha", "fixture", "experiment_sha", "anchor", "api_cap"}
            and owner["schema"] == "selection_call_session_v1" and type(owner["fixture"]) is bool
            and type(owner["api_cap"]) is int and owner["api_cap"] == 20
            and isinstance(owner["run_id"], str) and bool(re.fullmatch("[a-z0-9_]{1,80}", owner["run_id"]))
            and ((owner["fixture"] and owner["experiment_sha"] is None and owner["anchor"] is None)
                 or (not owner["fixture"] and V.S._valid_sha(owner["experiment_sha"])
                     and isinstance(owner["anchor"], str) and bool(re.fullmatch("[0-9a-f]{40}", owner["anchor"])))),
            "session 범위 또는 소유자 형식 불일치")
    verify_sources(owner["caller_protocol_sha"])
    return owner


def query_view(snapshot: dict, condition: str) -> dict:
    value = V.checked(snapshot)
    require(condition in CONDITIONS and value["inputs"]["masked"] is False, "호출 조건 또는 source snapshot")
    if condition == "production":
        return deepcopy(value["view"])
    require(value["inputs"]["policy"] == "model_adaptive", "노출 query는 adaptive 상태만 허용")
    shadow = V.build_snapshot(**{**value["inputs"], "masked": True})
    require(shadow["view"]["status"] == "open" and bool(shadow["view"]["eligible_probes"])
            and value["view"]["observations"][-1]["outcome"] in ("normal", "abnormal"),
            "노출 query의 유효한 다음 선택 상태가 아님")
    view = deepcopy(shadow["view"])
    if condition == "exposure_full":
        view["observations"][-1] = deepcopy(value["view"]["observations"][-1])
        prefix = f"observation:{len(view['observations']) - 1}:"
        view["evidence_facts"].update({key: val for key, val in value["view"]["evidence_facts"].items()
                                       if key.startswith(prefix)})
    return {**view, "schema": "selection_preupdate_exposure_view_v1", "used_for_decision": False}


def pair_disposition(full: dict, masked: dict) -> str:
    full, masked = V.checked(full), V.checked(masked)
    require(not full["observation_masked_control"] and masked["observation_masked_control"]
            and full["inputs"]["policy"] == "model_adaptive"
            and full["binding"] == masked["binding"] and full["ledger_digest"] == masked["ledger_digest"]
            and full["pre_receipt_anchor"] == masked["pre_receipt_anchor"], "다른 상태의 full/masked pair")
    a, b = full["view"], masked["view"]
    if any(view["status"] != "open" or view["budget_left"] <= 0 for view in (a, b)):
        return "host_transition_only"
    fields = ("remaining_candidates", "eligible_probes", "probe_catalog", "fixed_order", "budget_left",
              "execution_budget_s", "execution_budget_scope", "recorded_probe_gpu_s", "recorded_cost_scope")
    return ("receipt_exposure_pair" if all(a[key] == b[key] for key in fields)
            else "combined_host_view_change")


def _capsule(snapshot: dict, condition: str, history_relative: str, history_sha: str, owner: dict,
             replicate: int = 1) -> dict:
    value = V.checked(snapshot)
    view = query_view(value, condition)
    inputs = {key: val for key, val in value["inputs"].items() if key != "ledger_path"}
    return {"schema": "selection_call_capsule_v1", "snapshot_inputs": inputs,
            "history_relative": history_relative, "history_sha256": history_sha,
            "context_digest": V.L.digest(value["binding"]), "ledger_digest": value["ledger_digest"],
            "pre_receipt_anchor": value["pre_receipt_anchor"], "operational_view_digest": value["view_digest"],
            "view_digest": V.L.digest(view), "condition": condition, "replicate": replicate, "model": MODEL,
            "params": F.request_params(MODEL), "session_digest": V.L.digest(owner),
            "caller_protocol_sha": owner["caller_protocol_sha"]}


def freeze(root: Path, snapshot: dict, condition: str = "production", replicate: int = 1) -> Path:
    root = Path(root).resolve()
    owner = session(root)
    value = V.checked(snapshot)
    require(type(replicate) is int and 1 <= replicate <= 20, "반복 식별자 형식")
    require(value["inputs"]["run_id"] == owner["run_id"], "다른 run의 snapshot")
    live = Path(value["inputs"]["ledger_path"])
    require(live.is_relative_to(root), "batch 밖 live 원장")
    data = live.read_bytes() if live.exists() else b""
    require(V.checked(snapshot) == value, "원장 사본 저장 중 상태 변경")
    relative = f"histories/{V.L.digest(value['binding'])}/{value['ledger_digest']}.jsonl"
    history = inside(root, relative)
    history.parent.mkdir(parents=True, exist_ok=True)
    try:
        with history.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        require(history.read_bytes() == data, "기존 불변 원장 사본 불일치")
    frozen = V.build_snapshot(**{**value["inputs"], "ledger_path": history})
    require(frozen["ledger_digest"] == value["ledger_digest"]
            and frozen["view_digest"] == value["view_digest"], "원장 사본의 의미가 달라짐")
    capsule = _capsule(frozen, condition, relative, raw_sha(data), owner, replicate)
    raw = capsule_bytes(capsule)
    path = inside(root, f"capsules/{raw_sha(raw)}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
    return path


def prepared(root: Path, capsule_path: Path) -> tuple[dict, dict, dict, list]:
    root, capsule_path = Path(root).resolve(), Path(capsule_path).resolve()
    require(capsule_path.parent == root / "capsules"
            and capsule_path.name == raw_sha(capsule_path.read_bytes()) + ".json", "capsule 경로·SHA 불일치")
    capsule, owner = MC.read(capsule_path), session(root)
    require(capsule_path.read_bytes() == capsule_bytes(capsule), "비정규 capsule 직렬화")
    require(capsule["session_digest"] == V.L.digest(owner), "다른 session의 capsule")
    history = inside(root, capsule["history_relative"])
    require(raw_sha(history.read_bytes()) == capsule["history_sha256"], "불변 원장 사본 변경")
    snapshot = V.build_snapshot(**capsule["snapshot_inputs"], ledger_path=history)
    require(capsule["history_relative"] == f"histories/{V.L.digest(snapshot['binding'])}/{snapshot['ledger_digest']}.jsonl",
            "다른 logical call의 원장 별칭")
    require(snapshot["inputs"]["run_id"] == owner["run_id"]
            and capsule == _capsule(snapshot, capsule["condition"], capsule["history_relative"],
                                    capsule["history_sha256"], owner, capsule["replicate"])
            and type(capsule["replicate"]) is int and 1 <= capsule["replicate"] <= 20,
            "capsule 상태·model·params 결합 불일치")
    view = query_view(snapshot, capsule["condition"])
    extra = ("\nNon-executable observation query: candidates and eligible_probes are held at the state before "
             "the latest outcome. Use the exposed evidence to choose hold or the next probe. This query never executes."
             if capsule["condition"] != "production" else "")
    messages = [{"role": "system", "content": V.SYSTEM + extra},
                {"role": "user", "content": V.canonical(view).decode()}]
    return capsule, snapshot, view, messages


def planned_state(capsule: dict) -> dict:
    return {key: deepcopy(capsule[key]) for key in (
        "context_digest", "ledger_digest", "pre_receipt_anchor", "operational_view_digest", "view_digest",
        "history_sha256", "condition", "replicate")}


def _choice(snapshot: dict, view: dict, condition: str, content: str | None, status: str = "completed") -> dict:
    if condition == "production":
        reply = None if content is None and status == "completed" else {
            "context_digest": V.L.digest(snapshot["binding"]), "view_digest": snapshot["view_digest"],
            "content": content, "status": status}
        return V.select(snapshot, reply)
    result = {"action": "hold", "probe": None, "fact_ids": [], "model_status": "unknown",
              "codes": [], "used_for_decision": False, "approval_granted": False}
    if status != "completed":
        return result
    check = V.S.format_review(content, {**snapshot["inputs"]["packet"], "facts": view["evidence_facts"]})
    if check["status"] == "valid":
        decision = check["decision"]
        codes = ([] if all(key in view["evidence_facts"] for key in decision["fact_ids"]) else ["S_FACT_UNKNOWN"])
        if decision["action"] == "continue" and decision["probe"] not in view["eligible_probes"]:
            codes.append("S_PROBE_INELIGIBLE")
        return {**result, **decision, "model_status": "invalid" if codes else "valid", "codes": codes}
    return {**result, "model_status": "invalid", "codes": check["codes"]}


def transport(result: dict) -> str:
    if result.get("provider_model") not in (None, MODEL):
        return "unknown"
    if result.get("infra_error"):
        return "infra_error"
    if result.get("error_type") or result.get("provider_model") != MODEL:
        return "unknown"
    content = result.get("content")
    if content is None or (isinstance(content, str) and not content.strip()) or result.get("finish_reason") == "length":
        return "infra_error"
    if not isinstance(content, str) or result.get("finish_reason") != "stop":
        return "unknown"
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = result.get(name)
        if value is not None and (type(value) is not int or value < 0):
            return "unknown"
    counts = [result.get(name) for name in ("input_tokens", "output_tokens", "total_tokens")]
    if all(value is not None for value in counts) and counts[0] + counts[1] != counts[2]:
        return "unknown"
    return "completed" if V.number(result.get("wall_s")) and result["wall_s"] >= 0 else "unknown"


def safe_result(result: dict) -> dict:
    allowed = {"response_id", "provider_model", "finish_reason", "content", "reasoning_content",
               "input_tokens", "output_tokens", "total_tokens", "wall_s", "error_type", "infra_error", "status_code"}
    require(isinstance(result, dict), "provider 결과는 객체여야 함")
    clean = {key: deepcopy(val) for key, val in result.items() if key in allowed}
    bad = []
    for key in ("content", "reasoning_content", "response_id", "provider_model", "finish_reason"):
        if clean.get(key) is not None and not isinstance(clean[key], str):
            clean[key] = None
            bad.append(key)
    for key in ("error_type", "infra_error"):
        if clean.get(key) is not None and (not isinstance(clean[key], str)
                or not re.fullmatch("[A-Za-z_][A-Za-z0-9_]{0,79}", clean[key])):
            clean[key] = None
            bad.append(key)
    for key in ("input_tokens", "output_tokens", "total_tokens", "status_code"):
        if clean.get(key) is not None and (type(clean[key]) is not int or clean[key] < 0):
            clean[key] = None
            bad.append(key)
    if clean.get("wall_s") is not None and (not V.number(clean["wall_s"]) or clean["wall_s"] < 0):
        clean["wall_s"] = None
        bad.append("wall_s")
    if bad:
        clean.update(error_type="ProviderResponseContractError", infra_error=None,
                     response_contract_fields=sorted(bad))
    return clean


class FixtureResponses:
    """합성 자료만 반환한다. 임의 callable을 실행하지 않는다."""

    def __init__(self, rows: list):
        require(isinstance(rows, list) and all(isinstance(row, dict) or type(row) in
                (TimeoutError, ConnectionError, ValueError, RuntimeError) for row in rows), "합성 응답 자료 형식")
        self.rows, self.sent = iter(deepcopy(rows)), []

    def __call__(self, messages, params):
        self.sent.append((deepcopy(messages), deepcopy(params)))
        row = next(self.rows)
        if isinstance(row, Exception):
            raise row
        return deepcopy(row)


def is_wsl() -> bool:
    return sys.platform.startswith("linux") and bool(os.environ.get("WSL_DISTRO_NAME"))


class NvidiaCompletion:
    """커밋된 계획 행을 소비하고 기존 API 예약 아래에서만 실제 provider를 연결한다."""

    def __init__(self, root: Path, capsule_path: Path, experiment: Path, anchor: str):
        require(is_wsl(), "실제 transport는 WSL에서만 허용")
        self.root, self.capsule_path = Path(root).resolve(), Path(capsule_path).resolve()
        self.owner = session(self.root)
        execution = execution_manifest(experiment, anchor, self.owner["caller_protocol_sha"])
        require(not self.owner["fixture"] and self.owner["anchor"] == anchor
                and self.owner["experiment_sha"] == MC.file_sha(experiment)
                and self.owner["run_id"] == execution["run_id"], "실제 transport의 실행 결합 불일치")
        self.capsule, _, _, _ = prepared(self.root, self.capsule_path)
        matched = [row for row in execution["planned_calls"] if row["state"] == planned_state(self.capsule)]
        require(len(matched) == 1, "사전등록에 없는 입력·조건·반복")
        self.row, self.used = matched[0], set()
        for prior in execution["planned_calls"][:self.row["order"] - 1]:
            receipt = MC.read(inside(self.root, f"plan_receipts/{prior['row_id']}.json"))
            require(set(receipt) == {"schema", "row_digest", "experiment_sha", "session_digest", "capsule_sha256"}
                    and receipt["schema"] == "selection_call_plan_consumed_v1"
                    and receipt["row_digest"] == V.L.digest(prior)
                    and receipt["experiment_sha"] == self.owner["experiment_sha"]
                    and receipt["session_digest"] == V.L.digest(self.owner)
                    and V.S._valid_sha(receipt["capsule_sha256"]), "이전 계획 행 receipt 불일치")
            prior_capsule = inside(self.root, f"capsules/{receipt['capsule_sha256']}.json")
            require(planned_state(prepared(self.root, prior_capsule)[0]) == prior["state"], "이전 계획 입력 불일치")
            review(self.root, prior_capsule, inside(self.root, f"calls/{receipt['capsule_sha256']}.json"))
        MC.write_new(inside(self.root, f"plan_receipts/{self.row['row_id']}.json"), {
            "schema": "selection_call_plan_consumed_v1", "row_digest": V.L.digest(self.row),
            "experiment_sha": self.owner["experiment_sha"], "session_digest": V.L.digest(self.owner),
            "capsule_sha256": MC.file_sha(self.capsule_path)})

    def __call__(self, messages, params):
        verify_sources(self.owner["caller_protocol_sha"], self.owner["anchor"])
        require(params == self.capsule["params"], "실제 호출 params 불일치")
        active = MC.read(MC._active_path(self.root / "api"))
        budget = MC.audit_budget(self.root / "api")
        pending = [row["reservation"] for row in budget["rows"] if row["response"] is None]
        require(active.get("pid") == os.getpid() and len(pending) == 1, "실제 호출의 소유 API 예약 없음")
        reservation = pending[0]
        require(reservation["kind"] == "diagnosis"
                and reservation["packet_sha256"] == MC.file_sha(self.capsule_path)
                and reservation["messages_sha256"] == V.L.digest(messages)
                and reservation["output_key"] == f"selection/{MC.file_sha(self.capsule_path)}/request{len(self.used) + 1}"
                and len(self.used) < self.row["max_requests"]
                and reservation["request_index"] not in self.used, "다른 입력 또는 이미 소비한 실제 예약")
        self.used.add(reservation["request_index"])
        return F.openai_complete(messages, params)


def engine(snapshot: dict, view: dict, messages: list, condition: str, complete) -> dict:
    calls, sent = [], deepcopy(messages)
    feedback_used = infra_used = False
    stage = "first"
    reason = "not_used"
    choice = _choice(snapshot, view, condition, None)
    needs_model = (condition != "production" or choice.get("reason") == "model_unavailable")
    if needs_model:
        for _ in range(3):
            try:
                result = complete(deepcopy(sent), F.request_params(MODEL))
            except MC.BudgetStop:
                choice, reason = _choice(snapshot, view, condition, None, "unknown"), "budget_exhausted"
                break
            state = transport(result)
            choice = _choice(snapshot, view, condition, result.get("content"),
                             "completed" if state == "completed" else state)
            calls.append({"step": stage, "messages_digest": V.L.digest(sent), "result": deepcopy(result),
                          "transport": state, "choice": choice})
            reason = state
            if state == "infra_error" and not infra_used:
                infra_used = True
                stage = "infra_retry"
                continue
            if state == "completed" and choice["model_status"] == "invalid" and not feedback_used:
                require(all(code in FEEDBACK for code in choice["codes"]), "등록되지 않은 피드백 코드")
                text = "Contract validation rejected the output.\n" + "\n".join(
                    f"{code}: {FEEDBACK[code]}" for code in choice["codes"])
                sent = [*deepcopy(messages), {"role": "assistant", "content": result["content"]},
                        {"role": "user", "content": text}]
                feedback_used = True
                stage = "feedback"
                continue
            break
    costs = {name: (sum(row["result"][name] for row in calls)
                   if all(type(row["result"].get(name)) is int and row["result"][name] >= 0 for row in calls)
                   else None) for name in ("input_tokens", "output_tokens", "total_tokens")}
    costs["wall_s"] = (sum(row["result"]["wall_s"] for row in calls)
                       if all(V.number(row["result"].get("wall_s")) and row["result"]["wall_s"] >= 0
                              for row in calls) else None)
    return {"schema": "selection_call_dialogue_v1", "condition": condition, "calls": calls,
            "requests": len(calls), "selection": choice, "stop_reason": reason,
            "feedback_used": any(row["messages_digest"] != V.L.digest(messages) for row in calls),
            "infra_retry_used": any(row["step"] == "infra_retry" for row in calls), "costs": costs,
            "used_for_decision": condition == "production", "approval_granted": False}


def audit(root: Path) -> dict:
    root = Path(root).resolve()
    session(root)
    value = MC.audit_budget(root / "api")
    groups = {}
    for row in value["rows"]:
        reservation = row["reservation"]
        sha = reservation["packet_sha256"]
        require(V.S._valid_sha(sha) and reservation["kind"] == "diagnosis", "다른 종류의 API 예약")
        path = inside(root, f"capsules/{sha}.json")
        _capsule_value, _snapshot, _view, _messages = prepared(root, path)
        group = groups.setdefault(sha, [])
        require(reservation["output_key"] == f"selection/{sha}/request{len(group) + 1}"
                and len(group) < 3, "다른 capsule 또는 중복 logical call 예약")
        group.append(row)
    return {**value, "groups": groups}


def decide(root: Path, capsule_path: Path, complete) -> dict:
    """CPU 응답 자료 또는 사전등록을 검증한 WSL transport만 허용한다."""
    root, capsule_path = Path(root).resolve(), Path(capsule_path).resolve()
    owner = session(root)
    require((type(complete) is FixtureResponses and owner["fixture"])
            or (type(complete) is NvidiaCompletion and not owner["fixture"]
                and complete.root == root and complete.capsule_path == capsule_path), "호출 transport 권한 없음")
    capsule, snapshot, view, messages = prepared(root, capsule_path)
    sha = MC.file_sha(capsule_path)
    output = inside(root, f"calls/{sha}.json")
    budget = audit(root)
    require(not budget["pending"], "미완료 API 예약")
    require(not output.exists() and sha not in budget["groups"], "이미 시작하거나 끝난 logical call")
    def checked_complete(sent, params):
        return safe_result(complete(deepcopy(sent), deepcopy(params)))
    metered = MC.MeteredComplete(root / "api", "diagnosis", sha, f"selection/{sha}", checked_complete)
    try:
        record = engine(snapshot, view, messages, capsule["condition"], metered)
        record.update(capsule_sha256=sha, session_digest=capsule["session_digest"],
                      request_receipts=metered.receipts)
        MC.write_new(output, record)
        return review(root, capsule_path, output)
    except Exception as error:
        failure = output.with_suffix(".failure.json")
        if not failure.exists():
            MC.write_new(failure, {"schema": "selection_call_failure_v1", "capsule_sha256": sha,
                                   "error_type": type(error).__name__, "phase": "dialogue_or_review"})
        raise


def review(root: Path, capsule_path: Path, output: Path) -> dict:
    capsule, snapshot, view, messages = prepared(root, capsule_path)
    sha = MC.file_sha(capsule_path)
    require(Path(output).resolve() == inside(root, f"calls/{sha}.json"), "record 출력 경로 불일치")
    record = MC.read(output)
    budget = audit(root)
    rows = budget["groups"].get(sha, [])
    require(not budget["pending"] and record["requests"] == len(rows), "미완료 API 원장 또는 요청 수 불일치")
    raw = []
    expected_receipts = []
    for row in rows:
        reservation, response = row["reservation"], row["response"]
        raw.append(response["result"])
        expected_receipts.append({"request_index": reservation["request_index"],
            "reservation_sha256": MC.digest(reservation),
            "response_sha256": MC.file_sha(Path(root) / "api/responses" / f"{reservation['request_index']:04d}.json")})
    results = iter(raw)
    def replay(sent, params):
        index = replay.index
        if index >= len(rows):
            require((not rows and budget["remaining"] == 0)
                    or (bool(rows) and rows[-1]["reservation"]["request_index"] == MC.API_CAP),
                    "공용 상한 전에 조기 budget_exhausted 기록")
            raise MC.BudgetStop("기록된 요청 끝")
        require(rows[index]["reservation"]["messages_sha256"] == V.L.digest(sent)
                and params == capsule["params"], "실제 메시지·params 재계산 불일치")
        replay.index += 1
        return deepcopy(next(results))
    replay.index = 0
    expected = engine(snapshot, view, messages, capsule["condition"], replay)
    expected.update(capsule_sha256=sha, session_digest=capsule["session_digest"], request_receipts=expected_receipts)
    require(record == expected and replay.index == len(rows), "선택·피드백·비용·API receipt 재계산 불일치")
    return record


def exposure_summary(root: Path, full_path: Path, masked_path: Path) -> dict:
    full, a, va, _ = prepared(root, full_path)
    masked, b, vb, _ = prepared(root, masked_path)
    require(full["condition"] == "exposure_full" and masked["condition"] == "exposure_masked"
            and a["binding"] == b["binding"] and a["ledger_digest"] == b["ledger_digest"]
            and a["pre_receipt_anchor"] == b["pre_receipt_anchor"]
            and full["replicate"] == masked["replicate"], "다른 노출 query pair")
    def surface(view):
        return {key: val for key, val in view.items() if key not in ("evidence_facts", "observations")}
    require(surface(va) == surface(vb), "노출 query의 decision surface가 다름")
    outputs = [inside(root, f"calls/{MC.file_sha(path)}.json") for path in (full_path, masked_path)]
    result = {"schema": "selection_exposure_summary_v1", "used_for_decision": False,
              "pair_complete": all(path.is_file() for path in outputs), "view_sensitivity": None,
              "model_advantage": False, "recovery_comparison": False}
    if not result["pair_complete"]:
        return {**result, "disposition": "pair_incomplete"}
    records = [review(root, source, output) for source, output in zip((full_path, masked_path), outputs, strict=True)]
    valid = [row["selection"]["model_status"] == "valid" for row in records]
    choices = [(row["selection"]["action"], row["selection"]["probe"]) for row in records]
    latest = f"observation:{len(va['observations']) - 1}:"
    indices = [row["request_receipts"][0]["request_index"] if row["request_receipts"] else None for row in records]
    return {**result, "disposition": "auxiliary_preupdate_exposure", "contract_valid": valid,
            "view_sensitivity": choices[0] != choices[1] if all(valid) else None,
            "explicit_latest_fact_use": (any(key.startswith(latest) for key in records[0]["selection"]["fact_ids"])
                                         if valid[0] else None),
            "first_request_indices": indices, "requests": [row["requests"] for row in records]}


def validate_plan(value: dict, protocol_sha: str) -> dict:
    """입력 상태·반복·pair·순서·최악 요청 수를 결과 전에 고정한다."""
    required = {"schema", "run_id", "caller_protocol_sha", "model", "params", "api_cap", "conditions",
                "predictions", "result_wording", "trusted_host_input_required", "planned_calls"}
    require(set(value) == required and value["schema"] == "selection_call_execution_v1"
            and value["caller_protocol_sha"] == protocol_sha and value["model"] == MODEL
            and value["params"] == F.request_params(MODEL) and type(value["api_cap"]) is int
            and value["api_cap"] == 20 and isinstance(value["run_id"], str)
            and bool(re.fullmatch("[a-z0-9_]{1,80}", value["run_id"]))
            and isinstance(value["conditions"], list) and bool(value["conditions"])
            and all(isinstance(item, str) for item in value["conditions"])
            and len(set(value["conditions"])) == len(value["conditions"])
            and set(value["conditions"]) <= set(CONDITIONS)
            and value["trusted_host_input_required"] is True
            and isinstance(value["predictions"], dict) and bool(value["predictions"])
            and isinstance(value["result_wording"], dict) and bool(value["result_wording"])
            and isinstance(value["planned_calls"], list) and 1 <= len(value["planned_calls"]) <= 6,
            "실행 사전등록 범위")
    rows, pairs, states, names = value["planned_calls"], {}, set(), set()
    hashes = {"context_digest", "ledger_digest", "pre_receipt_anchor", "operational_view_digest",
              "view_digest", "history_sha256"}
    for index, row in enumerate(rows, 1):
        require(isinstance(row, dict) and set(row) == {"row_id", "pair_id", "order", "state", "max_requests"}
                and isinstance(row["row_id"], str) and bool(re.fullmatch("[a-z0-9_]{1,80}", row["row_id"]))
                and row["row_id"] not in names and type(row["order"]) is int and row["order"] == index
                and type(row["max_requests"]) is int and row["max_requests"] == 3
                and isinstance(row["state"], dict) and set(row["state"]) == hashes | {"condition", "replicate"},
                "계획 행·순서·요청 수 형식")
        state = row["state"]
        require(all(V.S._valid_sha(state[key]) for key in hashes) and state["condition"] in CONDITIONS
                and type(state["replicate"]) is int and 1 <= state["replicate"] <= 20
                and V.L.digest(state) not in states, "계획 입력 결합·중복")
        states.add(V.L.digest(state))
        names.add(row["row_id"])
        if state["condition"] == "production":
            require(row["pair_id"] is None, "production은 보조 pair가 아님")
        else:
            require(isinstance(row["pair_id"], str) and bool(re.fullmatch("[a-z0-9_]{1,80}", row["pair_id"])),
                    "보조 pair 식별자")
            pairs.setdefault(row["pair_id"], []).append(row)
    require(sum(row["max_requests"] for row in rows) <= value["api_cap"]
            and set(value["conditions"]) == {row["state"]["condition"] for row in rows}, "총 요청 배분·조건 불일치")
    full_first = 0
    for pair in pairs.values():
        require(len(pair) == 2 and {row["state"]["condition"] for row in pair}
                == {"exposure_full", "exposure_masked"}, "불완전 보조 pair 계획")
        common = [{key: val for key, val in row["state"].items() if key not in ("view_digest", "condition")}
                  for row in pair]
        require(common[0] == common[1], "pair 입력·반복 불일치")
        full_first += pair[0]["state"]["condition"] == "exposure_full"
    require(abs(2 * full_first - len(pairs)) <= 1, "보조 pair 순서 균형 불일치")
    return value


def execution_manifest(path: Path, anchor: str, protocol_sha: str) -> dict:
    verify_sources(protocol_sha, anchor)
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT / "bench/protocols") and path.suffix == ".json", "실행 사전등록 위치")
    raw = path.read_bytes()
    value = validate_plan(json.loads(raw), protocol_sha)
    relative = path.relative_to(ROOT).as_posix()
    for version in (f"{anchor}:{relative}", f":{relative}"):
        data = subprocess.run(["git", "show", version], cwd=ROOT, capture_output=True)
        require(data.returncode == 0 and data.stdout.replace(b"\r\n", b"\n") == raw.replace(b"\r\n", b"\n"),
                "실행 사전등록 commit/index 불일치")
    return value


def cleanup_worker(pid_file: Path, capsule_path: Path) -> str:
    """capsule·PID 시작 시각·새 worker argv가 같은 소유 프로세스만 정리한다."""
    if not Path(pid_file).is_file():
        return "unverified_no_pid"
    code = r'''import hashlib,json,os,signal,sys,time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text()); capsule=Path(sys.argv[2]); pid=r['pid']; proc=Path('/proc')/str(pid)
assert type(pid) is int and pid>1 and hashlib.sha256(capsule.read_bytes()).hexdigest()==r['packet_sha256']
def same():
 if not (proc/'stat').exists(): return False
 fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
 if int(fields[19])!=r['start_ticks']: raise RuntimeError('pid identity changed')
 args=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
 script=next((i for i,x in enumerate(args) if x.endswith('/selection_comparison_call_v1.py')),None)
 if script is None or args[script:]!=r['argv'] or str(capsule) not in args: raise RuntimeError('worker args differ')
 return True
if same():
 os.kill(pid,signal.SIGTERM)
 for _ in range(30):
  if not proc.exists(): break
  time.sleep(.1)
 if same(): os.kill(pid,signal.SIGKILL)
 for _ in range(30):
  if not proc.exists(): break
  time.sleep(.1)
 if proc.exists(): raise RuntimeError('worker remains')
'''
    command = '"$HOME/.venvs/rl-triage/bin/python" -c ' + shlex.join(
        [code, MC.wsl_path(pid_file), MC.wsl_path(capsule_path)])
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command], check=True,
                   capture_output=True, timeout=15)
    return "verified"


def launch(root: Path, capsule_path: Path, experiment: Path, anchor: str) -> dict:
    """Windows supervisor. 실제 호출은 별도 커밋 사전등록과 WSL worker가 있어야 한다."""
    root, capsule_path = Path(root).resolve(), Path(capsule_path).resolve()
    require(os.name == "nt", "Windows supervisor 경로만 지원")
    owner = session(root)
    execution = execution_manifest(experiment, anchor, owner["caller_protocol_sha"])
    require(not owner["fixture"] and owner["anchor"] == anchor
            and owner["experiment_sha"] == MC.file_sha(experiment)
            and execution["run_id"] == owner["run_id"], "실제 supervisor의 사전등록 결합 불일치")
    capsule, _snapshot, _view, _messages = prepared(root, capsule_path)
    require(capsule["condition"] in execution["conditions"], "실제 호출에 허용되지 않은 조건")
    sha = MC.file_sha(capsule_path)
    pid_file = inside(root, f"workers/{sha}.json")
    require(not pid_file.exists() and not inside(root, f"calls/{sha}.json").exists()
            and sha not in audit(root)["groups"], "이미 시작한 worker")
    args = [MC.wsl_path(ROOT / "evals/selection_comparison_call_v1.py"), "--protocol-sha", owner["caller_protocol_sha"],
            "--anchor", anchor, "--experiment", MC.wsl_path(experiment), "--batch-root", MC.wsl_path(root),
            "--capsule", MC.wsl_path(capsule_path), "--pid-file", MC.wsl_path(pid_file)]
    command = '. "$HOME/.config/nvidia/env" && exec "$HOME/.venvs/rl-triage/bin/python" -B ' + shlex.join(args)
    try:
        process = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
                                 capture_output=True, timeout=MC.TRANSPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        receipt = {"schema": "selection_call_transport_failure_v1", "phase": "worker_timeout", "cleanup": "unverified"}
        try:
            receipt["cleanup"] = cleanup_worker(pid_file, capsule_path)
        except Exception as error:
            receipt["cleanup_error_type"] = type(error).__name__
        MC.write_new(inside(root, f"calls/{sha}.transport_failure.json"), receipt)
        raise TimeoutError("공통뷰 worker의 소유 프로세스 정리 결과를 보존함") from None
    try:
        require(process.returncode == 0 and not pid_file.exists(), "worker 실패 또는 PID 파일 잔여")
        return review(root, capsule_path, inside(root, f"calls/{sha}.json"))
    except Exception as error:
        receipt = {"schema": "selection_call_transport_failure_v1", "phase": "worker_exit_or_review",
                   "returncode": process.returncode, "error_type": type(error).__name__, "cleanup": "not_needed"}
        if pid_file.exists():
            try:
                receipt["cleanup"] = cleanup_worker(pid_file, capsule_path)
            except Exception as cleanup_error:
                receipt.update(cleanup="unverified", cleanup_error_type=type(cleanup_error).__name__)
        MC.write_new(inside(root, f"calls/{sha}.transport_failure.json"), receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-sources", action="store_true")
    parser.add_argument("--protocol-sha", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--capsule", type=Path)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()
    verify_sources(args.protocol_sha, args.anchor)
    if args.verify_sources:
        print("source protocol/commit/index/worktree: verified")
        return 0
    require(sys.platform.startswith("linux") and bool(os.environ.get("WSL_DISTRO_NAME")) and args.experiment is not None
            and args.batch_root is not None and args.capsule is not None, "실제 caller는 WSL과 실행 사전등록이 필요함")
    execution = execution_manifest(args.experiment, args.anchor, args.protocol_sha)
    owner = initialize(args.batch_root, execution["run_id"], args.protocol_sha, fixture=False,
                       experiment_sha=MC.file_sha(args.experiment), anchor=args.anchor)
    capsule = MC.read(args.capsule)
    require(capsule["session_digest"] == V.L.digest(owner) and capsule["condition"] in execution["conditions"],
            "실제 실행의 session 또는 허용 조건 불일치")
    pid_file = inside(args.batch_root, f"workers/{MC.file_sha(args.capsule)}.json")
    require(args.pid_file is not None and args.pid_file.resolve() == pid_file, "worker PID 출력 위치 불일치")
    identity = MC._worker_identity(args.capsule)
    MC.write_new(pid_file, identity)
    try:
        complete = NvidiaCompletion(args.batch_root, args.capsule, args.experiment, args.anchor)
        decide(args.batch_root, args.capsule, complete)
    finally:
        require(MC.read(pid_file) == identity, "worker PID 소유자 변경")
        pid_file.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
