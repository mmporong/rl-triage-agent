"""적용성 계약을 거치는 기본 교정 CLI. 원장은 evals/loops/<케이스>/correction_ledger.jsonl(적용성 등록 표).

  python evals/correct_v2.py propose <케이스> --manifest <적용성 manifest JSON> --selection-actor model|registered_rule
                                     [--budget-gpu-s 7200]
  python evals/correct_v2.py approve <케이스> <request_id> --approver <이름>
  python evals/correct_v2.py reject  <케이스> <request_id> --approver <이름> --reason <이유>
  python evals/correct_v2.py status  <케이스>
  python evals/correct_v2.py replay   # 공개 loop 기록 네 건을 임시 폴더에서 재생(API·GPU·네트워크 없음)

- evals/correct.py는 공개 시연 프로토콜이 파일 SHA로 묶고 있어 그대로 둔다. 이 진입점은 그 loop 읽기
  (_loop_state·_confirming_request)를 재사용하고, 원장에 쓰기 전에 combination_contract_v1의 여섯 관문을 둔다.
- loop가 none_supported·unidentifiable이거나 확정 메커니즘이 등록 밖이면 manifest 없이 HOLD를 남긴다.
- 확정된 등록 메커니즘은 manifest가 있어야 한다. manifest의 결함 정책 체크포인트와 범위 관측은 loop 사전등록의
  체크포인트·확정 probe receipt와 같아야 하며, 다르면 원장에 쓰기 전에 ValueError로 멈춘다.
- manifest의 결함 계열이 loop 확정 메커니즘과 다르면 계약이 diagnosis_unconfirmed HOLD로 끝낸다(U1 함수군은 reward 계열).
- 선택 주체(model|registered_rule)는 loop 파일에 없으므로 호출자가 밝히고, 요청 근거와 digest에 묶인다.
- run은 실행 함수를 가진 실행기가 부른다(실행 함수 계약은 evals/correct.py와 같다). 소비 직전에 같은 근거로 관문을 다시 확인한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import combination_contract_v1 as K  # noqa: E402
import correct as CR  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger, LedgerError  # noqa: E402
from rl_triage.correction_registry import HOLD  # noqa: E402

LOOPS = CR.LOOPS
DEFAULT_BUDGET_GPU_S = CR.DEFAULT_BUDGET_GPU_S
ACTORS = ("model", "registered_rule")
# 계약 메커니즘 → loop 확정 범주. U1 함수군 교정은 P_reward로 확정한 reward loop에 연결한다.
LOOP_CATEGORY = {"reward": "reward", "reward_family": "reward", "actuator": "actuator", "termination": "termination"}

LOOP_FILES = ("prereg.json", "state.json", "ledger.jsonl")
FIXTURES = ROOT / "bench" / "fixtures" / "combination_contract_v1" / "cases"
# 공개 loop 기록 → (공개 manifest fixture, 선택 주체를 읽는 시연 결과). manifest가 없는 기록은 loop 종료만 재생한다.
REPLAYS = {
    "h02_s2080": ("go2_f1_ready", "evals/results/vl_aux_f1_s2080_20261016/demo_result.json"),
    "u1_s2040": ("go2_u1_legacy_scope_insufficient", "evals/results/e2e_demo_u1_s2040_20261013/demo_result.json"),
    "e01_s2027": (None, None),
    "h05_s2027": (None, None),
}
EXPECTED = {"h02_s2080": ("proposed", None), "u1_s2040": (HOLD, "scope_insufficient"),
            "e01_s2027": (HOLD, "loop status unidentifiable"), "h05_s2027": (HOLD, "unregistered_mechanism")}


def _ledger(case: str, loops: Path) -> CorrectionLedger:
    led = CorrectionLedger(CR._dir(case, loops) / "correction_ledger.jsonl", registry=K.registry())
    foreign = sorted({e["request_id"] for e in led.events()
                      if e["event"] == "proposed" and e.get("registry_digest") != led.registry_digest})
    if foreign:
        raise LedgerError(f"{case}: 다른 등록 표로 만든 요청 {foreign}이 있는 원장에는 쓰지 않는다")
    return led


def _bind(manifest: dict, prereg: dict, confirming: dict) -> None:
    """manifest가 이 loop의 결함 정책과 확정 관측을 가리키는지 원장에 쓰기 전에 확인한다."""
    stages = manifest.get("stages") if isinstance(manifest, dict) else None
    if not isinstance(stages, dict):
        raise ValueError("manifest에 stages가 없다")
    if "source" in stages:
        try:
            checkpoint = stages["source"]["fault"]["checkpoint"]["sha256"]
        except (KeyError, TypeError) as e:
            raise ValueError("manifest에 결함 정책 체크포인트가 없다") from e
        if checkpoint != prereg.get("checkpoint_sha256"):
            raise ValueError("manifest 결함 정책이 loop 사전등록 체크포인트와 다르다")
    scope = stages.get("scope")
    if isinstance(scope, dict) and scope.get("probe") is not None and (
            scope["probe"] != confirming["probe"]
            or scope.get("measurement") != confirming["receipt"].get("measurement")):
        raise ValueError("manifest 범위 관측이 loop 확정 probe receipt와 다르다")


def contract_state(case: str, manifest: dict, selection_actor: str | None, loops: Path = LOOPS) -> dict:
    """loop 기록을 계약의 고리 상태로 옮긴다. loop 상태 digest와 확정 probe 요청을 함께 묶는다."""
    if selection_actor not in ACTORS:
        raise ValueError(f"선택 주체는 {ACTORS} 중 하나다({selection_actor!r})")
    prereg, state, loop_led = CR._loop_state(case, loops)
    mech, req = CR._confirming_request(case, state, loop_led)
    _bind(manifest, prereg, req)
    fault = manifest.get("fault")
    family = K.FAULTS[fault][0] if isinstance(fault, str) and fault in K.FAULTS else None
    return {"status": state["status"], "hypotheses": [family] if LOOP_CATEGORY.get(family) == mech else [mech],
            "observed": dict(state.get("observed", {})), "selection_actor": selection_actor,
            "loop": {"case": case, "mechanism": mech, "loop_state_digest": CR.loop_state_digest(prereg, state),
                     "checkpoint_sha256": prereg.get("checkpoint_sha256"), "confirming_request_id": req["request_id"]}}


def propose(case: str, manifest: dict | None = None, selection_actor: str | None = None,
            budget_gpu_s: int = DEFAULT_BUDGET_GPU_S, loops: Path = LOOPS, root: Path = ROOT) -> dict:
    """확정된 loop에서 계약이 ready일 때만 교정 하나를 제안한다. 그 밖은 {"decision": "HOLD"}(요청 0)."""
    prereg, state, _ = CR._loop_state(case, loops)
    led = _ledger(case, loops)
    lsd = CR.loop_state_digest(prereg, state)
    status = state.get("status")
    if status in ("none_supported", "unidentifiable"):
        led.hold(lsd, "", f"loop status {status}")
        return {"decision": HOLD, "reason": f"loop status {status}"}
    if status != "confirmed" or len(state.get("hypotheses", [])) != 1:
        raise LedgerError(f"{case}: loop 상태가 confirmed가 아니다({status!r})")
    mech = state["hypotheses"][0]
    if mech not in set(LOOP_CATEGORY.values()):
        led.hold(lsd, mech, "unregistered_mechanism")
        return {"decision": HOLD, "reason": "unregistered_mechanism", "mechanism": mech}
    if manifest is None:
        raise ValueError(f"{case}: 확정된 {mech} 교정 제안에는 적용성 manifest가 필요하다")
    event = K.propose(manifest, contract_state(case, manifest, selection_actor, loops), led, root=root,
                      budget_gpu_s=budget_gpu_s)
    if event["event"] == "held":
        return {"decision": HOLD, "reason": event["reason"], "contract": event.get("evidence")}
    return {"decision": "proposed", **event}


def run(case: str, rid: str, execute, manifest: dict, selection_actor: str, loops: Path = LOOPS, root: Path = ROOT,
        tag: str | None = None) -> dict:
    led = _ledger(case, loops)
    if led.orphans():
        raise LedgerError(f"결과 없이 끝난 요청 {led.orphans()}이 있다. 먼저 recover로 닫는다")
    req = led.requests().get(rid)
    if req is None:
        raise LedgerError(f"{rid}: 없는 요청")
    # 관문 재확인·승인 없음·거절·재소비·loop 상태나 근거 변경이면 여기서 멈춘다.
    K.consume(manifest, contract_state(case, manifest, selection_actor, loops), led, rid, root=root)
    request = {"case": case, "request_id": rid, "correction_id": req["correction_id"],
               "recovery_mode": req["recovery_mode"], "protocol_case_arm": req["protocol_case_arm"],
               "budget_gpu_s": req["budget_gpu_s"], "tag": tag or f"correct_{rid}",
               "restoration_targets": req["evidence"]["restoration_targets"]}
    t0 = time.perf_counter()
    res = execute(request)  # 예외는 그대로 전파한다. 요청은 running(고아)으로 남는다.
    gpu_s = res.get("gpu_s", time.perf_counter() - t0)
    return led.receipt(rid, res.get("tag", request["tag"]), res["verdict"], res.get("result_sha"), gpu_s,
                       res.get("exit_code"))


def status(case: str, loops: Path = LOOPS) -> dict:
    led = _ledger(case, loops)
    return {"requests": led.requests(), "holds": [e for e in led.events() if e["event"] == "held"],
            "orphans": led.orphans()}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _fixture_manifest(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))["manifest"]


def _selection_actor(demo: dict, request_id: str) -> str:
    """시연 기록에서 확정 probe를 고른 주체를 읽는다."""
    started = demo["diagnosis"]["started"]
    if started.get("request_id") != request_id:
        raise ValueError("시연 기록의 probe 요청이 loop 확정 요청과 다르다")
    return "model" if started.get("model_probe_used") is True else "registered_rule"


def _summary(out: dict) -> dict:
    contract = out.get("contract") or out.get("evidence", {}).get("combination_contract")
    row = {"decision": out["decision"], "reason": out.get("reason"),
           "contract_status": None if contract is None else contract["status"],
           "trace": None if contract is None else [entry["stage"] for entry in contract["trace"]]}
    if out["decision"] == "proposed":
        row.update(correction_id=out["correction_id"], recovery_mode=out["recovery_mode"],
                   restoration_targets=out["evidence"]["restoration_targets"],
                   selection_actor=out["evidence"]["selection_actor"])
    return row


def replay(loops: Path = LOOPS, root: Path = ROOT) -> dict:
    """공개 loop 기록을 임시 폴더로 복사해 propose까지 재생한다. 저장소의 loop·원장에는 쓰지 않는다."""
    cases, inputs = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for case, (fixture, demo_path) in REPLAYS.items():
            (work / case).mkdir()
            for name in LOOP_FILES:
                shutil.copyfile(Path(loops) / case / name, work / case / name)
            used = {f"evals/loops/{case}/{name}": _sha(Path(loops) / case / name) for name in LOOP_FILES}
            manifest = actor = None
            if fixture is not None:
                manifest = _fixture_manifest(fixture)
                _, state, loop_led = CR._loop_state(case, work)
                _, req = CR._confirming_request(case, state, loop_led)
                actor = _selection_actor(json.loads((Path(root) / demo_path).read_text(encoding="utf-8")),
                                         req["request_id"])
                used[f"bench/fixtures/combination_contract_v1/cases/{fixture}.json"] = _sha(FIXTURES / f"{fixture}.json")
                used[demo_path] = _sha(Path(root) / demo_path)
            cases[case] = _summary(propose(case, manifest, actor, loops=work, root=root))
            inputs.update(used)
    ok = all((cases[case]["decision"], cases[case]["reason"] if want[1] else None) == want
             for case, want in EXPECTED.items())
    return {"schema": "correct_v2_replay_v1", "status": "pass" if ok else "fail", "cases": cases,
            "inputs_sha256_lf": inputs}


def _read_manifest(path: Path | None) -> dict | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    # 공개 fixture 파일은 manifest를 감싸 담는다.
    if isinstance(value, dict) and value.get("schema") == "combination_contract_history_fixture_v1":
        return value["manifest"]
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="적용성 계약을 거치는 교정 제안·승인")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("case")
    p.add_argument("--manifest", type=Path, help="호스트가 관측 원본으로 만든 적용성 manifest JSON")
    p.add_argument("--selection-actor", choices=ACTORS, help="확정 probe를 고른 주체")
    p.add_argument("--budget-gpu-s", type=int, default=DEFAULT_BUDGET_GPU_S)
    for name in ("approve", "reject"):
        q = sub.add_parser(name)
        q.add_argument("case")
        q.add_argument("request_id")
        q.add_argument("--approver", required=True)
        if name == "reject":
            q.add_argument("--reason", required=True)
    s = sub.add_parser("status")
    s.add_argument("case")
    sub.add_parser("replay", help="공개 loop 기록 네 건을 임시 폴더에서 재생")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "propose":
            out = propose(args.case, _read_manifest(args.manifest), args.selection_actor, args.budget_gpu_s)
        elif args.cmd == "approve":
            out = _ledger(args.case, LOOPS).approve(args.request_id, args.approver)
        elif args.cmd == "reject":
            out = _ledger(args.case, LOOPS).reject(args.request_id, args.approver, args.reason)
        elif args.cmd == "status":
            out = status(args.case)
        else:
            out = replay()
        print(json.dumps(out, ensure_ascii=False, indent=1))
    except (LedgerError, ValueError, OSError) as e:
        print(f"거부: {e}", file=sys.stderr)
        return 2
    return 1 if args.cmd == "replay" and out["status"] != "pass" else 0


if __name__ == "__main__":
    sys.exit(main())
