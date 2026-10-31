"""이미 본 비회복 사례에서 갱신 수치 제공 전후의 다음 검사 제안을 비교한다.

이 모듈은 검사·승인·교정·GPU 실행을 하지 않는다. 개발 비교이며 회복률 또는
새 seed 성능을 추정하지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_post_correction_inputs_v1 as POST
import optimizer_runtime_facts_v1 as FACT
import runtime_observation_view_v1 as RV
import selection_comparison_controller_v1 as C

SCHEMA = "runtime_observation_development_protocol_v1"
INPUT_SCHEMA = "runtime_observation_development_inputs_v1"
CONDITIONS = ("telemetry_only", "telemetry_and_runtime")
RUN_ID = "runtime_observation_dev_s2307_v1"
SOURCE_FILES = tuple(sorted({
    *RV.SOURCE_FILES, *C.SOURCE_FILES,
    "evals/runtime_observation_development_v1.py",
    "tests/test_runtime_observation_development_v1.py",
    "evals/lineage_post_correction_inputs_v1.py",
    "evals/lineage_recovery_inputs_v1.py",
    "evals/lineage_selection_inputs_v1.py",
    "evals/lineage_nim_transport_v1.py",
    "evals/lineage_nim_transport_v2.py",
    "evals/lineage_nim_transport_v3.py",
    "tests/test_lineage_nim_transport_v3.py",
}))


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON 객체가 필요하다")
    return value


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def ref(path: Path) -> dict:
    path = Path(path).resolve()
    require(path.is_file(), "근거 파일이 없다")
    return {"path": str(path), "sha256": sha(path)}


def condition_packet(base: dict, runtime_facts: dict, condition: str) -> dict:
    require(condition in CONDITIONS, "등록되지 않은 조건")
    require(set(runtime_facts) == set(FACT.FACT_IDS), "runtime fact ID 16개 결합이 다르다")
    require(not any(key.startswith("runtime:") for key in base["facts"]),
            "원래 packet에 runtime fact가 이미 있다")
    packet = deepcopy(base)
    if condition == "telemetry_and_runtime":
        packet["facts"].update(runtime_facts)
    return C.S.validate_packet(packet)


def materialize(post_path: Path, case_runtime: Path) -> dict:
    post, _telemetry = POST.checked_observation(post_path)
    reference = Path(post["private_metadata"]["sources"]["reference_collected"])
    envelope = POST.make_selection_packet(post_path, reference)
    base = envelope["packet"]
    runtime_envelope, facts = FACT.checked(
        case_runtime, expected_checkpoint_sha256=base["input_bindings"]["case"]["checkpoint_sha256"])
    require(all(key.startswith("runtime:case:") for key in facts), "case runtime 역할이 다르다")
    all_facts = {**facts, **FACT.unavailable_facts("reference")}
    return {
        "schema": INPUT_SCHEMA,
        "source_refs": {"post": ref(post_path), "case_runtime": ref(case_runtime),
                        "reference": ref(reference)},
        "runtime_binding": runtime_envelope["source_binding"],
        "reference_runtime_availability": "unavailable_no_same_checkpoint_monitor",
        "conditions": {name: condition_packet(base, all_facts, name) for name in CONDITIONS},
        "scope": {"already_seen_seed": 2307, "development_only": True,
                  "new_seed_evaluation": False, "approval": False, "gpu_execution": False},
    }


def prepare_inputs(post_path: Path, case_runtime: Path, out: Path) -> dict:
    value = materialize(Path(post_path).resolve(), Path(case_runtime).resolve())
    write(Path(out), value)
    return value


def checked_inputs(path: Path) -> dict:
    value = read(path)
    refs = value["source_refs"]
    for item in refs.values():
        require(sha(Path(item["path"])) == item["sha256"], "개발 입력 원문 SHA가 다르다")
    expected = materialize(Path(refs["post"]["path"]), Path(refs["case_runtime"]["path"]))
    require(value == expected, "개발 packet이 검증 원문에서 재계산한 값과 다르다")
    return value


def prepare_protocol(transport_protocol: Path, inputs: Path) -> dict:
    checked_inputs(inputs)
    transport = Path(transport_protocol).resolve()
    require(transport.is_relative_to(ROOT), "공개 transport protocol이 필요하다")
    return {
        "schema": SCHEMA, "purpose": "posthoc_same_input_next_probe_development_comparison",
        "conditions": list(CONDITIONS), "order": list(CONDITIONS),
        "input_sha256": sha(inputs),
        "components": {
            "view": {"path": RV.PROTOCOL_RELATIVE, "sha256_lf": sha(RV.PROTOCOL, lf=True)},
            "transport": {"path": transport.relative_to(ROOT).as_posix(),
                          "sha256_lf": sha(transport, lf=True)},
        },
        "source_sha256_lf": {name: sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "model": C.CALL.MODEL, "params": C.CALL.F.request_params(C.CALL.MODEL),
        "budget": {"logical_choices": 2, "max_requests_each": 3, "global_request_cap": 6,
                   "gpu_jobs": 0, "correction_jobs": 0},
        "predictions": {"telemetry_only": "optimizer_probe_may_be_missed",
                        "telemetry_and_runtime": "optimizer_probe_expected_with_runtime_fact_citation"},
        "criteria": {
            "runtime_observation_used": "valid P_optimizer_updates choice cites at least one runtime:case: fact",
            "changed_next_probe": "both choices valid and selected probe differs",
            "unchanged_or_invalid": "preserve actual result; no semantic retry",
        },
        "wording": {
            "success": "이미 본 사례의 개발 비교에서 갱신 계측 수치를 추가한 입력이 옵티마이저 검사를 선택했다.",
            "negative": "이미 본 사례의 개발 비교에서 갱신 계측 추가가 기대한 검사 선택으로 이어지지 않았다.",
            "scope": "입력 보강과 모델 선택의 개발 관찰이며 버전 우위·회복 효과·일반화 근거가 아니다.",
        },
        "scope": {"new_seed": False, "posthoc_design": True, "approval": False,
                  "gpu_execution": False, "old_measurement_overwrite": False},
    }


def verify_protocol(path: Path, inputs: Path, anchor: str) -> dict:
    value = read(path)
    transport = ROOT / value["components"]["transport"]["path"]
    require(value == prepare_protocol(transport, inputs), "개발 protocol 내용·입력 결합이 다르다")
    bound = {**value["source_sha256_lf"], path.resolve().relative_to(ROOT).as_posix(): sha(path, lf=True),
             **{item["path"]: item["sha256_lf"] for item in value["components"].values()}}
    result = subprocess.run(["git", "merge-base", "--is-ancestor", anchor, "HEAD"], cwd=ROOT, check=False)
    require(result.returncode == 0, "개발 anchor가 HEAD 조상이 아니다")
    for name, digest in bound.items():
        require(sha(ROOT / name, lf=True) == digest, "개발 source 현재 SHA가 다르다")
        blob = subprocess.run(["git", "show", f"{anchor}:{name}"], cwd=ROOT, capture_output=True, check=False)
        require(blob.returncode == 0 and hashlib.sha256(blob.stdout.replace(b"\r\n", b"\n")).hexdigest()
                == digest, "개발 source anchor blob이 다르다")
    RV.verify_protocol(RV.PROTOCOL, anchor)
    return value


def select(snapshot: dict, complete: Callable) -> tuple[dict, dict, dict]:
    inputs = C.dialogue_input(snapshot)
    require(inputs["needs_model"], "이 상태에는 모델 선택이 필요하지 않다")
    engine = C.CALL.engine(snapshot, inputs["view"], inputs["messages"], "production", complete)
    dialogue = {"schema": "selection_controller_dialogue_v1",
                "input_digest": C.L.digest(inputs),
                "safe_results": [C.CALL.safe_result(row["result"]) for row in engine["calls"]],
                "engine": engine}
    choice = C.choose(snapshot, "model_adaptive", dialogue)
    spec = C.request_spec(snapshot, choice) if choice["action"] == "continue" else None
    return choice, engine, {"dialogue_input": inputs, "request_spec": spec}


def run(protocol_path: Path, inputs_path: Path, anchor: str, queue: Path, out: Path) -> dict:
    import lineage_nim_transport_v3 as T

    protocol = verify_protocol(protocol_path, inputs_path, anchor)
    inputs = checked_inputs(inputs_path)
    require(not out.exists(), "개발 output은 fresh 경로여야 한다")
    out.mkdir(parents=True)
    results = []
    try:
        with RV.view_scope(RV.PROTOCOL, anchor):
            for condition in CONDITIONS:
                checked_inputs(inputs_path)
                packet = inputs["conditions"][condition]
                parent = {"run_id": RUN_ID, "policy": "model_adaptive", "epoch": 1,
                          **packet["prior_receipt"]}
                snapshot = C.V.build_snapshot(packet, RUN_ID, "model_adaptive",
                    RV.file_digest(RV.PROTOCOL), out / condition / "probes.ledger.jsonl", parent=parent)
                write(out / condition / "snapshot.json", snapshot)
                complete = T.file_completion(queue, condition, C.L.digest(snapshot["binding"]),
                                             snapshot["view_digest"], 180)
                choice, engine, audit = select(snapshot, complete)
                row = {"condition": condition, "choice": choice, "engine": engine, **audit,
                       "approval": False, "probe_executed": False}
                write(out / condition / "result.json", row)
                results.append(row)
        a, b = results
        valid = lambda row: row["choice"]["model_status"] == "valid" and row["choice"]["action"] == "continue"
        runtime_used = (valid(b) and b["choice"]["probe"] == "P_optimizer_updates"
                        and any(item.startswith("runtime:case:") for item in b["choice"]["fact_ids"]))
        summary = {"schema": "runtime_observation_development_result_v1", "status": "complete",
                   "input_sha256": sha(inputs_path), "protocol_sha256": sha(protocol_path),
                   "runtime_observation_used": runtime_used,
                   "changed_next_probe": bool(valid(a) and valid(b) and a["choice"]["probe"] != b["choice"]["probe"]),
                   "conditions": [{"condition": row["condition"], "choice": row["choice"],
                                   "requests": row["engine"]["requests"]} for row in results],
                   "wording": protocol["wording"]["success" if runtime_used else "negative"],
                   "scope": protocol["scope"]}
        write(out / "result.json", summary)
        return summary
    except BaseException as error:
        write(out / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                                     "completed_conditions": len(results)})
        raise
    finally:
        T.close(queue, "development_measurement_finished")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-inputs")
    prepare.add_argument("--post", type=Path, required=True)
    prepare.add_argument("--case-runtime", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    register = sub.add_parser("prepare-protocol")
    register.add_argument("--transport-protocol", type=Path, required=True)
    register.add_argument("--inputs", type=Path, required=True)
    register.add_argument("--out", type=Path, required=True)
    execute = sub.add_parser("run")
    for name in ("protocol", "inputs", "queue", "out"):
        execute.add_argument("--" + name, type=Path, required=True)
    execute.add_argument("--anchor", required=True)
    args = parser.parse_args()
    if args.command == "prepare-inputs":
        prepare_inputs(args.post, args.case_runtime, args.out)
    elif args.command == "prepare-protocol":
        write(args.out, prepare_protocol(args.transport_protocol, args.inputs))
    else:
        run(args.protocol, args.inputs, args.anchor, args.queue, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
