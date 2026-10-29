"""등록된 parent 입력에서 F1 case 검사 한 건만 실행한다.

기존 parent의 자원 입장, 물리 lease, GPU 원문, backend, evidence 봉인을
재사용한다. reference/NONE 실행과 모델 호출은 수행하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "AGENTS.md").is_file() and (parent / "bench").is_dir())
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in os.sys.path:
        os.sys.path.insert(0, str(_folder))

import run_agentic_brev_probe_parent_v1 as P

SCHEMA = "agentic_brev_probe_case_parent_v1"
SOURCE = "bench/run_agentic_brev_probe_case_v1.py"
TEST_SOURCE = "tests/test_run_agentic_brev_probe_case_v1.py"
SOURCES = frozenset(set(P.SOURCES) | {SOURCE, TEST_SOURCE})
ARM = "F1"


def require(ok: bool, reason: str) -> None:
    P.require(ok, reason)


def _verify_result(prepared: dict, result: dict) -> dict:
    require(result.get("schema") == SCHEMA + "_result"
            and result.get("status") in {"complete", "HOLD", "pending_unknown"}
            and result.get("reference_execution_count") == 0
            and result.get("model_api_calls") == 0
            and result.get("redispatch") is False,
            "case-only outcome contract")
    completed = result.get("completed")
    require(isinstance(completed, list) and len(completed) <= 1
            and (result["status"] != "complete" or len(completed) == 1),
            "case-only completion cardinality")
    if completed:
        expected = prepared["operations"][ARM]["op"]["operation_id"]
        require(completed[0]["operation_id"] == expected, "case operation changed")
        P._verify_completion(prepared, completed[0])
    return result


def lookup(prepared: dict) -> dict:
    folder = prepared["folder"]
    if not (folder / "intent.json").exists() or not (folder / "outcome.json").exists():
        return {"schema": SCHEMA + "_result", "status": "pending_unknown",
                "redispatch": False, "completed": [], "reference_execution_count": 0,
                "model_api_calls": 0}
    intent = P.E.J._read(folder / "intent.json")
    require(intent["request_ref"] == prepared["request_ref"]
            and intent["authority"] == prepared["authority"]
            and intent["order"] == [ARM], "existing case intent differs")
    return _verify_result(prepared, P.E.J._read(folder / "outcome.json"))


def run(request_ref: dict, *, execute: bool = False, allow_fixture: bool = False,
        home: Path | None = None, now=None, services=None) -> dict:
    prepared = P.prepare(request_ref, allow_fixture=allow_fixture, home=home, now=now)
    if prepared["status"] == "HOLD":
        return prepared
    if not prepared["fixture"]:
        require(SOURCES <= prepared["protocol"]["sources_lf"].keys(),
                "case wrapper source closure")
    require(prepared["operations"][ARM]["op"]["probe"] == "P_reward",
            "registered P_reward case only")
    if not prepared["fixture"]:
        require(prepared["snapshot"]["inputs"]["packet"]["input_bindings"]["case"] == {
            "checkpoint_sha256": P.U.INITIAL["case"]["checkpoint_sha256"],
            "telemetry_sha256": P.U.INITIAL["case"]["telemetry_sha256"],
        }, "s2221 case binding changed")
    folder = prepared["folder"]
    if folder.exists():
        return lookup(prepared)
    if not execute:
        return {"status": "prepared", "dispatched": False, "order": [ARM],
                "reference_execution_count": 0, "model_api_calls": 0,
                "authority_kind": "registered_consumed_pair_prefix_case_subset"}
    fixture = prepared["fixture"]
    require(fixture or (os.name == "posix" and services is None),
            "actual requires POSIX/default services")
    require(not fixture or (services is not None and services.evidence_kind == P.E.FIXTURE),
            "explicit CPU lifetime services required")
    services = services or P.ActualServices()
    folder.mkdir(parents=True, exist_ok=False)
    P.E.write_new(folder / "intent.json", {
        "schema": SCHEMA + "_intent", "request_ref": request_ref,
        "authority": prepared["authority"], "order": [ARM],
        "gpu_binding": P.GPU_BINDING, "reference_execution_count": 0,
        "model_api_calls": 0,
    })
    details = prepared["operations"][ARM]
    details["unit"].mkdir(parents=True, exist_ok=False)
    P.E.write_new(details["unit"] / "parent-intent.json", {
        "request_ref": request_ref, "pair_id": prepared["execution"]["pair_id"],
        "case_only": True, "reference_execution_count": 0,
    })
    completed: list[dict] = []
    result = {"schema": SCHEMA + "_result", "status": "pending_unknown",
              "redispatch": False, "completed": completed,
              "reference_execution_count": 0, "model_api_calls": 0,
              "host_completion_seal_required": True,
              "c_loop_update_permitted": False}
    try:
        with P._owner_scope(services):
            P.E.write_new(folder / "canonical-lease-root.json", services.observation())
            basis = services.validate_basis(prepared)
            if not fixture:
                require(all(basis["_recording_target"][key]
                            == prepared["protocol"]["provider_target"][key]
                            for key in ("resource_id", "resource_name", "instance_type")),
                        "lease/provider target mismatch")
            with P._acquire(prepared, services) as lease:
                snapshot = P.U.read_ref(lease)
                require(snapshot["resource_id"] == prepared["protocol"]["provider_target"]["resource_id"]
                        and snapshot["protocol_sha256_lf"] == prepared["basis_sha"],
                        "acquired lease resource/protocol")
                boot_ref = services.boot(folder / "boot.txt")

                def check() -> None:
                    require(P._OWNER == threading.get_ident(), "case lifetime owner thread")
                    before = services.verify(prepared, lease, basis)
                    require(before["lease_id"] == snapshot["lease_id"]
                            and before["resource_id"] == snapshot["resource_id"]
                            and before["evidence_kind"] == prepared["binding"]["evidence_kind"]
                            and (fixture or before["live_lock_proof"] is True),
                            "physical mutex inactive/lost/fixture")
                    logs = folder / "lease-observations"
                    log = logs / f"{len(list(logs.glob('*'))) + 1:05d}"
                    log.mkdir(parents=True, exist_ok=False)
                    services.record_lock(prepared, log, lease, before)
                    after = services.verify(prepared, lease, basis)
                    require(after["lease_id"] == before["lease_id"]
                            and after["owner"] == before["owner"],
                            "physical owner changed during observation")
                    P.E.write_new(log / "proof.json", {
                        "physical_mutex_only": True, "probe_authority_issued": False,
                        "lease": lease, "live": after,
                        "raw_sha256": {path.name: P.E.J.file_sha(path) for path in log.iterdir()},
                    })

                check()
                proof_folder, proof, capture_ref, capture, model_ref = P._proof(
                    prepared, ARM, lease, basis, services, check, boot_ref, None)
                plan, plan_ref = P._plan(prepared, ARM, proof_folder, proof, capture_ref,
                                         capture, model_ref, lease, boot_ref)
                completed.append(P._dispatch(prepared, ARM, plan, plan_ref, services, check))
                check()
                P._verify_completion(prepared, completed[0])
                result.update(status="complete", gpu_uuid=proof["gpu_uuid"],
                              boot_sha256=boot_ref["sha256"])
                _verify_result(prepared, result)
                check()
            result["lease_released"] = True
    except BaseException as error:  # noqa: BLE001 - 부분 원문 보존과 재실행 차단
        result.update(status="HOLD" if isinstance(error, Exception) else "pending_unknown",
                      error_type=type(error).__name__, reason=str(error), raw_preserved=True)
    if (folder / "lease-release.json").exists():
        result["lease_release"] = P.U.ref(folder / "lease-release.json")
    P.E.write_new(folder / "outcome.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = run({"path": str(args.request.resolve(strict=True)),
                  "sha256": args.request_sha256}, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"prepared", "complete"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
