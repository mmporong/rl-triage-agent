"""검증된 학습 observer 기록을 승인된 CPU probe receipt로 재사용한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import optimizer_runtime_facts_v1 as FACT
import runtime_training_inputs_v1 as INPUT

PLAN_SCHEMA = "recorded_optimizer_probe_plan_v1"
AUTH_SCHEMA = "recorded_optimizer_probe_authorization_v1"
RECEIPT_SCHEMA = "recorded_optimizer_probe_receipt_v1"
LEDGER_SCHEMA = "recorded_optimizer_probe_ledger_event_v1"
PROBE = "P_recorded_optimizer_updates"
EVIDENCE_KINDS = {"actual_training_reuse", "development_fixture"}
PARAMS = {"updates": [0, 1, 2], "source": "verified_training_observer_records",
          "new_training": False, "gpu_execution": False}
_LOCK = RLock()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda item: (
        _ for _ in ()).throw(ValueError(item)))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    require(not path.exists(), f"{path.name}은 fresh 파일이어야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def _runtime_checkpoint(path: Path) -> str:
    value = read(path)
    checkpoint = value.get("envelope", {}).get("source_binding", {}).get("checkpoint_sha256")
    require(isinstance(checkpoint, str) and re.fullmatch(r"[0-9a-f]{64}", checkpoint) is not None,
            "runtime artifact checkpoint binding이 없다")
    return checkpoint


def _source(
    collected_input: Path, runtime_artifact: Path, role: str,
) -> dict[str, Any]:
    require(role in {"case", "reference"}, "role은 case 또는 reference여야 한다")
    collected_input, runtime_artifact = (Path(collected_input).resolve(),
                                         Path(runtime_artifact).resolve())
    collected, input_metadata = INPUT.checked(collected_input)
    checkpoint = _runtime_checkpoint(runtime_artifact)
    envelope, facts = FACT.checked(runtime_artifact, expected_checkpoint_sha256=checkpoint)
    require(collected["source_binding"]["checkpoint_sha256"] == checkpoint,
            "collected와 runtime artifact checkpoint가 다르다")
    fact_metadata_path = runtime_artifact.parent / FACT.METADATA_NAME
    artifact_value = read(runtime_artifact)
    metadata_ref = artifact_value.get("private_metadata_ref")
    require(isinstance(metadata_ref, dict)
            and metadata_ref.get("file") == FACT.METADATA_NAME
            and metadata_ref.get("sha256") == sha(fact_metadata_path),
            "runtime fact metadata ref가 바뀌었다")
    fact_metadata = read(fact_metadata_path)
    sources = fact_metadata.get("sources")
    require(isinstance(sources, dict) and fact_metadata.get("role") == role,
            "runtime fact metadata role/source가 다르다")
    sources = cast(dict[str, str], sources)
    require(Path(sources["manifest"]).resolve() == Path(input_metadata["manifest"]).resolve()
            and Path(sources["owned_result"]).resolve() == Path(input_metadata["owned_result"]).resolve()
            and Path(sources["training_evidence"]).resolve().parent
            == Path(input_metadata["evidence_dir"]).resolve(),
            "collected와 runtime facts가 같은 학습 원문을 가리키지 않는다")
    raw_sha = fact_metadata.get("raw_source_sha256")
    if not isinstance(raw_sha, dict):
        raise TypeError("runtime fact raw source SHA 형식")
    require(set(raw_sha) >= {"manifest", "guard_records", "observer_records", "checkpoint"},
        "runtime fact raw source SHA가 부족하다")
    require(set(raw_sha) == set(sources)
            and all(Path(sources[name]).is_file()
                    and sha(Path(sources[name])) == expected
                    for name, expected in raw_sha.items()),
            "runtime fact raw source bytes가 metadata SHA와 다르다")
    private_training = input_metadata.get("private_training")
    if not isinstance(private_training, dict):
        raise TypeError("collected private training 형식")
    require(type(private_training.get("seed")) is int
            and isinstance(private_training.get("arm"), str),
            "collected private training seed/arm이 없다")
    return {
        "role": role, "checkpoint_sha256": checkpoint,
        "seed": private_training["seed"], "source_arm": private_training["arm"],
        "telemetry_sha256": collected["source_binding"]["telemetry_sha256"],
        "collected_binding_digest": collected["source_binding"]["binding_digest"],
        "runtime_binding_digest": envelope["source_binding"]["binding_digest"],
        "facts": facts,
        "raw_source_sha256": deepcopy(raw_sha),
        "collected_ref": {"path": str(collected_input), "sha256": sha(collected_input)},
        "runtime_ref": {"path": str(runtime_artifact), "sha256": sha(runtime_artifact)},
    }


def _ledger(path: Path) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    if not path.exists():
        return []
    rows = []
    previous = None
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        require(isinstance(row, dict) and row.get("schema") == LEDGER_SCHEMA
                and row.get("sequence") == index and row.get("previous_digest") == previous,
                "recorded probe ledger chain이 다르다")
        claimed = row.get("event_digest")
        body = {key: value for key, value in row.items() if key != "event_digest"}
        require(claimed == digest(body), "recorded probe ledger event digest가 다르다")
        previous = claimed
        rows.append(row)
    return rows


def _append(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _ledger(path)
    body = {"schema": LEDGER_SCHEMA, "sequence": len(rows) + 1,
            "previous_digest": rows[-1]["event_digest"] if rows else None, **event}
    row = {**body, "event_digest": digest(body)}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(row).decode("utf-8"))
    return row


def _events(ledger_path: Path, request_id: str) -> list[dict[str, Any]]:
    return [row for row in _ledger(ledger_path) if row.get("request_id") == request_id]


def prepare(
    collected_input: Path,
    runtime_artifact: Path,
    *,
    role: str,
    request_id: str,
    params: dict[str, Any],
    ledger_path: Path,
    evidence_kind: str,
    out: Path,
) -> dict[str, Any]:
    """실행 없이 검증된 observer 재사용 계획을 등록한다."""
    require(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", request_id) is not None,
            "request ID 형식")
    require(params == PARAMS, "recorded optimizer probe params가 고정 계약과 다르다")
    require(evidence_kind in EVIDENCE_KINDS, "fixture/actual evidence kind가 다르다")
    out, ledger_path = Path(out).resolve(), Path(ledger_path).resolve()
    require(not out.exists(), "recorded probe plan out은 fresh 디렉터리여야 한다")
    source = _source(collected_input, runtime_artifact, role)
    plan = {
        "schema": PLAN_SCHEMA, "probe": PROBE, "role": role, "request_id": request_id,
        "params": deepcopy(params), "params_digest": digest(params),
        "evidence_kind": evidence_kind, "source": source,
        "ledger_path": str(ledger_path),
        "scope": {"cpu_reuse_only": True, "new_training": False, "gpu_execution": False,
                  "new_observation": False, "recovery_claim": False},
    }
    plan_path = out / "plan.json"
    with _LOCK:
        require(not _events(ledger_path, request_id), "request ID는 일회용이어야 한다")
        out.mkdir(parents=True, exist_ok=False)
        write_new(plan_path, plan)
        _append(ledger_path, {"event": "proposed", "request_id": request_id,
                              "plan_sha256": sha(plan_path),
                              "params_digest": plan["params_digest"]})
    return deepcopy(plan)


def _verify_plan(plan_path: Path) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("probe") == PROBE
            and plan.get("params") == PARAMS
            and plan.get("params_digest") == digest(PARAMS)
            and plan.get("evidence_kind") in EVIDENCE_KINDS,
            "recorded probe plan schema/params가 다르다")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise TypeError("recorded probe plan source가 없다")
    expected = _source(Path(source["collected_ref"]["path"]),
                       Path(source["runtime_ref"]["path"]), plan["role"])
    require(source == expected, "recorded probe plan이 현재 검증 원문과 다르다")
    require(Path(plan["ledger_path"]).is_absolute(), "ledger path는 절대경로여야 한다")
    return plan


def authorize(plan_path: Path, *, approved_by: str, out: Path) -> dict[str, Any]:
    """명시적 host 승인 파일을 만들고 proposed 요청을 approved로 전이한다."""
    require(isinstance(approved_by, str) and bool(approved_by.strip()), "승인 주체가 없다")
    plan_path, out = Path(plan_path).resolve(), Path(out).resolve()
    plan = _verify_plan(plan_path)
    ledger_path = Path(plan["ledger_path"])
    with _LOCK:
        events = _events(ledger_path, plan["request_id"])
        require([row["event"] for row in events] == ["proposed"], "승인 전 원장 상태가 다르다")
        require(events[0]["plan_sha256"] == sha(plan_path), "proposed plan SHA가 다르다")
        authorization = {
            "schema": AUTH_SCHEMA, "approved": True, "approved_by": approved_by,
            "request_id": plan["request_id"], "probe": PROBE, "role": plan["role"],
            "plan": {"path": str(plan_path), "sha256": sha(plan_path)},
            "params_digest": plan["params_digest"],
            "source_binding": {
                key: plan["source"][key] for key in (
                    "seed", "source_arm", "checkpoint_sha256", "telemetry_sha256",
                    "collected_binding_digest", "runtime_binding_digest")},
        }
        write_new(out, authorization)
        _append(ledger_path, {"event": "approved", "request_id": plan["request_id"],
                              "plan_sha256": sha(plan_path), "authorization_sha256": sha(out),
                              "params_digest": plan["params_digest"]})
    return deepcopy(authorization)


def _authorization(path: Path, plan_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    value = read(path)
    require(value.get("schema") == AUTH_SCHEMA and value.get("approved") is True
            and value.get("request_id") == plan["request_id"]
            and value.get("probe") == PROBE and value.get("role") == plan["role"]
            and value.get("plan") == {"path": str(plan_path), "sha256": sha(plan_path)}
            and value.get("params_digest") == plan["params_digest"],
            "recorded probe authorization 결합이 다르다")
    expected_binding = {key: plan["source"][key] for key in (
        "seed", "source_arm", "checkpoint_sha256", "telemetry_sha256",
        "collected_binding_digest", "runtime_binding_digest")}
    require(value.get("source_binding") == expected_binding, "authorization source binding이 다르다")
    return value


def _measurement(source: dict[str, Any], role: str) -> dict[str, Any]:
    facts = source["facts"]
    prefix = f"runtime:{role}:"
    require(facts[prefix + "available"] is True, "recorded probe에는 available 원문이 필요하다")
    expected_steps = facts[prefix + "expected_optimizer_steps"]
    expected_batches = facts[prefix + "expected_minibatches"]
    return {
        "updates_observed": facts[prefix + "updates_observed"],
        "expected_optimizer_steps": expected_steps,
        "optimizer_steps_min": facts[prefix + "optimizer_steps_min"],
        "optimizer_steps_max": facts[prefix + "optimizer_steps_max"],
        "step_count_matches_config": (
            facts[prefix + "optimizer_steps_min"]
            == facts[prefix + "optimizer_steps_max"] == expected_steps),
        "expected_minibatches": expected_batches,
        "minibatches_min": facts[prefix + "minibatches_min"],
        "minibatches_max": facts[prefix + "minibatches_max"],
        "minibatch_count_matches_config": (
            facts[prefix + "minibatches_min"]
            == facts[prefix + "minibatches_max"] == expected_batches),
    }


def execute(plan_path: Path, authorization_path: Path, *, out: Path) -> dict[str, Any]:
    """승인된 일회 요청을 consume하고 검증된 기존 기록만 receipt로 투영한다."""
    plan_path, authorization_path, out = (Path(plan_path).resolve(),
                                           Path(authorization_path).resolve(), Path(out).resolve())
    plan = _verify_plan(plan_path)
    authorization = _authorization(authorization_path, plan_path, plan)
    ledger_path = Path(plan["ledger_path"])
    measurement = _measurement(plan["source"], plan["role"])
    outcome = ("normal" if measurement["step_count_matches_config"] is True
               and measurement["minibatch_count_matches_config"] is True else "abnormal")
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "complete", "probe": PROBE,
        "outcome": outcome,
        "role": plan["role"], "request_id": plan["request_id"],
        "evidence_kind": plan["evidence_kind"], "params_digest": plan["params_digest"],
        "plan": {"path": str(plan_path), "sha256": sha(plan_path)},
        "authorization": {"path": str(authorization_path), "sha256": sha(authorization_path)},
        "source_binding": deepcopy(authorization["source_binding"]),
        "raw_source_sha256": deepcopy(plan["source"]["raw_source_sha256"]),
        "measurement": measurement,
        "scope": {"reused_verified_training_records": True, "new_training": False,
                  "gpu_execution": False, "new_observation": False, "recovery_claim": False},
    }
    with _LOCK:
        events = _events(ledger_path, plan["request_id"])
        require([row["event"] for row in events] == ["proposed", "approved"],
                "consume 전 원장 상태가 다르다")
        require(all(row.get("plan_sha256") == sha(plan_path)
                    and row.get("params_digest") == plan["params_digest"] for row in events)
                and events[1]["authorization_sha256"] == sha(authorization_path),
                "approved plan/params/authorization 결합이 다르다")
        _append(ledger_path, {"event": "consumed", "request_id": plan["request_id"],
                              "plan_sha256": sha(plan_path),
                              "authorization_sha256": sha(authorization_path),
                              "params_digest": plan["params_digest"]})
        write_new(out, receipt)
        _append(ledger_path, {"event": "receipt", "request_id": plan["request_id"],
                              "plan_sha256": sha(plan_path), "receipt_sha256": sha(out),
                              "status": "complete", "params_digest": plan["params_digest"]})
    return deepcopy(receipt)


def checked_receipt(plan_path: Path, receipt_path: Path) -> dict[str, Any]:
    """현재 원문·승인·done ledger와 receipt를 모두 다시 검증한다."""
    plan_path, receipt_path = Path(plan_path).resolve(), Path(receipt_path).resolve()
    plan = _verify_plan(plan_path)
    receipt = read(receipt_path)
    auth_ref = receipt.get("authorization")
    if not isinstance(auth_ref, dict):
        raise TypeError("receipt authorization ref가 없다")
    authorization_path = Path(auth_ref["path"]).resolve()
    authorization = _authorization(authorization_path, plan_path, plan)
    measurement = _measurement(plan["source"], plan["role"])
    outcome = ("normal" if measurement["step_count_matches_config"] is True
               and measurement["minibatch_count_matches_config"] is True else "abnormal")
    expected = {
        "schema": RECEIPT_SCHEMA, "status": "complete", "probe": PROBE,
        "outcome": outcome,
        "role": plan["role"], "request_id": plan["request_id"],
        "evidence_kind": plan["evidence_kind"], "params_digest": plan["params_digest"],
        "plan": {"path": str(plan_path), "sha256": sha(plan_path)},
        "authorization": {"path": str(authorization_path), "sha256": sha(authorization_path)},
        "source_binding": deepcopy(authorization["source_binding"]),
        "raw_source_sha256": deepcopy(plan["source"]["raw_source_sha256"]),
        "measurement": measurement,
        "scope": {"reused_verified_training_records": True, "new_training": False,
                  "gpu_execution": False, "new_observation": False, "recovery_claim": False},
    }
    require(receipt == expected, "recorded optimizer receipt가 현재 검증 원문과 다르다")
    events = _events(Path(plan["ledger_path"]), plan["request_id"])
    require([row["event"] for row in events] == ["proposed", "approved", "consumed", "receipt"]
            and all(row.get("plan_sha256") == sha(plan_path)
                    and row.get("params_digest") == plan["params_digest"] for row in events)
            and events[1].get("authorization_sha256") == sha(authorization_path)
            and events[2].get("authorization_sha256") == sha(authorization_path)
            and events[-1].get("receipt_sha256") == sha(receipt_path),
            "recorded optimizer request가 done ledger 상태가 아니다")
    return deepcopy(expected)
