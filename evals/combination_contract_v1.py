"""로봇·결함·등록 교정의 적용성 관문과 기존 교정 원장을 연결한다.

manifest는 호스트가 출처를 읽어 구성한다. 모델이 ready나 교정 인자를 지정하지 않는다.
과거 기록의 재생은 계약 개발이며 새 조합의 실측 결과와 구분한다.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rl_triage import correction_scope as S
from rl_triage import grounded_recovery as C
from rl_triage import probe_loop as L
from rl_triage.correction_ledger import CorrectionLedger, LedgerError
import correction_adapter as A
import reward_family_correction as F
import e2e_demo_v4 as E

SCHEMA = "combination_applicability_manifest_v1"
ORDER = ("applicability", "injection", "source", "scope", "noop", "recovery")
ROBOTS = {"Go2": ("Isaac-Velocity-Flat-Unitree-Go2-v0", "fixed_eval_v1", 0.25,
                   "bench/protocols/fixed_eval_v1.json"),
          "ANYmal-C": ("Isaac-Velocity-Flat-Anymal-C-v0", "fixed_eval_v1_anymal_c", 0.5,
                       "bench/protocols/fixed_eval_v1_anymal_c.json")}
FAULTS = {"F1": ("reward", ("track_lin_vel_xy_exp",)),
          "F2": ("actuator", ("DCMotor._clip_effort",)),
          "F6": ("termination", ("time_out",)),
          "U1": ("reward_family", F.FUNCTIONS)}
SHA = re.compile(r"[0-9a-f]{64}")
BANDS = {
    "Go2": ("evals/results/p0c_fixed_eval_20261009/summary.json", ("bands",),
            "85c51c4b618370d3fc7926893db60f34f72f5617f8744f1440eb1bf288cfdf24"),
    "ANYmal-C": ("evals/results/anymal_c_band_20261014/result.json", ("analysis", "bands"),
                 "9db68f194fadf89faa1fdb29ad0a1e5d590522f7f8f9fc986c2d609ec0e67798"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def registered_bands(robot: str) -> dict:
    path, keys, expected_sha = BANDS[robot]
    if sha(ROOT / path) != expected_sha:
        raise ValueError("기존 행동 밴드 원본 결합이 다르다")
    row = json.loads((ROOT / path).read_text(encoding="utf-8"))
    for key in keys:
        row = row[key]
    return row


def registry() -> dict:
    """기존 v4 등록 표와 이미 등록된 U1 함수군 교정을 원장 형식으로 연결한다."""
    result = E.registry_v3()
    result["reward_family"] = {"correction_id": F.CORRECTION_ID,
        "adapter_target": F.MODULE + ":" + ",".join(F.REGISTRY[F.CORRECTION_ID]["qualnames"]),
        "protocol_case": "combination_u1_family_v1", "recovery_modes": ["restart"]}
    return result


def target_paths(correction_id: str) -> dict[str, str]:
    if correction_id == F.CORRECTION_ID:
        return {name: F.MODULE + ":" + name for name in F.FUNCTIONS}
    spec = A.TARGETS[correction_id]
    return {spec["qualname"]: spec["module"] + ":" + spec["qualname"]}


def fault_paths(fault: str) -> dict[str, str]:
    mechanism, _ = FAULTS[fault]
    return target_paths(registry()[mechanism]["correction_id"])


def make_manifest(robot: str, task: str, fault: str, correction_id: str,
                  stages: dict, evidence: list[dict]) -> dict:
    value = {"schema": SCHEMA, "robot": robot, "task": task, "fault": fault,
             "correction_id": correction_id, "stages": stages, "evidence": evidence}
    # JSON 밖 객체와 NaN을 통과시키지 않고 호출자의 가변 객체와 분리한다.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _bindings(manifest: dict, root: Path) -> None:
    refs = manifest["evidence"]
    if not isinstance(refs, list) or not refs:
        raise ValueError("원본 SHA 결합이 없다")
    seen = set()
    observations = []
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256_lf"}:
            raise ValueError("원본 참조 필드가 다르다")
        name = ref["path"]
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or Path(name).is_absolute() or ".." in Path(name).parts or name in seen):
            raise ValueError("원본 상대경로가 유효하지 않다")
        seen.add(name)
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("원본 경로가 계약 root 밖이다")
        if not SHA.fullmatch(str(ref["sha256_lf"])) or sha(path) != ref["sha256_lf"]:
            raise ValueError("원본 SHA 결합이 다르다")
        if path.suffix == ".json":
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("schema") == "combination_observation_v1":
                observations.append(record)
    identity = {key: manifest[key] for key in ("robot", "task", "fault", "correction_id")}
    if (len(observations) != 1 or observations[0].get("combination") != identity
            or observations[0].get("stages") != manifest["stages"]):
        raise ValueError("호스트 관측 원본과 manifest 사실이 다르다")


def _identity(m: dict) -> None:
    if not isinstance(m, dict) or set(m) != {"schema", "robot", "task", "fault", "correction_id", "stages", "evidence"}:
        raise ValueError("manifest 필드가 다르다")
    if m["schema"] != SCHEMA or m["robot"] not in ROBOTS or m["fault"] not in FAULTS:
        raise ValueError("등록하지 않은 manifest 조합이다")
    if m["task"] != ROBOTS[m["robot"]][0]:
        raise ValueError("robot/task 결합이 다르다")
    ids = {entry["correction_id"] for entry in registry().values()}
    if m["correction_id"] not in ids or not isinstance(m["stages"], dict) or set(m["stages"]) - set(ORDER):
        raise ValueError("등록하지 않은 교정 또는 단계다")
    if m["fault"] != "U1" and m["correction_id"] != registry()[FAULTS[m["fault"]][0]]["correction_id"]:
        raise ValueError("결함 계열과 등록 교정이 다르다")
    if m["fault"] == "U1" and m["correction_id"] not in (F.CORRECTION_ID, "restore_body_frame_reward"):
        raise ValueError("U1 교정 계열이 다르다")


def _mapping(value, keys: set, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(label + " 필드가 다르다")
    return value


def _fingerprints(row: dict, names: set) -> None:
    _mapping(row, names, "함수 지문")
    if any(not isinstance(v, str) or not SHA.fullmatch(v) for v in row.values()):
        raise ValueError("함수 지문이 SHA-256이 아니다")


def _applicability(m, row):
    _mapping(row, {"targets"}, "적용성")
    expected = fault_paths(m["fault"])
    _mapping(row["targets"], set(expected), "주입 표적")
    blocked = []
    for name, target in row["targets"].items():
        _mapping(target, {"path", "exists", "active", "effect_reachable"}, "표적")
        if target["path"] != expected[name] or any(type(target[k]) is not bool for k in ("exists", "active", "effect_reachable")):
            raise ValueError("주입 표적 경로 또는 적용성 형식이 다르다")
        if not all(target[k] for k in ("exists", "active", "effect_reachable")):
            blocked.append(name)
    return ("not_applicable", {"blocked_targets": blocked}) if blocked else (None, {"active_targets": sorted(expected)})


def _injection(m, row):
    _mapping(row, {"before", "after", "bound"}, "주입")
    names = set(FAULTS[m["fault"]][1])
    for key in ("before", "after"):
        _fingerprints(row[key], names)
    _mapping(row["bound"], names, "활성 경로 결합")
    if any(row["bound"][name] is not True or row["before"][name] == row["after"][name] for name in names):
        return "invalid", {"reason": "injection_readback_mismatch"}
    return None, {"changed_targets": sorted(names)}


def _source(m, row):
    _mapping(row, {"none", "fault", "bands"}, "행동 실패")
    task, protocol, scale, path = ROBOTS[m["robot"]]
    for name in ("none", "fault"):
        report = row[name]
        if report.get("task") != task or report.get("protocol") != protocol:
            raise ValueError("고정 평가 robot/task/protocol이 다르다")
        expected = {"name": report["name"], "checkpoint_file": report["checkpoint"]["file"],
                    "checkpoint_sha256": report["checkpoint"]["sha256"], "action_scale": scale,
                    "protocol_sha256_lf": sha(ROOT / path)}
        # ANYmal의 기존 validate_report와 같은 본문 검증 재사용 방식이다.
        C.validate_fixed_report({**report, "task": C.TASK, "protocol": C.PROTOCOL}, expected)
    bands = _mapping(row["bands"], {"survival_ratio_min", "err_xy_ratio_max", "err_yaw_ratio_max", "fall_frac_max"}, "밴드")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in bands.values()):
        raise ValueError("행동 밴드가 유한한 음이 아닌 수가 아니다")
    if bands != registered_bands(m["robot"]):
        raise ValueError("기존 고정 행동 밴드와 다르다")
    result = E.source_gate(row["none"], row["fault"], bands)
    return (None if result["outcome"] == "failure_present" else result["outcome"]), result


def _scope(m, row):
    if FAULTS[m["fault"]][0] not in ("reward", "reward_family"):
        return "invalid", {"reason": "scope_verification_unregistered"}
    _mapping(row, {"probe", "measurement"} if "verification_kind" not in row else
             {"probe", "measurement", "verification_kind"}, "범위")
    targets = set(target_paths(m["correction_id"]))
    active = set(FAULTS[m["fault"]][1])
    missing = sorted(active - targets)
    if row.get("verification_kind") == "registered_target_readback":
        if m["fault"] != "U1" or row["probe"] is not None or row["measurement"] is not None:
            raise ValueError("등록 함수군 readback 범위 계약이 다르다")
        measured = {"verdict": "not_applicable", "reason": "probe 대신 등록 함수군 readback을 사용"}
    else:
        if "verification_kind" in row or row["probe"] != "P_reward":
            raise ValueError("등록하지 않은 범위 관측이다")
        measured = S.check(row["probe"], row["measurement"], sorted(targets))
        if measured["verdict"] == "scope_unknown":
            return "invalid", {"scope": measured}
        missing = sorted(set(missing) | set(measured.get("uncovered_terms", [])))
        if not measured.get("abnormal_terms"):
            return "invalid", {"reason": "confirming_reward_measurement_not_abnormal", "scope": measured}
    detail = {"uncovered_targets": missing, "restoration_targets": sorted(targets), "probe_scope": measured}
    return ("scope_insufficient" if missing else None), detail


def _noop(m, row):
    _mapping(row, {"before", "after", "unrelated_changed"}, "NONE no-op")
    targets = set(target_paths(m["correction_id"]))
    for key in ("before", "after"):
        _fingerprints(row[key], targets)
    clean = m["stages"]["injection"]["before"]
    if (row["before"] != row["after"] or row["unrelated_changed"] != []
            or any(row["before"][name] != clean[name] for name in targets)):
        return "invalid", {"reason": "none_correction_not_noop"}
    return None, {"unchanged_targets": sorted(targets)}


def _recovery(m, row):
    _mapping(row, {"mode"}, "회복 방식")
    mech = FAULTS[m["fault"]][0]
    if mech == "reward_family":
        mode = "restart"
        rule = "U1 함수군 확장 사전등록: restart"
    elif mech == "termination":
        mode, rule = "resume", "기존 termination 등록 규칙: resume"
    else:
        recommendation = E.recommend(mech, m["stages"]["source"]["fault"])
        mode, rule = recommendation["mode"], recommendation["rule"]
    if row["mode"] != mode or mode not in registry()[mech]["recovery_modes"]:
        return "invalid", {"reason": "unregistered_recovery_mode", "expected": mode}
    return None, {"recovery_mode": mode, "rule": rule}


GATES = {"applicability": _applicability, "injection": _injection, "source": _source,
         "scope": _scope, "noop": _noop, "recovery": _recovery}


def evaluate(manifest: dict, root: Path = ROOT) -> dict:
    """앞 관문에서 멈추면 뒤 관문을 호출하거나 결과를 생성하지 않는다."""
    trace = []
    result = {"status": "invalid", "trace": trace, "reason": None, "active_targets": [],
              "restoration_targets": [], "recovery_mode": None, "manifest_digest": None}
    try:
        _identity(manifest)
        result["manifest_digest"] = L.digest(manifest)
        _bindings(manifest, Path(root))
        for stage in ORDER:
            status, details = GATES[stage](manifest, manifest["stages"][stage])
            trace.append({"stage": stage, "status": status or "passed", "details": details})
            if status:
                result.update(status=status, reason=stage)
                return result
        result.update(status="ready", active_targets=sorted(FAULTS[manifest["fault"]][1]),
                      restoration_targets=sorted(target_paths(manifest["correction_id"])),
                      recovery_mode=manifest["stages"]["recovery"]["mode"])
    except (KeyError, TypeError, ValueError, OSError) as error:
        result["reason"] = "contract_evidence_invalid:" + type(error).__name__
    return result


def _state(manifest: dict, state: dict) -> str:
    mechanism = FAULTS[manifest["fault"]][0]
    expected_probe = "target_readback" if manifest["stages"]["scope"].get("verification_kind") else "P_reward"
    if (state.get("status") != "confirmed" or state.get("hypotheses") != [mechanism]
            or state.get("observed", {}).get(expected_probe) != "abnormal"
            or state.get("selection_actor") not in ("model", "registered_rule")):
        raise LedgerError("확정 상태·관측·선택 주체가 교정 계약과 다르다")
    if expected_probe == "target_readback" and state["selection_actor"] != "registered_rule":
        raise LedgerError("등록 target readback은 모델 선택 근거가 아니다")
    return mechanism


def state_digest(manifest: dict, state: dict) -> str:
    return L.digest({"manifest": manifest, "loop_state": state})


def propose(manifest: dict, state: dict, ledger: CorrectionLedger, *, root: Path = ROOT,
            budget_gpu_s: int = 2400) -> dict:
    checked = evaluate(manifest, root)
    digest = state_digest(manifest, state)
    if checked["status"] != "ready":
        return ledger.hold(digest, "unknown", checked["status"], checked)
    try:
        mechanism = _state(manifest, state)
    except LedgerError:
        return ledger.hold(digest, "unknown", "diagnosis_unconfirmed", checked)
    if ledger.registry_digest != L.digest(registry()):
        raise LedgerError("적용성 등록 표와 교정 원장의 등록 표가 다르다")
    entry = registry()[mechanism]
    return ledger.propose(digest, mechanism, L.digest(manifest["evidence"]), manifest["correction_id"],
        checked["recovery_mode"], {"case": entry["protocol_case"], "arm": "corrected"}, budget_gpu_s,
        {"combination_contract": checked, "restoration_targets": checked["restoration_targets"],
         "selection_actor": state["selection_actor"], "manifest_digest": checked["manifest_digest"]})


def consume(manifest: dict, state: dict, ledger: CorrectionLedger, request_id: str, *, root: Path = ROOT) -> dict:
    checked = evaluate(manifest, root)
    if checked["status"] != "ready":
        raise LedgerError("실행 직전 적용성 관문이 ready가 아니다")
    _state(manifest, state)
    if ledger.registry_digest != L.digest(registry()):
        raise LedgerError("실행 시 등록 표가 다르다")
    request = ledger.requests().get(request_id)
    if request is None or request.get("evidence", {}).get("combination_contract") != checked:
        raise LedgerError("제안 뒤 적용성 검사 결과가 바뀌었다")
    return ledger.consume(request_id, state_digest(manifest, state))
