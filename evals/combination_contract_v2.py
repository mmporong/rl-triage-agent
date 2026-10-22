"""적용성 계약 v2: v1 여섯 관문에 모터·종료 범주의 범위 관문과 확정 상태 검사를 등록한다.

combination_contract_v1.py는 조합 실행 프로토콜이 SHA로 묶고 있어 그대로 두고, 그 관문 함수를 재사용한다.
바뀌는 것은 두 곳뿐이다.
- 범위(scope): 보상 계열은 v1과 같다. 모터(F2)는 P_torque, 종료(F6)는 P_episode가 확정 probe다. 두 probe는
  항별 측정이 없으므로(correction_scope.check가 not_applicable) 동결 분류로 확정 측정이 abnormal인지 다시 계산하고,
  결함 활성 표적이 등록 교정의 복원 표적에 모두 들어 있는지 구조로 대조한다.
- 확정 상태(_state): 범주별 확정 probe가 abnormal이어야 한다.
등록 표(registry)는 v1과 같다. 같은 교정 원장을 그대로 쓸 수 있다.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import combination_contract_v1 as V1  # noqa: E402
from rl_triage import correction_scope as S  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger, LedgerError  # noqa: E402

VERSION = "combination_contract_v2"
ORDER, FAULTS, ROBOTS, SCHEMA = V1.ORDER, V1.FAULTS, V1.ROBOTS, V1.SCHEMA
registry, target_paths, make_manifest, sha = V1.registry, V1.target_paths, V1.make_manifest, V1.sha
# 항별 측정이 없는 범주의 확정 probe. 보상 계열은 v1 규칙(P_reward 또는 등록 target readback)을 따른다.
CONFIRMING = {"actuator": "P_torque", "termination": "P_episode"}


def _scope(m, row):
    mechanism = FAULTS[m["fault"]][0]
    if mechanism in ("reward", "reward_family"):
        return V1._scope(m, row)
    probe = CONFIRMING.get(mechanism)
    if probe is None:
        return "invalid", {"reason": "scope_verification_unregistered"}
    V1._mapping(row, {"probe", "measurement", "reference"}, "범위")
    if row["probe"] != probe:
        raise ValueError("등록하지 않은 범위 관측이다")
    if not isinstance(row["measurement"], dict) or not (row["reference"] is None or isinstance(row["reference"], dict)):
        raise ValueError("범위 측정값 형식이 다르다")
    targets = set(target_paths(m["correction_id"]))
    measured = {**S.check(probe, row["measurement"], sorted(targets)),
                "classification": L.classify(probe, row["measurement"], row["reference"])}
    if measured["classification"] != "abnormal":
        return "invalid", {"reason": "confirming_measurement_not_abnormal", "scope": measured}
    missing = sorted(set(FAULTS[m["fault"]][1]) - targets)
    detail = {"uncovered_targets": missing, "restoration_targets": sorted(targets), "probe_scope": measured,
              "scope_basis": "registered_targets_cover_active_fault_targets"}
    return ("scope_insufficient" if missing else None), detail


GATES = {**V1.GATES, "scope": _scope}


def evaluate(manifest: dict, root: Path = ROOT) -> dict:
    """v1.evaluate와 같은 순서·중단 규칙에 v2 관문 표를 쓴다."""
    trace = []
    result = {"status": "invalid", "trace": trace, "reason": None, "active_targets": [],
              "restoration_targets": [], "recovery_mode": None, "manifest_digest": None, "contract": VERSION}
    try:
        V1._identity(manifest)
        result["manifest_digest"] = L.digest(manifest)
        V1._bindings(manifest, Path(root))
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


def confirming_probe(manifest: dict) -> str:
    mechanism = FAULTS[manifest["fault"]][0]
    if mechanism in CONFIRMING:
        return CONFIRMING[mechanism]
    return "target_readback" if manifest["stages"]["scope"].get("verification_kind") else "P_reward"


def _state(manifest: dict, state: dict) -> str:
    mechanism = FAULTS[manifest["fault"]][0]
    expected_probe = confirming_probe(manifest)
    if (state.get("status") != "confirmed" or state.get("hypotheses") != [mechanism]
            or state.get("observed", {}).get(expected_probe) != "abnormal"
            or state.get("selection_actor") not in ("model", "registered_rule")):
        raise LedgerError("확정 상태·관측·선택 주체가 교정 계약과 다르다")
    if expected_probe == "target_readback" and state["selection_actor"] != "registered_rule":
        raise LedgerError("등록 target readback은 모델 선택 근거가 아니다")
    return mechanism


state_digest = V1.state_digest


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
