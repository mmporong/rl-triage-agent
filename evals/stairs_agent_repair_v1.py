"""계단 실패의 검사 선택과 고정 과제 정책 적응 계약. 기존 결함 표는 바꾸지 않는다."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
ACCEPTED_MODELS = {MODEL, "private/nvidia/nemotron-3.5-super-120b-a12b"}
PROBES = ("P_stairs_task_integrity", "P_stairs_stability")
ADAPT = "fine_tune_stairs_300"
TASK = {"robot": "Go2", "direction": "down", "command_x_m_s": 0.25,
        "step_height_m": 0.05, "step_width_m": 0.3, "duration_s": 30}
RECIPE = {"updates": 300, "num_envs": 1024, "learning_rate": 0.0001,
          "load_optimizer": False, "command_x_m_s": 0.25,
          "reward_change": False, "checkpoint_selection": "final_only"}
ACCEPTANCE = {"min_progress_m": 3.75, "max_height_change_m": -0.3,
              "min_body_up_z": 0.5, "max_tracking_rmse_xy_m_s": 0.25,
              "inverted_samples": 0, "reset_count": 0}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def assess(rows: list[dict]) -> dict:
    if len(rows) != 1501:
        raise ValueError("30초 연속 상태 1501행이 필요하다")
    for i, row in enumerate(rows):
        if (row["frame"] != i or row["sim_time_s"] != i / 50
                or row["episode_length_step"] != i or row["terminated"] is not False):
            raise ValueError("상태 시간·종료·리셋 계약 위반")
        for field, length in (("root_pos_w", 3), ("root_quat_w", 4), ("command_body", 3),
                              ("root_lin_vel_w", 3)):
            if (len(row[field]) != length or any(isinstance(v, bool) or not math.isfinite(v)
                                                 for v in row[field])):
                raise ValueError("상태 shape/유한값 계약 위반")
        q = row["root_quat_w"]
        if abs(sum(v*v for v in q)-1) > 0.001:
            raise ValueError("quaternion 정규화 위반")
        up = q[0]**2-q[1]**2-q[2]**2+q[3]**2
        if abs(up-row["body_up_world_z"]) > 0.00001:
            raise ValueError("자세 수치와 quaternion 불일치")
    inverted = [r["sim_time_s"] for r in rows if r["body_up_world_z"] < 0]
    command_ok = all(r["command_body"] == [0.25, 0.0, 0.0] for r in rows)
    first, last = rows[0], rows[-1]
    dx = last["root_pos_w"][0]-first["root_pos_w"][0]
    dz = last["root_pos_w"][2]-first["root_pos_w"][2]
    squared = []
    for row in rows[1:]:
        w, x, y, z = row["root_quat_w"]
        vx, vy, vz = row["root_lin_vel_w"]
        tx, ty, tz = 2*(-y*vz+z*vy), 2*(-z*vx+x*vz), 2*(-x*vy+y*vx)
        body_x, body_y = vx+w*tx-y*tz+z*ty, vy+w*ty-z*tx+x*tz
        squared.append((body_x-0.25)**2+body_y**2)
    rmse = math.sqrt(sum(squared)/len(squared))
    minimum_up = min(r["body_up_world_z"] for r in rows)
    return {"command_preserved": command_ok, "reset_count": 0,
            "inverted_samples": len(inverted), "first_inverted_s": inverted[0] if inverted else None,
            "minimum_body_up_z": minimum_up, "tracking_rmse_xy_m_s": rmse,
            "dx_m": dx, "dz_m": dz, "acceptance": ACCEPTANCE,
            "accepted": command_ok and not inverted and dx >= ACCEPTANCE["min_progress_m"]
            and dz <= ACCEPTANCE["max_height_change_m"] and minimum_up >= ACCEPTANCE["min_body_up_z"]
            and rmse <= ACCEPTANCE["max_tracking_rmse_xy_m_s"]}


def probe(probe_id: str, source: dict) -> dict:
    if probe_id not in PROBES or source.get("schema") != "stairs_agent_source_v1":
        raise ValueError("등록 검사/입력이 아니다")
    a = source["assessment"]
    normal = a["command_preserved"] if probe_id == PROBES[0] else a["accepted"]
    return {"schema": "stairs_agent_probe_receipt_v1", "probe_id": probe_id,
            "source_digest": digest(source), "result": "normal" if normal else "abnormal",
            "measurement": a, "root_cause_confirmed": False,
            "scope": "실제 재생 trace의 명령·자세·진행 검사이며 학습 분포 원인 증명은 아니다"}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def packet(source: dict, receipts: list[dict]) -> dict:
    observed = {}
    for receipt in receipts:
        p = receipt["probe_id"]
        if p in observed or receipt != probe(p, source):
            raise ValueError("검사 원문/계보가 다르거나 중복이다")
        observed[p] = receipt
    eligible = [p for p in PROBES if p not in observed]
    if (not eligible and observed[PROBES[0]]["result"] == "normal"
            and observed[PROBES[1]]["result"] == "abnormal"):
        eligible.append(ADAPT)
    visible_source = {k: v for k, v in source.items() if k != "assessment"}
    return {"schema": "stairs_agent_decision_packet_v1", "task": TASK,
            "facts": {"source": visible_source, **observed}, "eligible_actions": eligible+["HOLD"],
            "recipe": RECIPE, "root_cause_status": "unconfirmed",
            "disclosure": "설계자가 seed200 실패와 속도 대조를 본 뒤 만든 개발 계약이다."
            "추가 학습은 새 과제 적응 후보이며 코드 결함 복원으로 확정하지 않는다."}


def validate_decision(value: dict, context: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"action", "cited_facts", "reason"}:
        raise ValueError("action/cited_facts/reason 세 필드가 필요하다")
    if value["action"] not in context["eligible_actions"]:
        raise ValueError("실행 가능한 등록 action이 아니다")
    cited = value["cited_facts"]
    if (not isinstance(cited, list) or not cited or len(set(cited)) != len(cited)
            or any(not isinstance(c, str) or c not in context["facts"] for c in cited)
            or not isinstance(value["reason"], str) or not value["reason"].strip()):
        raise ValueError("실제 fact ID 인용과 이유가 필요하다")
    if value["action"] == ADAPT and not set(PROBES).issubset(cited):
        raise ValueError("과제 무결성과 실패 검사를 모두 인용해야 한다")
    return value


def messages(context: dict) -> list[dict]:
    return [{"role": "system", "content":
             "You are RL Triage, choosing a next registered check or a policy adaptation. "
             "Return one JSON object with exactly action, cited_facts (fact IDs), reason (Korean). "
             "Choose only eligible_actions. Propose adaptation only when task integrity is normal "
             "and stability abnormal. This is not proof of a reward/actuator/optimizer bug. "
             "Never change commanded speed, stairs, observations, rewards or acceptance criteria. "
             "HOLD if evidence is insufficient. A model proposal grants no execution authority."},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, allow_nan=False)}]


def authorize_adaptation(source: dict, receipts: list[dict], decision: dict,
                         registered_operator_approval: bool) -> dict:
    context = packet(source, receipts)
    validate_decision(decision, context)
    if decision["action"] != ADAPT or registered_operator_approval is not True:
        raise ValueError("유효한 모델 제안과 실행 전 등록된 호스트 승인이 필요하다")
    return {"schema": "stairs_agent_adaptation_authorization_v1", "source_digest": digest(source),
            "decision_digest": digest(decision), "receipts_digest": digest(receipts),
            "recipe": RECIPE, "task": TASK, "selected_by": "model",
            "root_cause_confirmed": False}
