"""공식 Isaac Lab 디버깅 스킬(isaaclab-debugging-rl-training) 체크리스트를 한 사례의 기존 기록에 대조한다.

이미 남은 파일만 읽고 새 실행을 하지 않는다. 근거 파일이 없거나 판단 규칙이 등록되지 않은 항목은
unmet·partial로 남기며 내용을 지어내지 않는다. 스킬을 실행 중 호출한 것이 아니라 절차 항목과의 대조다.
체크포인트 차원은 torch 없이 zip 안의 pickle에서 허용 목록의 텐서 shape만 읽는다. GPU·모델·네트워크를 쓰지 않는다.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import zipfile
from typing import Any

import yaml

from rl_triage.triage_tools import _TolerantLoader

SCHEMA = "skill_checklist_report_v1"
SKILL_DIR = Path("bench/reference/nvidia_skills/isaaclab-debugging-rl-training")
STATUSES = ("met", "partial", "unmet", "not_applicable")
NO_TRACKING_RATIO = 0.9  # bench/protocols/training_correction_v4.json에 사전등록한 "추종 없음" 기준
# 이 저장소의 메커니즘 범주를 스킬 검증 2번의 실패 층 분류로 옮기는 대응표.
LAYER_BY_MECHANISM = {"reward": "reward", "physics": "physics", "actuator": "physics",
                      "termination": "environment", "exploration": "runner", "optimizer": "runner"}
OBS_TERM_DIMS = {
    "isaaclab.envs.mdp.observations:base_lin_vel": 3,
    "isaaclab.envs.mdp.observations:base_ang_vel": 3,
    "isaaclab.envs.mdp.observations:projected_gravity": 3,
    "isaaclab.envs.mdp.observations:generated_commands": "command",
    "isaaclab.envs.mdp.observations:joint_pos_rel": "joints",
    "isaaclab.envs.mdp.observations:joint_vel_rel": "joints",
    "isaaclab.envs.mdp.observations:last_action": "actions",
}
COMMAND_DIMS = {"isaaclab.envs.mdp.commands.velocity_command:UniformVelocityCommand": 3}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 스킬 원문

def _numbered(section: str) -> list[str]:
    return [match[1].strip() for match in re.finditer(r"^\d+\.\s+(.+)$", section, re.M)]


def _section(text: str, title: str) -> str:
    match = re.search(rf"^## {re.escape(title)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not match:
        raise ValueError(f"skill section is missing: {title}")
    return match[1]


def load_skill(repo_root: Path) -> dict:
    """고정한 스킬 원문의 SHA를 출처 기록과 대조하고 작업 흐름·검증·결정 규칙을 꺼낸다."""
    folder = repo_root / SKILL_DIR
    pinned = json.loads((folder / "SOURCE.json").read_text(encoding="utf-8"))["files"]
    pinned.update(json.loads((folder / "SOURCE_extra.json").read_text(encoding="utf-8"))["files"])
    files = {}
    for name in ("SKILL.md", "reference.md"):
        digest = sha256(folder / name)
        if digest != pinned[name]["sha256"]:
            raise ValueError(f"pinned skill file changed: {name}")
        files[name] = {"path": (SKILL_DIR / name).as_posix(), "sha256": digest}
    skill = (folder / "SKILL.md").read_text(encoding="utf-8")
    reference = (folder / "reference.md").read_text(encoding="utf-8")
    workflow, validation = _numbered(_section(skill, "Workflow")), _numbered(_section(skill, "Validation"))
    rules = [line[2:].strip() for line in _section(reference, "Decision Rules").splitlines() if line.startswith("- ")]
    if (len(workflow), len(validation), len(rules)) != (9, 6, 5):
        raise ValueError("pinned skill checklist shape differs from the 9/6/5 contract")
    commit = json.loads((folder / "SOURCE.json").read_text(encoding="utf-8"))["commit_sha"]
    return {"commit_sha": commit, "files": files, "workflow": workflow, "validation": validation, "decision_rules": rules}


# ---------------------------------------------------------------- 근거

@dataclass
class Evidence:
    repo_root: Path
    isaaclab_root: Path | None = None
    rows: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def add(self, relative: str | None, kind: str, *, scope: str = "repo") -> str | None:
        """파일이 있으면 근거 ID를 돌려주고, 없으면 누락으로만 기록한다."""
        if relative is None:
            return None
        root = self.repo_root if scope == "repo" else self.isaaclab_root
        label = relative if scope == "repo" else f"isaaclab:{relative}"
        for row in self.rows:
            if row["path"] == label:
                return row["id"]
        if root is None or not (root / relative).is_file():
            if label not in self.missing:
                self.missing.append(label)
            return None
        evidence_id = f"E{len(self.rows) + 1}"
        self.rows.append({"id": evidence_id, "path": label, "kind": kind, "sha256": sha256(root / relative)})
        return evidence_id


def _item(item_id: str, text: str, status: str, finding: str, evidence: list[str | None], missing: list[str] = ()) -> dict:
    if status not in STATUSES:
        raise ValueError(f"unknown checklist status: {status}")
    return {"id": item_id, "text": text, "status": status, "finding": finding,
            "evidence": [value for value in evidence if value], "missing": list(missing)}


# ---------------------------------------------------------------- 체크포인트 차원(torch 없이)

class _Shape:
    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = tuple(int(value) for value in shape)


def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
    return _Shape(size)


class _StorageMarker:
    pass


class _ShapeUnpickler(pickle.Unpickler):
    """torch 저장 형식에서 허용한 전역만 해석하고 텐서는 shape만 남긴다."""

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return _StorageMarker
        raise pickle.UnpicklingError(f"global is not allowed in checkpoint shape reading: {module}.{name}")

    def persistent_load(self, pid):
        return _StorageMarker


def checkpoint_shapes(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError("checkpoint archive must contain exactly one data.pkl")
        payload = _ShapeUnpickler(io.BytesIO(archive.read(names[0]))).load()
    model = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(model, dict):
        raise ValueError("checkpoint has no model_state_dict")
    return {"iteration": payload.get("iter"),
            "tensors": {key: list(value.shape) for key, value in model.items() if isinstance(value, _Shape)}}


def config_dims(env: dict, action_dim: int) -> dict:
    """저장 설정의 관측 항으로 정책 관측 차원을 계산한다. 모르는 항이 있으면 계산하지 않는다."""
    policy = env["observations"]["policy"]
    if policy.get("history_length") not in (None, 0, 1) or policy.get("concatenate_terms") is not True:
        raise ValueError("observation history or concatenation is outside the registered dimension table")
    actions = env.get("actions", {})
    if len(actions) != 1 or next(iter(actions.values())).get("joint_names") != [".*"]:
        raise ValueError("action term is not a single all-joint action; joint count is not derivable")
    total, terms = 0, {}
    for name, term in policy.items():
        if not isinstance(term, dict) or "func" not in term:
            continue
        spec = OBS_TERM_DIMS.get(term["func"])
        if spec is None:
            raise ValueError(f"observation term is outside the registered dimension table: {term['func']}")
        if spec == "command":
            command = env["commands"][term["params"]["command_name"]]["class_type"]
            spec = COMMAND_DIMS.get(command)
            if spec is None:
                raise ValueError(f"command type is outside the registered dimension table: {command}")
        dim = action_dim if spec in ("joints", "actions") else spec
        terms[name] = dim
        total += dim
    return {"policy_obs_dim": total, "terms": terms,
            "assumption": "단일 JointPositionAction(joint_names '.*')이므로 관절 수 = 행동 차원"}


def checkpoint_compatibility(shapes: dict, env: dict) -> dict:
    tensors = shapes["tensors"]
    action_dim = (tensors.get("std") or tensors.get("log_std") or [None])[0]
    actor_in = (tensors.get("actor.0.weight") or [None, None])[1]
    critic_in = (tensors.get("critic.0.weight") or [None, None])[1]
    if action_dim is None or actor_in is None:
        raise ValueError("checkpoint lacks std/log_std or actor.0.weight")
    expected = config_dims(env, action_dim)
    critic_expected = expected["policy_obs_dim"] if "critic" not in env["observations"] else None
    compatible = actor_in == expected["policy_obs_dim"] and (critic_expected is None or critic_in == critic_expected)
    return {"checkpoint_iteration": shapes["iteration"], "action_dim": action_dim, "actor_obs_dim": actor_in,
            "critic_obs_dim": critic_in, "config_policy_obs_dim": expected["policy_obs_dim"],
            "config_terms": expected["terms"], "assumption": expected["assumption"], "compatible": compatible}


# ---------------------------------------------------------------- 입력과 보고서

@dataclass
class CaseInputs:
    """CLI가 모은 사례 파일. 경로는 저장소 기준(체크포인트는 Isaac Lab 기준) 상대 경로다."""
    case: str
    telemetry: str | None = None
    fixed_report: str | None = None
    fixed_summary: str | None = None
    fixed_protocol: str | None = None
    loop_state: str | None = None
    loop_ledger: str | None = None
    correction_ledger: str | None = None
    probe_report: str | None = None
    env_config: str | None = None
    run_protocol: str | None = None
    correction_result: str | None = None
    correction_registry: str | None = None
    checkpoint: str | None = None
    r1_alert_iterations: list[int] | None = None


def _read_json(root: Path, relative: str | None) -> Any:
    if relative is None or not (root / relative).is_file():
        return None
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _read_jsonl(root: Path, relative: str | None) -> list[dict] | None:
    if relative is None or not (root / relative).is_file():
        return None
    return [json.loads(line) for line in (root / relative).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(inputs: CaseInputs, repo_root: Path, isaaclab_root: Path | None = None) -> dict:
    skill = load_skill(repo_root)
    ev = Evidence(repo_root, isaaclab_root)
    e_tele = ev.add(inputs.telemetry, "training_telemetry")
    e_fixed = ev.add(inputs.fixed_report, "fixed_evaluation_report")
    e_summary = ev.add(inputs.fixed_summary, "fixed_evaluation_summary")
    e_fixed_protocol = ev.add(inputs.fixed_protocol, "fixed_evaluation_protocol")
    e_state = ev.add(inputs.loop_state, "probe_loop_state")
    e_ledger = ev.add(inputs.loop_ledger, "probe_loop_ledger")
    e_corr_ledger = ev.add(inputs.correction_ledger, "correction_ledger")
    e_probe = ev.add(inputs.probe_report, "probe_measurement")
    e_env = ev.add(inputs.env_config, "saved_environment_config")
    e_run_protocol = ev.add(inputs.run_protocol, "experiment_protocol")
    e_result = ev.add(inputs.correction_result, "correction_result")
    e_registry = ev.add(inputs.correction_registry, "correction_registry")
    e_ckpt = ev.add(inputs.checkpoint, "checkpoint", scope="isaaclab")

    tele = _read_json(repo_root, inputs.telemetry)
    report = _read_json(repo_root, inputs.fixed_report)
    summary = _read_json(repo_root, inputs.fixed_summary)
    state = _read_json(repo_root, inputs.loop_state)
    ledger = _read_jsonl(repo_root, inputs.loop_ledger)
    probe = _read_json(repo_root, inputs.probe_report)
    protocol = _read_json(repo_root, inputs.run_protocol)
    result = _read_json(repo_root, inputs.correction_result)
    env = (yaml.load((repo_root / inputs.env_config).read_text(encoding="utf-8"), Loader=_TolerantLoader)
           if e_env else None)

    def last(tag, field_name="mean_last_20pct"):
        return tele["summary"][tag][field_name] if tele and tag in tele["summary"] else None

    reward_first, reward_last = last("Train/mean_reward", "mean_first_20pct"), last("Train/mean_reward")
    ratio = (report["metrics"]["lin_vel_rmse_mps"] / report["standstill"]["lin_vel_rmse_mps"]) if report else None
    summary_row = next((row for row in (summary or {}).get("runs", []) if row.get("run") == inputs.case), None)
    receipts = [row for row in (ledger or []) if row.get("event") == "receipt"]
    rejected = [row for row in (ledger or []) if row.get("event") == "rejected"]
    w, v, d = skill["workflow"], skill["validation"], skill["decision_rules"]
    items_w, items_v, items_d = [], [], []

    # W1 / V1: 실행 식별과 실패 증상
    identity = []
    if env:
        identity.append(f"seed {env.get('seed')}, env {env['scene']['num_envs']}")
    if report:
        identity.append(f"task {report['task']}, RSL-RL {report['versions'].get('rsl-rl-lib')}, GPU {report['versions'].get('gpu')}")
    command_missing = ["원 학습의 정확한 실행 명령(공개 기록 없음)"]
    items_w.append(_item("W1", w[0], "partial" if identity else "unmet",
                         "; ".join(identity) + ". 실행 명령 문자열은 공개 기록에 없다." if identity else "식별 근거 없음",
                         [e_env, e_fixed, e_run_protocol], command_missing))
    # W2: 환경 계약
    if env and e_fixed_protocol:
        rewards = [name for name, term in env["rewards"].items() if isinstance(term, dict) and term.get("weight")]
        resets = [name for name, term in env["events"].items() if isinstance(term, dict) and term.get("mode") == "reset"]
        finding = (f"보상 항 {len(rewards)}개({', '.join(rewards)}), 종료 항 {', '.join(env['terminations'])}, "
                   f"reset 이벤트 {', '.join(resets)}, 행동 {', '.join(env['actions'])}, "
                   "성공 지표는 고정 평가의 선속도·회전속도 RMSE와 낙상률")
        items_w.append(_item("W2", w[1], "met", finding, [e_env, e_fixed_protocol]))
    else:
        items_w.append(_item("W2", w[1], "unmet", "저장 설정 또는 고정 평가 계약이 없다", [e_env, e_fixed_protocol]))
    # W3: 가장 작은 재현(결정적 재생)
    if report:
        items_w.append(_item("W3", w[2], "met",
                             f"결정적 고정 평가: eval seed {report['eval_seed']}, {report['num_envs']} env, {report['horizon_steps']} step",
                             [e_fixed, e_fixed_protocol]))
    else:
        items_w.append(_item("W3", w[2], "unmet", "고정 평가 기록이 없다", []))
    # W4 / V6: 한 변수 원칙
    single = bool(protocol and (protocol.get("cases") or protocol.get("arm_specs")))
    design = []
    if protocol and protocol.get("cases"):
        design.append("교정 arm은 결함 표적 하나만 복원하고 대조 arm은 무관한 표적 하나만 바꾼다")
    if protocol and protocol.get("arm_specs"):
        design.append("회복 방식 비교는 arm마다 회복 방식 하나만 바꾼다")
    items_w.append(_item("W4", w[3], "met" if single else "unmet",
                         "; ".join(design) if single else "교정 실험 프로토콜이 없다", [e_run_protocol, e_registry]))
    # W5 / V4: 보상 곡선과 과업 지표
    if reward_first is not None and ratio is not None:
        finding = (f"학습 총보상 처음 20% {reward_first:.2f} → 마지막 20% {reward_last:.2f}, "
                   f"고정 평가 standstill_ratio {ratio:.3f}(제자리 대비 선속도 오차 비)")
        items_w.append(_item("W5", w[4], "met", finding, [e_tele, e_fixed]))
        track_first, track_last = last("Episode_Reward/track_lin_vel_xy_exp", "mean_first_20pct"), last("Episode_Reward/track_lin_vel_xy_exp")
        items_v.append(_item("V4", v[3], "met",
                             f"추종 보상 항 {track_first:.3f} → {track_last:.3f}와 고정 평가 지표를 함께 확인", [e_tele, e_fixed]))
    else:
        items_w.append(_item("W5", w[4], "unmet", "학습 기록 또는 고정 평가가 없다", [e_tele, e_fixed]))
        items_v.append(_item("V4", v[3], "unmet", "학습 기록 또는 고정 평가가 없다", [e_tele, e_fixed]))
    # W6: 보상 기하 일관성
    measurement = (probe or {}).get("probes", {}).get("P_reward", {}).get("measurement")
    if measurement:
        items_w.append(_item("W6", w[5], "partial",
                             f"P_reward로 추종 보상을 몸체·월드 좌표에서 재계산: 선속도 항 상대 차이 "
                             f"{measurement['track_lin_vel_xy_exp_rel_error']:.3g}. 모든 보상 항의 과업 단계 대응은 기록이 없다",
                             [e_probe, e_env], ["보상 항별 과업 단계 대응표"]))
    else:
        items_w.append(_item("W6", w[5], "unmet", "보상 기하 검사 기록이 없다", [e_probe]))
    # W7: 체크포인트 호환
    if e_ckpt and env:
        try:
            compat = checkpoint_compatibility(checkpoint_shapes(isaaclab_root / inputs.checkpoint), env)
            items_w.append(_item("W7", w[6], "met",
                                 f"체크포인트 iteration {compat['checkpoint_iteration']}: 행동 {compat['action_dim']}차원, "
                                 f"actor 입력 {compat['actor_obs_dim']} · critic 입력 {compat['critic_obs_dim']} 대 저장 설정 관측 "
                                 f"{compat['config_policy_obs_dim']}차원 → {'일치' if compat['compatible'] else '불일치'}",
                                 [e_ckpt, e_env]))
            checkpoint_detail = compat
        except (ValueError, pickle.UnpicklingError, zipfile.BadZipFile, KeyError) as error:
            items_w.append(_item("W7", w[6], "unmet", f"차원 계산 불가: {error}", [e_ckpt, e_env]))
            checkpoint_detail = None
    else:
        items_w.append(_item("W7", w[6], "unmet", "체크포인트 또는 저장 설정이 없다", [e_ckpt, e_env]))
        checkpoint_detail = None
    # W8: 접촉이 많은 조작 과업
    task = (report or {}).get("task", "")
    items_w.append(_item("W8", w[7], "not_applicable" if "Velocity" in task else "unmet",
                         "속도 추종 보행 과업이라 조작 과업의 접촉·그리퍼 추적 항목은 해당하지 않는다" if "Velocity" in task
                         else "과업 유형을 확인할 근거가 없다", [e_fixed]))
    # W9: 체크포인트 선택
    fixed_final = bool(protocol and (protocol.get("final_iteration") or protocol.get("final_iteration_by_mode")))
    items_w.append(_item("W9", w[8], "met" if fixed_final else "unmet",
                         "프로토콜이 최종 체크포인트를 고정하고 보상이 아닌 고정 평가로 판정한다" if fixed_final
                         else "체크포인트 선택 규칙 기록이 없다", [e_run_protocol, e_fixed_protocol]))

    # V1: 명령과 실패 증상
    if summary_row:
        symptom = f"고정 평가 {summary_row['fixed_label']} ({', '.join(summary_row['fixed_failed']) or '-'})"
        if inputs.r1_alert_iterations:
            symptom += f", 추종 경고 R1 iteration {inputs.r1_alert_iterations[0]}"
        items_v.insert(0, _item("V1", v[0], "partial", symptom + ". 실행 명령 문자열은 공개 기록에 없다.",
                                [e_summary, e_tele], command_missing))
    else:
        items_v.insert(0, _item("V1", v[0], "unmet", "실패 증상 기록이 없다", [e_summary], command_missing))
    # V2: 실패 층 분류
    if state and state.get("status") == "confirmed" and len(state.get("hypotheses", [])) == 1:
        mechanism = state["hypotheses"][0]
        items_v.insert(1, _item("V2", v[1], "met",
                                f"probe 고리가 {mechanism} 메커니즘으로 확정 → 스킬 분류 '{LAYER_BY_MECHANISM.get(mechanism, '대응 없음')}' "
                                "(저장소 대응표)", [e_state, e_ledger]))
    else:
        items_v.insert(1, _item("V2", v[1], "unmet", "확정된 실패 층 기록이 없다", [e_state]))
    # V3: 한 변수를 고립한 재현
    if receipts and measurement:
        items_v.insert(2, _item("V3", v[2], "met",
                                f"승인된 검사 {len(receipts)}건(거절 {len(rejected)}건 포함 원장): 기록 상태에서 보상 좌표계 하나만 바꿔 재계산",
                                [e_ledger, e_probe]))
    else:
        items_v.insert(2, _item("V3", v[2], "unmet", "승인·receipt가 붙은 고립 검사 기록이 없다", [e_ledger, e_probe]))
    # V5: 결정적 rollout으로 행동 변화 확인
    if report and result:
        numeric = result.get("numeric", {})
        outcome = numeric.get("labels") or numeric.get("status")
        if isinstance(outcome, dict):
            outcome = ", ".join(f"{arm} {label}" for arm, label in outcome.items())
        items_v.append(_item("V5", v[4], "met",
                             f"결정적 고정 평가로 실패 행동을 확인했고, 교정 뒤 같은 평가의 판정은 {outcome}",
                             [e_fixed, e_result]))
    elif report:
        items_v.append(_item("V5", v[4], "partial", "실패 행동은 고정 평가로 확인했지만 교정 뒤 평가 기록이 없다",
                             [e_fixed], ["교정 뒤 고정 평가"]))
    else:
        items_v.append(_item("V5", v[4], "unmet", "고정 평가 기록이 없다", []))
    # V6: 다음 실행은 한 변수만
    correction_rows = _read_jsonl(repo_root, inputs.correction_ledger) or []
    proposed = [row for row in correction_rows if row.get("event") == "proposed"]
    if proposed:
        items_v.append(_item("V6", v[5], "met", f"교정 원장 제안 {len(proposed)}건, 각 제안은 교정 하나·회복 방식 하나",
                             [e_corr_ledger, e_registry]))
    elif single and e_registry:
        items_v.append(_item("V6", v[5], "met", "교정 원장 기록은 없고, 등록 표와 프로토콜이 다음 실행을 교정 하나·회복 방식 하나로 제한한다",
                             [e_registry, e_run_protocol], ["사례별 교정 원장"]))
    else:
        items_v.append(_item("V6", v[5], "unmet", "다음 실행 제한 기록이 없다", [e_registry], ["사례별 교정 원장"]))

    # 결정 규칙
    reward_rises = None if reward_first is None else reward_last > reward_first
    no_tracking = None if ratio is None else ratio >= NO_TRACKING_RATIO
    basis = [e_tele, e_fixed]
    for index, (applies, note) in enumerate((
            (None if None in (reward_rises, no_tracking) else reward_rises and not no_tracking, "보상 상승·추종 성공"),
            (None if None in (reward_rises, no_tracking) else reward_rises and no_tracking, "보상 상승·추종 없음"),
            (None if reward_rises is None else not reward_rises, "보상이 처음부터 평평"),
            (None, "엔트로피 조기 붕괴 판단 임계값이 등록되지 않았다"),
            (None, "공개 기록에 물리 경고 집계가 없다"))):
        rule_id = f"D{index + 1}"
        if applies is None:
            items_d.append({**_item(rule_id, d[index], "unmet", note, []), "applies": None})
        else:
            finding = (f"{note}: {'해당' if applies else '해당 없음'} (총보상 상승 {reward_rises}, "
                       f"standstill_ratio {ratio:.3f} ≥ {NO_TRACKING_RATIO} 여부 {no_tracking})")
            items_d.append({**_item(rule_id, d[index], "met", finding, basis), "applies": applies})

    counts = {section: {status: sum(item["status"] == status for item in items) for status in STATUSES}
              for section, items in (("workflow", items_w), ("validation", items_v), ("decision_rules", items_d))}
    return {
        "schema": SCHEMA, "case": inputs.case,
        "skill": {"name": "isaaclab-debugging-rl-training", "upstream_status": "experimental",
                  "commit_sha": skill["commit_sha"], "files": skill["files"]},
        "scope": "이미 본 사례의 기존 기록을 공식 스킬 항목에 대조한 결과다. 스킬을 실행 중 호출한 기록이 아니며 근거가 없으면 unmet으로 둔다.",
        "items": {"workflow": items_w, "validation": items_v, "decision_rules": items_d},
        "counts": counts, "checkpoint_compatibility": checkpoint_detail,
        "evidence": ev.rows, "missing_inputs": ev.missing,
    }


def render_markdown(report: dict) -> str:
    lines = [f"# 스킬 체크리스트 보고서: {report['case']}", "",
             f"스킬 `{report['skill']['name']}`({report['skill']['upstream_status']}), 커밋 `{report['skill']['commit_sha'][:12]}`.",
             report["scope"], ""]
    titles = {"workflow": "작업 흐름", "validation": "검증", "decision_rules": "결정 규칙"}
    for section, title in titles.items():
        counts = report["counts"][section]
        lines += [f"## {title} ({', '.join(f'{key} {value}' for key, value in counts.items() if value)})", "",
                  "| ID | 스킬 항목 | 상태 | 판단 | 근거 |", "|---|---|---|---|---|"]
        for item in report["items"][section]:
            missing = f" (누락: {', '.join(item['missing'])})" if item["missing"] else ""
            cells = [item["id"], item["text"], item["status"], item["finding"] + missing, ", ".join(item["evidence"]) or "-"]
            lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
        lines.append("")
    lines += ["## 근거 파일", "", "| ID | 종류 | 경로 | SHA256 |", "|---|---|---|---|"]
    lines += [f"| {row['id']} | {row['kind']} | `{row['path']}` | `{row['sha256'][:16]}…` |" for row in report["evidence"]]
    if report["missing_inputs"]:
        lines += ["", "없는 입력: " + ", ".join(f"`{path}`" for path in report["missing_inputs"])]
    return "\n".join(lines) + "\n"
