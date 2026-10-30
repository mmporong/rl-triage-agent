"""검사 선택→검사 결과→판단 변화→교정→회복/HOLD를 공개 원문으로 따라가는 단일 재생 HTML을 만든다.

화면의 수치·배지는 커밋된 공개 결과 파일의 JSON pointer 값으로만 만들고, 원문 경로와 sha256을 HTML에
결합한다. 검사 선택 주체(model/rule/host)는 원문 필드로만 정하며, 모델 세대가 다른 실행을 한 줄로 합치지
않는다. 결과가 없는 단계는 pending, 공개 manifest·영상이 없는 슬롯은 대기로 둔다. GPU·모델·네트워크를
쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import build_replay_page as B  # 커밋 상태·해시 판정과 문구 검사를 그대로 쓴다
import build_transferred_vision_replay_v1 as TV  # standalone HTML 도우미

OUTPUT = Path("docs/replay/agentic-story-v1.html")
SCHEMA = "agentic_story_replay_v1"
MODEL_3_SUPER = "nvidia/nemotron-3-super-120b-a12b"
MODEL_3_5 = "nvidia/nemotron-3.5-super-vl-120b-a12b"
STAGES = (("select", "검사 선택"), ("result", "검사 결과"), ("judge", "판단 변화"), ("correct", "교정"),
          ("outcome", "회복/HOLD"))
STATES = {"confirmed": "confirmed", "not_recovered": "not_recovered", "recovered": "recovered", "HOLD": "HOLD",
          "execution_failed": "실행 실패", "pending": "pending", "normal": "normal", "abnormal": "abnormal"}
SELECTORS = ("model", "rule", "host")
# 선택 주체는 원문 필드 이름과 값의 쌍으로만 정한다. 표에 없는 쌍은 화면에 올리지 않는다.
SELECTOR_BASIS = {
    ("selection_source", "model_selected"): "model",
    ("selection_source", "registered_next"): "rule",
    ("selection_source", "registered_new_probe"): "rule",
    ("selected_by", "model"): "model",
    ("selected_by", "rule"): "rule",
    ("stage", "model"): "model",
    ("first_diagnosis_valid_after_feedback", True): "model",
    ("model_loop", "inactive"): "host",
}
# 3.5 경로는 첫 모델 선택까지만 공개 결과가 있다. 뒤 단계에 결과 배지가 붙으면 화면을 만들지 않는다.
FRESH_OPEN_STATES = (None, "pending", "HOLD", "execution_failed")
SOURCE_STATUS = {"ready": "확인", "waiting": "대기", "mismatch": "불일치"}
VIDEO_ROLES = {
    "input_reference": "입력 녹화 · 정상 정책 · 교정 영상 아님",
    "input_case": "입력 녹화 · F1 결함 정책 · 교정 영상 아님",
    "stairs_playback": "공식 rough 정책 재생 · 교정 영상 아님 · 판정 없음",
}

R1 = "evals/results/r1_repair_s2071_20261016"
RESIDUAL_V1 = "evals/results/residual_failure_public_20261017_v2"
RESIDUAL_V2 = "evals/results/residual_success_public_20261017"
U1_TAGS = ("evals/results/u1_scope_extension_s2051_20261015", "evals/results/novel_u1_scope_confirmation_s2062_20261016")
FRESH_CHOICE = "evals/results/fresh_frame_probe_choice_dev_s2221_20261029_v1"
FRESH_PREFIX = "evals/results/fresh_frame_runtime_prefix_dev_s2221_20261029_v1/summary.json"
FRESH_OPERATION = "evals/results/fresh_frame_runtime_operation_dev_s2221_20261029_v2/summary.json"
FRESH_RENDER = "evals/results/agentic_f1_fresh_render_dev_20261029h_v1/summary.json"
FRESH_MEDIA = "docs/media/agentic_f1_fresh_render_dev_20261029h_v1/source_manifest.json"
FRESH_CREATION = "evals/results/fresh_brev_render_creation_20261029f_v1/summary.json"
RENDER_AUDIT = "evals/results/vision_render_integrity_audit_20261028_v1/summary.json"
STAIRS_SCHEMA = "stairs_playback_manifest_v2"
STAIRS_TASKS = {"biped": ("Isaac-Velocity-Rough-H1-v0", "Isaac-Velocity-Rough-H1-Play-v0"),
                "quadruped": ("Isaac-Velocity-Rough-Unitree-Go2-v0", "Isaac-Velocity-Rough-Unitree-Go2-Play-v0")}
STAIRS_SLOTS = (("biped", "up", "H1 계단 상행"), ("biped", "down", "H1 계단 하행"),
                ("quadruped", "up", "Go2 계단 상행"), ("quadruped", "down", "Go2 계단 하행"))
SCOPE = ("이미 본 결함 종류의 개발 기록을 공개 원문으로 다시 보여 주는 화면이다. "
         "진단 정확도·교정 성공률·일반화·GPU 절감의 근거가 아니다.")
EXTERNAL = re.compile(r"https?:|(?:src|href)\s*=\s*[\"']?//|@import|url\(", re.IGNORECASE)


def _sha256_lf(path: Path) -> str:
    """Git의 text/eol=lf 정본과 같은 줄바꿈으로 공개 JSON 해시를 계산한다."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def at(document: Any, pointer: str) -> Any:
    """JSON pointer(RFC 6901의 단순형, 이스케이프 없음)로 원문 값을 읽는다."""
    value = document
    for part in pointer.split("/")[1:]:
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


class Story:
    """화면에 쓰는 공개 원문을 모으고 원문마다 모델 실행을 표시한다."""

    def __init__(self, root: Path):
        self.sources = B.Sources(root)
        self.root = self.sources.root
        self.ids: dict[str, str] = {}
        self.docs: dict[str, Any] = {}
        self.models: dict[str, str | None] = {}
        self.stale_videos: set[str] = set()

    def doc(self, relative: str, model: str | None) -> tuple[Any, str]:
        row = self.sources.file(relative, kind="public_result")
        if row["status"] != "ready":
            raise ValueError(f"공개 원문이 커밋 상태와 다르다: {relative} ({row['reason']})")
        row["sha256"] = _sha256_lf(self.root / relative)
        if relative not in self.ids:
            self.ids[relative] = f"S{len(self.ids) + 1:02d}"
            self.docs[relative] = B._read(self.root / relative)
            self.models[relative] = model
        elif self.models[relative] != model:
            raise ValueError(f"한 원문을 서로 다른 모델 실행으로 읽었다: {relative}")
        return self.docs[relative], relative


def cell(story: Story, relative: str, pointers: tuple[str, ...], render: Callable[..., str], *,
         state: str | None = None, basis: str | None = None, evidence: str | None = None) -> dict:
    """원문 pointer 값으로 문장을 만든다. basis는 선택 주체를, evidence는 원문 기록 ID를 가리킨다."""
    document = story.docs[relative]
    values = [at(document, pointer) for pointer in pointers]
    if state == "=value":
        state = values[0]
    selector = None
    if basis is not None:
        selector = _selector(basis, at(document, basis))
    row = {"kind": "fact" if basis is None else "selection", "text": render(*values), "state": state,
           "selector": selector, "source": relative,
           "refs": [{"pointer": pointer, "value": value} for pointer, value in zip(pointers, values)],
           "basis": None if basis is None else {"pointer": basis, "value": at(document, basis)},
           "evidence": None if evidence is None else {"pointer": evidence, "value": str(at(document, evidence))[:12]}}
    if state not in (None, *STATES):
        raise ValueError(f"등록 밖 상태 배지다: {state!r}")
    return row


def _selector(pointer: str, value: Any) -> str:
    try:
        selector = SELECTOR_BASIS.get((pointer.rsplit("/", 1)[-1], value))
    except TypeError:
        selector = None
    if selector is None:
        raise ValueError(f"선택 주체를 원문에서 정할 수 없다: {pointer}={value!r}")
    return selector


def _index(rows: list[dict], **match: Any) -> int:
    found = [index for index, row in enumerate(rows) if all(row.get(key) == value for key, value in match.items())]
    if len(found) != 1:
        raise ValueError(f"원문 행이 하나가 아니다: {match}")
    return found[0]


def _row(label: str, stages: dict[str, list[dict]]) -> dict:
    return {"label": label, "stages": {key: stages.get(key, []) for key, _ in STAGES},
            "final": stages["outcome"][0]["state"]}


def _rmse(a: float, b: float, c: float) -> str:
    return f"선속도 RMSE {a:.3f} → {b:.3f} m/s · 정상 {c:.3f}"


def lane_r1(story: Story) -> dict:
    summary, s = story.doc(f"{R1}/summary.json", MODEL_3_SUPER)
    evidence, e = story.doc(f"{R1}/model_evidence.json", MODEL_3_SUPER)
    loop, p = story.doc(f"{R1}/probe_and_correction.json", MODEL_3_SUPER)
    if (summary.get("schema") != "public_r1_repair_summary_v2" or summary.get("status") != "complete"
            or {evidence[kind]["call"]["provider_model"] for kind in ("diagnosis", "correction")} != {MODEL_3_SUPER}):
        raise ValueError("R1 공개 결과의 범위·모델이 다르다")
    select, result = [], []
    for index, probe in enumerate(loop["probes"]):
        base = f"/probes/{index}"
        select.append(cell(story, p, (f"{base}/order", f"{base}/probe"), lambda order, name: f"{order}. {name}",
                           basis=f"{base}/selection_source", evidence=f"{base}/request_id"))
        if "control" in probe["measurement"]:
            result.append(cell(story, p, (f"{base}/outcome", f"{base}/probe", f"{base}/measurement/case/optimizer_steps",
                                          f"{base}/measurement/control/optimizer_steps"),
                               lambda outcome, name, case, control: (f"{name} {outcome} · optimizer step "
                                                                     f"{'/'.join(map(str, case))} (대조 {'/'.join(map(str, control))})"),
                               state="=value"))
        else:
            result.append(cell(story, p, (f"{base}/outcome", f"{base}/probe"), lambda outcome, name: f"{name} {outcome}",
                               state="=value"))
    stages = {
        "select": select, "result": result,
        "judge": [cell(story, s, ("/diagnostic_sequence/0/raw_ranking_first", "/diagnostic_sequence/0/raw_optimizer_rank"),
                       lambda first, rank: f"모델 첫 순위 {first} · optimizer {rank}위",
                       basis="/diagnostic_sequence/0/stage"),
                  cell(story, p, ("/correction/confirmed_mechanism",), lambda name: f"검사 결과로 {name} 확정",
                       state="confirmed")],
        "correct": [cell(story, p, ("/correction/correction_id", "/correction/recovery_mode"),
                         lambda name, mode: f"등록 교정 {name} · {mode}", evidence="/correction/request_id"),
                    cell(story, s, ("/diagnostic_sequence/2/epoch_transition",),
                         lambda epochs: "PPO epoch " + "→".join(map(str, epochs))),
                    cell(story, e, ("/correction/review_status", "/correction/decision/agrees_with_rule"),
                         lambda status, agree: f"모델 교정 제안 {status} · 규칙과 {'일치' if agree else '불일치'} · 첨부 근거")],
        "outcome": [cell(story, s, ("/outcome",), lambda outcome: outcome, state="=value"),
                    cell(story, s, ("/fixed_evaluation_metrics/R1/lin_vel_rmse_mps",
                                    "/fixed_evaluation_metrics/CORRECTED/lin_vel_rmse_mps",
                                    "/fixed_evaluation_metrics/NONE/lin_vel_rmse_mps"), _rmse)],
    }
    counts = {name: sum(row["selector"] == name for row in select) for name in SELECTORS}
    return {"id": "r1", "title": "R1 PPO epoch 결함 · 등록 종단 수리", "model": MODEL_3_SUPER,
            "model_label": "Nemotron 3 Super · 기록된 실행", "rows": [_row(f"seed {summary['source_training_seed']}", stages)],
            "notes": [f"모델은 첫 검사 {counts['model']}개만 골랐고 이후 {counts['rule']}개 검사는 등록 규칙이 골랐다.",
                      cell(story, s, ("/design_coupling",), str)]}


def lane_residual_v1(story: Story) -> dict:
    summary, s = story.doc(f"{RESIDUAL_V1}/summary.json", MODEL_3_SUPER)
    audit, a = story.doc(f"{RESIDUAL_V1}/model_contract_audit.json", MODEL_3_SUPER)
    if summary.get("schema") != "residual_failure_public_summary_v1" or audit.get("model") != MODEL_3_SUPER:
        raise ValueError("잔여 결함 v1 공개 결과의 범위·모델이 다르다")
    rows = []
    for k, case in enumerate(summary["cases"]):
        seed, base = case["seed"], f"/cases/{k}"
        first = _index(audit["dialogues"], seed=seed, epoch=1, kind="diagnosis")
        second = _index(audit["dialogues"], seed=seed, epoch=2, kind="diagnosis")
        repair = _index(audit["dialogues"], seed=seed, epoch=1, kind="correction")
        rows.append(_row(f"seed {seed}", {
            "select": [cell(story, s, (f"{base}/first_probe",), lambda probe: f"epoch 1 · {probe}",
                            basis=f"{base}/first_diagnosis_valid_after_feedback"),
                       cell(story, a, (f"/dialogues/{first}/requests", f"/dialogues/{first}/status"),
                            lambda n, status: f"epoch 1 진단 요청 {n}회 · 최종 {status}",
                            evidence=f"/dialogues/{first}/record_sha256"),
                       cell(story, a, (f"/dialogues/{second}/status", f"/dialogues/{second}/codes",
                                       f"/dialogues/{second}/requests"),
                            lambda status, codes, n: f"epoch 2 진단 {status} · {', '.join(codes)} · 요청 {n}회 → 검사 선택 없음",
                            state="HOLD", evidence=f"/dialogues/{second}/record_sha256")],
            "result": [cell(story, s, (f"{base}/first_probe_outcome", f"{base}/first_probe",
                                       f"{base}/first_probe_measurement/track_lin_vel_xy_exp_rel_error"),
                            lambda outcome, probe, error: f"{probe} {outcome} · 선속도 항 상대오차 {error:.3f}",
                            state="=value"),
                       cell(story, s, (f"{base}/epoch2_probes_executed",), lambda n: f"epoch 2 검사 {n}개 실행")],
            "judge": [cell(story, s, (f"{base}/evaluation_outcomes/FIRST", f"{base}/epoch2_opened"),
                           lambda outcome, opened: f"첫 교정 {outcome} → epoch 2 {'열림' if opened else '닫힘'}",
                           state="=value")],
            "correct": [cell(story, s, (f"{base}/applied_corrections/0", f"{base}/recovery_mode"),
                             lambda name, mode: f"등록 교정 {name} · {mode}"),
                        cell(story, a, (f"/dialogues/{repair}/status",), lambda status: f"모델 교정 제안 {status}",
                             evidence=f"/dialogues/{repair}/record_sha256"),
                        cell(story, s, (f"{base}/second_correction_executed",),
                             lambda done: "두 번째 교정 실행" if done else "두 번째 교정 미실행")],
            "outcome": [cell(story, s, (f"{base}/outcome", f"{base}/reason"), lambda outcome, reason: f"{outcome} · {reason}",
                             state="=value"),
                        cell(story, s, (f"{base}/metrics/COMPOUND/lin_vel_rmse_mps", f"{base}/metrics/FIRST/lin_vel_rmse_mps",
                                        f"{base}/metrics/NONE/lin_vel_rmse_mps"), _rmse)],
        }))
    return {"id": "residual_v1", "title": "F1+R1 복합 결함 · 첫 판(v1)", "model": MODEL_3_SUPER,
            "model_label": "Nemotron 3 Super · 기록된 실행", "rows": rows,
            "notes": [cell(story, s, ("/wording",), str)]}


def lane_residual_v2(story: Story) -> dict:
    summary, s = story.doc(f"{RESIDUAL_V2}/summary.json", MODEL_3_SUPER)
    audit, a = story.doc(f"{RESIDUAL_V2}/model_audit.json", MODEL_3_SUPER)
    if summary.get("schema") != "residual_success_public_summary_v2" or audit.get("model") != MODEL_3_SUPER:
        raise ValueError("잔여 결함 v2 공개 결과의 범위·모델이 다르다")
    rows = []
    for k, case in enumerate(summary["cases"]):
        seed, base = case["seed"], f"/cases/{k}"
        epochs = [_index(audit["diagnosis_rows"], seed=seed, epoch=epoch) for epoch in (1, 2)]
        stages: dict[str, list[dict]] = {"select": [], "result": [], "correct": []}
        for i in epochs:
            row = f"/diagnosis_rows/{i}"
            stages["select"].append(cell(story, a, (f"{row}/epoch", f"{row}/probe"), lambda epoch, probe: f"epoch {epoch} · {probe}",
                                         basis=f"{row}/selected_by", evidence=f"{row}/diagnosis/record_sha256"))
            stages["result"].append(cell(story, a, (f"{row}/epoch", f"{row}/probe", f"{row}/confirmed_component"),
                                         lambda epoch, probe, part: f"epoch {epoch} · {probe} → {part} 확정",
                                         state="confirmed"))
            stages["correct"].append(cell(story, a, (f"{row}/epoch", f"{row}/correction_id", f"{row}/correction/status"),
                                          lambda epoch, name, status: f"epoch {epoch} · 등록 교정 {name} · 모델 제안 {status}",
                                          evidence=f"{row}/correction/record_sha256"))
        stages["judge"] = [cell(story, s, (f"{base}/repair_verdicts/0",), lambda verdict: f"첫 교정 {verdict} → 두 번째 고리",
                                state="=value"),
                           cell(story, s, ("/rule_transition/numerator", "/rule_transition/denominator"),
                                lambda n, d: f"규칙 전환 {n}/{d}")]
        stages["outcome"] = [cell(story, s, (f"{base}/outcome",), lambda outcome: outcome, state="=value"),
                             cell(story, s, (f"{base}/metrics/COMPOUND/lin_vel_rmse_mps", f"{base}/metrics/BOTH/lin_vel_rmse_mps",
                                             f"{base}/metrics/NONE/lin_vel_rmse_mps"), _rmse),
                             cell(story, s, (f"{base}/final_matches_none/checkpoint_sha256_equal",),
                                  lambda same: "최종 체크포인트 = 같은 seed 정상" if same else "최종 체크포인트 ≠ 같은 seed 정상")]
        rows.append(_row(f"seed {seed}", stages))
    notes = [cell(story, s, ("/wording",), str)]
    if all(case["final_matches_none"]["checkpoint_sha256_equal"] for case in summary["cases"]):
        notes.append("두 교정 뒤 체크포인트가 같은 seed 정상 학습과 같아, 회복은 예상된 확인이며 새로운 회복 효과가 아니다.")
    return {"id": "residual_v2", "title": "F1+R1 복합 결함 · 두 고리(v2)", "model": MODEL_3_SUPER,
            "model_label": "Nemotron 3 Super · 기록된 실행", "rows": rows, "notes": notes}


def lane_u1(story: Story) -> dict:
    rows, notes = [], []
    for tag in U1_TAGS:
        summary, s = story.doc(f"{tag}/summary.json", None)
        scope, c = story.doc(f"{tag}/scope_evidence.json", None)
        _restart, r = story.doc(f"{tag}/evaluations/u1_family_restart.json", None)
        _control, n = story.doc(f"{tag}/evaluations/none_family_control.json", None)
        if scope.get("api_calls") != 0 or scope.get("model_loop") != "inactive" or summary.get("status") != "complete":
            raise ValueError("U1 범위 확장은 모델 호출 0회 실행이어야 한다")
        receipt = "/artifact_sha256_lf/probe_receipt" if "artifact_sha256_lf" in scope else "/artifact_sha256/probe_receipt"
        rows.append(_row(f"seed {summary['source_training_seed']}", {
            "select": [cell(story, c, ("/measured_probe_terms", "/api_calls"),
                            lambda terms, calls: f"등록 검사 · 측정 항 {len(terms)}개 · API {calls}회", basis="/model_loop",
                            evidence=receipt)],
            "result": [cell(story, c, ("/probe_outcome", "/probe_measurement/track_lin_vel_xy_exp_rel_error",
                                       "/probe_measurement/track_ang_vel_z_exp_rel_error"),
                            lambda outcome, lin, yaw: f"{outcome} · 선속도 항 {lin:.3f} · 회전 항 {yaw:.3f}", state="=value")],
            "judge": [cell(story, c, ("/scope_verdict", "/uncovered_terms", "/source_registered_terms"),
                           lambda verdict, missing, terms: f"{verdict} · 원전 {len(terms)}항 중 {len(missing)}항 미포함")],
            "correct": [cell(story, s, ("/scope/restored_terms", "/corrected_u1_restart_training_seed"),
                             lambda count, seed: f"함수군 {count}항 복원 · restart seed {seed}")],
            "outcome": [cell(story, s, ("/corrected_u1_restart/recovery_label",), lambda label: label, state="=value"),
                        cell(story, r, ("/fixed_evaluation/metrics/lin_vel_rmse_mps",),
                             lambda value: f"교정 재학습 선속도 RMSE {value:.3f} m/s"),
                        cell(story, n, ("/fixed_evaluation/metrics/lin_vel_rmse_mps",),
                             lambda value: f"같은 seed 정상 {value:.3f} m/s")],
        }))
        notes.append(cell(story, s, ("/design_limit",), str))
    notes.insert(0, "모델 호출이 없는 실행이라 모델 판단이나 3.5 경로의 회복으로 세지 않는다.")
    return {"id": "u1", "title": "U1 이식 보상 결함 · 함수군 교정", "model": None,
            "model_label": "모델 호출 0회 · 모델 판단 고리 비활성", "rows": rows, "notes": notes}


def lane_fresh(story: Story) -> dict:
    choice, c = story.doc(f"{FRESH_CHOICE}/summary.json", MODEL_3_5)
    response, m = story.doc(f"{FRESH_CHOICE}/model_response.json", MODEL_3_5)
    prefix, _x = story.doc(FRESH_PREFIX, MODEL_3_5)
    operation, o = story.doc(FRESH_OPERATION, MODEL_3_5)
    render, r = story.doc(FRESH_RENDER, MODEL_3_5)
    media, _d = story.doc(FRESH_MEDIA, MODEL_3_5)
    _creation, f = story.doc(FRESH_CREATION, MODEL_3_5)
    if (choice.get("status") != "valid_registered_probe_choice" or choice["model"]["returned"] != MODEL_3_5
            or response.get("returned_model") != MODEL_3_5 or choice["response"]["path"] != m
            or choice["response"]["sha256"] != story.sources.rows[m]["sha256"]
            or prefix.get("model_result") != c or operation.get("target") != choice["decision"]["probe"]
            or media.get("result") != r):
        raise ValueError("3.5 fresh-frame 원문 결합이 다르다")
    later = {"probe_executed": choice["decision"]["probe_executed"], "cause_confirmed": choice["interpretation"]["cause_confirmed"],
             "functional_e2e_completed": operation["functional_e2e_completed"],
             "new_recovery_result": choice["interpretation"]["new_recovery_result"],
             "probe_receipt_created": prefix["probe_receipt_created"]}
    if any(value is not False for value in later.values()):
        raise ValueError(f"3.5 경로에 새 단계 결과가 생겼다. 화면 계약을 먼저 갱신한다: {later}")
    videos = []
    for role, key in (("input_reference", "reference"), ("input_case", "case")):
        ref = media["policies"][key]
        if ref != render["policies"][key]["video"] or not TV.HEX64.fullmatch(ref["sha256"]):
            raise ValueError("영상 manifest와 녹화 요약의 SHA가 다르다")
        videos.append(video_slot(story, ref["path"], ref["sha256"], role, f"seed {media['seed']}"))
    stages = {
        "select": [cell(story, c, ("/decision/probe", "/model/returned"), lambda probe, model: f"{probe} · {model}",
                        basis="/decision/selected_by", evidence="/response/sha256"),
                   cell(story, c, ("/citation_counts/telemetry", "/citation_counts/frame", "/citation_counts/same_play"),
                        lambda telemetry, frame, same: f"인용 telemetry {telemetry} · frame {frame} · same_play {same}"),
                   cell(story, c, ("/decision/contract_valid", "/decision/approval_granted"),
                        lambda valid, approved: f"계약 {'valid' if valid else 'invalid'} · 승인 {'기록됨' if approved else '없음'}")],
        "result": [cell(story, c, ("/decision/probe_executed",), lambda done: "검사 미실행", state="pending"),
                   cell(story, o, ("/runtime_state", "/status"), lambda state, status: f"runtime {state} · {status}",
                        state="pending")],
        "judge": [cell(story, c, ("/interpretation/cause_confirmed", "/interpretation/image_selection_benefit_established"),
                       lambda cause, image: f"원인 미확정 · 영상 선택 기여 {'확인' if image else '미측정'}", state="pending")],
        "correct": [cell(story, o, ("/functional_e2e_completed",), lambda done: "교정 단계 미도달", state="pending")],
        "outcome": [cell(story, c, ("/interpretation/new_recovery_result",), lambda value: "새 회복 결과 없음", state="pending")],
    }
    counts = choice["citation_counts"]
    return {"id": "fresh35", "title": "F1 보상 결함 · 새 3.5 fresh-frame 경로", "model": MODEL_3_5,
            "model_label": "Nemotron 3.5 Super VL · 첫 모델 선택만", "rows": [_row(f"seed {choice['already_seen_seed']}", stages)],
            "inputs": [cell(story, r, ("/policies/case/frames", "/policies/case/fps", "/new_training"),
                            lambda frames, fps, train: f"같은 Play 녹화 {frames} frame · {fps} fps · 새 학습 {train}"),
                       cell(story, f, ("/outcome", "/new_vm_count"), lambda outcome, vms: f"이전 렌더 환경 생성 {outcome} · 새 VM {vms}",
                            state="execution_failed")],
            "videos": videos,
              "notes": [(f"공개 결과는 {choice['decision']['probe']} 선택과 telemetry 인용 "
                         f"{counts['telemetry']}·frame 인용 {counts['frame']}까지다. "
                         "검사·교정·회복은 pending이다."),
                      "옛 3 Super 결과와 모델 호출 0회 U1 결과를 이 경로의 회복으로 합치지 않는다."]}


def video_slot(story: Story, relative: str, expected: str, role: str, label: str) -> dict:
    """커밋된 파일이 manifest SHA와 같고 stale-render 목록에 없을 때만 영상을 ready로 둔다."""
    if role not in VIDEO_ROLES:
        raise ValueError(f"등록 밖 영상 역할이다: {role}")
    row = story.sources.file(relative, kind="video", expected_sha256=expected)
    status, reason = row["status"], row["reason"]
    if row["sha256"] in story.stale_videos:
        status, reason = "mismatch", "stale_render_clip"
    return {"label": label, "role": role, "caption": VIDEO_ROLES[role], "path": relative, "sha256": row["sha256"],
            "expected_sha256": expected, "status": status, "reason": reason}


def stairs_slots(story: Story) -> list[dict]:
    """H1·Go2 계단 슬롯은 커밋된 공개 manifest와 같은 SHA의 영상이 있을 때만 ready다."""
    found: dict[tuple, list[tuple[str, dict]]] = {}
    for path in sorted(story.root.glob("docs/media/**/manifest.json")):
        try:
            document = B._read(path)
        except (OSError, ValueError):
            continue
        if isinstance(document, dict) and document.get("schema") == STAIRS_SCHEMA:
            key = (document.get("robot_role"), document.get("direction"))
            found.setdefault(key, []).append((path.relative_to(story.root).as_posix(), document))
    slots = []
    for role, direction, label in STAIRS_SLOTS:
        slot = {"id": f"stairs_{role}_{direction}", "label": label, "status": "waiting", "reason": "public_manifest_missing",
                "manifest": None, "video": None}
        candidates = found.get((role, direction), [])
        if len(candidates) > 1:
            slot.update(status="mismatch", reason="duplicate_manifest")
        elif candidates:
            relative, document = candidates[0]
            row = story.sources.file(relative, kind="stairs_manifest")
            slot["manifest"] = {"path": relative, "sha256": row["sha256"]}
            video = document.get("video") if isinstance(document.get("video"), dict) else {}
            if row["status"] != "ready":
                slot.update(status=row["status"], reason=f"manifest_{row['reason']}")
            elif (document.get("training_task"), document.get("play_task")) != STAIRS_TASKS[role]:
                slot.update(status="mismatch", reason="task_mismatch")
            elif not isinstance(video.get("path"), str) or not TV.HEX64.fullmatch(str(video.get("sha256"))):
                slot.update(status="mismatch", reason="video_ref_invalid")
            else:
                try:
                    video_relative = (Path(relative).parent / TV._relative(video["path"])).as_posix()
                except ValueError:
                    slot.update(status="mismatch", reason="video_ref_invalid")
                else:
                    slot["video"] = video_slot(story, video_relative, video["sha256"], "stairs_playback", label)
                    slot.update(status=slot["video"]["status"], reason=slot["video"]["reason"])
        slots.append(slot)
    return slots


def build(root: Path = ROOT) -> dict:
    story = Story(root)
    audit, _ = story.doc(RENDER_AUDIT, None)
    story.stale_videos = {row["video_sha256"] for row in audit["native_source_hashes"].values()}
    lanes = [lane_r1(story), lane_residual_v1(story), lane_residual_v2(story), lane_u1(story), lane_fresh(story)]
    sources = {}
    for path, row in story.sources.rows.items():
        sources[path] = {"sid": story.ids.get(path), "path": row["path"], "kind": row["kind"], "sha256": row["sha256"],
                         "status": row["status"], "reason": row["reason"],
                         "model": story.models.get(path) if path in story.ids else None}
    data = {"schema": SCHEMA, "scope": SCOPE, "lanes": lanes, "stairs": stairs_slots(story), "sources": sources}
    check_story(data, story.root)
    return data


def _cells(data: dict):
    for lane in data["lanes"]:
        for row in lane["rows"]:
            for stage, cells in row["stages"].items():
                for item in cells:
                    yield lane, stage, item
        for item in [*lane.get("inputs", []), *[note for note in lane["notes"] if isinstance(note, dict)]]:
            yield lane, "input", item


def check_story(data: dict, root: Path) -> None:
    """원문 값·선택 주체·모델 실행 분리·영상 상태를 다시 확인한다. 하나라도 어긋나면 화면을 만들지 않는다."""
    sources, problems, raw = data["sources"], [], {}
    for lane, stage, item in _cells(data):
        source = sources.get(item["source"])
        if source is None or source["status"] != "ready" or source["sid"] is None:
            problems.append(f"{lane['id']}: 확인되지 않은 원문 {item['source']}")
            continue
        if source["model"] != lane["model"]:
            problems.append(f"{lane['id']}: 다른 모델 실행의 원문을 섞었다 ({item['source']})")
        if item["source"] not in raw:
            raw[item["source"]] = B._read(root / item["source"])
        document = raw[item["source"]]
        for ref in [*item["refs"], *([item["basis"]] if item["basis"] else [])]:
            if at(document, ref["pointer"]) != ref["value"]:
                problems.append(f"{lane['id']}: 원문 값과 다르다 {item['source']}#{ref['pointer']}")
        if item["state"] not in (None, *STATES) or item["selector"] not in (None, *SELECTORS):
            problems.append(f"{lane['id']}: 등록 밖 배지 {item['state']!r}/{item['selector']!r}")
        basis = item["basis"]
        expected = None if basis is None else SELECTOR_BASIS.get((basis["pointer"].rsplit("/", 1)[-1], basis["value"]))
        if item["kind"] != ("fact" if basis is None else "selection") or expected != item["selector"]:
            problems.append(f"{lane['id']}: 선택 주체가 원문 필드와 다르다 ({item['text']})")
        if item["selector"] == "model" and lane["model"] is None:
            problems.append(f"{lane['id']}: 모델 호출 0회 실행에 model 선택을 붙였다")
        if lane["model"] == MODEL_3_5 and stage not in ("select", "input") and item["state"] not in FRESH_OPEN_STATES:
            problems.append(f"{lane['id']}: 3.5 경로의 미완료 단계에 결과 배지를 붙였다 ({item['state']})")
    for lane in data["lanes"]:
        for row in lane["rows"]:
            if row["final"] != row["stages"]["outcome"][0]["state"]:
                problems.append(f"{lane['id']}: 최종 배지가 회복 단계 원문과 다르다")
            if not any(item["kind"] == "selection" or item["state"] == "HOLD" for item in row["stages"]["select"]):
                problems.append(f"{lane['id']}: 검사 선택 단계에 원문 선택 주체가 없다")
        for video in lane.get("videos", []):
            if video["role"] not in VIDEO_ROLES or video["caption"] != VIDEO_ROLES[video["role"]]:
                problems.append(f"{lane['id']}: 등록 밖 영상 표시 {video['role']}")
    for slot in data["stairs"]:
        video = slot["video"]
        if slot["status"] == "ready" and (video is None or video["status"] != "ready" or slot["manifest"] is None):
            problems.append(f"{slot['id']}: manifest·영상 없이 ready로 표시했다")
    texts = [text for text in B.text_fields({"lanes": data["lanes"], "stairs": data["stairs"], "scope": data["scope"]})]
    problems += [problem for text in texts for problem in B.claim_violations(text)]
    problems += B.path_violations(json.dumps(data, ensure_ascii=False))
    if problems:
        raise ValueError(f"agentic story replay is not publishable: {problems[:5]}")


STYLE = """
:root{--nv:#76b900;--nv-dark:#4d7800;--ink:#17220f;--muted:#5d6b57;--line:#dbe5d5;--soft:#f6f9f3}
*{box-sizing:border-box} html,body{margin:0;background:#fff;color:var(--ink);font-family:"Noto Sans KR","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif}
#viewport{position:fixed;inset:0;overflow:hidden;background:#fff}
#stage{position:absolute;width:1920px;height:1080px;overflow:hidden;padding:36px 56px 70px;background:#fff}
.scene{display:none;height:100%;flex-direction:column;gap:14px;overflow:hidden}
.scene.active{display:flex}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;border-bottom:4px solid var(--nv);padding-bottom:12px}
.kicker{color:var(--nv-dark);font-weight:800;letter-spacing:.06em;font-size:15px}
h1{font-size:38px;line-height:1.12;margin:4px 0 6px} .scope{margin:0;color:var(--muted);font-size:17px}
.model{border:2px solid var(--nv);border-radius:12px;padding:8px 14px;font-size:17px;font-weight:800;text-align:right}
.model small{display:block;font:500 12px Consolas,monospace;color:var(--muted)}
.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.flow div{border:2px solid var(--nv);border-radius:12px;padding:12px 16px;font-size:22px;font-weight:800;position:relative;background:#fff}
.flow div:not(:last-child):after{content:"→";position:absolute;right:-12px;top:50%;transform:translate(50%,-50%);color:var(--nv-dark);font-size:22px}
.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:15px;color:var(--muted);align-items:center}
table{border-collapse:collapse;width:100%} th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top;font-size:17px}
th{color:var(--nv-dark);font-size:15px}
.grid{display:grid;grid-template-columns:150px repeat(5,minmax(0,1fr));gap:10px;flex:1;min-height:0}
.grid.rows-1{grid-template-rows:auto minmax(0,1fr)} .grid.rows-2{grid-template-rows:auto minmax(0,1fr) minmax(0,1fr)}
.colhead{font-weight:800;color:var(--nv-dark);font-size:18px;border-bottom:3px solid var(--nv);padding:2px 2px 6px}
.rowhead{font-weight:800;font-size:19px;padding-top:6px;display:flex;flex-direction:column;gap:8px;align-items:flex-start}
.cell{border:1px solid var(--line);border-radius:10px;background:var(--soft);padding:9px 10px;overflow:hidden;display:flex;flex-direction:column;gap:8px;min-height:0}
.step{font-size:17px;line-height:1.28;overflow-wrap:anywhere} .chips{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:2px}
.ev{font:12px/1.25 Consolas,"SFMono-Regular",monospace;color:var(--muted);word-break:break-all}
.sel,.st,.src{font-size:13px;font-weight:800;padding:1px 7px;border-radius:4px;border:1px solid transparent;white-space:nowrap}
.sel-model{color:#155a96;border-color:#155a96;background:#fff}.sel-rule{color:#7a5600;border-color:#a77d00;background:#fff}
.sel-host{color:#3f3f3f;border-color:#777;background:#fff}
.st-recovered{background:var(--nv);color:#112300}.st-confirmed{background:#dcefbf;color:#2d4a00}
.st-not_recovered{background:#fde3d4;color:#8a3200}.st-HOLD{background:#fff0bd;color:#6b4e00;border-color:#d4a900}
.st-execution_failed{background:#f7d2d2;color:#8a0000}.st-pending{background:#fff;color:#666;border:1px dashed #999}
.st-normal{background:#eef2ea;color:#43513d}.st-abnormal{background:#ffe1e1;color:#8a0000}
.src-ready{background:#dcefbf;color:#2d4a00}.src-waiting{background:#fff;color:#666;border:1px dashed #999}.src-mismatch{background:#f7d2d2;color:#8a0000}
.inputs{display:grid;grid-template-columns:330px 330px 1fr;gap:12px;height:230px;flex:none}
figure{margin:0;border-radius:10px;overflow:hidden;background:#0c1209;display:flex;flex-direction:column}
figure video{width:100%;height:186px;object-fit:cover;display:block;background:#0c1209}
figcaption{color:#fff;font-size:13px;padding:5px 9px} .placeholder{height:186px;display:flex;align-items:center;justify-content:center;color:#ddd;font-size:15px;padding:12px;text-align:center}
.notes{margin:0;padding-left:20px;color:var(--muted);font-size:15px;line-height:1.35;flex:none}
.slots{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;flex:1;min-height:0}
.slot{border:2px dashed #b9c7b1;border-radius:14px;padding:16px;display:flex;flex-direction:column;gap:10px;overflow:hidden}
.slot.ready{border-style:solid;border-color:var(--nv)} .slot h2{margin:0;font-size:24px}
.mono{font:12px/1.3 Consolas,"SFMono-Regular",monospace;word-break:break-all}
.sources td{font-size:13px;padding:5px 8px} .sources td.mono{font-size:12px}
#nav{position:absolute;left:56px;right:56px;bottom:18px;display:flex;align-items:center;gap:14px;font-size:15px;color:var(--muted)}
#nav button{font:inherit;font-weight:800;color:var(--ink);background:#fff;border:2px solid var(--nv);border-radius:999px;padding:5px 18px;cursor:pointer}
#nav .spacer{flex:1}
"""

SCRIPT = """(()=>{const s=document.getElementById('stage'),v=document.getElementById('viewport'),scenes=[...document.querySelectorAll('.scene')];let index=0;
function fit(){const k=Math.min(innerWidth/1920,innerHeight/1080);s.style.transform=`scale(${k})`;s.style.transformOrigin='top left';s.style.left=`${(innerWidth-1920*k)/2}px`;s.style.top=`${(innerHeight-1080*k)/2}px`}
function measure(scene){const clipped=[...scene.querySelectorAll('.cell,.slot,.inputs,.notes,table')].concat([scene]).filter(e=>e.scrollHeight>e.clientHeight+1||e.scrollWidth>e.clientWidth+1).map(e=>e.dataset.id||e.id||e.className);return{scene:scene.id,width:1920,height:1080,overflowX:Math.max(0,s.scrollWidth-1920),overflowY:Math.max(0,s.scrollHeight-1080),clipped}}
function show(i){index=(i+scenes.length)%scenes.length;scenes.forEach((e,k)=>e.classList.toggle('active',k===index));document.getElementById('pos').textContent=`${index+1} / ${scenes.length} · ${scenes[index].dataset.title}`;
const m=measure(scenes[index]);window.__STORY_LAYOUT__=m;v.dataset.overflow=m.overflowX+m.overflowY+m.clipped.length===0?'0':'1'}
window.__storyMeasureAll=()=>{const keep=index,out=scenes.map((e,k)=>{show(k);return measure(e)});show(keep);return out};
document.getElementById('prev').onclick=()=>show(index-1);document.getElementById('next').onclick=()=>show(index+1);
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}});
const byHash=()=>scenes.findIndex(e=>'#'+e.id===location.hash);addEventListener('hashchange',()=>{const k=byHash();if(k>=0&&k!==index)show(k)});
addEventListener('resize',fit);fit();show(Math.max(0,byHash()))})();"""


def _chip(kind: str, value: str | None, label: str | None = None) -> str:
    return "" if value is None else f'<span class="{kind} {kind}-{TV._escape(value)}">{TV._escape(label or value)}</span>'


def _step(item: dict, sources: dict) -> str:
    sid = sources[item["source"]]["sid"]
    pointer = (item["basis"] or item["refs"][0])["pointer"]
    evidence = f"{sid}#{pointer}" + (f" · id {item['evidence']['value']}" if item["evidence"] else "")
    chips = _chip("sel", item["selector"]) + _chip("st", item["state"], STATES.get(item["state"] or ""))
    return (f'<div class="step">{f"<div class=chips>{chips}</div>" if chips else ""}<div>{TV._escape(item["text"])}</div>'
            f'<div class="ev">{TV._escape(evidence)}</div></div>')


def _notes(notes: list, sources: dict) -> str:
    items = []
    for note in notes:
        if isinstance(note, dict):
            where = sources[note["source"]]["sid"] + "#" + note["refs"][0]["pointer"]
            items.append(f'<li>{TV._escape(note["text"])} <span class="ev">({TV._escape(where)})</span></li>')
        else:
            items.append(f"<li>{TV._escape(note)}</li>")
    return f'<ul class="notes">{"".join(items)}</ul>'


def _video(video: dict, root: Path, output: Path) -> str:
    caption = f'<figcaption>{TV._escape(video["label"])} · {TV._escape(video["caption"])}</figcaption>'
    if video["status"] != "ready":
        status = f"{SOURCE_STATUS[video['status']]} · {video['reason']}"
        return f'<figure><div class="placeholder">영상 {TV._escape(status)}</div>{caption}</figure>'
    return (f'<figure><video muted controls preload="metadata" data-sha256="{video["sha256"]}" '
            f'src="{TV._escape(TV._web_path(root, output, video["path"]))}"></video>{caption}</figure>')


def _top(kicker: str, title: str, scope: str, side: str = "") -> str:
    return (f'<header class="top"><div><div class="kicker">{TV._escape(kicker)}</div><h1>{TV._escape(title)}</h1>'
            f'<p class="scope">{TV._escape(scope)}</p></div>{side}</header>')


def _selection_counts(lane: dict) -> str:
    items = [item for row in lane["rows"] for item in row["stages"]["select"] if item["kind"] == "selection"]
    return " · ".join(f"{name} {sum(item['selector'] == name for item in items)}" for name in SELECTORS
                      if any(item["selector"] == name for item in items))


def _overview(data: dict, total: int) -> str:
    rows = []
    for lane in data["lanes"]:
        finals = " ".join(f'{TV._escape(row["label"])} {_chip("st", row["final"], STATES[row["final"]])}' for row in lane["rows"])
        rows.append(f'<tr><td>{TV._escape(lane["title"])}</td><td>{TV._escape(lane["model_label"])}</td>'
                    f'<td>{TV._escape(_selection_counts(lane))}</td><td>{finals}</td></tr>')
    ready = sum(slot["status"] == "ready" for slot in data["stairs"])
    rows.append(f'<tr><td>H1·Go2 계단 슬롯</td><td>공식 rough 정책 재생 · 모델 판단 없음</td><td>-</td>'
                f'<td>ready {ready}/{len(data["stairs"])}</td></tr>')
    legend = ("".join(_chip("sel", name) for name in SELECTORS) + "<span>검사 선택 주체(원문 필드로만 표시)</span>"
              + "".join(_chip("st", key, label) for key, label in STATES.items()))
    flow = "".join(f"<div>{TV._escape(label)}</div>" for _, label in STAGES)
    return ('<section class="scene" id="overview" data-title="개요">'
            + _top(f"NVIDIA · RL TRIAGE · 1/{total}", "검사 선택부터 회복/HOLD까지, 공개 원문으로 따라가기", data["scope"])
            + f'<div class="flow">{flow}</div><div class="legend">{legend}</div>'
            f'<table><thead><tr><th>실행</th><th>모델</th><th>검사 선택 주체(개수)</th><th>seed별 최종 상태</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>'
            '<ul class="notes"><li>각 칸 아래의 S번호#pointer는 마지막 화면의 원문 경로·sha256과 그 안의 JSON 위치다.</li>'
            '<li>모델 세대가 다른 실행(3 Super·3.5)과 모델 호출 0회 실행은 서로 다른 화면에 두고 합산하지 않는다.</li></ul></section>')


def _lane(lane: dict, number: int, total: int, data: dict, root: Path, output: Path) -> str:
    sources = data["sources"]
    model = (f'<div class="model">{TV._escape(lane["model_label"])}'
             f'<small>{TV._escape(lane["model"] or "model: none")}</small></div>')
    inputs = ""
    if lane.get("videos") or lane.get("inputs"):
        videos = "".join(_video(video, root, output) for video in lane.get("videos", []))
        steps = "".join(_step(item, sources) for item in lane.get("inputs", []))
        inputs = f'<div class="inputs">{videos}<div class="cell" data-id="{lane["id"]}-inputs">{steps}</div></div>'
    head = "<div></div>" + "".join(f'<div class="colhead">{TV._escape(label)}</div>' for _, label in STAGES)
    body = []
    for index, row in enumerate(lane["rows"]):
        body.append(f'<div class="rowhead">{TV._escape(row["label"])}{_chip("st", row["final"], STATES[row["final"]])}</div>')
        for key, _ in STAGES:
            steps = "".join(_step(item, sources) for item in row["stages"][key])
            body.append(f'<div class="cell" data-id="{lane["id"]}-{index}-{key}">{steps}</div>')
    return (f'<section class="scene" id="lane-{lane["id"]}" data-title="{TV._escape(lane["title"])}">'
            + _top(f"NVIDIA · RL TRIAGE · {number}/{total}", lane["title"], data["scope"], model) + inputs
            + f'<div class="grid rows-{len(lane["rows"])}">{head}{"".join(body)}</div>'
            + _notes(lane["notes"], sources) + "</section>")


def _stairs(data: dict, number: int, total: int, root: Path, output: Path) -> str:
    cards = []
    for slot in data["stairs"]:
        status = _chip("src", slot["status"], SOURCE_STATUS[slot["status"]])
        manifest = ("공개 manifest 없음" if slot["manifest"] is None
                    else f'{TV._escape(slot["manifest"]["path"])}<br>{slot["manifest"]["sha256"] or "-"}')
        video = "" if slot["video"] is None else _video(slot["video"], root, output)
        cards.append(f'<div class="slot{" ready" if slot["status"] == "ready" else ""}" data-id="{slot["id"]}">'
                     f'<h2>{TV._escape(slot["label"])}</h2><div>{status} <span class="mono">{TV._escape(slot["reason"] or "")}</span></div>'
                     f'<div class="mono">{manifest}</div>{video}</div>')
    return ('<section class="scene" id="stairs" data-title="계단 슬롯">'
            + _top(f"NVIDIA · RL TRIAGE · {number}/{total}", "H1·Go2 계단 재생 슬롯",
                   "커밋된 공개 manifest(stairs_playback_manifest_v2)와 같은 SHA의 영상이 있을 때만 ready다. "
                   "공식 rough 정책 재생이며 교정 영상이나 계단 성공 판정이 아니다.")
            + f'<div class="slots">{"".join(cards)}</div></section>')


def _sources(data: dict, number: int, total: int) -> str:
    rows = []
    for row in sorted(data["sources"].values(), key=lambda item: (item["sid"] is None, item["sid"] or "", item["path"] or "")):
        rows.append(f'<tr><td>{TV._escape(row["sid"] or row["kind"])}</td><td class="mono">{TV._escape(row["path"])}</td>'
                    f'<td class="mono">{row["sha256"] or "-"}</td><td>{_chip("src", row["status"], SOURCE_STATUS[row["status"]])}</td>'
                    f'<td class="mono">{TV._escape(row["model"] or "-")}</td></tr>')
    return ('<section class="scene" id="sources" data-title="원문 결합">'
            + _top(f"NVIDIA · RL TRIAGE · {number}/{total}", "원문 경로와 sha256",
                   "화면의 모든 수치·배지는 아래 커밋된 파일의 값이다. 상태는 커밋과 작업 트리 바이트가 같은지를 뜻한다.")
            + '<table class="sources"><thead><tr><th>ID</th><th>경로</th><th>sha256</th><th>상태</th><th>모델 실행</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></section>')


def render(data: dict, root: Path = ROOT, output: Path = OUTPUT) -> str:
    total = len(data["lanes"]) + 3
    scenes = [_overview(data, total)]
    scenes += [_lane(lane, index, total, data, root, output) for index, lane in enumerate(data["lanes"], start=2)]
    scenes += [_stairs(data, total - 1, total, root, output), _sources(data, total, total)]
    bindings = TV._script_json({"schema": SCHEMA, "sources": data["sources"]})
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="color-scheme" content="light">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RL Triage · 근거로 따라가는 결과 재생</title>
<link rel="icon" href="data:,"><style>{STYLE}</style></head><body><div id="viewport"><main id="stage">
{"".join(scenes)}
<nav id="nav"><button id="prev" type="button">← 이전</button><button id="next" type="button">다음 →</button><span id="pos"></span><span class="spacer"></span><span>1920 × 1080 · 외부 리소스 0</span></nav>
</main></div><script type="application/json" id="story-sources">{bindings}</script>
<script>{SCRIPT}</script></body></html>
"""
    if EXTERNAL.search(html) or B.path_violations(html):
        raise ValueError("HTML이 외부 리소스나 비공개 경로를 참조한다")
    return html


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="파일을 쓰지 않고 현재 HTML이 원문과 같은지만 확인한다")
    args = parser.parse_args(argv)
    content = render(build(root), root, OUTPUT)
    target = root / OUTPUT
    if args.check:
        current = target.is_file() and not target.is_symlink() and target.read_text(encoding="utf-8") == content
        print(json.dumps({"status": "current" if current else "stale", "output": OUTPUT.as_posix()}, ensure_ascii=False))
        return 0 if current else 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "written", "output": OUTPUT.as_posix()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
