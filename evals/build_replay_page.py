"""발표 재생 화면 데이터(docs/replay/data.js)를 승인된 공개 결과에서 만든다.

승인은 저장소에 커밋돼 있고 작업 트리가 그 커밋과 같은 파일을 뜻한다. 공개 묶음은 각 publisher의
replay까지 통과해야 한다. 파일이 없거나 커밋 전이면 대기, 수정됐거나 해시·replay가 맞지 않으면
불일치로 표시하고 그 숫자를 화면에 넣지 않는다. GPU·모델·네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

OUTPUT = Path("docs/replay/data.js")
SLOTS = Path("docs/replay/slots.json")
SCHEMA = "replay_page_data_v1"
SEEDS = (2026, 2027, 2028)
CASES = ("baseline_p0c", "h01", "h02", "h03", "h04", "h05", "h06")
CURVE_METRICS = {
    "error_xy": ("Metrics/base_velocity/error_vel_xy", "명령 추종 오차(기록값, 낮을수록 좋음)"),
    "tracking_reward": ("Episode_Reward/track_lin_vel_xy_exp", "선속도 추종 보상 항"),
    "episode_length": ("Train/mean_episode_length", "학습 에피소드 길이(step)"),
    "total_reward": ("Train/mean_reward", "학습 총보상 평균"),
    "noise_std": ("Policy/mean_noise_std", "기록된 행동 노이즈 std"),
}
R1 = {"window_iterations": 20, "eligible_iteration_min": 60, "persistence_iterations": 10, "error_ratio": 2.0}
FORBIDDEN_ALWAYS = ("최초", "보류 평가", "보류 세트 성능")
NEEDS_NEGATION = ("일반화", "절감", "우월", "규칙보다", "보다 정확", "보다 낫")
NEGATIONS = ("아니", "아닙", "않", "없", "금지", "철회", "말하지")
PRIVATE_PATTERNS = (re.compile(r"[A-Za-z]:[\\/]"), re.compile(r"/home/|/Users/|LIMMM", re.I),
                    re.compile(r"bench/private", re.I), re.compile(r'"(pid|process_id|process_identity)"'),
                    re.compile(r"https?://"))
STATUS = ("ready", "waiting", "mismatch")
STATUS_WORDS = ("대기", "불일치")
# 하단 범위 문구. 장면의 scope가 있으면 공통 문구(SCOPE_COMMON)를 덮어쓴다. 장면별로 사실에 맞게 쓴다.
SCOPE_COMMON = "승인된 공개 결과만 읽는 발표 재생 화면. 이미 본 합성 결함 개발 사례이며 일반화·GPU 절감 주장이 아니다."
SCOPE_NEW_SEED_DEMO = "승인된 공개 결과만 읽는 발표 재생 화면. 이미 본 결함 종류의 새 seed 단일 실행이며 일반화·GPU 절감 주장이 아니다."
SCOPE_UPSTREAM_SMOKE = ("다른 프레임워크에서 원인이 확정된 결함을 우리가 이식해 주입한 사례이며, 이미 쓴 seed 2026의 단일 smoke 실행이다. "
                        "진단 정확도·교정 성공률·일반화의 근거가 아니다.")
SCOPE_HOLD = "외부 원인 결함을 주입한 사례(T1)와 결함 없는 정상 입력(T2)의 이미 본 개발 기록이며, 정확도·일반화의 근거가 아니다."
SCOPE_UPSTREAM_DEMO = ("다른 프레임워크에서 원인이 확정된 결함을 우리가 이식해 주입한 사례이며, 새 seed 단일 실행이다. "
                       "진단 정확도·교정 성공률·일반화의 근거가 아니다.")
MOTOR_CURVE_STATES = {"complete": "완료", "failed": "기록 실패", "invalid": "무효", "not_executed": "미실행"}
RECOVERY_LABELS = {"recovered": "회복", "not_recovered": "미회복", "invalid": "판정 보류"}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Sources:
    """화면 숫자의 근거 파일과 승인 상태를 모은다."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.rows: dict[str, dict] = {}
        self.replays: dict[str, dict | None] = {}

    def _committed_state(self, relative: str) -> tuple[str, str | None]:
        if _git(self.root, "ls-files", "--error-unmatch", "--", relative).returncode != 0:
            return "waiting", "not_committed"
        if _git(self.root, "status", "--porcelain", "--", relative).stdout.strip():
            return "mismatch", "modified_since_commit"
        return "ready", None

    def file(self, relative: str | None, *, kind: str = "file", expected_sha256: str | None = None) -> dict:
        key = relative or f"slot:{kind}"
        if key in self.rows:
            return self.rows[key]
        row = {"id": key, "path": relative, "kind": kind, "sha256": None, "status": "waiting", "reason": "path_not_set"}
        if relative is not None:
            path = self.root / relative
            if not path.is_file():
                row["reason"] = "missing"
            else:
                row["sha256"] = _sha(path)
                row["status"], row["reason"] = self._committed_state(relative)
                if row["status"] == "ready" and expected_sha256 is not None and row["sha256"] != expected_sha256:
                    row["status"], row["reason"] = "mismatch", "sha_mismatch"
        self.rows[key] = row
        return row

    def bundle(self, relative: str, replay: Callable[[Path], dict], *, kind: str) -> tuple[dict, dict | None]:
        """공개 묶음 폴더 전체가 커밋돼 있고 publisher replay를 통과해야 확인으로 둔다."""
        if relative in self.replays:
            return self.rows[relative], self.replays[relative]
        self.replays[relative] = None
        manifest = f"{relative}/artifact_manifest.json"
        row = {"id": relative, "path": manifest, "kind": kind, "sha256": None, "status": "waiting", "reason": "missing"}
        self.rows[relative] = row
        folder = self.root / relative
        if not (folder / "artifact_manifest.json").is_file():
            return row, None
        row["sha256"] = _sha(folder / "artifact_manifest.json")
        if not _git(self.root, "ls-files", "--", relative).stdout.strip():
            row["reason"] = "not_committed"
            return row, None
        if _git(self.root, "status", "--porcelain", "--", relative).stdout.strip():
            row["status"], row["reason"] = "mismatch", "modified_since_commit"
            return row, None
        try:
            result = replay(folder)
        except Exception as error:  # 화면은 원인 대신 불일치만 표시한다.
            row["status"], row["reason"] = "mismatch", f"replay_failed:{type(error).__name__}"
            return row, None
        row["status"], row["reason"] = "ready", None
        self.replays[relative] = result
        return row, result

    def load(self, relative: str, *, kind: str = "file") -> tuple[Any, str]:
        row = self.file(relative, kind=kind)
        return (_read(self.root / relative) if row["status"] == "ready" else None), row["id"]

    def ready(self, key: str) -> bool:
        return self.rows[key]["status"] == "ready"


def cell(sources: Sources, value: Any, source: str, digits: int | None = 3) -> dict:
    """근거가 확인된 숫자만 값으로 남긴다."""
    return {"value": value if sources.ready(source) else None, "source": source, "digits": digits}


def smooth(values: list[float], window: int) -> list[float | None]:
    return [None if t < window - 1 else statistics.fmean(values[t - window + 1:t + 1]) for t in range(len(values))]


def r1_alerts(series: dict[str, list[float]], faults: dict[str, str]) -> dict[str, dict]:
    """사후 후보 경고 R1을 공개 텔레메트리로 다시 계산한다(다른 두 seed의 정상 실행을 기준으로 둔다)."""
    smoothed = {run: smooth(values, R1["window_iterations"]) for run, values in series.items()}
    result = {}
    for case in CASES:
        for seed in SEEDS:
            run = f"{case}_s{seed}"
            refs = [f"baseline_p0c_s{other}" for other in SEEDS if other != seed]
            streak, events = 0, []
            for t, value in enumerate(smoothed[run]):
                flag = (t >= R1["eligible_iteration_min"] and value is not None
                        and value > R1["error_ratio"] * max(smoothed[ref][t] for ref in refs))
                streak = streak + 1 if flag else 0
                if streak == R1["persistence_iterations"]:
                    events.append(t)
            result[run] = {"case": case, "fault": faults[case], "seed": seed, "alert_iterations": events}
    return result


def _fmt_sources(sources: Sources, paths: list[str]) -> list[str]:
    return [sources.file(path)["id"] for path in paths]


def media_video(sources: Sources, folder: str) -> dict:
    """비교 영상은 같은 폴더 source_manifest.json의 mp4 SHA와 맞고 manifest가 확인일 때만 확인으로 둔다."""
    manifest, manifest_id = sources.load(f"{folder}/source_manifest.json", kind="media_manifest")
    expected = manifest["comparison"]["mp4"]["sha256"] if manifest else None
    video = sources.file(f"{folder}/comparison.mp4", kind="video", expected_sha256=expected)
    if manifest is None:
        video.update(status="waiting" if sources.rows[manifest_id]["status"] == "waiting" else "mismatch",
                     reason="media_manifest_not_ready")
    return video


def scene_failure(sources: Sources, slots: dict) -> dict:
    rows = []
    for seed in SEEDS:
        fault, fault_id = sources.load(f"evals/results/p0c_fixed_eval_20261009/runs/h02_s{seed}.json")
        clean, clean_id = sources.load(f"evals/results/p0c_fixed_eval_20261009/runs/baseline_p0c_s{seed}.json")
        ratio = (lambda report: report["metrics"]["lin_vel_rmse_mps"] / report["standstill"]["lin_vel_rmse_mps"])
        rows.append([str(seed),
                     cell(sources, fault and fault["metrics"]["lin_vel_rmse_mps"], fault_id),
                     cell(sources, fault and fault["standstill"]["lin_vel_rmse_mps"], fault_id),
                     cell(sources, fault and ratio(fault), fault_id),
                     cell(sources, fault and fault["metrics"]["fall_rate"], fault_id),
                     cell(sources, clean and clean["metrics"]["lin_vel_rmse_mps"], clean_id),
                     cell(sources, clean and ratio(clean), clean_id)])
    training = []
    for seed in SEEDS:
        fault, fault_id = sources.load(f"bench/runs_p0c/h02_s{seed}.telemetry.json", kind="telemetry")
        clean, clean_id = sources.load(f"bench/runs_p0c/baseline_p0c_s{seed}.telemetry.json", kind="telemetry")
        last = (lambda tele, tag: tele["summary"][tag]["mean_last_20pct"])
        training.append([str(seed),
                         cell(sources, fault and last(fault, "Train/mean_episode_length"), fault_id, 0),
                         cell(sources, fault and last(fault, "Train/mean_reward"), fault_id, 1),
                         cell(sources, clean and last(clean, "Train/mean_reward"), clean_id, 1)])
    media = []
    video = media_video(sources, "docs/media/go2-correction-reward_resume_dev3_20261013")
    media.append({"label": "seed 2027 · 이전 / v3 교정 이어 학습 / 대조", "source": video["id"]})
    v4_videos = []
    for tag in slots["v4_tags"]:
        v4_videos.append(media_video(sources, f"docs/media/go2-correction-{tag}")["id"])
        seed = re.search(r"_s(\d{4})_", tag)
        media.append({"label": f"seed {seed[1] if seed else tag} · 회복 방식 여섯 정책을 같은 Play 조건으로 재생",
                      "source": v4_videos[-1]})
    notes = ["standstill_ratio는 선속도 RMSE를 제자리 정책 RMSE로 나눈 값이다. 1에 가까우면 명령 추종이 없다는 뜻이며 정지 자체를 증명하지 않는다.",
             "이미 본 seed의 합성 결함 개발 사례다. 실제 사용자 실패의 해결을 뜻하지 않는다."]
    if v4_videos and all(sources.ready(source) for source in v4_videos):
        notes.append("v4 영상 4.5초 지점: F1 이전·고쳐서 이어 학습·std 재설정 정책은 출발점 근처에서 몸을 낮춘 채 머물고, "
                     "처음부터 재학습·정상 이어 학습·정상 원본 정책은 걸어서 멀어진다. 재생 한 번의 관찰이며 영상은 "
                     "판정 근거가 아니다. 판정은 고정 평가 수치로 한다.")
    return {
        "id": "failure", "title": "실패 입력: 에피소드는 버티지만 명령을 따르지 않는다",
        "lead": "보상 좌표 결함(F1)을 넣고 300회 학습한 Go2 정책이다. 고정 평가 26조건에서 선속도 오차가 제자리 정책과 같은 수준이다.",
        "blocks": [
            {"type": "table", "wide": True, "title": "고정 평가(26조건, 1040 env, 1000 step)",
             "columns": ["seed", "F1 선속도 RMSE (m/s)", "제자리 RMSE", "standstill_ratio", "F1 낙상률",
                         "정상 RMSE", "정상 ratio"], "rows": rows},
            {"type": "table", "title": "학습 기록 마지막 20% 평균",
             "columns": ["seed", "F1 에피소드 길이", "F1 총보상", "정상 총보상"], "rows": training},
            {"type": "video", "items": media},
        ],
        "notes": notes,
    }


def scene_curves(sources: Sources) -> dict:
    series, ids = {}, []
    for seed in SEEDS:
        for name in (f"h02_s{seed}", f"baseline_p0c_s{seed}"):
            tele, source = sources.load(f"bench/runs_p0c/{name}.telemetry.json", kind="telemetry")
            ids.append(source)
            if tele is not None:
                series[name] = {metric: tele["series"][tag] for metric, (tag, _) in CURVE_METRICS.items()}
    complete = len(series) == 2 * len(SEEDS)
    panels = []
    for metric, (_, title) in CURVE_METRICS.items():
        if not complete:
            panels.append({"title": title, "x0": 19, "envelope": None, "lines": []})
            continue
        clean = [smooth(series[f"baseline_p0c_s{seed}"][metric], R1["window_iterations"])[19:] for seed in SEEDS]
        digits = 1 if metric == "episode_length" else 4
        panels.append({
            "title": title, "x0": 19,
            "envelope": {"label": "정상 3회 관측 범위",
                         "lo": [round(min(values), digits) for values in zip(*clean)],
                         "hi": [round(max(values), digits) for values in zip(*clean)]},
            "lines": [{"label": f"F1 seed {seed}",
                       "values": [round(value, digits) for value in
                                  smooth(series[f"h02_s{seed}"][metric], R1["window_iterations"])[19:]]}
                      for seed in SEEDS],
        })
    return {
        "id": "curves", "title": "학습 곡선: 에피소드 길이만으로는 추종을 알 수 없다",
        "lead": "원본 P0-C 학습 기록의 20회 이동 평균이다. 회색 띠는 정상 3회의 관측 범위, 선은 F1 seed 2026~2028이다.",
        "blocks": [{"type": "chart", "wide": True, "status_sources": ids, "panels": panels}],
        "notes": ["이미 본 seed의 사후 기술이다. 회색 띠는 관측 범위이며 신뢰구간이 아니다.",
                  "학습 기록의 오차는 고정 평가 RMSE와 다른 지표다."],
    }


def scene_warning(sources: Sources) -> dict:
    key, key_id = sources.load("bench/answer_key_p0c.json")
    series, ids = {}, [key_id]
    for case in CASES:
        for seed in SEEDS:
            tele, source = sources.load(f"bench/runs_p0c/{case}_s{seed}.telemetry.json", kind="telemetry")
            ids.append(source)
            if tele is not None:
                series[f"{case}_s{seed}"] = tele["series"][CURVE_METRICS["error_xy"][0]]
    ready = key is not None and len(series) == len(CASES) * len(SEEDS)
    rows = []
    if ready:
        faults = {case: ("NONE" if case == "baseline_p0c" else key["cases"][case]["fault"]) for case in CASES}
        categories = {case: ("정상" if case == "baseline_p0c" else key["cases"][case]["category"]) for case in CASES}
        alerts = r1_alerts(series, faults)
        for case in CASES:
            runs = [alerts[f"{case}_s{seed}"] for seed in SEEDS]
            first = [str(run["alert_iterations"][0]) if run["alert_iterations"] else "-" for run in runs]
            source = sources.file(f"bench/runs_p0c/{case}_s{SEEDS[0]}.telemetry.json", kind="telemetry")["id"]
            rows.append([f"{faults[case]} ({categories[case]})",
                         {"value": f"{sum(bool(run['alert_iterations']) for run in runs)}/3", "source": source, "digits": None},
                         {"value": " / ".join(first), "source": source, "digits": None}])
    return {
        "id": "warning", "title": "단순 경고 R1은 '추종이 무너졌다'까지만 말한다",
        "lead": "R1: 20회 이동 평균 추종 오차가 다른 두 seed 정상 실행 최댓값의 2배를 넘는 상태가 10회 연속이면 경고한다(iteration 60 이후).",
        "blocks": [{"type": "table", "title": "결함 범주별 경고(seed 2026 / 2027 / 2028)", "status_sources": ids,
                    "columns": ["결함 (범주)", "경고한 실행", "첫 경고 iteration"], "rows": rows}],
        "notes": ["R1은 다른 결함에서도 울리고 짧은 timeout 결함(F6)은 놓친다. 원인을 가르는 다음 검사가 필요하다.",
                  "이미 본 seed에 사후로 적용한 후보 규칙이며 사전등록 평가가 아니다."],
    }


def scene_probe(sources: Sources) -> dict:
    fault_rows, other_max, other_ids = [], None, []
    for case in CASES:
        for seed in SEEDS:
            probe, source = sources.load(f"evals/results/p1a_probes_holdout_20261009/probes/{case}_s{seed}.json")
            value = probe and probe["probes"]["P_reward"]["measurement"]["track_lin_vel_xy_exp_rel_error"]
            if case == "h02":
                fault_rows.append([f"F1 seed {seed}", cell(sources, value, source, 3)])
            else:
                other_ids.append(source)
                if value is not None:
                    other_max = value if other_max is None else max(other_max, value)
    all_ready = all(sources.ready(source) for source in other_ids)
    fault_rows.append(["다른 결함·정상 18개 중 최댓값",
                       {"value": other_max if all_ready else None, "source": other_ids[0], "digits": None, "format": "sci"}])
    loop, loop_id = sources.load("evals/results/live_loop_20261009/summary_v2.json")
    loop_rows = [] if loop is None else [
        ["사례", cell(sources, loop["cases"], loop_id, 0)],
        ["잘못된 확정", cell(sources, loop["wrong_confirmations"], loop_id, 0)],
        ["receipt 분류 일치", {"value": f"{loop['receipt_matches']}/{loop['probe_attempts']}", "source": loop_id, "digits": None}],
        ["수용 조건 전체 충족", {"value": "아니오" if loop["accepted"] is not True else "예", "source": loop_id, "digits": None}],
    ]
    order_path = "evals/results/p1a_loop_compare_20261009/loop_compare.jsonl"
    order_id = sources.file(order_path)["id"]
    order_rows = []
    if sources.ready(order_id):
        records = [json.loads(line) for line in (sources.root / order_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        for strategy, title in (("discriminate:llm_agent", "에이전트 가설 순위"), ("discriminate:rules", "규칙 가설 순위"),
                                ("fixed", "고정 순서"), ("exhaustive", "전수 검사")):
            row = [title]
            for seed in SEEDS:
                match = [r for r in records if r["run"] == f"h02_s{seed}" and r["strategy"] == strategy]
                value = None
                if len(match) == 1:
                    found = match[0]
                    state = "보상 확정" if found["status"] == "confirmed" and found["conclusion"] == "reward" else found["status"]
                    value = f"{state} · 검사 {found['probes_used']}개"
                row.append({"value": value, "source": order_id, "digits": None})
            order_rows.append(row)
    return {
        "id": "probe", "title": "다음 검사 하나: 보상 좌표 재계산(P_reward)",
        "lead": "기록된 상태로 선속도 추종 보상을 몸체 좌표와 월드 좌표에서 다시 계산한다. 두 값의 상대 차이가 F1 정책에서만 크게 나타난다.",
        "blocks": [
            {"type": "table", "title": "P_reward 상대 차이(이미 본 21개 실행)", "columns": ["실행", "상대 차이"],
             "rows": fault_rows},
            {"type": "table", "title": "F1 확정까지 쓴 검사 수(이미 본 3개 실행)", "status_sources": [order_id],
             "columns": ["검사 순서", "seed 2026", "seed 2027", "seed 2028"], "rows": order_rows},
            {"type": "table", "title": "이미 본 8개 사례의 고리 실행 기록(캐시된 순위, 새 모델 호출 없음)",
             "columns": ["항목", "값"], "rows": loop_rows, "status_sources": [loop_id]},
        ],
        "notes": ["probe 작성자와 결함 작성자가 같아 고리 동작 확인으로만 쓴다. F1에서는 네 검사 순서 모두 보상 결함으로 확정했고 차이는 검사 수뿐이다.",
                  "수용 조건이 모두 충족되지 않은 기록은 그대로 표시한다."],
    }


def scene_correction(sources: Sources, slots: dict) -> dict:
    protocol, protocol_id = sources.load("bench/protocols/training_correction_v4.json")
    targets = [] if protocol is None else [
        [name, {"value": f"{row['module']}.{row['qualname']}", "source": protocol_id, "digits": None}]
        for name, row in sorted(protocol["correction_source_contract"]["targets"].items())]
    rule = [] if protocol is None else [
        ["회복 방식 규칙", {"value": protocol["recovery_mode_rule"]["rule"], "source": protocol_id, "digits": None}],
        ["규칙 근거", {"value": protocol["recovery_mode_rule"]["derived_from"], "source": protocol_id, "digits": None}],
        ["std 재설정 값", cell(sources, protocol["resume_inputs"]["init_noise_std"], protocol_id, 1)],
    ]
    import replay_demo_slots as DS
    return {
        "id": "correction", "title": "교정 제안: 등록된 교정 하나만 바꾼다", "dense": True,
        "lead": "결함 범주마다 미리 등록한 교정 중 하나를 고르고 다른 표적은 바꾸지 않는다. 회복 방식은 실행 전에 등록한 규칙으로 권고한다.",
        "blocks": [
            {"type": "table", "title": "등록 교정과 대상 함수", "columns": ["교정", "대상"], "rows": targets,
             "status_sources": [protocol_id]},
            {"type": "table", "title": "사전등록 규칙(v4 프로토콜)", "columns": ["항목", "내용"], "rows": rule,
             "status_sources": [protocol_id]},
            *DS.correction_proposal_blocks(sources, slots.get("correction_proposal"), wide=False),
            *DS.correction_proposal_blocks(sources, slots.get("motor_correction_proposal"), kind="motor_correction_proposal",
                                           label=DS.MOTOR_PROPOSAL_LABEL, wide=False),
        ],
        "notes": ["규칙은 seed 2027 단일 관찰에서 만들었고 v4 결과로 적중 여부만 기록한다."],
    }


def scene_demo(sources: Sources, slots: dict) -> dict:
    """보상(seed 2030)·모터(seed 2033) 종단 시연을 나란히 둔다. 결과가 승인되지 않은 시연은 자리 표시로 남는다."""
    import replay_demo_slots as DS
    reward = DS.end_to_end_blocks(sources, slots.get("end_to_end"))
    motor = DS.end_to_end_blocks(sources, slots.get("motor_end_to_end"), kind="motor_end_to_end",
                                 label=DS.MOTOR_END_TO_END_LABEL)
    return {
        "id": "demo", "title": "종단 시연: 경고부터 교정 판정까지", "dense": True, "scope": SCOPE_NEW_SEED_DEMO,
        "lead": "새 seed에서 새 Nemotron 판단, 승인된 검사, 교정 제안 승인, Isaac 재학습, 고정 평가 판정을 한 번에 이었다. "
                "왼쪽은 보상 결함, 오른쪽은 모터 결함이다.",
        "blocks": [*reward, *motor],  # 비용은 마지막 장면의 종단 시연 비용 표에 둔다
        "notes": ["결함과 probe는 같은 설계자가 만들었고 이미 본 결함 종류의 새 seed 한 건씩이다. "
                  "진단 정확도·일반화·교정 성공률·GPU 절감의 근거가 아니다.",
                  "승인은 사용자가 위임한 운영자 승인이며 승인자 신원은 인증하지 않았다."],
    }


def scene_nat_demo(sources: Sources, slots: dict) -> dict:
    """NAT 결정 경로 시연: 단계 표와 NeMo Agent Toolkit 호출 기록을 나란히 둔다."""
    import replay_demo_slots as DS
    relative = slots.get("nat_end_to_end")
    return {
        "id": "nat_demo", "title": "NAT 결정 경로 시연: 진단과 교정 제안을 도구 호출로 제출", "dense": True,
        "scope": SCOPE_NEW_SEED_DEMO,
        "lead": "진단 결정과 교정 제안을 NeMo Agent Toolkit 도구 호출 워크플로가 제출했다. 승인·검사·재학습·고정 평가는 앞 시연과 같다.",
        "blocks": [*DS.end_to_end_blocks(sources, relative, kind="nat_end_to_end", label=DS.NAT_END_TO_END_LABEL),
                   *DS.nat_record_blocks(sources, relative)],
        "notes": ["도구 호출 순서와 횟수는 공개 결과에 없어 설정된 도구 목록만 보인다. 설정은 시연 프로토콜이 SHA로 묶은 파일이다.",
                  "NAT 경로 비교 요약의 채택 판정에 따라 쓴 경로이며 단일 호출과의 우열을 보여 주지 않는다. "
                  "이미 본 결함 종류의 새 seed 한 건이다."],
    }


def scene_recovery(sources: Sources, slots: dict) -> dict:
    import publish_training_correction as Q3
    import publish_training_correction_v4 as Q4
    v3_row, v3 = sources.bundle("evals/results/reward_resume_dev3_20261013", Q3.replay, kind="public_bundle_v3")
    v3_rows = []
    for arm, label in (("clean_reference", "정상 기준 이어 학습"), ("fault_sham", "잘못된 교정(대조)"),
                       ("corrected", "등록 교정 이어 학습")):
        v3_rows.append([label, cell(sources, v3 and v3["behavior"][arm]["err_xy"], v3_row["id"])])
    v3_rows.append(["판정", {"value": v3 and v3["status"], "source": v3_row["id"], "digits": None}])
    labels = RECOVERY_LABELS
    arms = (("resume_corrected", "고쳐서 이어 학습"), ("resume_corrected_std_reset", "이어 학습 + std 재설정"),
            ("restart_corrected", "고쳐서 처음부터 재학습"), ("clean_resume_reference", "정상 기준(이어 학습)"))
    v4_rows = [[title] for _, title in arms] + [["원본 F1 · 규칙 권고와 결과"], ["재학습 체크포인트 = 원본 정상 체크포인트"]]
    protocol_v4, _ = sources.load("bench/protocols/training_correction_v4.json")
    columns, ready = ["방식"], []
    for tag in slots["v4_tags"]:
        row, replay = sources.bundle(f"evals/results/{tag}", Q4.replay, kind="public_bundle_v4")
        if replay is not None:
            ready.append(row["id"])
        seed = re.search(r"_s(\d{4})_", tag)
        seed = seed[1] if seed else tag
        columns += [f"{seed} 판정", f"{seed} RMSE (m/s)", f"{seed} ratio"]
        signal = replay["mechanical_signal"] if replay else {}
        for index, (arm, _) in enumerate(arms):
            behavior = replay["behavior"].get(arm) if replay else None
            label = (labels.get(replay["labels"][arm]) if arm in replay["labels"] else "기준") if replay else None
            v4_rows[index] += [{"value": label, "source": row["id"], "digits": None},
                               cell(sources, behavior and behavior["err_xy"], row["id"]),
                               cell(sources, signal.get(arm), row["id"])]
        rule = replay["recovery_mode_rule"] if replay else {}
        v4_rows[-2] += [{"value": f"{rule.get('recommended')} · {rule.get('outcome')}" if replay else None,
                         "source": row["id"], "digits": None},
                        "-", cell(sources, signal.get("original_fault"), row["id"])]
        same = None
        if replay and protocol_v4 and seed in protocol_v4["resume_inputs"]["seeds"]:
            evidence = _read(sources.root / f"evals/results/{tag}/training_evidence.json")["arms"]
            restart = evidence.get("restart_corrected", {}).get("checkpoint", {}).get("sha256")
            original = protocol_v4["resume_inputs"]["seeds"][seed]["runs"]["NONE"]["hashes"]["checkpoint"]
            same = None if restart is None else ("같음" if restart == original else "다름")
        v4_rows[-1] += [{"value": same, "source": row["id"], "digits": None}, "-", "-"]
    blocks = [
        {"type": "table", "wide": True, "title": "v4 seed 2026·2028: 같은 300회 예산의 회복 방식",
         "columns": columns, "rows": v4_rows,
         "status_sources": [sources.rows[f"evals/results/{tag}"]["id"] for tag in slots["v4_tags"]]},
        {"type": "table", "title": "v3 seed 2027: 고쳐서 이어 학습 300회", "columns": ["조건", "선속도 RMSE (m/s)"],
         "rows": v3_rows, "status_sources": [v3_row["id"]]},
    ]
    if len(ready) == len(slots["v4_tags"]) == 2:
        summary = Q4.aggregate([sources.root / tag for tag in ready])
        blocks.append({"type": "table", "title": "두 seed 요약(사전 예측 대비)", "columns": ["방식", "회복 seed 수", "예측", "예측 유지"],
                       "rows": [[arm, {"value": f"{row['recovered']}/{len(row['labels'])}", "source": ready[0], "digits": None},
                                 {"value": row["prediction"] or "예측 없음", "source": ready[0], "digits": None},
                                 {"value": {True: "예", False: "아니오", None: "-"}[row["prediction_held_all_seeds"]],
                                  "source": ready[0], "digits": None}] for arm, row in summary["arms"].items()]})
    return {
        "id": "recovery", "title": "회복 방식 비교: 고쳐서 이어 학습할까, 처음부터 다시 할까", "dense": True,
        "lead": "같은 300회 예산으로 이어 학습, 탐색 재설정 이어 학습, 처음부터 재학습을 비교한다. 이어 학습 계열은 이어 학습한 정상 정책, 재학습은 원본 정상 정책과 비교한다.",
        "blocks": blocks,
        "notes": ["seed마다 단일 실행이다. 이미 본 seed의 합성 결함 개발 결과이며 일반화나 AI 교정 선택을 입증하지 않는다.",
                  "재학습은 교정한 코드가 정상 학습과 같아져 같은 seed의 정상 정책을 다시 만든다. 체크포인트가 같으면 회복은 예상된 확인이며 새로운 회복 효과가 아니다."],
    }


def scene_motor(sources: Sources, slots: dict) -> dict:
    """모터 결함(F2) 고쳐서 이어 학습의 seed별 판정과 중간 저장본 예산 반응. 공개 묶음이 없으면 칸을 비운다."""
    import run_training_correction_motor as W
    tags = slots["motor_tags"]
    iterations = [W.RESUME_ITERATION, *W.BUDGET_CHECKPOINTS, W.FINAL_ITERATION]
    judged, budget, ready = [], [], []
    for tag in tags:
        row, replay = sources.bundle(f"evals/results/{tag}", W.replay, kind="public_bundle_motor")
        seed = re.search(r"_s(\d{4})_", tag)
        seed = seed[1] if seed else tag
        if replay is None:
            judged_values, budget_values = [None] * 5, [None] * (len(iterations) + 3)
        else:
            ready.append(row["id"])
            treated = replay["behavior"].get("resume_corrected")
            reference = replay["behavior"].get("clean_resume_reference")
            curve, rule = replay["budget_curve"], replay["recovery_mode_rule"]
            complete = curve["status"] == "complete"
            if complete and [point["iteration"] for point in curve["points"]] != iterations:
                raise ValueError(f"{tag}: budget curve 점이 화면 열({iterations})과 다르다")
            # 확인된 묶음 안에서 비어 있는 값은 "-"로 둔다. None은 화면에서 근거 미확인 표시로 읽힌다.
            judged_values = [RECOVERY_LABELS.get(replay["labels"]["resume_corrected"], "-"),
                             treated["err_xy"] if treated else "-", reference["err_xy"] if reference else "-",
                             treated["err_xy"] / reference["err_xy"] if treated and reference and reference["err_xy"]
                             else "-",
                             f"{rule['recommended']} · {rule['outcome']}"]
            budget_values = [MOTOR_CURVE_STATES.get(curve["status"], curve["status"]),
                             *([point["err_xy_ratio"] for point in curve["points"]] if complete
                               else ["-"] * len(iterations)),
                             (curve.get("first_healthy_iteration") or "없음") if complete else "-",
                             {True: "예", False: "아니오"}.get(curve.get("budget_limited_candidate"), "-")]
        judged.append([seed, *[_motor_cell(sources, value, row["id"]) for value in judged_values]])
        budget.append([seed, *[_motor_cell(sources, value, row["id"]) for value in budget_values]])
    notes = ["seed마다 단일 실행이다. 이미 본 seed의 합성 결함 개발 결과이며 일반화나 AI 교정 선택을 입증하지 않는다."]
    if ready:
        notes.append(W.FIXED["result_wording"]["budget_curve"])
    if len(ready) == len(tags) == 2:
        summary = W.aggregate_public([sources.root / source for source in ready])
        update = summary["rule_update"]
        held = {True: "예", False: "아니오", None: "-"}[summary["prediction_held_all_seeds"]]
        judged.append(["두 seed",
                       _motor_cell(sources, f"회복 {summary['recovered']}/{len(summary['labels'])} · 예측 유지 {held}", ready[0]),
                       "-", "-", "-", _motor_cell(sources, f"규칙 갱신 {update['decision']}", ready[0])])
        notes.append(update["wording"])
    status_sources = [sources.rows[f"evals/results/{tag}"]["id"] for tag in tags]
    return {
        "id": "motor", "title": "모터 결함: 고쳐서 이어 학습과 예산 반응",
        "lead": ("모터 곡선 결함(F2)을 고친 뒤 정상 이어 학습과 같은 300회를 이어 학습한다. 실행 전 예측은 두 seed 모두 미회복이다. "
                 "중간 저장본(+51·+151·+251회)은 학습 없이 다시 평가해 같은 iteration의 정상 이어 학습과 비교한다."),
        "blocks": [
            {"type": "table", "wide": True, "title": "seed별 판정(최종 +299회, 같은 seed 정상 이어 학습 대비)",
             "columns": ["seed", "판정", "고쳐서 이어 학습 RMSE (m/s)", "정상 이어 학습 RMSE (m/s)", "오차 비(고쳐서 / 정상)",
                         "규칙 권고 · 결과"],
             "rows": judged, "status_sources": status_sources},
            {"type": "table", "wide": True, "title": "예산 반응(보조 기록): 같은 iteration 정상 이어 학습 대비 선속도 오차 비",
             "columns": ["seed", "기록", *[f"+{k - W.RESUME_ITERATION}" for k in iterations],
                         "처음 healthy (iteration 번호)", "budget_limited_candidate"],
             "rows": budget, "status_sources": status_sources},
        ],
        "notes": notes,
    }


def _motor_cell(sources: Sources, value: Any, source: str) -> dict:
    return cell(sources, value, source, 3 if isinstance(value, float) else 0 if isinstance(value, int) else None)


def scene_hold(sources: Sources) -> dict:
    t1, t1_id = sources.load("evals/results/t1_loop_compare_20261009/loop_compare.json")
    names = {"exhaustive": "전수 검사", "fixed": "고정 순서", "random": "무작위 순서",
             "discriminate:agent": "에이전트 순위 가르기", "discriminate:rules": "규칙 순위 가르기"}
    t1_rows = [] if t1 is None else [
        [names.get(name, name), cell(sources, row["cells"], t1_id, 0), cell(sources, row["correct"], t1_id, 0),
         cell(sources, row["wrong"], t1_id, 0), cell(sources, row["none_supported"], t1_id, 0)]
        for name, row in t1["table"].items()]
    t2, t2_id = sources.load("evals/results/t2_gate_20261009/summary_v2.json")
    t2_rows = [] if t2 is None else [
        [row["case"], {"value": row["behavior_label"], "source": t2_id, "digits": None},
         {"value": "열지 않음" if row["loop_opened"] is False else "열림", "source": t2_id, "digits": None},
         cell(sources, row["interventions"], t2_id, 0)] for row in t2["rows"]]
    return {
        "id": "hold", "title": "멈추는 경우: 지원 밖이면 확정하지 않는다", "scope": SCOPE_HOLD,
        "lead": "외부 원인 결함(T1)은 등록 메커니즘 밖이라 확정 대신 지원 밖으로 끝나야 하고, 정상 입력(T2)에서는 고리를 열지 않아야 한다.",
        "blocks": [
            {"type": "table", "wide": True, "title": "T1 외부 원인 결함(사례×방식)", "status_sources": [t1_id],
             "columns": ["방식", "사례", "맞는 확정", "잘못된 확정", "지원 밖 판정"], "rows": t1_rows},
            {"type": "table", "title": "T2 정상 입력", "status_sources": [t2_id],
             "columns": ["사례", "행동 라벨", "고리", "개입"], "rows": t2_rows},
        ],
        "notes": ["두 시험 모두 사전등록 뒤 실행한 개발 사례다. 표본이 작아 안전성 보장으로 말하지 않는다."],
    }


UPSTREAM_SMOKE_PROTOCOL = "bench/protocols/upstream_u1_smoke_v1.json"
UPSTREAM_SMOKE_RESULT = "evals/results/upstream_u1_smoke_20261013/result.json"
UPSTREAM_DEMO_PROTOCOL = "bench/protocols/e2e_demo_u1_v1.json"
# 운영 기준의 정의. 근거는 시연 프로토콜 scope_gate.enforce_outcome(강제 모드: 범위가 충분하지 않으면 재학습 전에 종료)이다
UPSTREAM_OPERATING_NOTE = ("운영 기준은 강제 모드다(시연 프로토콜 scope_gate.enforce_outcome: 범위가 충분하지 않으면 재학습 전에 종료). "
                           "이 시연의 재학습은 사전등록한 대조 모드의 예외다.")


def _short_ref(url: str | None, pattern: str) -> str | None:
    """원전 URL에서 커밋 앞 7자리나 PR 번호만 꺼낸다. 화면에는 URL을 싣지 않는다."""
    match = re.search(pattern, url or "")
    return match[1] if match else None


def scene_upstream(sources: Sources) -> dict:
    """설계자 밖 결함 U1: 원전 출처와 이식, 등록 교정이 덮는 범위, smoke 결과."""
    smoke_protocol, protocol_id = sources.load(UPSTREAM_SMOKE_PROTOCOL)
    demo_protocol, demo_protocol_id = sources.load(UPSTREAM_DEMO_PROTOCOL)
    smoke, smoke_id = sources.load(UPSTREAM_SMOKE_RESULT)
    origin = []
    if smoke_protocol is not None:
        source = smoke_protocol["source"]
        commit = _short_ref(source["workaround_commit"]["url"], r"/commit/([0-9a-f]{7})")
        pr = _short_ref(source["fix_pr"]["url"], r"/pull/(\d+)")
        origin = [
            ["원전 프레임워크", cell(sources, source["framework"], protocol_id, None)],
            ["원전 수정", cell(sources, f"우회 커밋 {commit} · 백엔드 수정 PR #{pr}({source['fix_pr']['merged']} 병합)",
                           protocol_id, None)],
            ["수정 커밋 메시지 발췌", cell(sources, source["fix_pr"]["excerpts"][0], protocol_id, None)],
            ["Isaac Lab으로 옮긴 보상 항(가중치)", cell(sources, " · ".join(
                f"{row['isaaclab_function']}({row['weight']})" for row in smoke_protocol["mapping"]), protocol_id, None)],
        ]
    if demo_protocol is not None:
        scope = demo_protocol["registered_correction_scope"]
        origin += [
            ["등록 교정이 되돌리는 함수", cell(sources, f"{scope['correction_id']} → {scope['target']}(네 함수 중 하나)",
                                       demo_protocol_id, None)],
            ["코드 지문으로 보이는가", cell(sources, demo_protocol["b_code_wording"], demo_protocol_id, None)],
        ]
    smoke_rows = []
    if smoke is not None:
        swapped = smoke["readback"]["swapped"]
        smoke_rows = [
            ["결과 · 사전 예측 일치", cell(sources, f"{smoke['outcome']} · {'예' if smoke['prediction_match'] else '아니오'}",
                                       smoke_id, None)],
            ["고정 평가 판정(같은 seed 정상 대비)", cell(sources, f"{smoke['verdict']['label']} · 실패 검사 "
                                                         f"{', '.join(smoke['verdict']['failed']) or '없음'}", smoke_id, None)],
            ["standstill ratio", cell(sources, smoke["standstill_ratio"], smoke_id, 3)],
            ["주입 확인(교체된 함수)", cell(sources, f"{sum(swapped.values())}/{len(swapped)} · 확인 줄 "
                                               f"{'ok' if smoke['readback']['ok'] else '실패'}", smoke_id, None)],
            ["정상 학습과의 params 차이", cell(sources, ", ".join(smoke["params_differences"]) or "없음", smoke_id, None)],
            ["학습 parent wall(s)", cell(sources, smoke["parent_wall_s"]["train_s"], smoke_id, 0)],
        ]
    return {
        "id": "upstream", "title": "설계자 밖 결함: 다른 프레임워크에서 원인이 확정된 결함을 옮겼다", "dense": True,
        "scope": SCOPE_UPSTREAM_SMOKE,
        "lead": ("UniLab(MotrixSim·MuJoCo 백엔드)에서 원인이 확정된 결함(보상 4항이 몸체 속도 0을 읽음)의 의미를 Isaac Lab "
                 "Go2 flat에 옮겼다. 등록 교정은 네 함수 중 하나만 되돌린다."),
        "blocks": [
            {"type": "table", "title": "원전과 이식", "columns": ["항목", "내용"], "rows": origin,
             "status_sources": [protocol_id, demo_protocol_id]},
            {"type": "table", "title": f"U1 smoke(seed {smoke['seed'] if smoke else '-'}, 4096 env × 300회)",
             "columns": ["항목", "기록"], "rows": smoke_rows, "status_sources": [smoke_id]},
        ],
        "notes": ["결함 의미는 외부 공개 수정에서 왔지만 이식·주입 방식·검사·범위 점검 규칙은 결함을 안 뒤에 만들었다. "
                  "설계자 결합을 줄일 뿐 없애지 못한다."],
    }


def scene_upstream_demo(sources: Sources, slots: dict) -> dict:
    """U1 종단 시연 단계와 교정 범위 점검·재학습 대조. 시연 결과가 승인되지 않았으면 자리 표시로 남는다."""
    import replay_demo_slots as DS
    relative = slots.get("upstream_end_to_end")
    return {
        "id": "upstream_demo", "title": "U1 시연: 재학습 전에 교정 범위를 점검하고 실측과 대조한다", "dense": True,
        "scope": SCOPE_UPSTREAM_DEMO,
        "lead": "확정 검사의 항별 오차로 교정 범위를 재학습 전에 점검하고, 재학습을 한 번 실행해 실측과 대조한다.",
        "blocks": [
            *DS.end_to_end_blocks(sources, relative, kind="upstream_end_to_end", label=DS.UPSTREAM_END_TO_END_LABEL),
            *DS.scope_blocks(sources, relative),
        ],
        "notes": ["seed 하나의 단일 실행이며 진단 정확도·일반화·교정 성공률의 근거가 아니다. 승인은 위임된 운영자 승인이며 "
                  "승인자 신원은 인증하지 않았다.",
                  UPSTREAM_OPERATING_NOTE],
    }


def scene_nvidia(sources: Sources, slots: dict) -> dict:
    import nat_eval_rescore as N
    import replay_demo_slots as DS
    roles = [
        ("Nemotron (nvidia/nemotron-3-super-120b-a12b)", "가설 순위와 다음 검사 제안", "모델 호출 기록",
         "evals/results/t1_agent_20261009/rankings.json"),
        ("Isaac Lab 2.1.1 · Isaac Sim 4.5", "검사 실행 · 재학습 · 고정 평가", "실행 기록",
         "evals/results/p1a_probes_holdout_20261009/probes/h02_s2026.json"),
        ("OpenShell", "분석 코드의 파일·네트워크 권한 경계", "샌드박스 검사 기록",
         "evals/results/sandbox_kernel_tests.txt"),
        ("애플리케이션 원장", "승인 · 1회 소비 · receipt", "실행 기록",
         "evals/results/live_loop_20261009/summary_v2.json"),
    ]
    rows = [[name, job, kind, {"value": "근거 파일", "source": sources.file(path)["id"], "digits": None}]
            for name, job, kind, path in roles]
    rows.insert(1, ["NeMo Agent Toolkit", "시연에서 진단 결정·교정 제안을 도구 호출로 제출", "시연 결정 기록",
                    DS.nat_role_cell(sources, slots.get("nat_end_to_end"))])
    # 재채점 행은 공개 요약이 커밋되고 공개 파일만으로 replay를 통과할 때만 값을 보인다.
    rescore_row, rescore = sources.bundle(slots["nat_rescore"], N.replay_public, kind="public_nat_rescore")
    rows.insert(2, ["NeMo Agent Toolkit eval (nat eval --skip_workflow)", "저장된 결정 기록을 모델 호출 없이 같은 검증기로 다시 채점",
                    "재채점 대조 기록",
                    {"value": rescore and f"{rescore['items']}항목 · 판정 차이 0", "source": rescore_row["id"], "digits": None}])
    rows.insert(3, ["NeMo Agent Toolkit HITL 미들웨어(선택)", "승인 전 요청을 보여 주고 운영자 확인을 받음. approve만 원장 승인 한 줄",
                    "설정·테스트", cell(sources, "근거 파일", sources.file("configs/nat_hitl_ledger_example.yml")["id"], None)])
    skill = [
        ["Reward increases are not proof that the task behavior improved", "보상 크기 대신 고정 평가 26조건의 행동 지표로 판정"],
        ["Map every reward term to a named task phase and check consistent geometry", "P_reward: 추종 보상을 몸체·월드 좌표에서 다시 계산"],
        ["Classify the failed layer (environment, reward, reset, physics, runner, checkpoint)", "가설 순위를 메커니즘 범주로 기록하고 지원 밖이면 HOLD"],
        ["Any recommended next run changes only one variable", "등록 교정 하나만 복원하고 회복 방식만 바꿔 비교"],
    ]
    return {
        "id": "nvidia", "title": "NVIDIA 기술의 역할", "dense": True,
        "lead": "각 구성이 맡는 일과 그 근거 파일의 종류를 구분한다. 승인과 1회 소비는 OpenShell이 아니라 애플리케이션 원장이 맡는다.",
        "blocks": [
            {"type": "table", "wide": True, "title": "구성별 역할", "columns": ["구성", "하는 일", "근거 종류", "근거"], "rows": rows},
            {"type": "table", "wide": True, "title": "Isaac Lab 공식 스킬 isaaclab-debugging-rl-training(experimental) 항목과의 대응",
             "columns": ["공식 스킬 항목(요지)", "이 고리의 대응"], "rows": skill},
        ],
        "notes": ["대응표는 절차 비교이며 실행 중 그 스킬을 호출했다는 뜻이 아니다.",
                  "NeMo Agent Toolkit 행은 NAT 결정 경로 시연 한 건의 기록이다. HITL 행의 근거는 설정·테스트이며 실행 기록이 아니다. "
                  "HITL 확인은 선택 기능이고 기록된 시연에는 쓰지 않았다.",
                  "재채점은 저장된 출력이 공개 요약과 같은 판정을 내는지 확인한 것이다. 결정의 정확도나 단일 호출과의 우열을 보여 주지 않으며 모델을 다시 부르지 않는다."],
    }


def scene_cost(sources: Sources, slots: dict) -> dict:
    import publish_training_correction as Q3
    import publish_training_correction_v4 as Q4
    rows = []
    for tag, label in (("reward_resume_dev3_20261013", "v3 보상 사례(3 arm)"), ("motor_resume_dev_20261013", "v3 모터 사례(3 arm)"),
                       ("timeout_resume_dev_20261013", "v3 timeout 사례(3 arm)")):
        row, replay = sources.bundle(f"evals/results/{tag}", Q3.replay, kind="public_bundle_v3")
        rows.append([label, cell(sources, replay and replay["total_parent_wall_s"], row["id"], 0),
                     "학습·고정 평가 parent wall"])
    for tag in slots["v4_tags"]:
        row, replay = sources.bundle(f"evals/results/{tag}", Q4.replay, kind="public_bundle_v4")
        rows.append([f"v4 {tag}(4 arm)", cell(sources, replay and replay["total_parent_wall_s"], row["id"], 0),
                     "학습·고정 평가 parent wall"])
    t1, t1_id = sources.load("evals/results/t1_loop_compare_20261009/loop_compare.json")
    rows.append(["T1 전수 검사 probe", cell(sources, t1 and t1["table"]["exhaustive"]["gpu_s_measured"], t1_id, 1),
                 "probe 실행 시간 합(각 probe elapsed_s)"])
    t2, t2_id = sources.load("evals/results/t2_gate_20261009/summary_v2.json")
    rows.append(["T2 정상 입력 관문", cell(sources, t2 and t2["costs"]["cpu_s"], t2_id, 3), "관문 Python CPU"])
    commands = [
        r".\.venv\Scripts\python.exe evals/publish_training_correction.py replay evals/results/reward_resume_dev3_20261013",
        *[rf".\.venv\Scripts\python.exe evals/publish_training_correction_v4.py replay evals/results/{tag}" for tag in slots["v4_tags"]],
        r".\.venv\Scripts\python.exe evals/build_replay_page.py",
    ]
    import replay_demo_slots as DS
    demos = DS.cost_block(sources, [(slots.get("end_to_end"), "end_to_end", "보상 결함"),
                                    (slots.get("motor_end_to_end"), "motor_end_to_end", "모터 결함"),
                                    (slots.get("nat_end_to_end"), "nat_end_to_end", "보상 결함(NAT 경로)"),
                                    (slots.get("upstream_end_to_end"), "upstream_end_to_end", "설계자 밖 결함 U1")])
    return {
        "id": "cost", "title": "비용과 재현", "dense": True,
        "lead": "숫자는 공개 결과 파일에서 다시 계산한다. 재현 명령은 GPU·모델·네트워크 없이 공개 묶음의 해시와 판정을 재생한다.",
        "blocks": [
            {"type": "table", "title": "실행 비용(초)", "columns": ["실행", "값(s)", "범위"], "rows": rows},
            {"type": "commands", "title": "재현 명령(저장소 루트에서)", "lines": commands},
            demos,
        ],
        "notes": ["parent wall time은 부모 프로세스 실행부터 종료 확인까지이며 GPU 커널 시간이나 다른 방법 대비 절감량이 아니다.",
                  "replay는 공개 파일의 수치를 다시 계산할 뿐 실행 프로세스의 신원을 재인증하지 않는다."],
    }


def build(root: Path = ROOT) -> dict:
    import replay_d3
    sources = Sources(root)
    slots = _read(root / SLOTS)
    scenes = [scene_failure(sources, slots), scene_curves(sources), scene_warning(sources),
              scene_probe(sources), scene_correction(sources, slots), scene_demo(sources, slots),
              scene_nat_demo(sources, slots), scene_recovery(sources, slots),
              scene_motor(sources, slots), scene_hold(sources), scene_upstream(sources), scene_upstream_demo(sources, slots),
              replay_d3.scene_d3(sources, slots), scene_nvidia(sources, slots), scene_cost(sources, slots)]
    for index, scene in enumerate(scenes, start=1):
        scene["number"] = index
    data = {"schema": SCHEMA,
            "scope": SCOPE_COMMON,
            "status_labels": {"ready": "확인", "waiting": "대기", "mismatch": "불일치"},
            "scenes": scenes, "sources": sources.rows}
    check_public(data)
    return data


def text_fields(value: Any, key: str | None = None):
    """화면에 보이는 문장만 꺼낸다. 경로·해시·근거 id는 문장 검사에서 뺀다."""
    if key in {"path", "sha256", "source", "id", "status_sources", "reason", "kind", "lines"}:
        return
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from text_fields(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from text_fields(child, key)
    elif isinstance(value, str):
        yield value


def claim_violations(text: str) -> list[str]:
    problems = [word for word in FORBIDDEN_ALWAYS if word in text]
    for sentence in re.split(r"[.\n。!?]", text):
        if any(word in sentence for word in NEEDS_NEGATION) and not any(word in sentence for word in NEGATIONS):
            problems.append(sentence.strip())
        if re.search(r"GPU\s*\d+(\.\d+)?\s*배", sentence):
            problems.append(sentence.strip())
    return problems


def path_violations(text: str) -> list[str]:
    return [pattern.pattern for pattern in PRIVATE_PATTERNS if pattern.search(text)]


def check_public(data: dict) -> None:
    problems = [problem for text in text_fields(data) for problem in claim_violations(text)]
    serialized = json.dumps(data, ensure_ascii=False)
    problems += path_violations(serialized)
    # 상태 표시 단어는 근거 상태 칸에만 나와야 한다. 장면 문구에 섞이면 "근거가 대기 중"으로 읽힌다.
    problems += [f"status word in scene text: {text[:40]}" for scene in data.get("scenes", [])
                 for text in text_fields(scene) if any(word in text for word in STATUS_WORDS)]
    if any(row["status"] not in STATUS for row in data["sources"].values()):
        problems.append("unknown source status")
    if problems:
        raise ValueError(f"replay page content is not publishable: {problems[:5]}")


def render(data: dict) -> str:
    return "window.REPLAY_DATA = " + json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False) + ";\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="파일을 쓰지 않고 현재 data.js와 같은지만 확인한다")
    args = parser.parse_args(argv)
    content = render(build(ROOT))
    output = ROOT / OUTPUT
    if args.check:
        same = output.is_file() and output.read_text(encoding="utf-8") == content
        print(json.dumps({"status": "current" if same else "stale"}, ensure_ascii=False))
        return 0 if same else 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8", newline="\n")
    counts = {}
    for row in json.loads(content[len("window.REPLAY_DATA = "):-2])["sources"].values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"status": "written", "path": OUTPUT.as_posix(), "sources": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
