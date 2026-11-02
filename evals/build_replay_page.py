"""발표 재생 화면 데이터(docs/replay/data.js)를 승인된 공개 결과에서 만든다.

승인은 저장소에 커밋돼 있고 작업 트리가 그 커밋과 같은 파일을 뜻한다. 공개 묶음은 각 publisher의
replay까지 통과해야 한다. 파일이 없거나 커밋 전이면 대기, 수정됐거나 해시·replay가 맞지 않으면
불일치로 표시하고 그 숫자를 화면에 넣지 않는다. GPU·모델·네트워크를 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

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
SCOPE_SOURCE_COLLECTION = ("커밋된 원천 학습·고정 평가의 수집 결과다. "
                           "모델의 진단·영상 기여·교정·회복 결과로 합산하지 않는다.")
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


def manifest_result(sources: Sources, relative: str, *, manifest_name: str,
                    manifest_schema: str, summary_schema: str, kind: str,
                    summary_name: str = "summary.json") -> tuple[dict, dict | None]:
    """manifest가 선언한 공개 파일 전체가 커밋·SHA 일치할 때만 summary를 반환한다."""
    if relative in sources.replays:
        return sources.rows[relative], sources.replays[relative]
    sources.replays[relative] = None
    manifest_relative = f"{relative}/{manifest_name}"
    row = {"id": relative, "path": manifest_relative, "kind": kind, "sha256": None,
           "status": "waiting", "reason": "missing"}
    sources.rows[relative] = row
    manifest_path = sources.root / manifest_relative
    if not manifest_path.is_file():
        return row, None
    row["sha256"] = _sha(manifest_path)
    if not _git(sources.root, "ls-files", "--", relative).stdout.strip():
        row["reason"] = "not_committed"
        return row, None
    if _git(sources.root, "status", "--porcelain", "--", relative).stdout.strip():
        row["status"], row["reason"] = "mismatch", "modified_since_commit"
        return row, None
    try:
        manifest = _read(manifest_path)
        if manifest.get("schema") != manifest_schema:
            raise ValueError("public manifest schema/files 불일치")
        declared = manifest["files"]
        if isinstance(declared, list):
            if any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in declared):
                raise ValueError("public manifest list entry 불일치")
            mapped = {item["path"]: item["sha256"] for item in declared}
            if len(mapped) != len(declared):
                raise ValueError("public manifest 중복 경로")
            declared = mapped
        if not isinstance(declared, dict):
            raise TypeError("public manifest files 불일치")
        expected_files = {manifest_name, *declared}
        actual_files = {path.relative_to(sources.root / relative).as_posix()
                        for path in (sources.root / relative).rglob("*") if path.is_file()}
        if actual_files != expected_files:
            raise ValueError("public manifest inventory 불일치")
        for name, entry in declared.items():
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != name:
                raise ValueError("public manifest 상대 경로 불일치")
            expected = entry if isinstance(entry, str) else entry.get("sha256") if isinstance(entry, dict) else None
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("public manifest SHA 형식 불일치")
            source_row = sources.file(f"{relative}/{name}", kind=f"{kind}_file",
                                      expected_sha256=expected)
            if source_row["status"] != "ready":
                raise ValueError("public manifest 파일 승인 불일치")
        if summary_name not in declared:
            raise ValueError("public summary가 manifest에 없다")
        summary = _read(sources.root / relative / summary_name)
        if summary.get("schema") != summary_schema:
            raise ValueError("public summary schema 불일치")
    except (OSError, ValueError, TypeError, KeyError) as error:
        row["status"], row["reason"] = "mismatch", f"manifest_failed:{type(error).__name__}"
        return row, None
    row["status"], row["reason"] = "ready", None
    sources.replays[relative] = summary
    return row, summary


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
    feedback = DS.feedback_demo_note(sources, slots.get("motor_feedback_end_to_end"), slots.get("motor_end_to_end"))
    return {
        "id": "demo", "title": "종단 시연: 경고부터 교정 판정까지", "dense": True, "scope": SCOPE_NEW_SEED_DEMO,
        "lead": "새 seed에서 새 Nemotron 판단, 승인된 검사, 교정 제안 승인, Isaac 재학습, 고정 평가 판정을 한 번에 이었다. "
                "왼쪽은 보상 결함, 오른쪽은 모터 결함이다.",
        "blocks": [*reward, *motor],  # 비용은 마지막 장면의 종단 시연 비용 표에 둔다
        "notes": ["결함과 probe는 같은 설계자가 만들었고 이미 본 결함 종류의 새 seed 한 건씩이다. "
                  "진단 정확도·일반화·교정 성공률·GPU 절감의 근거가 아니다.",
                  "승인은 사용자가 위임한 운영자 승인이며 승인자 신원은 인증하지 않았다.",
                  *([feedback] if feedback else [])],
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


def _policy_comparison(summary: dict | None, policy: str, *, option_contract: bool = False) -> list[Any]:
    if summary is None:
        return [None] * 7
    rows = [row for row in summary.get("rows", []) if row.get("policy") == policy]
    fault_rows = [row for row in rows if row.get("gate", {}).get("run") is True]
    missing_entries = summary.get("missing_rows", summary.get("missing_policy_rows", []))
    missing = sum((entry.get("policy") if isinstance(entry, dict) else str(entry).split("/")[-1]) == policy
                  for entry in missing_entries)
    invalid = sum(row.get("status") == "invalid" for row in rows)
    coverage = f"{len(rows)}/8 · {invalid} · {missing}"
    if policy == "strong_registered_rule":
        validity = "-"
        subject = "등록 규칙"
    elif option_contract:
        validity = f"{sum(row.get('first_response_valid') is True for row in fault_rows)}/{len(fault_rows)}"
        subject = "모델"
    else:
        first = sum(row.get("first_pass_valid_actions", 0) for row in fault_rows)
        validity = f"{first}/{len(fault_rows)}"
        subject = "모델"
    return [subject, coverage, validity,
            sum(row.get("correct_confirmed_family") is True for row in fault_rows),
            sum(row.get("decision_outcome") == "HOLD" for row in fault_rows),
            sum(row.get("compound_partial") is True for row in fault_rows),
            (f"{sum(row.get('paired_probes', 0) for row in fault_rows)} · "
             f"{sum(row.get('model_requests', 0) for row in fault_rows)}")]


def scene_prospective_comparison(sources: Sources, slots: dict) -> dict:
    comparison_path = slots["prospective_agentic_comparison"]
    audit_path = slots["recorded_optimizer_probe_audit"]
    option_path = slots["registry_action_contract"]
    recovery_path = slots["registry_action_recovery"]
    comparison_row, comparison = manifest_result(
        sources, comparison_path, manifest_name="artifact_manifest.json",
        manifest_schema="prospective_agentic_public_manifest_v1",
        summary_schema="prospective_agentic_public_summary_v1", kind="prospective_agentic_comparison")
    _audit_row, audit = manifest_result(
        sources, audit_path, manifest_name="manifest.json",
        manifest_schema="recorded_optimizer_probe_audit_public_manifest_v1",
        summary_schema="recorded_optimizer_probe_audit_public_v1", kind="recorded_optimizer_probe_audit")
    option_row, option = manifest_result(
        sources, option_path, manifest_name="artifact_manifest.json",
        manifest_schema="registry_action_contract_public_manifest_v1",
        summary_schema="registry_action_contract_public_summary_v1", kind="registry_action_contract")
    recovery_row, recovery = manifest_result(
        sources, recovery_path, manifest_name="artifact_manifest.json",
        manifest_schema="registry_action_recovery_public_manifest_v1",
        summary_schema="registry_action_recovery_public_summary_v1", kind="registry_action_recovery")
    policy_rows = []
    old_labels = {"eager_model": "기존 · 즉시", "evidence_guided_model": "기존 · 근거",
                  "strong_registered_rule": "기존 · 등록 규칙"}
    for policy in ("eager_model", "evidence_guided_model", "strong_registered_rule"):
        values = _policy_comparison(comparison, policy)
        policy_rows.append([old_labels[policy],
                            *[cell(sources, value, comparison_row["id"], None) for value in values]])
    option_labels = {"eager_model": "option · 즉시", "evidence_guided_model": "option · 근거"}
    for policy in ("eager_model", "evidence_guided_model"):
        values = _policy_comparison(option, policy, option_contract=True)
        policy_rows.append([option_labels[policy],
                            *[cell(sources, value, option_row["id"], None) for value in values]])

    audit_rows = []
    for seed in ("s2320", "s2321"):
        seed_relative = f"{audit_path}/seeds/{seed}.json"
        seed_source = sources.file(seed_relative, kind="recorded_optimizer_probe_audit_file")
        value = (_read(sources.root / seed_relative)
                 if audit is not None and seed_source["status"] == "ready" else None)
        recorded = value and value["methods"]["P_recorded_optimizer_updates"]
        fresh = value and value["methods"]["P_optimizer_updates"]
        audit_rows.append([
            seed,
            cell(sources, recorded and recorded["parent_wall_s"], seed_source["id"], 2),
            cell(sources, fresh and fresh["parent_wall_s"], seed_source["id"], 2),
            cell(sources, recorded and recorded["sdk_launches"], seed_source["id"], 0),
            cell(sources, fresh and fresh["sdk_launches"], seed_source["id"], 0),
            cell(sources, "일치" if value and value.get("agreement") else None, seed_source["id"], None),
        ])
    source_metrics = recovery and recovery["reports"]["source_r1_s2320"]["metrics"]
    corrected_metrics = recovery and recovery["reports"]["corrected_s3332"]["metrics"]
    checks = recovery and recovery["recovery"]["verdict"]["checks"]
    runtime = recovery and recovery["runtime_first3"]
    recovery_rows = [
        ["결과", cell(sources, recovery and
         f"{recovery['outcome']} · {sum(value is True for value in checks.values())}/{len(checks)} 기준 PASS",
         recovery_row["id"], None)],
        ["선택 → 교정", cell(sources, recovery and
         "recorded count 검사 → PPO epoch 복원 · restart",
         recovery_row["id"], None)],
        ["범위", cell(sources, recovery and
         f"s{recovery['selection']['source_seed']} R1 → s{recovery['repair']['recovery_seed']} restart · 1 case",
         recovery_row["id"], None)],
        ["XY RMSE", cell(sources, recovery and
         f"{source_metrics['lin_vel_rmse_mps']:.4f} → {corrected_metrics['lin_vel_rmse_mps']:.4f} m/s",
         recovery_row["id"], None)],
        ["Yaw RMSE", cell(sources, recovery and
         f"{source_metrics['yaw_rate_rmse_radps']:.4f} → {corrected_metrics['yaw_rate_rmse_radps']:.4f} rad/s",
         recovery_row["id"], None)],
        ["교정 평가", cell(sources, recovery and
         (f"생존 {corrected_metrics['mean_survival_s']:.4f}s · 낙상 "
          f"{round(corrected_metrics['fall_rate'] * 26)}/26 ({corrected_metrics['fall_rate'] * 100:.2f}%) · "
          f"first-3 {runtime['observed_steps']}/{runtime['expected_steps']} step · "
          f"API {recovery['study_separation']['recovery_api_requests']}"), recovery_row["id"], None)],
    ]
    return {
        "id": "prospective_comparison", "title": "새 source 진단 비교와 기록 재사용 검사", "dense": True,
        "lead": "같은 8개 source의 기존 3정책과 노출 후 개발한 option 2정책, 첫 3 update 기록 재사용 검사를 함께 본다.",
        "blocks": [
            {"type": "table", "wide": True, "title": "정책별 판정(각 8행, 결함 분모 6행)",
             "columns": ["판 · 정책", "선택 주체", "행 / 무효 / 미완료", "첫 유효", "원인 확정", "HOLD", "복합 부분", "검사 · API"],
             "rows": policy_rows},
            {"type": "table", "title": "Recorded 검사 감사(첫 3 update)",
             "columns": ["seed", "recorded parent(s)", "fresh parent(s)", "recorded SDK", "fresh SDK", "판정"],
             "rows": audit_rows},
            {"type": "table", "title": "사전 고정 R1 후속 평가", "columns": ["항목", "공개 기록"],
             "rows": recovery_rows},
        ],
        "notes": [
            "Option 계약은 첫 결정 12/12 유효지만 진단 성공과 다르다. 근거 6건은 HOLD, 즉시는 R1 2건 확정·복합 2건 부분 확정·F1 2건 HOLD였다.",
            "계약은 서로 다르고 option은 결과 노출 후 개발했다. 정상 gate는 모델 호출 0이다. R1 후속 1 case는 모델 검사 선택→host 교정이며 참조 seed는 새 회복이 아니고 낙상은 1/26이다.",
            "Recorded 검사는 API·SDK 0회로 2/2 판정을 재현했지만 parent wall은 커널 시간이 아니다. 기본 모드 채택, 모델 우월·조기 탐지·일반 회복·비용 절감 근거가 아니다.",
        ],
    }


def _guided_partition_values(summary: dict | None, *, diagnostic_v2: bool = False) -> list[Any]:
    """근거 유도 정책의 결함 6행을 공개 summary에서 집계한다."""
    if summary is None:
        return [None] * 9
    rows = [row for row in summary.get("rows", [])
            if row.get("gate", {}).get("run")
            and (diagnostic_v2 or row.get("policy") == "evidence_guided_model")]
    if len(rows) != 6:
        return [None] * 9
    return [
        f"{sum(row.get('first_response_valid') is True for row in rows)}/6",
        sum(row.get("correct_confirmed_family") is True for row in rows),
        sum(row.get("decision_outcome") == "HOLD" for row in rows),
        sum(row.get("compound_partial") is True for row in rows),
        sum("read_evidence" in row.get("observed_actions", []) for row in rows),
        sum(row.get("paired_probes", 0) for row in rows),
        sum(row.get("model_requests", 0) for row in rows),
        summary.get("dialogue_files") if diagnostic_v2 else "-",
        "후속 개발" if diagnostic_v2 else "기존 관측",
    ]


def _partition_case_values(row: dict | None) -> list[Any]:
    if row is None:
        return [None] * 6
    expected = " + ".join(row.get("expected_family", [])) or "-"
    confirmed = row.get("confirmed_family") or "-"
    if row.get("correct_confirmed_family") is True:
        extent = "복합 부분 확정" if row.get("compound_partial") is True else "단일 결함 확정"
    else:
        extent = "확정 없음"
    return [expected, row.get("first_action"), row.get("decision_outcome"), confirmed, extent,
            f"{row.get('paired_probes', 0)} / {row.get('model_requests', 0)}"]


def scene_registered_partitions(sources: Sources, slots: dict) -> dict:
    """등록 partition 공개 뒤 추가한 설명 메타데이터의 개발 결과를 별도 장면으로 둔다."""
    option_path = slots["registry_action_contract"]
    diagnostic_path = slots["diagnostic_option_study_v1"]
    option_row, option = manifest_result(
        sources, option_path, manifest_name="artifact_manifest.json",
        manifest_schema="registry_action_contract_public_manifest_v1",
        summary_schema="registry_action_contract_public_summary_v1", kind="registry_action_contract")
    diagnostic_row, diagnostic = manifest_result(
        sources, diagnostic_path, manifest_name="artifact_manifest.json",
        manifest_schema="diagnostic_option_study_public_manifest_v1",
        summary_schema="diagnostic_option_study_public_summary_v1", kind="diagnostic_option_study")
    comparison = []
    for label, values, source in (
        ("O1 · 기존 근거 유도", _guided_partition_values(option), option_row["id"]),
        ("O2 · 설명 메타데이터 추가", _guided_partition_values(diagnostic, diagnostic_v2=True),
         diagnostic_row["id"]),
    ):
        comparison.append([label, *[cell(sources, value, source, None) for value in values]])
    by_case = {row.get("case_id"): row for row in (diagnostic or {}).get("rows", [])
               if row.get("gate", {}).get("run")}
    cases = []
    for case_id in ("c02", "c03", "c04", "c06", "c07", "c08"):
        cases.append([case_id, *[cell(sources, value, diagnostic_row["id"], None)
                                for value in _partition_case_values(by_case.get(case_id))]])
    return {
        "id": "registered_partitions", "title": "등록 partition 설명을 노출한 후속 개발", "dense": True,
        "lead": "같은 관측 원자료·도구·예산에서 registry 예측 메타데이터와 설명을 보강한 사후 개발 결과를 기존 근거 유도 계약과 나눠 본다.",
        "blocks": [
            {"type": "table", "wide": True, "title": "근거 유도 계약 비교(결함 6행)",
             "columns": ["계약", "첫 유효", "원인 확정", "HOLD", "복합 부분", "read_evidence", "paired", "API", "대화 파일", "구분"],
             "rows": comparison},
            {"type": "table", "wide": True, "title": "O2 공개 사례별 판정",
             "columns": ["case", "예상 family", "첫 행동", "최종 결과", "확정 family", "확정 범위", "paired / API"],
             "rows": cases},
        ],
        "notes": [
            "O2는 O1 결과 공개 뒤 만든 후속 개발이며, 같은 관측 원자료·도구·예산에 registry 예측 메타데이터를 새로 노출했다.",
            "원인 확정은 단일 결함과 복합의 부분 확정 합계이며 둘 다 seed2321에서 나왔다. 독립된 두 계보의 성공이나 두 실패의 전체 해결이 아니다.",
            "새 교정·회복과 새 seed 평가는 없으며, 모델 3→3.5 효과·기본 채택·규칙 우위를 입증하지 않는다.",
        ],
    }


def _reasoning_case(sources: Sources, relative: str, case_id: str, mode: str,
                    ready: bool) -> tuple[dict | None, str]:
    path = f"{relative}/cases/{case_id}-{mode}.json"
    row = sources.file(path, kind="reasoning_diagnosis_case")
    return (_read(sources.root / path) if ready and row["status"] == "ready" else None,
            row["id"])


def _contains_observation(value: Any, target: str, outcome: str) -> bool:
    if isinstance(value, dict):
        if value.get("target") == target and value.get("outcome") == outcome:
            return True
        return any(_contains_observation(child, target, outcome) for child in value.values())
    if isinstance(value, list):
        return any(_contains_observation(child, target, outcome) for child in value)
    if isinstance(value, str) and value.startswith(("{", "[")):
        try:
            return _contains_observation(json.loads(value), target, outcome)
        except json.JSONDecodeError:
            return False
    return False


def _reasoning_confirmation_release(sources: Sources, relative: str | None) -> tuple[dict, dict | None]:
    """새 source reasoning 확인 묶음의 release→base 해시 사슬을 검증한다."""
    key = relative or "slot:reasoning_confirmation"
    if key in sources.replays:
        return sources.rows[key], sources.replays[key]
    sources.replays[key] = None
    row = {"id": key, "path": f"{relative}/release-manifest.json" if relative else None,
           "kind": "reasoning_confirmation_release", "sha256": None,
           "status": "waiting", "reason": "path_not_set" if relative is None else "missing"}
    sources.rows[key] = row
    if relative is None:
        return row, None
    root = sources.root / relative
    release_path = root / "release-manifest.json"
    if not release_path.is_file():
        return row, None
    row["sha256"] = _sha(release_path)
    if not _git(sources.root, "ls-files", "--", relative).stdout.strip():
        row["reason"] = "not_committed"
        return row, None
    if _git(sources.root, "status", "--porcelain", "--", relative).stdout.strip():
        row["status"], row["reason"] = "mismatch", "modified_since_commit"
        return row, None
    try:
        release = _read(release_path)
        if release.get("schema") != "prospective_reasoning_confirmation_release_manifest_v2":
            raise ValueError("reasoning release manifest schema 불일치")
        if release.get("status") != "complete" or release.get("verified_base_artifact_count") != 44:
            raise ValueError("reasoning release 상태 불일치")
        release_files = release.get("files")
        if (not isinstance(release_files, dict)
                or set(release_files) != {"release-provenance.json", "release-summary.json"}):
            raise ValueError("reasoning release 파일 목록 불일치")
        for name, expected in release_files.items():
            source_row = sources.file(f"{relative}/{name}", kind="reasoning_confirmation_release_file",
                                      expected_sha256=expected)
            if source_row["status"] != "ready":
                raise ValueError("reasoning release 파일 승인 불일치")
        provenance = _read(root / "release-provenance.json")
        release_summary = _read(root / "release-summary.json")
        if provenance.get("schema") != "prospective_reasoning_confirmation_release_provenance_v2":
            raise ValueError("reasoning release provenance schema 불일치")
        base_manifest_path = root / "verified-base" / "artifact-manifest.json"
        base_manifest_sha = _sha(base_manifest_path)
        if base_manifest_sha != release.get("verified_base_manifest_sha256"):
            raise ValueError("reasoning base manifest SHA 불일치")
        if base_manifest_sha != provenance.get("verified_base_manifest_sha256"):
            raise ValueError("reasoning provenance base manifest SHA 불일치")
        base_manifest = _read(base_manifest_path)
        if base_manifest.get("schema") != "prospective_reasoning_confirmation_public_manifest_v1":
            raise ValueError("reasoning base manifest schema 불일치")
        declared = base_manifest.get("files")
        if base_manifest.get("status") != "complete" or base_manifest.get("artifact_count") != 44:
            raise ValueError("reasoning base manifest 상태 불일치")
        if not isinstance(declared, dict) or len(declared) != 44:
            raise ValueError("reasoning base manifest 수량 불일치")
        actual = {path.relative_to(root / "verified-base").as_posix()
                  for path in (root / "verified-base").rglob("*") if path.is_file()}
        if actual != {"artifact-manifest.json", *declared}:
            raise ValueError("reasoning base inventory 불일치")
        top_level = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        expected_top_level = {"release-manifest.json", *release_files,
                              *(f"verified-base/{name}" for name in actual)}
        if top_level != expected_top_level:
            raise ValueError("reasoning release inventory 불일치")
        for name, expected in declared.items():
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != name:
                raise ValueError("reasoning base 상대 경로 불일치")
            source_row = sources.file(f"{relative}/verified-base/{name}",
                                      kind="reasoning_confirmation_base_file",
                                      expected_sha256=expected)
            if source_row["status"] != "ready":
                raise ValueError("reasoning base 파일 승인 불일치")
        base_summary = _read(root / "verified-base" / "summary.json")
        if _sha(root / "verified-base" / "summary.json") != release_summary.get(
                "verified_base_summary_sha256"):
            raise ValueError("reasoning base summary SHA 불일치")
        if base_summary.get("schema") != "prospective_reasoning_confirmation_public_summary_v1":
            raise ValueError("reasoning base summary schema 불일치")
        case_names = [name for name in declared if name.startswith("cases/")]
        call_names = [name for name in declared if name.startswith("calls/")]
        if len(case_names) != 24 or len(call_names) != 18:
            raise ValueError("reasoning 공개 행·호출 수 불일치")
        if base_summary.get("complete_rows") != 24 or base_summary.get("model_requests") != 18:
            raise ValueError("reasoning summary 행·호출 수 불일치")
        policies = base_summary.get("policies")
        if not isinstance(policies, dict):
            raise ValueError("reasoning policy summary 불일치")
        for mode in ("legacy_none", "enabled_low"):
            metrics = policies.get(mode, {})
            if (metrics.get("rows") != 8 or metrics.get("fault_rows") != 6
                    or metrics.get("compound_complete") != 0
                    or metrics.get("wrong_family_confirmations") != 0):
                raise ValueError("reasoning policy 집계 불일치")
        uncertainty = base_summary.get("uncertainty", {})
        if (base_summary.get("independent_source_seeds") != 2
                or uncertainty.get("independent_units") != 2
                or uncertainty.get("independent_unit") != "source_seed"):
            raise ValueError("reasoning 독립 seed 수 불일치")
        policy_audit = release_summary.get("policy_audit", {})
        if (release_summary.get("assessment") != "not_confirmed"
                or release_summary.get("complete_rows") != 24
                or policy_audit.get("status") != "registered_rule_prose_code_mismatch"
                or policy_audit.get("rule_description_matches_actual_sequence") is not False
                or policy_audit.get("rule_arm_unqualified_for_stated_sequence") is not True):
            raise ValueError("reasoning release 판정·정책 감사 불일치")
        accounting = release_summary.get("provider_request_accounting", {})
        if (accounting.get("model_requests_including_replay") != 18
                or accounting.get("replayed_preserved_requests") != 1
                or accounting.get("new_provider_requests") != 17):
            raise ValueError("reasoning provider 호출 원장 불일치")
    except (OSError, ValueError, TypeError, KeyError) as error:
        row["status"], row["reason"] = "mismatch", f"release_failed:{type(error).__name__}"
        return row, None
    row["status"], row["reason"] = "ready", None
    result = {"base": base_summary, "release": release_summary, "provenance": provenance}
    sources.replays[key] = result
    return row, result


def scene_reasoning_followup(sources: Sources, slots: dict) -> dict:
    """기록한 첫 reasoning 선택을 실제 진단 루프로 이어 간 개발 결과를 표시한다."""
    reasoning_path = slots["reasoning_diagnosis_study"]
    confirmation_path = slots.get("reasoning_confirmation")
    reasoning_row, summary = manifest_result(
        sources, reasoning_path, manifest_name="artifact_manifest.json",
        manifest_schema="reasoning_diagnosis_study_public_manifest_v1",
        summary_schema="reasoning_diagnosis_study_public_summary_v1",
        kind="reasoning_diagnosis_study")
    confirmation_row, confirmation = _reasoning_confirmation_release(sources, confirmation_path)
    new_summary = (confirmation or {}).get("base", {})
    mode_rows = []
    for mode, label in (("legacy_none", "reasoning 없음"), ("enabled_low", "reasoning low")):
        metrics = (summary or {}).get("mode_metrics", {}).get(mode, {})
        new_metrics = new_summary.get("policies", {}).get(mode, {})
        single_correct = (new_metrics.get("correct_confirmed_families", 0)
                          - new_metrics.get("compound_partial", 0))
        values = [
            f"{metrics.get('correct_confirmed_family')}/6" if summary is not None else None,
            f"{new_metrics.get('correct_confirmed_families')}/6" if confirmation is not None else None,
            f"{single_correct}/4" if confirmation is not None else None,
            f"{new_metrics.get('compound_partial')}/2" if confirmation is not None else None,
            new_metrics.get("model_requests") if confirmation is not None else None,
            new_metrics.get("paired_probes") if confirmation is not None else None,
        ]
        mode_rows.append([label,
                          cell(sources, values[0], reasoning_row["id"], None),
                          *[cell(sources, value, confirmation_row["id"], None)
                            for value in values[1:]]])

    contrasts = new_summary.get("seed_paired_contrasts", []) if confirmation is not None else []
    contrast_by_seed = {row.get("seed"): row for row in contrasts if isinstance(row, dict)}
    seed_rows = []
    for seed in (2430, 2431):
        contrast = contrast_by_seed.get(seed, {})
        seed_rows.append([
            str(seed),
            cell(sources, f"{contrast.get('legacy_none_correct')}/3" if contrast else None,
                 confirmation_row["id"], None),
            cell(sources, f"{contrast.get('enabled_low_correct')}/3" if contrast else None,
                 confirmation_row["id"], None),
        ])

    legacy, legacy_id = _reasoning_case(
        sources, reasoning_path, "c07", "legacy_none", summary is not None)
    low, low_id = _reasoning_case(
        sources, reasoning_path, "c07", "enabled_low", summary is not None)
    legacy_dialogue_path = f"{reasoning_path}/dialogues/c07/legacy_none/step-02.json"
    legacy_dialogue_row = sources.file(legacy_dialogue_path, kind="reasoning_diagnosis_dialogue")
    legacy_dialogue = (_read(sources.root / legacy_dialogue_path)
                       if summary is not None and legacy_dialogue_row["status"] == "ready" else None)
    legacy_ok = bool(legacy and legacy.get("actions") == ["run_probe", "hold"]
                     and legacy.get("decision_outcome") == "HOLD"
                     and legacy.get("actual_sdk_role_executions") == 2
                     and _contains_observation(legacy_dialogue, "P_reward", "normal"))
    low_ok = bool(low and low.get("actions") == ["read_evidence", "run_probe"]
                  and low.get("decision_outcome") == "confirmed"
                  and low.get("correct_complete_family") is True
                  and low.get("actual_sdk_role_executions") == 0
                  and low.get("probe_execution_audit", [{}])[0].get("probe") ==
                  "P_recorded_optimizer_updates")
    case_rows = [
        ["reasoning 없음", cell(sources, "P_reward → normal", legacy_dialogue_row["id"], None)
         if legacy_ok else cell(sources, None, legacy_dialogue_row["id"], None),
         cell(sources, "모델 HOLD", legacy_id, None) if legacy_ok else cell(sources, None, legacy_id, None),
         cell(sources, 2, legacy_id, 0) if legacy_ok else cell(sources, None, legacy_id, 0)],
        ["reasoning low", cell(sources, "count 읽기 → recorded count abnormal", low_id, None)
         if low_ok else cell(sources, None, low_id, None),
         cell(sources, "optimizer_update_count 확정", low_id, None)
         if low_ok else cell(sources, None, low_id, None),
         cell(sources, 0, low_id, 0) if low_ok else cell(sources, None, low_id, 0)],
    ]
    return {
        "id": "reasoning_followup", "title": "Reasoning 후속 확인: 개발 결과와 새 source 결과",
        "lead": "개발 2/6 대 5/6 뒤 새 seed에서는 4/6 대 3/6이었다. 사전 판정은 not_confirmed다.",
        "blocks": [
            {"type": "table", "title": "개발 기록 → 새 source 확인(각 mode 결함 6행)",
             "columns": ["요청 mode", "기존 개발 확정", "새 확정", "단일 결함", "복합 일부", "새 자료 요청", "paired 검사"],
             "rows": mode_rows},
            {"type": "table", "title": "새 source seed별 확정(각 seed 결함 3행)",
             "columns": ["source seed", "reasoning 없음", "reasoning low"], "rows": seed_rows},
            {"type": "table", "title": "옛 개발 예시 · c07 R1",
             "columns": ["요청 mode", "관측 경로", "최종 판정", "새 SDK role"], "rows": case_rows},
        ],
        "notes": [
            "새 자료의 판정은 not_confirmed이며 기본 mode는 바꾸지 않았다. 복합 결함 전체 확정은 두 mode 모두 0/2다.",
            "정상 6행은 host gate라 API와 probe가 0이다. LLM의 정상 판정으로 세지 않았다.",
            "모델 API 18회에는 보존 응답 replay 1회와 새 요청 17회가 포함된다. post-response host 계약 수정 뒤 같은 source로 이어 간 노출 후 개발이다.",
            "독립 단위는 source seed 2개다. 6개 결함 행을 독립 seed로 세지 않으며 원래 4-arm 확인도 아니다.",
            "등록한 규칙 문구와 동결 코드의 실제 순서가 달라 규칙 arm은 강한 기준선으로 해석하지 않는다. SDK upper bound 38은 실측 실행 횟수가 아니다.",
            "옛 c07 표는 기존 개발 자료다. 새 source 성능, 회복, 영상 이득, LLM 우월의 근거로 바꾸지 않는다.",
        ],
    }


def scene_four_way_development(sources: Sources, slots: dict) -> dict:
    row, summary = manifest_result(sources, slots["local_pair_four_arm"], manifest_name="manifest.json",
        manifest_schema="local_pair_four_arm_public_v1_manifest",
        summary_schema="local_pair_four_arm_public_v1_summary", kind="local_pair_four_arm")
    rows = []
    for arm, title in (("registry_next", "등록 규칙"), ("fixed_eligible", "고정 순서"),
                       ("logs_only", "3.5 수치"), ("logs_plus_frames", "같은 3.5 + 6프레임")):
        metrics = (summary or {}).get("arms", {}).get(arm, {})
        confirmed = metrics.get("correct_confirmation", {})
        values = [f"{confirmed.get('successes')}/{confirmed.get('total')}" if metrics else None,
                  metrics.get("hold"), metrics.get("paired_probe_count"), metrics.get("provider_calls"),
                  metrics.get("raw_role_execution_count"), metrics.get("raw_owned_child_wall_s")]
        rows.append([title, *[cell(sources, value, row["id"], 2 if i == 5 else None)
                             for i, value in enumerate(values)]])
    return {"id": "four_way_development", "title": "같은 수치에서 네 방식: 검사 3회와 1회",
        "lead": ("두 개발 계보에서 수치와 영상 방식 모두 P_reward 한 번으로 확정했다. 프레임 추가로 확정 결과나 검사 수가 달라지지 않았다."
                 if summary else "공개 manifest와 파일 SHA가 확인된 경우에만 네 방식 비교의 수치를 표시한다."),
        "blocks": [{"type": "table", "title": "두 F1 계보 합계 · source 2430 / 2431",
                    "columns": ["방식", "올바른 확정", "HOLD", "짝 검사", "API", "SDK 역할", "자식 wall(s)"],
                    "rows": rows},
                   {"type": "table", "title": "공통 입력과 실제 소비", "columns": ["항목", "값"],
                    "rows": [["공통 수치", cell(sources, "학습 48 + 같은 Play 54", row["id"], None)],
                             ["새 API / SDK 역할", cell(sources,
                              f"{summary['actual_api_calls']} / {summary['actual_sdk_role_executions']}" if summary else None,
                              row["id"], None)]]}],
        "notes": ["독립 계보는 2개다. 이미 본 보상 결함 개발 결과이며 새 계보 본평가나 모델 버전 비교가 아니다.",
                  ("영상 방식도 학습 수치만 인용했다. raw frame 인용 0이며 영상 이득을 입증하지 않는다."
                   if summary else "영상 인용과 실제 검사 결과를 분리하며 원인 확정과 회복을 구분한다."),
                  "자식 wall은 시작·종료·정리 확인까지다. GPU kernel 시간이 아니다. 정상 host gate 8행은 모델의 정상 식별로 세지 않았다."]}


def scene_five_type_sources(sources: Sources, slots: dict) -> dict:
    row, summary = manifest_result(sources, slots["five_type_sources"], manifest_name="manifest.json",
        manifest_schema="lineage_source_collection_public_v2_manifest",
        summary_schema="lineage_source_collection_public_v2_summary", kind="five_type_sources")
    cases = (summary or {}).get("case_plan", [])
    rows = []
    for source_type, label in (("NONE", "정상"), ("F1", "보상 좌표"), ("F2", "모터"),
                               ("COMPOUND", "보상 + PPO epoch"), ("F5", "마찰 · 교정 미지원")):
        group = [case for case in cases if case["source_type"] == source_type]
        values = ([len(group), sum(case["valid_after_dedup"] is True for case in group),
                   sum(case["classification"] == "eligible" for case in group),
                   sum(case["classification"] == "out_of_support" for case in group)] if summary else [None] * 4)
        rows.append([label, *[cell(sources, value, row["id"], None) for value in values]])
    counts = (summary or {}).get("counts", {})
    totals = [[label, cell(sources, counts.get(key), row["id"], None)] for key, label in (
        ("training_sdk_completed", "학습 완료"), ("fixed26_sdk_completed", "26조건 평가 완료"),
        ("actual_model_api_calls", "이 원천 단계의 모델 호출"), ("repairs", "이 원천 단계의 교정"))]
    return {"id": "five_type_sources", "title": "다섯 유형의 새 학습 입력 확보",
        "lead": "정상 사례도 다른 seed의 정상 정책과 비교한다. 결함·참조 학습과 고정 평가를 진단 결과와 분리한다.",
        "blocks": [{"type": "table", "title": "등록한 사례와 적용 범위",
                    "columns": ["유형", "계획 사례", "중복 제외 후", "적격", "교정 미지원"], "rows": rows},
                   {"type": "table", "title": "실제 원천 수집 · 진단과 회복의 분모가 아님",
                    "columns": ["항목", "횟수"], "rows": totals}],
        "notes": ["적격은 비교 입력의 적용성 판정이다. 원인 확정이나 회복 성공을 뜻하지 않는다.",
                  "F5는 원천 학습과 진단 대상으로 남기고 교정 학습은 허용하지 않는다. 실패가 없는 다른 seed로 바꾸지 않는다.",
                  "이미 알려진 결함 종류의 새 seed 반복이다. 새로운 결함 종류에 대한 일반화·GPU 절감의 근거가 아니다."],
        "scope": SCOPE_SOURCE_COLLECTION}


def scene_five_policy_diagnosis(sources: Sources, slots: dict) -> dict:
    relative = slots["five_policy_diagnosis"]
    row, summary = manifest_result(
        sources, relative, manifest_name="manifest.json",
        manifest_schema="lineage_five_policy_public_manifest_v1",
        summary_schema="lineage_five_policy_summary_v1", kind="five_policy_diagnosis")
    cases_relative = f"{relative}/cases.json"
    cases_source = sources.file(cases_relative, kind="five_policy_diagnosis_file")
    cases = (_read(sources.root / cases_relative)
             if summary is not None and cases_source["status"] == "ready" else None)
    case_rows = (cases or {}).get("rows", [])
    labels = {
        "registry_next": "등록 규칙", "fixed_eligible": "고정 순서",
        "logs_only": "3.5 수치", "logs_plus_frames": "3.5 + 프레임",
        "optimizer_first_registered_rule": "optimizer 우선 등록 규칙",
    }
    outcome_rows = []
    boundary_rows = []
    for policy in labels:
        metrics = (summary or {}).get("policies", {}).get(policy, {})
        tokens = metrics.get("provider_tokens") or {}
        policy_cases = [item for item in case_rows if item.get("policy") == policy]
        incomplete_cost_rows = max(
            metrics.get("infra_unknown_rows", 0) if metrics else 0,
            sum(item.get("technical_missing") is True for item in policy_cases),
        )
        parent_wall = metrics.get("parent_wall_s")
        parent_wall_value = (
            f"{parent_wall:.2f}s 기록 · 동일 범위 비교 불가"
            if parent_wall is not None and incomplete_cost_rows else parent_wall
        )
        outcome_values = [
            (f"{metrics.get('correct_any_family')}/{metrics.get('qualified_fault_cases')}"
             if metrics else None),
            metrics.get("wrong_confirmation"), metrics.get("hold_rows"),
            (f"{metrics.get('compound_partial')} / {metrics.get('compound_full')}"
             if metrics else None),
            metrics.get("paired_probes"), metrics.get("model_calls"),
            parent_wall_value,
        ]
        outcome_rows.append([labels[policy], *[
            cell(sources, value, row["id"], 2 if index == 6 else None)
            for index, value in enumerate(outcome_values)
        ]])
        f5 = [item for item in policy_cases if item.get("qualification", {}).get("arm") == "F5"]
        f5_value = (f"{sum(item.get('correct_any_family') is True for item in f5)} 진단 · "
                    f"{sum(item.get('hold_reason') == 'repair_unsupported' for item in f5)} repair_unsupported"
                    if cases else None)
        usage_complete = metrics and metrics.get("provider_usage_unknown_calls") == 0
        provider_complete = metrics and metrics.get("provider_unknown_events") == 0
        boundary_values = [
            (f"{metrics.get('normal_model_shadow_proposal')} / "
             f"{metrics.get('normal_model_observed')}" if metrics else None),
            metrics.get("normal_host_gate"), f5_value,
            tokens.get("total_tokens") if usage_complete else None,
            metrics.get("provider_wall_s") if provider_complete else None,
        ]
        boundary_rows.append([labels[policy], *[
            cell(sources, value, cases_source["id"] if index == 2 else row["id"],
                 2 if index == 4 else None)
            for index, value in enumerate(boundary_values)
        ]])
    lead = "공개 manifest와 파일 SHA가 확인된 경우에만 10개 독립 case의 다섯 정책 진단을 표시한다."
    if summary is not None:
        lead = (f"독립 case {summary['denominator']['independent_cases']}개를 정책마다 같은 완료 조건으로 비교했다. "
                f"등록 주대조 판정은 {summary['primary']['outcome']}, "
                f"등록 영상대조 판정은 {summary['secondary']['outcome']}이다.")
    return {
        "id": "five_policy_diagnosis", "title": "다섯 정책 진단: 확정·HOLD와 실행 기록", "dense": True,
        "lead": lead,
        "blocks": [
            {"type": "table", "wide": True, "title": "정책별 진단과 실행 기록 범위",
             "columns": ["정책", "올바른 확정 / 결함", "오확정", "HOLD", "복합 부분 / 전체",
                         "짝 검사", "API", "parent wall 기록(s)"], "rows": outcome_rows},
            {"type": "table", "wide": True, "title": "정상 gate·F5와 모델 API 비용",
             "columns": ["정책", "정상 native shadow 제안 / 관측", "정상 host gate",
                         "F5", "provider tokens", "provider wall(s)"], "rows": boundary_rows},
        ],
        "notes": [
            "정상 native shadow와 host gate를 분리한다. 정상 행의 host gate는 모델이 정상을 확정한 결과가 아니다.",
            "첫 원인 확정으로 끝나는 고리다. 복합 전체 확정 0은 전체 원인 탐색의 실패율이 아니다. F5의 repair_unsupported도 교정 성공이 아니다.",
            "분모는 독립 case checkpoint이며 반복 호출·프레임은 분모를 늘리지 않는다. parent wall은 GPU kernel 시간이나 비용 절감량이 아니다.",
            "기술 중단 행이 포함된 정책의 parent wall은 미완결 관측을 포함하므로 동일 범위 비용으로 비교하지 않는다.",
            "3.5·프레임 효과는 등록 대조의 공개 판정만 표시한다. 모델 버전 우위·GPU 절감·새 결함 일반화의 근거가 아니다.",
        ],
    }


def scene_five_policy_recovery(sources: Sources, slots: dict) -> dict:
    relative = slots["five_policy_recovery"]
    row, summary = manifest_result(
        sources, relative, manifest_name="manifest.json",
        manifest_schema="five_policy_recovery_public_manifest_v1",
        summary_schema="five_policy_recovery_public_summary_v1", kind="five_policy_recovery")
    cases_relative = f"{relative}/cases.json"
    cases_source = sources.file(cases_relative, kind="five_policy_recovery_file")
    cases = (_read(sources.root / cases_relative)
             if summary is not None and cases_source["status"] == "ready" else None)
    observations_relative = slots.get("five_policy_recovery_observations")
    if observations_relative is None:
        observations_source = sources.file(None, kind="five_policy_recovery_observations")
        observations = None
    else:
        observations_source, observations = manifest_result(
            sources, observations_relative, manifest_name="manifest.json",
            manifest_schema="five_policy_recovery_observations_manifest_v1",
            summary_schema="five_policy_recovery_observations_v1",
            kind="five_policy_recovery_observations")
        if observations is not None and row["status"] != "ready":
            observations_source["status"] = "waiting"
            observations_source["reason"] = "canonical_manifest_not_ready"
            sources.replays[observations_relative] = None
            observations = None
        elif observations is not None and observations.get("canonical_manifest_sha256") != row["sha256"]:
            observations_source["status"] = "mismatch"
            observations_source["reason"] = "canonical_manifest_sha256_mismatch"
            sources.replays[observations_relative] = None
            observations = None
        elif observations is not None:
            try:
                observation_rows = observations["rows"]
                optimizer = observations["optimizer_counts"]
                fresh = optimizer["fresh_three_update_inspection"]
                count_records = [optimizer["first_training"], fresh["case"],
                                 fresh["reference"], optimizer["second_training"]]
                expected_targets = {(case, policy) for case, policy in (
                    ("case-03", "fixed_eligible"), ("case-03", "logs_only"),
                    ("case-07", "fixed_eligible"), ("case-07", "logs_only"))}
                manifest_value = _read(sources.root / observations_relative / "manifest.json")
                if (manifest_value.get("status") != "complete"
                        or observations.get("status") != "complete"
                        or not isinstance(observation_rows, list) or len(observation_rows) != 4
                        or {(item["case_id"], item["policy"]) for item in observation_rows}
                        != expected_targets
                        or any(not isinstance(item["records_sha256"], str)
                               or len(item["records_sha256"]) != 64
                               or item["updates"] != [0, 1, 2]
                               or item["expected_steps"] != [20, 20, 20]
                               or not isinstance(item["observed_steps"], list)
                               or len(item["observed_steps"]) != 3
                               or item["observed_minibatches"] != item["observed_steps"]
                               for item in count_records)
                        or optimizer["first_training"]["observed_steps"] != [4, 4, 4]
                        or fresh["case"]["observed_steps"] != [4, 4, 4]
                        or fresh["reference"]["observed_steps"] != [20, 20, 20]
                        or optimizer["second_training"]["observed_steps"] != [20, 20, 20]
                        or not isinstance(observations["probe_physical_roles"], list)
                        or len(observations["probe_physical_roles"]) != 24
                        or observations["probe_physical_roles_count"] != 24
                        or observations["first_training_fixed_physical_roles"] != 8
                        or observations["second_training_fixed_physical_roles"] != 2
                        or observations["second_total_physical_roles"] != 26
                        or observations["R4_total_physical_roles"] != 34
                        or observations["second_model_requests"] != 6
                        or observations["continuation_new_model_or_probe_calls"] != 0
                        or observations["prior_failed_parent_wall_s"] != 10783.745328
                        or abs(observations["continuation_parent_wall_s"] - 1865.516) > 1e-6
                        or not isinstance(observations["probe_owned_wall_sum_s"], (int, float))):
                    raise ValueError("recovery observation contract 불일치")
                for item in observation_rows:
                    steps = item["steps"]
                    if (not isinstance(steps, list) or len(steps) > 3
                            or [step["step"] for step in steps] != list(range(1, len(steps) + 1))
                            or item["first_outcome"] != "not_recovered"
                            or item["final_outcome"] not in {"recovered", "not_recovered", "HOLD"}
                            or item["candidate_status"] not in {"validated_candidate", "HOLD"}
                            or any(step["probe"] not in {
                                "P_episode", "P_noise", "P_torque", "P_value",
                                "P_optimizer_updates"} or step["outcome"] not in {
                                    "normal", "abnormal"}
                                   or step["selected_by"] not in {"model", "fixed_order"}
                                   or not isinstance(step["fact_ids"], list)
                                   or not all(isinstance(fact, str) for fact in step["fact_ids"])
                                   or not isinstance(step["pair_result_sha256"], str)
                                   or len(step["pair_result_sha256"]) != 64
                                   or not isinstance(step["pair_parent_wall_s"], (int, float))
                                   or not isinstance(step["owned_wall_s"], (int, float))
                                   or step["pair_parent_wall_s"] < 0 or step["owned_wall_s"] < 0
                                   for step in steps)):
                        raise ValueError("recovery observation steps 불일치")
                if (sum(item["model_calls"] for item in observation_rows)
                        != observations["second_model_requests"]
                        or abs(sum(step["owned_wall_s"] for item in observation_rows
                                   for step in item["steps"])
                               - observations["probe_owned_wall_sum_s"]) > 1e-9):
                    raise ValueError("recovery observation 합계 불일치")
            except (KeyError, TypeError, ValueError):
                observations_source["status"] = "mismatch"
                observations_source["reason"] = "observation_contract_mismatch"
                sources.replays[observations_relative] = None
                observations = None
    public_rows = (cases or {}).get("rows", [])
    observations_by_target = {
        (item["case_id"], item["policy"]): item
        for item in (observations or {}).get("rows", [])}
    media_folder = slots.get("five_policy_recovery_media")
    recovery_video = (media_video(sources, media_folder) if media_folder is not None
                      else sources.file(None, kind="five_policy_recovery_media"))
    targets = (("case-03", "fixed_eligible"), ("case-03", "logs_only"),
               ("case-07", "fixed_eligible"), ("case-07", "logs_only"))
    pipeline_rows = []
    cost_rows = []
    for case_id, policy in targets:
        item = next((value for value in public_rows
                     if value.get("case_id") == case_id and value.get("policy") == policy), None)
        source_label = (f"{item['source_arm']} · seed {item['source_seed']}"
                        if item is not None else None)
        observed = observations_by_target.get((case_id, policy))
        probe_labels = {"P_episode": "episode", "P_noise": "noise", "P_torque": "torque",
                        "P_value": "value", "P_optimizer_updates": "updates"}
        probe_sequence = (" → ".join(
            f"{probe_labels[step['probe']]} {'이상' if step['outcome'] == 'abnormal' else '정상'}"
            for step in observed["steps"]) if observed is not None else None)
        pipeline_rows.append([
            f"{case_id} · {policy}",
            cell(sources, source_label, cases_source["id"], None),
            cell(sources, probe_sequence, observations_source["id"], None),
            cell(sources, item and item.get("first_outcome"), cases_source["id"], None),
            cell(sources, ("필요" if item and item.get("second_required") else "불필요")
                 if item else None, cases_source["id"], None),
            cell(sources, item and item.get("second_outcome"), cases_source["id"], None),
            cell(sources, item and item.get("final_outcome"), cases_source["id"], None),
        ])
        first = (item or {}).get("first_training")
        second_cost = (item or {}).get("second_diagnosis_cost")
        second = (item or {}).get("second_training")
        first_values = ([first.get("training_execution_wall_s"), first.get("fixed_execution_wall_s")]
                        if first else [None, None])
        first_value = (f"학습 {float(first_values[0]):.3f}s · 평가 {float(first_values[1]):.3f}s"
                       if all(value is not None for value in first_values) else None)
        diagnosis_values = ([second_cost.get("paired_probe_sdk_executions"),
                             second_cost.get("model_api_calls")] if second_cost else [None, None])
        diagnosis_value = (f"SDK {diagnosis_values[0]} · API {diagnosis_values[1]}"
                           if all(value is not None for value in diagnosis_values) else None)
        provider_tokens = ((second_cost.get("provider_tokens") or {}).get("total_tokens")
                           if second_cost else None)
        provider_wall = second_cost.get("provider_wall_s") if second_cost else None
        provider_value = (f"tokens {provider_tokens} · wall {float(provider_wall):.3f}s"
                          if provider_tokens is not None and provider_wall is not None else None)
        second_values = ([second.get("training_execution_wall_s"), second.get("fixed_execution_wall_s")]
                         if second else [None, None])
        second_value = (f"학습 {float(second_values[0]):.3f}s · 평가 {float(second_values[1]):.3f}s"
                        if all(value is not None for value in second_values) else None)
        cost_rows.append([
            f"{case_id} · {policy}",
            cell(sources, first_value, cases_source["id"], None),
            cell(sources, diagnosis_value, cases_source["id"], None),
            cell(sources, provider_value, cases_source["id"], None),
            cell(sources, second_value, cases_source["id"], None),
        ])
    lead = "공개 manifest와 파일 SHA가 확인된 경우에만 사전 선택한 네 recovery pipeline을 표시한다."
    if summary is not None:
        outcomes = summary["final_outcomes"]
        lead = (f"원래 source 계보 {summary['independent_source_lineages']}개에서 pipeline "
                f"{summary['pipeline_denominator']}개를 분리했다. 최종 결과는 회복 {outcomes['recovered']}, "
                f"미회복 {outcomes['not_recovered']}, HOLD {outcomes['HOLD']}다.")
    if summary is not None and observations is not None:
        counts = observations["optimizer_counts"]
        before = counts["fresh_three_update_inspection"]["case"]["observed_steps"]
        reference = counts["fresh_three_update_inspection"]["reference"]["observed_steps"]
        after = counts["second_training"]["observed_steps"]
        lead += (f" 회복 행의 첫 세 update는 case {before}, reference {reference}, "
                 f"교정 뒤 {after}로 기록됐다.")
    return {
        "id": "five_policy_recovery", "title": "사전 선택 네 pipeline: FIRST·SECOND·최종 회복", "dense": True,
        "lead": lead,
        "blocks": [
            {"type": "table", "wide": True, "title": "원래 source와 단계별 판정",
             "columns": ["pipeline", "원래 source", "SECOND 검사", "FIRST 결과", "SECOND", "SECOND 결과", "최종"],
             "rows": pipeline_rows},
            {"type": "table", "wide": True, "title": "단계별 실제 비용",
             "columns": ["pipeline", "FIRST 학습 · 평가", "SECOND SDK · API",
                         "SECOND provider 비용", "SECOND 학습 · 평가"],
             "rows": cost_rows},
            {"type": "video", "items": [{
                "label": ("Normal / Compound / After two registered corrections · 등록 실험 재생. "
                          "다른 세 pipeline은 HOLD이며 영상은 회복 판정을 새로 만들지 않는다."),
                "source": recovery_video["id"],
            }]},
        ],
        "notes": [
            "네 pipeline은 두 원래 source 계보에서 갈라진다. pipeline 4개와 독립 source 2개를 같은 분모로 세지 않는다.",
            "FIRST 결과, 필요한 경우의 SECOND 진단·학습, 최종 회복을 분리한다. not_recovered를 자동 재시도나 회복으로 바꾸지 않는다.",
            "4→20 검사는 같은 설정의 fresh 3-update 관측이다. 과거 checkpoint 실행값으로 추정하지 않는다.",
            "영상은 등록된 비교의 재생이며 모델 버전·VL 우위의 근거가 아니다. 회복 판정은 fixed26 영수증에서 가져온다.",
            "비용이 원문 receipt에 없으면 숫자를 채우지 않는다. 모델 우위·GPU 절감·등록 범위 밖 회복의 근거가 아니다.",
        ],
    }


def scene_stairs_motion(sources: Sources, slots: dict) -> dict:
    row, audit = manifest_result(sources, slots["stairs_motion_integrity"], manifest_name="manifest.json",
        manifest_schema="stairs_motion_integrity_public_manifest_v1",
        summary_schema="stairs_motion_integrity_batch_v1", summary_name="audit.json", kind="stairs_motion_integrity")
    by_clip = {item["clip_id"]: item for item in (audit or {}).get("rows", [])}
    rows = []
    labels = {"upright_completion_contradicted": "완료 해석과 자세 충돌",
              "original_failure_preserved": "기존 실패 유지",
              "no_inversion_observed_not_full_stability_proof": "뒤집힘 미관측"}
    for key, title in (("biped-stairs-up", "H1 상행"), ("biped-stairs-down", "H1 하행"),
                       ("quadruped-stairs-up-step-0p05", "Go2 상행 5cm"),
                       ("quadruped-stairs-down-step-0p05", "Go2 하행 5cm")):
        entry = by_clip.get(key, {})
        values = [entry.get("original_assessment", {}).get("outcome"),
                  entry.get("body_up_world_z", {}).get("min"),
                  entry.get("body_up_world_z", {}).get("negative_count"),
                  labels.get(entry.get("integrity_result"))]
        rows.append([title, *[cell(sources, value, row["id"], 3 if i == 1 else None)
                             for i, value in enumerate(values)]])
    return {"id": "stairs_motion_integrity", "title": "계단 영상 판정: 거리와 높이 외에 몸체 자세 확인",
        "lead": ("Go2 하행의 기존 completed 판정에서 12.26초부터 몸체 위쪽 축이 world 아래쪽을 향한 13표본을 찾았다."
                 if audit else "공개 manifest와 파일 SHA가 확인된 경우에만 기존 계단 영상의 자세 관측을 표시한다."),
        "blocks": [{"type": "table", "title": "기존 네 영상의 사후 CPU 감사",
                    "columns": ["영상", "기존 판정", "최소 up-z", "음수 표본", "추가 관측"], "rows": rows}],
        "notes": ["Go2 하행의 기존 결과는 보존하고 안정적 보행 시연 완료로 수용하지 않는다. 상행의 진행 실패도 유지한다.",
                  "몸체 위쪽 축의 world-z 내적을 사용했다. 음수 없음도 안정성 전체의 증명이 아니며, 낙상 원인은 확정하지 않았다.",
                  "공식 기학습 정책 재생의 사후 감사다. 모델이 발견한 결과나 RL Triage의 계단 교정·회복 성과가 아니다."]}


def scene_stairs_command(sources: Sources, slots: dict) -> dict:
    row, summary = manifest_result(sources, slots["go2_stairs_command"], manifest_name="manifest.json",
        manifest_schema="go2_stairs_command_contrast_manifest_v1",
        summary_schema="go2_stairs_command_contrast_public_v1", kind="go2_stairs_command")
    rows = []
    for direction, title in (("up", "상행"), ("down", "하행")):
        entry = next((item for item in (summary or {}).get("contrasts", []) if item["direction"] == direction), {})
        values = [entry.get("motion_accepted_0p25"), entry.get("motion_accepted_0p75"),
                  entry.get("initial_state_exact_match"), entry.get("checkpoint_terrain_runtime_exact_match")]
        rows.append([title, *[cell(sources, "통과" if value is True else "불수용" if value is False else None,
                                 row["id"], None) for value in values[:2]],
                     *[cell(sources, "동일" if value is True else None, row["id"], None) for value in values[2:]]])
    return {"id": "stairs_command_contrast", "title": "같은 Go2 정책: 명령 속도 변경 후 계단 상하행 통과",
        "lead": ("명령만 0.25에서 0.75m/s로 바꾼 재생에서 상행·하행 모두 진행·높이·자세 기준을 통과했다."
                 if summary else "등록 결과와 영상 SHA가 확인된 뒤 계단 명령 대조를 표시한다."),
        "blocks": [{"type": "table", "title": "seed 200 · 5cm 단차 · 공식 기학습 정책",
                    "columns": ["방향", "0.25m/s", "0.75m/s", "초기 상태", "정책·지형·runtime"], "rows": rows}],
        "notes": ["0.25m/s 상행 실패와 하행 뒤집힘을 재현·보존했다. 전체 네 조건 판정은 partial이며 두 0.75m/s 조건만 수용했다.",
                  "수용 기준: 625프레임·50fps·종료 증거, 진행·높이 통과, termination 없음, body-up-z 음수 0표본.",
                  "정책·seed·지형·초기 상태를 고정한 단일 계보 개발 재생이다. 새 학습·모델 판단·RL Triage 교정·회복 성과가 아니다."]}


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
              replay_d3.scene_d3(sources, slots), scene_nvidia(sources, slots),
              scene_prospective_comparison(sources, slots), scene_registered_partitions(sources, slots),
              scene_reasoning_followup(sources, slots),
              scene_four_way_development(sources, slots), scene_five_type_sources(sources, slots),
              scene_five_policy_diagnosis(sources, slots), scene_five_policy_recovery(sources, slots),
              scene_stairs_motion(sources, slots),
              scene_stairs_command(sources, slots),
              scene_cost(sources, slots)]
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
