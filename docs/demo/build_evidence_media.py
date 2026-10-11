"""동결된 공개 실행 기록을 검증한 뒤 근거 영상, GIF, 포스터를 만든다.

모델·Isaac·GPU를 호출하지 않는다. Python 표준 라이브러리, PowerShell의
System.Drawing, 설치된 ffmpeg/ffprobe만 사용한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / "docs" / "demo" / "active-diagnosis-v2.json"
DEFAULT_OUTPUT = ROOT / "docs" / "media" / "evidence-v2-20261011"
DEFAULT_BASELINE_VIDEO = ROOT / "docs" / "media" / "baseline_s42_play.mp4"
DEFAULT_FAILURE_VIDEO = ROOT / "docs" / "media" / "c04_s42_play.mp4"
DEFAULT_BRIDGE = ROOT / "evals" / "results" / "bridge_smoke.json"
DEFAULT_RELABEL = ROOT / "evals" / "results" / "p0b1_relabel_20261008" / "relabel.json"
FROZEN_COMMIT = "d0a9bfe51a86ca0f837ef23b2b06133ddadb9053"
PUBLIC_CARD_NAMES = tuple(f"{index:02d}_{name}.png" for index, name in enumerate((
    "title", "reference_overlay", "inspect", "measure", "decision", "audit", "recovery", "control"
)))


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedEvidence:
    snapshot: dict[str, Any]
    audit: dict[str, Any]
    selected: dict[str, Any]
    primary_count: int
    source_hashes: dict[str, str]
    recovery: dict[str, Any]
    frozen_commit: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def git_blob_sha256_lf(root: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=root, capture_output=True
    )
    if completed.returncode:
        raise EvidenceError(f"동결 커밋에 code source가 없습니다: {relative}")
    return hashlib.sha256(completed.stdout.replace(b"\r\n", b"\n")).hexdigest()


def _repo_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise EvidenceError(f"저장소 밖 source 경로: {relative}") from exc
    return path


def select_scene(snapshot: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    cases = {case["case"]: case for case in snapshot.get("cases", [])}
    audited = {case["case"]: case for case in audit.get("cases", [])}
    order = snapshot.get("registered_order", [])
    if not order or set(order) != set(cases):
        raise EvidenceError("registered_order와 snapshot 사례 목록이 맞지 않습니다")
    for case_name in order:
        case = cases[case_name]
        check = audited.get(case_name, {})
        if case.get("state") == "verified" and case.get("proof") is True and check.get("valid") is True and check.get("proof_changed") is True:
            return case
    return cases[order[0]]


def _source_file(root: Path, source_hashes: dict[str, str], relative: str) -> Path:
    path = _repo_path(root, relative)
    if not path.is_file() or source_hashes.get(relative) != sha256_lf(path):
        raise EvidenceError(f"필수 결속 파일이 source manifest와 다릅니다: {relative}")
    return path


def _verify_receipts(root: Path, tag: str, case: str, result: dict[str, Any], source_hashes: dict[str, str]) -> None:
    checkpoint = result.get("checkpoint_sha256")
    fields = ("at", "event", "request_id", "outcome", "measurement", "gpu_s", "exit_code", "over_budget")
    receipts = result.get("observed", [])
    if not receipts:
        return
    ledger_relative = f"evals/results/{tag}/cases/{case}/ledger.jsonl"
    ledger_path = _source_file(root, source_hashes, ledger_relative)
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for receipt in receipts:
        try:
            canonical = {field: receipt[field] for field in fields}
        except KeyError as exc:
            raise EvidenceError(f"receipt 필드 누락: {exc.args[0]}") from exc
        if canonical_digest(canonical) != receipt.get("receipt_digest"):
            raise EvidenceError(f"receipt digest 불일치: {receipt.get('request_id')}")
        if receipt.get("checkpoint_sha256") != checkpoint:
            raise EvidenceError(f"receipt checkpoint 불일치: {receipt.get('request_id')}")
        probe, request_id = receipt.get("probe"), receipt.get("request_id")
        if not isinstance(probe, str) or not isinstance(request_id, str):
            raise EvidenceError("receipt의 probe/request_id가 잘못되었습니다")
        base = f"evals/results/{tag}"
        report_relative = f"{base}/probes/{case}__{probe}.json"
        execution_relative = f"{base}/execution/{case}__{probe}.json"
        resources_relative = f"{base}/resources/{case}__{probe}.json"
        report_path = _source_file(root, source_hashes, report_relative)
        execution_path = _source_file(root, source_hashes, execution_relative)
        resources_path = _source_file(root, source_hashes, resources_relative)
        report, execution, resources = load_json(report_path), load_json(execution_path), load_json(resources_path)
        expected_report_sha = source_hashes[report_relative]
        cell = report.get("probes", {}).get(probe, {})
        checks = (
            receipt.get("report_sha256_lf") == expected_report_sha,
            report.get("name") == case,
            report.get("checkpoint", {}).get("sha256") == checkpoint,
            cell.get("measurement") == receipt.get("measurement"),
            cell.get("exit") == receipt.get("exit_code"),
            execution.get("case") == resources.get("case") == case,
            execution.get("probe") == resources.get("probe") == probe,
            execution.get("checkpoint_sha256") == resources.get("checkpoint_sha256") == checkpoint,
            execution.get("report_sha256_lf") == resources.get("report_sha256_lf") == expected_report_sha,
            execution.get("gpu_wall_s") == execution.get("execution_wall_s") == receipt.get("gpu_s"),
            execution.get("exit_code") == resources.get("launcher_exit_code") == receipt.get("exit_code"),
        )
        if not all(checks):
            raise EvidenceError(f"receipt report/execution 결속 불일치: {request_id}")
        request_events = [row for row in ledger if row.get("request_id") == request_id]
        if [row.get("event") for row in request_events] != ["proposed", "approved", "consumed", "receipt"]:
            raise EvidenceError(f"receipt 원장 순서 불일치: {request_id}")
        if request_events[0].get("probe") != probe or request_events[-1] != canonical:
            raise EvidenceError(f"receipt 요청/원장 결속 불일치: {request_id}")


def _verify_recovery(bridge_path: Path, relabel_path: Path) -> dict[str, Any]:
    bridge = load_json(bridge_path)
    relabel = load_json(relabel_path)
    bridge_runs = {(row["case_id"], row["seed"], row["suspected"]): row for row in bridge}
    correct = bridge_runs.get(("c01", 42, "ch3"))
    benign = bridge_runs.get(("c01", 42, "ch1"))
    runs = {row["run"]: row for row in relabel.get("runs", [])}
    recovered = runs.get("c01_revert_ch3_s42")
    unresolved = runs.get("c01_revert_ch1_s42")
    if not correct or correct.get("recovered") is not True or not benign or benign.get("recovered") is not False:
        raise EvidenceError("bridge_smoke의 c01 원복 결과가 예상 계약과 다릅니다")
    if not recovered or recovered.get("behavior_label") != "healthy" or recovered.get("current_recovered") is not True:
        raise EvidenceError("재학습 회복 사례의 판정이 맞지 않습니다")
    if not unresolved or unresolved.get("behavior_label") != "undetermined" or unresolved.get("short_horizon") is not True:
        raise EvidenceError("무해 변경 원복 사례의 1초 horizon 판정이 맞지 않습니다")
    return {
        "bridge_sha256_lf": sha256_lf(bridge_path),
        "relabel_sha256_lf": sha256_lf(relabel_path),
        "correct": correct,
        "benign": benign,
        "recovered": recovered,
        "unresolved": unresolved,
    }


def verify_evidence(root: Path, snapshot_path: Path, bridge_path: Path, relabel_path: Path, frozen_commit: str = FROZEN_COMMIT) -> VerifiedEvidence:
    snapshot = load_json(snapshot_path)
    if snapshot.get("schema") != "active_replay_summary_v2":
        raise EvidenceError("지원하지 않는 replay schema입니다")
    sources = snapshot.get("sources")
    if not isinstance(sources, list) or not sources:
        raise EvidenceError("source manifest가 비어 있습니다")
    source_hashes: dict[str, str] = {}
    for item in sources:
        relative, expected = item.get("path"), item.get("sha256_lf")
        if not isinstance(relative, str) or relative in source_hashes:
            raise EvidenceError(f"중복되거나 잘못된 source 경로: {relative!r}")
        path = _repo_path(root, relative)
        if not path.is_file() or sha256_lf(path) != expected:
            raise EvidenceError(f"source LF SHA-256 불일치: {relative}")
        source_hashes[relative] = expected

    audit_relative = snapshot.get("audit")
    if audit_relative not in source_hashes:
        raise EvidenceError("감사 파일이 source manifest에 없습니다")
    audit = load_json(_repo_path(root, audit_relative))
    if audit.get("schema") != "active_diagnosis_audit_v2" or audit.get("valid") is not True:
        raise EvidenceError("동결 감사가 유효하지 않습니다")
    if audit.get("tag") != snapshot.get("tag"):
        raise EvidenceError("감사 tag와 replay tag가 다릅니다")

    protocol_hash = audit.get("protocol_sha256_lf")
    protocol_sources = [path for path, digest in source_hashes.items() if path.startswith("bench/protocols/") and digest == protocol_hash]
    if len(protocol_sources) != 1:
        raise EvidenceError("동결 protocol 결속이 맞지 않습니다")
    protocol = load_json(_repo_path(root, protocol_sources[0]))
    code_sources = protocol.get("code_sha256_lf")
    if not isinstance(code_sources, dict) or not code_sources:
        raise EvidenceError("protocol code source 결속이 비어 있습니다")
    for relative, expected in code_sources.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or git_blob_sha256_lf(root, frozen_commit, relative) != expected:
            raise EvidenceError(f"동결 커밋 code source 불일치: {relative}")
    audited = {case["case"]: case for case in audit.get("cases", [])}
    cases = {case["case"]: case for case in snapshot.get("cases", [])}
    if set(audited) != set(cases):
        raise EvidenceError("감사 사례와 replay 사례가 다릅니다")
    for case_name, case in cases.items():
        result_relative = f"evals/results/{snapshot['tag']}/cases/{case_name}/result.json"
        check = audited[case_name]
        if check.get("valid") is not True or source_hashes.get(result_relative) != check.get("result_sha256_lf"):
            raise EvidenceError(f"감사와 result 결속이 맞지 않습니다: {case_name}")
        result = load_json(_repo_path(root, result_relative))
        if result != case.get("result"):
            raise EvidenceError(f"replay에 내장된 result가 원본과 다릅니다: {case_name}")
        if result.get("case") != case_name or result.get("tag") != snapshot.get("tag") or result.get("protocol_sha256_lf") != protocol_hash:
            raise EvidenceError(f"result 식별자 결속이 맞지 않습니다: {case_name}")
        _verify_receipts(root, snapshot["tag"], case_name, result, source_hashes)

    selected = select_scene(snapshot, audit)
    primary_count = sum(case.get("proof") is True and audited[case["case"]].get("proof_changed") is True for case in cases.values())
    recovery = _verify_recovery(bridge_path, relabel_path)
    return VerifiedEvidence(snapshot, audit, selected, primary_count, source_hashes, recovery, frozen_commit)


def card_spec(evidence: VerifiedEvidence) -> dict[str, Any]:
    snapshot, selected, recovery = evidence.snapshot, evidence.selected, evidence.recovery
    result = selected["result"]
    decision = result.get("decisions", [{}])[0]
    receipt = result.get("observed", [{}])[0]
    total = len(snapshot["registered_order"])
    hold_count = sum(case["result"].get("status") == "hold" for case in snapshot["cases"])
    recovered, unresolved = recovery["recovered"], recovery["unresolved"]
    return {"cards": [
        {"filename": "00_title.png", "eyebrow": "동결 실행 기록 · 공개 근거", "title": "실패에서 다음 검사까지", "body": ["1 실패 관찰  →  2 검사 선택  →  3 실제 측정", "4 판단·보류  →  5 별도 개선 확인", f"모델 사례 {total}건 기록 감사 통과 · 판단 갱신 증거 {evidence.primary_count}/{total}"], "badge": f"PRIMARY {evidence.primary_count}/{total}", "badge_kind": "danger", "accent": "blue", "footer": "seen seed 2027 통합 기록 · 정확도·일반화 평가 아님"},
        {"filename": "01_reference_overlay.png", "layout": "reference_overlay", "eyebrow": "과거 100 iter 정책 재생", "title": "기존 기준과 결함 장면", "body": [], "badge": "참고 장면", "badge_kind": "info", "accent": "blue", "footer": "V2 선택 사례나 회복 후 장면이 아닙니다. 두 영상은 참고용 실제 재생입니다."},
        {"filename": "02_inspect.png", "eyebrow": f"등록 첫 사례 · {selected['case']}", "title": "AI가 보상 검사를 선택", "body": ["검사 전 가장 의심한 원인: 보상", "승인 후 같은 정책으로 보상 계산을 검사합니다.", "저장된 실제 모델 판단을 재생합니다."], "evidence": f"raw: {decision.get('probe', '검사 없음')} · checkpoint {result.get('checkpoint_sha256')}", "badge": "AI 검사 선택", "badge_kind": "info", "accent": "blue", "footer": f"동결 기록 재생 · commit {evidence.frozen_commit[:7]} · case {selected['case']}"},
        {"filename": "03_measure.png", "eyebrow": "실제 검사 receipt", "title": "보상 계산 검사는 정상", "body": ["예상 보상과 기록된 보상이 일치했습니다.", f"선속도 보상 상대 오차  {receipt.get('measurement', {}).get('track_lin_vel_xy_exp_rel_error'):.4g}", f"각속도 보상 상대 오차  {receipt.get('measurement', {}).get('track_ang_vel_z_exp_rel_error'):.4g}", f"launcher 기록 {receipt.get('gpu_s')}초 · exit code {receipt.get('exit_code')}"], "evidence": f"raw: track_lin_vel_xy_exp_rel_error / track_ang_vel_z_exp_rel_error · request {receipt.get('request_id')}", "badge": "실제 측정", "badge_kind": "success", "accent": "green", "footer": "receipt, report, checkpoint 해시를 함께 검증했습니다."},
        {"filename": "04_decision.png", "eyebrow": "판단과 후속 행동", "title": "검사와 어긋난 판단은 보류", "body": ["검사 결과와 모델 판단이 맞지 않아 원인을 확정하지 않았습니다.", "다음 검사를 실행하기 전 판단을 다시 검증해야 합니다."], "evidence": f"raw: {result.get('stop_reason')} · unique support / raw ranking 1위 조건 불충족", "badge": "보류", "badge_kind": "warning", "accent": "orange", "footer": f"선택 사례 {selected['case']} · 진단 성공이나 원인 확정으로 세지 않음"},
        {"filename": "05_audit.png", "eyebrow": "독립 감사 결과", "title": f"기록 검증 {total}/{total} · 판단 변화 {evidence.primary_count}/{total}", "body": [f"모델 실행: 완료 {total - hold_count}건 · 보류 {hold_count}건", "검사 때문에 판단이 바뀌었다는 근거는 아직 없습니다.", "정책 회복은 이 진단 사례에서 측정하지 않았습니다."], "badge": "과장 없이 표시", "badge_kind": "danger", "accent": "red", "footer": "판단 변화·정답·회복을 서로 다른 결과로 기록"},
        {"filename": "06_recovery.png", "eyebrow": "별도 원복·재학습 검증 · 다른 case/run", "title": "원복 후 생존 구간 회복", "body": [f"c01_revert_ch3_s42: episode ratio {recovery['correct']['ratios']['Train/mean_episode_length']}", f"survival {recovered['behavior']['survival_s']:.4f}/{recovered['behavior']['horizon_s']:.0f}초 · fall fraction {recovered['behavior']['fall_frac']:.4f}", f"행동 판정 {recovered['behavior_label']} · active 사례와 같은 사건이 아닙니다.", "수치는 확인했으며, 로봇 회복 장면은 아직 없습니다."], "badge": "별도 측정", "badge_kind": "success", "accent": "green", "footer": "bridge_smoke.json + recovery relabel.json"},
        {"filename": "07_control.png", "eyebrow": "별도 대조 · 무해 변경 원복", "title": "1초 관찰로는 회복 여부 미결정", "body": ["무해한 변경만 되돌린 별도 대조 실험", "에피소드 길이: 기준의 5% · 행동 관찰: 1초", "관찰 시간이 짧아 행동 개선 여부는 미결정입니다."], "evidence": f"raw: c01_revert_ch1_s42 · horizon_ratio {unresolved['compared']['horizon_ratio']} · behavior_label {unresolved['behavior_label']}", "badge": "미결정", "badge_kind": "warning", "accent": "orange", "footer": "원복 실패와 행동 불건강은 같은 판정이 아닙니다."},
        {"filename": "poster.png", "eyebrow": "RL Triage Agent · 근거 영상", "title": "실패 → 검사 → 측정 → HOLD", "body": [f"active 판단 갱신 증거 {evidence.primary_count}/{total}", f"등록 첫 사례 {selected['case']}: {result.get('status', '').upper()}", "별도 재학습 회복 수치는 확인, 회복 영상은 미확보"], "badge": "검증된 범위만", "badge_kind": "info", "accent": "blue", "footer": "실제 재생 · 동결 기록 재생 · 미측정을 화면에서 구분"},
    ]}


def run(command: list[str], cwd: Path = ROOT) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"명령 실패 ({completed.returncode}): {' '.join(command)}\n{completed.stderr[-4000:]}")


def _ps_quote(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _ffmpeg_card(card: Path, duration_s: int, target: Path) -> None:
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-loop", "1", "-framerate", "30", "-i", str(card), "-t", str(duration_s), "-vf", "format=yuv420p", "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(target)])


def _ffmpeg_reference(baseline: Path, failure: Path, overlay: Path, duration_s: int, target: Path) -> None:
    filters = (
        f"color=white:s=1280x720:d={duration_s}[bg];"
        "[0:v]scale=590:300:force_original_aspect_ratio=decrease,pad=590:300:(ow-iw)/2:(oh-ih)/2:white[a];"
        "[1:v]scale=590:300:force_original_aspect_ratio=decrease,pad=590:300:(ow-iw)/2:(oh-ih)/2:white[b];"
        "[bg][a]overlay=30:225[x];[x][b]overlay=660:225[y];[y][2:v]overlay=0:0,format=yuv420p[v]"
    )
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-stream_loop", "-1", "-i", str(baseline), "-stream_loop", "-1", "-i", str(failure), "-loop", "1", "-i", str(overlay), "-t", str(duration_s), "-filter_complex", filters, "-map", "[v]", "-an", "-r", "30", "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(target)])


def _probe_media(path: Path) -> dict[str, Any]:
    completed = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=codec_name,width,height,avg_frame_rate", "-of", "json", str(path)], text=True, encoding="utf-8", capture_output=True, check=True)
    return json.loads(completed.stdout)


def verify_artifact_manifest(output: Path, required_names: set[str] | None = None) -> None:
    manifest = load_json(output / "manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise EvidenceError("artifact manifest가 JSON 객체가 아닙니다")
    required = required_names or (set(PUBLIC_CARD_NAMES) | {"poster.png", "evidence.mp4", "evidence-summary.gif"})
    if not required <= set(artifacts):
        raise EvidenceError("필수 공개 artifact가 manifest에서 빠졌습니다")
    for name, item in artifacts.items():
        path = output / name
        if not path.is_file() or path.stat().st_size != item.get("bytes") or hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256"):
            raise EvidenceError(f"artifact hash/size 불일치: {name}")


def render(evidence: VerifiedEvidence, output: Path, baseline_video: Path, failure_video: Path, snapshot_path: Path, bridge_path: Path, relabel_path: Path) -> None:
    if output.exists():
        raise FileExistsError(f"출력 폴더가 이미 있습니다: {output}")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="evidence-media-") as temp_name:
        temp = Path(temp_name)
        spec_path = temp / "cards.json"
        spec_path.write_text(json.dumps(card_spec(evidence), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        script = ROOT / "docs" / "demo" / "render_evidence_cards.ps1"
        bootstrap = (
            f"& ([scriptblock]::Create((Get-Content -Raw -Encoding UTF8 -LiteralPath {_ps_quote(script)}))) "
            f"-SpecPath {_ps_quote(spec_path)} -OutputDir {_ps_quote(output)}"
        )
        run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", bootstrap])

        segments = []
        sequence = [("00_title.png", 6), ("reference", 9), ("02_inspect.png", 7), ("03_measure.png", 7), ("04_decision.png", 7), ("05_audit.png", 7), ("06_recovery.png", 7), ("07_control.png", 5)]
        for index, (name, duration_s) in enumerate(sequence):
            segment = temp / f"segment-{index:02d}.mp4"
            if name == "reference":
                _ffmpeg_reference(baseline_video, failure_video, output / "01_reference_overlay.png", duration_s, segment)
            else:
                _ffmpeg_card(output / name, duration_s, segment)
            segments.append(segment)
        concat_list = temp / "concat.txt"
        concat_list.write_text("".join(f"file '{path.as_posix()}'\n" for path in segments), encoding="utf-8", newline="\n")
        video = output / "evidence.mp4"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", "-movflags", "+faststart", str(video)])

        gif = output / "evidence-summary.gif"
        gif_filter = (
            "[0:v]trim=start=6:end=9,setpts=PTS-STARTPTS[a];"
            "[0:v]trim=start=15:end=18,setpts=PTS-STARTPTS[b];"
            "[0:v]trim=start=22:end=25,setpts=PTS-STARTPTS[c];"
            "[0:v]trim=start=29:end=32,setpts=PTS-STARTPTS[d];"
            "[0:v]trim=start=43:end=46,setpts=PTS-STARTPTS[e];"
            "[a][b][c][d][e]concat=n=5:v=1:a=0,fps=10,scale=640:-1:flags=lanczos,split[x][y];[x]palettegen=max_colors=128[p];[y][p]paletteuse=dither=bayer"
        )
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(video), "-filter_complex", gif_filter, str(gif)])

    artifacts = {}
    artifact_paths = [video, gif, output / "poster.png"] + [output / name for name in PUBLIC_CARD_NAMES]
    for path in artifact_paths:
        info: dict[str, Any] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        if path.suffix.lower() in {".mp4", ".gif"}:
            probe = _probe_media(path)
            info["duration_s"] = round(float(probe["format"]["duration"]), 3)
            stream = probe["streams"][0]
            info.update({"width": stream["width"], "height": stream["height"], "codec": stream["codec_name"]})
        artifacts[path.name] = info

    selected = evidence.selected
    manifest = {
        "schema": "evidence_media_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_replay_commit": evidence.frozen_commit,
        "selected_case": selected["case"],
        "selection": "first_registered_primary" if selected.get("proof") else "first_registered_honest_hold_fallback",
        "primary_verified": evidence.primary_count,
        "primary_total": len(evidence.snapshot["registered_order"]),
        "chapters": [
            {"name": "기존 기준/결함 재생", "kind": "actual_video_replay", "scope": "legacy_100_iter_reference_only", "recovery_video": False},
            {"name": "V2 active 진단", "kind": "frozen_record_replay", "scope": evidence.snapshot["scope"], "recovery_measured": False},
            {"name": "별도 원복·재학습 검증", "kind": "measured_separate_run", "same_case_as_active": False, "recovery_video": False},
        ],
        "sources": {
            "active_snapshot": {"path": snapshot_path.relative_to(ROOT).as_posix(), "sha256_lf": sha256_lf(snapshot_path), "verified_entries": len(evidence.source_hashes)},
            "bridge_smoke": {"path": bridge_path.relative_to(ROOT).as_posix(), "sha256_lf": evidence.recovery["bridge_sha256_lf"]},
            "recovery_relabel": {"path": relabel_path.relative_to(ROOT).as_posix(), "sha256_lf": evidence.recovery["relabel_sha256_lf"]},
            "baseline_video": {"path": baseline_video.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(baseline_video.read_bytes()).hexdigest(), "classification": "actual_replay_reference_only"},
            "failure_video": {"path": failure_video.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(failure_video.read_bytes()).hexdigest(), "classification": "actual_replay_failure_only"},
        },
        "artifacts": artifacts,
        "disclosures": ["V2 active 사례의 회복은 미측정", "별도 재학습 회복은 수치로 확인했지만 회복 로봇 영상은 미확보", "판단 갱신, 정답, 회복은 서로 다른 판정"],
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    verify_artifact_manifest(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-video", type=Path, default=DEFAULT_BASELINE_VIDEO)
    parser.add_argument("--failure-video", type=Path, default=DEFAULT_FAILURE_VIDEO)
    parser.add_argument("--frozen-commit", default=FROZEN_COMMIT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    for executable in ("ffmpeg", "ffprobe", "powershell"):
        if shutil.which(executable) is None:
            raise SystemExit(f"필수 실행 파일을 찾지 못했습니다: {executable}")
    commit = subprocess.run(["git", "rev-parse", "--verify", f"{args.frozen_commit}^{{commit}}"], cwd=ROOT, text=True, encoding="utf-8", capture_output=True, check=True).stdout.strip()
    evidence = verify_evidence(ROOT, args.snapshot.resolve(), DEFAULT_BRIDGE, DEFAULT_RELABEL, commit)
    if args.verify_only:
        print(json.dumps({"valid": True, "selected_case": evidence.selected["case"], "primary": f"{evidence.primary_count}/{len(evidence.snapshot['registered_order'])}", "sources": len(evidence.source_hashes)}, ensure_ascii=False))
        return 0
    render(evidence, args.output.resolve(), args.baseline_video.resolve(), args.failure_video.resolve(), args.snapshot.resolve(), DEFAULT_BRIDGE, DEFAULT_RELABEL)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
