"""B4 영상 관찰 병목의 프레임 준비와 순수 요청/응답 어댑터.

이 모듈은 API를 호출하지 않는다. 평가 영상 18개에서 CPU ffmpeg로 같은 규칙의
JPEG 여섯 장을 만들고, 호출기가 사용할 메시지와 검증 결과를 구성한다. 영상
관찰은 고정 평가를 대신하지 않는 보조 기록이다.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = ROOT / "bench/private/codex_support_20261015/task19/v2/prepared_assets_bound"
DEFAULT_MANIFEST = DEFAULT_ASSET_ROOT / "manifest.json"
FRAME_COUNT = 6
MAX_JPEG_BYTES = 180_000
ENUM = {"yes", "no", "unclear"}

# id, clip, fixed-eval group, standstill ratio, fall rate, checkpoint, clip sha, report, report sha
_ROWS = (
    ("motor_before", "docs/media/go2-correction-motor_resume_dev_20261013/before_original.mp4", "failure", .9066, .0385, "70ab48fb4d073b2ccd7e28fe8a69863393bfc67f26074790eaa2cab94a03e45a", "3e424cb64760d71ec7cf0536d3dd426bedfbd61afb8e998f82e2b9c82d9983cd", "evals/results/motor_resume_dev_20261013/original_reports/fault.json", "66e32c00f0e2d3254e62e5b3d2760df5b1a5a0d3b1e929325c2d27bb362b4854"),
    ("motor_corrected", "docs/media/go2-correction-motor_resume_dev_20261013/after_corrected.mp4", "normal", .1601, 0.0, "6cc5b4610e98b772e011e398621dc501bdc94141d61bd4455820ba0e84344dd6", "06f7952cd400af44271f5f35df52e002cec8b37b611a6824c705feede1fbf401", "evals/results/motor_resume_dev_20261013/reports/corrected.json", "53b5fa7d56442dd9f6e67d6e260738838c726bd904e34e30b1c730225ec8c597"),
    ("reward_dev_before", "docs/media/go2-correction-reward_resume_dev3_20261013/before_original.mp4", "failure", 1.0122, .0692, "3814f73d8104c72f92a4ed368af459af9ee7f26353c2c00cc91f123384578f14", "e417e9093b37d896d7e80c576de6945479a9b584266df10faf3e8b297f70eccf", "evals/results/p0c_fixed_eval_20261009/runs/h02_s2027.json", "7668039f7f21f55500a41b27336d92e0c3eaaa2e11c099e0c3d43afbfdb1481c"),
    ("reward_dev_corrected", "docs/media/go2-correction-reward_resume_dev3_20261013/after_corrected.mp4", "failure", .9866, 0.0, "eb33ebec393f0c62f7511d05fd794610a0cfc560b61f6d3ea9e0870b87954020", "0ec50b01311313cef0721d462839aacaade9e8709f3f8d674ead088b0c9fe1e6", "evals/results/reward_resume_dev3_20261013/reports/corrected.json", "3c3af3990f883fb9ecb98104426742d3d26b0067069093b153d1a6838123300e"),
    ("reward_dev_sham", "docs/media/go2-correction-reward_resume_dev3_20261013/after_sham.mp4", "failure", 1.0201, 0.0, "ef08b538466392c90b40604c56f744f3ea53693191528983c48763bd68f4493a", "3574d5d474a86b5879168244ed0bb038a6d5aa05643ff6aa167438af5bf2efff", "evals/results/reward_resume_dev3_20261013/reports/fault_sham.json", "d3a71f0d167fe3851b3113781b2f7d91a92f81f02d6f26651e0f35d901330d00"),
    ("reward_2026_before", "docs/media/go2-correction-reward_v4_s2026_20261013/before_original.mp4", "failure", .9967, 0.0, "d1621a289b31d883f2d0da825689ecd10998ee2d3a53c8c4cf2d887fb866546c", "d2e7c4689ed1ee15a6723a20dbc2c990e1a2b2a9ca4790f28c774f79434484e5", "evals/results/p0c_fixed_eval_20261009/runs/h02_s2026.json", "f7e2ebab9a78488e7ef37e8a90e415b93ee095242b92163edade06a5cb283827"),
    ("reward_2026_resume", "docs/media/go2-correction-reward_v4_s2026_20261013/after_resume_corrected.mp4", "failure", .9823, 0.0, "37ff93769014b9abd0008729e92ae10f4fa950bc1e8b24155ab9376385fb6ec4", "bc89df09d359179be988b8929446c33178f033611223197a365f6b0076fbddf0", "evals/results/reward_v4_s2026_20261013/reports/resume_corrected.json", "e542905533ed616a9b22b2d249329367cf3bcce791f9b61ee2f46b79161af1cf"),
    ("reward_2026_std", "docs/media/go2-correction-reward_v4_s2026_20261013/after_std_reset.mp4", "failure", .9477, .001, "17f9cac434193a0b14647e352ac1276ff3098699f86ecdc4c7cd4c728981bf92", "b59187d83f3323a9a60489e9f4928e205d05647a818beb091cf89445e38fb34f", "evals/results/reward_v4_s2026_20261013/reports/resume_corrected_std_reset.json", "b7fae39c7651a44e7de0cf1a59e1c70d928ca8d66ca3468f947dcb604cfba07d"),
    ("reward_2026_none", "docs/media/go2-correction-reward_v4_s2026_20261013/reference_original_none.mp4", "normal", .1256, 0.0, "2e1e84cdbe5b348d1b6c3fe99c4894c95029ab5a18d3ebde69cd1e1743e80b2f", "65f01dcdc100799b86431e8370c941996456def5969c6301d788fa0a682036f7", "evals/results/motor_w1_s2026_20261013/original_reports/clean_reference.json", "c8855549b0541fe12a7d612f3e04eea5a285dee12d0ce69fbdb4b43ee4b884cf"),
    ("reward_2026_clean_resume", "docs/media/go2-correction-reward_v4_s2026_20261013/reference_clean_resume.mp4", "normal", .1218, .0385, "08f037318cfc0ee0c41a87a62ecf870b5633f53bbb7b480cbfe66b78bbe0695f", "07df564b84a855967cb5a8e18677abcd3b1b33d0aa8c62a5e90ff295edf85a3b", "evals/results/motor_w1_s2026_20261013/reports/clean_resume_reference.json", "b92b4334e26a17950ec302a4b9defe11e285e6a0f8c0f65ff7cc6d71a42789d3"),
    ("reward_2028_before", "docs/media/go2-correction-reward_v4_s2028_20261013/before_original.mp4", "failure", .9889, 0.0, "e63f4b0cddb0213e1f124ee6f0349218237511af0bdf0e75e12386b9d1856b2e", "602adc020d5ad7d37db7c1a40122814a79cdd36135419df89a6b95f112420dab", "evals/results/p0c_fixed_eval_20261009/runs/h02_s2028.json", "9a70f99717c5107245787812bfb4ff6276a345c7a94635ecd3a90cb43d38e5e3"),
    ("reward_2028_resume", "docs/media/go2-correction-reward_v4_s2028_20261013/after_resume_corrected.mp4", "failure", .9664, 0.0, "3c5908e677cd489a1a2cd36cb85690e5a67bbe8d4269e35c3c9183baa2bea6a4", "aaafaf5890028bb4d05453111892678c324f2aef39eab688fce2b590f88713ef", "evals/results/reward_v4_s2028_20261013/reports/resume_corrected.json", "1d28a2d262b7a892e55065fb85a5f9bf69b8157a6983db075827690627b899f8"),
    ("reward_2028_std", "docs/media/go2-correction-reward_v4_s2028_20261013/after_std_reset.mp4", "failure", .9814, 0.0, "155e7ddc9c01030516c8aef419a95ad8c7bf495490d90f171b4a5b4f97210672", "ffc5637bc6c1879dedbaec7030deec79f9a852e457883cc03c98ba78ff1b7ec3", "evals/results/reward_v4_s2028_20261013/reports/resume_corrected_std_reset.json", "bc4141149ccda7c50fab3fcaf990d8c1cc6391e51d7d4062fbba7f2ee92f75ac"),
    ("reward_2028_none", "docs/media/go2-correction-reward_v4_s2028_20261013/reference_original_none.mp4", "normal", .1195, 0.0, "d8c67eec0b8108b7e7092a80a3efcdc0338285c06c87ca8b9f7b8c65641024e3", "dba0c5549370f78a62459d42c325bb5c2dd50f4abbfc564a3fa12ca0306c6e0c", "evals/results/motor_w1_s2028_20261013/original_reports/clean_reference.json", "40617eb003b5316b3bf86fc7767ca9b6a9aaeee96cc5671ba6cb8b6e4587bdcc"),
    ("reward_2028_clean_resume", "docs/media/go2-correction-reward_v4_s2028_20261013/reference_clean_resume.mp4", "normal", .1128, .0385, "f0aacea12b8cd64b6ff5d69657580e8ed40987270cb5bcdbbbcf9e1c0be92c77", "615b4ab8f8f071fbf0636d55a38fc14be3bb7e0f539962f816ca849baf77356a", "evals/results/motor_w1_s2028_20261013/reports/clean_resume_reference.json", "83e4352f8d177ad41d358776ca104e36d304e69544b19df39933e8eeff99ceb1"),
    ("demo_2030_fault", "docs/media/go2-e2e-e2e_reward_s2030_20261013/before_fault.mp4", "failure", 1.0176, .0231, "6133080cf77710da55c157ba05109c818787cf3c906778b320b2073947dddc39", "c87ebe2ca6084781f82d41aa491ca4ce5e2bfb748979155d13faefbc812c4e27", "evals/results/e2e_reward_s2030_20261013/runs/h02_s2030.json", "a0771acdcd80da76d1e723042d190e90d877338a268998f779a3048b9ed2f5dc"),
    ("demo_2030_none", "docs/media/go2-e2e-e2e_reward_s2030_20261013/reference_clean.mp4", "normal", .1247, 0.0, "2feae4b805a9e5ed37ab29e99b8ba1b348e05837eeccb1ddd42d6cfeb1b1d7a1", "7c72ecc10763fbad6c86f59b5b14488867973bbb5a8f6740fb341b0bfc6d63d9", "evals/results/e2e_reward_s2030_20261013/runs/baseline_p0c_s2030.json", "3328f38043d99b9c80f1c12b41c2485cb2fffeb0f8ca073c88af57f619548716"),
    ("demo_2030_restart", "docs/media/go2-e2e-e2e_reward_s2030_20261013/after_restart_fix.mp4", "normal", .12544545755336192, None, "e1b06d6d320023473bc130a6fd7fa5730395e16b595e2f512855b37a33c63b4b", "3edf9e248bed4abf9934ba8902383ec6f10eec9639f263dc032adf658196d759", "evals/results/e2e_reward_s2030_20261013/demo_result.json", "fd1a72b73928ad85e3b12b5ee9053957c03529b8e9550fb406016d5a5497b351"),
)

PREFLIGHT = {
    "id": "preflight_non_eval",
    "clip": "docs/media/go2-policy-20261012_t1_v3/simulation.mp4",
    "source_sha256": "dabc1f5d2cebf4ac7e9c0f71c515c0d6ca93630fc045e9ed602c42e29495fe06",
}
GRAPH_TELEMETRY = (
    ("normal", "bench/runs_p0c/baseline_p0c_s2026.telemetry.json", "01841ca7b3bc8b7fb03b2ccffe763720f7dbc28a3d26bd1fca57cd1f49f48ee9"),
    ("fault", "bench/runs_p0c/h02_s2026.telemetry.json", "1eb3b4d09ad204b2b8c79aaf3c8950eb02dfe32c9c6c2464c0c3be168b31d250"),
)
MEDIA_MANIFEST_SHA = {
    "docs/media/go2-correction-motor_resume_dev_20261013/source_manifest.json": "f3f87250ef93377a1e170c2fdf0d11016454c46db181ee9d43438c52b2b158da",
    "docs/media/go2-correction-reward_resume_dev3_20261013/source_manifest.json": "6a09800109cbbde5cd942d0cfc57d93b952dda1790fe011ccc74bf0d5436b265",
    "docs/media/go2-correction-reward_v4_s2026_20261013/source_manifest.json": "9ead6de8d2ea3b76c06d4499ab11563c81c2ec44f24aff2a7340406e6504e6af",
    "docs/media/go2-correction-reward_v4_s2028_20261013/source_manifest.json": "b0721b6ddd67a940b4be6044934931630208aac7e26298217b7e8364f6c596a5",
    "docs/media/go2-e2e-e2e_reward_s2030_20261013/source_manifest.json": "2ff7a85aae28c13ebdb8b5815a69ef0e5f72bbd5648e0d4e6d828d96695ef4b6",
}

SOURCE_FILES = set(
    ["evals/nemotron_bottleneck_vision_v2.py", "tests/test_nemotron_bottleneck_vision_v2.py", PREFLIGHT["clip"]]
    + [r[1] for r in _ROWS]
    + [r[7] for r in _ROWS]
    + [r[1] for r in GRAPH_TELEMETRY]
    + list(MEDIA_MANIFEST_SHA)
)

VIDEO_SYSTEM = (
    "You observe six chronologically ordered frames from one locomotion rollout. "
    "Use only visible motion and pose evidence. Return exactly one JSON object with "
    "walking, in_place, and falling. Each value must be yes, no, or unclear."
)
VIDEO_QUESTION = (
    "Across the six ordered frames, is the robot walking with visible horizontal "
    "displacement, staying mostly in place, or falling? Do not infer a commanded "
    "velocity or evaluation score because neither is shown."
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _messages_sha(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical(messages)).hexdigest()


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    value = float(result.stdout.strip())
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid video duration: {path}")
    return value


def _extract_frames(source: Path, destination: Path) -> list[dict[str, Any]]:
    duration = _duration(source)
    frames: list[dict[str, Any]] = []
    for index in range(FRAME_COUNT):
        timestamp = round(duration * (2 * index + 1) / (2 * FRAME_COUNT), 6)
        output = destination / f"frame_{index + 1:02d}.jpg"
        for width, quality in ((640, 4), (480, 7)):
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source), "-ss", f"{timestamp:.6f}",
                 "-frames:v", "1", "-vf", f"scale='min({width},iw)':-2", "-q:v", str(quality), "-y", str(output)],
                check=True, capture_output=True,
            )
            if output.stat().st_size <= MAX_JPEG_BYTES:
                break
        if output.stat().st_size > MAX_JPEG_BYTES:
            raise ValueError(f"JPEG exceeds byte ceiling: {output}")
        frames.append({"file": output.name, "timestamp_s": timestamp, "sha256": _sha(output), "bytes": output.stat().st_size})
    return frames


def _draw_line(pixels: bytearray, width: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < len(pixels) // (width * 3):
            offset = (y0 * width + x0) * 3
            pixels[offset:offset + 3] = bytes(color)
        if x0 == x1 and y0 == y1:
            return
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _render_graph(repo: Path, destination: Path) -> dict[str, Any]:
    loaded: dict[str, dict[str, list[float]]] = {}
    telemetry_binding = []
    for name, logical, expected_sha in GRAPH_TELEMETRY:
        source = repo / logical
        if _sha(source) != expected_sha:
            raise ValueError(f"telemetry source changed: {logical}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        loaded[name] = payload["series"]
        telemetry_binding.append({"logical_id": logical, "sha256": expected_sha})
    width, height = 960, 540
    pixels = bytearray([250] * width * height * 3)
    panels = ((30, 250, "Episode_Reward/track_lin_vel_xy_exp"), (290, 510, "Metrics/base_velocity/error_vel_xy"))
    colors = {"normal": (33, 102, 172), "fault": (230, 85, 13)}
    for top, bottom, key in panels:
        left, right = 55, 930
        _draw_line(pixels, width, left, top, left, bottom, (60, 60, 60))
        _draw_line(pixels, width, left, bottom, right, bottom, (60, 60, 60))
        all_values = [float(v) for name in ("normal", "fault") for v in loaded[name][key]]
        low, high = min(all_values), max(all_values)
        pad = max((high - low) * .05, 1e-9)
        low, high = low - pad, high + pad
        for name in ("normal", "fault"):
            values = [float(v) for v in loaded[name][key]]
            points = []
            for i, value in enumerate(values):
                x = left + round((right - left) * i / max(1, len(values) - 1))
                y = bottom - round((bottom - top) * (value - low) / (high - low))
                points.append((x, y))
            for first, second in zip(points, points[1:]):
                _draw_line(pixels, width, *first, *second, colors[name])
    ppm = destination.with_suffix(".ppm")
    ppm.write_bytes(f"P6\n{width} {height}\n255\n".encode() + bytes(pixels))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(ppm), "-frames:v", "1", "-q:v", "3", "-y", str(destination)], check=True, capture_output=True)
    ppm.unlink()
    return {"file": destination.name, "sha256": _sha(destination), "bytes": destination.stat().st_size, "telemetry": telemetry_binding}


def _source_binding(repo: Path, clip: str, source_sha: str, checkpoint: str) -> dict[str, str]:
    manifest_logical = (Path(clip).parent / "source_manifest.json").as_posix()
    manifest_path = repo / manifest_logical
    manifest_sha = MEDIA_MANIFEST_SHA[manifest_logical]
    if _sha(manifest_path) != manifest_sha:
        raise ValueError(f"media manifest changed: {manifest_logical}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [item for item in manifest.get("clips", []) if item.get("file") == Path(clip).name]
    if len(matches) != 1:
        raise ValueError(f"clip mapping missing or ambiguous: {clip}")
    mapped = matches[0]
    if mapped.get("sha256") != source_sha or mapped.get("checkpoint_sha256") != checkpoint:
        raise ValueError(f"clip/checkpoint mapping changed: {clip}")
    return {"logical_id": clip, "sha256": source_sha,
            "media_manifest_logical_id": manifest_logical, "media_manifest_sha256": manifest_sha}


def _verify_label_report(path: Path, case_id: str, checkpoint: str, ratio: float, fall_rate: float | None) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if case_id == "demo_2030_restart":
        result = report.get("correction", {}).get("result", {})
        if (result.get("label") != "recovered" or result.get("verdict", {}).get("label") != "healthy"
                or not math.isclose(float(result.get("standstill_ratio", math.nan)), ratio, abs_tol=1e-12)):
            raise ValueError("demo restart label binding changed")
        return
    if report.get("checkpoint", {}).get("sha256") != checkpoint:
        raise ValueError(f"label checkpoint changed: {case_id}")
    observed_ratio = report["metrics"]["lin_vel_rmse_mps"] / report["standstill"]["lin_vel_rmse_mps"]
    if round(observed_ratio, 4) != ratio or round(float(report["metrics"]["fall_rate"]), 4) != fall_rate:
        raise ValueError(f"fixed-eval label changed: {case_id}")


def prepare_assets(repo_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """원본 결합을 확인하고 평가·preflight 프레임과 그래프를 새 폴더에 만든다."""
    repo, output = Path(repo_root).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets: dict[str, Any] = {}
    for row in _ROWS:
        case_id, clip, label, ratio, fall_rate, checkpoint, source_sha, report, report_sha = row
        source, report_path = repo / clip, repo / report
        if _sha(source) != source_sha:
            raise ValueError(f"video source changed: {clip}")
        if _sha(report_path) != report_sha:
            raise ValueError(f"label source changed: {report}")
        source_binding = _source_binding(repo, clip, source_sha, checkpoint)
        _verify_label_report(report_path, case_id, checkpoint, ratio, fall_rate)
        case_dir = output / case_id
        case_dir.mkdir()
        assets[case_id] = {
            "source": source_binding,
            "label_binding": {"group": label, "standstill_ratio": ratio, "fall_rate": fall_rate,
                              "checkpoint_sha256": checkpoint, "report_logical_id": report, "report_sha256": report_sha},
            "frames": _extract_frames(source, case_dir),
            "asset_dir": case_id,
        }
    preflight_source = repo / PREFLIGHT["clip"]
    if _sha(preflight_source) != PREFLIGHT["source_sha256"]:
        raise ValueError("preflight source changed")
    preflight_dir = output / PREFLIGHT["id"]
    preflight_dir.mkdir()
    assets[PREFLIGHT["id"]] = {
        "source": {"logical_id": PREFLIGHT["clip"], "sha256": PREFLIGHT["source_sha256"]},
        "frames": _extract_frames(preflight_source, preflight_dir), "asset_dir": PREFLIGHT["id"],
    }
    graph = _render_graph(repo, output / "learning_curve.jpg")
    manifest = {"schema": 1, "frame_count": FRAME_COUNT, "max_jpeg_bytes": MAX_JPEG_BYTES,
                "assets": assets, "graph": graph}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _load_manifest(manifest: str | Path | dict[str, Any] | None) -> tuple[dict[str, Any], Path]:
    if manifest is None:
        manifest = Path(os.environ.get("RL_TRIAGE_VISION_MANIFEST", DEFAULT_MANIFEST))
    if isinstance(manifest, dict):
        root = Path(manifest.get("_asset_root", "."))
        return manifest, root
    path = Path(manifest)
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def _data_url(path: Path, expected_sha: str, expected_bytes: int) -> str:
    if _sha(path) != expected_sha or path.stat().st_size != expected_bytes or expected_bytes > MAX_JPEG_BYTES:
        raise ValueError(f"prepared frame binding failed: {path.name}")
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _video_case(case_id: str, entry: dict[str, Any], root: Path, *, preflight: bool = False) -> dict[str, Any]:
    if len(entry["frames"]) != FRAME_COUNT:
        raise ValueError(f"frame count mismatch: {case_id}")
    content: list[dict[str, Any]] = [{"type": "text", "text": VIDEO_QUESTION}]
    frame_binding = []
    for frame in entry["frames"]:
        path = root / entry["asset_dir"] / frame["file"]
        content.append({"type": "image_url", "image_url": {"url": _data_url(path, frame["sha256"], frame["bytes"]), "detail": "low"}})
        frame_binding.append({k: frame[k] for k in ("timestamp_s", "sha256", "bytes")})
    messages = [{"role": "system", "content": VIDEO_SYSTEM}, {"role": "user", "content": content}]
    binding = {"source": entry["source"], "frames": frame_binding, "messages_sha256": _messages_sha(messages)}
    packet: dict[str, Any] = {"preflight": preflight}
    if not preflight:
        binding["label_binding"] = entry["label_binding"]
        packet["fixed_eval_group"] = entry["label_binding"]["group"]
    return {"id": case_id, "bottleneck": "B4", "kind": "vision", "messages": messages,
            "request_params": _request_params(), "binding": binding, "packet": packet}


def _request_params() -> dict[str, Any]:
    schema = {"name": "locomotion_observation", "strict": True, "schema": {"type": "object", "additionalProperties": False,
              "properties": {key: {"type": "string", "enum": sorted(ENUM)} for key in ("walking", "in_place", "falling")},
              "required": ["walking", "in_place", "falling"]}}
    return {"temperature": 0, "max_tokens": 300, "stream": False,
            "response_format": {"type": "json_schema", "json_schema": schema},
            "extra_body": {"reasoning_effort": "none"}}


def build_cases(manifest: str | Path | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload, root = _load_manifest(manifest)
    cases = []
    checkpoints = set()
    for row in _ROWS:
        case_id, clip, label, ratio, fall_rate, checkpoint, source_sha, report, report_sha = row
        entry = payload["assets"][case_id]
        manifest_logical = (Path(clip).parent / "source_manifest.json").as_posix()
        expected_source = {"logical_id": clip, "sha256": source_sha,
                           "media_manifest_logical_id": manifest_logical,
                           "media_manifest_sha256": MEDIA_MANIFEST_SHA[manifest_logical]}
        expected_label = {"group": label, "standstill_ratio": ratio, "fall_rate": fall_rate,
                          "checkpoint_sha256": checkpoint, "report_logical_id": report, "report_sha256": report_sha}
        if entry["source"] != expected_source or entry["label_binding"] != expected_label:
            raise ValueError(f"source or label binding mismatch: {case_id}")
        if checkpoint in checkpoints:
            raise ValueError(f"duplicate checkpoint: {case_id}")
        checkpoints.add(checkpoint)
        cases.append(_video_case(case_id, entry, root))
    if len(cases) != 18 or sum(c["packet"]["fixed_eval_group"] == "normal" for c in cases) != 7:
        raise AssertionError("fixed inventory must be 7 normal and 11 failure")
    return cases


def build_preflight_cases(manifest: str | Path | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload, root = _load_manifest(manifest)
    entry = payload["assets"][PREFLIGHT["id"]]
    if entry["source"] != {"logical_id": PREFLIGHT["clip"], "sha256": PREFLIGHT["source_sha256"]}:
        raise ValueError("preflight source binding mismatch")
    return [_video_case(PREFLIGHT["id"], entry, root, preflight=True)]


def _graph_case(case_id: str, question: str, schema: dict[str, list[str]], payload: dict[str, Any], root: Path) -> dict[str, Any]:
    graph = payload["graph"]
    image_url = _data_url(root / graph["file"], graph["sha256"], graph["bytes"])
    messages = [{"role": "system", "content": "Read the plotted curves only. Return exactly the requested JSON object."},
                {"role": "user", "content": [{"type": "text", "text": question},
                                                 {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}]}]
    response_schema = {"name": case_id, "strict": True, "schema": {"type": "object", "additionalProperties": False,
                       "properties": {key: {"type": "string", "enum": values} for key, values in schema.items()},
                       "required": list(schema)}}
    return {"id": case_id, "bottleneck": "B4", "kind": "graph", "messages": messages,
            "request_params": {"temperature": 0, "max_tokens": 200, "stream": False,
                               "response_format": {"type": "json_schema", "json_schema": response_schema},
                               "extra_body": {"reasoning_effort": "none"}},
            "binding": {"graph_sha256": graph["sha256"], "telemetry": graph["telemetry"],
                        "messages_sha256": _messages_sha(messages)},
            "packet": {"allowed": schema}}


def build_graph_cases(manifest: str | Path | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload, root = _load_manifest(manifest)
    expected = [{"logical_id": p, "sha256": s} for _, p, s in GRAPH_TELEMETRY]
    if payload["graph"]["telemetry"] != expected:
        raise ValueError("graph telemetry binding mismatch")
    return [
        _graph_case("graph_endpoint_order", "Blue is the normal run and orange is the reward-fault run. The upper panel is tracking reward and the lower panel is velocity error. Which color ends higher in the upper panel and lower in the lower panel? Return top_higher and bottom_lower.", {"top_higher": ["blue", "orange", "similar", "unclear"], "bottom_lower": ["blue", "orange", "similar", "unclear"]}, payload, root),
        _graph_case("graph_fault_trend", "Blue is the normal run and orange is the reward-fault run. Does the orange tracking reward visibly improve over training, and does its velocity error visibly decrease? Return orange_tracking_improves and orange_error_decreases.", {"orange_tracking_improves": sorted(ENUM), "orange_error_decreases": sorted(ENUM)}, payload, root),
    ]


def _parse_json_object(content: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None, ["response_not_json"]
    if not isinstance(value, dict):
        return None, ["response_not_object"]
    return value, []


def validate_response(case: dict[str, Any], content: str) -> dict[str, Any]:
    value, errors = _parse_json_object(content)
    allowed = case.get("packet", {}).get("allowed")
    if allowed is None:
        allowed = {key: sorted(ENUM) for key in ("walking", "in_place", "falling")}
    if value is not None:
        if set(value) != set(allowed):
            errors.append("response_keys_mismatch")
        for key, choices in allowed.items():
            if value.get(key) not in choices:
                errors.append(f"invalid_enum:{key}")
    valid = not errors
    metrics: dict[str, Any] = {"capability_valid": valid, "label_alignment": None}
    group = case.get("packet", {}).get("fixed_eval_group")
    policy_label = "failed" if group == "failure" else group
    if valid and group in {"normal", "failure"}:
        expected = "yes" if group == "normal" else "no"
        metrics["label_alignment"] = value["walking"] == expected
        metrics["alignment_basis"] = "walking_vs_fixed_eval_group"
    return {"valid": valid, "errors": errors, "decision_key": value if valid else None,
            "metrics": metrics, "policy_label": policy_label,
            "label_alignment": metrics["label_alignment"]}


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    """응답 기록용 사본. 이미지 바이트와 내부 packet은 제거한다."""
    return {key: json.loads(json.dumps(case[key]))
            for key in ("id", "bottleneck", "kind", "request_params", "binding")}


def prepare(repo_root: str | Path = ROOT, output_dir: str | Path = DEFAULT_ASSET_ROOT) -> dict[str, Any]:
    """최초에는 자산을 만들고, 이후에는 결합된 기존 자산 전체를 검증한다."""
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        build_preflight_cases(manifest_path)
        build_cases(manifest_path)
        build_graph_cases(manifest_path)
        return payload
    return prepare_assets(repo_root, output)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepare", nargs="?")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    prepare(args.repo_root, args.output)
    print(json.dumps({"outcome": "prepared", "manifest": str(Path(args.output) / "manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
