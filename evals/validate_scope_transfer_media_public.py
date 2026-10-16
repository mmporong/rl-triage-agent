"""U1·ANYmal-C 공개 비교 영상의 해시·manifest·ffprobe 계약을 검사한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("bench/protocols/replay_additional_checks_v2.json")
CASES = {
    "u1": {
        "folder": "docs/media/scope-transfer-u1-s2051-v1",
        "task": "Isaac-Velocity-Flat-Unitree-Go2-Play-v0",
        "protocol_path": "bench/protocols/scope_transfer_media_u1_s2051_v1.json",
    },
    "anymal": {
        "folder": "docs/media/scope-transfer-anymal-s2047-v1",
        "task": "Isaac-Velocity-Flat-Anymal-C-Play-v0",
        "protocol_path": "bench/protocols/scope_transfer_media_anymal_s2047_v1.json",
    },
}
FILENAMES = {
    "after_corrected.mp4", "after_sham.mp4", "before_original.mp4",
    "comparison.gif", "comparison.mp4", "source_manifest.json",
}
CLIPS = (
    ("before_original", "failed", "unhealthy"),
    ("after_corrected", "corrected_restart", "healthy"),
    ("after_sham", "none_control", "healthy"),
)
PRIVATE = (re.compile(r"[A-Za-z]:[\\/]"), re.compile(r"/home/|/Users/|bench/private", re.I))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def committed_clean_state(root: Path, relative: str) -> tuple[bool, bool]:
    tracked = git(root, "ls-files", "--error-unmatch", "--", relative)
    if tracked.returncode not in (0, 1):
        raise ValueError(f"git_ls_files_failed:{relative}")
    status = git(root, "status", "--porcelain", "--", relative)
    if status.returncode != 0:
        raise ValueError(f"git_status_failed:{relative}")
    is_tracked = tracked.returncode == 0
    return is_tracked, is_tracked and not status.stdout.strip()


def require_committed_clean(root: Path, relative: str) -> None:
    committed, clean = committed_clean_state(root, relative)
    if not (committed and clean):
        raise ValueError(f"not_committed_clean:{relative}")


def probe(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise ValueError(f"ffprobe_failed:{path.name}")
    row = json.loads(result.stdout)
    stream = row["streams"][0]
    return {"width": stream["width"], "height": stream["height"],
            "frame_rate": stream["avg_frame_rate"], "frame_count": int(stream["nb_frames"]),
            "duration_s": float(row["format"]["duration"])}


def validate_manifest(case: str, spec: dict, manifest: dict, recording: dict | None = None,
                      root: Path = ROOT) -> None:
    if (manifest.get("schema") != "scope_transfer_media_artifact_v1"
            or manifest.get("case_id") != spec["case_id"]
            or manifest.get("media_claim") != "visual_playback_only_fixed_eval_labels_preserved"
            or manifest.get("playback") != {"task": spec["task"], "seed": 42, "num_envs": 1,
                                             "video_length": 300,
                                             "kit_args": "--/app/vulkan=false --/app/window/hideUi=true",
                                             "overrides": ["env.viewer.eye=[3.0,3.0,2.0]",
                                                           "env.viewer.lookat=[0.0,0.0,0.35]"]}
            or manifest.get("protocol") != {"path": spec["protocol_path"],
                                             "sha256_lf": spec["protocol_sha256"]}):
        raise ValueError(f"{case}:manifest_identity_or_playback")
    clips = manifest.get("clips", [])
    if len(clips) != 3:
        raise ValueError(f"{case}:clip_count")
    for clip, (clip_id, role, label) in zip(clips, CLIPS):
        filename = clip_id + ".mp4" if clip_id != "after_sham" else "after_sham.mp4"
        if (clip.get("id"), clip.get("role"), clip.get("fixed_eval_label"), clip.get("file")) != (
                clip_id, role, label, filename):
            raise ValueError(f"{case}:clip_binding")
        if clip.get("sha256") != spec["files"][filename]["sha256"]:
            raise ValueError(f"{case}:clip_sha")
    expected_comparison = {
        "mp4": ("comparison.mp4", spec["files"]["comparison.mp4"]["sha256"]),
        "gif": ("comparison.gif", spec["files"]["comparison.gif"]["sha256"]),
    }
    for key, (filename, digest) in expected_comparison.items():
        if (manifest.get("comparison", {}).get(key, {}).get("file"),
                manifest.get("comparison", {}).get(key, {}).get("sha256")) != (filename, digest):
            raise ValueError(f"{case}:comparison_binding")
    serialized = json.dumps(manifest, ensure_ascii=False)
    if any(pattern.search(serialized) for pattern in PRIVATE):
        raise ValueError(f"{case}:private_path")
    if recording is not None:
        if (recording.get("schema") != "scope_transfer_media_protocol_v1"
                or recording.get("case_id") != case or recording.get("task") != spec["task"]
                or recording.get("media_claim") != manifest["media_claim"]
                or recording.get("seed") != 42 or recording.get("num_envs") != 1
                or recording.get("video_length") != 300
                or recording.get("kit_args") != manifest["playback"]["kit_args"]
                or recording.get("overrides") != manifest["playback"]["overrides"]):
            raise ValueError(f"{case}:recording_protocol_contract")
        fields = ("id", "role", "logical_run_id", "checkpoint_sha256", "fixed_eval_label")
        if ([{field: row[field] for field in fields} for row in recording.get("clips", [])]
                != [{field: row[field] for field in fields} for row in manifest["clips"]]):
            raise ValueError(f"{case}:policy_checkpoint_binding")
        if recording.get("public_sources") != manifest.get("public_sources"):
            raise ValueError(f"{case}:public_source_inventory")
        if recording.get("private_sources") != manifest.get("private_sources"):
            raise ValueError(f"{case}:private_source_logical_binding")
        for item in manifest["public_sources"]:
            path = root / item["path"]
            if not path.is_file() or sha256_lf(path) != item["sha256_lf"]:
                raise ValueError(f"{case}:public_source_sha")
        task_sources = recording.get("play_task_sources", [])
        if {row.get("role") for row in task_sources} != {"registration", "environment", "agent"}:
            raise ValueError(f"{case}:play_task_source_inventory")
        for item in task_sources:
            if (not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
                    or not item.get("relative_path") or not item.get("required_symbols")):
                raise ValueError(f"{case}:play_task_source_binding")


def validate(root: Path = ROOT, *, require_committed: bool, run_ffprobe: bool = True) -> dict:
    protocol = json.loads((root / PROTOCOL).read_text(encoding="utf-8"))
    if protocol.get("schema") != "additional_checks_replay_contract_v2" or set(protocol.get("media", {})) != set(CASES):
        raise ValueError("replay_protocol_contract")
    if require_committed:
        require_committed_clean(root, PROTOCOL.as_posix())
    base = protocol["base_protocol"]
    if sha256(root / base["path"]) != base["sha256"]:
        raise ValueError("base_protocol_sha")
    cases = {}
    for case, spec in protocol["media"].items():
        if any(spec.get(field) != value for field, value in CASES[case].items()):
            raise ValueError(f"{case}:registered_case_binding")
        if set(spec["files"]) != FILENAMES:
            raise ValueError(f"{case}:file_inventory")
        recording_path = root / spec["protocol_path"]
        if sha256(recording_path) != spec["protocol_sha256"]:
            raise ValueError(f"{case}:recording_protocol_sha")
        recording = json.loads(recording_path.read_text(encoding="utf-8"))
        folder = root / spec["folder"]
        actual_names = {path.name for path in folder.iterdir() if path.is_file()}
        if actual_names != FILENAMES:
            raise ValueError(f"{case}:folder_inventory")
        for filename, expected in spec["files"].items():
            path = folder / filename
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"{case}:file_binding:{filename}")
        manifest = json.loads((folder / "source_manifest.json").read_text(encoding="utf-8"))
        validate_manifest(case, spec, manifest, recording, root)
        probes = {}
        if run_ffprobe:
            for filename in sorted(FILENAMES - {"source_manifest.json"}):
                probes[filename] = probe(folder / filename)
            for clip in manifest["clips"]:
                if probes[clip["file"]] != clip["ffprobe"]:
                    raise ValueError(f"{case}:ffprobe:{clip['file']}")
            for key in ("mp4", "gif"):
                item = manifest["comparison"][key]
                if probes[item["file"]] != item["ffprobe"]:
                    raise ValueError(f"{case}:ffprobe:{item['file']}")
        paths = [f"{spec['folder']}/{name}" for name in sorted(FILENAMES)]
        states = [committed_clean_state(root, path) for path in paths]
        committed = all(state[0] for state in states)
        clean = all(state[1] for state in states)
        if require_committed and not (committed and clean):
            raise ValueError(f"{case}:not_committed_clean")
        cases[case] = {"status": "ready" if committed and clean else "precommit_verified",
                       "file_count": len(paths), "total_bytes": sum(row["bytes"] for row in spec["files"].values()),
                       "ffprobe": probes}
    return {"schema": "scope_transfer_media_public_verification_v1",
            "status": "ready" if all(row["status"] == "ready" for row in cases.values()) else "precommit_verified",
            "require_committed": require_committed, "cases": cases, "gpu_calls": 0, "model_calls": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-uncommitted", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(validate(require_committed=not args.allow_uncommitted), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
