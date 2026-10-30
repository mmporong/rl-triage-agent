"""등록된 기존 H 한 대에서 H1·Go2 계단 영상 네 클립을 순차 수집한다.

이 모듈은 MainLifecycle v5와 Driver v9의 Start/boot/copy/script/Stop 경계를 재사용한다.
등록 전 protocol, 다른 source blob, 다른 체크포인트, 다른 비용 원점은 실행 전에 거부한다.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

SCHEMA = "stairs_capture_host_registration_v2"
RUNNER_SCHEMA = "stairs_playback_spec_v1"
RESOURCE = {
    "resource_id": "d3cxwsmgd",
    "resource_name": "rl-triage-l40s-renderer-20261029h",
    "instance_type": "gpu-l40s-d.1gpu-16vcpu-96gb",
}
ENV_REF = {
    "path": "bench/protocols/agentic_runtime_text_s2221_20261030_v49_environment.json",
    "sha256": "23abf27328047dfa32a8dad99302986977d5dcf1b3803ffa4c0095d1debc7968",
}
LIFECYCLE_REF = {
    "path": "bench/protocols/agentic_runtime_text_s2221_20261030_v49_lifecycle.json",
    "sha256": "4ee5190fd9fcc6d07c84083266f75857dfdb89b4b5084d81724311ea959381f3",
}
LEASE_REF = {
    "path": "bench/protocols/agentic_runtime_text_s2221_20261030_v49_lease_basis.json",
    "sha256": "bd85e2565899bc8c0e7540460fae54c2aa6c82c672f44e3a6a995d35c9669e93",
}
ENV_ANCHOR = "3816e4614d6d5ec299aad1eea745fd634b99fedf"
DEADLINE = "2026-10-30T06:05:36.779993+00:00"
STOP_CUTOFF = "2026-10-30T05:45:36.779993+00:00"
CHECKPOINTS = {
    "biped": ("h1_rough_checkpoint.pt", 7150946,
              "15710657f43b714ddfe249d1a56db1d27682fb7801101f88dbe5a6bf17898a88"),
    "quadruped": ("go2_rough_checkpoint.pt", 6881762,
                  "f2aa77bf0349c10ec2a00071162e37123d0cf80f3447c3989c17a618e1b24ad2"),
}
CLIPS = (
    ("biped-stairs-up", "biped", "up", "Isaac-Velocity-Rough-H1-v0",
     "Isaac-Velocity-Rough-H1-Play-v0"),
    ("biped-stairs-down", "biped", "down", "Isaac-Velocity-Rough-H1-v0",
     "Isaac-Velocity-Rough-H1-Play-v0"),
    ("quadruped-stairs-up", "quadruped", "up", "Isaac-Velocity-Rough-Unitree-Go2-v0",
     "Isaac-Velocity-Rough-Unitree-Go2-Play-v0"),
    ("quadruped-stairs-down", "quadruped", "down", "Isaac-Velocity-Rough-Unitree-Go2-v0",
     "Isaac-Velocity-Rough-Unitree-Go2-Play-v0"),
)


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False, timeout=60,
    )
    if result.returncode:
        raise ValueError("Git source 결합 실패")
    return result.stdout


def _public_source(repo: Path, anchor: str, reference: dict[str, Any]) -> Path:
    if set(reference) != {"path", "sha256_lf"}:
        raise ValueError("공개 source ref 필드가 다르다")
    relative = reference["path"]
    pure = PurePosixPath(relative)
    if (not isinstance(relative, str) or pure.is_absolute() or pure.as_posix() != relative
            or ".." in pure.parts or relative.startswith("bench/private/")):
        raise ValueError("공개 source 경로가 아니다")
    path = repo.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink() or sha(path, lf=True) != reference["sha256_lf"]:
        raise ValueError("공개 source 현재 바이트가 다르다")
    blob = _git(repo, "show", f"{anchor}:{relative}")
    if hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() != reference["sha256_lf"]:
        raise ValueError("공개 source anchor blob이 다르다")
    return path


def validate_registration(path: Path, repo: Path, *, host_source: Path | None = None,
                          require_committed: bool = True) -> dict[str, Any]:
    if require_committed:
        relative = path.resolve(strict=True).relative_to(repo.resolve(strict=True)).as_posix()
        if relative.startswith("bench/private/"):
            raise ValueError("private preregistration은 실행할 수 없다")
        if _git(repo, "show", "HEAD:" + relative) != path.read_bytes():
            raise ValueError("현재 HEAD에 커밋된 preregistration 바이트가 아니다")
    registration = read(path)
    if registration.get("schema") != SCHEMA or registration.get("status") != "registered_executable":
        raise ValueError("커밋된 실행 가능 stairs 등록이 필요하다")
    anchor = registration.get("source_anchor")
    if not isinstance(anchor, str) or len(anchor) != 40:
        raise ValueError("source anchor가 등록되지 않았다")
    _git(repo, "merge-base", "--is-ancestor", anchor, "HEAD")
    sources = registration.get("sources")
    if not isinstance(sources, dict) or set(sources) != {"host", "runner", "lifecycle_adapter", "driver"}:
        raise ValueError("실행 source closure가 다르다")
    host = _public_source(repo, anchor, sources["host"])
    if host_source is not None and sha(host_source, lf=True) != sha(host, lf=True):
        raise ValueError("실행 중 host source가 등록 blob과 다르다")
    _public_source(repo, anchor, sources["runner"])
    if registration.get("resource") != RESOURCE:
        raise ValueError("등록 자원 신원이 다르다")
    if registration.get("environment") != {"anchor": ENV_ANCHOR, **ENV_REF}:
        raise ValueError("Time12 environment 결합이 다르다")
    if registration.get("lifecycle") != LIFECYCLE_REF:
        raise ValueError("Time12 lifecycle 결합이 다르다")
    if registration.get("lease_basis") != LEASE_REF:
        raise ValueError("Time12 lease 결합이 다르다")
    budget = registration.get("budget")
    expected_budget = {
        "budget_origin": "2026-10-28T21:05:36.779993+00:00",
        "compute_cap_usd": "75", "program_cap_usd": "240", "lease_batch_budget_usd": "55",
        "deadline_utc": DEADLINE, "stop_cutoff_utc": STOP_CUTOFF, "reset": False,
    }
    if budget != expected_budget:
        raise ValueError("Time12 비용 원점/상한이 바뀌었다")
    expected_clips = [
        {"clip_id": clip, "robot_role": role, "direction": direction,
         "training_task": training, "play_task": play}
        for clip, role, direction, training, play in CLIPS
    ]
    if registration.get("clips") != expected_clips:
        raise ValueError("4클립 순서가 다르다")
    if registration.get("smoke") != {"tag": "stairs-s172-host176-biped-up-v2",
                                      "mode": "first_full_clip", "clip_id": "biped-stairs-up"}:
        raise ValueError("첫 full clip GPU 확인 등록이 다르다")
    if registration.get("execution") != {"new_vm": 0, "install": 0, "new_training": 0,
                                           "model_api": 0, "maximum_stop_requests": 1,
                                           "automatic_retry": False}:
        raise ValueError("실행 범위가 다르다")
    acceptance = registration.get("clip_acceptance")
    if (not isinstance(acceptance, dict)
            or set(acceptance) != {"manifest", "trace", "video_glob", "app_startup_marker",
                                   "result_ready_marker",
                                   "app_close_entered_marker", "app_close_marker",
                                   "trace_rows", "video_frames", "fps"}
            or acceptance["trace_rows"] != 626 or acceptance["video_frames"] != 625
            or acceptance["fps"] != 50
            or any(not isinstance(acceptance[key], str) or not acceptance[key]
                   for key in ("manifest", "trace", "video_glob", "app_startup_marker",
                               "result_ready_marker",
                               "app_close_entered_marker", "app_close_marker"))):
        raise ValueError("clip 수용 계약이 다르다")
    return registration


def validate_checkpoints(checkpoint_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for role, (name, size, digest) in CHECKPOINTS.items():
        path = checkpoint_dir / name
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != size
                or sha(path) != digest):
            raise ValueError(f"{role} 공식 checkpoint 바이트가 다르다")
        result[role] = path
    return result


def validate_plan(plan_path: Path, registration: dict[str, Any]) -> dict[str, Any]:
    plan = read(plan_path)
    if plan.get("resource_id") != RESOURCE["resource_id"]:
        raise ValueError("lifecycle plan 자원이 다르다")
    if plan.get("environment_protocol_anchor") != ENV_ANCHOR:
        raise ValueError("lifecycle plan environment anchor가 다르다")
    for key, expected in (("environment_phase", ENV_REF), ("lifecycle_protocol", LIFECYCLE_REF)):
        ref = plan.get(key)
        if not isinstance(ref, dict) or ref.get("sha256") != expected["sha256"]:
            raise ValueError(f"lifecycle plan {key}가 다르다")
        if PurePosixPath(str(ref.get("path", "")).replace("\\", "/")).name != Path(expected["path"]).name:
            raise ValueError(f"lifecycle plan {key} 파일이 다르다")
    driver = plan.get("driver")
    if not isinstance(driver, dict) or driver.get("sha256") != registration["sources"]["driver"]["sha256"]:
        raise ValueError("Driver v9 source 결합이 다르다")
    return plan


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_specs(registration: dict[str, Any], remote_checkpoints: dict[str, str]) -> list[dict[str, Any]]:
    contract = registration["runner_contract"]
    specs = []
    for clip_id, role, direction, training, play in CLIPS:
        specs.append({
            "schema": RUNNER_SCHEMA, "clip_id": clip_id, "robot_role": role, "direction": direction,
            "training_task": training, "play_task": play, "seed": 172,
            "checkpoint": {"path": remote_checkpoints[role], "sha256": CHECKPOINTS[role][2]},
            "source": {"isaaclab_commit": contract["isaaclab_commit"],
                       "isaaclab_files_lf": {**contract["common_files_lf"],
                                             **contract["role_files_lf"][role]},
                       "runner_sha256_lf": contract["runner_sha256_lf"]},
            "capture": {"frames": 625, "fps": 50, "step_height_m": 0.1,
                        "step_width_m": 0.3, "command_x_m_s": 0.25,
                        "camera": "asset_root_three_quarter"},
        })
    return specs


def _encoded(value: Any) -> str:
    return repr(base64.b64encode(json.dumps(value, separators=(",", ":")).encode()).decode())


def _clip_body(config: dict[str, Any]) -> str:
    return f"""python3 - <<'PY'
import base64,hashlib,json,os,subprocess
from pathlib import Path
c=json.loads(base64.b64decode({_encoded(config)}));repo=Path(c['repo'])
env={{**os.environ,'PYTHONPATH':os.pathsep.join(str(repo/p) for p in ['bench','evals','src']),'OMNI_KIT_ACCEPT_EULA':'YES','PYTHONUNBUFFERED':'1'}}
r=subprocess.run([c['python'],'-B',str(repo/c['runner']),'--spec',c['spec'],'--out',c['out']],cwd=repo,env=env,timeout=900)
if r.returncode!=0:raise SystemExit(r.returncode)
out=Path(c['out']);a=c['acceptance'];spec=json.loads(Path(c['spec']).read_text());manifest=json.loads((out/a['manifest']).read_text())
if manifest.get('source')!=spec['source'] or manifest.get('checkpoint_sha256')!=spec['checkpoint']['sha256']:raise RuntimeError('manifest source/checkpoint binding')
trace=out/a['trace'];videos=list(out.glob(a['video_glob']));startup=json.loads((out/a['app_startup_marker']).read_text());ready=json.loads((out/a['result_ready_marker']).read_text());entered=json.loads((out/a['app_close_entered_marker']).read_text());marker=json.loads((out/a['app_close_marker']).read_text())
if len(trace.read_text().splitlines())!=a['trace_rows'] or len(videos)!=1:raise RuntimeError('trace/video cardinality')
if hashlib.sha256(trace.read_bytes()).hexdigest()!=manifest['state_trace']['sha256']:raise RuntimeError('trace SHA mismatch')
if hashlib.sha256(videos[0].read_bytes()).hexdigest()!=manifest['video']['sha256']:raise RuntimeError('video SHA mismatch')
if startup.get('schema')!='stairs_playback_shutdown_marker_v1' or startup.get('event')!='app-startup' or startup.get('sequence')!=1:raise RuntimeError('app startup marker missing')
if ready.get('schema')!='stairs_playback_shutdown_marker_v1' or ready.get('event')!='result-ready' or ready.get('sequence')!=2 or ready.get('manifest_sha256')!=hashlib.sha256((out/a['manifest']).read_bytes()).hexdigest():raise RuntimeError('result ready marker missing')
if entered.get('schema')!='stairs_playback_shutdown_marker_v1' or entered.get('event')!='app-close-entered' or entered.get('sequence')!=3:raise RuntimeError('app close entered marker missing')
if marker.get('schema')!='stairs_playback_shutdown_marker_v1' or marker.get('event')!='app-close-returned' or marker.get('sequence')!=4:raise RuntimeError('app close marker missing')
markers=[startup,ready,entered,marker]
if len(set(m.get('pid') for m in markers))!=1 or [m.get('sequence') for m in markers]!=[1,2,3,4] or any(markers[i]['monotonic_ns']>markers[i+1]['monotonic_ns'] for i in range(3)):raise RuntimeError('shutdown marker lineage mismatch')
probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate','-of','json',str(videos[0])],capture_output=True,text=True,check=True,timeout=60)
stream=json.loads(probe.stdout)['streams'][0]
if int(stream['nb_read_frames'])!=a['video_frames'] or stream['r_frame_rate']!=str(a['fps'])+'/1':raise RuntimeError('video frame/fps mismatch')
PY
"""


def _pack_body(remote_root: str) -> str:
    return f"""python3 - <<'PY'
import hashlib,json,tarfile
from pathlib import Path
root=Path({remote_root!r});archive=root.with_suffix('.tar.gz');rows=[]
for p in sorted(root.rglob('*')):
 if p.is_dir():continue
 if not p.is_file() or p.is_symlink() or p.stat().st_nlink!=1:raise RuntimeError('non-regular evidence')
 rows.append({{'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}})
(root/'collection-inventory.json').write_text(json.dumps({{'files':rows}},sort_keys=True)+'\\n',encoding='utf-8')
with tarfile.open(archive,'x:gz') as t:t.add(root,arcname='evidence',recursive=True)
print('STAIRS_ARCHIVE='+json.dumps({{'path':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(rows)}}))
PY
"""


def run_sequence(owner: Any, registration: dict[str, Any], checkpoints: dict[str, Path], out: Path,
                 backup: Path, *, collector: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """이미 구성된 MainLifecycle owner에서 단일 Start와 네 순차 clip을 수행한다."""
    owner.driver.budget(4200)
    owner.start()
    booted = False
    primary: BaseException | None = None
    completed: list[str] = []
    collection: dict[str, Any] | None = None
    try:
        owner.boot()
        booted = True
        remote_root = owner.driver.metadata["remote"] + "/stairs-host173"
        owner.script("stairs-prepare", (
            "python3 - <<'PY'\nfrom pathlib import Path\n"
            f"root=Path({remote_root!r});root.mkdir(exist_ok=False);"
            "(root/'checkpoints').mkdir();(root/'specs').mkdir()\nPY\n"
        ), 60)
        remote_checkpoints = {
            role: remote_root + "/checkpoints/" + CHECKPOINTS[role][0] for role in CHECKPOINTS
        }
        for role in ("biped", "quadruped"):
            owner.copy("upload-" + role, str(checkpoints[role]),
                       Path(owner.lifecycle["resource_name"] + ":" + remote_checkpoints[role]), 300)
        specs = build_specs(registration, remote_checkpoints)
        spec_root = out / "specs"
        for spec in specs:
            write_new(spec_root / (spec["clip_id"] + ".json"), spec)
        repo_remote = owner.driver.metadata["repo"]
        python_remote = owner.driver.metadata["python"]
        runner_relative = registration["sources"]["runner"]["path"]
        for spec in specs:
            spec_path = spec_root / (spec["clip_id"] + ".json")
            remote_spec = remote_root + "/specs/" + spec_path.name
            owner.copy("upload-spec-" + spec["clip_id"], str(spec_path),
                       Path(owner.lifecycle["resource_name"] + ":" + remote_spec), 120)
            owner.script("stairs-" + spec["clip_id"], _clip_body({
                "repo": repo_remote, "python": python_remote, "runner": runner_relative,
                "spec": remote_spec, "out": remote_root + "/clips/" + spec["clip_id"],
                "acceptance": registration["clip_acceptance"],
            }), 960)
            completed.append(spec["clip_id"])
    except Exception as error:  # noqa: BLE001 - 실패 원문과 부분 산출물을 회수한 뒤 다시 발생시킨다.
        primary = error
        write_new(out / "primary-failure.json", {
            "error_type": type(error).__name__, "reason": str(error),
            "completed_clips": completed, "remaining_clips_not_started":
            [row[0] for row in CLIPS if row[0] not in completed], "automatic_retry": False,
        })
    finally:
        if booted:
            try:
                remote_root = owner.driver.metadata["remote"] + "/stairs-host173"
                receipt = owner.script("stairs-pack", _pack_body(remote_root), 300)
                text = Path(receipt["stdout_path"]).read_text(encoding="utf-8")
                packet = json.loads(next(line.split("STAIRS_ARCHIVE=", 1)[1]
                                         for line in text.splitlines() if "STAIRS_ARCHIVE=" in line))
                archive = out / "retrieved.tar.gz"
                owner.copy("stairs-collect", owner.lifecycle["resource_name"] + ":" + packet["path"],
                           archive, 300)
                if sha(archive) != packet["sha256"]:
                    raise ValueError("회수 archive SHA가 다르다")
                collect = collector or owner.host_module.collect_verified
                collection = collect(archive, out / "retrieved", backup)
                write_new(out / "collection.json", {**packet, "copy_verification": collection})
            except Exception as collect_error:  # noqa: BLE001 - 회수 실패도 별도 증거로 보존한다.
                write_new(out / "collection-failure.json", {
                    "error_type": type(collect_error).__name__, "reason": str(collect_error),
                    "primary_failure_preserved": primary is not None,
                })
                if primary is None:
                    primary = collect_error
    if primary is not None:
        raise primary
    return {"status": "complete", "completed_clips": completed, "collection": collection,
            "new_vm": 0, "new_training": 0, "model_api": 0}


def execute(registration_path: Path, plan_path: Path, lifecycle_source: Path,
            checkpoint_dir: Path, out: Path, backup: Path) -> dict[str, Any]:
    repo = Path.cwd().resolve(strict=True)
    registration = validate_registration(registration_path, repo, host_source=Path(__file__))
    plan = validate_plan(plan_path, registration)
    adapter_ref = registration["sources"]["lifecycle_adapter"]
    if sha(lifecycle_source) != adapter_ref["sha256"]:
        raise ValueError("MainLifecycle v5 source가 다르다")
    checkpoints = validate_checkpoints(checkpoint_dir)
    if out.exists() or backup.exists():
        raise FileExistsError("새 C/D 결과 경로가 필요하다")
    out.mkdir(parents=True)
    lifecycle_plan = dict(plan)
    lifecycle_plan["output"] = str(out / "lifecycle")
    lifecycle_plan_path = out / "lifecycle-plan.json"
    write_new(lifecycle_plan_path, lifecycle_plan)
    write_new(out / "intent.json", {
        "schema": "stairs_capture_host_intent_v1", "registration_sha256": sha(registration_path),
        "source_anchor": registration["source_anchor"], "clips": [row[0] for row in CLIPS],
        "checkpoint_sha256": {role: CHECKPOINTS[role][2] for role in CHECKPOINTS},
        "new_vm": 0, "install": 0, "new_training": 0, "model_api": 0,
    })
    lifecycle = _load("stairs_host_main_lifecycle_v5", lifecycle_source)
    with lifecycle.MainLifecycle(lifecycle_plan_path) as owner:
        result = run_sequence(owner, registration, checkpoints, out, backup)
    write_new(out / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--lifecycle-source", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(execute(args.registration, args.plan, args.lifecycle_source,
                             args.checkpoint_dir, args.out, args.backup), ensure_ascii=False))


if __name__ == "__main__":
    main()
