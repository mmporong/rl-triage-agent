"""부모 쪽 공용 U1 실행 API: 설계자 밖 결함을 건 학습·probe 하나·교정 학습을 소유 Job으로 돌리고 산출물을 남긴다.

U1 시연(evals/e2e_demo_u1.py)과 D3가 import해 쓴다. 각 함수는 seed·run_name·출력 경로를 인자로 받고, Isaac 쪽
bench/upstream_isaac.py를 Isaac Sim 번들 파이썬(<IsaacLab>/_isaac_sim/python.bat)으로 Isaac Lab 폴더에서 부른다. 소유 Job과
watchdog은 run_runtime_readback_smoke.run_owned다(owned 인자로 바꿀 수 있다). GPU 사용 여부 확인은 호출자가 한다.

- train: run_case.ps1과 같은 train.py 인자·meta 키·텔레메트리 위치. 확인 줄 [upstream_fault](ok)과 [upstream_aux]가 하나씩
  없으면 거부한다.
- probe: probe 하나를 결함을 건 환경에서 잰다. run_probes(--per-probe-output --metered)·loop.isaac_execute와 같은
  산출물(evals/results/<tag>/probes·resources·execution)을 쓴다. 확인 줄이 없거나 ok가 아니면 측정값을 버리고 종료 코드 1로
  돌려줘 고리가 unknown으로 받는다. loop_executor는 loop.run에 넘길 실행·사전 확인 함수를 만든다.
- correction_train: capture(원본 확인) → 결함 주입 → apply(등록 교정 하나) 뒤 처음부터 학습한다(receipt·runner hook 없는
  단순 경로). 교정 readback은 등록 표적만 대조하므로 그 밖 함수 상태는 보조 지문 파일을 본다.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

ENTRY = ROOT / "bench" / "upstream_isaac.py"
PROBES = ROOT / "evals" / "probes.py"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
RUN_NAME_RE = r"[A-Za-z0-9_]{1,80}"
RESULTS = ROOT / "evals" / "results"


def sha(path: Path, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n")


def finished(execution: dict) -> bool:
    """소유 Job이 배정됐고 시간 초과·예외 없이 스스로 끝났으며 남은 프로세스가 없다. 종료 코드는 따로 본다."""
    return (execution.get("job_assigned_before_release") is True and execution.get("natural_exit") is True
            and execution.get("timed_out") is False and execution.get("error_type") is None
            and execution.get("cleanup_verified") is True)


def lines(text: str, tag: str) -> list[dict]:
    """로그에서 '<tag> <JSON>' 줄을 모두 읽는다. 해석할 수 없는 줄은 None으로 남긴다."""
    rows = []
    for line in text.splitlines():
        if line.startswith(tag + " "):
            try:
                rows.append(json.loads(line[len(tag) + 1:]))
            except json.JSONDecodeError:
                rows.append(None)
    return rows


def readbacks(folder: Path) -> tuple[dict | None, dict | None]:
    """소유 Job 폴더 stdout.log의 확인 줄([upstream_fault])과 보조 지문 줄([upstream_aux]). 하나씩이 아니면 None."""
    text = (folder / "stdout.log").read_text(encoding="utf-8", errors="replace") if (folder / "stdout.log").exists() else ""
    fault, aux = lines(text, "[upstream_fault]"), lines(text, "[upstream_aux]")
    return (fault[0] if len(fault) == 1 else None), (aux[0] if len(aux) == 1 else None)


def _python(isaaclab: Path) -> str:
    return str(Path(isaaclab) / "_isaac_sim" / "python.bat")


def _run(owned):
    if owned is not None:
        return owned
    import run_runtime_readback_smoke as P
    return P.run_owned


def _check_name(run_name: str) -> None:
    if not isinstance(run_name, str) or not re.fullmatch(RUN_NAME_RE, run_name):
        raise ValueError(f"run_name은 영문·숫자·밑줄만 쓴다: {run_name!r}")


def find_run_dirs(isaaclab: Path, run_name: str) -> list[Path]:
    log_root = Path(isaaclab) / "logs/rsl_rl/unitree_go2_flat"
    if not log_root.is_dir():
        return []
    return sorted(d for d in log_root.iterdir()
                  if d.is_dir() and re.fullmatch(rf"\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d_{re.escape(run_name)}", d.name))


def train_args(seed: int, run_name: str, num_envs: int, max_iterations: int) -> list[str]:
    """run_case.ps1과 같은 train.py 인자."""
    return ["--task", TASK, "--num_envs", str(num_envs), "--max_iterations", str(max_iterations), "--seed", str(seed),
            "--headless", "--run_name", run_name]


def _launch(folder: Path, argv: list[str], isaaclab: Path, watchdog_s: float, owned) -> dict:
    folder.mkdir(parents=True)
    execution = _run(owned)(folder, argv, Path(isaaclab), watchdog_s)
    write_new(folder / "execution.json", execution)
    return execution


def _single_run_dir(isaaclab: Path, run_name: str) -> Path:
    run_dirs = find_run_dirs(isaaclab, run_name)
    if len(run_dirs) != 1:
        raise ValueError(f"{run_name}: SDK 실행 폴더가 하나가 아니다({len(run_dirs)})")
    return run_dirs[0]


# --- 학습 ---
def train(isaaclab: Path, *, fault: str, seed: int, run_name: str, telemetry_path: Path, meta_path: Path, work_dir: Path,
          num_envs: int = 4096, max_iterations: int = 300, watchdog_s: float = 1800, extract_watchdog_s: float = 300,
          owned=None) -> dict:
    """결함을 건 학습 한 번과 텔레메트리 추출. meta는 run_case.ps1과 같은 키에 확인 줄·보조 지문을 더한다."""
    _check_name(run_name)
    telemetry_path, meta_path, work_dir = Path(telemetry_path), Path(meta_path), Path(work_dir)
    if telemetry_path.exists() or meta_path.exists() or work_dir.exists() or find_run_dirs(isaaclab, run_name):
        raise FileExistsError(f"{run_name}: 학습 산출물이 이미 있다. 덮어쓰지 않는다")
    argv = [_python(isaaclab), str(ENTRY), "train", "--fault", fault, "--",
            *train_args(seed, run_name, num_envs, max_iterations)]
    execution = _launch(work_dir / "train", argv, isaaclab, watchdog_s, owned)
    readback, aux = readbacks(work_dir / "train")
    if not finished(execution) or execution.get("exit_code") != 0:
        raise RuntimeError(f"{run_name}: 학습이 끝나지 않았다(exit={execution.get('exit_code')}). {work_dir / 'train'}의 로그를 본다")
    if readback is None or readback.get("ok") is not True or readback.get("fault") != fault or aux is None:
        raise ValueError(f"{run_name}: 결함 확인 줄([upstream_fault] ok)이나 보조 지문 줄이 하나가 아니다")
    run_dir = _single_run_dir(isaaclab, run_name)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    extract = _launch(work_dir / "extract", [_python(isaaclab), str(ROOT / "bench" / "extract_tb.py"), str(run_dir),
                                             str(telemetry_path)], isaaclab, extract_watchdog_s, owned)
    # 추출은 성공해도 0이 아닌 종료 코드를 남길 때가 있어 결과 파일로 판정한다(run_case.ps1과 같다).
    if not finished(extract) or not telemetry_path.is_file():
        raise RuntimeError(f"{run_name}: 텔레메트리 추출이 끝나지 않았거나 파일이 없다")
    case_id = run_name[: -len(f"_s{seed}")] if run_name.endswith(f"_s{seed}") else run_name
    meta = {"case_id": case_id, "seed": seed, "num_envs": num_envs, "max_iterations": max_iterations, "task": TASK,
            "fault": fault, "train_exit_code": execution["exit_code"], "wall_time_s": round(execution["wall_s"], 1),
            "run_dir": str(run_dir), "entry": "bench/upstream_isaac.py train",
            "upstream_readback": readback, "upstream_aux": aux}
    write_new(meta_path, meta)
    return {"meta_path": str(meta_path), "meta": meta, "execution": execution, "extract": extract, "run_dir": str(run_dir),
            "telemetry_path": str(telemetry_path)}


# --- probe 하나 ---
def probe_paths(tag: str, run_name: str, probe: str, results: Path | None = None) -> dict[str, Path]:
    base = (results or RESULTS) / tag
    name = f"{run_name}__{probe}.json"
    return {"report": base / "probes" / name, "resources": base / "resources" / name, "execution": base / "execution" / name}


def _last_checkpoint(run_dir: Path) -> Path:
    from run_fixed_eval import last_checkpoint
    return last_checkpoint(run_dir)


def probe(isaaclab: Path, *, fault: str, seed: int, run_name: str, probe: str, tag: str, work_dir: Path,
          watchdog_s: float = 600, owned=None, results: Path | None = None) -> dict:
    """결함을 건 환경에서 probe 하나를 잰다(계측 포함). 측정값은 확인 줄이 ok일 때만 돌려준다."""
    from rl_triage.probe_loop import PROBES as REGISTERED
    _check_name(run_name)
    if probe not in REGISTERED:
        raise ValueError(f"모르는 probe {probe!r}")
    paths = probe_paths(tag, run_name, probe, results)
    work_dir = Path(work_dir)
    if any(p.exists() for p in paths.values()) or work_dir.exists():
        raise FileExistsError(f"{paths['report'].name}: 기존 측정을 재사용하지 않는다. 새 tag를 쓴다")
    checkpoint = _last_checkpoint(_single_run_dir(isaaclab, run_name))
    work_dir.mkdir(parents=True)
    jobs_file, metrics = work_dir / "jobs.json", work_dir / "metrics.json"
    # 동결 probes.run은 jobs의 fault로 hidden_faults를 건다. NONE으로 두어 아무것도 하지 않게 하고 결함은 upstream_isaac가 건다.
    write_new(jobs_file, {"fault": "NONE", "upstream_fault": fault, "seed": seed, "jobs": [
        {"name": run_name, "checkpoint": str(checkpoint), "probes": [probe], "output": str(paths["report"])}]})
    argv = [_python(isaaclab), str(ENTRY), "probes", "--fault", fault, "--", "--jobs", str(jobs_file),
            "--metrics", str(metrics), "--headless"]
    started_at = datetime.now(timezone.utc).isoformat()
    cpu0 = time.process_time()
    execution = _launch(work_dir / "run", argv, isaaclab, watchdog_s, owned)
    parent_cpu = time.process_time() - cpu0
    readback, aux = readbacks(work_dir / "run")
    readback_ok = readback is not None and readback.get("ok") is True and readback.get("fault") == fault and aux is not None
    report = json.loads(paths["report"].read_text(encoding="utf-8")) if paths["report"].exists() else {}
    record = report.get("probes", {}).get(probe, {})
    meter_path = metrics.with_suffix(".post_close.json") if metrics.with_suffix(".post_close.json").exists() else metrics
    meter = json.loads(meter_path.read_text(encoding="utf-8")) if meter_path.exists() else None
    report_sha = sha(paths["report"], True) if paths["report"].exists() else None
    rows = (meter or {}).get("measurements") or []
    meter_ok = (meter is not None and meter.get("jobs_sha256") == sha(jobs_file) and meter.get("completed") is True
                and meter.get("probes_sha256_lf") == sha(PROBES, True) and len(rows) == 1 and rows[0].get("probe") == probe
                and rows[0].get("checkpoint_sha256") == report.get("checkpoint", {}).get("sha256"))
    code = execution.get("exit_code") if execution.get("exit_code") != 0 else record.get("exit")
    if not finished(execution) or not readback_ok or not meter_ok:
        code = code if code not in (0, None) else 1
    if meter_ok and report_sha is not None:
        write_new(paths["resources"], {
            "schema": "probe_resources_v1", "case": run_name, "probe": probe, "checkpoint_sha256": rows[0]["checkpoint_sha256"],
            "report_sha256_lf": report_sha, "cpu_s": meter["cpu_s"] + parent_cpu, "launcher_exit_code": execution.get("exit_code"),
            "cpu_scope": "isaac_python_wrapper_and_probe_driver_processes", "isaac_cpu_scope": meter["cpu_scope"],
            "app_close_cpu_included": meter["phase"] == "after_main", "simulator_steps_scope": meter["simulator_steps_scope"],
            "meter_sha256_lf": sha(meter_path, True), "probes_sha256_lf": meter["probes_sha256_lf"],
            "upstream_fault": fault, **rows[0]})
    resource = json.loads(paths["resources"].read_text(encoding="utf-8")) if paths["resources"].exists() else None
    wall = execution.get("wall_s")
    meta = {"schema": "probe_execution_v1", "case": run_name, "probe": probe,
            "checkpoint_sha256": report.get("checkpoint", {}).get("sha256"), "report_sha256_lf": report_sha,
            "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "gpu_wall_s": None if wall is None else round(wall, 1), "execution_wall_s": None if wall is None else round(wall, 1),
            "exit_code": code, "parent_cpu_s": parent_cpu,
            "cpu_s": None if resource is None else resource["cpu_s"],
            "simulator_steps": None if resource is None else resource["simulator_steps"], "metered": True,
            "upstream_fault": fault, "upstream_readback_ok": readback_ok}
    write_new(paths["execution"], meta)
    return {"measurement": record.get("measurement") if code == 0 else None, "gpu_s": wall or 0.0, "exit_code": code,
            "readback": readback, "aux": aux, "paths": {k: str(v) for k, v in paths.items()}}


def loop_executor(isaaclab: Path, *, fault: str, seed: int, tag: str, work_root: Path, watchdog_s: float = 600,
                  owned=None, results: Path | None = None):
    """loop.run(case, rid, execute, preflight=...)에 넘길 (execute, preflight). case는 결함 실행 이름이다."""
    def preflight(case: str, name: str) -> None:
        paths = probe_paths(tag, case, name, results)
        if any(p.exists() for p in paths.values()):
            raise SystemExit(f"{paths['report'].name}: 기존 측정을 새 승인으로 재사용하지 않는다. 새 tag를 쓴다")

    def execute(case: str, name: str) -> tuple[dict | None, float, int | None]:
        out = probe(isaaclab, fault=fault, seed=seed, run_name=case, probe=name, tag=tag,
                    work_dir=Path(work_root) / f"{case}__{name}", watchdog_s=watchdog_s, owned=owned, results=results)
        return out["measurement"], out["gpu_s"], out["exit_code"]

    return execute, preflight


# --- 교정 학습(단순 경로) ---
def correction_train(isaaclab: Path, *, fault: str, correction_id: str, seed: int, run_name: str, work_dir: Path,
                     contract: dict, num_envs: int = 4096, max_iterations: int = 300, watchdog_s: float = 1800,
                     owned=None) -> dict:
    """capture → 결함 → apply(등록 교정 하나) 뒤 처음부터 학습한다. 교정 readback·보조 지문·체크포인트를 돌려준다."""
    _check_name(run_name)
    work_dir = Path(work_dir)
    if work_dir.exists() or find_run_dirs(isaaclab, run_name):
        raise FileExistsError(f"{run_name}: 교정 학습 산출물이 이미 있다. 덮어쓰지 않는다")
    work_dir.mkdir(parents=True)
    contract_path, readback_path = work_dir / "contract.json", work_dir / "correction.json"
    write_new(contract_path, contract)
    argv = [_python(isaaclab), str(ENTRY), "correct", "--fault", fault, "--correction", correction_id,
            "--contract", str(contract_path), "--readback", str(readback_path), "--",
            *train_args(seed, run_name, num_envs, max_iterations)]
    execution = _launch(work_dir / "train", argv, isaaclab, watchdog_s, owned)
    if not finished(execution) or execution.get("exit_code") != 0:
        raise RuntimeError(f"{run_name}: 교정 학습이 끝나지 않았다(exit={execution.get('exit_code')})")
    aux_path = readback_path.with_name(readback_path.stem + ".aux.json")
    if not readback_path.is_file() or not aux_path.is_file():
        raise ValueError(f"{run_name}: 교정 readback이나 보조 지문 파일이 없다")
    correction = json.loads(readback_path.read_text(encoding="utf-8"))
    if correction.get("registered_target_state") != "restored_to_clean" or correction.get("correction_id") != correction_id:
        raise ValueError(f"{run_name}: 등록 표적이 원본으로 돌아가지 않았다")
    run_dir = _single_run_dir(isaaclab, run_name)
    checkpoint = run_dir / f"model_{max_iterations - 1}.pt"
    if not checkpoint.is_file():
        raise ValueError(f"{run_name}: 마지막 체크포인트가 없다")
    return {"execution": execution, "correction": correction, "aux": json.loads(aux_path.read_text(encoding="utf-8")),
            "readback": readbacks(work_dir / "train")[0], "run_dir": str(run_dir),
            "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)}}
