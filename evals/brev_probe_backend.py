"""Brev transport boundary for frozen active-diagnosis Isaac probes.

The model remains on the Windows/WSL path.  This module only checks remote GPU
readiness and runs the already registered probe catalogue on one pre-existing
Brev instance.  All public failures are typed and deliberately omit command,
credential, host, and absolute-path text.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
import uuid


DEFAULT_INSTANCE = "marked-lavender-warbler"
DEFAULT_WORKSPACE = "rl-triage-active-evidence-20261011"
TRANSPORT_TIMEOUT_S = 120.0
EXPECTED_ENVIRONMENT = {"ubuntu": "22.04", "python": "3.10.12", "isaac_sim": "4.5.0.0",
                        "isaac_lab_sha": "90b79bb2d44feb8d833f260f2bf37da3487180ba",
                        "torch": "2.7.0+cu128", "cuda": "12.8.57", "rsl_rl": "2.3.3", "gymnasium": "1.2.0",
                        "python_relpath": "venv/bin/python", "isaaclab_relpath": "IsaacLab",
                        "owner_marker_basename": ".rl-triage-active-evidence-owner.json"}
INSTANCE_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+){2,5}")
SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
MARKER_RE = re.compile(r"^RL_TRIAGE_JSON:([A-Za-z0-9_-]+)$")
REMOTE_PREP_CODE = (
    "import base64,json,pathlib,sys; ps=[pathlib.Path(x) for x in sys.argv[1:]]; "
    "ok=not any(p.exists() for p in ps); [p.parent.mkdir(parents=True,exist_ok=True) for p in ps]; "
    "v={'ok':ok}; print('RL_TRIAGE_JSON:'+base64.urlsafe_b64encode(json.dumps(v,separators=(',',':')).encode()).decode().rstrip('='))"
)


class BrevBackendError(RuntimeError):
    """Safe, typed cloud-boundary failure."""

    def __init__(self, stage: str, error_type: str = "BrevBackendError"):
        self.stage = stage
        self.error_type = error_type
        self.launched = False
        self.launcher_wall_s: float | None = 0.0
        super().__init__(f"{stage}: {error_type}")


class BrevGPUWaitError(RuntimeError):
    def __init__(self, snapshot: dict):
        self.snapshot = snapshot
        super().__init__("Brev GPU is busy")


class BrevCleanupUnverified(BrevBackendError):
    """The transport failed after launch, so owned-process cleanup is unproven."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def windows_to_wsl(path: Path) -> str:
    value = path.resolve().as_posix()
    match = re.fullmatch(r"([A-Za-z]):/(.+)", value)
    if match is None:
        raise BrevBackendError("local transport path", "PathContractError")
    return f"/mnt/{match[1].lower()}/{match[2]}"


def validate_backend_config(value: object) -> dict:
    if not isinstance(value, dict) or value.get("name") not in {"native_windows", "brev"}:
        raise ValueError("probe_backend 형식 오류")
    if value["name"] == "native_windows":
        if set(value) != {"name"}:
            raise ValueError("native_windows backend에 미등록 필드가 있다")
        return {"name": "native_windows"}
    if set(value) != {"name", "instance", "workspace_basename", "environment", "owner_marker_sha256"}:
        raise ValueError("brev backend 필드 불일치")
    instance, workspace = value.get("instance"), value.get("workspace_basename")
    if not isinstance(instance, str) or INSTANCE_RE.fullmatch(instance) is None:
        raise ValueError("Brev instance 이름 형식 오류")
    if not isinstance(workspace, str) or SLUG_RE.fullmatch(workspace) is None:
        raise ValueError("Brev workspace 이름 형식 오류")
    if value.get("environment") != EXPECTED_ENVIRONMENT:
        raise ValueError("Brev environment 고정값 불일치")
    owner_sha = value.get("owner_marker_sha256")
    if not isinstance(owner_sha, str) or re.fullmatch(r"[0-9a-f]{64}", owner_sha) is None:
        raise ValueError("Brev owner marker SHA 형식 오류")
    return {"name": "brev", "instance": instance, "workspace_basename": workspace,
            "environment": dict(EXPECTED_ENVIRONMENT), "owner_marker_sha256": owner_sha}


def backend_config(name: str, instance: str = DEFAULT_INSTANCE,
                   workspace_basename: str = DEFAULT_WORKSPACE,
                   owner_marker_sha256: str | None = None) -> dict:
    value: dict[str, object] = {"name": name}
    if name == "brev":
        value.update(instance=instance, workspace_basename=workspace_basename,
                     environment=dict(EXPECTED_ENVIRONMENT), owner_marker_sha256=owner_marker_sha256)
    return validate_backend_config(value)


def environment_matches(actual: object, expected: object = EXPECTED_ENVIRONMENT) -> bool:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    exact = ("ubuntu", "python", "isaac_sim", "isaac_lab_sha", "torch", "cuda", "rsl_rl", "gymnasium",
             "python_relpath", "isaaclab_relpath", "owner_marker_basename")
    return all(actual.get(k) == expected.get(k) for k in exact)


def _encoded_marker(value: dict) -> str:
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    return "RL_TRIAGE_JSON:" + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def parse_marker(stdout: str) -> dict:
    matches = [m for line in stdout.splitlines() if (m := MARKER_RE.fullmatch(line.strip()))]
    if len(matches) != 1:
        raise BrevBackendError("transport envelope", "MarkerContractError")
    token = matches[0][1]
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        value = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as error:
        raise BrevBackendError("transport envelope", "MarkerContractError") from error
    if not isinstance(value, dict):
        raise BrevBackendError("transport envelope", "MarkerContractError")
    return value


def _remote_python(code: str, *args: str, executable: str = "python3") -> str:
    """Return one quoted command; code is fixed by this module and args are data."""
    return shlex.join([executable, "-c", code, *args])


def validate_probe_artifacts(report: object, meter: object, *, case: str, probe: str,
                             checkpoint_sha256: str, probes_sha256_lf: str,
                             jobs_sha256: str) -> tuple[dict, dict]:
    if not isinstance(report, dict) or not isinstance(meter, dict):
        raise BrevBackendError("artifact verification", "ArtifactBindingMismatch")
    rows = meter.get("measurements")
    if (report.get("name") != case or not isinstance(report.get("checkpoint"), dict)
            or report["checkpoint"].get("sha256") != checkpoint_sha256
            or not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
            or rows[0].get("probe") != probe or rows[0].get("checkpoint_sha256") != checkpoint_sha256
            or rows[0].get("completed") is not True or meter.get("completed") is not True
            or meter.get("jobs_sha256") != jobs_sha256
            or meter.get("probes_sha256_lf") != probes_sha256_lf):
        raise BrevBackendError("artifact verification", "ArtifactBindingMismatch")
    return report, meter


def contains_private_text(value: object) -> bool:
    text = json.dumps(value, ensure_ascii=False)
    patterns = (r"[A-Za-z]:[\\/]", r"/(?:home|mnt|root|Users)/", r"(?i)bearer\s+",
                r"(?i)(?:api[_-]?key|token|credential)\s*[:=]")
    return any(re.search(pattern, text) for pattern in patterns)


def validate_cleanup_evidence(outcome: object) -> dict:
    """Require proof that the owned process group and GPU work are gone."""
    if not isinstance(outcome, dict):
        raise BrevCleanupUnverified("probe cleanup", "CleanupEvidenceMissing")
    pids = outcome.get("cleanup_compute_pids")
    memory = outcome.get("cleanup_memory_mib")
    valid = (
        outcome.get("cleanup_checked") is True
        and outcome.get("cleanup_check_error") is False
        and outcome.get("cleanup_group_alive") is False
        and isinstance(pids, list) and not pids
        and isinstance(memory, int) and not isinstance(memory, bool) and 0 <= memory <= 2500
        and outcome.get("cleanup_confirmed") is True
        and isinstance(outcome.get("cleanup_wall_s"), (int, float))
        and not isinstance(outcome.get("cleanup_wall_s"), bool)
        and outcome["cleanup_wall_s"] >= 0
    )
    if not valid:
        raise BrevCleanupUnverified("probe cleanup", "CleanupUnverified")
    return {"cleanup_checked": True, "cleanup_check_error": False,
            "cleanup_group_alive": False, "cleanup_compute_pids": [],
            "cleanup_memory_mib": memory, "cleanup_wall_s": outcome["cleanup_wall_s"]}


class BrevProbeBackend:
    def __init__(self, root: Path, protocol: dict, *, runner=subprocess.run):
        config = validate_backend_config(protocol.get("probe_backend"))
        if config["name"] != "brev":
            raise ValueError("BrevProbeBackend에는 brev 설정이 필요하다")
        self.root = Path(root)
        self.protocol = protocol
        self.instance = config["instance"]
        self.workspace_basename = config["workspace_basename"]
        self.runner = runner
        self._home: str | None = None
        self._wsl_home: str | None = None

    def _remaining(self, deadline: float | None) -> float:
        if deadline is None:
            return TRANSPORT_TIMEOUT_S
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BrevBackendError("probe execution", "BudgetExceeded")
        return remaining

    def _run(self, args: list[str], stage: str, deadline: float | None = None) -> subprocess.CompletedProcess:
        try:
            proc = self.runner(args, cwd=self.root, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=self._remaining(deadline))
        except subprocess.TimeoutExpired as error:
            raise BrevBackendError(stage, "TransportTimeout") from error
        except (OSError, subprocess.SubprocessError) as error:
            raise BrevBackendError(stage, type(error).__name__) from error
        if proc.returncode != 0:
            raise BrevBackendError(stage, "TransportFailed")
        return proc

    def _exec(self, command: str, stage: str, deadline: float | None = None) -> dict:
        proc = self._run(["wsl", "-d", "Ubuntu", "--", self._brev_executable(deadline),
                          "exec", self.instance, command],
                         stage, deadline)
        return parse_marker(proc.stdout)

    def _copy(self, source: str, destination: str, stage: str, deadline: float | None) -> None:
        self._run(["wsl", "-d", "Ubuntu", "--", self._brev_executable(deadline),
                   "copy", source, destination], stage, deadline)

    def _brev_executable(self, deadline: float | None = None) -> str:
        if self._wsl_home is None:
            proc = self._run(["wsl", "-d", "Ubuntu", "--", "printenv", "HOME"],
                             "Brev CLI discovery", deadline)
            home = proc.stdout.strip()
            if (not home or "\n" in home or re.fullmatch(r"/[A-Za-z0-9._/-]+", home) is None
                    or ".." in home.split("/")):
                raise BrevBackendError("Brev CLI discovery", "PathContractError")
            self._wsl_home = home.rstrip("/")
        return f"{self._wsl_home}/.local/bin/brev"

    def remote_home(self, deadline: float | None = None) -> str:
        if self._home is None:
            code = "import base64,json,os; v={'home':os.path.expanduser('~')}; print('RL_TRIAGE_JSON:'+base64.urlsafe_b64encode(json.dumps(v,separators=(',',':')).encode()).decode().rstrip('='))"
            value = self._exec(_remote_python(code), "remote home", deadline)
            home = value.get("home")
            if not isinstance(home, str) or re.fullmatch(r"/[A-Za-z0-9._/-]+", home) is None or ".." in home.split("/"):
                raise BrevBackendError("remote home", "RemoteContractError")
            self._home = home.rstrip("/")
        return self._home

    def workspace(self, deadline: float | None = None) -> str:
        return f"{self.remote_home(deadline)}/{self.workspace_basename}"

    def _verify_sources(self, deadline: float | None) -> None:
        expected = self.protocol.get("code_sha256_lf")
        if not isinstance(expected, dict):
            raise BrevBackendError("source verification", "ProtocolContractError")
        payload = base64.urlsafe_b64encode(json.dumps(expected, separators=(",", ":")).encode()).decode()
        code = (
            "import base64,hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); "
            "e=json.loads(base64.urlsafe_b64decode(sys.argv[2])); "
            "a={k:hashlib.sha256((r/k).read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest() for k in e}; "
            "ok=a==e; v={'ok':ok}; print('RL_TRIAGE_JSON:'+base64.urlsafe_b64encode(json.dumps(v,separators=(',',':')).encode()).decode().rstrip('='))"
        )
        value = self._exec(_remote_python(code, self.workspace(deadline), payload),
                           "source verification", deadline)
        if value != {"ok": True}:
            raise BrevBackendError("source verification", "SourceDigestMismatch")

    def gpu_snapshot(self) -> dict:
        code = (
            "import base64,json,subprocess; "
            "q=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]; "
            "u,m,t=[int(x.strip()) for x in q.split(',')]; "
            "p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True); "
            "pids=sorted({int(x.strip()) for x in p.stdout.splitlines() if x.strip().isdigit()}); "
            "reasons=(['other_gpu_compute'] if pids else [])+(['unexplained_large_allocation'] if m>2500 else []); "
            "v={'gpu_utilization_pct':u,'memory_mib':m,'memory_total_mib':t,'other_compute_pids':pids,'ready':not reasons,'busy_reasons':reasons}; "
            "print('RL_TRIAGE_JSON:'+base64.urlsafe_b64encode(json.dumps(v,separators=(',',':')).encode()).decode().rstrip('='))"
        )
        value = self._exec(_remote_python(code), "gpu snapshot")
        required = {"gpu_utilization_pct", "memory_mib", "memory_total_mib",
                    "other_compute_pids", "ready", "busy_reasons"}
        if set(value) != required or not isinstance(value.get("ready"), bool):
            raise BrevBackendError("gpu snapshot", "RemoteContractError")
        if not value["ready"]:
            raise BrevGPUWaitError(value)
        return value

    def preflight(self, case: str, probe: str, tag: str) -> None:
        for value in (case, probe, tag):
            if not isinstance(value, str) or SLUG_RE.fullmatch(value) is None:
                raise BrevBackendError("probe preflight", "SlugValidationError")
        if probe not in self.protocol.get("probe_arguments", {}):
            raise BrevBackendError("probe preflight", "ProbeCatalogError")
        for folder in ("probes", "execution", "resources", "meters"):
            if (self.root / "evals/results" / tag / folder / f"{case}__{probe}.json").exists():
                raise BrevBackendError("probe preflight", "NoOverwriteError")

    def _ensure_remote_checkpoint(self, local: Path, remote: str, expected: str,
                                  deadline: float | None) -> None:
        if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
            raise BrevBackendError("checkpoint verification", "CheckpointDigestMismatch")
        check_code = (
            "import base64,hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); e=sys.argv[2]; "
            "a=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None; "
            "v={'exists':p.exists(),'matches':a==e}; print('RL_TRIAGE_JSON:'+base64.urlsafe_b64encode(json.dumps(v,separators=(',',':')).encode()).decode().rstrip('='))"
        )
        state = self._exec(_remote_python(check_code, remote, expected), "checkpoint preflight", deadline)
        if state.get("exists") and not state.get("matches"):
            raise BrevBackendError("checkpoint preflight", "CheckpointConflict")
        if state.get("exists"):
            return
        workspace = self.workspace(deadline)
        upload = f"{workspace}/private/uploads/{local.stem}-{uuid.uuid4().hex}{local.suffix}"
        prepared = self._exec(_remote_python(REMOTE_PREP_CODE, upload),
                              "checkpoint upload preflight", deadline)
        if prepared != {"ok": True}:
            raise BrevBackendError("checkpoint upload preflight", "NoOverwriteError")
        self._copy(windows_to_wsl(local), f"{self.instance}:{upload}", "checkpoint copy", deadline)
        install_code = (
            "import base64,hashlib,json,os,pathlib,sys; s=pathlib.Path(sys.argv[1]); d=pathlib.Path(sys.argv[2]); e=sys.argv[3]; "
            "d.parent.mkdir(parents=True,exist_ok=True); a=hashlib.sha256(s.read_bytes()).hexdigest(); "
            "ok=a==e and not d.exists(); os.link(s,d) if ok else None; s.unlink(missing_ok=True); "
            "v={'ok':ok}; print('RL_TRIAGE_JSON:'+base64.urlsafe_b64encode(json.dumps(v,separators=(',',':')).encode()).decode().rstrip('='))"
        )
        installed = self._exec(_remote_python(install_code, upload, remote, expected),
                               "checkpoint install", deadline)
        if installed != {"ok": True}:
            raise BrevBackendError("checkpoint install", "CheckpointDigestMismatch")

    def _fault(self, case: str) -> tuple[str, int]:
        import run_probes
        key = json.loads(run_probes.KEY_PATH.read_text(encoding="utf-8")) if run_probes.KEY_PATH.exists() else None
        key_t1 = json.loads(run_probes.T1_KEY_PATH.read_text(encoding="utf-8")) if run_probes.T1_KEY_PATH.exists() else None
        try:
            return run_probes.fault_of(case, key, key_t1), int(case.rsplit("_s", 1)[1])
        except (OSError, ValueError, KeyError, TypeError, SystemExit) as error:
            raise BrevBackendError("private job creation", type(error).__name__) from error

    def _verify_environment(self) -> dict:
        workspace = self.workspace()
        code = (
            "import base64,hashlib,importlib.metadata as m,json,pathlib,platform,subprocess,sys; "
            "o=dict(x.split('=',1) for x in pathlib.Path('/etc/os-release').read_text().splitlines() if '=' in x); "
            "ver=lambda *ns: next((m.version(n) for n in ns if any((d.metadata.get('Name') or '').lower()==n.lower() for d in m.distributions())),None); "
            "v={'ubuntu':o.get('VERSION_ID','').strip(chr(34)),'ubuntu_version':o.get('VERSION','').strip(chr(34)),'python':platform.python_version(),"
            "'isaac_sim':ver('isaacsim'),'isaac_lab_sha':subprocess.check_output(['git','-C',sys.argv[1]+'/IsaacLab','rev-parse','HEAD'],text=True).strip(),"
            "'torch':ver('torch'),'cuda':ver('nvidia-cuda-runtime-cu12'),'rsl_rl':ver('rsl-rl-lib','rsl_rl'),'gymnasium':ver('gymnasium'),"
            "'python_relpath':'venv/bin/python','isaaclab_relpath':'IsaacLab',"
            "'owner_marker_basename':'.rl-triage-active-evidence-owner.json',"
            "'owner_marker_sha256':hashlib.sha256((pathlib.Path(sys.argv[1])/'.rl-triage-active-evidence-owner.json').read_bytes()).hexdigest()}; "
            "print('RL_TRIAGE_JSON:'+base64.urlsafe_b64encode(json.dumps(v,separators=(',',':')).encode()).decode().rstrip('='))"
        )
        actual = self._exec(_remote_python(code, workspace,
                                           executable=f"{workspace}/venv/bin/python"),
                            "environment verification")
        expected = self.protocol["probe_backend"]["environment"]
        if (not environment_matches(actual, expected)
                or actual.get("owner_marker_sha256") != self.protocol["probe_backend"]["owner_marker_sha256"]):
            raise BrevBackendError("environment verification", "EnvironmentMismatch")
        return actual

    def prepare_execution(self, case: str, probe: str, tag: str, checkpoint: Path,
                          checkpoint_sha256: str, checkpoint_dependencies: dict | None,
                          owner_binding: dict) -> dict:
        """Stage immutable private inputs and return a public approval binding."""
        self.preflight(case, probe, tag)
        started_perf = time.perf_counter()
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha256:
            raise BrevBackendError("checkpoint verification", "CheckpointDigestMismatch")
        self._verify_sources(None)
        environment = self._verify_environment()
        workspace = self.workspace()
        remote_checkpoint = f"{workspace}/private/checkpoints/{case}/{checkpoint.name}"
        self._ensure_remote_checkpoint(checkpoint, remote_checkpoint, checkpoint_sha256, None)
        checkpoint_set = [{"role": "last", "file": checkpoint.name, "sha256": checkpoint_sha256}]
        dependency = (checkpoint_dependencies or {}).get(probe)
        if dependency is not None:
            if (not isinstance(dependency, dict) or set(dependency) != {"file", "sha256"}
                    or SLUG_RE.fullmatch(str(dependency.get("file"))) is None):
                raise BrevBackendError("checkpoint dependency", "ProtocolContractError")
            dependency_path = checkpoint.with_name(dependency["file"])
            dependency_remote = f"{workspace}/private/checkpoints/{case}/{dependency['file']}"
            self._ensure_remote_checkpoint(dependency_path, dependency_remote, dependency["sha256"], None)
            checkpoint_set.append({"role": probe + ":from", **dependency})
        fault, seed = self._fault(case)
        token = uuid.uuid4().hex
        remote_report = f"{workspace}/results/{tag}/probes/{case}__{probe}.json"
        remote_metrics = f"{workspace}/private/metrics/{tag}-{case}-{probe}-{token}.json"
        remote_jobs = f"{workspace}/private/jobs/{tag}-{case}-{probe}-{token}.json"
        remote_log = f"{workspace}/private/logs/{tag}-{case}-{probe}-{token}.txt"
        jobs = {"fault": fault, "seed": seed, "jobs": [{"name": case, "checkpoint": remote_checkpoint,
                "probes": [probe], "output": remote_report}]}
        private_root = self.root / "bench/private/active_cloud"
        private_root.mkdir(parents=True, exist_ok=True)
        jobs_path = private_root / f"jobs_{tag}_{case}_{probe}_{token}.json"
        with jobs_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(jobs, handle, indent=1); handle.write("\n")
        prepared = self._exec(_remote_python(REMOTE_PREP_CODE, remote_jobs, remote_log, remote_metrics,
                                             remote_metrics.rsplit(".", 1)[0] + ".post_close.json",
                                             remote_report), "remote artifact preflight")
        if prepared != {"ok": True}:
            raise BrevBackendError("remote artifact preflight", "NoOverwriteError")
        self._copy(windows_to_wsl(jobs_path), f"{self.instance}:{remote_jobs}", "jobs copy", None)
        public = {"name": "brev", "instance": self.instance,
                  "workspace_basename": self.workspace_basename, "environment": environment,
                  "environment_digest": digest(environment),
                  "source_sha256_lf": self.protocol["code_sha256_lf"],
                  "source_set_digest": digest(self.protocol["code_sha256_lf"]),
                  "checkpoint_set": checkpoint_set,
                  "jobs_sha256": hashlib.sha256(jobs_path.read_bytes()).hexdigest(),
                  "owner_binding_digest": digest(owner_binding)}
        public["preapproval_transport_wall_s"] = time.perf_counter() - started_perf
        return {"public": public, "started_perf": started_perf, "jobs_path": jobs_path,
                "remote_checkpoint": remote_checkpoint, "remote_report": remote_report,
                "remote_metrics": remote_metrics, "remote_jobs": remote_jobs, "remote_log": remote_log}

    def execute(self, case: str, probe: str, tag: str, checkpoint: Path,
                checkpoint_sha256: str, budget_s: int,
                checkpoint_dependencies: dict | None = None,
                plan: dict | None = None, watchdog_s: float | None = None) -> tuple[dict | None, float, int | None]:
        self.preflight(case, probe, tag)
        started_at, cpu_start = now(), time.process_time()
        if budget_s != self.protocol.get("probe_execution_budget_s"):
            raise BrevBackendError("probe budget", "ProtocolContractError")
        deadline = None
        execution_state: dict[str, object] = {"launched": False, "launcher_wall_s": 0.0}
        try:
            if plan is None:
                plan = self.prepare_execution(case, probe, tag, checkpoint, checkpoint_sha256,
                                              checkpoint_dependencies, {})
            public = plan.get("public") if isinstance(plan, dict) else None
            if not isinstance(public, dict):
                raise BrevBackendError("execution plan", "ProtocolContractError")
            remote_report, remote_metrics = plan["remote_report"], plan["remote_metrics"]
            remote_jobs, remote_log = plan["remote_jobs"], plan["remote_log"]
            jobs_path = Path(plan["jobs_path"])
            workspace = self.workspace(deadline)
            private_root = self.root / "bench/private/active_cloud"
            with tempfile.TemporaryDirectory(dir=private_root) as temp_name:
                temp = Path(temp_name)
                argv = [f"{workspace}/venv/bin/python", f"{workspace}/evals/metered_probes.py",
                        "--jobs", remote_jobs, "--metrics", remote_metrics, "--headless"]
                encoded_argv = base64.urlsafe_b64encode(json.dumps(argv).encode()).decode()
                manifest = {"root": workspace, "sources": public["source_sha256_lf"],
                            "checkpoints": public["checkpoint_set"], "checkpoint_dir": f"{workspace}/private/checkpoints/{case}",
                            "jobs": remote_jobs, "jobs_sha256": public["jobs_sha256"]}
                encoded_manifest = base64.urlsafe_b64encode(json.dumps(manifest).encode()).decode()
                watchdog_s = (self.protocol.get("per_case_isaac_wall_limit_s", 360)
                              if watchdog_s is None else watchdog_s)
                if isinstance(watchdog_s, bool) or not isinstance(watchdog_s, (int, float)) or watchdog_s <= 0:
                    raise BrevBackendError("probe watchdog", "CaseWallLimitExceeded")
                if public.get("launcher_watchdog_s", watchdog_s) != watchdog_s:
                    raise BrevBackendError("probe watchdog", "ProtocolContractError")
                run_code = (
                    "import base64,hashlib,json,os,pathlib,signal,subprocess,sys,time; a=json.loads(base64.urlsafe_b64decode(sys.argv[1])); m=json.loads(base64.urlsafe_b64decode(sys.argv[6])); "
                    "log=pathlib.Path(sys.argv[2]); log.parent.mkdir(parents=True,exist_ok=True); "
                    "out=pathlib.Path(sys.argv[3]); met=pathlib.Path(sys.argv[4]); "
                    "h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest(); "
                    "source_ok=all(h(pathlib.Path(m['root'])/k)==v for k,v in m['sources'].items()); "
                    "checkpoint_ok=all(hashlib.sha256((pathlib.Path(m['checkpoint_dir'])/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in m['checkpoints']); "
                    "jobs_ok=hashlib.sha256(pathlib.Path(m['jobs']).read_bytes()).hexdigest()==m['jobs_sha256']; "
                    "q=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]; "
                    "cp=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True); "
                    "remote_busy=int(q.strip())>2500 or any(x.strip().isdigit() for x in cp.stdout.splitlines()); "
                    "blocked=out.exists(); p=None; rc=None; fh=None; timed=False; cleaned=False; launcher_wall=0.0; cleanup_wall=0.0; cleanup_checked=False; cleanup_error=False; group_alive=None; cleanup_pids=None; cleanup_memory=None; "
                    "\nif source_ok and checkpoint_ok and jobs_ok and not blocked and not remote_busy:\n"
                    " child_env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES')\n"
                    " fh=log.open('x'); p=subprocess.Popen(a,cwd=sys.argv[5],env=child_env,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)\n"
                    " def stop(sig,frame):\n"
                    "  os.killpg(p.pid,signal.SIGTERM) if p.poll() is None else None\n"
                    " signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGHUP,stop)\n"
                    " t=time.perf_counter()\n"
                    " try:\n"
                    "  try: rc=p.wait(timeout=float(sys.argv[7]))\n"
                    "  except subprocess.TimeoutExpired:\n"
                    "   timed=True; os.killpg(p.pid,signal.SIGTERM)\n"
                    "   try: p.wait(timeout=10)\n"
                    "   except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL); p.wait(timeout=10)\n"
                    "   rc=124\n"
                    " finally:\n"
                    "  launcher_wall=time.perf_counter()-t\n"
                    "  fh.close()\n"
                    " ct=time.perf_counter()\n"
                    " try:\n"
                    "  def alive():\n"
                    "   try: os.killpg(p.pid,0); return True\n"
                    "   except ProcessLookupError: return False\n"
                    "  group_alive=alive()\n"
                    "  if group_alive:\n"
                    "   os.killpg(p.pid,signal.SIGTERM); dl=time.perf_counter()+10\n"
                    "   while alive() and time.perf_counter()<dl: time.sleep(.1)\n"
                    "   if alive():\n"
                    "    os.killpg(p.pid,signal.SIGKILL); dl=time.perf_counter()+10\n"
                    "    while alive() and time.perf_counter()<dl: time.sleep(.1)\n"
                    "  group_alive=alive()\n"
                    "  cq=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)\n"
                    "  mq=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]\n"
                    "  cleanup_pids=sorted({int(x.strip()) for x in cq.stdout.splitlines() if x.strip().isdigit()}); cleanup_memory=int(mq.strip()); cleanup_checked=True\n"
                    "  cleaned=not group_alive and not cleanup_pids and cleanup_memory<=2500\n"
                    " except Exception:\n"
                    "  cleanup_error=True; cleaned=False\n"
                    " finally: cleanup_wall=time.perf_counter()-ct\n"
                    "post=met.with_suffix('.post_close.json'); chosen=post if post.exists() else met; "
                    "v={'blocked':blocked,'remote_busy':remote_busy,'spawned':p is not None,'binding_ok':source_ok and checkpoint_ok and jobs_ok,'timed_out':timed,'cleanup_confirmed':cleaned,'cleanup_checked':cleanup_checked,'cleanup_check_error':cleanup_error,'cleanup_group_alive':group_alive,'cleanup_compute_pids':cleanup_pids,'cleanup_memory_mib':cleanup_memory,'cleanup_wall_s':cleanup_wall,'exit_code':rc,'launcher_wall_s':launcher_wall,'report':out.exists(),'metrics':chosen.exists(),'metrics_path':str(chosen)}; "
                    "print('RL_TRIAGE_JSON:'+base64.urlsafe_b64encode(json.dumps(v,separators=(',',':')).encode()).decode().rstrip('='))"
                )
                launch_call = time.perf_counter()
                transport_deadline = time.monotonic() + float(watchdog_s) + TRANSPORT_TIMEOUT_S
                try:
                    outcome = self._exec(_remote_python(run_code, encoded_argv, remote_log, remote_report,
                                                        remote_metrics, f"{workspace}/IsaacLab", encoded_manifest,
                                                        str(watchdog_s)), "probe execution", transport_deadline)
                except BrevBackendError as error:
                    raise BrevCleanupUnverified("probe execution", error.error_type) from error
                launch_transport_wall = time.perf_counter() - launch_call
                execution_state = {"launched": outcome.get("spawned") is True,
                                   "launcher_wall_s": outcome.get("launcher_wall_s")}
                if outcome.get("blocked"):
                    raise BrevBackendError("probe execution", "NoOverwriteError")
                if outcome.get("remote_busy"):
                    raise BrevBackendError("probe execution", "RemoteGPUNotReady")
                if outcome.get("binding_ok") is not True:
                    raise BrevBackendError("probe execution", "RemoteBindingMismatch")
                cleanup = validate_cleanup_evidence(outcome)
                retrieval_start = time.perf_counter()
                local_report, local_metrics = temp / "report.json", temp / "metrics.json"
                if outcome.get("report"):
                    self._copy(f"{self.instance}:{remote_report}", windows_to_wsl(local_report),
                               "report copy", deadline)
                if outcome.get("metrics") and isinstance(outcome.get("metrics_path"), str):
                    self._copy(f"{self.instance}:{outcome['metrics_path']}", windows_to_wsl(local_metrics),
                               "metrics copy", deadline)
                if not local_report.exists() or not local_metrics.exists():
                    raise BrevBackendError("artifact retrieval", "ArtifactMissing")
                report, meter = json.loads(local_report.read_text(encoding="utf-8")), json.loads(local_metrics.read_text(encoding="utf-8"))
                raw_report = private_root / f"raw_report_{tag}_{case}_{probe}_{uuid.uuid4().hex}.json"
                with raw_report.open("xb") as handle:
                    handle.write(local_report.read_bytes())
                if contains_private_text(report) or contains_private_text(meter):
                    raise BrevBackendError("artifact privacy", "PrivateTextDetected")
                report_sha, meter_sha = sha(local_report), sha(local_metrics)
                report, meter = validate_probe_artifacts(
                    report, meter, case=case, probe=probe, checkpoint_sha256=checkpoint_sha256,
                    probes_sha256_lf=self.protocol["code_sha256_lf"]["evals/probes.py"],
                    jobs_sha256=hashlib.sha256(jobs_path.read_bytes()).hexdigest())
                rows = meter["measurements"]
                destination = self.root / "evals/results" / tag
                report_path = destination / "probes" / f"{case}__{probe}.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                with report_path.open("xb") as handle:
                    handle.write(local_report.read_bytes())
                meter_path = destination / "meters" / report_path.name
                meter_path.parent.mkdir(parents=True, exist_ok=True)
                with meter_path.open("xb") as handle:
                    handle.write(local_metrics.read_bytes())
                parent_cpu = time.process_time() - cpu_start
                resource = {"schema": "probe_resources_v1", "case": case, "probe": probe,
                            "checkpoint_sha256": checkpoint_sha256, "report_sha256_lf": report_sha,
                            "cpu_s": meter.get("cpu_s"), "launcher_exit_code": outcome.get("exit_code"),
                            "cpu_scope": "remote_isaac_python_wrapper_plus_windows_brev_parent_reported_separately",
                            "isaac_cpu_scope": meter.get("cpu_scope"),
                            "app_close_cpu_included": meter.get("phase") == "after_main",
                            "simulator_steps_scope": meter.get("simulator_steps_scope"),
                            "meter_sha256_lf": meter_sha,
                            "probes_sha256_lf": meter.get("probes_sha256_lf"),
                            "backend_binding_sha256": digest(public),
                            "jobs_sha256": public["jobs_sha256"], **rows[0]}
                resource_path = destination / "resources" / report_path.name
                resource_path.parent.mkdir(parents=True, exist_ok=True)
                with resource_path.open("x", encoding="utf-8", newline="\n") as handle:
                    json.dump(resource, handle, indent=1, allow_nan=False); handle.write("\n")
            retrieval_wall = time.perf_counter() - retrieval_start
            launcher_wall = float(outcome["launcher_wall_s"])
            end_to_end_wall = time.perf_counter() - float(plan["started_perf"])
            cleanup_wall = float(cleanup["cleanup_wall_s"])
            transport_wall = (float(public["preapproval_transport_wall_s"])
                              + max(0.0, launch_transport_wall - launcher_wall - cleanup_wall))
            execution = {"schema": "probe_execution_v1", "case": case, "probe": probe,
                         "checkpoint_sha256": checkpoint_sha256, "report_sha256_lf": report_sha,
                         "started_at": started_at, "finished_at": now(), "gpu_wall_s": round(launcher_wall, 1),
                         "execution_wall_s": round(launcher_wall, 1), "remote_launcher_wall_s": launcher_wall,
                         "transport_wall_s": transport_wall, "retrieval_wall_s": retrieval_wall,
                         "end_to_end_wall_s": end_to_end_wall, "timed_out": outcome.get("timed_out") is True,
                         "cleanup_confirmed": True, "cleanup_scope": "owned_process_group_and_post_gpu_snapshot",
                         "cleanup_wall_s": cleanup_wall, "cleanup_group_alive": cleanup["cleanup_group_alive"],
                         "cleanup_compute_pids": cleanup["cleanup_compute_pids"],
                         "cleanup_memory_mib": cleanup["cleanup_memory_mib"],
                         "cleanup_check_error": cleanup["cleanup_check_error"],
                         "launcher_watchdog_s": watchdog_s,
                         "backend_binding_sha256": digest(public), "jobs_sha256": public["jobs_sha256"],
                         "exit_code": outcome.get("exit_code"),
                         "parent_cpu_s": parent_cpu, "cpu_s": (None if resource["cpu_s"] is None else parent_cpu + resource["cpu_s"]),
                         "simulator_steps": resource.get("simulator_steps"), "metered": True}
            execution_path = self.root / "evals/results" / tag / "execution" / f"{case}__{probe}.json"
            execution_path.parent.mkdir(parents=True, exist_ok=True)
            with execution_path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(execution, handle, indent=1, allow_nan=False); handle.write("\n")
            rec = report.get("probes", {}).get(probe, {})
            code = outcome.get("exit_code") if outcome.get("exit_code") != 0 else rec.get("exit")
            return rec.get("measurement") if code == 0 else None, launcher_wall, code
        except BrevBackendError as error:
            if execution_state["launched"]:
                error.launched = True
                value = execution_state["launcher_wall_s"]
                error.launcher_wall_s = float(value) if isinstance(value, (int, float)) else None
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise BrevBackendError("probe backend", type(error).__name__) from error
