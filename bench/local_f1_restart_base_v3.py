"""v3 source closure와 Isaac runtime/package pin을 제공한다."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / ".git").exists() and (
            parent / "bench/local_f1_restart_base_v2.py"
        ).is_file():
            return parent
    raise FileNotFoundError("rl-triage 저장소 root를 찾지 못했다")


_DISCOVERED_ROOT = _repo_root(Path(__file__))
sys.path.insert(0, str(_DISCOVERED_ROOT / "bench"))

import local_f1_restart_base_v2 as V2

ROOT = V2.ROOT
read, sha, write_new, binding, judge = (
    V2.read,
    V2.sha,
    V2.write_new,
    V2.binding,
    V2.judge,
)

SOURCE_FILES = (
    "bench/local_f1_restart_base_v2.py",
    "bench/local_f1_restart_prepare_v2.py",
    "bench/local_f1_restart_terminal_v2.py",
    "bench/local_f1_restart_child_v2.py",
    "bench/local_f1_restart_worker_v2.py",
    "bench/local_f1_restart_base_v3.py",
    "bench/local_f1_restart_worker_v3.py",
    "bench/finalize_local_f1_restart_v3.py",
    "bench/protocols/local_f1_restart_s3221_v3.json",
    "bench/brev_recovery_worker_v2.py",
    "bench/brev_recovery_worker_v3.py",
    "bench/residual_training_v1.py",
    "bench/hidden_faults.py",
    "bench/optimizer_faults.py",
    "bench/r1_repair_training.py",
    "bench/residual_faults_v1.py",
    "bench/residual_probe_v1.py",
    "bench/extract_tb.py",
    "evals/fixed_eval.py",
    "evals/optimizer_observability.py",
    "bench/protocols/fixed_eval_v1.json",
    "src/rl_triage/correction_ledger.py",
    "src/rl_triage/correction_registry.py",
    "src/rl_triage/grounded_recovery.py",
    "src/rl_triage/probe_loop.py",
    "src/rl_triage/recovery.py",
)


def source_map(root: Path = ROOT) -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source closure 누락: {missing}")
    return {name: sha(root / name, lf=True) for name in SOURCE_FILES}


def git_blob_map(root: Path, anchor: str, names: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for name in names:
        raw = subprocess.run(
            ["git", "-C", str(root), "show", f"{anchor}:{name}"],
            capture_output=True,
            check=True,
        ).stdout
        result[name] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    return result


def runtime_snapshot(
    lab: Path, expected_head: str, paths: dict[str, str]
) -> dict[str, str]:
    head = subprocess.run(
        ["git", "-C", str(lab), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=True,
    ).stdout.strip()
    if head != expected_head:
        raise ValueError("Isaac Lab HEAD 불일치")
    out = {}
    for relative, expected in paths.items():
        if subprocess.run(
            ["git", "-C", str(lab), "diff", "--quiet", "HEAD", "--", relative],
            check=False,
        ).returncode:
            raise ValueError(f"Isaac Lab runtime source dirty: {relative}")
        target = lab / relative
        if not target.is_file() or sha(target, lf=True) != expected:
            raise ValueError(f"Isaac Lab runtime source SHA 불일치: {relative}")
        out[relative] = expected
    return out


def package_snapshot(
    isaac_python: Path,
    expected_distribution: str,
    expected_version: str,
    paths: dict[str, str],
) -> dict[str, str]:
    script = (
        "import importlib.metadata as m,json,pathlib,rsl_rl;"
        "print(json.dumps({'root':str(pathlib.Path(rsl_rl.__file__).resolve().parent),"
        f"'version':m.version({expected_distribution!r})}}))"
    )
    process = subprocess.run(
        [str(isaac_python), "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=60,
        check=True,
    )
    metadata = json.loads(process.stdout.strip().splitlines()[-1])
    if metadata.get("version") != expected_version:
        raise ValueError("rsl-rl-lib version 불일치")
    root = Path(metadata["root"])
    out = {}
    for relative, expected in paths.items():
        target = root / relative
        if not target.is_file() or sha(target, lf=True) != expected:
            raise ValueError(f"rsl_rl package source SHA 불일치: {relative}")
        out[relative] = expected
    return out
