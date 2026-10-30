from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

CANDIDATE = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(CANDIDATE))

import local_f1_restart_base_v3 as BASE
import local_f1_restart_worker_v3 as WORKER

TEMPLATE = CANDIDATE / "protocols/local_f1_restart_s3221_v3.template.json"


def protocol() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def test_actual_runtime_and_package_snapshots() -> None:
    value = protocol()
    lab = Path.home() / "IsaacLab"
    isaac_python = lab / "_isaac_sim/python.bat"
    if not (lab / ".git").exists() or not isaac_python.is_file():
        pytest.skip("이 호스트에 검증 대상 Isaac Lab 설치가 없다")
    assert (
        BASE.runtime_snapshot(
            lab, value["runtime"]["isaaclab_head"], value["runtime"]["source_sha256_lf"]
        )
        == value["runtime"]["source_sha256_lf"]
    )
    assert (
        BASE.package_snapshot(
            isaac_python,
            value["python_package"]["distribution"],
            value["python_package"]["version"],
            value["python_package"]["source_sha256_lf"],
        )
        == value["python_package"]["source_sha256_lf"]
    )


def test_runtime_snapshot_rejects_wrong_head() -> None:
    value = protocol()
    lab = Path.home() / "IsaacLab"
    if not (lab / ".git").exists():
        pytest.skip("이 호스트에 검증 대상 Isaac Lab checkout이 없다")
    with pytest.raises(ValueError, match="HEAD 불일치"):
        BASE.runtime_snapshot(lab, "0" * 40, value["runtime"]["source_sha256_lf"])


def test_build_jobs_separates_namespaces_and_f1_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    value = protocol()
    value["status"] = "registered"
    value["registration_anchor"] = subprocess.run(
        ["git", "-C", str(BASE.ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    auth = {
        "none_checkpoint": {
            "path": str(tmp_path / "none.pt"),
            "sha256": value["reference_checkpoint_sha256"],
        },
        "fixed_eval": {"path": str(BASE.ROOT / "bench/protocols/fixed_eval_v1.json")},
    }
    repo_sources = {name: "1" * 64 for name in WORKER.TRAINING.MINIMUM_BOUND_SOURCES}
    repo_sources["bench/local_f1_restart_worker_v3.py"] = "1" * 64
    captured: dict = {}

    monkeypatch.setattr(WORKER.V2, "authorization", lambda _path: auth)
    monkeypatch.setattr(WORKER, "read", lambda _path: copy.deepcopy(value))
    monkeypatch.setattr(WORKER, "source_map", lambda: repo_sources)
    monkeypatch.setattr(
        WORKER,
        "runtime_snapshot",
        lambda *_args: value["runtime"]["source_sha256_lf"],
    )
    monkeypatch.setattr(
        WORKER,
        "package_snapshot",
        lambda *_args: value["python_package"]["source_sha256_lf"],
    )
    monkeypatch.setattr(WORKER, "sha", lambda *_args, **_kwargs: "2" * 64)
    monkeypatch.setattr(
        WORKER.TRAINING,
        "verify_source_binding",
        lambda manifest, runtime_root=None: captured.update(
            {"manifest": manifest, "runtime_root": runtime_root}
        ),
    )

    train, evaluate = WORKER.build_jobs(
        tmp_path / "authorization.json",
        tmp_path / "stage",
        Path.home() / "IsaacLab",
        value["registration_anchor"],
        Path.home() / "IsaacLab/_isaac_sim/python.bat",
    )

    assert train["source_sha256_lf"] == repo_sources
    assert train["runtime_source_sha256_lf"] == value["runtime"]["source_sha256_lf"]
    assert (
        train["python_package_source_sha256_lf"]
        == value["python_package"]["source_sha256_lf"]
    )
    assert train["source_checkpoint_sha256"] == WORKER.F1_SHA256
    assert (
        evaluate["artifacts"]["NONE"]["checkpoint"]["sha256"]
        == value["reference_checkpoint_sha256"]
    )
    assert (
        captured["manifest"]["runtime_source_sha256_lf"]
        == value["runtime"]["source_sha256_lf"]
    )
    assert captured["runtime_root"] == Path.home() / "IsaacLab"
