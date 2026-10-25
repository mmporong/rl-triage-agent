from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import run_vision_brev_annotated_v1 as C  # pyright: ignore[reportMissingImports]
import vision_brev_render_backend_v3 as B  # pyright: ignore[reportMissingImports]


def test_pinned_popen_translates_only_exact_owned_gate_and_preserves_training() -> None:
    calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def delegate(args: Any, *positional: Any, **kwargs: Any) -> str:
        calls.append((args, positional, kwargs))
        return "process"

    python = Path("/opt/isaac/python")
    worker_v1 = Path("/repo/bench/vision_play_worker_v1.py")
    worker_v2 = Path("/repo/bench/vision_play_worker_v2.py")
    popen = B.PinnedVisionWorkerPopen(delegate, python, worker_v1, worker_v2)
    owned = [
        popen.python,
        "-I",
        "-S",
        "-c",
        B.V2.OWNED_WORKER_GATE,
        popen.python,
        "-B",
        popen.worker_v1,
        "--job",
        "/runtime/job.json",
    ]
    assert popen(owned, cwd="/lab") == "process"
    assert calls[-1] == (
        [*owned[:7], popen.worker_v2, *owned[8:]],
        (),
        {"cwd": "/lab"},
    )

    training = [
        str(python),
        "-u",
        "/repo/bench/brev_recovery_worker_v2.py",
        "--request",
        "r.json",
    ]
    assert popen(training, stdout=subprocess.PIPE) == "process"
    assert calls[-1] == (training, (), {"stdout": subprocess.PIPE})


@pytest.mark.parametrize("case", ["v1", "v2", "gate"])
def test_pinned_popen_rejects_direct_or_malformed_worker_argv(case: str) -> None:
    popen = B.PinnedVisionWorkerPopen(
        lambda *_args, **_kwargs: pytest.fail("거부할 argv가 delegate에 전달됐다"),
        Path("/opt/isaac/python"),
        Path("/repo/bench/vision_play_worker_v1.py"),
        Path("/repo/bench/vision_play_worker_v2.py"),
    )
    argv = {
        "v1": [popen.python, "-B", popen.worker_v1, "--job", "job.json"],
        "v2": [popen.python, "-B", popen.worker_v2, "--job", "job.json"],
        "gate": [popen.python, "-I", "-S", "-c", B.V2.OWNED_WORKER_GATE, "wrong"],
    }[case]
    with pytest.raises(ValueError, match="owned-gate"):
        popen(argv)


def _source_protocol() -> dict[str, Any]:
    return {
        "source_sha256_lf": {
            relative: B._sha256_lf(ROOT / relative) for relative in B.SOURCE_FILES
        }
    }


def test_backend_factory_requires_new_source_closure_and_wraps_delegate(
    monkeypatch,
) -> None:
    protocol = _source_protocol()
    captured: dict[str, Any] = {}
    sentinel = object()

    def parent(
        cls: type[Any],
        lab: Path,
        python: Path,
        source: dict[str, Any],
        anchor: str,
        runtime: Path,
        plan: list[dict[str, Any]],
        **kwargs: Any,
    ) -> object:
        captured.update({"cls": cls, "source": source, "popen": kwargs["popen"]})
        return sentinel

    monkeypatch.setattr(
        B.V2.AttestedVisionBrevBackend, "from_runtime", classmethod(parent)
    )
    delegate = object()
    result = B.AnnotatedVisionBrevBackend.from_runtime(
        Path("/lab"),
        Path("/opt/isaac/python"),
        protocol,
        "a" * 40,
        Path("/repo/.triage-runtime/vision-brev/test"),
        [],
        popen=delegate,
    )
    assert result is sentinel
    assert captured["cls"] is B.AnnotatedVisionBrevBackend
    assert captured["source"] is protocol
    assert isinstance(captured["popen"], B.PinnedVisionWorkerPopen)
    assert captured["popen"].delegate is delegate

    missing = _source_protocol()
    del missing["source_sha256_lf"]["bench/vision_play_worker_v2.py"]
    with pytest.raises(ValueError, match="source 결합"):
        B.AnnotatedVisionBrevBackend.from_runtime(
            Path("/lab"),
            Path("/opt/isaac/python"),
            missing,
            "a" * 40,
            Path("/repo/.triage-runtime/vision-brev/test"),
            [],
            popen=delegate,
        )
    changed = _source_protocol()
    changed["source_sha256_lf"]["bench/vision_play_worker_v2.py"] = "0" * 64
    with pytest.raises(ValueError, match="LF SHA"):
        B._verify_annotated_source_closure(changed)


def _base_args(tmp_path: Path, phase: str, command: str) -> list[str]:
    return [
        "--phase",
        phase,
        "--command",
        command,
        "--repo",
        str(tmp_path),
        "--protocol",
        "source.json",
        "--protocol-sha256-lf",
        "1" * 64,
        "--render-protocol",
        "render.json",
        "--render-protocol-sha256-lf",
        "2" * 64,
        "--preflight-protocol",
        "preflight.json",
        "--preflight-protocol-sha256-lf",
        "3" * 64,
        "--source-anchor",
        "a" * 40,
        "--resource-proof",
        "resource.json",
        "--resource-sha256",
        "4" * 64,
    ]


@pytest.mark.parametrize("phase", ["preflight", "render"])
def test_active_controllers_share_annotated_backend_factory(
    tmp_path: Path,
    monkeypatch,
    capsys,
    phase: str,
) -> None:
    captured: dict[str, Any] = {}

    class Batch:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured["factory"] = kwargs["backend_factory"]

        def verify(self) -> dict[str, Any]:
            return {"status": "verified"}

    monkeypatch.setattr(C.PREFLIGHT, "PreflightBatch", Batch)
    monkeypatch.setattr(C.RENDER, "RenderBatch", Batch)
    args = [
        *_base_args(tmp_path, phase, "verify"),
        "--isaaclab",
        "lab",
        "--python",
        "python",
    ]
    assert C.main(args) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "verified"}
    assert (
        captured["factory"].__func__
        is B.AnnotatedVisionBrevBackend.from_runtime.__func__
    )


def test_preflight_offline_settle_returns_without_backend_creation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    expected = {"status": "HOLD", "reason": "preflight_stage_failed"}
    monkeypatch.setattr(C.PREFLIGHT, "settle_preflight", lambda *_args: expected)
    monkeypatch.setattr(
        C.PREFLIGHT,
        "PreflightBatch",
        lambda *_args, **_kwargs: pytest.fail(
            "offline settle 뒤 backend batch를 만들었다"
        ),
    )
    assert C.main(_base_args(tmp_path, "preflight", "advance")) == 2
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize("command", ["export", "finalize"])
def test_render_offline_commands_reuse_existing_functions_without_backend(
    tmp_path: Path,
    monkeypatch,
    capsys,
    command: str,
) -> None:
    expected = {"status": "offline_complete", "command": command}
    monkeypatch.setattr(
        C.RENDER,
        "RenderBatch",
        lambda *_args, **_kwargs: pytest.fail("backend batch 생성"),
    )
    monkeypatch.setattr(
        C.RENDER, "export_materials", lambda *_args, **_kwargs: expected
    )
    monkeypatch.setattr(
        C.RENDER, "finalize_materials", lambda *_args, **_kwargs: expected
    )
    args = _base_args(tmp_path, "render", command)
    if command == "finalize":
        args += ["--materials", "materials.json"]
    assert C.main(args) == 0
    assert json.loads(capsys.readouterr().out) == expected
