from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import vision_annotator_pull_checkpoint_v1 as D  # pyright: ignore[reportMissingImports]
import vision_annotator_pull_worker_v1 as WORKER  # pyright: ignore[reportMissingImports]


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = tmp_path / "job.json"
    job.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    return {
        "job": {
            "mode": D.MODE,
            "callback_variant": D.VARIANT,
            "profile": RECORD.fixed_profile(),
        },
        "job_path": job,
        "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
        "output_directory": output,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def _capture(ordinal: int) -> dict[str, Any]:
    return {
        "schema": "vision_annotator_pull_capture_v1",
        "ordinal": ordinal,
        "arm": {"ordinal": ordinal, "marker_code": ordinal},
        "physics_before": {"simulation_time_s": 0.01, "sha256": "a" * 64},
        "physics_after": {"simulation_time_s": 0.01, "sha256": "a" * 64},
        "raw": {
            "rgb": np.full((2, 2, 4), ordinal, dtype=np.uint8),
            "semantic_segmentation": {
                "data": np.full((2, 2), ordinal, dtype=np.uint32),
                "info": {"idToLabels": {str(ordinal): {"class": f"marker-{ordinal}"}}},
            },
            "CameraParams": {
                "cameraViewTransform": np.eye(4),
                "cameraProjection": np.eye(4),
                "renderProductResolution": [2, 2],
            },
            "IsaacReadTimes": {"frameNumber": ordinal + 1, "simulationTime": 0.01},
        },
    }


def _worker() -> dict[str, Any]:
    return {
        "schema": WORKER.SCHEMA,
        "collection": {
            "schema": "vision_annotator_pull_collection_report_v1",
            "status": "collected_unvalidated",
            "capture_count": 4,
            "captures": [_capture(index) for index in range(4)],
            "partial_capture": None,
            "atomicity_claimed": False,
        },
        "validation": {
            "approved": True,
            "camera_sequence": ["A", "A", "B", "A"],
            "marker_codes": [0, 1, 2, 3],
            "atomicity_claimed": False,
        },
        "policy_probe": {
            "environment_steps": 1,
            "annotators_detached_before_policy_step": True,
        },
        "lifecycle": {"annotators_attached": True, "annotators_detached": True},
        "writer_calls": {"create": 0, "attach": 0, "callback": 0, "detach": 0},
        "diagnostic_complete": True,
    }


def test_nested_fixed_profile_and_raw_npz_receipts_are_preserved(tmp_path: Path) -> None:
    validated = _validated(tmp_path)
    sealed: list[dict[str, Any]] = []

    result = D.run_annotator_pull_diagnostic(
        validated,
        object(),
        seal_report=lambda value: sealed.append(dict(value)),
        run_worker=lambda *_args, **_kwargs: _worker(),
    )

    assert D._camera_profile(RECORD.fixed_profile()) == RECORD.fixed_profile()["camera"]
    assert result["diagnostic_complete"] is True
    assert result["approved"] is False
    assert result["capture_approved"] is False
    assert result["atomicity_claimed"] is False
    assert len(result["raw_artifacts"]) == 4
    assert all(row["npz"]["bytes"] > 0 for row in result["raw_artifacts"])
    assert result["worker"]["writer_calls"] == {
        "create": 0,
        "attach": 0,
        "callback": 0,
        "detach": 0,
    }
    assert sealed == [result]


def test_failure_preserves_complete_and_partial_raw_payloads(tmp_path: Path) -> None:
    validated = _validated(tmp_path)
    failed = _worker()
    failed["diagnostic_complete"] = False
    failed["collection"]["captures"] = [_capture(0)]
    failed["collection"]["capture_count"] = 1
    partial = _capture(1)
    partial.pop("schema")
    failed["collection"]["partial_capture"] = partial
    sealed: list[dict[str, Any]] = []

    def fail(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        error = ValueError("pull failed")
        raise WORKER.AnnotatorPullWorkerFailure(failed, error) from error

    with pytest.raises(D.AnnotatorPullDiagnosticFailure) as caught:
        D.run_annotator_pull_diagnostic(
            validated,
            object(),
            seal_report=lambda value: sealed.append(dict(value)),
            run_worker=fail,
        )

    assert isinstance(caught.value.primary, ValueError)
    assert len(caught.value.report["raw_artifacts"]) == 2
    assert [row["complete_capture"] for row in caught.value.report["raw_artifacts"]] == [
        True,
        False,
    ]
    assert sealed == [caught.value.report]


def test_nested_profile_rejects_full_profile_at_render_helper_boundary() -> None:
    profile = RECORD.fixed_profile()
    assert D._camera_profile(profile) == profile["camera"]
    with pytest.raises(TypeError, match="profile.camera"):
        D._camera_profile({**profile, "camera": None})


def test_public_worker_uses_real_nested_profile_setup_without_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validated = _validated(tmp_path)
    validated["job"]["mode"] = WORKER.MODE
    profile = RECORD.fixed_profile()
    order: list[str] = []

    class Command:
        def __setitem__(self, _key: Any, _value: Any) -> None:
            order.append("command")

        def new_tensor(self, value: Any) -> Any:
            return value

    class Sim:
        current_time = 0.01

        def set_camera_view(self, **_kwargs: Any) -> None:
            order.append("camera_view")

    base = type(
        "Base",
        (),
        {
            "sim": Sim(),
            "command_manager": type(
                "Commands", (), {"get_command": lambda _self, _name: Command()}
            )(),
            "reset": lambda _self, seed: order.append(f"reset:{seed}"),
        },
    )()

    class Env:
        unwrapped = base

        def close(self) -> None:
            order.append("env_close")

    class Context:
        def get_stage(self) -> Any:
            return type("Stage", (), {"GetPrimAtPath": lambda _self, path: path})()

        def reset_renderer_accumulation(self) -> None:
            order.append("renderer_reset")

    class Attr:
        def Set(self, _value: Any) -> None:
            pass

    class Camera:
        def GetProjectionAttr(self) -> Attr:
            return Attr()

        def GetHorizontalApertureAttr(self) -> Attr:
            return Attr()

        def GetFocalLengthAttr(self) -> Attr:
            return Attr()

        def GetClippingRangeAttr(self) -> Attr:
            return Attr()

    class RenderProduct:
        path = "/Render/Products/RP"

        def destroy(self) -> None:
            order.append("render_product_destroy")

    class Marker:
        def bind_and_validate(self, _stage: Any) -> None:
            order.append("marker_bind")

        def set_code(self, _stage: Any, code: int) -> None:
            order.append(f"marker:{code}")

    rep = type(
        "Rep",
        (),
        {
            "create": type(
                "Create", (), {"render_product": lambda _self, *_args: RenderProduct()}
            )(),
            "orchestrator": object(),
        },
    )()
    writer_calls: list[str] = []
    deps = {name: object() for name in WORKER.PREPARED._DEPENDENCY_KEYS} | {
        "profile": lambda: profile,
        "settings": lambda: object(),
        "enable_extension": lambda _name: True,
        "startup": lambda _rep, _settings: {"before": {}, "after": {}},
        "registration_probe": lambda: {
            name: True for name in WORKER.SDK.ANNOTATORS
        },
        "usd_context": lambda: (Context(), object()),
        "make_env": lambda _task, _cfg: Env(),
        "rep": rep,
        "camera": lambda _prim: Camera(),
        "root_position": lambda _base: [0.0, 0.0, 0.0],
        "marker": lambda _rep, _root: Marker(),
        "fxaa_required": lambda _job: False,
        "renderer_readback": lambda *_args: {"renderer": "RayTracedLighting"},
        "readiness": lambda *_args: None,
        "physics_snapshot": lambda _base: {"root_position_world_m": [0.0, 0.0, 0.0]},
        "writer": lambda *_args: writer_calls.append("writer"),
    }
    monkeypatch.setattr(WORKER.SDK, "_configure_env", lambda _profile, _deps: object())

    def collect(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["camera_profile"] is profile["camera"]
        for ordinal, (kind, code) in enumerate((("A", 0), ("A", 1), ("B", 2), ("A", 3))):
            kwargs["prepare_arm"](ordinal, kind, code)
        return {
            "operation_trace": [
                *({"operation": "attach"} for _ in range(4)),
                *({"operation": "detach"} for _ in range(4)),
            ],
            "cleanup_errors": [],
        }

    monkeypatch.setattr(WORKER.COLLECTOR, "collect_aaba", collect)
    monkeypatch.setattr(WORKER.VALIDATOR, "validate", lambda _value: {"approved": True})

    def policy_factory(
        _validated: Any, make_env: Any, state: dict[str, bool], probe: dict[str, Any]
    ) -> Any:
        def create(task: str, cfg: Any) -> Any:
            env = make_env(task, cfg)
            original_close = env.close

            def close() -> None:
                assert state == {"capture_completed": True, "annotators_detached": True}
                probe.update(
                    {
                        "environment_steps": 1,
                        "annotators_detached_before_policy_step": True,
                    }
                )
                original_close()

            env.close = close
            return env

        return create

    result = WORKER.run_worker(
        validated,
        type("App", (), {"update": lambda _self: order.append("app_update")})(),
        dependencies=deps,
        policy_env_factory=policy_factory,
    )

    assert result["diagnostic_complete"] is True
    assert writer_calls == []
    assert [row for row in order if row.startswith("marker:")] == [
        "marker:0",
        "marker:1",
        "marker:2",
        "marker:3",
    ]
    assert order[-2:] == ["render_product_destroy", "env_close"]
