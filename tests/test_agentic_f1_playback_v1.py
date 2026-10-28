"""F1 재생 등록: NONE은 v1 그대로, F1은 새 schema·재생 주입 없음, F2는 거부. callback은 원본 함수 관측을 receipt에 남긴다."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_f1_playback_v1 as PB  # pyright: ignore[reportMissingImports]
import agentic_fault_playback_v1 as V1  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]

SHA = "1d73d523c97c5c89c988febfbd3d5e533e90681cd0e6132d6ffe13888fa95731"
F1_ID = "source-s2221-f1"


def test_none_is_v1_object_and_f1_is_new_schema_with_real_hidden_faults() -> None:
    assert PB.registration("source-s2221-none", "b" * 64, "NONE") == V1.registration("source-s2221-none", "b" * 64, "NONE")
    value = PB.registration(F1_ID, SHA, "F1")
    base = V1.registration(F1_ID, SHA, "NONE")
    assert {key: value[key] for key in base if key not in ("schema", "fault_id")} == {
        key: base[key] for key in base if key not in ("schema", "fault_id")}
    assert value["schema"] == PB.SCHEMA and value["fault_id"] == "F1"
    assert value["playback_injection"] == "none" and value["training_fault_category"] == "reward"
    assert value["hidden_faults_sha256_lf"] == R.sha256_file(ROOT / "bench/hidden_faults.py", lf=True)


@pytest.mark.parametrize(("logical_id", "sha", "fault"), [
    (F1_ID, SHA, "F2"), (F1_ID, SHA, "F3"), (F1_ID, "abc", "F1"), ("Bad ID", SHA, "F1")])
def test_registration_rejects_f2_unknown_and_bad_format(logical_id: str, sha: str, fault: str) -> None:
    with pytest.raises(ValueError):
        PB.registration(logical_id, sha, fault)


def test_f1_category_must_be_reward(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PB, "_hidden_category", lambda: "actuator")
    with pytest.raises(ValueError, match="보상 결함"):
        PB.registration(F1_ID, SHA, "F1")


def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    for name in PB.SOURCE_FILES:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(f"# fixture {name}\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(V1, "ROOT", repo.resolve())
    monkeypatch.setattr(PB, "_hidden_category", lambda: "reward")
    return repo, {name: R.sha256_file(repo / name, lf=True) for name in PB.SOURCE_FILES}


def _validated(tmp_path: Path, repo: Path, sources: dict[str, str], registration: dict[str, Any]) -> dict[str, Any]:
    stage = tmp_path / "stage"
    stage.mkdir()
    return {"job": {"checkpoint": {"logical_id": F1_ID, "sha256": SHA}, "profile": R.fixed_profile(),
                    "source": {"files_lf": sources}},
            "protocol": {"playback_registrations": {F1_ID: registration}, "source_sha256_lf": sources},
            "checkpoint_sha256": SHA, "repo": str(repo), "job_path": str(stage / "job.json"), "job_sha256": "0" * 64}


def test_f1_callback_runs_without_injection_and_writes_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, sources = _repo(tmp_path, monkeypatch)
    registration = PB.registration(F1_ID, SHA, "F1")
    validated = _validated(tmp_path, repo, sources, registration)
    seen: list[Any] = []
    run_real = PB.make_run_real(registration, runner=lambda v, app, **kw: seen.append(kw) or "played",
                                installer=lambda checked: {"fault_id": checked["fault_id"], "class_method_changed": False})
    assert run_real(validated, object(), camera_capture_factory="factory") == "played"
    assert seen == [{"camera_capture_factory": "factory"}]
    stage = Path(validated["job_path"]).parent
    intent = json.loads((stage / "playback-intent.json").read_text(encoding="utf-8"))
    receipt = json.loads((stage / "playback-receipt.json").read_text(encoding="utf-8"))
    assert intent["registration"] == registration and receipt["status"] == "completed"
    assert receipt["installation"] == {"fault_id": "F1", "class_method_changed": False}


@pytest.mark.parametrize("case", ["protocol", "checkpoint", "source", "no-app"])
def test_f1_callback_rejects_mismatch_before_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    repo, sources = _repo(tmp_path, monkeypatch)
    registration = PB.registration(F1_ID, SHA, "F1")
    validated = _validated(tmp_path, repo, sources, registration)
    app: Any = object()
    if case == "protocol":
        validated["protocol"]["playback_registrations"][F1_ID] = dict(registration, playback_injection="motor")
    elif case == "checkpoint":
        validated["checkpoint_sha256"] = "c" * 64
    elif case == "source":
        (repo / "bench/agentic_f1_playback_v1.py").write_text("# changed\n", encoding="utf-8")
    else:
        app = None
    run_real = PB.make_run_real(registration, runner=lambda *a, **k: pytest.fail("재생되면 안 된다"),
                                installer=lambda checked: {})
    with pytest.raises(ValueError):
        run_real(validated, app)


def _fake_isaac(monkeypatch: pytest.MonkeyPatch, *, injected: bool) -> None:
    def track_lin_vel_xy_exp(env: Any) -> Any:
        return env

    def _track_lin_vel_world(env: Any) -> Any:
        return env

    rewards = ModuleType("isaaclab.envs.mdp.rewards")
    track_lin_vel_xy_exp.__module__ = rewards.__name__
    if injected:  # hidden_faults._swap_code와 같은 방식: 함수 객체는 두고 본문만 바꾼다
        track_lin_vel_xy_exp.__code__ = _track_lin_vel_world.__code__
    rewards.track_lin_vel_xy_exp = track_lin_vel_xy_exp  # type: ignore[attr-defined]
    actuator = ModuleType("isaaclab.actuators.actuator_pd")

    def _clip_effort(self: Any, effort: Any) -> Any:
        return effort

    _clip_effort.__module__, _clip_effort.__qualname__ = actuator.__name__, "DCMotor._clip_effort"
    actuator.DCMotor = type("DCMotor", (), {"_clip_effort": _clip_effort})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, rewards.__name__, rewards)
    monkeypatch.setitem(sys.modules, actuator.__name__, actuator)


@pytest.mark.parametrize("injected", [False, True])
def test_default_installer_observes_original_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, injected: bool
) -> None:
    repo, sources = _repo(tmp_path, monkeypatch)
    registration = PB.registration(F1_ID, SHA, "F1")
    validated = _validated(tmp_path, repo, sources, registration)
    _fake_isaac(monkeypatch, injected=injected)
    played: list[bool] = []
    run_real = PB.make_run_real(registration, runner=lambda *a, **k: played.append(True))
    receipt_path = Path(validated["job_path"]).parent / "playback-receipt.json"
    if injected:
        with pytest.raises(ValueError, match="이미 바뀌었다"):
            run_real(validated, object())
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert played == [] and receipt["status"] == "failed" and receipt["installation"] == {}
        return
    run_real(validated, object())
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert played == [True] and receipt["installation"]["playback_injection"] == "none"
    assert receipt["installation"]["reward_function"]["code_name"] == "track_lin_vel_xy_exp"
    assert receipt["installation"]["clip_function"]["qualname"] == "DCMotor._clip_effort"


def test_none_delegates_to_v1_and_foreign_objects_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(PB, "_V1_MAKE_RUN_REAL", lambda value, **kwargs: calls.append((value, kwargs)) or "v1")
    none = V1.registration("source-s2221-none", "b" * 64, "NONE")

    def runner(*_args: Any, **_kwargs: Any) -> None:
        return None

    def installer(checked: Any) -> Any:
        return checked

    assert PB.make_run_real(none, runner=runner, installer=installer) == "v1"
    assert calls == [(none, {"runner": runner, "installer": installer})]
    for value in (V1.registration("go2-s2033-f2", "b" * 64, "F2"), dict(none, schema=PB.SCHEMA),
                  dict(PB.registration(F1_ID, SHA, "F1"), schema=V1.SCHEMA)):
        with pytest.raises(ValueError, match="계약 밖"):
            PB.make_run_real(value)


def test_import_does_not_load_isaac_or_torch() -> None:
    code = ("import sys; sys.path[:0]=[sys.argv[1]]; import agentic_f1_playback_v1; "
            "print([m for m in sys.modules if m.split('.')[0] in ('torch','isaaclab','isaacsim','omni','carb')])")
    result = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "bench")], capture_output=True, text=True,
                            check=True)
    assert result.stdout.strip() == "[]"
