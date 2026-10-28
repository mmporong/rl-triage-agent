"""상태 dry-run child v2의 종료 단계 증거와 fast shutdown 적용 조건을 SIDECAR1 실제 dry-run과 fake App·session으로 검증한다."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import record_agentic_state_sidecar_v1 as S1  # pyright: ignore[reportMissingImports]
import record_agentic_state_sidecar_v2 as S2  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import test_agentic_state_sidecar_v1 as T1  # pyright: ignore[reportMissingImports]

SOURCE_ID = "motor-s2033-none-state-smoke"
SCRIPT = ROOT / S2.SOURCE_PATH


class Settings:
    """carb.settings fake. 설정 변경 순서를 events에 남긴다."""

    def __init__(self, events: list[Any], *, startup: bool = False, readback: bool = True) -> None:
        self.values = {S2.SETTING: startup}
        self.events = events
        self.readback = readback

    def get_as_bool(self, key: str) -> bool:
        return self.values[key]

    def set_bool(self, key: str, value: bool) -> None:
        self.events.append(("set", key, value))
        self.values[key] = value if self.readback else not value


class App:
    def __init__(self, events: list[Any], out: Path, *, fail: bool = False) -> None:
        self.events, self.out, self.fail = events, out, fail

    def close(self, **kwargs: Any) -> None:
        folder = self.out / S2.SHUTDOWN_DIR
        self.events.append(("app.close", kwargs, sorted(path.name for path in folder.iterdir()) if folder.is_dir() else []))
        if self.fail:
            raise RuntimeError("app close failed")


class Session(T1.FakeSession):
    def __init__(self, events: list[Any], out: Path, *, hook: Callable[[Path], object] | None = None,
                 fail_close: bool = False, fail_advance: bool = False) -> None:
        super().__init__(T1._base(T1.FakeContact()), events, fail_advance=fail_advance)
        self._out = out
        self.hook, self.fail_close = hook, fail_close

    def close(self) -> None:
        super().close()
        if self.hook is not None:
            self.hook(self._out)  # 봉인 뒤·app close 전 상태를 바꾼다(변조·실패 흉내)
        if self.fail_close:
            raise RuntimeError("session close failed")


def _checkpoint(tmp: Path) -> tuple[Path, str]:
    path = tmp / "model.pt"
    path.write_bytes(b"policy")
    return path, R.sha256_file(path)


def run(tmp: Path, *, frames: int = 1, startup: bool = False, readback: bool = True, app_fail: bool = False,
        **session: Any) -> tuple[int, list[Any], Path, Settings]:
    events: list[Any] = []
    checkpoint, sha = _checkpoint(tmp)
    out = tmp / "state-smoke"
    settings = Settings(events, startup=startup, readback=readback)
    code = S2.run_state_dry_run(
        checkpoint_path=checkpoint, checkpoint_sha256=sha, source_id=SOURCE_ID, out_dir=out, frames=frames,
        launch_app=lambda: App(events, out, fail=app_fail), settings_factory=lambda: settings,
        session_factory=lambda path: Session(events, out, **session),
    )
    return code, events, out, settings


def verify(out: Path, tmp: Path, **changes: Any) -> dict[str, Any]:
    args = {"checkpoint_sha256": R.sha256_file(tmp / "model.pt"), "source_id": SOURCE_ID, "frames": 1, **changes}
    return S2.verify_shutdown_evidence(out, **args)


# --- 성공 순서 ----------------------------------------------------------------------------------


def test_success_order_is_seal_session_close_check_setting_then_app_close(tmp_path: Path) -> None:
    code, events, out, settings = run(tmp_path)
    assert code == 0
    session_close = next(item for item in events if item[0] == "session.close")
    assert session_close[1] == ["app-startup.json", "dry_run_result.json", "session-close-entered.json",
                                "sidecar_manifest.json"]
    assert [item[0] for item in events] == ["session.close", "set", "app.close"]
    assert events[2][1] == {"wait_for_replicator": False}
    assert events[2][2] == ["app-close-entered.json", "app-startup.json", "session-close-entered.json",
                            "session-close-returned.json", "shutdown-policy.json"]
    assert settings.values[S2.SETTING] is True
    result = verify(out, tmp_path)
    assert result["child_pid"] == os.getpid() and result["app_close_returned"] is True
    assert result["dry_run_result_sha256"] == R.sha256_file(out / "dry_run_result.json")
    assert S2.shutdown_markers(out) == list(S2.ORDER)
    policy = R.read_object(out / S2.SHUTDOWN_DIR / S2.POLICY)
    assert policy["before"] is False and policy["after"] is True and policy["wait_for_replicator"] is False
    with pytest.raises(FileExistsError):  # 같은 출력·영수증을 다시 쓰지 않는다
        run(tmp_path)


def test_startup_fast_shutdown_true_fails_before_session_and_closes_app(tmp_path: Path) -> None:
    code, events, out, _settings = run(tmp_path, startup=True)
    assert code == 1 and [item[0] for item in events] == ["app.close"]
    assert events[0][1] == {"wait_for_replicator": False}
    assert (out / "dry_run_failure.json").is_file() and not (out / S2.SHUTDOWN_DIR).exists()


# --- fast shutdown 거부 --------------------------------------------------------------------------


def _rewrite(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _other_cp(out: Path) -> None:  # manifest SHA까지 결과에 맞춘 타 CP
    manifest = out / "sidecar" / S1.MANIFEST_NAME
    _rewrite(manifest, lambda value: value["source"].update(checkpoint_sha256="0" * 64))
    _rewrite(out / "dry_run_result.json", lambda value: value.update(sidecar_manifest_sha256=R.sha256_file(manifest)))


TAMPER: dict[str, Callable[[Path], object]] = {
    "frames": lambda out: _rewrite(out / "dry_run_result.json", lambda value: value.update(frames=2)),
    "other-cp": _other_cp,
    "manifest": lambda out: _rewrite(out / "sidecar" / S1.MANIFEST_NAME, lambda value: value.update(extra=1)),
    "failure": lambda out: (out / "dry_run_failure.json").write_text("{}", encoding="utf-8"),
    "missing": lambda out: (out / "dry_run_result.json").unlink(),
}


@pytest.mark.parametrize("case", sorted(TAMPER))
def test_tampered_or_missing_seal_refuses_fast_shutdown(tmp_path: Path, case: str) -> None:
    code, events, out, settings = run(tmp_path, hook=TAMPER[case])
    assert code == 1 and settings.values[S2.SETTING] is False and "set" not in [item[0] for item in events]
    assert events[-1][1] == {"wait_for_replicator": False}  # SDK close는 이어서 시도한다
    assert (out / S2.SHUTDOWN_DIR / S2.REFUSAL).is_file() and not (out / S2.SHUTDOWN_DIR / S2.POLICY).exists()
    entered = R.read_object(out / S2.SHUTDOWN_DIR / S2.APP_ENTERED)
    assert entered["fast_shutdown_applied"] is False and entered["policy_sha256"] is None
    with pytest.raises(S2.ShutdownEvidenceError):
        verify(out, tmp_path)


def test_session_close_error_keeps_first_error_and_refuses(tmp_path: Path) -> None:
    code, events, out, settings = run(tmp_path, fail_close=True)
    assert code == 1 and settings.values[S2.SETTING] is False and events[-1][0] == "app.close"
    error = R.read_object(out / S2.SHUTDOWN_DIR / S2.SESSION_ERROR)
    assert error["error_message"] == "session close failed"
    assert "session close" in R.read_object(out / S2.SHUTDOWN_DIR / S2.REFUSAL)["error_message"]
    with pytest.raises(S2.ShutdownEvidenceError, match="집합"):
        verify(out, tmp_path)


@pytest.mark.parametrize("case", ["dry-run-failure", "readback", "app-close-error"])
def test_failure_paths_never_end_as_fast_success(tmp_path: Path, case: str) -> None:
    if case == "dry-run-failure":
        code, _events, out, settings = run(tmp_path, frames=2, fail_advance=True)
        assert settings.values[S2.SETTING] is False
    elif case == "readback":
        code, _events, out, _settings = run(tmp_path, readback=False)
    else:
        code, _events, out, _settings = run(tmp_path, app_fail=True)
        assert R.read_object(out / S2.SHUTDOWN_DIR / S2.APP_ERROR)["error_message"] == "app close failed"
    assert code == 1
    with pytest.raises(S2.ShutdownEvidenceError):
        verify(out, tmp_path, frames=2 if case == "dry-run-failure" else 1)


# --- 부모 검증 ------------------------------------------------------------------------------------


def test_parent_rejects_other_args_foreign_marker_extra_file_and_policy_change(tmp_path: Path) -> None:
    code, _events, out, _settings = run(tmp_path)
    assert code == 0
    for changes in ({"checkpoint_sha256": "0" * 64}, {"source_id": "other"}, {"frames": 2}):
        with pytest.raises(S2.ShutdownEvidenceError):
            verify(out, tmp_path, **changes)
    folder = out / S2.SHUTDOWN_DIR
    original = (folder / S2.SESSION_RETURNED).read_bytes()
    _rewrite(folder / S2.SESSION_RETURNED, lambda value: value.update(pid=value["pid"] + 1))
    with pytest.raises(S2.ShutdownEvidenceError, match="프로세스"):
        verify(out, tmp_path)
    (folder / S2.SESSION_RETURNED).write_bytes(original)
    _rewrite(folder / S2.POLICY, lambda value: value["evidence"].update(dry_run_result_sha256="0" * 64))
    with pytest.raises(S2.ShutdownEvidenceError, match="정책"):
        verify(out, tmp_path)


# --- 실제 CLI 흐름(subprocess) -------------------------------------------------------------------

CHILD = """
import os, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path[:0] = {paths!r}
import record_agentic_state_sidecar_v1 as S1
import record_agentic_state_sidecar_v2 as S2
import test_agentic_state_sidecar_v1 as T1
mode = {mode!r}
out = Path(sys.argv[sys.argv.index("--out") + 1])

class Settings:
    def __init__(self): self.value = False
    def get_as_bool(self, key): return self.value
    def set_bool(self, key, value): self.value = value

SETTINGS = Settings()

class App:
    def close(self, **kwargs):
        assert kwargs == {{"wait_for_replicator": False}}
        if mode == "sdk-exit" and SETTINGS.value is True:
            os._exit(0)  # SDK가 Python 반환 전에 자연 종료하는 경우의 모의
        if mode == "app-crash":
            os._exit(3)

class Session(T1.FakeSession):
    def close(self):
        if mode == "session-crash":
            os._exit(3)  # 봉인 뒤 session close 중 비정상 종료의 모의(-11 대신 3)

S2._launch_app = App
S2._carb_settings = lambda: SETTINGS
S1._open_play_session = lambda path: Session(T1._base(T1.FakeContact()), [])
raise SystemExit(S2.main(sys.argv[1:]))
"""


def _cli(tmp: Path, mode: str | None) -> tuple[subprocess.CompletedProcess[str], Path]:
    checkpoint, sha = _checkpoint(tmp)
    out = tmp / "state-smoke"
    args = ["state-dry-run", "--checkpoint", str(checkpoint), "--checkpoint-sha256", sha, "--source-id", SOURCE_ID,
            "--out", str(out), "--frames", "1"]  # UNIT1 _state_smoke와 같은 인자, script만 v2
    if mode is None:
        command = [sys.executable, "-B", str(SCRIPT), *args]
    else:
        paths = [str(ROOT / name) for name in ("bench", "evals", "src", "tests")]
        command = [sys.executable, "-B", "-c", CHILD.format(paths=paths, mode=mode), *args]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                          cwd=str(ROOT), timeout=300, check=False), out


@pytest.mark.parametrize(("mode", "exit_code", "returned"), [("returns", 0, True), ("sdk-exit", 0, False)])
def test_cli_success_with_or_without_python_return_from_app_close(
    tmp_path: Path, mode: str, exit_code: int, returned: bool
) -> None:
    process, out = _cli(tmp_path, mode)
    assert process.returncode == exit_code, process.stderr
    result = verify(out, tmp_path)
    assert result["app_close_returned"] is returned and result["child_pid"] != os.getpid()


def test_cli_crash_in_session_close_leaves_only_entry_marker_and_is_rejected(tmp_path: Path) -> None:
    process, out = _cli(tmp_path, "session-crash")
    assert process.returncode == 3
    assert S2.shutdown_markers(out) == [S2.APP_STARTUP, S2.SESSION_ENTERED]
    assert (out / "dry_run_result.json").is_file()  # 봉인은 남지만 성공이 아니다
    with pytest.raises(S2.ShutdownEvidenceError):
        verify(out, tmp_path)


def test_cli_crash_inside_app_close_needs_parent_exit_code_rejection(tmp_path: Path) -> None:
    process, out = _cli(tmp_path, "app-crash")
    assert process.returncode == 3
    assert verify(out, tmp_path)["app_close_returned"] is False  # 증거만으로는 구분되지 않는다: 부모 exit 0 확인이 필수


def test_real_script_entry_seals_sdk_import_failure_and_import_is_cpu_only(tmp_path: Path) -> None:
    probe = subprocess.run(
        [sys.executable, "-B", "-c", "import sys; sys.path[:0] = [r'%s']; import record_agentic_state_sidecar_v2; "
         "print(sorted(name for name in ('isaaclab', 'torch', 'carb', 'omni') if name in sys.modules))"
         % (ROOT / "bench")],
        capture_output=True, text=True, encoding="utf-8", timeout=120, check=True)
    assert probe.stdout.strip() == "[]"
    process, out = _cli(tmp_path, None)  # 이 CPU 환경에는 Isaac이 없어 AppLauncher import가 실패한다
    assert process.returncode == 1
    failure = R.read_object(out / "dry_run_failure.json")
    assert failure["completed"] is False and failure["records_written"] == 0
    assert S2.shutdown_markers(out) == []
