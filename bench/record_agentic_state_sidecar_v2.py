"""상태 dry-run child v2: SIDECAR1 ``run_state_dry_run``을 그대로 실행하고 종료 단계 증거와 기존 fast shutdown 정책을 더한다.

- 실행 본체·상태 reader·profile·sidecar·``dry_run_result.json``·실패 봉인은 SIDECAR1 그대로다. app·session proxy만 주입한다.
- 종료 단계마다 ``<out>/shutdown/``에 write-once marker를 남긴다(pid·monotonic_ns). 새 관측·라벨·임계값은 없다.
- 실제 AppLauncher(SIDECAR1 기본 인자: headless, 카메라 없음, fast_shutdown=False)를 만든 뒤에만 ``carb.settings``를
  import하고 startup ``/app/fastShutdown`` false를 확인한다.
- fast shutdown은 session close가 정상 반환되고 봉인 결과·sidecar·CP·source·frame이 모두 맞으며 실패가 없을 때만
  적용한다. 순서는 기존 ``vision_shutdown_policy_v1``과 같다: 근거 확인 → false→true 전환·readback → 영수증 →
  ``close(wait_for_replicator=False)``. SDK가 close 안에서 자연 종료하면 반환 marker는 남지 않는다.
- 조건이 어긋나면 fastShutdown을 바꾸지 않고 거부 영수증을 남긴 뒤 SDK close를 시도하고, child는 1로 끝난다.
  실패를 성공으로 바꾸거나 강제 종료 코드를 만들지 않는다.
- 부모는 ``verify_shutdown_evidence``로 봉인·session close·정책·app close 진입을 확인한다. 이 정책이 실제 SDK 종료
  충돌을 해결하는지는 GPU 실행 전에 주장하지 않는다.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import record_agentic_state_sidecar_v1 as SIDECAR1
import record_vision_evidence_v1 as R
import vision_shutdown_policy_v1 as POLICY1

SOURCE_PATH = "bench/record_agentic_state_sidecar_v2.py"
TEST_PATH = "tests/test_agentic_state_sidecar_v2.py"
SOURCE_FILES = (SOURCE_PATH, TEST_PATH)
SETTING = POLICY1.SETTING
APP_FACTORY = "record_agentic_state_sidecar_v1._default_app_factory"
SHUTDOWN_DIR = "shutdown"
MARKER_SCHEMA = "agentic_state_shutdown_marker_v1"
POLICY_SCHEMA = "agentic_state_shutdown_policy_v1"
REFUSAL_SCHEMA = "agentic_state_shutdown_refusal_v1"
EVIDENCE_SCHEMA = "agentic_state_shutdown_evidence_v1"
POLICY_PHASE = "after_sealed_state_result_and_session_close_before_sdk_close"
APP_STARTUP = "app-startup.json"
SESSION_ENTERED = "session-close-entered.json"
SESSION_RETURNED = "session-close-returned.json"
SESSION_ERROR = "session-close-error.json"
POLICY = "shutdown-policy.json"
REFUSAL = "shutdown-refusal.json"
APP_ENTERED = "app-close-entered.json"
APP_RETURNED = "app-close-returned.json"
APP_ERROR = "app-close-error.json"
REQUIRED = (APP_STARTUP, SESSION_ENTERED, SESSION_RETURNED, POLICY, APP_ENTERED)
OPTIONAL = (APP_RETURNED,)
ORDER = (APP_STARTUP, SESSION_ENTERED, SESSION_RETURNED, POLICY, APP_ENTERED, APP_RETURNED)
_RUNTIME_KEYS = ("pid", "sequence", "monotonic_ns")
_SEQUENCE = itertools.count(1)  # 같은 프로세스 안 기록 순서(monotonic_ns는 OS 해상도에 따라 같을 수 있다)


class ShutdownEvidenceError(ValueError):
    """봉인·종료 단계 증거가 계약과 다르다. 부모는 거부한다."""


class ShutdownRefused(RuntimeError):
    """fast shutdown 조건이 맞지 않아 정책을 적용하지 않았다(child 실패로 끝난다)."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ShutdownEvidenceError(reason)


def _stamp() -> dict[str, int]:
    return {"pid": os.getpid(), "sequence": next(_SEQUENCE), "monotonic_ns": time.monotonic_ns()}


def _error(error: BaseException) -> dict[str, Any]:
    return {"error_type": f"{type(error).__module__}.{type(error).__qualname__}", "error_message": str(error),
            "traceback": traceback.format_exception(type(error), error, error.__traceback__)}


def _marker(out: Path, name: str, **fields: Any) -> Path:
    path = out / SHUTDOWN_DIR / name
    R.write_new(path, {"schema": MARKER_SCHEMA, "event": name.removesuffix(".json"), **_stamp(), **fields})
    return path


def _safe_marker(out: Path, name: str, **fields: Any) -> None:
    """오류 경로의 marker. 기록 실패가 원래 오류를 가리지 않게 한다."""
    try:
        _marker(out, name, **fields)
    except BaseException:  # noqa: BLE001 - 원래 오류를 우선한다.
        traceback.print_exc()


def _regular(path: Path, label: str) -> Path:
    try:
        return POLICY1._single_regular(path, label)
    except ValueError as error:
        raise ShutdownEvidenceError(str(error)) from error


def _terminal(out: Path, *, checkpoint_sha256: str, source_id: str, frames: int) -> dict[str, str]:
    """SIDECAR1 봉인 결과와 sidecar가 CLI 인자와 맞는지 확인하고 근거 SHA를 돌려준다."""
    sidecar = out / "sidecar"
    _require(not (out / "dry_run_failure.json").exists() and not (sidecar / SIDECAR1.FAILURE_NAME).exists(),
             "dry-run 또는 sidecar 실패가 봉인됐다")
    result_path = _regular(out / "dry_run_result.json", "dry-run result")
    manifest_path = _regular(sidecar / SIDECAR1.MANIFEST_NAME, "sidecar manifest")
    try:
        result = R.read_object(result_path)
        manifest, records = SIDECAR1.load_sidecar(sidecar)
    except (OSError, TypeError, ValueError, KeyError, SIDECAR1.SidecarError) as error:
        raise ShutdownEvidenceError(f"봉인 결과를 읽을 수 없다: {error}") from error
    _require(result.get("schema") == SIDECAR1.DRY_RUN_SCHEMA and result.get("completed") is True
             and result.get("frames") == frames and result.get("native_recording_touched") is False
             and result.get("used_for_decision") is False, "dry-run 결과가 완료·frame 계약과 다르다")
    manifest_sha = R.sha256_file(manifest_path)
    _require(result.get("sidecar_manifest_sha256") == manifest_sha, "dry-run 결과의 sidecar manifest SHA가 다르다")
    source = manifest.get("source") or {}
    _require(source.get("checkpoint_sha256") == checkpoint_sha256 and source.get("source_id") == source_id,
             "sidecar source의 checkpoint·source ID가 인자와 다르다")
    _require(manifest.get("frame_count") == frames and len(records) == frames, "sidecar frame 수가 인자와 다르다")
    return {"dry_run_result_sha256": R.sha256_file(result_path), "sidecar_manifest_sha256": manifest_sha,
            "sidecar_records_sha256": manifest["records"]["sha256"]}


def _policy_value(evidence: Mapping[str, str], *, checkpoint_sha256: str, source_id: str, frames: int) -> dict[str, Any]:
    return {"schema": POLICY_SCHEMA, "completed": True, "phase": POLICY_PHASE, "setting": SETTING,
            "startup_fast_shutdown": False, "before": False, "after": True, "wait_for_replicator": False,
            "session_close": "returned", "checkpoint_sha256": checkpoint_sha256, "source_id": source_id,
            "frames": frames, "evidence": dict(evidence)}


def _present(out: Path) -> set[str]:
    folder = out / SHUTDOWN_DIR
    return {path.name for path in folder.iterdir()} if folder.is_dir() else set()


# --- child proxy ----------------------------------------------------------------------------------


class SessionProxy:
    """session.close 진입·반환·오류를 기록한다. 그 밖의 동작은 원래 session에 넘긴다."""

    def __init__(self, session: Any, out: Path) -> None:
        self._session = session
        self._out = out

    @property
    def base(self) -> Any:
        return self._session.base

    def advance(self) -> None:
        self._session.advance()

    def close(self) -> None:
        _marker(self._out, SESSION_ENTERED)
        try:
            self._session.close()
        except BaseException as error:
            _safe_marker(self._out, SESSION_ERROR, **_error(error))
            raise
        _marker(self._out, SESSION_RETURNED)


class AppProxy:
    """SDK close 전에 봉인 근거를 확인하고, 맞을 때만 기존 fast shutdown 정책을 적용한다."""

    def __init__(self, app: Any, settings: Any, out: Path, *, checkpoint_sha256: str, source_id: str,
                 frames: int) -> None:
        self._app = app
        self._settings = settings
        self._out = out
        self._expected = {"checkpoint_sha256": checkpoint_sha256, "source_id": source_id, "frames": frames}
        if settings.get_as_bool(SETTING) is not False:
            raise ValueError("App 생성 직후 fastShutdown이 false가 아니다")
        _marker(out, APP_STARTUP, startup_fast_shutdown=False, setting=SETTING, app_factory=APP_FACTORY)

    def update(self) -> Any:
        return self._app.update()

    def _apply(self) -> Path:
        present = _present(self._out)
        _require(SESSION_RETURNED in present and SESSION_ERROR not in present, "session close가 정상 반환되지 않았다")
        evidence = _terminal(self._out, **self._expected)
        if self._settings.get_as_bool(SETTING) is not False:
            raise ShutdownEvidenceError("봉인 확인 뒤 fastShutdown 시작값이 false가 아니다")
        self._settings.set_bool(SETTING, True)
        if self._settings.get_as_bool(SETTING) is not True:
            raise RuntimeError("fastShutdown true 전환 readback이 실패했다")
        path = self._out / SHUTDOWN_DIR / POLICY
        R.write_new(path, {**_policy_value(evidence, **self._expected), **_stamp()})
        return path

    def close(self) -> Any:
        policy: Path | None = None
        reason: dict[str, Any] | None = None
        try:
            policy = self._apply()
        except BaseException as error:  # noqa: BLE001 - 거부 사유를 남기고 SDK close는 이어간다.
            reason = _error(error)
            try:
                current = self._settings.get_as_bool(SETTING)
            except BaseException:  # noqa: BLE001
                current = None
            try:
                R.write_new(self._out / SHUTDOWN_DIR / REFUSAL,
                            {"schema": REFUSAL_SCHEMA, **_stamp(), "fast_shutdown_now": current, **reason})
            except BaseException:  # noqa: BLE001 - 거부 사실은 아래 예외로도 남는다.
                traceback.print_exc()
        entered_error: BaseException | None = None
        try:
            _marker(self._out, APP_ENTERED, fast_shutdown_applied=policy is not None, wait_for_replicator=False,
                    policy_sha256=None if policy is None else R.sha256_file(policy))
        except BaseException as error:  # noqa: BLE001 - marker 실패에도 SDK close는 시도한다.
            entered_error = error
        try:
            result = self._app.close(wait_for_replicator=False)
        except BaseException as error:
            _safe_marker(self._out, APP_ERROR, **_error(error))
            raise
        _marker(self._out, APP_RETURNED)
        if entered_error is not None:
            raise entered_error
        if reason is not None:
            raise ShutdownRefused(f"fast shutdown을 적용하지 않았다: {reason['error_message']}")
        return result


def _launch_app() -> Any:
    return SIDECAR1._default_app_factory()


def _carb_settings() -> Any:
    import carb.settings  # pyright: ignore[reportMissingImports]

    return carb.settings.get_settings()


def run_state_dry_run(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source_id: str,
    out_dir: Path,
    frames: int = 1,
    launch_app: Callable[[], Any] | None = None,
    settings_factory: Callable[[], Any] | None = None,
    session_factory: Callable[[Path], Any] | None = None,
    reader_factory: Callable[[Any, int], Any] = SIDECAR1._default_reader_factory,
) -> int:
    """SIDECAR1 ``run_state_dry_run``에 종료 증거 proxy를 주입해 실행한다. 반환값은 SIDECAR1과 같다."""
    out = Path(out_dir).absolute()
    expected = {"checkpoint_sha256": checkpoint_sha256, "source_id": source_id, "frames": frames}
    launch = launch_app or _launch_app
    settings_of = settings_factory or _carb_settings
    open_session = session_factory or SIDECAR1._open_play_session

    def app_factory() -> Any:
        app = launch()
        try:
            return AppProxy(app, settings_of(), out, **expected)
        except BaseException:
            app.close(wait_for_replicator=False)
            raise

    return SIDECAR1.run_state_dry_run(
        checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha256, source_id=source_id, out_dir=out,
        frames=frames, app_factory=app_factory, session_factory=lambda path: SessionProxy(open_session(path), out),
        reader_factory=reader_factory,
    )


# --- 부모 검증 ------------------------------------------------------------------------------------


def shutdown_markers(out_dir: Path) -> list[str]:
    """읽기 전용 진단. 있는 shutdown 파일 이름을 기록 순서(sequence)로 돌려준다. 읽을 수 없는 파일은 뒤에 둔다."""
    folder = Path(out_dir) / SHUTDOWN_DIR
    if not folder.is_dir():
        return []
    rows = []
    for path in folder.iterdir():
        try:
            order = R.read_object(path).get("sequence")
        except (OSError, TypeError, ValueError):
            order = None
        rows.append((0 if isinstance(order, int) else 1, order if isinstance(order, int) else 0, path.name))
    return [name for _, _, name in sorted(rows)]


def verify_shutdown_evidence(out_dir: Path, *, checkpoint_sha256: str, source_id: str, frames: int) -> dict[str, Any]:
    """부모 수용 전 검증. 봉인 결과·session close 반환·정책 영수증·app close 진입을 확인한다."""
    out = Path(out_dir).absolute()
    _require(out.is_dir() and not out.is_symlink() and not any(parent.is_symlink() for parent in out.parents),
             "dry-run 출력이 일반 디렉터리가 아니다")
    evidence = _terminal(out, checkpoint_sha256=checkpoint_sha256, source_id=source_id, frames=frames)
    folder = out / SHUTDOWN_DIR
    _require(folder.is_dir() and not folder.is_symlink(), "shutdown 증거 폴더가 없다")
    names = _present(out)
    _require(set(REQUIRED) <= names <= set(REQUIRED) | set(OPTIONAL),
             f"shutdown 증거 집합이 성공 계약과 다르다: {shutdown_markers(out)}")
    paths = {name: _regular(folder / name, name) for name in names}
    try:
        values = {name: R.read_object(path) for name, path in paths.items()}
    except (OSError, TypeError, ValueError) as error:
        raise ShutdownEvidenceError(f"shutdown 증거를 읽을 수 없다: {error}") from error
    pids = {value.get("pid") for value in values.values()}
    _require(len(pids) == 1 and isinstance(next(iter(pids)), int), "shutdown 증거의 프로세스가 하나가 아니다")
    for name, value in values.items():
        if name != POLICY:
            _require(value.get("schema") == MARKER_SCHEMA and value.get("event") == name.removesuffix(".json"),
                     f"{name} marker 형식이 다르다")
    sequence: list[Any] = [values[name].get("sequence") for name in ORDER if name in names]
    clock: list[Any] = [values[name].get("monotonic_ns") for name in ORDER if name in names]
    _require(all(isinstance(item, int) for item in sequence + clock)
             and all(left < right for left, right in itertools.pairwise(sequence))
             and all(left <= right for left, right in itertools.pairwise(clock)), "shutdown 단계 순서가 다르다")
    startup = values[APP_STARTUP]
    _require(startup.get("startup_fast_shutdown") is False and startup.get("setting") == SETTING
             and startup.get("app_factory") == APP_FACTORY, "App startup fastShutdown 기록이 다르다")
    policy = {key: value for key, value in values[POLICY].items() if key not in _RUNTIME_KEYS}
    expected = _policy_value(evidence, checkpoint_sha256=checkpoint_sha256, source_id=source_id, frames=frames)
    _require(policy == expected, "shutdown 정책 영수증이 봉인 결과·인자와 다르다")
    policy_sha = R.sha256_file(paths[POLICY])
    entered = values[APP_ENTERED]
    _require(entered.get("fast_shutdown_applied") is True and entered.get("wait_for_replicator") is False
             and entered.get("policy_sha256") == policy_sha, "app close 진입 기록이 정책과 다르다")
    return {"schema": EVIDENCE_SCHEMA, "child_pid": next(iter(pids)), "frames": frames,
            "checkpoint_sha256": checkpoint_sha256, "source_id": source_id, **evidence,
            "shutdown_policy_sha256": policy_sha, "app_close_returned": APP_RETURNED in names,
            "markers": {name: R.sha256_file(path) for name, path in sorted(paths.items())}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dry = commands.add_parser("state-dry-run")
    dry.add_argument("--checkpoint", required=True, type=Path)
    dry.add_argument("--checkpoint-sha256", required=True)
    dry.add_argument("--source-id", required=True)
    dry.add_argument("--out", required=True, type=Path)
    dry.add_argument("--frames", type=int, default=1)
    parsed = parser.parse_args(argv)
    return run_state_dry_run(checkpoint_path=parsed.checkpoint, checkpoint_sha256=parsed.checkpoint_sha256,
                             source_id=parsed.source_id, out_dir=parsed.out, frames=parsed.frames)


if __name__ == "__main__":
    raise SystemExit(main())
