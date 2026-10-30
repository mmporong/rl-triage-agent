"""Parent9에 영속 등록 GPU 관측 root와 경로 재결합 reader를 더한다."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if all((candidate / name).is_dir() for name in ("bench", "evals", "src")):
            return candidate
    raise RuntimeError("repository root unavailable")


ROOT = _repo_root()
for location in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    value = str(location)
    if value not in sys.path:
        sys.path.insert(0, value)

import run_agentic_brev_probe_parent_v9 as V9

V1 = V9.V1
V6 = V9.V8.V6
SCHEMA = V9.SCHEMA
SOURCE = "bench/run_agentic_brev_probe_parent_v10.py"
TEST_SOURCE = "tests/test_run_agentic_brev_probe_parent_v10.py"
SOURCES = V9.SOURCES | {SOURCE, TEST_SOURCE}


def _inputs(request_ref: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    request = V1.U.read_ref(request_ref)
    return V1.U.read_ref(request["protocol"]), V1.U.read_ref(request["binding"])


def registered_observation_roots(protocol: dict[str, Any], binding: dict[str, Any], *, home: Path | None = None
                                ) -> dict[str, dict[str, Any]]:
    layout = protocol.get("layout")
    if not isinstance(layout, dict) or set(layout) != {"repo", "isaaclab", "python"}:
        raise ValueError("HOME layout fields")
    base = (home or Path.home()).absolute()
    repo = base.joinpath(*V1.U._relative(layout["repo"])).absolute()
    operations = binding.get("expected_request", {}).get("execution", {}).get("operations")
    if not isinstance(operations, dict) or set(operations) != set(V1.ORDER):
        raise ValueError("exact observation operations")
    roots = binding.get("roots")
    if not isinstance(roots, list):
        raise TypeError("binding roots")
    registration = [row for row in roots if row.get("remote", "").endswith("/registration")]
    V1.require(len(registration) == 1, "single registration root")
    registration_local = Path(registration[0]["local"])
    V1.require(registration_local.is_absolute(), "registration local root")
    values: dict[str, dict[str, Any]] = {}
    for arm in V1.ORDER:
        operation = operations[arm]
        operation_id = operation["operation_id"]
        remote = str(PurePosixPath(operation["remote_stage"]).parent / "gpu-observations")
        local = registration_local / "gpu-observations" / operation_id
        V1.require(roots.count({"remote": remote, "local": str(local)}) == 1,
                   "registered stable GPU observation root")
        values[operation_id] = {"arm": arm, "remote": remote, "local": local, "repo": repo}
    V1.require(len(values) == 2, "distinct stable GPU observation roots")
    return values


def ensure_observation_roots(protocol: dict[str, Any], binding: dict[str, Any], *, home: Path | None = None
                            ) -> dict[str, dict[str, Any]]:
    values = registered_observation_roots(protocol, binding, home=home)
    pair_id = binding["expected_request"]["execution"]["pair_id"]
    repo = next(iter(values.values()))["repo"]
    pair_folder = repo.joinpath(*V1.U._relative(protocol["execution_root"])) / "pairs" / pair_id
    for value in values.values():
        local = value["local"]
        V1.require(not any(path.is_symlink() for path in (local, *local.parents)),
                   "GPU observation root symlink ancestor")
        local.mkdir(parents=True, exist_ok=True)
        V1.require(local.is_dir() and not local.is_symlink(), "GPU observation root directory")
        if not pair_folder.exists():
            V1.require(not any(local.iterdir()), "stale GPU observation root before pair start")
    return values


class ReboundReader:
    """원문은 등록 local root에 쓰고 receipt 경로만 등록 remote root로 돌려준다."""

    def __init__(self, reader: Any, local: Path, remote: str) -> None:
        self.reader, self.local, self.remote = reader, local.resolve(strict=True), PurePosixPath(remote)

    @property
    def last_receipt(self) -> dict[str, str] | None:
        reference = self.reader.last_receipt
        if reference is None:
            return None
        path = Path(reference["path"]).resolve(strict=True)
        V1.require(path.parent == self.local and path.name.endswith(".receipt.json"),
                   "GPU observation receipt outside registered local root")
        return {"path": str(self.remote / path.name), "sha256": reference["sha256"]}

    def __call__(self) -> list[int]:
        return self.reader()


class RegisteredObservationServices(V1.ActualServices):
    def __init__(self, roots: Mapping[str, Mapping[str, Any]]) -> None:
        super().__init__()
        self.observation_roots = roots

    def reader(self, capture: dict[str, Any], folder: Path) -> ReboundReader:
        folder = Path(folder)
        operation_id = folder.parent.name
        V1.require(folder.name == "gpu-observations" and operation_id in self.observation_roots,
                   "unregistered operation observation folder")
        value = self.observation_roots[operation_id]
        local = Path(value["local"])
        return ReboundReader(super().reader(capture, local), local, value["remote"])


def _prepare_impl(request_ref: dict, *, allow_fixture: bool = False, home: Path | None = None,
                  now=None) -> dict:
    protocol, binding = _inputs(request_ref)
    ensure_observation_roots(protocol, binding, home=home)
    return V6._prepare_impl(request_ref, allow_fixture=allow_fixture, home=home, now=now)


@contextmanager
def parent10_scope(roots: Mapping[str, Mapping[str, Any]] | None = None) -> Iterator[None]:
    with V9.parent9_scope():
        original_prepare, original_services = V1.prepare, V1.ActualServices
        V1.prepare = _prepare_impl
        if roots is not None:
            V1.ActualServices = lambda: RegisteredObservationServices(roots)
        try:
            yield
        finally:
            V1.prepare, V1.ActualServices = original_prepare, original_services


def prepare(*args, **kwargs):
    with parent10_scope():
        return _prepare_impl(*args, **kwargs)


def lookup(*args, **kwargs):
    with parent10_scope():
        return V1.lookup(*args, **kwargs)


def run(request_ref: dict, *, execute: bool = False, allow_fixture: bool = False,
        home: Path | None = None, now=None, services=None) -> dict:
    protocol, binding = _inputs(request_ref)
    roots = ensure_observation_roots(protocol, binding, home=home)
    V1.require(services is None or allow_fixture, "actual custom services forbidden")
    with parent10_scope(roots if services is None else None):
        return V1.run(request_ref, execute=execute, allow_fixture=allow_fixture, home=home, now=now,
                      services=services)


def verify_authority(*args, **kwargs):
    with parent10_scope():
        return V1.verify_authority(*args, **kwargs)


def _verify_completion(*args, **kwargs):
    with parent10_scope():
        return V1._verify_completion(*args, **kwargs)


def main() -> None:
    if "--model-transport" in sys.argv:
        V9.main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        result = run({"path": str(args.request.absolute()), "sha256": args.request_sha256}, execute=args.execute)
    except Exception as error:  # noqa: BLE001 - CLI 실패를 명시 HOLD로 표시한다.
        result = {"status": "HOLD", "dispatched": False,
                  "error_type": type(error).__name__, "reason": str(error)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def __getattr__(name: str):
    return getattr(V9, name)


if __name__ == "__main__":
    main()
