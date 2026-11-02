"""R4 epoch-2 loop의 typed view와 paired probe backend parent를 결합한다."""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import five_policy_recovery_epoch2_v1 as OLD
import local_pair_probe_host_v1 as BASE_HOST
import runtime_observation_view_v1 as RV

SOURCE_FILES = (
    "evals/recovery_epoch2_probe_boundary_v1.py",
    "tests/test_recovery_epoch2_probe_boundary_v1.py",
)
_ORIGINAL_VERIFY = OLD.verify_loop_protocol
_ORIGINAL_HOST = OLD._HOST
_LOCK = RLock()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("R4 protocol 구성요소는 repo 내부여야 한다") from error


def _ref(path: Path) -> dict[str, str]:
    checked = Path(path).resolve(strict=True)
    return {"path": _relative(checked), "sha256_lf": OLD.sha(checked, lf=True)}


def _isolated(module: ModuleType, name: str) -> ModuleType:
    path = Path(str(module.__file__)).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    clone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clone)
    return clone


def _facade(module: ModuleType, name: str) -> ModuleType:
    clone = ModuleType(name)
    clone.__dict__.update(module.__dict__)
    clone.__name__ = name
    return clone


def _derive_protocol(parent_path: Path, parent_anchor: str, parent: dict[str, Any]) -> dict[str, Any]:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", parent_anchor)), "parent anchor 형식")
    legacy_view = parent.get("view_protocol")
    require(isinstance(legacy_view, dict), "기존 selection view protocol이 없다")
    value = deepcopy(parent)
    value["selection_view_protocol"] = deepcopy(legacy_view)
    value["view_protocol"] = _ref(RV.PROTOCOL)
    value["parent_loop_protocol"] = {
        **_ref(parent_path),
        "anchor": parent_anchor,
    }
    sources = value.get("sources_lf")
    require(isinstance(sources, dict), "기존 R4 source 결합이 없다")
    sources = cast(dict[str, str], sources)
    value["sources_lf"] = {
        **deepcopy(sources),
        **{name: OLD.sha(ROOT / name, lf=True) for name in SOURCE_FILES},
    }
    return value


def prepare_protocol(parent_loop_path: Path, parent_anchor: str) -> dict[str, Any]:
    """기존 R4 loop를 검증한 뒤 runtime/selection view 역할을 분리한다."""
    parent_path = Path(parent_loop_path).resolve(strict=True)
    parent = cast(dict[str, Any], _ORIGINAL_VERIFY(parent_path, parent_anchor))
    return _derive_protocol(parent_path, parent_anchor, parent)


def verify_loop_protocol(
    path: Path,
    anchor: str,
    *,
    lab: Path | None = None,
) -> dict[str, Any]:
    """새 loop와 보존된 parent loop 양쪽의 기존 검증기를 모두 실행한다."""
    value = cast(dict[str, Any], OLD.TRANSPORT._verify_protocol(Path(path), anchor))
    parent_ref = value.get("parent_loop_protocol")
    require(isinstance(parent_ref, dict), "parent loop protocol ref가 없다")
    parent_ref = cast(dict[str, Any], parent_ref)
    parent_path = ROOT / str(parent_ref.get("path", ""))
    parent_anchor = parent_ref.get("anchor")
    require(
        isinstance(parent_anchor, str)
        and parent_ref == {**_ref(parent_path), "anchor": parent_anchor},
        "parent loop protocol ref가 바뀌었다",
    )
    parent_anchor = cast(str, parent_anchor)
    parent = cast(
        dict[str, Any],
        _ORIGINAL_VERIFY(parent_path, parent_anchor, lab=lab),
    )
    expected = _derive_protocol(parent_path, parent_anchor, parent)
    require(value == expected, "typed R4 loop가 parent에서 정확히 파생되지 않았다")
    runtime_ref = value["view_protocol"]
    require(runtime_ref == _ref(RV.PROTOCOL), "runtime view protocol 결합이 다르다")
    RV.verify_protocol(RV.PROTOCOL, anchor)
    return deepcopy(value)


def _backend_registrations(
    sources: Sequence[dict[str, Any]],
    parents: Mapping[str, dict[str, Any]],
    protocol: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    require(1 <= len(sources) <= 4, "R4 probe source 수 범위")
    targets = {f"{source.get('case_id')}:{source.get('policy')}" for source in sources}
    require(set(parents) == targets and len(targets) == len(sources), "R4 source/parent target 결합")
    registered: dict[str, str] | None = None
    selection_ref: dict[str, str] = _ref(BASE_HOST.C.V.PROTOCOL)
    if protocol is not None:
        registered_value = protocol.get("source_digests")
        require(isinstance(registered_value, dict), "R4 source digest 등록이 없다")
        registered = cast(dict[str, str], registered_value)
        selection_value = protocol.get("selection_view_protocol")
        require(isinstance(selection_value, dict), "selection view protocol ref가 없다")
        selection_ref = cast(dict[str, str], selection_value)
    selection_path = ROOT / str(selection_ref.get("path", ""))
    require(selection_ref == _ref(selection_path), "selection view protocol ref가 바뀌었다")
    selection_sha = BASE_HOST.C.V.file_digest(selection_path)

    result: dict[str, dict[str, Any]] = {}
    for source in sources:
        case_id, arm = source.get("case_id"), source.get("policy")
        target = f"{case_id}:{arm}"
        require((case_id, arm) in OLD.TARGETS, "등록되지 않은 R4 probe target")
        case_id, arm = cast(str, case_id), cast(str, arm)
        if registered is not None:
            require(registered.get(target) == OLD.digest(source), "R4 probe source digest가 미등록이다")
        public_packet, backend_packet = BASE_HOST._packets(
            source["packet"], source["backend_packet"])
        policy = OLD._DRIVER._storage_policy(arm)
        public_run_id = f"r4_{str(case_id).replace('-', '_')}_{arm}"
        public_parent = deepcopy(parents[target])
        expected_public_parent = {
            "run_id": public_run_id,
            "policy": policy,
            "epoch": 1,
            **deepcopy(public_packet["prior_receipt"]),
        }
        require(public_parent == expected_public_parent, "public parent가 source/arm과 다르다")
        public_binding = BASE_HOST.C.V.epoch_binding(
            public_packet, public_run_id, policy, selection_sha, public_parent)
        backend_run_id = "local_pair_backend_" + BASE_HOST.digest(public_binding)[:12]
        backend_parent = {
            "run_id": backend_run_id,
            "policy": policy,
            "epoch": 1,
            **deepcopy(backend_packet["prior_receipt"]),
        }
        require(
            backend_parent["checkpoint_sha256"]
            == backend_packet["input_bindings"]["case"]["checkpoint_sha256"],
            "backend prior checkpoint가 case input과 다르다",
        )
        packet_digest = OLD.digest(backend_packet)
        require(packet_digest not in result, "서로 다른 target의 backend packet이 같다")
        result[packet_digest] = {
            "target": target,
            "packet": backend_packet,
            "run_id": backend_run_id,
            "policy": policy,
            "parent": backend_parent,
            "public_binding": public_binding,
        }
    return result


@contextmanager
def loop_protocol_scope(
    protocol_path: Path,
    anchor: str,
) -> Iterator[dict[str, Any]]:
    """OLD loop verifier를 등록된 typed loop verifier로 scope 동안만 바꾼다."""
    with _LOCK:
        protocol_path = Path(protocol_path).resolve(strict=True)
        protocol = verify_loop_protocol(protocol_path, anchor)
        original_verify = OLD.verify_loop_protocol

        def scoped_verify(path: Path, execution_anchor: str, *, lab: Path | None = None) -> dict:
            require(
                Path(path).resolve() == protocol_path and execution_anchor == anchor,
                "다른 typed R4 loop를 요청했다",
            )
            return verify_loop_protocol(protocol_path, anchor, lab=lab)

        OLD.__dict__["verify_loop_protocol"] = scoped_verify
        try:
            yield deepcopy(protocol)
        finally:
            current = OLD.verify_loop_protocol
            OLD.__dict__["verify_loop_protocol"] = original_verify
            if current is not scoped_verify:
                raise RuntimeError("R4 loop verifier가 scope 중 외부에서 바뀌었다")


@contextmanager
def backend_parent_scope(
    sources: Sequence[dict[str, Any]],
    *,
    parents: Mapping[str, dict[str, Any]],
) -> Iterator[dict[str, dict[str, Any]]]:
    """backend parent를 OLD host의 격리 clone에만 연결한다."""
    with _LOCK:
        registrations = _backend_registrations(sources, parents)
        require(OLD._HOST is _ORIGINAL_HOST, "R4 probe host가 이미 바뀌었다")

        host = _isolated(BASE_HOST, "_recovery_epoch2_probe_host_v1")
        host.__dict__["OBS"] = OLD._PROBE
        controller = _facade(BASE_HOST.C, "_recovery_epoch2_backend_controller_v1")
        view = _facade(BASE_HOST.C.V, "_recovery_epoch2_backend_view_v1")
        original_build = BASE_HOST.C.V.build_snapshot

        def build_snapshot(
            packet: dict,
            run_id: str,
            policy: str,
            protocol_sha256: str,
            ledger_path: str | Path,
            parent: dict | None = None,
            masked: bool = False,
        ) -> dict:
            registration = registrations.get(OLD.digest(packet))
            if registration is None or packet != registration["packet"]:
                raise ValueError("backend packet이 등록 source와 다르다")
            require(run_id == registration["run_id"], "backend run_id가 public binding과 다르다")
            require(policy == registration["policy"], "backend policy가 source arm과 다르다")
            expected_parent = registration["parent"]
            require(parent is None or parent == expected_parent, "backend parent가 등록 계보와 다르다")
            return original_build(
                packet,
                run_id,
                policy,
                protocol_sha256,
                ledger_path,
                parent=deepcopy(expected_parent),
                masked=masked,
            )

        view.__dict__["build_snapshot"] = build_snapshot
        controller.__dict__["V"] = view
        host.__dict__["C"] = controller
        OLD.__dict__["_HOST"] = host
        try:
            yield {
                row["target"]: {
                    "backend_run_id": row["run_id"],
                    "backend_policy": row["policy"],
                    "backend_parent": deepcopy(row["parent"]),
                    "public_binding": deepcopy(row["public_binding"]),
                }
                for row in registrations.values()
            }
        finally:
            current_host = OLD._HOST
            OLD.__dict__["_HOST"] = _ORIGINAL_HOST
            if current_host is not host:
                raise RuntimeError("R4 probe boundary가 scope 중 외부에서 바뀌었다")


@contextmanager
def probe_boundary_scope(
    sources: Sequence[dict[str, Any]],
    *,
    parents: Mapping[str, dict[str, Any]],
    protocol_path: Path,
    anchor: str,
) -> Iterator[dict[str, dict[str, Any]]]:
    """등록 typed loop와 backend parent scope를 한 실행 경계로 묶는다."""
    with loop_protocol_scope(protocol_path, anchor) as protocol:
        _backend_registrations(sources, parents, protocol)
        with backend_parent_scope(sources, parents=parents) as registrations:
            yield registrations
