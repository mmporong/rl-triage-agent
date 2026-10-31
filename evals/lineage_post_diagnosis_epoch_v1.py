"""검증한 첫 교정 비회복 관측을 기존 lineage 진단기의 epoch 2로 연결한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import lineage_compound_retry_v1 as RETRY
import lineage_diagnosis_epoch_v1 as E0
import lineage_gpu_wait_v1 as GPU_WAIT
import lineage_lease_context_v2 as LEASE
import lineage_nim_transport_v1 as T
import lineage_post_correction_inputs_v1 as POST
import lineage_post_probe_backend_v1 as POST_BACKEND
import lineage_selection_inputs_v1 as INPUT0

SCHEMA = "lineage_post_diagnosis_epoch_v1"
PROTOCOL_SCHEMA = "lineage_post_diagnosis_epoch_protocol_v1"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
NEW_SOURCE_FILES = (
    "evals/lineage_post_diagnosis_epoch_v1.py",
    "tests/test_lineage_post_diagnosis_epoch_v1.py",
    "evals/lineage_post_recovery_candidate_v1.py",
    "tests/test_lineage_post_recovery_candidate_v1.py",
)
POST_SOURCE_FILES = (
    "evals/lineage_post_correction_inputs_v1.py",
    "tests/test_lineage_post_correction_inputs_v1.py",
)
GPU_WAIT_SOURCE_FILES = (
    "evals/lineage_gpu_wait_v1.py",
    "tests/test_lineage_gpu_wait_v1.py",
)
SOURCE_FILES = tuple(dict.fromkeys(
    (*NEW_SOURCE_FILES, *POST_SOURCE_FILES, *LEASE.SOURCE_FILES,
     *RETRY.DRIVER_SOURCE_FILES, *GPU_WAIT_SOURCE_FILES, *T.SOURCE_FILES)
))
_LOCK = RLock()
_LOCAL = threading.local()
_ORIGINAL_INPUT = E0.INPUT
_ORIGINAL_PROBE = E0.P
_ORIGINAL_SOURCE_BINDING = E0._source_binding
_ORIGINAL_BASE_RESULT = E0._base_result
_POST_INPUT = SimpleNamespace(
    PACKET_SCHEMA=INPUT0.PACKET_SCHEMA,
    make_selection_packet=POST.make_selection_packet,
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    payload = Path(path).read_bytes()
    if lf:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("component protocol은 repo 내부여야 한다") from error


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def prepare_protocol(post_backend_protocol: Path) -> dict:
    path = Path(post_backend_protocol).resolve()
    backend = _read(path)
    _require(backend.get("schema") == POST_BACKEND.PROTOCOL_SCHEMA,
             "typed post backend protocol schema가 다르다")
    return {
        "schema": PROTOCOL_SCHEMA,
        "phase": "epoch2_after_valid_not_recovered_receipt_first_diagnosis_only",
        "component_post_backend_protocol": {
            "path": _relative(path), "sha256_lf": _sha(path, lf=True),
        },
        "source_sha256_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "budget": {"max_decisions": 3, "execution_request_budget_s": 1200,
                   "model_logical_slots": 3, "model_post_cap_per_slot": 3,
                   "model_post_cap_total": 9},
        "input_contract": {
            "epoch": 2, "prior_outcome": "not_recovered",
            "case_source_type": "post_observation", "same_seed_none_reference": True,
            "different_corrections_max": 2,
        },
        "execution_contract": {
            "reuse_epoch0_runner": True, "result_relabelling_after_write": False,
            "probe_executor_injection": False, "complete_factory_injection": False,
            "lease_metadata_bounded_retry": True, "actual_backend_required": True,
        },
        "claims": {"correction": False, "training": False, "recovery": False,
                   "generalization": False},
    }


def verify_protocol(protocol_path: Path, *, lab: Path, anchor: str) -> dict:
    _require(isinstance(anchor, str) and HEX40.fullmatch(anchor) is not None,
             "execution anchor 형식이 다르다")
    path = Path(protocol_path).resolve()
    value = _read(path)
    component = value.get("component_post_backend_protocol")
    _require(isinstance(component, dict) and set(component) == {"path", "sha256_lf"},
             "typed post backend component 결합이 없다")
    assert isinstance(component, dict)
    component_path = ROOT / component["path"]
    _require(value == prepare_protocol(component_path), "epoch2 protocol 내용 또는 SHA가 다르다")
    POST_BACKEND.verify_protocol(component_path, Path(lab), anchor)
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {
        _relative(path): _sha(path, lf=True),
        component["path"]: component["sha256_lf"],
        **value["source_sha256_lf"],
    }
    for relative, expected_sha in bound.items():
        current = ROOT / relative
        _require(current.is_file() and _sha(current, lf=True) == expected_sha,
                 f"현재 epoch2 source/blob이 다르다: {relative}")
        committed = _git("show", f"{anchor}:{relative}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(committed).hexdigest() == expected_sha,
                 f"execution anchor epoch2 blob이 다르다: {relative}")
    return deepcopy(value)


def _checked_post_binding(envelope: dict, gate_case: dict) -> dict:
    private = envelope.get("private_metadata")
    _require(isinstance(private, dict), "epoch2 packet의 private metadata가 없다")
    assert isinstance(private, dict)
    post_ref, reference_ref = private.get("post_observation"), private.get("reference_collected")
    _require(isinstance(post_ref, dict) and isinstance(reference_ref, dict),
             "epoch2 post/reference 참조가 없다")
    assert isinstance(post_ref, dict) and isinstance(reference_ref, dict)
    post_path, reference_path = Path(post_ref.get("path", "")), Path(reference_ref.get("path", ""))
    _require(post_path.is_file() and _sha(post_path) == post_ref.get("sha256")
             and reference_path.is_file() and _sha(reference_path) == reference_ref.get("sha256"),
             "epoch2 post/reference 원문 SHA가 다르다")
    post, _post_telemetry = POST.checked_observation(post_path)
    reference, _reference_telemetry = INPUT0._checked_collected(reference_path)
    source = post["private_metadata"]["sources"]
    receipt = POST._read(Path(source["repair_receipt"]))
    packet = envelope["packet"]
    bindings = packet["input_bindings"]
    gate = post["private_metadata"]["second_epoch_gate"]
    _require(gate_case.get("case_id") == "case-07" and gate_case.get("arm") == "COMPOUND"
             and gate_case.get("seed") == receipt.get("seed") == reference["private_source_map"]["seed"]
             and gate_case.get("reference_job") == reference["private_source_map"]["job_id"],
             "원래 gate는 case-07 association으로만 결합돼야 한다")
    _require(reference["private_source_map"].get("arm") == "NONE",
             "epoch2 reference는 same-seed NONE이어야 한다")
    _require(bindings["case"]["checkpoint_sha256"] == gate["input_checkpoint_sha"]
             == receipt["new_checkpoint"]["sha256"]
             and bindings["case"]["telemetry_sha256"] == post["source_binding"]["telemetry_sha256"]
             and bindings["reference"]["checkpoint_sha256"]
             == reference["source_binding"]["checkpoint_sha256"]
             and bindings["reference"]["telemetry_sha256"]
             == reference["source_binding"]["telemetry_sha256"],
             "epoch2 packet checkpoint/telemetry 결합이 다르다")
    prior = packet.get("prior_receipt")
    _require(isinstance(prior, dict) and packet.get("epoch") == 2 and prior.get("verdict")
             == "not_recovered" and packet.get("applied_corrections") == [receipt["correction"]]
             and gate["prior_valid_not_recovered_receipt_sha"]
             == post["source_binding"]["repair_receipt_sha256"],
             "epoch2 prior not_recovered/applied correction gate가 다르다")
    return {
        "case": {
            "job_id": "post_observation", "source_type": "post_observation",
            "seed": receipt["seed"], "checkpoint_sha256": bindings["case"]["checkpoint_sha256"],
            "telemetry_sha256": bindings["case"]["telemetry_sha256"],
            "post_observation_sha256": post_ref["sha256"],
            "prior_not_recovered_receipt_sha256": post["source_binding"]["repair_receipt_sha256"],
            "applied_corrections": deepcopy(packet["applied_corrections"]),
        },
        "reference": {
            "job_id": reference["private_source_map"]["job_id"], "source_type": "collected_none",
            "seed": reference["private_source_map"]["seed"],
            "checkpoint_sha256": bindings["reference"]["checkpoint_sha256"],
            "telemetry_sha256": bindings["reference"]["telemetry_sha256"],
        },
        "second_epoch_gate": deepcopy(gate),
    }


def _epoch2_base_result(**kwargs: Any) -> dict:
    result = _ORIGINAL_BASE_RESULT(**kwargs)
    binding = kwargs.get("binding")
    _require(isinstance(binding, dict) and isinstance(binding.get("second_epoch_gate"), dict),
             "epoch2 base result의 검증 gate가 없다")
    assert isinstance(binding, dict)
    case = binding["case"]
    _require(isinstance(case, dict), "epoch2 case binding이 객체가 아니다")
    assert isinstance(case, dict)
    result.update({
        "schema": SCHEMA,
        "epoch": 2,
        "epoch2_registered": True,
        "prior_not_recovered_receipt_sha256": case["prior_not_recovered_receipt_sha256"],
        "current_checkpoint_gate": deepcopy(binding["second_epoch_gate"]),
        "applied_corrections_before": deepcopy(case["applied_corrections"]),
    })
    return result


@contextmanager
def adapter_scope() -> Iterator[None]:
    """현재 thread의 E0 입력·backend·binding·base-result만 epoch2로 바꾼다."""
    with _LOCK:
        depth = getattr(_LOCAL, "depth", 0)
        if depth:
            _require(E0.INPUT is _POST_INPUT and E0.P is POST_BACKEND
                     and E0._source_binding is _checked_post_binding
                     and E0._base_result is _epoch2_base_result,
                     "epoch2 adapter 재진입 중 전역 결합이 바뀌었다")
            _LOCAL.depth = depth + 1
            try:
                yield
            finally:
                _LOCAL.depth -= 1
            return
        _require(E0.INPUT is _ORIGINAL_INPUT and E0.P is _ORIGINAL_PROBE
                 and E0._source_binding is _ORIGINAL_SOURCE_BINDING
                 and E0._base_result is _ORIGINAL_BASE_RESULT,
                 "epoch0 runner 전역이 등록 원본이 아니다")
        E0.INPUT = _POST_INPUT
        E0.P = POST_BACKEND
        E0._source_binding = _checked_post_binding
        E0._base_result = _epoch2_base_result
        _LOCAL.depth = 1
        try:
            yield
        finally:
            _LOCAL.depth = 0
            E0._base_result = _ORIGINAL_BASE_RESULT
            E0._source_binding = _ORIGINAL_SOURCE_BINDING
            E0.P = _ORIGINAL_PROBE
            E0.INPUT = _ORIGINAL_INPUT


def run_epoch(out: Path, post_input: Path, reference_input: Path, policy: str, *,
              epoch_protocol: Path, anchor: str, **kwargs: Any) -> dict:
    """E0의 실제 기본 경로를 epoch2 typed 입력과 backend로 단회 실행한다."""
    _require("probe_executor" not in kwargs and "complete_factory" not in kwargs,
             "epoch2 actual backend/transport는 주입할 수 없다")
    lab = Path(str(kwargs.get("lab", ""))).resolve()
    protocol = verify_protocol(Path(epoch_protocol), lab=lab, anchor=anchor)
    component = (ROOT / protocol["component_post_backend_protocol"]["path"]).resolve()
    probe_protocol = Path(str(kwargs.get("probe_protocol", ""))).resolve()
    _require(probe_protocol == component, "epoch2 probe protocol이 등록 typed backend와 다르다")
    transport_root = kwargs.get("transport_root")
    slot_ids = kwargs.get("slot_ids", ())
    if transport_root is not None:
        session = T._session(Path(transport_root))
        _require(session["plan"] == RETRY.transport_plan(),
                 "epoch2는 새 case-07/3-slot/9-cap queue만 허용한다")
    if policy == "model_adaptive":
        expected_slots = tuple(f"case-07.model.{index}" for index in range(1, 4))
        _require(transport_root is not None and slot_ids == expected_slots,
                 "epoch2 model은 새 세 slot과 별도 transport가 필요하다")
    POST.checked_observation(Path(post_input))
    evidence = Path(out).resolve() / "lease-read-evidence.json"
    with RETRY.lease_read_scope(evidence):
        GPU_WAIT.B.lease_context()
    with adapter_scope():
        result = E0.run_epoch(
            Path(out), Path(post_input), Path(reference_input), policy,
            anchor=anchor, **kwargs,
        )
    _require(result.get("schema") == SCHEMA and result.get("epoch") == 2
             and result.get("epoch2_registered") is True
             and result.get("evidence_scope") == "actual_backend",
             "epoch2 runner 결과가 typed actual backend 계약과 다르다")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-protocol")
    prepare.add_argument("--post-backend-protocol", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--post-input", type=Path, required=True)
    run.add_argument("--reference-input", type=Path, required=True)
    run.add_argument("--policy", choices=E0.POLICIES, required=True)
    run.add_argument("--epoch-protocol", type=Path, required=True)
    run.add_argument("--probe-protocol", type=Path, required=True)
    run.add_argument("--anchor", required=True)
    run.add_argument("--lab", type=Path, required=True)
    run.add_argument("--gate", type=Path, required=True)
    run.add_argument("--gate-sha256", required=True)
    run.add_argument("--case-id", default="case-07")
    run.add_argument("--transport-root", type=Path)
    run.add_argument("--slot-id", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "prepare-protocol":
        _write_new(args.out, prepare_protocol(args.post_backend_protocol))
        return 0
    run_epoch(
        args.out, args.post_input, args.reference_input, args.policy,
        epoch_protocol=args.epoch_protocol, anchor=args.anchor, lab=args.lab,
        probe_protocol=args.probe_protocol, run_id=f"post_{args.policy}", gate_path=args.gate,
        gate_sha256=args.gate_sha256, case_id=args.case_id,
        transport_root=args.transport_root, slot_ids=tuple(args.slot_id),
        max_decisions=3, execution_budget_s=1200,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
