"""완료된 lineage source 한 건을 공통 selection view 입력으로 봉인한다.

CPU 전용 어댑터다. 학습, 모델 호출, probe 실행, 승인 또는 실행 원장 쓰기를 하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import lineage_source_batch_v1 as B
import lineage_source_training_v1 as CHILD
import selection_comparison_view_v1 as V

SCHEMA = "lineage_selection_collected_v1"
PACKET_SCHEMA = "lineage_selection_packet_v1"
SNAPSHOTS_SCHEMA = "lineage_selection_snapshots_v1"
POLICIES = ("registry_next", "fixed_eligible", "model_adaptive")
EXTRACTOR = ROOT / "bench" / "extract_tb.py"
EXTRACT_TIMEOUT_S = 120


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def _new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _new_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _event(log_dir: Path) -> Path:
    events = sorted(path for path in log_dir.glob("events.out.tfevents.*") if path.is_file())
    _require(len(events) == 1, "TensorBoard event는 정확히 하나여야 한다")
    return events[0]


def _validate_accepted(stage: Path, manifest: dict, evidence: dict) -> tuple[dict, dict]:
    accepted = _json(stage / "accepted.json")
    evidence_path = stage / "training" / "training_evidence.json"
    checkpoint = evidence["checkpoint"]
    _require(
        accepted.get("job_id") == manifest["job_id"]
        and accepted.get("purpose") == manifest["purpose"]
        and accepted.get("arm") == manifest["arm"]
        and accepted.get("seed") == manifest["seed"]
        and accepted.get("checkpoint") == checkpoint
        and accepted.get("evidence_sha256") == _sha(evidence_path),
        "accepted 결과가 manifest·evidence·checkpoint와 다르다",
    )
    return accepted, checkpoint


def collect_source(stage: Path, out: Path, isaac_python: Path) -> dict:
    """완료된 단일 source job을 검증하고 TensorBoard 원문과 요약을 봉인한다."""
    stage, out, isaac_python = Path(stage).resolve(), Path(out).resolve(), Path(isaac_python).resolve()
    _require(stage.is_dir(), "source stage가 없다")
    _require(isaac_python.is_file(), "Isaac Python 실행 파일이 없다")
    manifest = CHILD.validate_manifest(_json(stage / "manifest.json"))
    manifest_fingerprint = CHILD.manifest_fingerprint(manifest)
    evidence = B.completed(stage, manifest, _json(stage / "owned-result.json"))
    _accepted, checkpoint = _validate_accepted(stage, manifest, evidence)
    checkpoint_path = Path(checkpoint["path"]).resolve()
    _require(_sha(checkpoint_path) == checkpoint["sha256"], "checkpoint bytes SHA가 다르다")
    evidence_path = stage / "training" / "training_evidence.json"
    accepted_path = stage / "accepted.json"
    event = _event(Path(evidence["log_dir"]).resolve())
    event_sha_before = _sha(event)

    out.mkdir(parents=True, exist_ok=False)
    event_copy = out / "source.events.out.tfevents.raw"
    shutil.copyfile(event, event_copy)
    _require(_sha(event_copy) == event_sha_before, "복사한 TensorBoard event SHA가 다르다")
    telemetry = out / "telemetry.json"
    argv = [str(isaac_python), str(EXTRACTOR), str(Path(evidence["log_dir"]).resolve()), str(telemetry)]
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    receipt = {
        "schema": "lineage_tensorboard_extract_receipt_v1",
        "argv": argv,
        "cpu_only": True,
        "timeout_s": EXTRACT_TIMEOUT_S,
        "event_sha256_before": event_sha_before,
        "extractor_sha256_lf": _sha(EXTRACTOR, lf=True),
        "collector_sha256_lf": _sha(Path(__file__), lf=True),
    }
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            check=False,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=EXTRACT_TIMEOUT_S,
        )
        stdout, stderr = completed.stdout, completed.stderr
        receipt.update(returncode=completed.returncode, error_type=None)
    except Exception as exc:
        stdout = _subprocess_text(getattr(exc, "stdout", None))
        stderr = _subprocess_text(getattr(exc, "stderr", None))
        receipt.update(returncode=None, error_type=type(exc).__name__)
        receipt["event_sha256_after"] = _sha(event) if event.is_file() else None
        _new_text(out / "extract.stdout.txt", stdout)
        _new_text(out / "extract.stderr.txt", stderr)
        _new_json(out / "extract.receipt.json", receipt)
        raise
    _new_text(out / "extract.stdout.txt", stdout)
    _new_text(out / "extract.stderr.txt", stderr)
    event_sha_after = _sha(event)
    receipt["event_sha256_after"] = event_sha_after
    if completed.returncode != 0 or not telemetry.is_file():
        _new_json(out / "extract.receipt.json", receipt)
        raise RuntimeError("TensorBoard 추출 subprocess가 성공하지 않았다")
    _require(event_sha_after == event_sha_before, "추출 도중 source event가 바뀌었다")
    telemetry_value = _json(telemetry)
    _require(isinstance(telemetry_value.get("summary"), dict), "telemetry summary가 없다")
    telemetry_sha = _sha(telemetry)
    receipt["telemetry_sha256"] = telemetry_sha
    _new_json(out / "extract.receipt.json", receipt)

    collected = {
        "schema": SCHEMA,
        "source_binding": {
            "manifest_fingerprint": manifest_fingerprint,
            "manifest_sha256": _sha(stage / "manifest.json"),
            "owned_result_sha256": _sha(stage / "owned-result.json"),
            "accepted_sha256": _sha(accepted_path),
            "evidence_sha256": _sha(evidence_path),
            "checkpoint_sha256": checkpoint["sha256"],
            "event_sha256": event_sha_before,
            "telemetry_sha256": telemetry_sha,
            "extractor_sha256_lf": receipt["extractor_sha256_lf"],
            "collector_sha256_lf": receipt["collector_sha256_lf"],
        },
        "private_source_map": {
            "stage": str(stage),
            "job_id": manifest["job_id"],
            "arm": manifest["arm"],
            "seed": manifest["seed"],
            "run_name": manifest["run_name"],
            "manifest": str((stage / "manifest.json").resolve()),
            "owned_result": str((stage / "owned-result.json").resolve()),
            "accepted": str(accepted_path.resolve()),
            "training_evidence": str(evidence_path.resolve()),
            "checkpoint": str(checkpoint_path),
            "source_event": str(event),
            "event_copy": str(event_copy),
            "telemetry": str(telemetry),
            "extract_receipt": str((out / "extract.receipt.json").resolve()),
        },
        "scope": {
            "activity": "completed_source_cpu_materialization_only",
            "batch_requirement": "a_completed_single_job_is_sufficient",
            "excluded": ["training", "model_selection", "probe_execution", "approval"],
            "none_or_failure_absence_intervention_owner": "next_gate_controller",
        },
    }
    _new_json(out / "collected.json", collected)
    return deepcopy(collected)


def _checked_collected(path: Path) -> tuple[dict, dict]:
    path = Path(path).resolve()
    value = _json(path)
    _require(value.get("schema") == SCHEMA, "collected schema가 다르다")
    binding, source = value.get("source_binding"), value.get("private_source_map")
    _require(isinstance(binding, dict) and isinstance(source, dict), "collected 결합 또는 source map이 없다")
    checks = {
        "manifest_sha256": source["manifest"],
        "owned_result_sha256": source["owned_result"],
        "accepted_sha256": source["accepted"],
        "evidence_sha256": source["training_evidence"],
        "checkpoint_sha256": source["checkpoint"],
        "event_sha256": source["source_event"],
        "telemetry_sha256": source["telemetry"],
    }
    for field, raw_path in checks.items():
        _require(_sha(Path(raw_path)) == binding.get(field), f"collected {field} 원문이 바뀌었다")
    _require(_sha(Path(source["event_copy"])) == binding["event_sha256"], "봉인 event 복사본이 바뀌었다")
    _require(_sha(EXTRACTOR, lf=True) == binding.get("extractor_sha256_lf"), "extractor source가 바뀌었다")
    _require(_sha(Path(__file__), lf=True) == binding.get("collector_sha256_lf"), "collector source가 바뀌었다")
    receipt = _json(Path(source["extract_receipt"]))
    _require(
        receipt.get("cpu_only") is True
        and receipt.get("timeout_s") == EXTRACT_TIMEOUT_S
        and receipt.get("returncode") == 0
        and receipt.get("event_sha256_before") == binding["event_sha256"]
        and receipt.get("event_sha256_after") == binding["event_sha256"]
        and receipt.get("telemetry_sha256") == binding["telemetry_sha256"]
        and receipt.get("extractor_sha256_lf") == binding["extractor_sha256_lf"]
        and receipt.get("collector_sha256_lf") == binding["collector_sha256_lf"],
        "extract receipt 결합이 다르다",
    )
    manifest = CHILD.validate_manifest(_json(Path(source["manifest"])))
    _require(CHILD.manifest_fingerprint(manifest) == binding.get("manifest_fingerprint"),
             "manifest fingerprint가 다르다")
    return value, _json(Path(source["telemetry"]))


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _telemetry_projection(role: str, telemetry: dict) -> tuple[dict, dict]:
    summary = telemetry.get("summary")
    _require(isinstance(summary, dict), "telemetry summary가 객체가 아니다")
    facts, coverage = {}, {}
    for field in V.TELEMETRY_FIELDS:
        row = summary.get(field)
        fact_id = f"telemetry:{role}:{field}"
        if row is None:
            facts[fact_id] = None
            coverage[field] = {"n": 0, "observed_scalar_index_max": None}
            continue
        _require(isinstance(row, dict), f"등록 telemetry {field} summary 형식이 잘못됐다")
        mean = row.get("mean_last_20pct")
        n = row.get("n")
        _require(_finite_number(mean), f"등록 telemetry {field} 평균이 유한 수치가 아니다")
        _require(type(n) is int and n > 0, f"등록 telemetry {field} 표본 수가 잘못됐다")
        facts[fact_id] = mean
        coverage[field] = {"n": n, "observed_scalar_index_max": n - 1}
    return facts, coverage


def make_selection_packet(case_input: Path, reference_input: Path, *, budget_left: int = 7,
                          execution_budget_s: float = 1200) -> dict:
    """두 collected 입력에서 공통 selection packet과 비공개 provenance를 만든다."""
    case, case_telemetry = _checked_collected(Path(case_input))
    reference, reference_telemetry = _checked_collected(Path(reference_input))
    case_facts, case_coverage = _telemetry_projection("case", case_telemetry)
    reference_facts, reference_coverage = _telemetry_projection("reference", reference_telemetry)
    facts = {**case_facts, **reference_facts}
    packet = V.S.build_packet(
        epoch=1,
        applied_corrections=[],
        prior_receipt=None,
        facts=facts,
        input_bindings={
            "case": {"checkpoint_sha256": case["source_binding"]["checkpoint_sha256"],
                     "telemetry_sha256": case["source_binding"]["telemetry_sha256"]},
            "reference": {"checkpoint_sha256": reference["source_binding"]["checkpoint_sha256"],
                          "telemetry_sha256": reference["source_binding"]["telemetry_sha256"]},
        },
        budget_left=budget_left,
        execution_budget_s=execution_budget_s,
    )
    return {
        "schema": PACKET_SCHEMA,
        "packet": packet,
        "private_metadata": {
            "case": {"collected": str(Path(case_input).resolve()), "coverage": case_coverage,
                     "source": deepcopy(case["private_source_map"])},
            "reference": {"collected": str(Path(reference_input).resolve()), "coverage": reference_coverage,
                          "source": deepcopy(reference["private_source_map"])},
            "coverage_semantics": "extractor scalar count and zero-based observed index range; not a 300-step claim",
            "unregistered_telemetry_projected": False,
            "config_action_scale_verified": False,
        },
        "scope": {
            "activity": "cpu_packet_materialization_only",
            "excluded": ["model_callback", "probe_execution", "approval"],
            "none_or_failure_absence_intervention_owner": "next_gate_controller",
        },
    }


def build_snapshots(packet: dict, run_id: str, out: Path) -> dict:
    """같은 공개 view를 가진 세 정책 snapshot을 CPU에서 생성한다."""
    envelope = deepcopy(packet)
    raw_packet = envelope.get("packet") if envelope.get("schema") == PACKET_SCHEMA else envelope
    V.S.validate_packet(raw_packet)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol_sha = V.file_digest(V.PROTOCOL)
    snapshots = {}
    for policy in POLICIES:
        snapshot = V.build_snapshot(raw_packet, run_id, policy, protocol_sha, out / f"{policy}.ledger.jsonl")
        _new_json(out / f"{policy}.snapshot.json", snapshot)
        snapshots[policy] = snapshot
    _require(len({item["view_digest"] for item in snapshots.values()}) == 1,
             "세 정책의 공통 selection view가 다르다")
    result = {
        "schema": SNAPSHOTS_SCHEMA,
        "run_id": run_id,
        "protocol_sha256": protocol_sha,
        "view_digest": next(iter(snapshots.values()))["view_digest"],
        "snapshots": {policy: str((out / f"{policy}.snapshot.json").resolve()) for policy in POLICIES},
        "scope": {"activity": "cpu_snapshot_materialization_only",
                  "excluded": ["model_callback", "probe_execution", "approval"]},
    }
    _new_json(out / "snapshots.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--stage", type=Path, required=True)
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--isaac-python", type=Path, required=True)
    packet = commands.add_parser("packet")
    packet.add_argument("--case-input", type=Path, required=True)
    packet.add_argument("--reference-input", type=Path, required=True)
    packet.add_argument("--out", type=Path, required=True)
    snapshots = commands.add_parser("snapshots")
    snapshots.add_argument("--packet", type=Path, required=True)
    snapshots.add_argument("--run-id", required=True)
    snapshots.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect_source(args.stage, args.out, args.isaac_python)
    elif args.command == "packet":
        _new_json(args.out, make_selection_packet(args.case_input, args.reference_input))
    else:
        build_snapshots(_json(args.packet), args.run_id, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
