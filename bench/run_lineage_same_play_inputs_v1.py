# pyright: reportMissingImports=false
"""새 원천 20개의 same-Play 녹화와 열 쌍의 모델 입력을 순차 봉인한다."""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_gpu_admission_v1 as GPU
import lineage_same_play_capture_v1 as CAPTURE
import lineage_same_play_pair_inputs_v1 as PAIR
import lineage_source_batch_v1 as LEASE
import lineage_source_inputs_v2 as INPUT

B = CAPTURE.V1
SCHEMA = "lineage_same_play_input_batch_protocol_v1"
SOURCE_FILES = tuple(sorted({
    *CAPTURE.source_files(), *PAIR.SOURCE_FILES, *INPUT.BATCH.source_files(),
    "bench/run_lineage_same_play_inputs_v1.py", "tests/test_run_lineage_same_play_inputs_v1.py",
    "evals/lineage_gpu_admission_v1.py", "evals/lineage_source_batch_v1.py",
}))


def prepare(index: Path, lab: Path, isaac_python: Path, *, source_anchor: str,
            tag: str, config_out: Path) -> dict:
    """실제 입력이 확보된 뒤 공개 protocol 31개와 private config를 생성한다."""
    if not re.fullmatch(r"[a-z0-9_]{1,50}", tag):
        raise ValueError("registration tag")
    public, _metadata = INPUT.checked(index)
    if set(public["sources"]) != {f"source-{i:02d}" for i in range(1, 21)}:
        raise ValueError("source20 anonymous catalog")
    if not Path(isaac_python).is_file() or config_out.exists():
        raise ValueError("Isaac launcher or fresh private config")
    config = {"schema": "lineage_same_play_input_batch_config_v1", "index": B._ref(index),
              "lab": str(Path(lab).resolve()), "isaac_python": str(Path(isaac_python).resolve())}
    protocol_dir = ROOT / "bench/protocols"
    capture_protocols, pair_protocols = {}, {}
    for source_id in sorted(public["sources"]):
        path = protocol_dir / f"{tag}_{source_id.replace('-', '_')}.json"
        CAPTURE.prepare_protocol(index, source_id, lab, path, source_anchor=source_anchor)
        capture_protocols[source_id] = {"path": path.relative_to(ROOT).as_posix(),
                                        "sha256_lf": B._sha(path, lf=True)}
    for case_id in public["case_ids"]:
        path = protocol_dir / f"{tag}_{case_id.replace('-', '_')}_pair.json"
        PAIR.prepare_protocol(index, case_id, path, source_anchor=source_anchor)
        pair_protocols[case_id] = {"path": path.relative_to(ROOT).as_posix(),
                                  "sha256_lf": B._sha(path, lf=True)}
    B._write_new(config_out, config)
    value = {"schema": SCHEMA, "source_anchor": source_anchor,
             "sources_lf": {name: B._sha(ROOT / name, lf=True) for name in SOURCE_FILES},
             "source_index_sha256": B._sha(index), "execution_config_sha256": B._sha(config_out),
             "capture_protocols": capture_protocols, "pair_protocols": pair_protocols,
             "smoke_source_id": "source-01", "source_order": sorted(public["sources"]),
             "case_order": list(public["case_ids"]), "sdk_cap": 19, "api_cap": 0,
             "maximum_full_captures": 18, "wall_cap_s": 45000,
             "automatic_retries": 0, "replacement_captures": False,
             "scope": "same_play_input_materialization_not_diagnosis_or_recovery"}
    target = protocol_dir / f"{tag}_batch.json"
    B._write_new(target, value)
    return {"protocol": str(target), "private_config": str(config_out),
            "public_protocols": [row["path"] for row in [*capture_protocols.values(), *pair_protocols.values()]]
                                 + [target.relative_to(ROOT).as_posix()]}


def preflight(protocol_path: Path, anchor: str, config_path: Path) -> tuple[dict, dict, dict]:
    value, config = B._read(protocol_path), B._read(config_path)
    if (value.get("schema") != SCHEMA or value.get("sdk_cap") != 19
            or value.get("api_cap") != 0 or value.get("maximum_full_captures") != 18
            or value.get("automatic_retries") != 0 or value.get("replacement_captures") is not False
            or value.get("wall_cap_s") != 45000
            or value.get("execution_config_sha256") != B._sha(config_path)):
        raise ValueError("registered capture batch/config contract")
    if not set(SOURCE_FILES) <= set(value["sources_lf"]):
        raise ValueError("capture batch source closure")
    CAPTURE._verify_source_anchor(ROOT, value["source_anchor"], value["sources_lf"])
    B._verify_protocol_blob(protocol_path, ROOT, value["source_anchor"], anchor)
    if config.get("schema") != "lineage_same_play_input_batch_config_v1":
        raise ValueError("private capture config schema")
    index = Path(config["index"]["path"])
    if B._ref(index) != config["index"] or B._sha(index) != value["source_index_sha256"]:
        raise ValueError("source index changed")
    public, _metadata = INPUT.checked(index)
    if (value["source_order"] != sorted(public["sources"])
            or value["case_order"] != public["case_ids"]
            or set(value["capture_protocols"]) != set(public["sources"])
            or set(value["pair_protocols"]) != set(public["case_ids"])):
        raise ValueError("source20/case10 registration changed")
    captures = {}
    for source_id, item in value["capture_protocols"].items():
        path = ROOT / item["path"]
        if B._sha(path, lf=True) != item["sha256_lf"]:
            raise ValueError("capture protocol changed")
        captures[source_id] = CAPTURE.validate_protocol(path, index, Path(config["lab"]), execution_anchor=anchor)
    for case_id, item in value["pair_protocols"].items():
        path = ROOT / item["path"]
        if B._sha(path, lf=True) != item["sha256_lf"]:
            raise ValueError("pair input protocol changed")
        PAIR.verify_protocol(path, index, case_id, anchor)
    if sum(row["status"] != "hold_unsupported" for row in captures.values()) != 18:
        raise ValueError("expected eighteen full captures and two explicit unsupported holds")
    if captures[value["smoke_source_id"]]["playback"]["kind"] != "none":
        raise ValueError("capture smoke requires no-injection source")
    return value, config, captures


def execute(protocol_path: Path, anchor: str, config_path: Path, out: Path) -> dict:
    protocol, config, captures = preflight(protocol_path, anchor, config_path)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    initial = LEASE.lease_context()
    if initial["owner"] != LEASE.OWNER:
        raise ValueError("capture batch lease owner")
    B._write_new(out / "lease-initial.json", initial)
    started, sdk = time.monotonic(), 0
    index, lab, python = Path(config["index"]["path"]), Path(config["lab"]), Path(config["isaac_python"])
    artifacts = {}

    def boundary():
        lease = LEASE.lease_context()
        if (lease["leaseId"] != initial["leaseId"] or lease["owner"] != initial["owner"]
                or time.monotonic() - started > protocol["wall_cap_s"]
                or B._sha(config_path) != protocol["execution_config_sha256"]):
            raise ValueError("capture batch source/time/lease boundary changed")

    def record(source_id, mode):
        nonlocal sdk
        boundary()
        if sdk >= protocol["sdk_cap"]:
            raise ValueError("capture SDK cap")
        sdk += 1
        stage = out / ("smoke" if mode == "smoke" else source_id)
        B._write_new(out / "sdk-intents" / f"{sdk:02d}.json", {"source_id": source_id, "mode": mode})
        GPU.snapshot(out / "gpu-admission" / f"{sdk:02d}")
        return CAPTURE.execute(ROOT / protocol["capture_protocols"][source_id]["path"], index,
                               anchor, lab, stage, python, mode=mode)

    try:
        record(protocol["smoke_source_id"], "smoke")
        for source_id in protocol["source_order"]:
            entry = ROOT / protocol["capture_protocols"][source_id]["path"]
            supported = captures[source_id]["status"] != "hold_unsupported"
            if supported:
                record(source_id, "full")
            artifact = CAPTURE.export_artifact(entry, index, lab, out / source_id if supported else None,
                                               execution_anchor=anchor)
            B._write_new(out / "artifacts" / f"{source_id}.json", artifact)
            artifacts[source_id] = artifact
        _public, metadata = INPUT.checked(index)
        source_ids = {job: row["public_id"] for job, row in metadata["sources"].items()}
        pairs = {}
        for case in metadata["cases"]:
            boundary()
            case_id = case["case_id"]
            ids = {"case": source_ids[case["source_job"]], "reference": source_ids[case["reference_job"]]}
            pair_out = out / "pairs" / case_id
            PAIR.prepare_pair(index, case_id, artifacts[ids["case"]], artifacts[ids["reference"]], pair_out,
                              protocol_path=ROOT / protocol["pair_protocols"][case_id]["path"],
                              execution_anchor=anchor, lab=lab,
                              capture_protocols={role: ROOT / protocol["capture_protocols"][source_id]["path"]
                                                 for role, source_id in ids.items()})
            PAIR.load_pair(pair_out / "result.json", index)
            pairs[case_id] = B._ref(pair_out / "result.json")
        result = {"schema": "lineage_same_play_input_batch_result_v1", "status": "complete",
                  "sdk_intents": sdk, "api": 0, "source_artifacts": 20, "pairs": pairs,
                  "parent_wall_s": time.monotonic() - started, "diagnosis": False, "recovery": False}
        B._write_new(out / "result.json", result)
        return result
    except BaseException as error:
        B._write_new(out / "failure.json", {"status": "invalid", "error_type": type(error).__name__,
                                           "message": str(error), "sdk_intents": sdk, "automatic_retry": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        preflight(args.protocol, args.anchor, args.config)
        B._write_new(args.out, {"status": "complete", "api": 0, "gpu": 0})
    else:
        execute(args.protocol, args.anchor, args.config, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
