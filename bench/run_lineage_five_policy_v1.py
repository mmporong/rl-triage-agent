# pyright: reportMissingImports=false
"""등록된 새 10사례의 다섯 정책 비교를 공용 GPU lease 안에서 실행한다."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import lineage_five_policy_driver_v1 as DRIVER
import lineage_five_policy_nim_transport_v1 as NIM
import lineage_gpu_admission_v1 as GPU
import lineage_same_play_pair_inputs_v1 as PAIR
import lineage_source_batch_v1 as LEASE
import lineage_source_probe_inputs_v1 as HOST
import run_local_pair_four_arm_v1 as PREVIOUS

B = DRIVER.BASE
SOURCE_FILES = tuple(
    sorted(
        {
            *DRIVER.SOURCE_FILES,
            *HOST.SOURCE_FILES,
            *PREVIOUS.SOURCE_FILES,
            *PAIR.SOURCE_FILES,
            "bench/run_lineage_five_policy_v1.py",
            "tests/test_run_lineage_five_policy_v1.py",
        }
    )
)


def worker_command(root: Path, call: Path, protocol: Path, anchor: str) -> list[str]:
    args = [
        "-B",
        PREVIOUS.mounted(ROOT / "evals/lineage_five_policy_nim_transport_v1.py"),
        "--worker",
        "--root",
        PREVIOUS.mounted(root),
        "--call-dir",
        PREVIOUS.mounted(call),
        "--protocol",
        PREVIOUS.mounted(protocol),
        "--anchor",
        anchor,
    ]
    shell = 'set -euo pipefail; . "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" '
    return [
        "wsl.exe",
        "--distribution",
        "Ubuntu",
        "--exec",
        "bash",
        "-lc",
        shell + shlex.join(args),
    ]


def _worker_instance():
    spec = importlib.util.spec_from_file_location(
        "_lineage_five_policy_parent_worker_v1", PREVIOUS.__file__
    )
    if spec is None or spec.loader is None:
        raise ImportError("one-shot worker parent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__dict__["worker_command"] = worker_command
    return module


_WORKER = _worker_instance()


def _registered_ref_path(value: object, label: str) -> Path:
    B.require(
        isinstance(value, Mapping)
        and isinstance(value.get("path"), str)
        and isinstance(value.get("sha256"), str),
        label + " ref",
    )
    ref = cast(Mapping[str, Any], value)
    return Path(cast(str, ref["path"])).resolve()


def _nested_ref_paths(value: object) -> list[Path]:
    found: list[Path] = []
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            found.append(Path(value["path"]).resolve())
        for child in value.values():
            found.extend(_nested_ref_paths(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_nested_ref_paths(child))
    return found


def _base_case_roots(config: Mapping, row: Mapping) -> tuple[Path, ...]:
    index = _registered_ref_path(config.get("source_index"), "source index")
    refs = [row.get("source"), *(row.get("probe_inputs", {}).values())]
    if row.get("pair_input") is not None:
        refs.append(row.get("pair_input"))
    return DRIVER._input_roots((index, index.parent / DRIVER.INPUT.METADATA,
                                *(_registered_ref_path(ref, "case input") for ref in refs)))


def _registered_case_roots(config: Mapping, row: Mapping) -> tuple[Path, ...]:
    roots = list(_base_case_roots(config, row))
    refs = [row.get("source"), *(row.get("probe_inputs", {}).values())]
    if row.get("pair_input") is not None:
        refs.append(row.get("pair_input"))
    roots.extend(_registered_ref_path(ref, "case input") for ref in refs)
    index = roots[0]
    if index.is_file():
        B.require(B.sha(index) == config["source_index"]["sha256"],
                  "registered source index bytes changed before resolving data refs")
        public = B.read(index)
        metadata_ref = public.get("metadata_ref")
        B.require(
            isinstance(metadata_ref, Mapping)
            and isinstance(metadata_ref.get("file"), str)
            and metadata_ref["file"] == DRIVER.INPUT.METADATA,
            "source metadata ref",
        )
        metadata_path = index.parent / metadata_ref["file"]
        roots.append(metadata_path.resolve())
        if metadata_path.is_file():
            B.require(B.sha(metadata_path) == metadata_ref.get("sha256"),
                      "registered metadata bytes changed before resolving data refs")
            metadata = B.read(metadata_path)
            roots.extend(_nested_ref_paths(metadata))
            sources = metadata.get("sources")
            B.require(isinstance(sources, Mapping), "source metadata sources")
            for private in sources.values():
                if isinstance(private, Mapping):
                    material = private.get("material")
                    if isinstance(material, Mapping) and isinstance(
                        material.get("stage"), str
                    ):
                        stage = Path(material["stage"]).resolve()
                        roots.append(stage)
                        roots.extend(
                            stage / relative
                            for relative in (
                                "manifest.json",
                                "owned-result.json",
                                "accepted.json",
                                "training/training_evidence.json",
                                "training/records.jsonl",
                                "fixed-result.json",
                            )
                        )
    for ref in refs:
        path = _registered_ref_path(ref, "case input")
        if path.is_file() and path.suffix == ".json":
            B.require(B.sha(path) == ref["sha256"],
                      "registered case input bytes changed before resolving data refs")
            roots.extend(_nested_ref_paths(B.read(path)))
    return DRIVER._input_roots(tuple(dict.fromkeys(path.resolve() for path in roots)))


def _missing_source_result(
    case_id: str,
    arm: str,
    source_digest: str,
    source_index: object,
    gate: Mapping,
    output: Path,
) -> dict:
    result = {
        "schema": DRIVER.RESULT_SCHEMA,
        "arm": arm,
        "case_id": case_id,
        "run_id": case_id.replace("-", "_") + "_" + arm,
        "scope": "actual_backend",
        "selector_policy": arm,
        "host_storage_policy": None,
        "host_storage_alias": None,
        "source_digest": source_digest,
        "source_index": deepcopy(source_index),
        "checkpoint_bindings": None,
        "steps": [],
        "model_calls": 0,
        "native_normal_intervention_proposal": None,
        "hold_reason": "registered_input_unavailable",
        "input_unavailable": 1,
        "input_guard": deepcopy(gate),
        "semantic_invalid": 0,
        "infra_unknown": 0,
        "provider_unknown": 0,
        "frame_fact_citations": 0,
        "repair": False,
        "recovery": False,
        "heldout": False,
        "qualification": None,
        "end_state": "HOLD",
        "confirmed_subtype": None,
        "final_view_digest": None,
        "remaining_candidates": None,
        "scoring": None,
        "parent_wall_s": 0.0,
    }
    B.write(output / "input-guard.json", gate)
    B.write(output / "result.json", result)
    return result


def preflight(
    protocol_path: Path, anchor: str, config_path: Path
) -> tuple[dict, dict, dict]:
    protocol = DRIVER.verify_protocol(protocol_path, anchor)
    B.require(
        set(SOURCE_FILES) <= set(protocol["sources_lf"]),
        "five-policy runner source closure",
    )
    B.require(
        B.sha(config_path) == protocol["execution_config_sha256"],
        "private execution config changed",
    )
    config = B.read(config_path)
    B.require(
        set(config) == {"schema", "lab", "source_index", "cases"}
        and config["schema"] == "lineage_five_policy_execution_config_v1"
        and set(config["cases"]) == set(protocol["case_order"]),
        "ten case config contract",
    )
    lab = Path(config["lab"]).resolve(strict=True)
    tools = protocol["tool_protocols"]
    for entry in tools.values():
        B.require(
            B.sha(ROOT / entry["path"], lf=True) == entry["sha256_lf"],
            "tool protocol changed",
        )
    HOST.verify_protocol(ROOT / tools["paired_probe"]["path"], lab, anchor)
    DRIVER.HOST.RV.verify_protocol(ROOT / tools["view"]["path"], anchor)
    B.require(
        protocol["paired_cap"] == 150
        and protocol["sdk_role_cap"] == 300
        and protocol["wall_cap_s"] == 144000,
        "registered diagnosis execution upper bounds",
    )
    sources = {}
    for case_id in protocol["case_order"]:
        row = config["cases"][case_id]
        base_roots = _base_case_roots(config, row)
        resolved = DRIVER.GUARD.invoke_guarded("input", _registered_case_roots, config, row,
                                               registered_input_roots=base_roots)
        if isinstance(resolved, dict) and resolved.get("schema") == DRIVER.GUARD.SCHEMA:
            sources[case_id] = {"source": None, "inputs": {}, "input_hold": resolved,
                                "registered_input_roots": base_roots}
            continue
        roots = cast(tuple[Path, ...], resolved)
        raw_source: dict | None = None

        def load_case(_row=row, _case_id=case_id) -> dict:
            nonlocal raw_source
            index = B._bound_file(config["source_index"])
            source_path = B._bound_file(_row["source"])
            raw_source = B.read(source_path)
            B.require(
                B.digest(raw_source) == protocol["source_digests"][_case_id],
                "registered raw source digest changed",
            )
            source = DRIVER.checked_source(raw_source)
            B.require(
                source["case_id"] == _case_id and source["index"] == B.ref(index),
                "registered source identity changed",
            )
            pair_ref = _row.get("pair_input")
            if pair_ref is None:
                B.require(
                    source["providers"] is None and source["visual_evidence"] is None,
                    "model image/numeric Play inputs require bound capture provenance",
                )
            else:
                pair_source = PAIR.load_pair(B._bound_file(pair_ref), index)
                B.require(
                    pair_source == source,
                    "source and raw same-Play pair provenance differ",
                )
            inputs = {}
            for role in ("case", "reference"):
                path = B._bound_file(_row["probe_inputs"][role])
                envelope, _metadata = HOST.checked(path)
                binding = source["packet"]["input_bindings"][role]
                B.require(
                    all(
                        envelope["binding"][key] == binding[key]
                        for key in ("checkpoint_sha256", "telemetry_sha256")
                    ),
                    "paired role/checkpoint mismatch",
                )
                B.require(
                    envelope["index"] == B.ref(index),
                    "probe input belongs to another collected index",
                )
                inputs[role] = path
            return {"source": source, "inputs": inputs}

        loaded = DRIVER.GUARD.invoke_guarded(
            "input", load_case, registered_input_roots=roots
        )
        if isinstance(loaded, dict) and loaded.get("schema") == DRIVER.GUARD.SCHEMA:
            sources[case_id] = {
                "source": raw_source,
                "inputs": {},
                "input_hold": loaded,
                "registered_input_roots": roots,
            }
        else:
            sources[case_id] = {**loaded, "registered_input_roots": roots}
    return protocol, config, sources


def execute(protocol_path: Path, anchor: str, config_path: Path, output: Path) -> dict:
    protocol, config, sources = preflight(protocol_path, anchor, config_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    first_lease = LEASE.lease_context()
    B.require(first_lease["owner"] == LEASE.OWNER, "common lease owner")
    B.write(output / "lease-initial.json", first_lease)
    started, paired = time.monotonic(), 0
    results = []
    NIM.initialize(
        output / "nim", protocol_path=protocol_path, execution_anchor=anchor, api_cap=72
    )

    def context(_request=None):
        B.require(
            time.monotonic() - started < protocol["wall_cap_s"], "diagnosis wall cap"
        )
        now = LEASE.lease_context()
        B.require(
            now["leaseId"] == first_lease["leaseId"]
            and now["owner"] == first_lease["owner"],
            "GPU lease identity changed",
        )
        B.require(
            B.sha(config_path) == protocol["execution_config_sha256"],
            "config changed during batch",
        )
        for name, expected in protocol["sources_lf"].items():
            B.require(
                B.sha(ROOT / name, lf=True) == expected,
                "registered source changed: " + name,
            )
        return {"approved": True}

    def busy(operation_id, _intent):
        context()
        stage = output / "gpu-observations" / operation_id
        try:
            GPU.snapshot(stage)
        except RuntimeError:
            receipt = B.read(stage / "gpu_before.json")
            B.require(
                receipt.get("bounded_resample", {}).get("accepted") is False,
                "GPU exception without bound rejection evidence",
            )
            return {
                "status": "busy",
                "observation_id": B.sha(stage / "gpu_before.json"),
            }
        return {
            "status": "available",
            "observation_id": B.sha(stage / "gpu_before.json"),
        }

    for case_id in protocol["case_order"]:
        item = sources[case_id]
        source = item["source"]
        for arm in protocol["arm_order"][case_id]:
            context()

            if source is None:
                result = _missing_source_result(
                    case_id,
                    arm,
                    protocol["source_digests"][case_id],
                    config["source_index"],
                    item["input_hold"],
                    output / case_id / arm,
                )
                results.append(
                    {
                        "case_id": case_id,
                        "arm": arm,
                        "result": B.ref(output / case_id / arm / "result.json"),
                        "end_state": result["end_state"],
                        "scoring": None,
                    }
                )
                continue

            def dispatch(
                snapshot,
                choice,
                *,
                out,
                _source=source,
                _item=item,
                _case=case_id,
                _arm=arm,
            ):
                nonlocal paired
                context()
                B.require(paired < protocol["paired_cap"], "paired probe global cap")
                paired += 1  # intent를 차감하며 실패해도 반환하지 않는다.
                B.write(
                    output / "paired-intents" / f"{paired:04d}.json",
                    {
                        "case_id": _case,
                        "arm": _arm,
                        "source_digest": B.digest(_source),
                        "choice_digest": B.digest(choice),
                        "snapshot_digest": B.digest(snapshot),
                        "sdk_roles_reserved": 2,
                        "automatic_retry": False,
                    },
                )
                return HOST.execute_pair(
                    snapshot,
                    choice,
                    backend_packet=_source["backend_packet"],
                    case_input=_item["inputs"]["case"],
                    reference_input=_item["inputs"]["reference"],
                    probe_protocol=ROOT
                    / protocol["tool_protocols"]["paired_probe"]["path"],
                    view_protocol=ROOT / protocol["tool_protocols"]["view"]["path"],
                    anchor=anchor,
                    lab=Path(config["lab"]),
                    out=out,
                    verify_context=context,
                )

            def complete(request):
                context()
                return NIM.complete(
                    output / "nim",
                    request,
                    send=lambda body, call: _WORKER._worker_send(
                        output / "nim", protocol_path, anchor, body, call
                    ),
                )

            result = DRIVER.run_arm(
                source,
                arm=arm,
                run_id=case_id.replace("-", "_") + "_" + arm,
                output=output / case_id / arm,
                protocol_path=protocol_path,
                execution_anchor=anchor,
                complete=complete if arm in DRIVER.MODEL_ARMS else None,
                execute_probe=dispatch,
                verify_probe=HOST.verify_pair,
                verify_call=NIM.verify_call,
                busy_reader=busy,
                registered_input_roots=item["registered_input_roots"],
            )
            results.append(
                {
                    "case_id": case_id,
                    "arm": arm,
                    "result": B.ref(output / case_id / arm / "result.json"),
                    "end_state": result["end_state"],
                    "scoring": result["scoring"],
                }
            )
    unavailable_rows = sum(row["scoring"] is None for row in results)
    result = {
        "schema": "lineage_five_policy_batch_result_v1",
        "status": "complete",
        "execution_protocol": B.ref(protocol_path),
        "execution_anchor": anchor,
        "execution_config": B.ref(config_path),
        "planned_cases": 10,
        "planned_rows": 50,
        "rows": results,
        "input_unavailable_rows": unavailable_rows,
        "comparison_status": (
            "inconclusive_input_unavailable"
            if unavailable_rows
            else "eligible_for_registered_summary"
        ),
        "paired_intents": paired,
        "sdk_roles_reserved": paired * 2,
        "transport": NIM.status(output / "nim"),
        "parent_wall_s": time.monotonic() - started,
        "lease_id": first_lease["leaseId"],
        "repair": False,
        "recovery": False,
        "heldout": False,
        "model_version_gain": False,
    }
    B.write(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.preflight:
            _protocol, _config, sources = preflight(
                args.protocol, args.anchor, args.config
            )
            B.write(
                args.out,
                {"status": "complete", "case_ids": list(sources), "api": 0, "gpu": 0},
            )
        else:
            execute(args.protocol, args.anchor, args.config, args.out)
    except Exception as error:
        B.write(
            args.out.with_name(args.out.name + "-failure.json")
            if args.preflight
            else args.out / "failure.json",
            {
                "status": "invalid",
                "error_type": type(error).__name__,
                "message": str(error),
                "automatic_retry": False,
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
