"""검증된 recorded-vs-fresh audit를 경로 없는 공개 결과로 내보낸다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import diagnostic_fact_ledger_v1 as FACTS
import evidence_guided_selector_v1 as SEL
import prospective_diagnosis_inputs_v2 as INPUT
import recorded_optimizer_probe_audit_v1 as AUDIT

SCHEMA = "recorded_optimizer_probe_audit_public_v1"
MANIFEST_SCHEMA = "recorded_optimizer_probe_audit_public_manifest_v1"
PUBLIC_FILES = ("manifest.json", "protocol.json", "summary.json",
                "seeds/s2320.json", "seeds/s2321.json")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 객체여야 한다")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_public(value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    _require("bench/private" not in text.replace("\\", "/").lower(), "공개 결과에 private 경로가 있다")
    _require(re.search(r"(?:[A-Za-z]:[\\/]|/(?:home|mnt|Users)/)", text) is None,
             "공개 결과에 물리 경로가 있다")


def _state(case: dict[str, Any], reference: dict[str, Any], ledger_path: Path,
           target: str) -> tuple[dict[str, Any], dict[str, Any]]:
    packet = INPUT.make_packet(case["input"], reference["input"], budget_left=3)
    ledger = FACTS.checked(ledger_path)
    expected = FACTS.build(Path(case["metadata"]["runtime_facts"]["case"]["path"]),
                           Path(reference["metadata"]["runtime_facts"]["reference"]["path"]))
    _require(ledger == expected, "audit harness fact ledger가 source와 다르다")
    state = SEL.initialize(packet["packet"], ledger_path)
    if target == AUDIT.RECORDED.PROBE:
        state = SEL.advance(state, {"action": "read_evidence", "target": SEL.RESOURCE,
                                    "fact_ids": []})
        action = SEL.select_strong_rule(state)
    else:
        action = {"action": "run_probe", "target": AUDIT.FRESH_PROBE, "fact_ids": []}
    _require(action["target"] == target, "audit harness action이 method와 다르다")
    SEL.request_spec(state, action)
    return state, action


def _method(execution_root: Path, seed: int, target: str, case: dict[str, Any],
            reference: dict[str, Any], recorded: dict[str, Any]) -> dict[str, Any]:
    root = execution_root / f"s{seed}/{target}"
    host = _read(root / "probe/host-result.json")
    paths = {name: Path(host[name]).resolve() for name in (
        "case_plan", "case_receipt", "reference_plan", "reference_receipt")}
    _require(all(path.is_file() and path.is_relative_to(root.resolve()) for path in paths.values()),
             "host result proof가 audit method root를 이탈했다")
    state, action = _state(case, reference, root / "harness/fact-ledger.json", target)
    final = SEL.advance(state, action, **paths)
    measurements = {role: AUDIT._canonical(target, role, paths[f"{role}_plan"],
                                            paths[f"{role}_receipt"])
                    for role in ("case", "reference")}
    derived = {"measurements": measurements, "paired_outcome": final["observed"][target],
               "parent_wall_s": host["parent_wall_s"],
               "owned_child_wall_s": host["owned_child_wall_s"],
               "sdk_launches": 0 if target == AUDIT.RECORDED.PROBE else 2}
    _require(derived == recorded, "재계산한 measurement/outcome/cost가 parent result와 다르다")
    return derived


def export(result_path: Path, protocol_path: Path, source_batch_result: Path, *, anchor: str,
           lab: Path, out: Path, private_out: Path) -> dict[str, Any]:
    out, private_out = Path(out).resolve(), Path(private_out).resolve()
    _require(not out.exists() and not private_out.exists(), "export 출력은 fresh여야 한다")
    protocol = AUDIT.verify_protocol(protocol_path, anchor, lab,
                                     source_batch_result=source_batch_result)
    result = _read(result_path)
    _require(result.get("schema") == AUDIT.RESULT_SCHEMA and result.get("status") == "complete"
             and result.get("planned_seeds") == 2 and result.get("agreement_count") == 2
             and result.get("conditional_candidate") is True
             and result.get("model_requests") == 0 and result.get("repairs") == 0
             and result.get("sdk_launches") == 4,
             "audit parent result의 완료/상한/일치 조건이 다르다")
    _require(result.get("claims") == {"gpu_kernel_savings": False, "early_detection": False,
                                      "model_advantage": False, "recovery": False},
             "audit claim boundary가 다르다")
    bound = {**protocol, "_source_batch_path": str(Path(source_batch_result).resolve())}
    cases = AUDIT._cases(bound, anchor)
    public_rows = []
    for source_row in result["rows"]:
        seed = int(source_row["seed_id"][1:])
        _require(seed in AUDIT.SEEDS and source_row["order"] == list(AUDIT.ORDER[seed]),
                 "audit counterbalance order 불일치")
        case, reference = cases[seed]
        methods = {target: _method(Path(result_path).parent, seed, target, case, reference,
                                   source_row["methods"][target])
                   for target in AUDIT.ORDER[seed]}
        agreement = (methods[AUDIT.RECORDED.PROBE]["measurements"]
                     == methods[AUDIT.FRESH_PROBE]["measurements"]
                     and methods[AUDIT.RECORDED.PROBE]["paired_outcome"]
                     == methods[AUDIT.FRESH_PROBE]["paired_outcome"])
        _require(agreement is source_row["agreement"] is True,
                 "audit seed agreement 재계산 불일치")
        row = {"seed_id": source_row["seed_id"], "counterbalanced_order": source_row["order"],
               "methods": methods, "agreement": agreement,
               "recorded_parent_wall_lower": (
                   methods[AUDIT.RECORDED.PROBE]["parent_wall_s"]
                   < methods[AUDIT.FRESH_PROBE]["parent_wall_s"])}
        _safe_public(row)
        public_rows.append(row)
    summary = {
        "schema": SCHEMA, "status": "complete", "agreement": {"count": 2, "planned": 2},
        "conditional_candidate": True,
        "candidate_scope": "stored_first_three_updates_available",
        "candidate_statement": "보관된 첫 3 update 계측이 있는 조건의 더 가벼운 검사 경로",
        "model_requests": 0, "repairs": 0, "fresh_sdk_launches": 4,
        "recorded_sdk_launches": 0, "source_failure_required": False,
        "main_comparison_healthy_gate_changed": False,
        "claims": deepcopy(result["claims"]),
        "seeds": [{"seed_id": row["seed_id"], "ref": f"seeds/{row['seed_id']}.json"}
                  for row in public_rows],
    }
    _safe_public(summary)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, out / "protocol.json")
    _write(out / "summary.json", summary)
    for row in public_rows:
        _write(out / "seeds" / f"{row['seed_id']}.json", row)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol": {"path": "protocol.json", "sha256": _sha(out / "protocol.json"),
                     "registered_path": "bench/protocols/recorded_optimizer_probe_audit_20261101_v2.json",
                     "anchor": anchor},
        "actual_sources": {"parent_result_sha256": _sha(result_path),
                           "source_batch_sha256": _sha(source_batch_result)},
        "files": {name: _sha(out / name) for name in PUBLIC_FILES if name != "manifest.json"},
    }
    _safe_public(manifest)
    _write(out / "manifest.json", manifest)
    _require({path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()}
             == set(PUBLIC_FILES), "공개 파일 inventory 불일치")
    private_out.mkdir(parents=True, exist_ok=False)
    public_lines = "\n".join(PUBLIC_FILES) + "\n"
    (private_out / "PUBLIC_FILES.txt").write_text(public_lines, encoding="utf-8", newline="\n")
    sums = "\n".join(f"{_sha(out / name)}  {name}" for name in PUBLIC_FILES) + "\n"
    (private_out / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
    result_md = (
        "# Recorded optimizer probe audit v2\n\n"
        "두 seed 모두 recorded CPU 검사와 fresh 3-update 검사가 같은 count 및 판정을 냈다.\n"
        "후보 범위는 보관된 첫 3 update 계측이 있는 조건으로 제한한다.\n")
    (private_out / "RESULT.md").write_text(result_md, encoding="utf-8", newline="\n")
    _write(private_out / "provenance.json", {
        "result": {"path": str(Path(result_path).resolve()), "sha256": _sha(result_path)},
        "source_batch": {"path": str(Path(source_batch_result).resolve()),
                         "sha256": _sha(source_batch_result)},
        "protocol": {"path": str(Path(protocol_path).resolve()), "sha256": _sha(protocol_path)},
        "anchor": anchor, "lab": str(Path(lab).resolve())})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "protocol", "source-batch-result", "lab", "out", "private-out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    args = parser.parse_args(argv)
    export(args.result, args.protocol, args.source_batch_result, anchor=args.anchor,
           lab=args.lab, out=args.out, private_out=args.private_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
