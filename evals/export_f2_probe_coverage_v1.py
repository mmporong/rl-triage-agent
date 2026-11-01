# pyright: reportMissingImports=false
"""RI410 F2 oracle control의 검증된 SHA-only 공개 요약을 만든다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_probe_cost_audit_v1 as AUDIT
import run_f2_probe_coverage_v1 as RUN

SCHEMA = "f2_probe_coverage_public_export_v1"
TARGETS = ("case-05", "case-06")
OUTCOMES = {
    "normal": "signature_not_observed",
    "abnormal": "confirmed_signature",
    "unknown": "unknown_HOLD",
}
SOURCE_FILES = (
    "evals/export_f2_probe_coverage_v1.py",
    "tests/test_export_f2_probe_coverage_v1.py",
    "docs/F2-PROBE-COVERAGE.md",
    *RUN.SOURCE_FILES,
)
HEX64 = frozenset("0123456789abcdef")
DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object가 필요하다")
    return value


def write_new(path: Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _lf_sha(path: Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _sha64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _bound(value: object, label: str) -> Path:
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}, label)
    row = cast(Mapping[str, Any], value)
    path = Path(cast(str, row["path"])).resolve(strict=True)
    require(sha(path) == row["sha256"], label + " SHA")
    return path


def _fresh_pair_path(result_path: Path, case_id: str) -> Path:
    """RI410의 고정 `<run>/<case>/pair/result.json` 저장 계약을 검사한다."""
    require(case_id in TARGETS and Path(result_path).name == "result.json",
            "RI410 result/case contract")
    root = Path(result_path).resolve(strict=True).parent
    pair = root / case_id / "pair" / "result.json"
    require(pair.is_file() and pair.resolve().parent.parent.name == case_id,
            "fresh pair receipt missing")
    return pair.resolve()


def _frozen_snapshot(pair_path: Path, source: dict[str, Any], case_id: str,
                     temp_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pair = read(pair_path)
    ledger = pair.get("public_ledger")
    require(isinstance(ledger, dict) and isinstance(ledger.get("path"), str),
            "fresh pair public ledger")
    ledger_path = Path(ledger["path"]).resolve(strict=True)
    expected_ledger = pair_path.parents[2] / case_id / "oracle.ledger.jsonl"
    require(ledger_path == expected_ledger.resolve(strict=True), "fresh pair storage contract")
    temp_root.mkdir(parents=True, exist_ok=True)
    empty = temp_root / f"{case_id}.ledger.jsonl"
    empty.write_bytes(b"")
    snapshot = RUN.DRIVER.C.V.build_snapshot(
        source["packet"], f"f2_oracle_{case_id.replace('-', '_')}",
        RUN.HOST_STORAGE_POLICY, RUN.DRIVER.C.V.file_digest(RUN.DRIVER.C.V.PROTOCOL), empty)
    snapshot["inputs"]["ledger_path"] = str(ledger_path)
    choice = pair.get("public_choice")
    require(isinstance(choice, dict)
            and RUN.B.digest(snapshot) == pair.get("public_snapshot_sha256")
            and choice == RUN.oracle_choice(snapshot), "fresh pair frozen snapshot/choice")
    return snapshot, choice


def _reject_paths(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require("path" not in str(key).lower() and "lease" not in str(key).lower(),
                    "public path/lease field")
            _reject_paths(item)
    elif isinstance(value, list):
        for item in value:
            _reject_paths(item)
    elif isinstance(value, str):
        require(not DRIVE.match(value) and not value.startswith(("/", "\\\\"))
                and "\\" not in value and "/" not in value, "public path string")


def verify_source_anchor(anchor: str, repo: Path = ROOT) -> dict[str, str]:
    require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "source anchor")
    repo = Path(repo).resolve()
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    require(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                            anchor, head], check=False).returncode == 0, "source anchor ancestry")
    output = {}
    for name in sorted(set(SOURCE_FILES)):
        blob = subprocess.run(["git", "-C", str(repo), "show", f"{anchor}:{name}"],
                              check=True, capture_output=True).stdout
        actual = hashlib.sha256(blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
        expected = _lf_sha(repo / name)
        require(actual == expected, "exporter source blob changed: " + name)
        output[name] = expected
    return output


def build_export(result_path: Path, protocol_path: Path, registration_anchor: str,
                 plan_path: Path, *, lab: Path,
                 validator: Callable = RUN.verify_protocol,
                 inspect_step: Callable = AUDIT.inspect_step,
                 inspect_pair: Callable = AUDIT.inspect_pair,
                 pair_verifier: Callable = RUN.HOST.verify_pair,
                 source_verifier: Callable = verify_source_anchor) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", registration_anchor) is not None,
            "registration anchor")
    source_hashes = source_verifier(registration_anchor)
    require(set(source_hashes) == set(SOURCE_FILES)
            and all(_sha64(value) for value in source_hashes.values()),
            "exporter source closure")
    protocol, plan, material = validator(
        protocol_path, registration_anchor, plan_path, lab=lab)
    result_path = Path(result_path).resolve(strict=True)
    result = read(result_path)
    targets = result.get("targets")
    require(result.get("schema") == RUN.RESULT_SCHEMA and result.get("status") == "complete"
            and isinstance(targets, list)
            and [row.get("case_id") for row in targets] == list(TARGETS)
            and result.get("api_calls") == 0 and result.get("model_calls") == 0
            and result.get("training") is False and result.get("correction") is False
            and result.get("original_comparison_modified") is False
            and result.get("selection_performance_credit") == 0
            and type(result.get("new_pairs")) is int and 0 <= result["new_pairs"] <= 2
            and result.get("sdk_roles_reserved") == result["new_pairs"] * 2
            and isinstance(result.get("parent_wall_s"), (int, float))
            and 0 <= result["parent_wall_s"] <= 3600,
            "RI410 complete result/budget")
    require([row.get("case_id") for row in plan.get("targets", [])] == list(TARGETS),
            "RI410 fixed targets")
    require(_sha64(plan.get("batch_result", {}).get("sha256")), "original batch SHA")
    exported: list[dict[str, Any]] = []
    fresh_count = 0
    observed_sdk = 0
    known_wall = 0.0
    with tempfile.TemporaryDirectory(prefix="f2-coverage-export-") as folder:
        temporary = Path(folder)
        for summary, planned in zip(targets, plan["targets"], strict=True):
            case_id = summary["case_id"]
            outcome = summary.get("outcome")
            execution = summary.get("execution")
            require(summary.get("oracle_label") == "F2" and summary.get("probe") == RUN.PROBE
                    and outcome in OUTCOMES and summary.get("coverage") == OUTCOMES[outcome]
                    and summary.get("control_policy") == RUN.ORACLE_POLICY
                    and summary.get("host_storage_policy") == RUN.HOST_STORAGE_POLICY
                    and summary.get("selected_by") == "registered_rule"
                    and summary.get("selection_performance_credit") == 0,
                    "RI410 target summary")
            source = material["prepared"][case_id]["source"]
            require(RUN.B.digest(source) == planned["source_digest"], "RI410 source digest")
            historical = planned.get("historical")
            if execution == "reused_existing_receipt":
                require(historical is not None and planned.get("action") == "reuse",
                        "historical reuse plan")
                checked = inspect_step(
                    _bound(historical["detail"], "historical detail"), source,
                    historical["step_index"], temporary / "history" / case_id)
                require(checked == historical["verified"]
                        and checked.get("probe") == RUN.PROBE
                        and checked.get("outcome") == outcome, "historical reuse receipt")
                pair_sha = checked["pair"]["pair_result_sha256"]
                target_cost = {"new_sdk_roles_reserved": 0, "new_sdk_roles_observed": 0,
                               "known_owned_wall_s": 0.0, "reused_prior_receipt": True}
            else:
                require(execution == "executed_one_pair" and historical is None
                        and planned.get("action") == "execute_one_pair", "fresh pair plan")
                pair_path = _fresh_pair_path(result_path, case_id)
                snapshot, choice = _frozen_snapshot(pair_path, source, case_id,
                                                    temporary / "fresh")
                verified = pair_verifier(pair_path, snapshot, choice)
                audited = inspect_pair(pair_path)
                require(verified.get("probe") == RUN.PROBE
                        and verified.get("outcome") == outcome
                        and verified.get("retry_count") == 0
                        and audited.get("probe") == RUN.PROBE
                        and audited.get("outcome") == outcome
                        and audited.get("pair_result_sha256") == sha(pair_path),
                        "fresh pair host/audit receipt")
                roles = audited["reconstructed"]["sdk_roles_observed"]
                wall = audited["reconstructed"]["known_owned_wall_s"]
                require(type(roles) is int and 0 <= roles <= 2
                        and isinstance(wall, (int, float)) and 0 <= wall <= 3600,
                        "fresh pair observed cost")
                fresh_count += 1
                observed_sdk += roles
                known_wall += float(wall)
                pair_sha = audited["pair_result_sha256"]
                target_cost = {"new_sdk_roles_reserved": 2,
                               "new_sdk_roles_observed": roles,
                               "known_owned_wall_s": wall,
                               "reused_prior_receipt": False}
            require(_sha64(pair_sha), "pair result SHA")
            exported.append({
                "case_id": case_id, "fault": "F2", "probe": RUN.PROBE,
                "outcome": outcome, "coverage": OUTCOMES[outcome],
                "execution": execution, "pair_result_sha256": pair_sha,
                "control_policy": RUN.ORACLE_POLICY,
                "selection_performance_credit": 0, "cost": target_cost,
            })
    require(fresh_count == result["new_pairs"]
            and fresh_count * 2 == result["sdk_roles_reserved"], "RI410 fresh pair totals")
    public = {
        "schema": SCHEMA, "status": "verified",
        "scope": "posthoc_development_probe_coverage_control",
        "evidence": {
            "overlay_sha256": sha(Path(protocol_path).resolve(strict=True)),
            "private_plan_sha256": sha(Path(plan_path).resolve(strict=True)),
            "original_batch_sha256": plan["batch_result"]["sha256"],
            "registration_anchor": registration_anchor,
            "exporter_sources": [
                {"component": Path(name).name, "sha256": source_hashes[name]}
                for name in sorted(source_hashes)
            ],
        },
        "targets": exported,
        "new_cost": {"pairs": fresh_count, "sdk_roles_reserved": fresh_count * 2,
                     "sdk_roles_observed": observed_sdk,
                     "known_owned_wall_s": known_wall, "api_calls": 0},
        "claims": {"original_50_row_comparison_modified": False,
                   "selection_performance_credit": 0,
                   "diagnosis_success_credit": 0,
                   "recovery_or_media_credit": 0},
    }
    require(protocol.get("r3", {}).get("batch_sha256") == plan["batch_result"]["sha256"]
            and protocol.get("oracle_policy", {}).get("name") == RUN.ORACLE_POLICY
            and protocol.get("oracle_policy", {}).get("selection_performance_credit") == 0,
            "registered oracle overlay/original batch")
    _reject_paths(public)
    return deepcopy(public)


def export(result_path: Path, protocol_path: Path, registration_anchor: str,
           plan_path: Path, *, lab: Path, out: Path) -> dict[str, Any]:
    value = build_export(result_path, protocol_path, registration_anchor, plan_path, lab=lab)
    write_new(out, value)
    return value


def verify(export_path: Path, result_path: Path, protocol_path: Path,
           registration_anchor: str, plan_path: Path, *, lab: Path) -> dict[str, Any]:
    value = read(Path(export_path).resolve(strict=True))
    expected = build_export(result_path, protocol_path, registration_anchor, plan_path, lab=lab)
    require(value == expected, "public F2 coverage export changed")
    return deepcopy(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "verify"):
        child = sub.add_parser(name)
        child.add_argument("--result", type=Path, required=True)
        child.add_argument("--protocol", type=Path, required=True)
        child.add_argument("--registration-anchor", required=True)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--lab", type=Path, required=True)
        child.add_argument("--out" if name == "export" else "--export", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export(args.result, args.protocol, args.registration_anchor, args.plan,
               lab=args.lab, out=args.out)
    else:
        verify(args.export, args.result, args.protocol, args.registration_anchor,
               args.plan, lab=args.lab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
