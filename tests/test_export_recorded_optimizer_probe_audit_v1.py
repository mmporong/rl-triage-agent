# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import export_recorded_optimizer_probe_audit_v1 as E


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def measurements() -> dict:
    return {"case": {"role": "case", "expected_optimizer_steps": 20,
                     "optimizer_steps": [4, 4], "expected_minibatches": 20,
                     "minibatches": [4, 4], "outcome": "abnormal", "records": 3},
            "reference": {"role": "reference", "expected_optimizer_steps": 20,
                          "optimizer_steps": [20, 20], "expected_minibatches": 20,
                          "minibatches": [20, 20], "outcome": "normal", "records": 3}}


def method(target: str, seed: int) -> dict:
    recorded = target == E.AUDIT.RECORDED.PROBE
    return {"measurements": measurements(), "paired_outcome": "abnormal",
            "parent_wall_s": 40.0 + (seed - 2320) if recorded else 204.0 + (seed - 2320),
            "owned_child_wall_s": 0.0 if recorded else 55.6,
            "sdk_launches": 0 if recorded else 2}


def test_safe_public_rejects_private_and_absolute_paths() -> None:
    with pytest.raises(ValueError, match="private"):
        E._safe_public({"path": "bench/private/x.json"})
    with pytest.raises(ValueError, match="물리"):
        E._safe_public({"path": "C:\\secret\\x.json"})


def test_method_rejects_parent_cost_mismatch(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: Path) -> None:
    root = tmp_path / "s2320" / E.AUDIT.RECORDED.PROBE
    paths = {}
    for role in ("case", "reference"):
        paths[f"{role}_plan"] = str(write(root / "probe" / role / "plan.json", {}).resolve())
        paths[f"{role}_receipt"] = str(write(root / "probe" / role / "receipt.json", {}).resolve())
    write(root / "probe/host-result.json", {
        **paths, "parent_wall_s": 40.0, "owned_child_wall_s": 0.0})
    write(root / "harness/fact-ledger.json", {})
    monkeypatch.setattr(E, "_state", lambda *_args: ({}, {"target": E.AUDIT.RECORDED.PROBE}))
    monkeypatch.setattr(E.SEL, "advance", lambda *_args, **_kwargs: {
        "observed": {E.AUDIT.RECORDED.PROBE: "abnormal"}})
    monkeypatch.setattr(E.AUDIT, "_canonical", lambda target, role, plan, receipt: measurements()[role])
    bad = method(E.AUDIT.RECORDED.PROBE, 2320)
    bad["parent_wall_s"] = 41.0
    with pytest.raises(ValueError, match="parent result"):
        E._method(tmp_path, 2320, E.AUDIT.RECORDED.PROBE, {}, {}, bad)


def test_export_writes_only_safe_public_inventory(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    protocol = write(tmp_path / "protocol.json", {"schema": E.AUDIT.SCHEMA})
    source = write(tmp_path / "source.json", {"status": "complete"})
    rows = []
    for seed in E.AUDIT.SEEDS:
        methods = {target: method(target, seed) for target in E.AUDIT.ORDER[seed]}
        rows.append({"seed_id": f"s{seed}", "order": list(E.AUDIT.ORDER[seed]),
                     "methods": methods, "agreement": True,
                     "recorded_parent_wall_lower": True})
    result = write(tmp_path / "result.json", {
        "schema": E.AUDIT.RESULT_SCHEMA, "status": "complete", "planned_seeds": 2,
        "agreement_count": 2, "conditional_candidate": True, "model_requests": 0,
        "repairs": 0, "sdk_launches": 4, "rows": rows,
        "claims": {"gpu_kernel_savings": False, "early_detection": False,
                   "model_advantage": False, "recovery": False}})
    monkeypatch.setattr(E.AUDIT, "verify_protocol", lambda *_args, **_kwargs: {
        "source_batch": {"sha256": E._sha(source)}})
    monkeypatch.setattr(E.AUDIT, "_cases", lambda *_args: {2320: ({}, {}), 2321: ({}, {})})
    monkeypatch.setattr(E, "_method", lambda root, seed, target, case, reference, recorded:
                        recorded)
    summary = E.export(result, protocol, source, anchor="a" * 40, lab=tmp_path,
                       out=tmp_path / "public", private_out=tmp_path / "private")
    public = tmp_path / "public"
    assert summary["agreement"] == {"count": 2, "planned": 2}
    assert {path.relative_to(public).as_posix() for path in public.rglob("*") if path.is_file()} \
        == set(E.PUBLIC_FILES)
    for path in public.rglob("*.json"):
        E._safe_public(json.loads(path.read_text(encoding="utf-8")))
    assert (tmp_path / "private/PUBLIC_FILES.txt").read_text().splitlines() == list(E.PUBLIC_FILES)


def test_export_is_write_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        E.export(tmp_path / "r", tmp_path / "p", tmp_path / "s", anchor="a" * 40,
                 lab=tmp_path, out=out, private_out=tmp_path / "private")
