# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src", "tests")]

import export_lineage_five_policy_v1 as E
import lineage_five_policy_summary_v1 as S
import test_lineage_five_policy_summary_v1 as FIX


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    batch, _paths = FIX.make_batch(tmp_path)
    summary = tmp_path / "input-summary.json"
    S.summarize(batch, FIX.STUDY, summary, allow_fixture=True)
    return batch, summary


def test_fixture_export_preserves_summary_and_fifty_rows_without_paths(tmp_path: Path) -> None:
    batch, summary = fixture(tmp_path)
    out = tmp_path / "public"
    manifest = E.export(batch, FIX.STUDY, summary, out, allow_fixture=True)
    public_summary = E._read(out / "summary.json")
    cases = E._read(out / "cases.json")
    provenance = E._read(out / "provenance.json")
    original = E._read(summary)
    assert manifest["scope"] == "cpu_fixture_development_measurement"
    assert len(cases["rows"]) == 50
    assert public_summary["primary"] == original["primary"]
    assert public_summary["secondary"] == original["secondary"]
    assert public_summary["policies"] == original["policies"]
    assert public_summary["claims"] == original["claims"]
    assert provenance["fixture_promoted_to_actual"] is False
    text = "\n".join(path.read_text(encoding="utf-8") for path in out.glob("*.json"))
    assert str(tmp_path) not in text and "bench/private" not in text


def test_tampered_supplied_summary_is_rejected_before_output(tmp_path: Path) -> None:
    batch, summary = fixture(tmp_path)
    changed = E._read(summary)
    changed["policies"]["logs_only"]["paired_probes"] += 1
    summary.unlink()
    write(summary, changed)
    out = tmp_path / "public"
    with pytest.raises(ValueError, match="재계산"):
        E.export(batch, FIX.STUDY, summary, out, allow_fixture=True)
    assert not out.exists()


def test_tampered_result_source_is_rejected_before_output(tmp_path: Path) -> None:
    batch, summary = fixture(tmp_path)
    batch_value = E._read(batch)
    detail_path = Path(batch_value["rows"][0]["result"]["path"])
    detail = E._read(detail_path)
    detail["source_digest"] = "f" * 64
    detail_path.unlink()
    write(detail_path, detail)
    out = tmp_path / "public"
    with pytest.raises(ValueError):
        E.export(batch, FIX.STUDY, summary, out, allow_fixture=True)
    assert not out.exists()


def test_output_is_write_once(tmp_path: Path) -> None:
    batch, summary = fixture(tmp_path)
    out = tmp_path / "public"
    E.export(batch, FIX.STUDY, summary, out, allow_fixture=True)
    with pytest.raises(ValueError, match="fresh"):
        E.export(batch, FIX.STUDY, summary, out, allow_fixture=True)


def test_input_unavailable_nested_result_refs_are_digest_only(tmp_path: Path) -> None:
    batch = tmp_path / "batch.json"
    study = tmp_path / "study.json"
    raw = tmp_path / "private-result.json"
    write(batch, {"batch": True})
    write(study, {"study": True})
    write(raw, {"result": True})
    recomputed = {
        "scope": "actual_backend_inconclusive_input_unavailable",
        "input_unavailable": {
            "rows": 1,
            "ratio": None,
            "missing_rows": [{"case_id": "case-01", "policy": "logs_only",
                              "result": S._ref(raw)}],
            "observed_rows": [],
        },
        "claims": {"general_superiority": False},
    }
    public = E._public_summary(recomputed, batch, study)
    row = public["input_unavailable"]["missing_rows"][0]
    assert row["result"] == E._digest_ref(raw)
    assert public["input_unavailable"]["rows"] == 1
    assert public["input_unavailable"]["ratio"] is None
    E._assert_public(public)
    assert str(tmp_path) not in json.dumps(public)


def test_recursive_public_guard_rejects_private_paths_and_credentials() -> None:
    with pytest.raises(ValueError, match="경로"):
        E._assert_public({"nested": [{"result": "C:/Users/example/private.json"}]})
    with pytest.raises(ValueError, match="자격증명"):
        E._assert_public({"nested": {"authorization": "redacted"}})


def test_production_protocol_must_be_public_bench_protocol(tmp_path: Path) -> None:
    private_protocol = tmp_path / "bench" / "private" / "protocol.json"
    write(private_protocol, {"schema": "private"})
    with pytest.raises(ValueError, match="저장소"):
        E._public_protocol(S._ref(private_protocol), allow_fixture=False)

    repo_private = ROOT / ("bench/private/claude_reward_track_20261013/"
                           "codex_main_20261025/vision_goal/ri_full_completion_251/"
                           "replan_369/comparison_export_389/test-private-protocol.json")
    write(repo_private, {"schema": "private"})
    try:
        with pytest.raises(ValueError, match="bench/protocols"):
            E._public_protocol(S._ref(repo_private), allow_fixture=False)
    finally:
        repo_private.unlink()
