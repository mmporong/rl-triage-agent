from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import core_seed_guard_v2 as G


def _layout(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    repo, lab = tmp_path / "repo", tmp_path / "lab"
    for path in (repo / "bench/private/run", repo / "bench/protocols", repo / "evals", lab / "logs"):
        path.mkdir(parents=True)
    protocol = repo / "bench/protocols/five_type_recovery_first_s4552_s4557_20261101_v1.json"
    protocol.write_text('{"schema":"plan"}\n', encoding="utf-8")
    reference = {
        "path": str(protocol),
        "sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
    }
    return repo, lab, protocol, reference


def _receipt(repo: Path, value: dict) -> Path:
    path = repo / "bench/private/run/receipt.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_valid_protocol_range_end_is_fresh_but_registered_seed_is_used(tmp_path: Path) -> None:
    repo, lab, _protocol, reference = _layout(tmp_path)
    _receipt(repo, {"registered_seed": 4552, "protocol": reference})

    assert G.SOURCE_FILES == (
        "evals/core_seed_guard_v2.py",
        "tests/test_core_seed_guard_v2.py",
    )
    result = G.assert_fresh_seeds([4557], lab, repo=repo)
    assert result["status"] == "fresh"
    assert result["schema"] == "core_seed_freshness_v2"
    assert result["verified_planning_protocol_refs"] == [
        "bench/protocols/five_type_recovery_first_s4552_s4557_20261101_v1.json"
    ]
    with pytest.raises(G.V1.ORIGINAL.SeedReuseError, match="4552"):
        G.assert_fresh_seeds([4552], lab, repo=repo)


def test_wrong_protocol_sha_fails_closed_and_restores_parser(tmp_path: Path) -> None:
    repo, lab, _protocol, reference = _layout(tmp_path)
    reference["sha256"] = "0" * 64
    _receipt(repo, {"registered_seed": 4552, "protocol_ref": reference})
    old_files = G.V1.ORIGINAL._files
    old = G.V1.ORIGINAL._record_seeds

    with pytest.raises(ValueError, match="SHA가 원문과 다르다"):
        G.assert_fresh_seeds([4557], lab, repo=repo)
    assert G.V1.ORIGINAL._files is old_files
    assert G.V1.ORIGINAL._record_seeds is old


def test_unrelated_missing_protocol_is_not_inspected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, lab, _protocol, _reference = _layout(tmp_path)
    unrelated = {
        "path": str(repo / "bench/protocols/probe.json"),
        "sha256": "0" * 64,
    }
    _receipt(repo, {"registered_seed": 4552, "protocol": unrelated})
    monkeypatch.setattr(
        G,
        "_lexical_public_protocol",
        lambda *_args: pytest.fail("무관한 protocol 참조를 검사했다"),
    )

    assert G.assert_fresh_seeds([4557], lab, repo=repo)["status"] == "fresh"


def test_requested_seed_in_missing_protocol_fails_closed(tmp_path: Path) -> None:
    repo, lab, _protocol, _reference = _layout(tmp_path)
    missing = {
        "path": str(repo / "bench/protocols/missing_s4557.json"),
        "sha256": "0" * 64,
    }
    _receipt(repo, {"registered_seed": 4552, "protocol": missing})

    with pytest.raises(ValueError, match="공개 protocol 파일이 아니다"):
        G.assert_fresh_seeds([4557], lab, repo=repo)


@pytest.mark.parametrize(
    "actual",
    [
        {"registered_seed": 4557},
        {"argv": ["train", "--seed", "4557"]},
        {"checkpoint_path": "runs/model_s4557_final.pt"},
    ],
)
def test_structured_seed_argv_and_checkpoint_remain_used(tmp_path: Path, actual: dict) -> None:
    repo, lab, _protocol, reference = _layout(tmp_path)
    _receipt(repo, {**actual, "source_protocol": reference})

    with pytest.raises(G.V1.ORIGINAL.SeedReuseError, match="4557"):
        G.assert_fresh_seeds([4557], lab, repo=repo)


def test_isaac_log_seed_remains_used(tmp_path: Path) -> None:
    repo, lab, _protocol, reference = _layout(tmp_path)
    _receipt(repo, {"registered_seed": 4552, "protocol": reference})
    (lab / "logs/run_s4557_actual").mkdir()

    with pytest.raises(G.V1.ORIGINAL.SeedReuseError, match="4557"):
        G.assert_fresh_seeds([4557], lab, repo=repo)


def test_protocol_path_string_outside_reference_is_conservatively_used(tmp_path: Path) -> None:
    repo, lab, protocol, _reference = _layout(tmp_path)
    _receipt(repo, {"note": str(protocol)})

    with pytest.raises(G.V1.ORIGINAL.SeedReuseError, match="4557"):
        G.assert_fresh_seeds([4557], lab, repo=repo)


def test_provider_canary_is_not_enumerated_and_callbacks_restore(tmp_path: Path) -> None:
    repo, lab, _protocol, reference = _layout(tmp_path)
    _receipt(repo, {"registered_seed": 4552, "protocol": reference})
    excluded = repo / G.V1.EXCLUDED_RELATIVE
    excluded.mkdir(parents=True)
    (excluded / "result.json").write_text("not JSON s4557", encoding="utf-8")
    old_files = G.V1.ORIGINAL._files
    old_records = G.V1.ORIGINAL._record_seeds

    result = G.assert_fresh_seeds([4557], lab, repo=repo)
    assert result["status"] == "fresh"
    assert G.V1.ORIGINAL._files is old_files
    assert G.V1.ORIGINAL._record_seeds is old_records
