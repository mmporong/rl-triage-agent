from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "tests", ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import test_vision_selection_sources_v1 as FIXTURE  # pyright: ignore[reportMissingImports]
import vision_record_selection_inputs_v2 as I  # pyright: ignore[reportMissingImports]


def test_inputs_v2_requires_capture_first_readiness_v2() -> None:
    assert I.READINESS.SCHEMA == "vision_record_source_readiness_v2"
    assert I.READINESS.BINDING_SCHEMA == "vision_record_source_readiness_binding_v2"


@pytest.fixture(scope="module")
def source_case(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return cast(Any, FIXTURE.full_case).__wrapped__(tmp_path_factory)


def test_first_class_verifier_preserves_full_twenty_four_input_validation(
    source_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = I.BASE.verify_model_readiness
    calls: list[Path] = []

    def readiness(repo: Path, path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
        calls.append(path)
        return legacy(repo, path, protocol)

    monkeypatch.setattr(I.READINESS, "verify_model_readiness", readiness)
    result = I.verify_inputs(
        source_case["repo"], source_case["protocol"], source_case["locations"]
    )
    assert result["schema"] == I.BASE.VERIFIED_SCHEMA
    assert len(result["rows"]) == 24
    assert len(calls) == 4
    assert len({path.resolve() for path in calls}) == 4


def test_readiness_flag_or_bound_raw_mutation_is_rejected(
    source_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        I.READINESS, "verify_model_readiness", I.BASE.verify_model_readiness
    )
    locations = I.BASE._read(source_case["locations"])
    manifest_path = I.BASE._ref(
        source_case["repo"], locations["rows"][0]["input_manifest"], frozen=True
    )
    manifest = I.BASE._read(manifest_path)
    readiness_path = I.BASE._ref(source_case["repo"], manifest["model_readiness"])
    original = readiness_path.read_bytes()
    try:
        forged = json.loads(original)
        forged["model_use_ready"] = True
        forged["receipt"]["video_sha256"] = "f" * 64
        readiness_path.write_text(json.dumps(forged), encoding="utf-8")
        with pytest.raises(ValueError, match="SHA|readiness"):
            I.verify_inputs(
                source_case["repo"], source_case["protocol"], source_case["locations"]
            )
    finally:
        readiness_path.write_bytes(original)

    trace_path = I.BASE._ref(source_case["repo"], manifest["trace"])
    trace_original = trace_path.read_bytes()
    try:
        trace_path.write_bytes(trace_original + b"\n")
        with pytest.raises(ValueError, match="SHA"):
            I.verify_inputs(
                source_case["repo"], source_case["protocol"], source_case["locations"]
            )
    finally:
        trace_path.write_bytes(trace_original)


def test_materializer_injects_both_first_class_verifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def materialize(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"rows": 24}

    monkeypatch.setattr(I.MATERIALIZE, "materialize", materialize)
    result = I.materialize(
        tmp_path, tmp_path / "actual.json", tmp_path / "inputs", tmp_path / "protocol.json"
    )
    assert result == {"rows": 24}
    assert captured["kwargs"]["readiness_verifier"] is I.READINESS.verify_model_readiness
    assert captured["kwargs"]["input_verifier"] is I.verify_inputs


def test_cli_exposes_verify_and_materialize() -> None:
    with pytest.raises(SystemExit) as captured:
        I.main(["--help"])
    assert captured.value.code == 0
