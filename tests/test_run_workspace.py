import hashlib
import json
from pathlib import Path

import pytest

from rl_triage import cli
from rl_triage import run_workspace as W
from rl_triage import single_run as S
from rl_triage import triage_tools as T


def _write(path: Path, value: object) -> bytes:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _telemetry():
    return {
        "run_dir_name": "/private/run",
        "answer_key": "must-not-leak",
        "summary": {
            "Train/mean_reward": {
                "n": 3, "first": -1.0, "last": None, "min": -1.0, "max": 2.0,
                "mean_first_20pct": -1.0, "internal_label": "remove",
            },
        },
        "series": {"Train/mean_reward": [-1.0, 2.0, 0.0]},
    }


def test_prepare_inspect_report_and_existing_single_run_consumer(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.json"
    source_raw = _write(source, _telemetry())
    configuration = tmp_path / "configuration.json"
    configuration_raw = _write(configuration, {
        "rewards": {"tracking": {"weight": 10}},
        "observations": {"enabled": None},
    })
    facts = tmp_path / "facts.json"
    _write(facts, [
        {"id": "tracking_weight", "path": ["rewards", "tracking", "weight"]},
        {"id": "observations_enabled", "path": ["observations", "enabled"]},
    ])
    workspace = tmp_path / "prepared"

    assert cli.main([
        "prepare-run", "--telemetry", str(source), "--workspace", str(workspace),
        "--case", "trial-1", "--configuration", str(configuration), "--facts", str(facts),
    ]) == 0
    prepared_output = json.loads(capsys.readouterr().out)
    assert prepared_output["source_telemetry_sha256"] == hashlib.sha256(source_raw).hexdigest()
    prepared = json.loads((workspace / "cases/trial-1/telemetry.json").read_text())
    assert set(prepared) == {"summary", "series"}
    assert set(prepared["summary"]["Train/mean_reward"]) <= set(W.G.SUMMARY_FIELDS)
    assert "answer_key" not in json.dumps(prepared)
    assert (workspace / "cases/trial-1/configuration.json").read_bytes() == configuration_raw

    report = tmp_path / "report.json"
    assert cli.main([
        "inspect-run", "trial-1", "--workspace", str(workspace), "--report", str(report),
    ]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["status"] == "ready_for_assessment"
    assert inspected["model_evaluation"] == "not_run"
    assert inspected["reference_status"] == "absent"
    assert inspected["configuration"]["facts"][0]["value"] == 10
    assert inspected["evidence_state_counts"] == {"present": 6, "null": 2, "missing": 1}
    assert report.is_file()
    assert "/private/run" not in report.read_text()

    previous = T.WORKSPACE
    monkeypatch.setattr(T, "WORKSPACE", workspace)
    try:
        overview = S.telemetry_overview("trial-1")
        context = S.get_context("trial-1")
    finally:
        monkeypatch.setattr(T, "WORKSPACE", previous)
    assert overview["rows"][0]["last"] is None
    assert context["configuration_sha256"] == hashlib.sha256(configuration_raw).hexdigest()


@pytest.mark.parametrize("content, message", [
    ('{"summary":{},"summary":{},"series":{}}', "중복 JSON 키"),
    ('{"summary":{},"series":{"x":[NaN]}}', "유한하지 않은"),
    ('{"summary":[],"series":{}}', "summary와 series"),
])
def test_bad_telemetry_fails_before_target_creation(tmp_path, capsys, content, message):
    source = tmp_path / "source.json"
    source.write_text(content, encoding="utf-8")
    workspace = tmp_path / "prepared"
    assert cli.main([
        "prepare-run", "--telemetry", str(source), "--workspace", str(workspace), "--case", "one",
    ]) == 2
    assert message in json.loads(capsys.readouterr().out)["error"]
    assert not workspace.exists()


def test_empty_telemetry_is_not_ready_for_assessment(tmp_path):
    source = tmp_path / "source.json"
    _write(source, {"summary": {}, "series": {}})
    workspace = tmp_path / "prepared"
    with pytest.raises(ValueError, match="하나 이상의 관측 지표"):
        W.prepare_run(source, workspace, "one")
    assert not workspace.exists()


def test_new_command_usage_error_is_json_with_exit_two(capsys):
    assert cli.main(["prepare-run", "--telemetry", "input.json"]) == 2
    error = json.loads(capsys.readouterr().out)
    assert "required" in error["error"]


@pytest.mark.parametrize("case_id", ["../escape", "a/b", "..", ""])
def test_case_escape_is_rejected_without_target(tmp_path, case_id):
    source = tmp_path / "source.json"
    _write(source, _telemetry())
    workspace = tmp_path / "prepared"
    with pytest.raises(ValueError, match="case id"):
        W.prepare_run(source, workspace, case_id)
    assert not workspace.exists()


@pytest.mark.parametrize("facts_value, message", [
    ([], "비어 있지 않은"),
    ([{"id": "bad/id", "path": ["rewards", "x"]}], "fact id"),
    ([{"id": "missing", "path": ["rewards", "missing"]}], "없는 fact path"),
    ([{"id": "unsafe_root", "path": ["private", "value"]}], "허용되지 않은"),
])
def test_invalid_facts_are_rejected_before_publish(tmp_path, facts_value, message):
    source, configuration, facts = (tmp_path / name for name in (
        "source.json", "configuration.json", "facts.json",
    ))
    _write(source, _telemetry())
    _write(configuration, {"rewards": {"x": 1}, "private": {"value": 2}})
    _write(facts, facts_value)
    workspace = tmp_path / "prepared"
    with pytest.raises(ValueError, match=message):
        W.prepare_run(source, workspace, "one", configuration, facts)
    assert not workspace.exists()


def test_configuration_and_facts_must_be_paired_and_input_symlink_rejected(tmp_path):
    source = tmp_path / "source.json"
    config = tmp_path / "configuration.json"
    _write(source, _telemetry())
    _write(config, {"rewards": {"x": 1}})
    with pytest.raises(ValueError, match="함께 제공"):
        W.prepare_run(source, tmp_path / "one", "one", config, None)
    link = tmp_path / "source-link.json"
    link.symlink_to(source)
    with pytest.raises(PermissionError, match="심볼릭 링크"):
        W.prepare_run(link, tmp_path / "two", "one")


def test_existing_workspace_and_report_are_not_changed(tmp_path):
    source = tmp_path / "source.json"
    _write(source, _telemetry())
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        W.prepare_run(source, existing, "one")
    assert marker.read_text() == "keep"

    workspace = tmp_path / "prepared"
    W.prepare_run(source, workspace, "one")
    report = tmp_path / "report.json"
    report.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        W.inspect_run("one", workspace, report)
    assert report.read_text() == "keep"


@pytest.mark.parametrize("target", ["telemetry", "manifest", "reference"])
def test_inspect_detects_imported_file_and_manifest_tampering(tmp_path, target):
    source = tmp_path / "source.json"
    _write(source, _telemetry())
    workspace = tmp_path / "prepared"
    W.prepare_run(source, workspace, "one")
    if target == "telemetry":
        path = workspace / "cases/one/telemetry.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif target == "reference":
        _write(workspace / "reference/telemetry.json", {
            "reference_status": "absent", "summary": {"x": {}}, "series": {},
        })
    else:
        manifest = json.loads((workspace / "import.json").read_text())
        manifest["files"]["../escape.json"] = "0" * 64
        _write(workspace / "import.json", manifest)
    with pytest.raises((ValueError, PermissionError), match="manifest|SHA-256|파일 목록"):
        W.inspect_run("one", workspace)


def test_inspect_detects_configuration_context_and_hash_tampering(tmp_path):
    source, configuration, facts = (tmp_path / name for name in (
        "source.json", "configuration.json", "facts.json",
    ))
    _write(source, _telemetry())
    _write(configuration, {"rewards": {"x": 1}})
    _write(facts, [{"id": "x", "path": ["rewards", "x"]}])
    for relative in (
        "cases/one/configuration.json", "cases/one/context.json", "import.json",
    ):
        workspace = tmp_path / relative.replace("/", "-").replace(".", "-")
        W.prepare_run(source, workspace, "one", configuration, facts)
        path = workspace / relative
        if relative == "import.json":
            manifest = json.loads(path.read_text())
            manifest["schema"] = "tampered"
            _write(path, manifest)
        else:
            path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValueError):
            W.inspect_run("one", workspace)


def test_inspect_rechecks_files_before_publishing_result(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    _write(source, _telemetry())
    workspace = tmp_path / "prepared"
    W.prepare_run(source, workspace, "one")
    original = W.G.evidence_catalog

    def mutate_after_catalog(case_id):
        result = original(case_id)
        telemetry = workspace / "cases/one/telemetry.json"
        telemetry.write_bytes(telemetry.read_bytes() + b" ")
        return result

    monkeypatch.setattr(W.G, "evidence_catalog", mutate_after_catalog)
    with pytest.raises(ValueError, match="SHA-256"):
        W.inspect_run("one", workspace)


def test_workspace_outputs_do_not_invalidate_imported_inputs(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    _write(source, _telemetry())
    workspace = tmp_path / "prepared"
    W.prepare_run(source, workspace, "one")

    report = workspace / "report.json"
    first = W.inspect_run("one", workspace, report)
    assert first["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    monkeypatch.setattr(T, "WORKSPACE", workspace)
    assessment = S.write_assessment(
        "one", ["reward dynamics need another measurement"],
        "imported telemetry only", "no reference run", "measure contact force",
    )
    assert assessment["status"] == "hypotheses_only"
    (workspace / "scratch").mkdir()
    (workspace / "scratch" / "analysis.txt").write_text("output", encoding="utf-8")

    second = W.inspect_run("one", workspace)
    assert second["status"] == "ready_for_assessment"
    assert second["import_sha256"] == first["import_sha256"]


def test_report_cannot_be_created_inside_input_tree(tmp_path):
    source = tmp_path / "source.json"
    _write(source, _telemetry())
    workspace = tmp_path / "prepared"
    W.prepare_run(source, workspace, "one")
    report = workspace / "cases/one/report.json"
    with pytest.raises(ValueError, match="입력 트리"):
        W.inspect_run("one", workspace, report)
    assert not report.exists()


def test_unexpected_file_inside_input_tree_is_rejected(tmp_path):
    source = tmp_path / "source.json"
    _write(source, _telemetry())
    workspace = tmp_path / "prepared"
    W.prepare_run(source, workspace, "one")
    (workspace / "cases/one/unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="실제 파일 목록"):
        W.inspect_run("one", workspace)
