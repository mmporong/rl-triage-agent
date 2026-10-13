"""N3: 저장된 E4b·시연 출력의 NAT eval 결정적 재채점. 기존 검증기만 감싸고 저장 요약과 같아야 한다."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import nat_eval_rescore as N  # noqa: E402

PRIVATE = all((ROOT / "bench/private" / name).is_dir() for name in ("e4b", "e4b_v2", "e4b_v3", "e2e_reward_demo"))
needs_private = pytest.mark.skipif(not PRIVATE, reason="비공개 E4b·시연 기록이 없다")
NAT_RUN = {"type": "nat_run", "version": "v1", "set": "synthetic", "kind": "diagnosis", "packet": "none",
           "packet_sha256": "0" * 64, "attempt": None, "record": None}


def _row(ident: str, meta: dict, generated: dict, expected: dict) -> dict:
    return {"id": ident, "question": f"synthetic {ident}", "answer": N.dumps(expected),
            "generated_answer": N.dumps(generated), "item": N.dumps({"schema": N.SCHEMA, **meta})}


def _check(rows: list[dict], summaries: dict | None = None) -> dict:
    scored = N.score_rows(rows)
    return N.compare(N.aggregate(rows, scored), scored, summaries)


def _registered_evaluator_types() -> set[str]:
    from nat.cli.type_registry import GlobalTypeRegistry
    return {info.config_type.static_type() for info in GlobalTypeRegistry.get().get_registered_evaluators()}


def _synthetic_dataset(tmp_path: Path) -> Path:
    rows = [_row("missing", {**NAT_RUN, "run": 1}, {"missing": True}, {"status": "missing", "decision": None}),
            _row("wrong", {**NAT_RUN, "run": 2}, {"missing": True}, {"status": "valid", "decision": None}),
            {**_row("schema", {**NAT_RUN, "run": 3}, {"missing": True}, {}), "item": json.dumps({"schema": "other"})}]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return dataset


def test_config_declares_only_the_rescore_evaluator():
    config = yaml.safe_load(N.CONFIG.read_text(encoding="utf-8"))
    assert set(config) == {"eval"}  # llms·functions·workflow가 없으면 모델을 만들 수 없다
    assert config["eval"]["evaluators"] == {N.EVALUATOR_NAME: {"_type": N.EVALUATOR_TYPE}}


def _registry_snapshot() -> dict[str, set]:
    from nat.cli.type_registry import GlobalTypeRegistry
    return {name: set(value) for name, value in vars(GlobalTypeRegistry.get()).items() if isinstance(value, dict)}


def test_nat_eval_skip_workflow_runs_the_registered_evaluator_without_leaking_it(tmp_path):
    from nat.data_models.config import Config
    from nat.runtime.loader import PluginTypes, discover_and_register_plugins
    import nat.plugins.eval.cli.evaluate  # noqa: F401
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)
    Config.model_validate({})  # 다른 NAT 테스트처럼 설정 스키마를 먼저 쓴 상태에서도 등록이 반영돼야 한다
    before = _registry_snapshot()
    for attempt in ("first", "second"):  # push가 끝난 뒤 다시 등록해도 같은 결과
        scored = N.scored_from_nat_output(N.run_nat_eval(_synthetic_dataset(tmp_path), tmp_path / attempt))
        assert set(scored) == {"missing", "wrong", "schema"}
        assert scored["missing"]["score"] == 1.0 and scored["wrong"]["score"] == 0.0
        assert scored["schema"]["score"] == 0.0 and scored["schema"].get("error")  # NAT가 0점 처리한 오류도 드러난다
        with pytest.raises(ValueError, match="evaluator 오류"):
            N.require_no_evaluator_errors(scored)
        assert N.EVALUATOR_TYPE not in _registered_evaluator_types()
        assert _registry_snapshot() == before  # push 안에서 다른 등록이 생기거나 사라지지 않았다
        with pytest.raises(ValueError):  # 레지스트리와 함께 Config 스키마도 원래대로 돌아간다
            Config.model_validate({"eval": {"evaluators": {"x": {"_type": N.EVALUATOR_TYPE}}}})


def test_infra_errors_are_taken_from_the_bound_record(monkeypatch):
    bind = N._bind_nat_run
    monkeypatch.setattr(N, "_bind_nat_run", lambda meta, generated: None)
    generated = {"missing": False, "record_status": "infra_error", "submission": None, "rejected": False}
    result = N.score_entry({**NAT_RUN, "run": 1}, generated)
    assert result == {"comparable": {"status": "infra_error", "decision": None}, "status_source": "record"}
    with pytest.raises(ValueError, match="기록 없이"):
        bind({**NAT_RUN, "run": 1}, generated)


def test_missing_bound_run_files_are_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "run_path", lambda version, set_name, kind, run, attempt: tmp_path / f"run{run}_a{attempt}.json")
    inputs = {"diagnosis": {"packet": "none", "packet_sha256": "a" * 64}}
    record = tmp_path / "run1_a1.json"
    record.write_text(json.dumps({"packet_sha256": "a" * 64, "status": "invalid", "rejection_sha256": "b" * 64}),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="거절 기록 결합"):
        N.nat_run_entry("v1", "synthetic", "diagnosis", 1, inputs)
    record.write_text(json.dumps({"packet_sha256": "a" * 64, "status": "valid", "submission_sha256": "c" * 64}),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="제출 파일이 없다"):
        N.nat_run_entry("v1", "synthetic", "diagnosis", 1, inputs)


@needs_private
def test_rescore_reproduces_stored_summaries_and_demo_results():
    rows = N.build_dataset()
    scored = N.score_rows(rows)
    result = N.compare(N.aggregate(rows, scored), scored)
    assert result["pass"], result
    assert (result["items"], result["nat_run_items"], result["demo_items"]) == (67, 55, 4)
    statuses = Counter(item["rescored"]["comparable"]["status"] for item in scored.values() if "status" in item["expected"])
    assert statuses["valid"] > 0 and statuses["invalid"] > 0  # 유효·무효를 모두 다시 판정했다


@needs_private
def test_nat_eval_path_matches_the_direct_scoring_path(tmp_path):
    rows = N.build_dataset()
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    output = N.run_nat_eval(dataset, tmp_path / "out")
    via_nat = N.scored_from_nat_output(output)
    N.require_no_evaluator_errors(via_nat)
    direct = N.score_rows(rows)
    assert json.loads(output.read_text(encoding="utf-8"))["average_score"] == 1.0
    assert {k: v["score"] for k, v in via_nat.items()} == {k: v["score"] for k, v in direct.items()}
    assert N.compare(N.aggregate(rows, via_nat), via_nat)["pass"]


@needs_private
def test_changed_stored_values_are_reported_as_differences():
    rows = N.build_dataset()
    changed = copy.deepcopy(N.stored_results()[0])
    changed["v3"]["sets"]["h02_s2027"]["kinds"]["correction"]["counts"]["valid"] += 1
    changed["v2"]["adoption"]["decision"] = "adopt"
    first_valid = next(r for r in changed["v2"]["sets"]["s2030"]["kinds"]["diagnosis"]["runs"] if r["status"] == "valid")
    first_valid["decision"] = {**first_valid["decision"], "probe": "P_other"}
    del changed["v3"]["sets"]["h05_s2027"]["kinds"]["correction"]
    differences = _check(rows, changed)["differences"]
    assert {"v3/h02_s2027/correction/counts", "v2/adoption", "v2/s2030/diagnosis/runs",
            "v3/h05_s2027/kind coverage"} <= set(differences)


@needs_private
def test_binding_failures_are_errors_not_judgments():
    rows = N.build_dataset()
    by_type = {}
    for row in rows:
        by_type.setdefault(json.loads(row["item"])["type"], []).append(row)
    valid_run = next(r for r in by_type["nat_run"] if json.loads(r["answer"]).get("status") == "valid")
    tampered = copy.deepcopy(valid_run)
    generated = json.loads(tampered["generated_answer"])
    generated["submission"]["kind"] = "other"
    tampered["generated_answer"] = N.dumps(generated)
    with pytest.raises(ValueError, match="기록 파일과 다르다"):
        N.score_row(tampered)
    other_packets = {json.loads(r["item"])["packet"] for r in rows}
    for kind_rows in (by_type["single_call"], by_type["demo_call"]):
        row = copy.deepcopy(kind_rows[0])
        meta = json.loads(row["item"])
        meta["packet"] = next(p for p in sorted(other_packets) if p != meta["packet"])  # SHA는 그대로 둔다
        row["item"] = N.dumps(meta)
        with pytest.raises(ValueError, match="packet"):
            N.score_row(row)


def test_publish_refuses_uncommitted_code_existing_folders_and_bad_tags(tmp_path):
    def uncommitted(_repo, _paths):
        raise subprocess.CalledProcessError(1, ["git", "ls-files"])
    with pytest.raises(ValueError, match="커밋되지"):
        N.command_publish("never-run", "nat_eval_rescore_v9", verify_committed=uncommitted, results=tmp_path)
    (tmp_path / "nat_eval_rescore_v9").mkdir()
    with pytest.raises(FileExistsError):
        N.command_publish("never-run", "nat_eval_rescore_v9", verify_committed=uncommitted, results=tmp_path)
    with pytest.raises(ValueError, match="tag"):
        N.command_publish("never-run", "../escape", verify_committed=uncommitted, results=tmp_path)
    assert not (N.PRIVATE / "never-run").exists()


def _rewrite(folder: Path, summary: dict) -> None:
    raw = (json.dumps(summary, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    (folder / "summary.json").write_bytes(raw)
    manifest = {"schema": N.PUBLIC_SCHEMA, "files": {"summary.json": hashlib.sha256(raw).hexdigest()}}
    (folder / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@needs_private
def test_public_summary_replays_from_public_files_only(tmp_path):
    rows = N.build_dataset()
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    output = N.run_nat_eval(dataset, tmp_path / "out")
    scored = N.scored_from_nat_output(output)
    result = {"path": "nat_eval_skip_workflow", "average_score": json.loads(output.read_text(encoding="utf-8"))["average_score"],
              **N.compare(N.aggregate(rows, scored), scored)}
    with pytest.raises(ValueError, match="공개하지 않는다"):
        N.public_summary({**result, "pass": False}, rows, dataset, "0" * 40)
    folder = tmp_path / "public"
    N.write_public(folder, N.public_summary(result, rows, dataset, "0" * 40))
    view = N.replay_public(folder)
    assert (view["items"], view["item_types"]) == (67, {"demo_call": 4, "nat_run": 55, "single_call": 8})
    assert view["adoption"] == {v: N.stored_results()[0][v]["adoption"]["decision"] for v in N.MODULES}
    text = (folder / "summary.json").read_text(encoding="utf-8")
    assert "bench/private" not in text and '"content"' not in text and view["dataset_sha256"] == N.sha_raw(dataset)

    summary = json.loads(text)
    (folder / "summary.json").write_text(text.replace('"items": 67', '"items": 68'), encoding="utf-8")
    with pytest.raises(ValueError, match="파일 결합"):
        N.replay_public(folder)
    changed = copy.deepcopy(summary)
    changed["reproduced"]["v2"]["adoption"] = "adopt"
    _rewrite(folder, changed)
    with pytest.raises(ValueError, match="판정 필드"):
        N.replay_public(folder)
    changed = copy.deepcopy(summary)
    changed["stored_results_sha256_lf"][N.SUMMARIES["v1"]] = "0" * 64
    _rewrite(folder, changed)
    with pytest.raises(ValueError, match="저장 결과가 바뀌었다"):
        N.replay_public(folder)
    _rewrite(folder, summary)
    (folder / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="파일 결합"):
        N.replay_public(folder)


@needs_private
def test_scoring_delegates_to_the_existing_validators(monkeypatch):
    rows = N.build_dataset()

    def refuse(*_args):
        raise ValueError("validator replaced")
    monkeypatch.setattr(N.W1, "validate_submission", refuse)
    result = _check(rows)
    assert not result["pass"] and any(d.endswith("/counts") for d in result["differences"])
