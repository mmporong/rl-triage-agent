from __future__ import annotations

import json
import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "bench"))
import run_recovery_reset_v2 as H  # noqa: E402
import train_with_recovery_reset_v2 as CHILD  # noqa: E402


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout.strip()


def repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / "bound.txt").write_text("bound\n", encoding="utf-8")
    git(root, "add", "bound.txt")
    git(root, "commit", "-m", "anchor")
    return root, git(root, "rev-parse", "HEAD")


def test_source_anchor_allows_unrelated_descendant_commit(tmp_path):
    root, anchor_commit = repo(tmp_path)
    anchor = H.make_source_anchor(anchor_commit, root, ("bound.txt",))
    (root / "unrelated.txt").write_text("new\n", encoding="utf-8")
    git(root, "add", "unrelated.txt")
    git(root, "commit", "-m", "unrelated")
    assert H.verify_source_anchor(anchor, root, ("bound.txt",)) == anchor


@pytest.mark.parametrize("kind", ["working", "staged", "committed"])
def test_source_anchor_rejects_bound_file_changes(tmp_path, kind):
    root, anchor_commit = repo(tmp_path)
    anchor = H.make_source_anchor(anchor_commit, root, ("bound.txt",))
    (root / "bound.txt").write_text("changed\n", encoding="utf-8")
    if kind in {"staged", "committed"}:
        git(root, "add", "bound.txt")
    if kind == "committed":
        git(root, "commit", "-m", "bound changed")
    with pytest.raises(ValueError, match="결합 파일"):
        H.verify_source_anchor(anchor, root, ("bound.txt",))


def test_source_anchor_rejects_nonancestor_and_missing_commit(tmp_path):
    root, anchor_commit = repo(tmp_path)
    anchor = H.make_source_anchor(anchor_commit, root, ("bound.txt",))
    git(root, "switch", "--orphan", "other")
    (root / "bound.txt").write_text("bound\n", encoding="utf-8")
    git(root, "add", "bound.txt")
    git(root, "commit", "-m", "unrelated root")
    with pytest.raises(ValueError, match="조상이 아니다"):
        H.verify_source_anchor(anchor, root, ("bound.txt",))
    missing = {**anchor, "commit": "f" * 40}
    missing["bundle_sha256"] = H._bundle_sha(missing["commit"], missing["git_blobs"])
    with pytest.raises(ValueError, match="없거나"):
        H.verify_source_anchor(missing, root, ("bound.txt",))


def test_child_dispatches_v2_harness_and_preserves_intervention_hook(tmp_path, monkeypatch):
    output = tmp_path / "train"
    output.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    manifest = {"arm": "optimizer_only", "seed": 2026, "output_dir": str(output)}
    monkeypatch.setattr(CHILD.H, "read", lambda path: manifest if Path(path) == manifest_path
                        else json.loads(Path(path).read_text(encoding="utf-8")))
    monkeypatch.setattr(CHILD.H, "intervention_for_arm", lambda arm: arm)
    monkeypatch.setattr(CHILD.H, "reset_seed", lambda seed, arm: 123)
    seen = {}

    def execute(path, helper):
        seen.update(path=path, helper=helper)
        (output / "intervention.json").write_text('{"status":"applied"}', encoding="utf-8")
        return {"completed": True}

    monkeypatch.setattr(CHILD.R, "execute", execute)
    result = CHILD.execute(manifest_path)
    assert result == {"completed": True}
    assert seen == {"path": manifest_path, "helper": CHILD.H}


def test_verify_training_dispatches_v2_validator_without_recursion(tmp_path, monkeypatch):
    private = tmp_path / "private"
    folder = private / "tag"
    manifest_path = folder / "optimizer_only/train/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    anchor = {"schema": "recovery_reset_source_anchor_v2", "commit": "a" * 40,
              "git_blobs": {"x": "b" * 40}}
    anchor["bundle_sha256"] = H._bundle_sha(anchor["commit"], anchor["git_blobs"])
    manifest_path.write_text(json.dumps({"tag": "tag", "source_commit_sha": anchor["commit"]}), encoding="utf-8")
    (folder / "request.json").write_text(json.dumps({
        "source_anchor": anchor, "source_commit_sha": anchor["commit"],
        "protocol_sha256_lf": H.sha(protocol, True),
    }), encoding="utf-8")
    monkeypatch.setattr(H, "PRIVATE", private)
    monkeypatch.setattr(H, "PROTOCOL", protocol)
    monkeypatch.setattr(H, "verify_source_anchor", lambda value: value)
    calls = []
    monkeypatch.setattr(H, "_V1_VALIDATE_MANIFEST", lambda path: (calls.append(path) or ({"ok": True}, {})))
    monkeypatch.setattr(H, "_V1_VERIFY_TRAINING",
                        lambda _folder, _arm, _execution: H.V1.validate_manifest(manifest_path)[0])
    assert H.verify_training(folder, "optimizer_only", {}) == {"ok": True}
    assert calls == [manifest_path]


def test_v2_consumption_is_new_write_once_and_v1_sentinel_is_preserved(tmp_path, monkeypatch):
    private = tmp_path / "v2"
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    isaac = tmp_path / "IsaacLab"
    (isaac / "logs/rsl_rl/unitree_go2_flat").mkdir(parents=True)
    old = tmp_path / "v1-consumed.json"
    old.write_text("preserve", encoding="utf-8")
    anchor = {"schema": "recovery_reset_source_anchor_v2", "commit": "a" * 40,
              "git_blobs": {"x": "b" * 40}}
    anchor["bundle_sha256"] = H._bundle_sha(anchor["commit"], anchor["git_blobs"])
    monkeypatch.setattr(H, "PRIVATE", private)
    monkeypatch.setattr(H, "PROTOCOL", protocol)
    monkeypatch.setattr(H, "verify_protocol", lambda _lab: ({}, "a" * 40))
    monkeypatch.setattr(H, "make_source_anchor", lambda _commit: anchor)
    monkeypatch.setattr(H, "verify_source_anchor", lambda value: value)
    monkeypatch.setattr(H.M, "query_gpu", lambda profile: {
        "busy": False, "processes": [], "admission_policy": profile,
    })
    monkeypatch.setattr(H, "manifest_for", lambda folder, request, arm: {"arm": arm})
    monkeypatch.setattr(H.P, "run_owned", lambda *_args: (_ for _ in ()).throw(RuntimeError("stop")))
    monkeypatch.setattr(H, "summarize", lambda seed, reports, failed: {
        "status": "partial", "seed": seed, "scope": H.FIXED["scope"], "labels": {},
    })
    result = H.execute("v2_s2026", 2026, isaac)
    assert result["status"] == "partial"
    assert old.read_text(encoding="utf-8") == "preserve"
    consumed = list((private / "consumed").glob("*.json"))
    assert len(consumed) == 1
    with pytest.raises(FileExistsError, match="already been consumed"):
        H.execute("v2_s2026_second", 2026, isaac)
    assert len(list((private / "consumed").glob("*.json"))) == 1


def result_row(seed: int, commit: str, blobs: dict[str, str]) -> dict:
    anchor = {"schema": "recovery_reset_source_anchor_v2", "commit": commit,
              "git_blobs": blobs, "bundle_sha256": H._bundle_sha(commit, blobs)}
    return {"seed": seed, "source_anchor": anchor, "protocol_sha256_lf": "p" * 64,
            "labels": {arm: "not_recovered" for arm in H.ARMS}}


def test_aggregate_allows_distinct_commits_with_same_bound_blobs_and_rejects_drift(monkeypatch):
    rows = {
        "a": result_row(2026, "a" * 40, {"bound": "1" * 40}),
        "b": result_row(2028, "b" * 40, {"bound": "1" * 40}),
    }
    monkeypatch.setattr(H, "_validate_result_path", lambda path: (rows[path.name], {"path": path.name, "sha256": "x" * 64}))
    complete = H.aggregate([Path("a"), Path("b")])
    assert complete["status"] == "complete"
    assert set(complete["source_anchors"]) == {"2026", "2028"}
    rows["b"] = result_row(2028, "b" * 40, {"bound": "2" * 40})
    with pytest.raises(ValueError, match="bound source blobs differ"):
        H.aggregate([Path("a"), Path("b")])


def test_fixed_v2_contract_records_equal_budget_and_invalid_scope():
    assert H.FIXED["seeds"] == [2026, 2028]
    assert H.FIXED["arms"] == list(H.ARMS)
    assert H.FIXED["num_envs"] == 4096 and H.FIXED["max_iterations"] == 300
    assert H.FIXED["semantic_retries"] == 0
    assert isinstance(H.FIXED["v1_invalid_reason"], str) and H.FIXED["v1_invalid_reason"]
    assert isinstance(H.FIXED["selection_limit"], str) and H.FIXED["selection_limit"]


def test_v2_exposes_every_helper_attribute_used_by_training_runtime():
    tree = ast.parse((ROOT / "bench/train_with_recovery_mode.py").read_text(encoding="utf-8"))
    required = {node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "H"}
    assert required == {"sha", "validate_training_configuration", "write_new",
                        "validate_manifest", "validate_readback"}
    assert all(callable(getattr(H, name, None)) for name in required)


def test_prepared_protocol_binds_v2_sources_and_original_invalid_evidence():
    protocol = H.read(H.PROTOCOL)
    assert protocol["schema"] == "recovery_reset_protocol_v2"
    assert protocol["v1_invalid_evidence"] == H.invalid_evidence(Path.home() / "IsaacLab")
    checkpoint = protocol["v1_invalid_evidence"]["completed_checkpoint"]
    assert checkpoint["path"].startswith("<isaaclab>/logs/")
    H._reject_public_absolute_paths(protocol)
    assert protocol["source_files"]["evals/run_recovery_reset_v2.py"] == H.sha(
        H.ROOT / "evals/run_recovery_reset_v2.py", True)
    assert protocol["source_files"]["bench/train_with_recovery_reset_v2.py"] == H.sha(
        H.ROOT / "bench/train_with_recovery_reset_v2.py", True)


def test_gpu_contract_requires_idle_before_but_allows_external_busy_after():
    base = {"busy": False, "admission_policy": H.FIXED["gpu_admission_profile"], "processes": []}
    assert H.validate_gpu_receipt(base, require_idle=True) == base
    external_after = {**base, "busy": True, "processes": [{"pid": 999}]}
    assert H.validate_gpu_receipt(external_after, require_idle=False) == external_after
    with pytest.raises(ValueError, match="busy"):
        H.validate_gpu_receipt(external_after, require_idle=True)


@pytest.mark.parametrize("absolute", [
    r"C:\Users\fake\IsaacLab\model.pt",
    "C:/Users/fake/IsaacLab/model.pt",
    "/home/fake/IsaacLab/model.pt",
    "/Users/fake/IsaacLab/model.pt",
    "/mnt/c/Users/fake/IsaacLab/model.pt",
    r"\\server\share\model.pt",
    "//server/share/model.pt",
])
def test_public_protocol_path_guard_rejects_nested_absolute_paths(absolute):
    with pytest.raises(ValueError, match="절대경로"):
        H._reject_public_absolute_paths({"outer": [{"path": absolute}]})


@pytest.mark.parametrize("portable", [
    "<isaaclab>/logs/rsl_rl/run/model.pt",
    "bench/private/evidence.json",
])
def test_public_protocol_path_guard_allows_markers_and_repo_relative_paths(portable):
    H._reject_public_absolute_paths({"outer": [{"path": portable}]})
