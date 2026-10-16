import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import build_additional_checks_replay_v2 as R
import validate_scope_transfer_media_public as M


def protocol():
    return json.loads((ROOT / R.PROTOCOL).read_text(encoding="utf-8"))


@pytest.mark.parametrize("field,value", [
    ("scenes", []), ("validation", {}), ("scope", "changed"), ("model_calls", 1), ("gpu_calls", 1),
])
def test_v2_fixed_contract_rejects_changes(field, value):
    row = protocol()
    row[field] = value
    with pytest.raises(ValueError):
        R.validate_protocol(row)


@pytest.mark.parametrize("field,value", [
    ("folder", "docs/media/other"), ("task", "other-task"), ("protocol_path", "bench/protocols/other.json"),
])
def test_v2_case_bindings_reject_changes(field, value):
    row = protocol()
    row["media"]["u1"][field] = value
    with pytest.raises(ValueError):
        R.validate_protocol(row)


def test_actual_media_is_fully_verified_before_commit():
    result = M.validate(require_committed=False)
    assert result["status"] in {"precommit_verified", "ready"}
    assert set(result["cases"]) == {"u1", "anymal"}
    assert all(row["file_count"] == 6 for row in result["cases"].values())
    assert result["cases"]["u1"]["ffprobe"]["comparison.mp4"] == {
        "width": 1920, "height": 360, "frame_rate": "50/1", "frame_count": 299, "duration_s": 5.98,
    }
    assert result["cases"]["anymal"]["ffprobe"]["comparison.gif"] == {
        "width": 960, "height": 180, "frame_rate": "50/3", "frame_count": 90, "duration_s": 6.0,
    }


def test_recording_protocol_binds_policy_checkpoint_and_public_sources():
    row = protocol()
    spec = row["media"]["u1"]
    manifest = json.loads((ROOT / spec["folder"] / "source_manifest.json").read_text(encoding="utf-8"))
    recording = json.loads((ROOT / spec["protocol_path"]).read_text(encoding="utf-8"))
    M.validate_manifest("u1", spec, manifest, recording)
    changed = copy.deepcopy(recording)
    changed["clips"][0]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="policy_checkpoint_binding"):
        M.validate_manifest("u1", spec, manifest, changed)
    changed = copy.deepcopy(recording)
    changed["public_sources"].pop()
    with pytest.raises(ValueError, match="public_source_inventory"):
        M.validate_manifest("u1", spec, manifest, changed)


@pytest.mark.parametrize("mutation", ["case", "task", "clip_role", "clip_sha", "comparison_sha", "private_path"])
def test_manifest_cross_binding_rejects_mutations(mutation):
    row = protocol()
    spec = row["media"]["u1"]
    manifest = json.loads((ROOT / spec["folder"] / "source_manifest.json").read_text(encoding="utf-8"))
    manifest = copy.deepcopy(manifest)
    if mutation == "case":
        manifest["case_id"] = "anymal"
    elif mutation == "task":
        manifest["playback"]["task"] = "wrong"
    elif mutation == "clip_role":
        manifest["clips"][0]["role"] = "none_control"
    elif mutation == "clip_sha":
        manifest["clips"][0]["sha256"] = "0" * 64
    elif mutation == "comparison_sha":
        manifest["comparison"]["mp4"]["sha256"] = "0" * 64
    else:
        manifest["private_sources"][0]["path"] = "C:\\private\\file.json"
    with pytest.raises(ValueError):
        M.validate_manifest("u1", spec, manifest)


def test_replay_adds_both_media_blocks_and_preserves_old_numbers(monkeypatch):
    monkeypatch.setattr(R.B.Sources, "_committed_state", lambda *args: ("ready", None))
    monkeypatch.setattr(R.M, "validate", lambda *args, **kwargs: {"status": "ready"})
    data = R.build()
    assert [scene["id"] for scene in data["scenes"]] == [
        "u1-expanded-scope", "optimizer-update-observation", "anymal-c-transfer-playback",
    ]
    assert data["scenes"][0]["blocks"][1]["rows"][0][1]["value"] == pytest.approx(1.2231621716230434)
    assert data["scenes"][0]["blocks"][-1]["type"] == "video"
    assert data["scenes"][2]["blocks"][0]["type"] == "video"
    assert data["sources"]["docs/media/scope-transfer-u1-s2051-v1/comparison.mp4"]["status"] == "ready"
    assert data["sources"]["docs/media/scope-transfer-anymal-s2047-v1/comparison.mp4"]["status"] == "ready"
    for name in ("anymal_c_h02_s2047.json", "anymal_c_none_s2047.json"):
        source = data["sources"][f"evals/results/anymal_c_e2e_f1_s2047_20261014/runs/{name}"]
        assert source["status"] == "ready"
        assert source["sha256_lf"] in {
            "852472699e762bb31f61af9c0d88498f81e3291ce67ac389123fc3085f17eb80",
            "60b14311fe7f56f47ce3bf3b4a4f6c484c2fdbb6999166e001f4ca3384d765d8",
        }


def test_lf_manifest_hash_accepts_committed_crlf_file(tmp_path, monkeypatch):
    relative = "public/result.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{\r\n  "status": "ready"\r\n}\r\n')
    expected_lf = M.sha256_lf(path)
    sources = R.B.Sources(tmp_path)
    monkeypatch.setattr(sources, "_committed_state", lambda *args: ("ready", None))
    source_id = R.add_lf_source(sources, tmp_path, {"path": relative, "sha256_lf": expected_lf})
    row = sources.rows[source_id]
    assert row["status"] == "ready"
    assert row["sha256_lf"] == expected_lf
    assert row["sha256"] != expected_lf


@pytest.mark.parametrize("tracked,dirty", [(False, False), (True, True)])
def test_strict_ready_rejects_untracked_or_dirty_v2_protocol(tmp_path, monkeypatch, tracked, dirty):
    class Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode, self.stdout = returncode, stdout

    def fake_git(root, *args):
        if args[0] == "ls-files":
            return Result(0 if tracked else 1)
        return Result(0, " M protocol.json\n" if dirty else "")

    monkeypatch.setattr(M, "git", fake_git)
    with pytest.raises(ValueError, match="not_committed_clean"):
        M.require_committed_clean(tmp_path, "protocol.json")


def test_git_status_failure_is_fail_closed(tmp_path, monkeypatch):
    class Result:
        def __init__(self, returncode=0, stdout=""):
            self.returncode, self.stdout = returncode, stdout

    def fake_git(root, *args):
        return Result(0) if args[0] == "ls-files" else Result(128)

    monkeypatch.setattr(M, "git", fake_git)
    with pytest.raises(ValueError, match="git_status_failed"):
        M.require_committed_clean(tmp_path, "protocol.json")


def test_precommit_build_does_not_call_untracked_media_ready(monkeypatch):
    media_prefixes = tuple(
        f"{spec['folder']}/" for spec in protocol()["media"].values()
    )

    def simulated_commit_state(self, relative):
        if relative.startswith(media_prefixes):
            return "waiting", "not_committed"
        return "ready", None

    monkeypatch.setattr(R.B.Sources, "_committed_state", simulated_commit_state)
    data = R.build()
    for folder in ("scope-transfer-u1-s2051-v1", "scope-transfer-anymal-s2047-v1"):
        row = data["sources"][f"docs/media/{folder}/comparison.mp4"]
        assert row["status"] == "waiting"
        assert row["reason"] == "not_committed"


def test_v2_snapshot_and_html_entrypoint_are_current():
    assert R.B.render(R.build()) == (ROOT / R.OUTPUT).read_text(encoding="utf-8")
    html = (ROOT / "docs/replay/additional-checks.html").read_text(encoding="utf-8")
    assert '<script src="additional-checks-v2-data.js"></script>' in html
    assert '<body class="additional-checks-v2">' in html
    style = (ROOT / "docs/replay/style.css").read_text(encoding="utf-8")
    assert ".additional-checks-v2 .videos.single-video" in style
    assert "\n.videos.single-video" not in style
