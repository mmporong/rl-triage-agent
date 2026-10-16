from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from evals import nemotron_bottleneck_vision_v2 as V


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_manifest(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    assets = {}
    jpeg = b"synthetic-jpeg"
    digest = hashlib.sha256(jpeg).hexdigest()
    for row in V._ROWS:
        case_id, clip, label, ratio, fall, checkpoint, source_sha, report, report_sha = row
        folder = tmp_path / case_id
        folder.mkdir()
        frames = []
        for index in range(V.FRAME_COUNT):
            name = f"frame_{index + 1:02d}.jpg"
            (folder / name).write_bytes(jpeg)
            frames.append({"file": name, "timestamp_s": index + .5, "sha256": digest, "bytes": len(jpeg)})
        assets[case_id] = {
            "source": {"logical_id": clip, "sha256": source_sha,
                       "media_manifest_logical_id": (Path(clip).parent / "source_manifest.json").as_posix(),
                       "media_manifest_sha256": V.MEDIA_MANIFEST_SHA[(Path(clip).parent / "source_manifest.json").as_posix()]},
            "label_binding": {"group": label, "standstill_ratio": ratio, "fall_rate": fall,
                              "checkpoint_sha256": checkpoint, "report_logical_id": report,
                              "report_sha256": report_sha},
            "frames": frames, "asset_dir": case_id,
        }
    folder = tmp_path / V.PREFLIGHT["id"]
    folder.mkdir()
    frames = []
    for index in range(V.FRAME_COUNT):
        name = f"frame_{index + 1:02d}.jpg"
        (folder / name).write_bytes(jpeg)
        frames.append({"file": name, "timestamp_s": index + .5, "sha256": digest, "bytes": len(jpeg)})
    assets[V.PREFLIGHT["id"]] = {
        "source": {"logical_id": V.PREFLIGHT["clip"], "sha256": V.PREFLIGHT["source_sha256"]},
        "frames": frames, "asset_dir": V.PREFLIGHT["id"],
    }
    graph = tmp_path / "learning_curve.jpg"
    graph.write_bytes(jpeg)
    return {"_asset_root": str(tmp_path), "assets": assets,
            "graph": {"file": graph.name, "sha256": digest, "bytes": len(jpeg),
                      "telemetry": [{"logical_id": p, "sha256": s} for _, p, s in V.GRAPH_TELEMETRY]}}


def test_inventory_is_exactly_18_unique_checkpoints_with_frozen_groups():
    assert len(V._ROWS) == 18
    assert len({row[5] for row in V._ROWS}) == 18
    assert sum(row[2] == "normal" for row in V._ROWS) == 7
    assert sum(row[2] == "failure" for row in V._ROWS) == 11
    assert all("recovery-c01" not in row[1] for row in V._ROWS)


def test_all_frozen_sources_match_repository_bytes():
    for row in V._ROWS:
        assert _sha(ROOT / row[1]) == row[6]
        assert _sha(ROOT / row[7]) == row[8]
    assert _sha(ROOT / V.PREFLIGHT["clip"]) == V.PREFLIGHT["source_sha256"]
    for _, logical_id, expected in V.GRAPH_TELEMETRY:
        assert _sha(ROOT / logical_id) == expected
    for logical_id, expected in V.MEDIA_MANIFEST_SHA.items():
        assert _sha(ROOT / logical_id) == expected


def test_source_and_label_semantics_match_manifest_and_reports():
    for row in V._ROWS:
        case_id, clip, _, ratio, fall, checkpoint, source_sha, report, _ = row
        binding = V._source_binding(ROOT, clip, source_sha, checkpoint)
        assert binding["media_manifest_sha256"] == V.MEDIA_MANIFEST_SHA[binding["media_manifest_logical_id"]]
        V._verify_label_report(ROOT / report, case_id, checkpoint, ratio, fall)


def test_build_cases_keeps_labels_out_of_prompts_and_binds_images(tmp_path):
    cases = V.build_cases(_fake_manifest(tmp_path))
    assert len(cases) == 18
    for case in cases:
        prompt = json.dumps(case["messages"], ensure_ascii=False)
        assert "standstill" not in prompt
        assert '"normal"' not in prompt
        assert '"failure"' not in prompt
        assert sum(item.get("type") == "image_url" for item in case["messages"][1]["content"]) == 6
        assert case["binding"]["messages_sha256"]
        assert case["request_params"]["response_format"]["type"] == "json_schema"


def test_build_cases_rejects_changed_label_or_duplicate_checkpoint(tmp_path, monkeypatch):
    manifest = _fake_manifest(tmp_path)
    manifest["assets"][V._ROWS[0][0]]["label_binding"]["group"] = "normal"
    with pytest.raises(ValueError, match="binding mismatch"):
        V.build_cases(manifest)

    manifest = _fake_manifest(tmp_path / "second")
    changed = list(V._ROWS[1])
    changed[5] = V._ROWS[0][5]
    monkeypatch.setattr(V, "_ROWS", (V._ROWS[0], tuple(changed), *V._ROWS[2:]))
    manifest["assets"][changed[0]]["label_binding"]["checkpoint_sha256"] = changed[5]
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        V.build_cases(manifest)


def test_preflight_is_outside_evaluation_inventory(tmp_path):
    manifest = _fake_manifest(tmp_path)
    case = V.build_preflight_cases(manifest)[0]
    assert case["id"] == "preflight_non_eval"
    assert case["packet"] == {"preflight": True}
    assert "label_binding" not in case["binding"]
    assert V.PREFLIGHT["clip"] not in {row[1] for row in V._ROWS}


def test_validate_response_separates_capability_and_label_alignment(tmp_path):
    cases = V.build_cases(_fake_manifest(tmp_path))
    normal = next(case for case in cases if case["packet"]["fixed_eval_group"] == "normal")
    failure = next(case for case in cases if case["packet"]["fixed_eval_group"] == "failure")
    observed_walking = '{"walking":"yes","in_place":"no","falling":"no"}'
    normal_result = V.validate_response(normal, observed_walking)
    failure_result = V.validate_response(failure, observed_walking)
    assert normal_result["valid"] is True and normal_result["metrics"]["label_alignment"] is True
    assert failure_result["valid"] is True and failure_result["metrics"]["label_alignment"] is False
    assert normal_result["policy_label"] == "normal" and normal_result["label_alignment"] is True
    assert failure_result["policy_label"] == "failed" and failure_result["label_alignment"] is False
    malformed = V.validate_response(normal, '{"walking":"yes","in_place":"no"}')
    assert malformed["valid"] is False
    assert malformed["metrics"] == {"capability_valid": False, "label_alignment": None}
    assert malformed["label_alignment"] is None


def test_public_case_omits_image_bytes_and_private_packet(tmp_path):
    case = V.build_cases(_fake_manifest(tmp_path))[0]
    public = V.public_case(case)
    encoded = json.dumps(public)
    assert "synthetic-jpeg" not in encoded
    assert set(public) == {"id", "bottleneck", "kind", "request_params", "binding"}
    assert "messages" not in public and "packet" not in public
    assert re.search(r"[A-Za-z]:[/\\\\]", encoded) is None
    assert "bench/private" not in encoded


def test_graph_cases_are_two_separate_image_reading_checks(tmp_path):
    cases = V.build_graph_cases(_fake_manifest(tmp_path))
    assert [case["id"] for case in cases] == ["graph_endpoint_order", "graph_fault_trend"]
    assert all(case["kind"] == "graph" for case in cases)
    assert V.validate_response(cases[0], '{"top_higher":"blue","bottom_lower":"blue"}')["valid"]
    assert V.validate_response(cases[1], '{"orange_tracking_improves":"no","orange_error_decreases":"no"}')["valid"]


def test_umbrella_judge_consumes_policy_label_and_alignment(tmp_path):
    from evals import nemotron_bottleneck_compare_v2 as umbrella

    observations = []
    for case in V.build_cases(_fake_manifest(tmp_path)):
        walking = "yes" if case["packet"]["fixed_eval_group"] == "normal" else "no"
        content = json.dumps({"walking": walking, "in_place": "no", "falling": "no"})
        observations.append({"status": "valid", "verdict": V.validate_response(case, content)})
    judged = umbrella.judge_vision({"feasibility": {"status": "valid"},
                                    "observations": observations, "graphs": []})
    assert judged["status"] == "solved"
    assert judged["alignment"] == {"normal": 7, "failed": 11}


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg unavailable")
def test_extract_frames_uses_six_ordered_midpoint_timestamps(tmp_path):
    source = tmp_path / "test.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=white:s=320x180:d=1", "-r", "12", "-y", str(source)], check=True)
    output = tmp_path / "frames"
    output.mkdir()
    frames = V._extract_frames(source, output)
    assert len(frames) == 6
    assert [item["timestamp_s"] for item in frames] == sorted(item["timestamp_s"] for item in frames)
    assert all(0 < item["bytes"] <= V.MAX_JPEG_BYTES for item in frames)
