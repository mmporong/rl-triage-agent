"""추가 검사 재생 화면에 검증된 U1·ANYmal-C 비교 영상을 결합한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import build_additional_checks_replay as V1
import build_replay_page as B
import validate_scope_transfer_media_public as M

PROTOCOL = Path("bench/protocols/replay_additional_checks_v2.json")
OUTPUT = Path("docs/replay/additional-checks-v2-data.js")


def validate_protocol(protocol: dict, root: Path = ROOT) -> dict:
    if (protocol.get("schema") != "additional_checks_replay_contract_v2"
            or protocol.get("scenes") != ["u1-expanded-scope", "optimizer-update-observation",
                                           "anymal-c-transfer-playback"]
            or protocol.get("scope") != "public_result_replay_with_visual_playback_no_new_measurement"
            or protocol.get("model_calls") != 0 or protocol.get("gpu_calls") != 0
            or protocol.get("validation") != {
                "require_committed_unmodified_inputs_for_ready": True,
                "allow_uncommitted_precommit_verification": True,
                "hide_media_when_not_ready": True,
                "ffprobe_actual_files": True,
                "viewport": [1920, 1080], "overflow_px_max": 0,
            }):
        raise ValueError("v2 fixed contract differs")
    base = protocol.get("base_protocol", {})
    if base.get("path") != V1.PROTOCOL.as_posix() or M.sha256(root / base["path"]) != base.get("sha256"):
        raise ValueError("base protocol binding differs")
    old = json.loads((root / base["path"]).read_text(encoding="utf-8"))
    V1.validate_protocol(old)
    if set(protocol.get("media", {})) != {"u1", "anymal"}:
        raise ValueError("media cases differ")
    for case, spec in protocol["media"].items():
        if (spec.get("case_id") != case or set(spec.get("files", {})) != M.FILENAMES
                or any(spec.get(field) != value for field, value in M.CASES[case].items())):
            raise ValueError("media inventory differs")
    return old


def add_lf_source(sources: B.Sources, root: Path, item: dict) -> str:
    """커밋 상태는 Sources로 확인하고 manifest의 sha256_lf는 LF 기준으로 대조한다."""
    row = sources.file(item["path"])
    if row["status"] == "ready":
        actual = M.sha256_lf(root / item["path"])
        row["sha256_lf"] = actual
        if actual != item["sha256_lf"]:
            row.update(status="mismatch", reason="sha256_lf_mismatch")
    return row["id"]


def add_media(sources: B.Sources, case: str, spec: dict, root: Path) -> tuple[str, list[str]]:
    ids = []
    for filename, expected in spec["files"].items():
        row = sources.file(f"{spec['folder']}/{filename}",
                           kind="media_manifest" if filename == "source_manifest.json" else "video",
                           expected_sha256=expected["sha256"])
        ids.append(row["id"])
        path = root / spec["folder"] / filename
        if path.is_file() and path.stat().st_size != expected["bytes"]:
            row.update(status="mismatch", reason="size_mismatch")
    manifest_id = f"{spec['folder']}/source_manifest.json"
    video_id = f"{spec['folder']}/comparison.mp4"
    if all(sources.ready(source) for source in ids):
        manifest = json.loads((root / manifest_id).read_text(encoding="utf-8"))
        try:
            M.validate_manifest(case, spec, manifest)
            M.validate(root, require_committed=True, run_ffprobe=False)
            for item in manifest["public_sources"]:
                ids.append(add_lf_source(sources, root, item))
            if not all(sources.ready(source) for source in ids):
                raise ValueError("public_source_not_ready")
        except (KeyError, TypeError, ValueError):
            for source in ids:
                sources.rows[source].update(status="mismatch", reason="media_cross_validation_failed")
    elif sources.rows[video_id]["status"] == "ready":
        sources.rows[video_id].update(status="waiting", reason="media_bundle_not_ready")
    return video_id, ids


def build(root: Path = ROOT) -> dict:
    protocol = json.loads((root / PROTOCOL).read_text(encoding="utf-8"))
    validate_protocol(protocol, root)
    data = V1.build(root)
    sources = B.Sources(root)
    media = {}
    for case, spec in protocol["media"].items():
        media[case] = add_media(sources, case, spec, root)
    data["sources"].update(sources.rows)
    u1_video, u1_sources = media["u1"]
    data["scenes"][0]["blocks"].append({
        "type": "video", "wide": True, "title": "같은 Play 설정의 정책 비교",
        "items": [{"label": "U1 실패 / 네 항 복원 후 재학습 / NONE 대조", "source": u1_video}],
        "status_sources": u1_sources,
    })
    data["scenes"][0]["notes"].append(
        "영상은 시각 자료이며 회복 판정은 위의 고정 평가 수치를 따른다.")
    anymal_video, anymal_sources = media["anymal"]
    data["scenes"].append({
        "id": "anymal-c-transfer-playback", "number": 3,
        "title": "ANYmal-C에서도 같은 회복 흐름을 재생",
        "lead": "F1 실패 정책, 보상 결함 복원 후 처음부터 재학습한 정책, NONE 정책을 같은 Play 설정으로 나란히 본다.",
        "scope": "ANYmal-C seed 2047 결과를 본 뒤 만든 시각 자료다. 새 평가가 아니며 판정은 결합된 고정 평가 결과를 따른다.",
        "blocks": [{
            "type": "video", "wide": True, "title": "ANYmal-C 정책 비교",
            "items": [{"label": "F1 실패 / 복원 후 재학습 / NONE 대조", "source": anymal_video}],
            "status_sources": anymal_sources,
        }],
        "notes": ["영상은 정책 동작을 설명하며 성능 수치나 새로운 회복 판정을 만들지 않는다."],
    })
    for index, scene in enumerate(data["scenes"], start=1):
        scene["number"] = index
    data["scope"] = "공개 결과와 해시가 고정된 비교 영상에 결합한 검사·교정 재생 화면"
    B.check_public(data)
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = build()
    content = B.render(data)
    output = ROOT / OUTPUT
    if args.check:
        current = output.is_file() and output.read_text(encoding="utf-8") == content
        print(json.dumps({"status": "current" if current else "stale", "scenes": len(data["scenes"]),
                          "source_status": {state: sum(row["status"] == state for row in data["sources"].values())
                                            for state in ("ready", "waiting", "mismatch")}}, ensure_ascii=False))
        return 0 if current else 1
    output.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "written", "scenes": len(data["scenes"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
