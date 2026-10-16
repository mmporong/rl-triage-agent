"""VL 보조 관찰 시연을 검증하고 공개 allowlist 산출물로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import e2e_demo_v4 as E  # noqa: E402
import vl_aux_demo_v1 as DEMO  # noqa: E402
import vl_aux_observer_v1 as OBS  # noqa: E402


TAG = "vl_aux_f1_s2080_20261016"
PUBLIC_TAG = "vl_aux_demo_public_s2080_20261016"
RUN = ROOT / "bench/private/codex_support_20261015/task23/runs" / TAG
OUTPUT = ROOT / "evals/results" / PUBLIC_TAG
AUDIT = ROOT / "bench/private/codex_support_20261015/task24/export" / PUBLIC_TAG
PROTOCOL = ROOT / "bench/protocols/vl_aux_demo_v1/e2e_demo_v4_f1.json"
FIXED_REPORTS = {
    "none": ROOT / "evals/results/vl_aux_f1_s2080_20261016/runs/baseline_p0c_s2080.json",
    "fault": ROOT / "evals/results/vl_aux_f1_s2080_20261016/runs/h02_s2080.json",
    "corrected": RUN / "fix/eval/restart_corrected.json",
}
CLIPS = {"clip1": "none", "clip2": "fault", "clip3": "corrected"}
EXPECTED_SHA256 = {
    "protocol": "278a8c8310079c9227de724b3504f07e588d97b9014222c0521cb5a42bb05c7d",
    "result": "549f785336be2cf3d6d135050efdc4ecffe6008deef5bf054a7c7e990cce8667",
    "vl_aux_result": "98b3cc1a29fc74bd62818b233e30bf0e212220148ea0b23e5c94b7009d215dcc",
    "fixed_none": "8201eb95e1810f8b9e4d546b1446ef5be8a15b3be56d76b6f265740721975fdb",
    "fixed_fault": "00c04ad93b9f7569103fc21e038b70dbff3fc5a7b9d4063485f3d7356c5dddc8",
    "fixed_corrected": "daec4d19432cac48af362ad5f5346a21a1454332a934cd5b44b2860f12b313b6",
    "diagnosis_record": "5820ee4c8bc224a636236571e84409f1aeddb8a02c91d5c18623fe7c02e88f9f",
    "diagnosis_packet": "5b114b7135804f7774667ccf0277a37435dd28b6249b888897139df6b9969be1",
    "correction_record": "718dfb483fd7cf808607d9d18d571cc90b48c6f91f00a42eca4a22ea05fae255",
    "correction_packet": "61c789dcc5a3e3b8170cd8398c71bcbca0fb7ad3772e24a7dd6dd3dde66c5442",
}
STEP_SHA256 = {
    "begin": "732d1350727fda0e0e5ea57a07c1e61d4a1ca53c706b6520ac9c3d8be3f1d421",
    "train_NONE": "e6577d8b33816353d8cf08d0102f12ad0981c979656f283b65e1fce7ecc8acd1",
    "train_FAULT": "e475bae579696ee927adab8a1aad32004a1ecd3fe6c4febed4c8277b3ba2e0c2",
    "alarm": "668e102385c86794881bedcd502e8668eea5c3bb8cc2ff57c0be3aa8efff05cc",
    "evaluate": "26c574d16a87733b568994d7c95193acca53785668a5b200f5fb5e4d7e3ff070",
    "reference": "35a66e6ca0d0d96950260889b577d1ecfe35e17d939975edc5f32477dc527f4a",
    "packet": "6673750f26a6125d69db0071f65d4ae008fb93f8f244cf202ce686008ffd1c0c",
    "loop_start": "2d80a127bd0733aa6de4ce7dafea6f551289ea179ca0555391313e8e2c50f85d",
    "loop_close": "6201c7c60f5812661ac8ab7dcefdcc6e888cd43eb7c73429fefbc138b91fb5d7",
    "specificity": "9daf80a6894c804026dd0b025532ba5ac645620dab4f703207b2c5af272a280d",
    "correction": "688b2059fed598afd04ae56a5fdc84de279caa0871fc021da4f687127e267d7d",
}
CLIP_SHA256 = {
    "clip1": {
        "packet": "a5b47458d6ec3160b5005cb5233e87c2facaf57a89765a1541012d2db17ec73b",
        "observation": "7be1bccb0d71cb4693fb133310afe285667861f91eea18a1fbf1100a4d73c2b4",
        "manifest": "c9924a4cadde50141e5e5b85792b231e05cd7c2be18760f8d16981e7d1e99af5",
        "result": "1e641ce6942e6e3fa8cac0c5ed930cab4dab0b88546c8fd1afc8ac3d32840c38",
    },
    "clip2": {
        "packet": "e24dbc4fd5435f4ba519f9f8fefc072b48ae90d5b623e6ece51c2b114e5f33f6",
        "observation": "212d1f588acfc1cdf23511d07b80ec72c1d6461bcbff979cfdfb4336784590a6",
        "manifest": "d02b2d6cfefe236bf4ad59dbc07a74d10ef79ebb15d550838315e7ddfac0e44b",
        "result": "75a22b395fa878bad64ad4cf294048b5b91f9c5905e4b8c27c15116935bded59",
    },
    "clip3": {
        "packet": "4329040b3c197d74fd60055658282bc87befe8ec965d95aad2b6e8679c2e5800",
        "observation": "cc694fc98e16f37a9752101461e22adac62edf60acbf57b97a6f9a72adb7a810",
        "manifest": "df69f8b61c58c755c00ea721e5248e8dfa3cf729725e1ab60a0f786ad6623db3",
        "result": "ea3ec06c0fc8a5660bb23aa1896816e7f3853367c0d5c094cf8b3eb2df416e0b",
    },
}
FRAME_SHA256 = {
    "clip1": [
        "e701960a42ded334e42fd1ff645c2db449b343c567a07419a75c943d2a6a9158",
        "d3de5b846a22a600fed53cab166d61e56e16ba80e4e0011db6e7dea1a95cea75",
        "da0b715e7b35181f34ce0d05e2d7622af272aae8a906f1309e5d9c4a3119ca3e",
        "81f15e51e61e0bc9ddd05f648b0c1035116fa033b82eeaf49e7426b8f95254a9",
        "8e9d86b037a0a13bc09aa8a9bb6577104bcb2b4771250b816e0a2ea13669ff76",
        "86ddee896c89a00183cf0320a90920106986659e9292cb8998edcfade799f860",
    ],
    "clip2": [
        "38471bbda4b3069832622f57c917c075fc57ddb986db42c055b4517159eeb0d8",
        "225b8355fc2fdf73274469fcb30192aa1caaf130ff1308d6a448d6f0cf15f97b",
        "efa1e6a87fb34013fb2f40c50b11f61326c9e63ebf248dc3699f6e6d4c34bbd3",
        "a4d6f8e5ecb2a48c2d6fd12f41ddef47a56e21d3616bfd3dceafb2cf38b52edc",
        "b9cbb02844a9d573bec77a496d314f143e839d47f3aa2f0fd192c25f458992d6",
        "4ed8a6ccc524e0b9c04736a87df69cbd130b0fd4b94329897ff4e6fc51a89be3",
    ],
    "clip3": [
        "d21de9d627b887a4c02f9441692a9f79138b1e5865017c6b98fe5f06d3dc9541",
        "f9ae2d7a5bfe8cc3dac71f62f8f330135356ffb6f97d2d9000e8f94db3ec04a7",
        "7c2b1990b1b1c971c5aa4dd802f280abdd5f1649a3eb0b49d4d9f89cb07cd77d",
        "0dad3247110badd7da6f53bb2a703735238dddc616f033ccccf50d76cc8c7f5f",
        "4876dd41316bd838c2475e091c85b0be30c51ccfcb59b94d1e531537c4a49d90",
        "b185a52a4462d893801079888046815cdab4e9140593de6cd916234180b8087b",
    ],
}
JSON_FILES = {
    "summary.json", "observations.json", "evaluations/none.json",
    "evaluations/fault.json", "evaluations/corrected.json", "provider_identity.json",
    "provenance.json",
}
FRAME_FILES = {
    f"frames/{clip}/frame_{index:02d}.jpg"
    for clip in CLIPS for index in range(1, 7)
}
PUBLIC_FILES = JSON_FILES | FRAME_FILES
_FORBIDDEN = re.compile(
    r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|bench[\\/]private|codex_support|task23[\\/]",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


def _require_hash(path: Path, expected: str, logical_id: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"source artifact differs: {logical_id}")


def _verify_graph(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("steps_sha256") != STEP_SHA256:
        raise ValueError("result step graph differs")
    rows = []
    previous = None
    for index, (step, digest) in enumerate(STEP_SHA256.items()):
        path = RUN / "steps" / f"{index:02d}_{step}.json"
        _require_hash(path, digest, f"step_{step}")
        body = read(path)
        if body.get("step") != step or body.get("previous_sha256") != previous:
            raise ValueError(f"step graph link differs: {step}")
        rows.append({"step": step, "sha256": digest, "previous_sha256": previous})
        previous = digest
    return rows


def _public_evaluation(arm: str, report: dict[str, Any], source_sha: str,
                       label: str, verdict: dict[str, Any]) -> dict[str, Any]:
    conditions = [
        {
            "command": row["command"], "num_envs": row["num_envs"],
            "fall_count": row["fall_count"], "alive_steps_sum": row["alive_steps_sum"],
            "fall_rate": row["fall_rate"], "lin_vel_rmse_mps": row["lin_vel_rmse_mps"],
            "yaw_rate_rmse_radps": row["yaw_rate_rmse_radps"],
        }
        for row in report["by_condition"]
    ]
    if len(conditions) != 26:
        raise ValueError(f"fixed evaluation condition count differs: {arm}")
    return {
        "schema": "public_vl_aux_fixed_evaluation_v1", "arm": arm, "label": label,
        "verdict": verdict,
        "protocol": {
            "name": report["protocol"], "sha256_lf": report["protocol_sha256_lf"],
            "task": report["task"], "eval_seed": report["eval_seed"],
            "num_envs": report["num_envs"], "horizon_steps": report["horizon_steps"],
            "step_dt": report["step_dt"], "action_scale": report["action_scale"],
        },
        "checkpoint": report["checkpoint"], "metrics": report["metrics"],
        "standstill": report["standstill"], "conditions": conditions,
        "source_artifact_sha256": source_sha,
    }


def _verify_models() -> dict[str, Any]:
    records = {}
    with DEMO.configured():
        for kind in ("diagnosis", "correction"):
            record_path = RUN / f"model/{kind}.feedback1.json"
            packet_path = RUN / f"model/{kind}.packet.json"
            _require_hash(record_path, EXPECTED_SHA256[f"{kind}_record"], f"{kind}_record")
            _require_hash(packet_path, EXPECTED_SHA256[f"{kind}_packet"], f"{kind}_packet")
            verified = E.review_feedback_record(record_path, packet_path, kind)
            raw = read(record_path)
            if verified["status"] != "valid" or raw["requests"] != 1:
                raise ValueError(f"{kind} model record is not valid")
            records[kind] = {
                "requests": raw["requests"],
                "input_tokens": sum(row["input_tokens"] for row in raw["calls"]),
                "output_tokens": sum(row["output_tokens"] for row in raw["calls"]),
                "wall_s": sum(row["wall_s"] for row in raw["calls"]),
            }
    return records


def verify_and_build() -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    source_paths = {
        "protocol": PROTOCOL, "result": RUN / "result.json", "vl_aux_result": RUN / "vl_aux_result.json",
        "fixed_none": FIXED_REPORTS["none"], "fixed_fault": FIXED_REPORTS["fault"],
        "fixed_corrected": FIXED_REPORTS["corrected"],
        "diagnosis_record": RUN / "model/diagnosis.feedback1.json",
        "diagnosis_packet": RUN / "model/diagnosis.packet.json",
        "correction_record": RUN / "model/correction.feedback1.json",
        "correction_packet": RUN / "model/correction.packet.json",
    }
    for logical_id, expected in EXPECTED_SHA256.items():
        _require_hash(source_paths[logical_id], expected, logical_id)

    protocol, result, aux = read(PROTOCOL), read(RUN / "result.json"), read(RUN / "vl_aux_result.json")
    if (result.get("outcome") != "recovered" or result.get("demo_success") is not True
            or len(result.get("success_conditions", {})) != 12
            or not all(result["success_conditions"].values()) or result.get("unmet_conditions") != []
            or result.get("protocol_sha256_lf") != EXPECTED_SHA256["protocol"]
            or result.get("source_commit_sha") != ["110720c2b0188d48e39bbc8cdbc90a3d88fcc6bb"]):
        raise ValueError("base demo result differs")
    if (aux.get("base_result_sha256") != EXPECTED_SHA256["result"]
            or aux.get("outcome") != "recovered" or aux.get("demo_success") is not True
            or aux.get("auxiliary_complete") is not True or aux.get("expected_policies") != 3
            or aux.get("used_for_decision") is not False
            or aux.get("api_requests") != {"decision": 2, "observer": 3, "total": 5}):
        raise ValueError("VL auxiliary result differs")
    graph = _verify_graph(result)
    model_calls = _verify_models()
    decision_identity = []
    for kind in ("diagnosis", "correction"):
        raw = read(RUN / f"model/{kind}.feedback1.json")
        decision_identity.append({
            "call": kind, "requested_model": raw["request"]["model"],
            "returned_provider_model": raw["calls"][0]["provider_model"],
            "registered_alias": False,
        })

    fixed = {arm: read(path) for arm, path in FIXED_REPORTS.items()}
    gate = E.source_gate(fixed["none"], fixed["fault"], protocol["bands"])
    if gate != result["evaluation"]["gate"]:
        raise ValueError("source fixed-evaluation gate differs")
    corrected_verdict = E.RC.verdict(
        E.RC.compare(E.C._behavior(fixed["corrected"]), E.C._behavior(fixed["none"])), protocol["bands"]
    )
    if corrected_verdict != result["correction"]["result"]["verdict"]:
        raise ValueError("corrected fixed-evaluation verdict differs")

    public_frames: dict[str, Path] = {}
    observations = []
    observer_identity = []
    observer_input = observer_output = 0
    observer_wall = 0.0
    aux_by_clip = {row["clip_id"]: row for row in aux["auxiliary"]}
    for clip, arm in CLIPS.items():
        folder = RUN / "auxiliary" / clip
        paths = {
            "packet": folder / "frames/packet.json", "observation": folder / "observation.json",
            "manifest": folder / "play/result_manifest.json", "result": folder / "result.json",
        }
        for name, path in paths.items():
            _require_hash(path, CLIP_SHA256[clip][name], f"{clip}_{name}")
        observation = OBS.verify_record(paths["packet"], paths["observation"], RUN / "auxiliary_api")
        manifest, clip_result = read(paths["manifest"]), read(paths["result"])
        expected_aux = aux_by_clip[clip]
        if (clip_result != expected_aux or observation["status"] != "valid"
                or observation["verdict"]["valid"] is not True or observation["requests"] != 1
                or clip_result["used_for_decision"] is not False
                or manifest["used_for_recovery_decision"] is not False
                or manifest["label_forwarded_to_player_or_model"] is not False
                or manifest["source_sha256"]["checkpoint"] != clip_result["checkpoint_sha256"]
                or manifest["source_sha256"]["fixed_report"] != clip_result["fixed_report_sha256"]):
            raise ValueError(f"{clip} observation binding differs")
        decision = observation["verdict"]["decision_key"]
        agreement = (clip_result["fixed_evaluation_label"] == "healthy" and decision["walking"] == "yes") \
            or (clip_result["fixed_evaluation_label"] == "unhealthy" and decision["walking"] == "no")
        if not agreement or decision["falling"] != "no":
            raise ValueError(f"{clip} observation does not match the preregistered public statement")
        frame_rows = []
        for index, frame in enumerate(observation["frames"], 1):
            source = folder / f"frames/frames/frame_{index:02d}.jpg"
            expected_frame = FRAME_SHA256[clip][index - 1]
            _require_hash(source, expected_frame, f"{clip}_frame_{index:02d}")
            if frame["sha256"] != expected_frame or frame["bytes"] != source.stat().st_size:
                raise ValueError(f"{clip} frame metadata differs")
            relative = f"frames/{clip}/frame_{index:02d}.jpg"
            public_frames[relative] = source
            frame_rows.append({"path": relative, **frame})
        attempt = observation["attempts"][0]["result"]
        observer_identity.append({
            "call": clip, "requested_model": observation["model"],
            "returned_provider_model": attempt["provider_model"],
            "returned_provider_model_raw": attempt["provider_model_raw"],
            "registered_alias": attempt["provider_identity_registered"],
        })
        observer_input += attempt["input_tokens"]
        observer_output += attempt["output_tokens"]
        observer_wall += attempt["measured_wall_s"]
        observations.append({
            "clip": clip, "arm": arm, "fixed_evaluation_label": clip_result["fixed_evaluation_label"],
            "used_for_decision": False, "status": "valid", "decision": decision,
            "fixed_label_correspondence": True, "model": observation["model"],
            "provider_identity_policy": observation["provider_identity_policy"],
            "requests": observation["requests"],
            "tokens": {"input": attempt["input_tokens"], "output": attempt["output_tokens"],
                       "total": attempt["total_tokens"]},
            "model_wall_s": attempt["wall_s"], "measured_wall_s": attempt["measured_wall_s"],
            "binding": {"checkpoint_sha256": clip_result["checkpoint_sha256"],
                        "fixed_evaluation_sha256": clip_result["fixed_report_sha256"]},
            "frames": frame_rows,
        })

    audited = OBS.RequestLedger(RUN / "auxiliary_api").audit()
    if len(audited) != 3 or any(row["response"] is None for row in audited):
        raise ValueError("observer request ledger differs")
    total_input = observer_input + sum(row["input_tokens"] for row in model_calls.values())
    total_output = observer_output + sum(row["output_tokens"] for row in model_calls.values())
    decision_wall = sum(row["wall_s"] for row in model_calls.values())
    payloads = {
        "summary.json": {
            "schema": "public_vl_aux_demo_summary_v1", "status": "complete", "seed": 2080,
            "restart_seed": 3080, "fault": "F1", "outcome": "recovered", "demo_success": True,
            "base_success_conditions": {"passed": 12, "total": 12},
            "auxiliary": {"complete": True, "valid_observations": 3, "expected_observations": 3,
                          "fixed_label_correspondence": {"matched": 3, "total": 3},
                          "used_for_decision": False},
            "api": {"requests": {"decision": 2, "observer": 3, "total": 5, "cap": 16},
                    "tokens": {"input": total_input, "output": total_output,
                               "total": total_input + total_output},
                    "wall_s": {"decision_model": decision_wall, "observer_measured": observer_wall,
                               "combined": decision_wall + observer_wall}},
            "statement_scope": "새 seed 한 건의 통합 확인이며 영상 관찰은 회복 판정에 사용하지 않았다.",
        },
        "observations.json": {
            "schema": "public_vl_aux_observations_v1", "frame_sampling": "six temporal midpoints",
            "fixed_evaluation_label_in_prompt": False, "used_for_decision": False,
            "observations": observations,
        },
        "evaluations/none.json": _public_evaluation(
            "none", fixed["none"], EXPECTED_SHA256["fixed_none"], "healthy",
            {"label": "healthy", "checks": {"reference_policy": True}, "failed": []},
        ),
        "evaluations/fault.json": _public_evaluation(
            "fault", fixed["fault"], EXPECTED_SHA256["fixed_fault"], "unhealthy", gate["fault_verdict"],
        ),
        "evaluations/corrected.json": _public_evaluation(
            "corrected", fixed["corrected"], EXPECTED_SHA256["fixed_corrected"],
            corrected_verdict["label"], corrected_verdict,
        ),
        "provider_identity.json": {
            "schema": "public_vl_aux_provider_identity_v1",
            "identity_policy": "registered endpoint alias; model weights were not attested",
            "timing_scope": {
                "decision_model": "sum of client SDK-call wall_s measured inside openai_complete for two text decision calls",
                "observer_measured": "sum of client-measured wall_s for three visual observation calls",
                "combined": "arithmetic display across the two scopes; not a comparable latency or full experiment runtime",
                "timing_comparison_allowed": False,
            },
            "calls": decision_identity + observer_identity,
        },
        "provenance.json": {
            "schema": "public_vl_aux_demo_provenance_v1",
            "export_contract": "allowlisted fields and byte-identical sampled frames",
            "source_commit_sha": "110720c2b0188d48e39bbc8cdbc90a3d88fcc6bb",
            "protocol_sha256_lf": EXPECTED_SHA256["protocol"],
            "base_result_sha256": EXPECTED_SHA256["result"],
            "auxiliary_result_sha256": EXPECTED_SHA256["vl_aux_result"],
            "step_graph": graph,
            "model_record_validation": {"diagnosis": "valid", "correction": "valid",
                                        "observer_records": 3, "observer_ledger_entries": len(audited)},
            "source_artifact_sha256": EXPECTED_SHA256,
            "clip_artifact_sha256": CLIP_SHA256,
        },
    }
    verify_public_payloads(payloads, public_frames)
    return payloads, public_frames


def verify_public_payloads(payloads: dict[str, dict[str, Any]], frames: dict[str, Path]) -> None:
    if set(payloads) != JSON_FILES or set(frames) != FRAME_FILES:
        raise ValueError("public artifact inventory differs")
    for relative, payload in payloads.items():
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if _has_forbidden_string(payload):
            raise ValueError(f"private or user path in public payload: {relative}")
    for relative, source in frames.items():
        clip = Path(relative).parts[1]
        index = int(Path(relative).stem.split("_")[1])
        if sha256(source) != FRAME_SHA256[clip][index - 1]:
            raise ValueError(f"public frame source differs: {relative}")


def _has_forbidden_string(value: Any) -> bool:
    if isinstance(value, str):
        return _FORBIDDEN.search(value) is not None
    if isinstance(value, dict):
        return any(_has_forbidden_string(key) or _has_forbidden_string(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_forbidden_string(item) for item in value)
    return False


def _write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_new(payloads: dict[str, dict[str, Any]], frames: dict[str, Path]) -> None:
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("public output or export audit already exists")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for relative, payload in sorted(payloads.items()):
        _write_json_new(OUTPUT / relative, payload)
    for relative, source in sorted(frames.items()):
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    public_hashes = {relative: sha256(OUTPUT / relative) for relative in sorted(PUBLIC_FILES)}
    AUDIT.mkdir(parents=True, exist_ok=False)
    _write_json_new(AUDIT / "receipt.json", {
        "schema": "vl_aux_public_export_receipt_v1", "source_artifact_sha256": EXPECTED_SHA256,
        "public_artifact_sha256": public_hashes,
    })


def verify_export() -> dict[str, str]:
    payloads, frames = verify_and_build()
    actual = {path.relative_to(OUTPUT).as_posix() for path in OUTPUT.rglob("*") if path.is_file()}
    if actual != PUBLIC_FILES:
        raise ValueError("exported public artifact inventory differs")
    for relative, payload in payloads.items():
        if read(OUTPUT / relative) != payload:
            raise ValueError(f"exported public JSON differs: {relative}")
    for relative, source in frames.items():
        if sha256(OUTPUT / relative) != sha256(source):
            raise ValueError(f"exported public frame differs: {relative}")
    return {relative: sha256(OUTPUT / relative) for relative in sorted(PUBLIC_FILES)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "export", "verify"))
    args = parser.parse_args()
    if args.command == "verify":
        result = {"status": "pass", "public_sha256": verify_export()}
    else:
        payloads, frames = verify_and_build()
        if args.command == "export":
            write_new(payloads, frames)
        result = {"status": "exported" if args.command == "export" else "pass",
                  "public_files": sorted(PUBLIC_FILES)}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
