"""회수된 motor capture 네 정책(seed 2033·2046의 NONE·F2)을 기존 검증기로 묶어 상태 수치 facts를 만드는 CPU adapter.

순서는 등록 결합 → 회수 게이트 → 정책별 검증 → write-once 출력이다.

- 등록: 호출자가 명시한 공개 phase(repo 상대 경로와 LF SHA)의 batch·protocol SHA와 portable manifest(phase에
  등록된 SHA)의 원본 checkpoint SHA를 exact match한다. 네 logical ID·pair·F2/NONE은 등록값에서만 만든다.
  특정 phase 버전을 기본값으로 두지 않으므로 v5·이후 등록을 같은 입력 형식으로 받는다.
- 회수 게이트: host ``collection.json``의 archive SHA 검증·C/D 사본 ``all_match``가 봉인되고, 회수 root의
  파일 집합이 ``collection-inventory.json``과 같아야 한다. ``collection.json``이 없으면 ``Hold``(실측 HOLD)다.
- 정책: stage는 host 규칙 ``record-<logical_id>``, output은 ``stage/output``, sidecar는
  ``JOB1.sidecar_directory_for(stage)``로만 찾는다. 원격 job 경로는 바꾸지 않고 구조만 대조한다.
  facts는 ``agentic_state_facts_v1.build_state_facts``에 final join SHA를 등록값으로 넘겨 만든다.
- 판정: 기술 수용·거부만 기록한다. 행동 임계값·걷기·넘어짐·원인·회복 정답을 만들지 않고, 등록
  ``fixed_flags``(capture_approval·model_ready·used_for_decision = false)를 바꾸지 않는다. 최종 수용은 메인 몫이다.
- 공개 후보 facts에는 경로·job·정책·고장 ID를 넣지 않는다. 결합 식별자는 비공개 metadata에 둔다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import agentic_play_job_v1 as JOB1
import agentic_state_facts_v1 as FACTS
import record_agentic_state_sidecar_v1 as SIDECAR
import record_vision_evidence_v1 as R
import run_agentic_capture_unit_v2 as UNIT2
import run_agentic_play_record_v1 as RUN1

PORTABLE_SCHEMA = "agentic_capture_input_bundle_v1"
STAGE_MODE = "record"  # host v5 unit(): stage = remote / (mode + "-" + logical_id)
# stage 옆에 생기는 등록 산출물. admission은 UNIT1, incoming은 host v5, 관측은 UNIT2 규칙이다.
STAGE_SIBLINGS = (JOB1.SIDECAR_SUFFIX, "-admission", "-incoming", UNIT2.OBSERVATION_SUFFIX)
COLLECTION_NAME = "collection.json"
COLLECTION_FAILURE_NAME = "collection-failure.json"
ARCHIVE_NAME = "retrieved.tar.gz"
INVENTORY_NAME = "collection-inventory.json"
UNIT_RESULT_SCHEMA = "agentic_capture_unit_result_v1"
FIXED_FLAGS = {"capture_approval": False, "model_ready": False, "used_for_decision": False}
SCHEMA = "agentic_motor_capture_facts_v1"
METADATA_SCHEMA = "agentic_motor_capture_metadata_v1"
SUMMARY_SCHEMA = "agentic_motor_capture_acceptance_v1"
PUBLIC_NAME = "motor_capture_facts.json"
METADATA_NAME = "motor_capture_metadata.json"
SUMMARY_NAME = "motor_capture_acceptance.json"
_WSL = re.compile(r"^/mnt/([a-z])/(.*)$")
_WINDOWS = re.compile(r"^([A-Za-z]):[\\/](.*)$")


class AcceptanceError(ValueError):
    """등록·회수·결합 계약 위반. 부분 결과를 수용하지 않는다."""


class Hold(RuntimeError):
    """회수 봉인 전이라 실측을 처리하지 않는다(입력이 없는 상태이며 실패가 아니다)."""


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise AcceptanceError(reason)


def _exact(value: Any, keys: set[str], anchor: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == keys, f"{anchor} 필드가 계약과 다르다")
    return value


def host_path(text: Any) -> Path:
    """host 기록 경로를 이 OS 경로로 바꾼다. WSL ``/mnt/<d>/``와 Windows ``<D>:/``만 1:1로 대응한다."""
    _require(isinstance(text, str) and bool(text), "기록 경로가 문자열이 아니다")
    if os.name == "nt":
        match = _WSL.match(text)
        path = Path(f"{match[1].upper()}:/", match[2]) if match else Path(text)
    else:
        match = _WINDOWS.match(text)
        path = Path("/mnt", match[1].lower(), match[2].replace("\\", "/")) if match else Path(text)
    _require(path.is_absolute(), "기록 경로가 절대 경로가 아니다")
    return path


def _same_path(text: Any, expected: Path) -> bool:
    return os.path.normcase(str(host_path(text))) == os.path.normcase(str(Path(expected).absolute()))


def _no_link(path: Path, anchor: str) -> Path:
    path = Path(path).absolute()
    try:
        JOB1.COMMON._check_ancestors(path, anchor)
    except ValueError as error:
        raise AcceptanceError(str(error)) from error
    return path


def _regular(path: Path, anchor: str) -> Path:
    path = _no_link(path, anchor)
    _require(path.is_file(), f"{anchor}가 일반 파일이 아니다")
    return path


def _read(path: Path, anchor: str) -> dict[str, Any]:
    try:
        return R.read_object(_regular(path, anchor))
    except (OSError, TypeError, ValueError) as error:
        raise AcceptanceError(f"{anchor}를 읽을 수 없다: {error}") from error


def _relative(text: Any, anchor: str) -> PurePosixPath:
    _require(isinstance(text, str) and bool(text) and "\\" not in text and ":" not in text, f"{anchor} 형식")
    path = PurePosixPath(text)
    _require(not path.is_absolute() and ".." not in path.parts and "." not in path.parts
             and str(path) == text, f"{anchor} 경로가 root 밖이거나 정규형이 아니다")
    return path


def _walk_files(root: Path) -> set[str]:
    """symlink·reparse point 없이 일반 파일만 모은다(따라가지 않는다)."""
    found: set[str] = set()
    pending = [root]
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                _require(not stat.S_ISLNK(info.st_mode) and not JOB1.COMMON._is_reparse(info),
                         "회수 root 안에 symlink/reparse point가 있다")
                path = Path(entry.path)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    found.add(path.relative_to(root).as_posix())
                else:
                    raise AcceptanceError("회수 root 안에 일반 파일·디렉터리 외 항목이 있다")
    return found


# --- 등록 ----------------------------------------------------------------------------------------


def load_registration(
    portable_manifest: Path, *, phase: str, phase_sha256_lf: str, repo: Path = ROOT
) -> dict[str, Any]:
    """명시한 phase·두 protocol·portable manifest에서 네 정책의 등록값을 만든다."""
    repo = Path(repo).absolute()
    phase_path = _regular(repo / _relative(phase, "phase"), "phase")
    _require(R.sha256_file(phase_path, lf=True) == phase_sha256_lf, "phase SHA가 등록값과 다르다")
    value = _read(phase_path, "phase")
    _require(value.get("planned_recordings") == 4 and value.get("frames_per_recording") == R.FRAME_COUNT
             and value.get("fps") == R.FPS, "phase 녹화 수·frame·fps가 계약과 다르다")
    meaning = value.get("interpretation") or {}
    _require(meaning.get("behavior_labels_created") is False and meaning.get("fixed_eval_is_behavior_ground_truth") is False
             and meaning.get("f2_all_12_joint_torque_speed_change") is True, "phase 해석 고정값이 다르다")
    vms: Any = value.get("vms")
    _require(isinstance(vms, list) and len(vms) == 1 and isinstance(vms[0], dict), "phase VM 등록이 하나가 아니다")
    resource_id = vms[0].get("resource_id")
    batches = vms[0].get("batches")
    _require(isinstance(resource_id, str) and isinstance(batches, list) and len(batches) == 2, "phase batch가 둘이 아니다")
    portable_path = _regular(Path(portable_manifest), "portable manifest")
    portable_sha = R.sha256_file(portable_path)
    _require(portable_sha == value.get("input_manifest_sha256"), "portable manifest SHA가 phase 등록값과 다르다")
    portable = _read(portable_path, "portable manifest")
    _require(portable.get("schema") == PORTABLE_SCHEMA and isinstance(portable.get("files"), list), "portable schema")
    rows = {row.get("archive_path"): row for row in portable["files"] if isinstance(row, dict)}
    _require(len(rows) == len(portable["files"]), "portable manifest 경로가 중복됐다")
    policies: dict[str, dict[str, Any]] = {}
    protocols = []
    for batch in batches:
        batch = _exact(batch, {"protocol", "protocol_sha256_lf", "reference_id", "case_id"}, "phase batch")
        relative = _relative(batch["protocol"], "protocol")
        path = _regular(repo / relative, "protocol")
        _require(R.sha256_file(path, lf=True) == batch["protocol_sha256_lf"], "protocol SHA가 phase 등록값과 다르다")
        protocol = _read(path, "protocol")
        case, reference = batch["case_id"], batch["reference_id"]
        _require(protocol.get("schema") == JOB1.PROTOCOL_SCHEMA
                 and protocol.get("recording_target", {}).get("resource_id") == resource_id, "protocol 자원·schema")
        flags = protocol.get("preregistration", {}).get("fixed_flags", {})
        _require(all(flags.get(key) is False for key in FIXED_FLAGS), "등록 승인 플래그가 false 고정값과 다르다")
        _require(protocol.get("interpretation", {}).get("single_leg_fault") is False, "protocol 해석 고정값이 다르다")
        pairs: Any = protocol.get("pairs")
        _require(isinstance(pairs, list) and len(pairs) == 1, "protocol pair가 하나가 아니다")
        pair = _exact(pairs[0], {"pair_id", "seed", "case", "reference"}, "protocol pair")
        _require(pair["case"] == case and pair["reference"] == reference, "protocol pair가 phase batch와 다르다")
        checkpoints = {row["logical_id"]: row for row in protocol.get("checkpoints", [])}
        registrations = protocol.get("playback_registrations", {})
        _require(set(checkpoints) == {case, reference} == set(registrations), "protocol checkpoint 등록이 pair와 다르다")
        for identity, role, fault in ((reference, "reference", "NONE"), (case, "case", "F2")):
            row = checkpoints[identity]
            registered = registrations[identity]
            portable_row = rows.get(row["isaaclab_relative"])
            _require(registered.get("fault_id") == fault and registered.get("checkpoint_sha256") == row["sha256"],
                     "playback 등록의 fault·checkpoint가 다르다")
            _require(isinstance(portable_row, dict) and portable_row.get("sha256") == row["sha256"],
                     "portable manifest 원본 checkpoint SHA가 protocol과 다르다")
            policies[identity] = {
                "logical_id": identity, "role": role, "fault_id": fault, "pair": dict(pair),
                "checkpoint_sha256": row["sha256"], "protocol": relative.as_posix(),
                "protocol_sha256_lf": batch["protocol_sha256_lf"], "protocol_id": protocol["protocol_id"],
                "job_id": f"{protocol['protocol_id']}-{identity}", "registration": dict(registered),
            }
        protocols.append({"path": relative.as_posix(), "sha256_lf": batch["protocol_sha256_lf"]})
    _require(len(policies) == 4 and len({row["checkpoint_sha256"] for row in policies.values()}) == 4,
             "등록 정책·checkpoint가 넷으로 구분되지 않는다")
    return {"phase": phase, "phase_sha256_lf": phase_sha256_lf, "resource_id": resource_id,
            "portable_manifest_sha256": portable_sha, "protocols": protocols, "policies": policies}


def stage_name(logical_id: str) -> str:
    return f"{STAGE_MODE}-{logical_id}"


# --- 회수 게이트 ---------------------------------------------------------------------------------


def verify_collection(run_dir: Path, registration: Mapping[str, Any]) -> dict[str, Any]:
    """host 봉인과 inventory 파일 집합을 확인한다. 자료가 아직 없으면 ``Hold``."""
    run = _no_link(Path(run_dir), "run")
    _require(run.name == f"run-{registration['resource_id']}", "run 폴더가 등록 자원과 다르다")
    _require(not (run / COLLECTION_FAILURE_NAME).exists(), "host 회수 실패가 봉인됐다")
    if not (run / COLLECTION_NAME).exists():
        raise Hold("collection.json이 아직 없다(회수·C/D 대조 봉인 전)")
    collection = _exact(_read(run / COLLECTION_NAME, "collection"),
                        {"path", "sha256", "files", "local", "archive_sha_verified", "copy_verification"}, "collection")
    copy = _exact(collection["copy_verification"], {"files_verified", "local", "backup", "all_match"}, "copy")
    _require(collection["archive_sha_verified"] is True and copy["all_match"] is True, "archive·C/D 대조가 봉인되지 않았다")
    archive = _regular(run / ARCHIVE_NAME, "archive")
    _require(_same_path(collection["local"], archive) and R.sha256_file(archive) == collection["sha256"],
             "회수 archive 경로·SHA가 봉인값과 다르다")
    evidence = _no_link(run / "retrieved" / "evidence", "회수 root")
    _require(evidence.is_dir() and _same_path(copy["local"], evidence), "C 사본 root가 등록 회수 root와 다르다")
    backup = _no_link(host_path(copy["backup"]), "D 사본 root")
    _require(backup.is_dir(), "D 사본 root가 없다")
    inventory = _exact(_read(evidence / INVENTORY_NAME, "inventory"), {"files"}, "inventory")
    rows: dict[str, dict[str, Any]] = {}
    for raw in inventory["files"]:
        row = _exact(raw, {"path", "sha256", "bytes"}, "inventory row")
        path = _relative(row["path"], "inventory path").as_posix()
        _require(path not in rows and path != INVENTORY_NAME, "inventory 경로가 중복됐다")
        rows[path] = row
    _require(collection["files"] == len(rows) and copy["files_verified"] == len(rows) + 1, "회수 파일 수가 봉인값과 다르다")
    _require(_walk_files(evidence) == set(rows) | {INVENTORY_NAME}, "회수 root 파일 집합이 inventory와 다르다")
    return {"run": run, "evidence": evidence, "backup": backup, "inventory": rows,
            "collection_sha256": R.sha256_file(run / COLLECTION_NAME), "archive_sha256": collection["sha256"],
            "inventory_sha256": R.sha256_file(evidence / INVENTORY_NAME), "files": len(rows)}


def _record_entries(evidence: Path, policies: Mapping[str, Any]) -> list[str]:
    """회수 root 바로 아래 ``record-*`` 항목이 등록 stage와 그 고정 옆 경로뿐인지 본다."""
    allowed = {stage_name(identity) + suffix for identity in policies for suffix in ("", *STAGE_SIBLINGS)}
    extra = sorted(entry.name for entry in os.scandir(evidence)
                   if entry.name.startswith(STAGE_MODE + "-") and entry.name not in allowed)
    return [f"등록 밖 record 항목: {name}" for name in extra]


def _verify_refs(prefixes: Sequence[str], gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """stage·sidecar 아래 모든 파일을 inventory·C·D SHA로 다시 대조한다."""
    refs = []
    for path, row in sorted(gate["inventory"].items()):
        if not any(path.startswith(prefix + "/") for prefix in prefixes):
            continue
        local = gate["evidence"] / path
        backup = _regular(gate["backup"] / path, "D 사본 파일")
        _require(local.stat().st_size == row["bytes"] and R.sha256_file(local) == row["sha256"],
                 f"C 사본 SHA·크기가 inventory와 다르다: {path}")
        _require(R.sha256_file(backup) == row["sha256"], f"D 사본 SHA가 inventory와 다르다: {path}")
        refs.append({"path": path, "sha256": row["sha256"], "bytes": row["bytes"]})
    _require(bool(refs), "stage 파일이 inventory에 없다")
    return refs


# --- 정책 -----------------------------------------------------------------------------------------


def _check_final(final: Mapping[str, Any], policy: Mapping[str, Any], job_sha: str) -> None:
    _require(final.get("schema") == RUN1.FINAL_SCHEMA, "final schema가 RUN1 계약과 다르다")
    _require(final.get("final_technical_complete") is True and final.get("acceptable_as_live_evidence") is True
             and final.get("live_evidence") is True, "final이 기술 완료·live evidence가 아니다")
    lease = final.get("lease") or {}
    _require(lease.get("active_at_finalize") is True and lease.get("live_lock_proof") is True, "final lease 증거가 live가 아니다")
    _require(final.get("behavior_label") is None and final.get("used_for_decision") is False, "final 판정 필드가 바뀌었다")
    _require(final.get("checkpoint_logical_id") == policy["logical_id"]
             and final.get("checkpoint_sha256") == policy["checkpoint_sha256"]
             and final.get("fault_id") == policy["fault_id"] and final.get("pair") == policy["pair"],
             "final checkpoint·fault·pair가 등록값과 다르다")
    _require(final.get("job_sha256") == job_sha, "final job SHA가 회수 job과 다르다")
    _require(final.get("playback_completed") is True and final.get("actuator_clip_consistent_with_registration") is True,
             "final playback·모터 함수 대조가 완료되지 않았다")
    sidecar = final.get("sidecar") or {}
    join = final.get("join") or {}
    _require(sidecar.get("status") == "sealed" and join.get("joined") is True
             and join.get("frame_count") == R.FRAME_COUNT and join.get("checkpoint_sha256") == policy["checkpoint_sha256"],
             "final sidecar·join이 625 frame 봉인 결합이 아니다")


def _check_job(job: Mapping[str, Any], policy: Mapping[str, Any], stage: str) -> None:
    _require(job.get("schema") == JOB1.JOB_SCHEMA and job.get("mode") == JOB1.MODE
             and job.get("job_id") == policy["job_id"], "job schema·ID가 등록값과 다르다")
    checkpoint = job.get("checkpoint") or {}
    _require(checkpoint.get("logical_id") == policy["logical_id"]
             and checkpoint.get("sha256") == policy["checkpoint_sha256"], "job checkpoint가 등록값과 다르다")
    protocol = job.get("protocol") or {}
    _require(protocol.get("sha256_lf") == policy["protocol_sha256_lf"]
             and PurePosixPath(str(protocol.get("path"))).name == PurePosixPath(policy["protocol"]).name,
             "job protocol이 등록 protocol과 다르다")
    output = PurePosixPath(str(job.get("output_directory")))
    remote_stage = output.parent
    sidecar = job.get("sidecar") or {}
    _require(output.is_absolute() and output.name == "output" and remote_stage.name == stage,
             "job output이 등록 stage/output 구조가 아니다")
    _require(PurePosixPath(str(sidecar.get("directory"))) == remote_stage.parent / (stage + JOB1.SIDECAR_SUFFIX)
             and sidecar.get("source_id") == policy["job_id"] and sidecar.get("expected_frames") == R.FRAME_COUNT,
             "job sidecar가 고정 stage 옆 경로·source·frame과 다르다")


def check_policy(policy: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    """정책 하나를 검증하고 facts를 만든다. 실패하면 AcceptanceError 또는 FactsError."""
    name = stage_name(policy["logical_id"])
    evidence: Path = gate["evidence"]
    stage = _no_link(evidence / name, "stage")
    _require(stage.is_dir(), "등록 stage가 회수되지 않았다")
    sidecar_dir = _no_link(JOB1.sidecar_directory_for(stage), "sidecar")
    refs = _verify_refs([name, sidecar_dir.name], gate)
    job_path = _regular(stage / "job.json", "job")
    job = _read(job_path, "job")
    _check_job(job, policy, name)
    final_path = _regular(stage / RUN1.FINAL_NAME, "final")
    _check_final(_read(final_path, "final"), policy, R.sha256_file(job_path))
    _require(not (stage / "unit-failure.json").exists(), "unit 실패가 기록됐다")
    unit = _read(stage / "unit-result.json", "unit result")
    _require(unit.get("schema") == UNIT_RESULT_SCHEMA and unit.get("mode") == STAGE_MODE
             and unit.get("completed") is True, "unit 결과가 record 완료가 아니다")
    output = stage / "output"
    trace = _regular(output / "evidence" / "raw_trace.json", "raw trace")
    native = _regular(output / "native_capture_manifest.json", "native manifest")
    facts = FACTS.build_state_facts(sidecar_dir, trace, native, expected=FACTS._expected_from_final(final_path))
    metadata = facts["metadata"]
    _require(metadata["job_id"] == policy["job_id"] and metadata["checkpoint_sha256"] == policy["checkpoint_sha256"]
             and metadata["sidecar_source"].get("source_id") == policy["job_id"]
             and metadata["sidecar_source"].get("checkpoint_sha256") == policy["checkpoint_sha256"],
             "facts 결합 식별자가 등록 job·checkpoint와 다르다")
    return {"stage": name, "refs": refs, "facts": facts}


# --- 종합·출력 ------------------------------------------------------------------------------------


def _public_clean(public: Mapping[str, Any], policies: Mapping[str, Any]) -> None:
    text = json.dumps(public, ensure_ascii=False)
    banned = {*policies, *(row["job_id"] for row in policies.values()),
              *(row["protocol_id"] for row in policies.values())}
    _require(not any(item in text for item in banned), "공개 facts에 job·정책 ID가 들어갔다")
    _require(re.search(r"(?:[A-Za-z]:[\\/]|/mnt/|/home/|/Users/|bench/private)", text) is None,
             "공개 facts에 경로가 들어갔다")


def evaluate(
    run_dir: Path, portable_manifest: Path, *, phase: str, phase_sha256_lf: str, repo: Path = ROOT
) -> dict[str, Any]:
    """``{"summary", "metadata", "public"}``. 회수 전이면 Hold, 등록·게이트 위반이면 AcceptanceError."""
    registration = load_registration(portable_manifest, repo=repo, phase=phase, phase_sha256_lf=phase_sha256_lf)
    policies = registration["policies"]
    gate = verify_collection(run_dir, registration)
    errors = _record_entries(gate["evidence"], policies)
    checked: dict[str, dict[str, Any]] = {}
    rows = []
    for identity in sorted(policies):
        policy = policies[identity]
        try:
            checked[identity] = check_policy(policy, gate)
            reason = None
        except (ValueError, SIDECAR.SidecarError, OSError, KeyError, TypeError) as error:  # AcceptanceError·FactsError 포함
            reason = f"{type(error).__name__}: {error}"
        rows.append({"logical_id": identity, "fault_id": policy["fault_id"], "pair_id": policy["pair"]["pair_id"],
                     "role": policy["role"], "technical_accepted": reason is None, "reason": reason})
    digests = [item["facts"]["metadata"]["binding_digest"] for item in checked.values()]
    videos = [item["facts"]["metadata"]["video_sha256"] for item in checked.values()]
    if len(set(digests)) != len(digests) or len(set(videos)) != len(videos):
        errors.append("정책 사이 binding·video가 중복됐다")
    complete = not errors and len(checked) == len(policies)
    public = None
    if complete:
        public = {
            "schema": SCHEMA, "policy_count": len(checked), "frame_count": R.FRAME_COUNT, "fps": R.FPS,
            "policies": sorted((item["facts"]["public"] for item in checked.values()),
                               key=lambda value: value["binding"]["binding_digest"]),
            "fixed_flags": dict(FIXED_FLAGS), "thresholds_created": 0, "behavior_label": None,
            "used_for_decision": False,
        }
        _public_clean(public, policies)
    metadata = {
        "schema": METADATA_SCHEMA,
        "registration": {key: registration[key] for key in ("phase", "phase_sha256_lf", "resource_id",
                                                           "portable_manifest_sha256", "protocols")},
        "collection": {"run": str(gate["run"]), "backup": str(gate["backup"]), "files": gate["files"],
                       "collection_sha256": gate["collection_sha256"], "archive_sha256": gate["archive_sha256"],
                       "inventory_sha256": gate["inventory_sha256"], "c_d_rehashed": True},
        "policies": [{**{key: policies[identity][key] for key in ("logical_id", "role", "fault_id", "pair",
                                                                  "checkpoint_sha256", "protocol", "job_id")},
                      "stage": item["stage"], "binding_digest": item["facts"]["metadata"]["binding_digest"],
                      "public_facts_sha256": item["facts"]["metadata"]["public_sha256"],
                      "video_sha256": item["facts"]["metadata"]["video_sha256"], "refs": item["refs"]}
                     for identity, item in sorted(checked.items())],
        "public_sha256": None if public is None else R.digest(public),
    }
    summary = {
        "schema": SUMMARY_SCHEMA, "status": "technical_complete" if complete else "rejected",
        "policies": rows, "errors": errors, "fixed_flags": dict(FIXED_FLAGS), "thresholds_created": 0,
        "behavior_label": None, "final_acceptance_owner": "main", "metadata_sha256": R.digest(metadata),
    }
    return {"summary": summary, "metadata": metadata, "public": public}


def write_outputs(result: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """write-once. 공개 후보 facts는 네 정책이 모두 기술 수용될 때만 쓴다."""
    out = Path(out_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"출력이 이미 있다: {out}")
    out.mkdir(parents=True)
    paths = {"summary": out / SUMMARY_NAME, "metadata": out / METADATA_NAME}
    R.write_new(paths["metadata"], result["metadata"])
    if result["public"] is not None:
        paths["public"] = out / PUBLIC_NAME
        R.write_new(paths["public"], result["public"])
    R.write_new(paths["summary"], result["summary"])
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="host run-<resource_id> 폴더")
    parser.add_argument("--portable-manifest", required=True, type=Path)
    parser.add_argument("--phase", required=True, help="repo 상대 등록 phase 경로")
    parser.add_argument("--phase-sha256-lf", required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.run, args.portable_manifest, phase=args.phase,
                          phase_sha256_lf=args.phase_sha256_lf, repo=args.repo)
    except Hold as error:
        print(json.dumps({"status": "HOLD", "reason": str(error)}, ensure_ascii=False))
        return 3
    except AcceptanceError as error:
        print(json.dumps({"status": "rejected_before_policies", "reason": str(error)}, ensure_ascii=False))
        return 2
    paths = write_outputs(result, args.out)
    print(json.dumps({"status": result["summary"]["status"], **{key: str(value) for key, value in paths.items()}},
                     ensure_ascii=False))
    return 0 if result["summary"]["status"] == "technical_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
