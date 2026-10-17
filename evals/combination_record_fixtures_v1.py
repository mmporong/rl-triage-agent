"""기존 다섯 조합 기록을 공통 적용성 manifest로 정규화한다.

공개 fixture는 원본 위치 대신 이 도구가 만든 공개 record를 참조한다. 원본 상대경로와
LF SHA 결합은 이 코드의 ``SOURCES``에만 있고, build 때마다 원본을 다시 대조한다.
과거 사례의 ``ready``는 이번 계약을 개발하기 위한 replay이며 당시 실행의 상태명이 아니다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import combination_contract_v1 as C  # noqa: E402

OUTPUT = ROOT / "bench/fixtures/combination_contract_v1"
SCHEMA = "combination_contract_history_fixture_v1"
RECORD_SCHEMA = "combination_observation_v1"
SHA = re.compile(r"[0-9a-f]{64}")
PUBLIC_SHA256 = "bddf50272870d7c297c2b4cdf804a80fe5b5c2d9354485736ef4d3ffe6f0c624"

# 원본 주소는 공개 fixture에 복사하지 않는다. build는 이 목록의 LF SHA와 실제 파일을 먼저 대조한다.
SOURCES = {
    "f1_none_noop": (
        "bench/private/codex_support_20261015/task39/records/f1_none_noop_v1.json",
        "7e63c79deb8b8e322cffbc261d372fbe081fe6f80213baaab428f5df55ef67d3"),
    "f1_none_noop_runner": (
        "bench/combination_f1_noop_v1.py",
        "8a64494bd5962487fdc2d1f2b72d6e0c6a3bad506c218dc63c97225575934145"),
    "f2_semantics": (
        "bench/private/codex_support_20261015/task13/semantics_result_v2_final.json",
        "341222cb09a49c87bf42253549cb5d01fd2223fbd945d05915f9f8d57f59c5da"),
    "f6_semantics": (
        "bench/private/codex_support_20261015/task27/semantics/anymal_termination_semantics_v1_final3.json",
        "1ba3e5b31e8e620da854fd1d209e0102eb1dc598f934d4ad99da833f64263e4f"),
    "f6_result": (
        "bench/private/codex_support_20261015/task27/runs/anymal_f6_smoke_s2100_20261017/result.json",
        "11808e18b73855fff9fe2d2e27c128d7e9b41d67c53f3fed7ba23ad989676611"),
    "f6_none": (
        "bench/private/codex_support_20261015/task27/runs/anymal_f6_smoke_s2100_20261017/evaluate/NONE.json",
        "ffd88eac4557b02c51082b9a143e388e6ad0c3051ba955b5183a8dbab1c1709a"),
    "f6_fault": (
        "bench/private/codex_support_20261015/task27/runs/anymal_f6_smoke_s2100_20261017/evaluate/F6.json",
        "271127e18a785de0d775b7cb49d1811bae76c4faa9f71197d04c0e47537b0452"),
    "f6_protocol": (
        "bench/protocols/anymal_termination_smoke_v1.json",
        "66cf669f708e30fee2c3515b6aad975c452572c5177fd9c993bf493f5118e412"),
    "u1_legacy_result": (
        "evals/results/e2e_demo_u1_s2040_20261013/demo_result.json",
        "a30f841acc064b15b717f066e2ef81afc025f339a8441c22615f2b37c1a4e21e"),
    "u1_legacy_none": (
        "evals/results/e2e_demo_u1_s2040_20261013/runs/baseline_p0c_s2040.json",
        "191dc2f02d51df1c705e4e41783340607e009d7139b3c409bc1b67771285961a"),
    "u1_legacy_fault": (
        "evals/results/e2e_demo_u1_s2040_20261013/runs/u1_s2040.json",
        "479aafb39938a53dcf566a522247f617ead62211bbfff3212281bb636004872a"),
    "u1_legacy_protocol": (
        "bench/protocols/e2e_demo_u1_v1.json",
        "410e86209c271039f3e93cd05a5c3d440e84e924ff60c2f60c7f85f64c02e52e"),
    "u1_full_none": (
        "bench/private/codex_support_20261015/task11/u1/source_results/novel_u1_scope_source_s2062/runs/baseline_p0c_s2062.json",
        "9b945305d8e69c73c748c5484d91f2367ec97ba2f1bc750d0cdc442ce4229d1e"),
    "u1_full_fault": (
        "bench/private/codex_support_20261015/task11/u1/source_results/novel_u1_scope_source_s2062/runs/u1_s2062.json",
        "263972be69007fedb697f13cc04f293bb52fcbbb5c9bcd4b921ae375a30666e1"),
    "u1_full_fault_readback": (
        "bench/private/codex_support_20261015/task11/u1/runs/novel_u1_scope_s2062_20261016/u1_family_restart/train/correction.json",
        "44d5c1292e99ccc0408ad8b0eba831fc18b46c6acd8db71e955af89728509787"),
    "u1_full_none_readback": (
        "bench/private/codex_support_20261015/task11/u1/runs/novel_u1_scope_s2062_20261016/none_family_control/train/correction.json",
        "8e7247f61ddf396de08121c008ab5d6e90987e57e8896c5977bf397c4fc2ef25"),
    "u1_full_scope": (
        "evals/results/novel_u1_scope_confirmation_s2062_20261016/scope_evidence.json",
        "959a4fd3199dbbaa05a841d2f27a0948cf75dd996c1d6be1db92c96530d3a1ee"),
    "u1_full_protocol": (
        "bench/protocols/novel_u1_scope_source/e2e_demo_u1_v1.json",
        "ca497986788d71ede3330fe602663494d3f172b46f1c16bc7fa26f65c1b818af"),
    "f1_result": (
        "evals/results/vl_aux_f1_s2080_20261016/demo_result.json",
        "549f785336be2cf3d6d135050efdc4ecffe6008deef5bf054a7c7e990cce8667"),
    "f1_none": (
        "evals/results/vl_aux_f1_s2080_20261016/runs/baseline_p0c_s2080.json",
        "6e0140d39ea2257dad1d8e302c8ba8a3313676231cd8eeaebc9779b416cb46cd"),
    "f1_fault": (
        "evals/results/vl_aux_f1_s2080_20261016/runs/h02_s2080.json",
        "1d255901423849474e3844f043ec40c9f199ca5eb5dbd57ffe65566222f9ee84"),
    "f1_readback": (
        "bench/private/codex_support_20261015/task23/runs/vl_aux_f1_s2080_20261016/fix/restart_corrected/train/correction.json",
        "064aaa23ac2aa48185dc4a5dcfc6369bec4adeb439ad09123461e4010de5f608"),
    "f1_protocol": (
        "bench/protocols/vl_aux_demo_v1/e2e_demo_v4_f1.json",
        "278a8c8310079c9227de724b3504f07e588d97b9014222c0521cb5a42bb05c7d"),
    "adapter": (
        "bench/correction_adapter.py",
        "71224801d07a136c999770a6296e017165579377ee96ab98fa36fc70aa11aaa8"),
    "adapter_test": (
        "tests/test_correction_adapter.py",
        "2b74ef86ef7ba7ba2767c45228b4394f814225027456580ad1d90e7d7bd67025"),
}

CASES = (
    "anymal_f2_not_applicable",
    "anymal_f6_source_failure_absent",
    "go2_u1_legacy_scope_insufficient",
    "go2_u1_extended_ready",
    "go2_f1_ready",
)
EXPECTED = {
    "anymal_f2_not_applicable": "not_applicable",
    "anymal_f6_source_failure_absent": "source_failure_absent",
    "go2_u1_legacy_scope_insufficient": "scope_insufficient",
    "go2_u1_extended_ready": "ready",
    "go2_f1_ready": "ready",
}


def sha(path: Path) -> str:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verified_source_path(root: Path, name: str) -> Path:
    relative, expected = SOURCES[name]
    path = Path(root) / relative
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError(f"원본 SHA 결합이 다르다: {name}")
    return path


def _source(root: Path, name: str) -> dict:
    path = _verified_source_path(root, name)
    if path.suffix != ".json":
        raise ValueError("관측 원본은 JSON이어야 한다")
    value = _json(path)
    if not isinstance(value, dict):
        raise ValueError("관측 원본은 객체여야 한다")
    return value


def verify_sources(root: Path = ROOT) -> dict[str, str]:
    """build 입력 전체가 고정 LF SHA와 일치하는지 검사한다."""
    for name in SOURCES:
        _verified_source_path(root, name)
    return {name: expected for name, (_, expected) in SOURCES.items()}


def _provenance(names: tuple[str, ...]) -> dict[str, str]:
    return {name: SOURCES[name][1] for name in names}


def _target(path: str, *, reachable: bool = True) -> dict:
    return {"path": path, "exists": True, "active": True, "effect_reachable": reachable}


def _record(case: str, source_fingerprints: dict[str, str], payload: dict) -> dict:
    return {"schema": RECORD_SCHEMA, "case": case,
            "measurement_kind": "post_result_contract_development_replay",
            "source_fingerprints": source_fingerprints, "payload": payload}


def _f2(root: Path) -> tuple[dict, dict]:
    sem = _source(root, "f2_semantics")
    if sem["outcome"] != "not_applicable" or sem["structural"]["calls_inherited_clip"] is not True:
        raise ValueError("F2 의미 기록이 예상과 다르다")
    record = _record("anymal_f2_not_applicable", _provenance(("f2_semantics",)), {
        "applicability": {"calls_inherited_clip": True,
            "lstm_compute_writes_clip_joint_velocity": sem["structural"]["lstm_compute_writes_clip_joint_velocity"],
            "effect_reachable": False},
    })
    stages = {"applicability": {"targets": {
        "DCMotor._clip_effort": _target("isaaclab.actuators.actuator_pd:DCMotor._clip_effort", reachable=False)}}}
    return record, C.make_manifest("ANYmal-C", C.ROBOTS["ANYmal-C"][0], "F2",
                                   "restore_motor_torque_speed", stages, [])


def _f6(root: Path) -> tuple[dict, dict]:
    sem = _source(root, "f6_semantics")
    result = _source(root, "f6_result")
    none, fault = _source(root, "f6_none"), _source(root, "f6_fault")
    protocol = _source(root, "f6_protocol")
    readback = sem["correction_readback"]
    if sem["outcome"] != "applicable" or result["outcome"] != "source_failure_absent":
        raise ValueError("F6 기존 상태가 예상과 다르다")
    bands = protocol["bands"]
    record = _record("anymal_f6_source_failure_absent",
        _provenance(("f6_semantics", "f6_result", "f6_none", "f6_fault", "f6_protocol")), {
            "none": none, "fault": fault, "bands": bands,
            "clean_fingerprint": readback["clean_fingerprint"],
            "fault_fingerprint": readback["fault_fingerprint"],
            "runtime_signature": result["runtime_injection_signature"]})
    stages = {
        "applicability": {"targets": {"time_out": _target("isaaclab.envs.mdp.terminations:time_out")}},
        "injection": {"before": {"time_out": readback["clean_fingerprint"]},
                      "after": {"time_out": readback["fault_fingerprint"]}, "bound": {"time_out": True}},
        "source": {"none": none, "fault": fault, "bands": bands},
    }
    return record, C.make_manifest("ANYmal-C", C.ROBOTS["ANYmal-C"][0], "F6",
                                   "restore_configured_timeout", stages, [])


def _u1_common(root: Path) -> tuple[dict, dict]:
    fault_readback = _source(root, "u1_full_fault_readback")
    none_readback = _source(root, "u1_full_none_readback")
    names = tuple(C.FAULTS["U1"][1])
    if (set(fault_readback["changed_targets"]) != set(names)
            or fault_readback["after_correction"] != fault_readback["before_fault"]
            or none_readback["before_fault"] != none_readback["after_fault"]
            or none_readback["after_fault"] != none_readback["after_correction"]):
        raise ValueError("U1 함수군 readback이 예상과 다르다")
    return fault_readback, none_readback


def _u1_legacy(root: Path) -> tuple[dict, dict]:
    result = _source(root, "u1_legacy_result")
    none, fault = _source(root, "u1_legacy_none"), _source(root, "u1_legacy_fault")
    protocol = _source(root, "u1_legacy_protocol")
    fault_readback, _ = _u1_common(root)
    scope = result["scope_check"]["result"]
    measurement = result["loop"]["requests"][0]["measurement"]
    record = _record("go2_u1_legacy_scope_insufficient",
        _provenance(("u1_legacy_result", "u1_legacy_none", "u1_legacy_fault", "u1_legacy_protocol",
                     "u1_full_fault_readback", "u1_full_none_readback")), {
            "none": none, "fault": fault, "bands": protocol["bands"],
            "before": fault_readback["before_fault"], "after": fault_readback["after_fault"],
            "scope_measurement": measurement,
            "development_note": "확장 readback으로 U1 네 활성 표적을 정규화하고 seed 2040의 구 교정 범위를 대조했다."})
    paths = {name: _target("isaaclab.envs.mdp.rewards:" + name) for name in C.FAULTS["U1"][1]}
    stages = {
        "applicability": {"targets": paths},
        "injection": {"before": fault_readback["before_fault"], "after": fault_readback["after_fault"],
                      "bound": dict.fromkeys(C.FAULTS["U1"][1], True)},
        "source": {"none": none, "fault": fault, "bands": protocol["bands"]},
        "scope": {"probe": "P_reward", "measurement": measurement},
    }
    return record, C.make_manifest("Go2", C.ROBOTS["Go2"][0], "U1",
                                   "restore_body_frame_reward", stages, [])


def _u1_full(root: Path) -> tuple[dict, dict]:
    none, fault = _source(root, "u1_full_none"), _source(root, "u1_full_fault")
    protocol = _source(root, "u1_full_protocol")
    scope = _source(root, "u1_full_scope")
    fault_readback, none_readback = _u1_common(root)
    measurement = scope["probe_measurement"]
    names = tuple(C.FAULTS["U1"][1])
    record = _record("go2_u1_extended_ready",
        _provenance(("u1_full_none", "u1_full_fault", "u1_full_fault_readback", "u1_full_none_readback",
                     "u1_full_scope", "u1_full_protocol")), {
            "none": none, "fault": fault, "bands": protocol["bands"],
            "before": fault_readback["before_fault"], "after": fault_readback["after_fault"],
            "none_noop": {"before": none_readback["before_fault"], "after": none_readback["after_correction"]},
            "scope_measurement": measurement})
    stages = {
        "applicability": {"targets": {name: _target("isaaclab.envs.mdp.rewards:" + name) for name in names}},
        "injection": {"before": fault_readback["before_fault"], "after": fault_readback["after_fault"],
                      "bound": dict.fromkeys(names, True)},
        "source": {"none": none, "fault": fault, "bands": protocol["bands"]},
        "scope": {"probe": "P_reward", "measurement": measurement},
        "noop": {"before": none_readback["before_fault"], "after": none_readback["after_correction"],
                 "unrelated_changed": []},
        "recovery": {"mode": "restart"},
    }
    return record, C.make_manifest("Go2", C.ROBOTS["Go2"][0], "U1",
                                   C.F.CORRECTION_ID, stages, [])


def _f1(root: Path) -> tuple[dict, dict]:
    result = _source(root, "f1_result")
    none, fault = _source(root, "f1_none"), _source(root, "f1_fault")
    protocol = _source(root, "f1_protocol")
    readback = _source(root, "f1_readback")
    _verified_source_path(root, "adapter")
    _verified_source_path(root, "adapter_test")
    clean = readback["before_fault"]["restore_body_frame_reward"]
    changed = readback["after_fault"]["restore_body_frame_reward"]
    noop = _source(root, "f1_none_noop")
    if (noop["status"] != "passed" or noop["correction_id"] != "restore_body_frame_reward"
            or noop["before"] != noop["after"] or noop["changed_targets"] or noop["restored_targets"]
            or noop["before"]["restore_body_frame_reward"] != clean
            or noop["source_sha256_lf"] != SOURCES["f1_none_noop_runner"][1]):
        raise ValueError("F1 선택 교정 NONE CPU 대조가 원본 지문과 다르다")
    measurement = result["loop"]["requests"][0]["measurement"]
    record = _record("go2_f1_ready",
        _provenance(("f1_result", "f1_none", "f1_fault", "f1_readback", "f1_protocol",
                     "adapter", "adapter_test", "f1_none_noop", "f1_none_noop_runner")), {
            "none": none, "fault": fault, "bands": protocol["bands"],
            "before": {"track_lin_vel_xy_exp": clean}, "after": {"track_lin_vel_xy_exp": changed},
            "scope_measurement": measurement,
            "none_noop_development_replay": {
                "verification": "correction_adapter.apply(clean_snapshot, restore_body_frame_reward)",
                "scope": noop["scope"], "status": noop["status"],
                "changed_targets": noop["changed_targets"], "restored_targets": noop["restored_targets"],
                "before": {"track_lin_vel_xy_exp": noop["before"]["restore_body_frame_reward"]},
                "after": {"track_lin_vel_xy_exp": noop["after"]["restore_body_frame_reward"]}}})
    stages = {
        "applicability": {"targets": {"track_lin_vel_xy_exp":
            _target("isaaclab.envs.mdp.rewards:track_lin_vel_xy_exp")}},
        "injection": {"before": {"track_lin_vel_xy_exp": clean},
                      "after": {"track_lin_vel_xy_exp": changed},
                      "bound": {"track_lin_vel_xy_exp": True}},
        "source": {"none": none, "fault": fault, "bands": protocol["bands"]},
        "scope": {"probe": "P_reward", "measurement": measurement},
        "noop": {"before": {"track_lin_vel_xy_exp": clean},
                 "after": {"track_lin_vel_xy_exp": clean}, "unrelated_changed": []},
        "recovery": {"mode": "restart"},
    }
    return record, C.make_manifest("Go2", C.ROBOTS["Go2"][0], "F1",
                                   "restore_body_frame_reward", stages, [])


BUILDERS = {
    "anymal_f2_not_applicable": _f2,
    "anymal_f6_source_failure_absent": _f6,
    "go2_u1_legacy_scope_insufficient": _u1_legacy,
    "go2_u1_extended_ready": _u1_full,
    "go2_f1_ready": _f1,
}


def _bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def payloads(root: Path = ROOT) -> dict[str, bytes]:
    """원본을 검증하고 공개 record와 manifest fixture를 메모리에서 만든다."""
    verify_sources(root)
    output: dict[str, bytes] = {}
    for case in CASES:
        record, manifest = BUILDERS[case](Path(root))
        record["combination"] = {key: manifest[key] for key in ("robot", "task", "fault", "correction_id")}
        record["stages"] = deepcopy(manifest["stages"])
        record_rel = f"records/{case}.json"
        record_raw = _bytes(record)
        output[record_rel] = record_raw
        public_path = f"bench/fixtures/combination_contract_v1/{record_rel}"
        manifest["evidence"] = [{"path": public_path, "sha256_lf": hashlib.sha256(record_raw).hexdigest()}]
        fixture = {"schema": SCHEMA, "case": case, "expected_status": EXPECTED[case],
                   "development_replay": True,
                   "historical_ready_is_contract_development_state": case.endswith("_ready"),
                   "manifest": manifest}
        output[f"cases/{case}.json"] = _bytes(fixture)
    index = {"schema": "combination_contract_history_index_v1", "cases": [
        {"case": case, "expected_status": EXPECTED[case], "fixture": f"cases/{case}.json"}
        for case in CASES]}
    output["index.json"] = _bytes(index)
    return output


def build(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    generated = payloads(root)
    output = Path(output)
    if output.exists():
        raise FileExistsError("공개 fixture 폴더가 이미 있다")
    for relative, raw in generated.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return {"status": "built", "files": len(generated), "cases": len(CASES)}


def verify_public(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    """공개 fixture만으로 상태 전이와 public record SHA를 재검증한다."""
    root, output = Path(root), Path(output)
    index = _json(output / "index.json")
    if index.get("schema") != "combination_contract_history_index_v1":
        raise ValueError("fixture index schema가 다르다")
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*.json")}
    expected_files = {"index.json"} | {f"cases/{case}.json" for case in CASES} | {
        f"records/{case}.json" for case in CASES}
    if actual_files != expected_files:
        raise ValueError("fixture 파일 집합이 다르다")
    inventory = {name: sha(output / name) for name in sorted(actual_files)}
    if C.L.digest(inventory) != PUBLIC_SHA256:
        raise ValueError("공개 fixture canonical SHA 결합이 다르다")
    statuses = {}
    for case in CASES:
        fixture = _json(output / f"cases/{case}.json")
        if (set(fixture) != {"schema", "case", "expected_status", "development_replay",
                            "historical_ready_is_contract_development_state", "manifest"}
                or fixture["schema"] != SCHEMA or fixture["case"] != case
                or fixture["expected_status"] != EXPECTED[case] or fixture["development_replay"] is not True):
            raise ValueError("fixture 본문이 다르다")
        checked = C.evaluate(deepcopy(fixture["manifest"]), root=root)
        if checked["status"] != EXPECTED[case]:
            raise ValueError(f"{case} 상태가 다르다: {checked}")
        statuses[case] = {"status": checked["status"], "trace": [row["stage"] for row in checked["trace"]]}
    return {"status": "valid", "files": len(actual_files), "cases": statuses}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    args = parser.parse_args(argv)
    result = build() if args.command == "build" else verify_public()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
