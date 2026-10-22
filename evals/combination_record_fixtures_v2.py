"""Go2 모터 결함(F2) 공개 시연 두 건을 적용성 계약 v2 manifest로 정규화한다.

v1 생성기(combination_record_fixtures_v1.py)와 같은 방식이다. 공개 fixture는 이 도구가 만든 공개 record만 참조하고,
원본 상대경로와 LF SHA는 이 코드의 ``SOURCES``에만 둔다. build 때마다 원본을 다시 대조한다.
과거 사례의 ``ready``는 계약 개발을 위한 replay이며 당시 실행의 상태명이 아니다.
선택 교정 NONE 대조는 시연과 같은 Isaac Python 3.10에서 bench/combination_noop_v2.py로 다시 만들었다.
함수 지문은 바이트코드로 계산하므로 Python 버전이 다르면 값이 달라진다.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import combination_contract_v2 as C  # noqa: E402
import combination_record_fixtures_v1 as R1  # noqa: E402

OUTPUT = ROOT / "bench/fixtures/combination_contract_v2"
PUBLIC_PREFIX = "bench/fixtures/combination_contract_v2"
SCHEMA, RECORD_SCHEMA = R1.SCHEMA, R1.RECORD_SCHEMA
PUBLIC_SHA256 = "f3329e855c89046f7ef5fcb9b55f85b1dde1d27ba68ee029504ecbd120a6a466"
CORRECTION = "restore_motor_torque_speed"
TARGET = "DCMotor._clip_effort"

SOURCES = {
    "s2046_result": (
        "evals/results/e2e_demo_v4_f2_s2046_20261014/demo_result.json",
        "5dced87d92740479037aed0e13296df50d7c20a04b038f24bf47adc3fe9db408"),
    "s2046_none": (
        "evals/results/e2e_demo_v4_f2_s2046_20261014/runs/baseline_p0c_s2046.json",
        "a1e0f89b545e6ce8f7159da3806aa5a12eaec9615587bc6f7c457ad5fb216c77"),
    "s2046_fault": (
        "evals/results/e2e_demo_v4_f2_s2046_20261014/runs/h06_s2046.json",
        "a51d8eef0b3579bda01913beb2b55e985a0ce2abb2493977b09125fa93a0a75c"),
    "s2046_protocol": (
        "bench/protocols/e2e_demo_v4_f2.json",
        "f166882dfb21aa42e459babf90c842746b243440d9f3fb5139242313042515da"),
    "s2046_readback": (
        "bench/private/e2e_demo_v4/e2e_demo_v4_f2_s2046_20261014/fix/restart_corrected/train/correction.json",
        "e3b6e72cc4892ecad35468f0cf7a5d4c882d8b70d75323db5e4829d445debb22"),
    "s2046_reference": (
        "evals/loops/h06_s2046/reference.json",
        "00dd9be3c63ebb4b58ce2623e8a3b34ad3ef47827b21979ca8d98c886ac8f915"),
    "s2033_result": (
        "evals/results/e2e_demo_v3_1_f2_s2033_20261013/demo_result.json",
        "a13bfe9ad5a610bb3a547b81add9f3c52d82be6fe4a4eada08b7267bec42c14d"),
    "s2033_none": (
        "evals/results/e2e_demo_v3_1_f2_s2033_20261013/runs/baseline_p0c_s2033.json",
        "ee1aab651e13edf321a54557e181080157cbd79868ad6326a77b604b16e3e4c6"),
    "s2033_fault": (
        "evals/results/e2e_demo_v3_1_f2_s2033_20261013/runs/h06_s2033.json",
        "bc68d85c196ccc3d0d391e98fb67826213e910f588f1ac4e4078e8d43f35427a"),
    "s2033_protocol": (
        "bench/protocols/e2e_demo_v3_1_f2.json",
        "1275ec610dd8be1418afd5c1924cc19d2e885400c1dfa9a01b3dc260fcb37d91"),
    "s2033_readback": (
        "bench/private/e2e_demo_v3_1/e2e_demo_v3_1_f2_s2033_20261013/fix/restart_corrected/train/correction.json",
        "e3b6e72cc4892ecad35468f0cf7a5d4c882d8b70d75323db5e4829d445debb22"),
    "s2033_reference": (
        "evals/loops/h06_s2033/reference.json",
        "30d34f1952c3a9019d127fae7db818fc6d1f85934c29f412fbe973f58690cc3a"),
    "f2_none_noop": (
        "bench/private/claude_reward_track_20261013/combination_v2/f2_none_noop_isaac310_v1.json",
        "648e53608535f655b113692e7ea82a65da105b19f8a2f1d95b01c53a292b827a"),
    "f2_none_noop_runner": (
        "bench/combination_noop_v2.py",
        "ac71972142cfacf2dc9813598cf9bc63e0ac02f0f180586fb15bc092a4e450b6"),
    "adapter": (
        "bench/correction_adapter.py",
        "71224801d07a136c999770a6296e017165579377ee96ab98fa36fc70aa11aaa8"),
    "adapter_test": (
        "tests/test_correction_adapter.py",
        "2b74ef86ef7ba7ba2767c45228b4394f814225027456580ad1d90e7d7bd67025"),
}

# case → (seed, 공개 loop 기록, 시연 회차)
CASES = {"go2_f2_ready_s2046": (2046, "h06_s2046", "e2e_demo_v4"),
         "go2_f2_ready_s2033": (2033, "h06_s2033", "e2e_demo_v3_1")}
EXPECTED = dict.fromkeys(CASES, "ready")


def _verified_source_path(root: Path, name: str) -> Path:
    relative, expected = SOURCES[name]
    path = Path(root) / relative
    if path.is_symlink() or not path.is_file() or R1.sha(path) != expected:
        raise ValueError(f"원본 SHA 결합이 다르다: {name}")
    return path


def _source(root: Path, name: str) -> dict:
    path = _verified_source_path(root, name)
    value = R1._json(path)
    if path.suffix != ".json" or not isinstance(value, dict):
        raise ValueError("관측 원본은 JSON 객체여야 한다")
    return value


def verify_sources(root: Path = ROOT) -> dict[str, str]:
    for name in SOURCES:
        _verified_source_path(root, name)
    return {name: expected for name, (_, expected) in SOURCES.items()}


def _provenance(names: tuple[str, ...]) -> dict[str, str]:
    return {name: SOURCES[name][1] for name in names}


def _f2(root: Path, case: str) -> tuple[dict, dict]:
    seed, loop_case, _ = CASES[case]
    key = f"s{seed}"
    result = _source(root, f"{key}_result")
    none, fault = _source(root, f"{key}_none"), _source(root, f"{key}_fault")
    protocol = _source(root, f"{key}_protocol")
    readback = _source(root, f"{key}_readback")
    reference = _source(root, f"{key}_reference")
    noop = _source(root, "f2_none_noop")
    _verified_source_path(root, "adapter")
    _verified_source_path(root, "adapter_test")
    clean = readback["before_fault"][CORRECTION]
    changed = readback["after_fault"][CORRECTION]
    if (readback["selected_correction"] != CORRECTION or readback["changed_targets"] != [CORRECTION]
            or readback["after_correction"][CORRECTION] != clean or clean == changed):
        raise ValueError("F2 교정 readback이 예상과 다르다")
    if (noop["status"] != "passed" or noop["correction_id"] != CORRECTION
            or noop["before"] != noop["after"] or noop["changed_targets"] or noop["restored_targets"]
            or noop["before"][CORRECTION] != clean
            or noop["source_sha256_lf"] != SOURCES["f2_none_noop_runner"][1]):
        raise ValueError("F2 선택 교정 NONE CPU 대조가 원본 지문과 다르다")
    confirming = [row for row in result["loop"]["requests"] if row["probe"] == "P_torque"]
    if (result["fault"] != "F2" or result["outcome"] != "recovered" or len(confirming) != 1
            or confirming[0]["outcome"] != "abnormal" or result["loop"]["status"] != "confirmed"):
        raise ValueError("F2 시연 기록이 예상과 다르다")
    measurement, ref = confirming[0]["measurement"], reference["P_torque"]
    record = R1._record(case, _provenance((f"{key}_result", f"{key}_none", f"{key}_fault", f"{key}_protocol",
                                           f"{key}_readback", f"{key}_reference", "f2_none_noop",
                                           "f2_none_noop_runner", "adapter", "adapter_test")), {
        "none": none, "fault": fault, "bands": protocol["bands"],
        "before": {TARGET: clean}, "after": {TARGET: changed},
        "scope_measurement": measurement, "scope_reference": ref, "loop_case": loop_case,
        "none_noop_development_replay": {
            "verification": f"correction_adapter.apply(clean_snapshot, {CORRECTION})",
            "runtime": noop["contract"]["runtime"], "scope": noop["scope"], "status": noop["status"],
            "changed_targets": noop["changed_targets"], "restored_targets": noop["restored_targets"],
            "before": {TARGET: noop["before"][CORRECTION]}, "after": {TARGET: noop["after"][CORRECTION]}},
        "development_note": "Go2는 DCMotor를 쓰므로 표적이 활성이다. 효과 도달은 결함 고정 평가 실패와 확정 P_torque로 확인했다."})
    stages = {
        "applicability": {"targets": {TARGET: R1._target("isaaclab.actuators.actuator_pd:" + TARGET)}},
        "injection": {"before": {TARGET: clean}, "after": {TARGET: changed}, "bound": {TARGET: True}},
        "source": {"none": none, "fault": fault, "bands": protocol["bands"]},
        "scope": {"probe": "P_torque", "measurement": measurement, "reference": ref},
        "noop": {"before": {TARGET: clean}, "after": {TARGET: clean}, "unrelated_changed": []},
        "recovery": {"mode": "restart"},
    }
    return record, C.make_manifest("Go2", C.ROBOTS["Go2"][0], "F2", CORRECTION, stages, [])


def payloads(root: Path = ROOT) -> dict[str, bytes]:
    """원본을 검증하고 공개 record와 manifest fixture를 메모리에서 만든다."""
    verify_sources(root)
    output: dict[str, bytes] = {}
    for case in CASES:
        record, manifest = _f2(Path(root), case)
        record["combination"] = {key: manifest[key] for key in ("robot", "task", "fault", "correction_id")}
        record["stages"] = deepcopy(manifest["stages"])
        record_rel = f"records/{case}.json"
        record_raw = R1._bytes(record)
        output[record_rel] = record_raw
        manifest["evidence"] = [{"path": f"{PUBLIC_PREFIX}/{record_rel}",
                                 "sha256_lf": hashlib.sha256(record_raw).hexdigest()}]
        fixture = {"schema": SCHEMA, "case": case, "expected_status": EXPECTED[case],
                   "development_replay": True, "historical_ready_is_contract_development_state": True,
                   "manifest": manifest}
        output[f"cases/{case}.json"] = R1._bytes(fixture)
    output["index.json"] = R1._bytes({"schema": "combination_contract_history_index_v1", "contract": C.VERSION,
        "cases": [{"case": case, "expected_status": EXPECTED[case], "fixture": f"cases/{case}.json"}
                  for case in CASES]})
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


def inventory_digest(output: Path = OUTPUT) -> str:
    output = Path(output)
    files = sorted(p.relative_to(output).as_posix() for p in output.rglob("*.json"))
    return C.L.digest({name: R1.sha(output / name) for name in files})


def verify_public(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    """공개 fixture만으로 상태 전이와 public record SHA를 재검증한다(비공개 원본 없이 실행)."""
    root, output = Path(root), Path(output)
    index = R1._json(output / "index.json")
    if index.get("schema") != "combination_contract_history_index_v1" or index.get("contract") != C.VERSION:
        raise ValueError("fixture index schema가 다르다")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*.json")}
    expected = {"index.json"} | {f"cases/{c}.json" for c in CASES} | {f"records/{c}.json" for c in CASES}
    if actual != expected:
        raise ValueError("fixture 파일 집합이 다르다")
    if inventory_digest(output) != PUBLIC_SHA256:
        raise ValueError("공개 fixture canonical SHA 결합이 다르다")
    statuses = {}
    for case in CASES:
        fixture = R1._json(output / f"cases/{case}.json")
        if (fixture.get("schema") != SCHEMA or fixture.get("case") != case
                or fixture.get("expected_status") != EXPECTED[case] or fixture.get("development_replay") is not True):
            raise ValueError("fixture 본문이 다르다")
        checked = C.evaluate(deepcopy(fixture["manifest"]), root=root)
        if checked["status"] != EXPECTED[case]:
            raise ValueError(f"{case} 상태가 다르다: {checked}")
        statuses[case] = {"status": checked["status"], "trace": [row["stage"] for row in checked["trace"]]}
    return {"status": "valid", "files": len(actual), "cases": statuses}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "digest"))
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build()
    elif args.command == "digest":
        result = {"inventory_digest": inventory_digest()}
    else:
        result = verify_public()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
