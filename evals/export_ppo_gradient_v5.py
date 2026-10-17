"""PPO 계산 검사 v5의 원본을 재검증하고 공개 증거를 write-once로 추출한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import ppo_gradient_check_v5 as Q
import ppo_gradient_check_v4 as V4

PRIVATE = ROOT / "bench/private/codex_support_20261015"
RUN = PRIVATE / "task37/runs/ppo_gradient_v5_20261017"
DRY = PRIVATE / "task37/dry_runs/ppo_gradient_v5_20261017_dry"
PREVIOUS = PRIVATE / "task36/runs/ppo_gradient_v4_20261017"
CPU = PRIVATE / "task38/cpu_counts.json"
AUDIT = PRIVATE / "task38/export_receipt.json"
OUTPUT = ROOT / "evals/results/ppo_gradient_v5_20261017"
ANCHOR = "4fc28e605f0769b1a88a1fbbcfe5589afa5f035c"
PROTOCOL_SHA_LF = "ae7b1c7387f27c852b281d92ecd9f08f1f4aa6e6e94d08791a2341605cfa5907"
RESULT_SHA_LF = "99025f3349e0b2ce9b3a0dea8bb647554b44092e0aa124017b13e03662000a43"
DRY_SHA_LF = "01ae3e9434364fd7e4a450e018fa4ea36c80b39801bb440de3bd028e00d413a1"
V4_SHA_LF = "0989938d5583afd5d2bef3b3b3863cc9cca6b59652b102528265980a591828a0"
# 고정 원본으로 build한 공개 파일들의 (상대경로, LF SHA) manifest를 코드에 결합한다.
PUBLIC_ROOT_SHA256 = "1877ab5bcb512091bb1840c5cc745af21cff818ffdfb34767b61967f6c3340ae"
FORBIDDEN = re.compile(r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|bench[\\/]private|codex_support", re.I)
ARTIFACTS = ("manifest.json", "records.jsonl", "training_evidence.json", "close_ready.json", "inspection.json")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def encode(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def public_safe(data: bytes) -> None:
    if FORBIDDEN.search(data.decode("utf-8-sig")):
        raise ValueError("공개 내용에 사용자 또는 비공개 경로가 있다")


def inventory(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): Q.sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def validate_source_inventory(folder: Path, expected_result_sha: str) -> dict:
    if Q.sha(folder / "result.json") != expected_result_sha:
        raise ValueError("고정 원본 결과 SHA가 다르다")
    source = Q.read(folder / "result.json")
    found = inventory(folder)
    del found["result.json"]
    if found != source["artifact_sha256_lf"]:
        raise ValueError("원본 artifact 목록 또는 SHA가 다르다")
    return source


def replay_source(module, folder: Path, source: dict, protocol: dict, *, dry=False) -> list[dict]:
    specs = [module.dry_spec()] if dry else module.specs()[:source["completed_runs"]]
    rows = []
    for spec in specs:
        arm = folder / spec["id"]
        execution = Q.read(arm / "execution.json")
        Q.OLD.verify_execution(execution)
        row = module.inspect_run(arm, spec, protocol, source["experiment_commit"])
        row["owned_process_wall_s"] = execution["wall_s"]
        if Q.read(arm / "inspection.json") != row:
            raise ValueError("저장 inspection과 원본 재계산이 다르다")
        rows.append(row)
    summary = module.summarize_dry(rows) if dry else module.summarize(rows)
    if any(source.get(key) != value for key, value in summary.items()):
        raise ValueError("원본 요약과 판정 재계산이 다르다")
    return rows


def execution_projection(source: dict, source_sha: str) -> dict:
    fields = ("job_assigned_before_release", "natural_exit", "timed_out", "error_type", "exit_code",
        "cleanup_verified", "wall_s")
    return {"schema": "ppo_public_execution_v1", **{k: source[k] for k in fields},
        "source_execution_sha256_lf": source_sha,
        "omitted": ["process_identities", "host_process_metadata"]}


def verify_and_build(isaaclab: Path) -> dict[str, bytes]:
    if Q.sha(Q.PROTOCOL) != PROTOCOL_SHA_LF:
        raise ValueError("등록 프로토콜 SHA가 다르다")
    protocol, _ = Q.verify(isaaclab, ANCHOR)
    source = validate_source_inventory(RUN, RESULT_SHA_LF)
    dry = validate_source_inventory(DRY, DRY_SHA_LF)
    if source["experiment_commit"] != ANCHOR or dry["experiment_commit"] != ANCHOR:
        raise ValueError("실험 커밋이 다르다")
    Q.verify_dry_run(DRY.name, isaaclab)
    rows = replay_source(Q, RUN, source, protocol)
    dry_rows = replay_source(Q, DRY, dry, protocol, dry=True)
    if source["outcome"] != "confirmed" or len(rows) != 8 or not all(source["checks"].values()):
        raise ValueError("confirmed 결과의 등록 조건이 다르다")
    previous = validate_source_inventory(PREVIOUS, V4_SHA_LF)
    old_protocol, _ = V4.verify(isaaclab, previous["experiment_commit"])
    replay_source(V4, PREVIOUS, previous, old_protocol)
    if previous["outcome"] != "observer_cost_exceeded":
        raise ValueError("v4 실측 상태가 다르다")

    payloads: dict[str, bytes] = {}
    source_roles = {}
    for group, folder, specs in (("runs", RUN, Q.specs()), ("dry_run", DRY, [Q.dry_spec()])):
        for spec in specs:
            arm = folder / spec["id"]
            for name in ARTIFACTS:
                key = f"{group}/{spec['id']}/{name}"
                payloads[key] = (arm / name).read_bytes()
                source_roles[key] = {"sha256_lf": Q.sha(arm / name),
                    "sha256_raw": hashlib.sha256(payloads[key]).hexdigest(), "export": "unchanged"}
            key = f"{group}/{spec['id']}/execution.json"
            sha = Q.sha(arm / "execution.json")
            payloads[key] = encode(execution_projection(Q.read(arm / "execution.json"), sha))
            source_roles[key] = {"sha256_lf": sha, "export": "allowlisted_projection"}

    payloads["result.json"] = encode({**Q.summarize(rows), "experiment_commit": ANCHOR,
        "protocol_sha256_lf": PROTOCOL_SHA_LF, "tag": source["tag"], "dry_run": False})
    payloads["dry_run_result.json"] = encode({**Q.summarize_dry(dry_rows), "experiment_commit": ANCHOR,
        "protocol_sha256_lf": PROTOCOL_SHA_LF, "tag": dry["tag"], "dry_run": True})
    fields = ("status", "outcome", "pairs", "checks", "completed_runs", "planned_runs", "experiment_commit", "protocol_sha256_lf")
    payloads["v4_comparison.json"] = encode({"schema": "ppo_observer_cost_comparison_v1",
        "v4": {k: previous[k] for k in fields}, "v5": {k: source[k] for k in fields},
        "metric": protocol["cost"]["metric"], "max_ratio": protocol["cost"]["upper_bound_each_seed"],
        "development_coupling": protocol["supersedes"]["development_coupling"],
        "same_seed_comparison": False, "gpu_time_saving_claimed": False})

    cpu = Q.read(CPU)
    if (cpu["device"] != "cpu" or cpu["updates"] != 3 or cpu["gpu_executions"] != 0
            or cpu["api_requests"] != 0 or not cpu["state_and_losses_exact"]
            or [r["label"] for r in cpu["runs"]] != ["OFF", "V4_ON", "V5_ON"]
            or [r["counts"]["item"] for r in cpu["runs"]] != [726, 2211, 726]):
        raise ValueError("CPU 호출 수 근거가 다르다")
    for name, expected in cpu["source_sha256_lf"].items():
        if Q.sha(ROOT / name) != expected:
            raise ValueError("CPU profile 소스 결합이 다르다")
    payloads["cpu_profile.json"] = encode({**cpu, "schema": "ppo_cpu_call_counts_public_v1",
        "scope": "이미 본 합성 CPU seed 17의 실제 RSL learn 전체 fixture 호출 수다. GPU 시간 판정에 쓰지 않는다.",
        "method": "bench/profile_ppo_observer_v1.py의 _exercise와 _profile_summary, cProfile 사용",
        "source_cpu_counts_sha256_lf": Q.sha(CPU)})
    payloads["provenance.json"] = encode({"schema": "ppo_gradient_public_provenance_v1",
        "experiment_commit": ANCHOR, "protocol": Q.PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_sha256_lf": PROTOCOL_SHA_LF,
        "source_results_sha256_lf": {"v5": RESULT_SHA_LF, "dry_run": DRY_SHA_LF, "v4": V4_SHA_LF},
        "source_roles": source_roles,
        "source_inventory_sha256": {"v5": digest(inventory(RUN)), "dry_run": digest(inventory(DRY)), "v4": digest(inventory(PREVIOUS))},
        "excluded": ["lease metadata", "launch commands", "host paths", "stdout/stderr", "GPU/process metadata"],
        "validation": {"source_commit_and_protocol_verified": True, "source_artifact_inventory_verified": True,
            "dry_run_verified": True, "v5_arms_recomputed": len(rows), "v4_arms_recomputed": 6},
        "payload_sha256_lf": {name: sha_bytes(data) for name, data in payloads.items()}})
    for data in payloads.values():
        public_safe(data)
    return payloads


def verify_payloads(payloads: dict[str, bytes]) -> None:
    for name, data in payloads.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("공개 상대경로가 유효하지 않다")
        public_safe(data)
    hashes = {name: sha_bytes(data) for name, data in payloads.items()}
    if not PUBLIC_ROOT_SHA256 or digest(hashes) != PUBLIC_ROOT_SHA256:
        raise ValueError("공개 canonical manifest 결합이 다르다")
    provenance = json.loads(payloads["provenance.json"])
    if provenance["payload_sha256_lf"] != {k: v for k, v in hashes.items() if k != "provenance.json"}:
        raise ValueError("공개 provenance 결합이 다르다")


def verify_public(folder: Path = OUTPUT) -> dict:
    payloads = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    verify_payloads(payloads)
    protocol = Q.read(Q.PROTOCOL)
    if Q.sha(Q.PROTOCOL) != PROTOCOL_SHA_LF:
        raise ValueError("공개 검증의 프로토콜이 다르다")
    for group, specs, name, summarize in (("runs", Q.specs(), "result.json", Q.summarize),
            ("dry_run", [Q.dry_spec()], "dry_run_result.json", Q.summarize_dry)):
        rows = []
        for spec in specs:
            arm = folder / group / spec["id"]
            row = Q.inspect_run(arm, spec, protocol, ANCHOR)
            execution = Q.read(arm / "execution.json")
            if (execution["exit_code"] != 0 or not execution["natural_exit"] or execution["timed_out"]
                    or execution["error_type"] is not None or not execution["cleanup_verified"]):
                raise ValueError("공개 실행 종료 증거가 다르다")
            row["owned_process_wall_s"] = execution["wall_s"]
            if row != Q.read(arm / "inspection.json"):
                raise ValueError("공개 검사 증거 재생 불일치")
            rows.append(row)
        recorded = Q.read(folder / name)
        if any(recorded.get(k) != v for k, v in summarize(rows).items()):
            raise ValueError("공개 판정 재계산 불일치")
    return {"status": "valid", "files": len(payloads), "outcome": "confirmed", "checks": 7,
        "recomputed_arms": 8, "dry_run_verified": True, "public_root_sha256": PUBLIC_ROOT_SHA256}


def export(isaaclab: Path) -> dict:
    if OUTPUT.exists() or AUDIT.exists():
        raise FileExistsError("기존 공개 결과·감사 기록은 덮어쓰지 않는다")
    before = {name: inventory(path) for name, path in (("v5", RUN), ("dry_run", DRY), ("v4", PREVIOUS))}
    payloads = verify_and_build(isaaclab)
    verify_payloads(payloads)
    after = {name: inventory(path) for name, path in (("v5", RUN), ("dry_run", DRY), ("v4", PREVIOUS))}
    if before != after:
        raise ValueError("추출 중 원본이 바뀌었다")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        path = OUTPUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    checked = verify_public()
    Q.write_new(AUDIT, {"source_before": before, "source_after": after, "verification": checked,
        "gpu_executions": 0, "api_requests": 0})
    return {"status": "exported", "files": len(payloads), "source_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "export", "verify"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    if args.command == "build":
        payloads = verify_and_build(args.isaaclab)
        result = {"files": len(payloads), "public_root_sha256": digest({n: sha_bytes(d) for n, d in payloads.items()})}
    elif args.command == "export":
        result = export(args.isaaclab)
    else:
        result = verify_public()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
