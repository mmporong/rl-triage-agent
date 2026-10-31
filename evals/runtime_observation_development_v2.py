"""동일 입력 개발 비교를 Windows/WSL 명시 경로 alias transport로 실행한다."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import lineage_nim_transport_v4 as T
import runtime_observation_development_v1 as D

SCHEMA = "runtime_observation_development_protocol_v2"
SOURCE_FILES = ("evals/runtime_observation_development_v2.py",
                "tests/test_runtime_observation_development_v2.py", *T.SOURCE_FILES)


def prepare_protocol(v1_protocol: Path, transport_protocol: Path) -> dict:
    v1, transport = D.read(v1_protocol), D.read(transport_protocol)
    D.require(v1["schema"] == D.SCHEMA and transport["schema"] == T.SCHEMA,
              "개발 또는 transport component schema가 다르다")
    D.require(transport["component_v3_protocol"] == v1["components"]["transport"],
              "동일 등록 요청 plan의 transport component가 아니다")
    return {"schema": SCHEMA, "component_development": {
                "path": v1_protocol.resolve().relative_to(ROOT).as_posix(),
                "sha256_lf": D.sha(v1_protocol, lf=True)},
            "component_transport": {"path": transport_protocol.resolve().relative_to(ROOT).as_posix(),
                                    "sha256_lf": D.sha(transport_protocol, lf=True)},
            "source_sha256_lf": {name: D.sha(ROOT / name, lf=True) for name in SOURCE_FILES},
            "replacement": {"prior_queue_provider_dispatches": 0,
                            "prior_queue_closed": True, "semantic_result_observed": False,
                            "reason": "cross_os_absolute_protocol_path_infrastructure_failure"},
            "scope": {"input_predictions_criteria_params_unchanged": True,
                      "gpu_jobs": 0, "same_seen_development_case": True}}


def verify_protocol(path: Path, inputs: Path, anchor: str) -> dict:
    value = D.read(path)
    v1 = ROOT / value["component_development"]["path"]
    transport = ROOT / value["component_transport"]["path"]
    D.require(value == prepare_protocol(v1, transport), "개발 v2 protocol 결합이 다르다")
    D.verify_protocol(v1, inputs, anchor)
    T.verify_protocol(transport, anchor)
    bound = {**value["source_sha256_lf"], path.resolve().relative_to(ROOT).as_posix(): D.sha(path, lf=True)}
    for name, digest in bound.items():
        blob = D.subprocess.run(["git", "show", f"{anchor}:{name}"], cwd=ROOT,
                                capture_output=True, check=False)
        D.require(D.sha(ROOT / name, lf=True) == digest and blob.returncode == 0
                  and hashlib.sha256(blob.stdout.replace(b"\r\n", b"\n")).hexdigest() == digest,
                  "개발 v2 source anchor 불일치")
    return value


def run(protocol: Path, inputs: Path, anchor: str, queue: Path, out: Path) -> dict:
    value = verify_protocol(protocol, inputs, anchor)
    with T.compatibility_scope():
        return D.run(ROOT / value["component_development"]["path"], inputs, anchor, queue, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    for name in ("v1-protocol", "transport-protocol", "out"):
        prepare.add_argument("--" + name, type=Path, required=True)
    execute = sub.add_parser("run")
    for name in ("protocol", "inputs", "queue", "out"):
        execute.add_argument("--" + name, type=Path, required=True)
    execute.add_argument("--anchor", required=True)
    args = parser.parse_args()
    if args.command == "prepare-protocol":
        D.write(args.out, prepare_protocol(args.v1_protocol, args.transport_protocol))
    else:
        run(args.protocol, args.inputs, args.anchor, args.queue, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
