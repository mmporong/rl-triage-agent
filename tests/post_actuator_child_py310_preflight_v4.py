"""Python 3.10에서 post child의 validate-only 경계를 독립 프로세스로 확인한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n").encode()


def write(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(root: Path, condition: str) -> Path:
    run = root / "bench/private/post_actuator_confirmation/runs" / condition.lower()
    consumed = root / "bench/private/post_actuator_confirmation/consumed"
    protocol = write(root / "bench/protocols/outer.json", {
        "schema": "post_actuator_operational_protocol_v5", "status": "registered",
        "condition": "NONE", "max_sdk_launches": 1,
        "child_entrypoint": "bench/post_actuator_child_v4.py",
    })
    base = {
        "operation_id": "operation", "scope_key": "scope", "condition": condition,
        "checkpoint_sha256": "a" * 64, "recovery_receipt_sha256": "b" * 64,
        "authority_sha256": "c" * 64, "effect_key": "effect",
    }
    intent = write(run / "intent.json", {
        "schema": "post_actuator_operation_intent_v2", "status": "reserved_pre_sdk",
        **base, "outer_protocol": protocol,
    })
    consumption = write(consumed / f"{condition.lower()}.json", {
        "schema": "post_actuator_global_consumption_v2", "status": "consumed_pre_sdk",
        **base, "outer_protocol": protocol, "intent": intent,
    })
    manifest = write(root / f"bench/private/producer/{condition}/manifest.json", {
        "schema": "actuator_envelope_runtime_manifest_v1", "condition": condition,
    })
    return Path(write(run / "child-context.json", {
        "schema": "post_actuator_child_context_v1", **base, "outer_protocol": protocol,
        "intent": intent, "consumption": consumption, "producer_manifest": manifest,
        "body_entrypoint": "bench.actuator_envelope_runtime.child_main",
    })["path"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for condition in ("NONE", "F2", "F3", "F4"):
        context = fixture(args.work_dir / condition.lower(), condition)
        completed = subprocess.run(
            [sys.executable, "-B", str(args.child), "--context", str(context),
             "--validate-only", "--repo-root", str((args.work_dir / condition.lower()).resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="strict", check=False,
        )
        payload = json.loads(completed.stdout.strip())
        expected = 0 if condition == "NONE" else 2
        if completed.returncode != expected:
            raise RuntimeError(f"{condition} validate-only exit differs")
        if condition == "NONE" and (payload.get("sdk_app_started") is not False
                                     or payload.get("forbidden_imports") != []):
            raise RuntimeError("NONE validation imported SDK modules")
        rows.append({"condition": condition, "exit_code": completed.returncode,
                     "status": payload.get("status")})
    print(json.dumps({"schema": "post_actuator_child_py310_preflight_v4", "rows": rows,
                      "sdk_app_launches": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
