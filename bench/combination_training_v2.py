"""Task39 ANYmal-C U1 source·교정 arm을 기존 restart 하네스로 실행한다(조합 실행기 v2용)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals")]

import isaac_training_guard_v1 as GUARD  # noqa: E402
import reward_family_correction as FAMILY  # noqa: E402
import train_with_recovery_mode as RUNNER  # noqa: E402


def _binding(env, contract: dict, expect_clean: bool) -> dict:
    import importlib
    module = importlib.import_module(FAMILY.MODULE)
    manager = env.unwrapped.reward_manager
    terms = {name: manager.get_term_cfg(name).func for name in manager.active_terms}
    clean = {name: FAMILY.BASE.fingerprint(getattr(module, name)) == contract["functions"][name]
             for name in FAMILY.FUNCTIONS}
    bound = {name: terms.get(name) is getattr(module, name) for name in FAMILY.FUNCTIONS}
    ok = all(bound.values()) and all(value is expect_clean for value in clean.values())
    row = {"schema": "combination_reward_binding_v1", "clean": clean,
           "bound_to_reward_terms": bound, "expected_clean": expect_clean, "ok": ok}
    if not ok:
        raise RuntimeError("ANYmal reward 함수군의 주입·교정 결합이 manifest와 다르다")
    return row


def install(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(Path(manifest["protocol_path"]).read_text(encoding="utf-8"))
    contract = protocol["correction_source_contract"]
    output = Path(manifest["output_dir"])
    fault, correction = manifest["fault"], manifest["correction_id"]
    if fault == "U1":
        import upstream_isaac
        import upstream_faults
        shim = upstream_isaac.hidden_faults_shim("U1")
        original_arm = shim.arm

        def arm(name):
            import gymnasium
            original_arm(name)
            armed_make = gymnasium.make

            def make(*args, **kwargs):
                env = armed_make(*args, **kwargs)
                row = _binding(env, contract, correction == FAMILY.CORRECTION_ID)
                GUARD.durable_new(output / "family_environment.json", row)
                return env

            gymnasium.make = make

        setattr(shim, "arm", arm)
    elif fault == "NONE":
        import gymnasium
        shim = ModuleType("hidden_faults")

        def before_env(name):
            if name != "NONE":
                raise ValueError("NONE arm에 다른 결함이 들어왔다")

        def arm(name):
            if name != "NONE":
                raise ValueError("NONE arm에 다른 결함이 들어왔다")
            original_make = gymnasium.make
            once = {"done": False}

            def make(*args, **kwargs):
                if once["done"]:
                    return original_make(*args, **kwargs)
                once["done"] = True
                shim._before_env(name)
                env = original_make(*args, **kwargs)
                row = _binding(env, contract, True)
                GUARD.durable_new(output / "family_environment.json", row)
                print("[combination_binding] " + json.dumps(row), flush=True)
                return env

            gymnasium.make = make

        setattr(shim, "_before_env", before_env)
        setattr(shim, "arm", arm)
        setattr(shim, "FAULTS", {"NONE": {"category": "control"}})
    else:
        raise ValueError("등록되지 않은 결함 arm")
    sys.modules["hidden_faults"] = shim
    sys.modules["correction_adapter"] = FAMILY


def _sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _empty_new(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def guarded_execute(manifest_path: Path, helper) -> dict:
    """train.py의 terminal close 앞에서 receipt 완결을 봉인한다."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(manifest["output_dir"])
    upstream = Path(manifest["isaaclab_root"]) / "scripts/reinforcement_learning/rsl_rl/train.py"
    original_run_path = RUNNER.runpy.run_path

    def run_path(path, run_name=None):
        if Path(path).resolve() != upstream.resolve() or run_name != "__main__":
            raise ValueError("등록 train.py __main__만 guarded 실행할 수 있다")

        def before_close(_app) -> None:
            helper.validate_manifest(manifest_path)
            receipt_path = output / "receipt.json"
            if not receipt_path.is_file():
                raise RuntimeError("terminal close 전에 학습 완료 receipt가 없다")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            manifest_sha = helper.sha(manifest_path, lf=False)
            if (receipt.get("completed") is not True or receipt.get("manifest_sha256") != manifest_sha
                    or receipt.get("actual_control_steps") != 300 * 24):
                raise ValueError("terminal close 전 receipt 예산·manifest 결합이 다르다")
            evidence = {"schema": "combination_training_evidence_v1", "manifest_sha256": manifest_sha,
                        "updates": 300, "receipt_sha256_lf": _sha_lf(receipt_path),
                        "recorded_before_app_close": True}
            GUARD.durable_new(output / GUARD.EVIDENCE, evidence)
            _empty_new(output / GUARD.RECORDS)
            GUARD.durable_new(output / GUARD.CLOSE_READY, {
                "schema": GUARD.CLOSE_SCHEMA, "manifest_sha256": manifest_sha, "updates": 300,
                "evidence_sha256_lf": _sha_lf(output / GUARD.EVIDENCE),
                "records_sha256_lf": _sha_lf(output / GUARD.RECORDS),
            })

        return GUARD.run_upstream(upstream, output, before_close)

    RUNNER.runpy.run_path = run_path
    try:
        receipt = RUNNER.execute(manifest_path, helper)
    finally:
        RUNNER.runpy.run_path = original_run_path
    GUARD.validate_completion(output, helper.sha(manifest_path, lf=False))
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--lease-id", required=True)
    args = parser.parse_args(argv)
    try:
        import combination_trial_v2 as H
        H.assert_lease(args.lease_id)
        H.validate_manifest(args.manifest)
        install(args.manifest)
        guarded_execute(args.manifest, H)
        H.assert_lease(args.lease_id)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        H.verify_source_anchor(manifest["source_commit_sha"], Path(manifest["isaaclab_root"]))
        return 0
    except BaseException as exc:
        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            GUARD.record_failure(Path(manifest["output_dir"]), exc, "combination_training")
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            sys.stdout.flush(); sys.stderr.flush()
        finally:
            os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
