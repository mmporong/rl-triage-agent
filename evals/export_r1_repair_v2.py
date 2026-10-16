"""R1 repair v2 실행 결과를 검증하고 공개 allowlist JSON으로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import r1_repair_contract as CONTRACT  # noqa: E402
import r1_repair_model_call as MODEL_CALL  # noqa: E402
from rl_triage import grounded_recovery as G  # noqa: E402
from rl_triage import recovery as REC  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger  # noqa: E402

TAG = "r1_repair_s2071_20261016"
RUN = ROOT / "bench/private/codex_support_20261015/task21/runs" / TAG
OUTPUT = ROOT / "evals/results" / TAG
AUDIT = ROOT / "bench/private/codex_support_20261015/task22/export" / TAG
INPUTS = {
    "protocol": ROOT / "bench/protocols/r1_repair_demo_v2.json",
    "fixed_eval_protocol": ROOT / "bench/protocols/fixed_eval_v1.json",
    "bands": ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json",
    "result": RUN / "result.json",
    "source_gate": RUN / "source_gate.json",
    "recovery": RUN / "recovery.json",
    "artifact_none": RUN / "artifact_none.json",
    "artifact_r1": RUN / "artifact_r1.json",
    "artifact_corrected": RUN / "artifact_corrected.json",
    "eval_none": RUN / "source_evaluation/none.json",
    "eval_r1": RUN / "source_evaluation/r1.json",
    "eval_corrected": RUN / "correction_evaluation/corrected.json",
    "readback_none": RUN / "training/none/readback.json",
    "readback_r1": RUN / "training/r1/readback.json",
    "readback_corrected": RUN / "correction_training/readback.json",
    "records_none": RUN / "training/none/records.jsonl",
    "records_r1": RUN / "training/r1/records.jsonl",
    "records_corrected": RUN / "correction_training/records.jsonl",
    "probe_optimizer": RUN / "probe_6_P_optimizer_updates/evidence.json",
    "probe_ledger": RUN / "probes.ledger.jsonl",
    "correction_ledger": RUN / "correction.ledger.jsonl",
    "model_diagnosis": RUN / "model/diagnosis.feedback1.json",
    "model_correction": RUN / "model/correction.feedback1.json",
}
EXPECTED_SHA256 = {
    "protocol": "b125f62107724bfc896c50f24c8124d571d3e8595916b07e9fac67c6af50ff61",
    "fixed_eval_protocol": "8d24652eeed47f462d78244fceec785164b7695e5b757ae8a4e9bd7947aa8a46",
    "bands": "7589a132d4e1a55e159527389a8b635a1766472ecde558da0a1a704c286f3f23",
    "result": "8b2320662f7b2ba233a3cdc906ec6dfe03b0a6272f0311fc48c0de6b4c5d36e7",
    "source_gate": "2ebc8f59729d6cd3ceb012f4dfc8eeb866041409c8ab56305124df708d48193b",
    "recovery": "1e3df3c1f0ab339294abc3e57661e343ee2831830927e40da344ef8371f31e7f",
    "artifact_none": "bfc78ee61931ab4453c3d0f8d179f6e7fe816398b3083a319e0e104d7f2d93b0",
    "artifact_r1": "d1af6d5f90221324660f7c35c17cf00968c0c1c799ff376df6404c7d929689aa",
    "artifact_corrected": "46bbd54affab826cfbf86c1e427e0894883bdc0ea128be214d5182125cbd7e43",
    "eval_none": "80aa8687624ff59a4a3c4d69ed1c4ad50a7b2d299a74bb41a1e5b4a0093cdccb",
    "eval_r1": "169c82a0924fe29806f592690f34e2d82574dd70fa4b24a75c154630b4f2d544",
    "eval_corrected": "67802d74413c345607238d119701ff0d2ec3bcf20eee92e14d69da4b6810192b",
    "readback_none": "71a4f841475e0dfd21dde563109a2ab9d542d9e783c1b3d57a113601c82a2cb9",
    "readback_r1": "92bb7c7d337af7e95ab1fbd8f032fc6a38f0002a5ef67b820649a29fc9fd269f",
    "readback_corrected": "084efe27b6ca428362031c4645cb385d8468addf058783821af2f6f6f8cd920d",
    "records_none": "e9506dc8af2e79b27f2868eb64ed3a9317b2804fc58002f7c50ca9652e1477f3",
    "records_r1": "00ea14f150dc5498de7abdf511a41677a9047a65b94d72ba189b3eae44c40a6d",
    "records_corrected": "fb7bad4f6ad1bb428aa2d1d99812fb2e31d5e8902e33135600c2b539fd3a22d6",
    "probe_optimizer": "8c9af3b77f4e51743ab0ceebc0847a03b8ebbf9419f58f0d77f7d89858255411",
    "probe_ledger": "7d1ec12fcd1e7455563358c36609d256c91a185060ce422ec8cdd325ea406124",
    "correction_ledger": "60ea23fec144c739a09eb661eb0df061915d59a79b1246e5d4ab50cd54253b33",
    "model_diagnosis": "e307a3303d07c61e61f5c50efeade94cd4ef2fa7a642c5a1e8b9be8328ec5b4d",
    "model_correction": "b624906fb06f835e9a153c8d409482566deb5b41b3dd854dc009b70af28d7175",
}
VALIDATION_INPUTS = {
    "diagnosis_packet": RUN / "model/diagnosis.packet.json",
    "diagnosis_reservation": RUN / "model/diagnosis.feedback1.reserved.json",
    "correction_packet": RUN / "model/correction.packet.json",
    "correction_reservation": RUN / "model/correction.feedback1.reserved.json",
    "telemetry_none": RUN / "training/none/telemetry.json",
    "telemetry_r1": RUN / "training/r1/telemetry.json",
    "telemetry_corrected": RUN / "correction_training/telemetry.json",
}
EXPECTED_VALIDATION_SHA256 = {
    "diagnosis_packet": "91a6644b320f2561938ef9ba3ceb1072e975fe8afc8cbfea3f975f73039d616e",
    "diagnosis_reservation": "b5b86a139e6b3b9af96c6d2899963c173d74894d5571e772a43b3d285da16f9a",
    "correction_packet": "607cf9164e21f8a1b6c602cd1c3915a76a77f2910deed74fd9e3e61b4bdc89c4",
    "correction_reservation": "1c046726c0c871e80432ef4fca369283dae88e882e5a1af67c94fcfc31931919",
    "telemetry_none": "00525a666a3985174a00d14e9445b372ed57f4a09208380566d416b977f0a789",
    "telemetry_r1": "16c1170aff602be5982cc5c0f24fcfe081db30d9b68703974265c77bc409e6e6",
    "telemetry_corrected": "d37a9fc3a2c120e82c9c2112bb4a2afb843e7c8faf36e0ed92bf801bf9b23df1",
}
PUBLIC_FILES = {
    "summary.json", "evaluations/none.json", "evaluations/r1.json", "evaluations/corrected.json",
    "training_evidence.json", "probe_and_correction.json", "model_evidence.json", "provenance.json",
}
_ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]|/(?:home|Users|mnt)/", re.IGNORECASE)
_PRIVATE = re.compile(r"bench[\\/]private|codex_support|task2[12][\\/]", re.IGNORECASE)
_DANGEROUS_PATH = re.compile(r"^(?:/|\\\\)|(?:^|[\\/])\.\.(?:[\\/]|$)")
_FORBIDDEN_KEYS = {"path", "run_dir", "records_path", "command", "args", "environment", "env",
                   "stack", "traceback", "api_key", "authorization", "secret"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _execution(value: dict) -> dict:
    keys = ("job_assigned_before_release", "natural_exit", "timed_out", "error_type", "exit_code",
            "cleanup_verified", "wall_s")
    row = {key: value.get(key) for key in keys}
    if row != {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
               "error_type": None, "exit_code": 0, "cleanup_verified": True, "wall_s": row["wall_s"]}:
        raise ValueError("execution did not finish cleanly")
    if not isinstance(row["wall_s"], (int, float)) or row["wall_s"] <= 0:
        raise ValueError("execution wall time is invalid")
    return row


def _public_fixed(arm: str, report: dict, verdict: dict, source_sha: str) -> dict:
    rows = [{"target_velocity": row["command"], **{key: row[key] for key in
             ("num_envs", "fall_count", "alive_steps_sum", "lin_error_sq_sum", "yaw_error_sq_sum",
              "fall_rate", "lin_vel_rmse_mps", "yaw_rate_rmse_radps")}} for row in report["by_condition"]]
    return {
        "schema": "public_r1_repair_fixed_evaluation_v1", "arm": arm,
        "label": verdict["label"], "verdict": verdict,
        "protocol": {"name": report["protocol"], "sha256_lf": report["protocol_sha256_lf"],
                     "task": report["task"], "eval_seed": report["eval_seed"],
                     "num_envs": report["num_envs"], "horizon_steps": report["horizon_steps"],
                     "step_dt": report["step_dt"], "action_scale": report["action_scale"]},
        "checkpoint": report["checkpoint"], "metrics": report["metrics"], "standstill": report["standstill"],
        "condition_count": len(rows), "conditions": rows, "source_artifact_sha256": source_sha,
    }


def _training(arm: str, artifact: dict, records: list[dict], readback_sha: str, records_sha: str) -> dict:
    expected = [4, 4, 4] if arm == "R1" else [20, 20, 20]
    observed_steps = [row["observed"]["optimizer_steps"] for row in records]
    observed_batches = [row["observed"]["minibatches"] for row in records]
    if len(records) != 3 or [row["update_index"] for row in records] != [0, 1, 2] \
            or any(row["error"] is not None for row in records) \
            or observed_steps != expected or observed_batches != expected:
        raise ValueError(f"{arm} raw first-three update counts differ")
    if artifact["inspection"] != {"status": "abnormal" if arm == "R1" else "normal",
                                  "optimizer_steps": expected, "minibatches": expected}:
        raise ValueError(f"{arm} inspection summary differs")
    configuration = artifact["configuration"]
    if configuration != {"seed": 3071 if arm == "CORRECTED" else 2071, "max_iterations": 300,
                          "num_learning_epochs": 5, "num_mini_batches": 4}:
        raise ValueError(f"{arm} training configuration differs")
    checkpoint = artifact["checkpoint_evidence"]
    if checkpoint["device"] != "cpu" or checkpoint["first"]["iteration"] != 0 \
            or checkpoint["final"]["iteration"] != 299 or checkpoint["final"]["sha256"] != artifact["checkpoint"]["sha256"]:
        raise ValueError(f"{arm} checkpoint evidence differs")
    readback = artifact["readback"]
    telemetry_names = ("Train/mean_reward", "Train/mean_episode_length",
                       "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw")
    return {
        "arm": arm, "seed": configuration["seed"], "fresh_restart": arm == "CORRECTED",
        "configuration": configuration, "readback": readback,
        "first_three_updates": {"status": artifact["inspection"]["status"],
                                "optimizer_steps": observed_steps, "minibatches": observed_batches},
        "checkpoint": {"file": Path(artifact["checkpoint"]["path"]).name,
                       "sha256": artifact["checkpoint"]["sha256"],
                       "first_iteration": checkpoint["first"]["iteration"],
                       "final_iteration": checkpoint["final"]["iteration"],
                       "model_state_keys": checkpoint["final"]["model_state_keys"],
                       "optimizer_state_keys": checkpoint["final"]["optimizer_state_keys"],
                       "verification_device": checkpoint["device"]},
        "execution": {name: _execution(artifact[name]) for name in
                      ("execution", "checkpoint_execution", "extract_execution")},
        "telemetry_final_20pct": {name: artifact["telemetry"]["summary"][name]["mean_last_20pct"]
                                  for name in telemetry_names},
        "source_artifact_sha256": {"readback": readback_sha, "first_three_records": records_sha},
    }


def _validate_readbacks(rows: dict[str, dict]) -> None:
    expected = {
        "NONE": (5, 5, None, 5), "R1": (5, 1, None, 1), "CORRECTED": (5, 1, 5, 5),
    }
    for arm, (before, fault, restored, after) in expected.items():
        row = rows[arm]
        got = (row["before"]["learning_epochs"], row["after_fault"]["learning_epochs"],
               None if row["restoration"] is None else row["restoration"]["runtime_value"],
               row["after"]["learning_epochs"])
        if row["arm"] != arm or got != (before, fault, restored, after):
            raise ValueError(f"{arm} epoch readback differs")
        if any(row[key]["mini_batches"] != 4 for key in ("before", "after_fault", "after")):
            raise ValueError(f"{arm} minibatch readback differs")


def _validate_ledgers(result: dict, artifacts: dict[str, dict]) -> tuple[list[dict], dict]:
    probe_ledger = CONTRACT.M.RegistryLedger(INPUTS["probe_ledger"], CONTRACT.registry())
    correction_ledger = CorrectionLedger(INPUTS["correction_ledger"], CONTRACT.correction_registry())
    probes, corrections = probe_ledger.events(), correction_ledger.events()
    probe_requests, correction_requests = probe_ledger.requests(), correction_ledger.requests()
    expected_probes = ["P_reward", "P_value", "P_physics", "P_episode", "P_noise", "P_optimizer_updates"]
    if len(probes) != 24 or [probes[i]["probe"] for i in range(0, 24, 4)] != expected_probes:
        raise ValueError("probe ledger inventory differs")
    if any([row["event"] for row in probes[i:i + 4]] != ["proposed", "approved", "consumed", "receipt"]
           for i in range(0, 24, 4)):
        raise ValueError("probe approval lifecycle differs")
    if any(len({row["request_id"] for row in probes[i:i + 4]}) != 1 for i in range(0, 24, 4)):
        raise ValueError("probe approval lifecycle crosses request ids")
    receipts = [probes[i + 3] for i in range(0, 24, 4)]
    prereg = {"checkpoint_sha256": artifacts["R1"]["checkpoint"]["sha256"], "seed": 2071,
              "protocol_sha256_lf": EXPECTED_SHA256["protocol"],
              "experiment_commit": result["experiment_commit"],
              "registry_digest": probe_ledger.registry_digest}
    prereg_digest = CONTRACT.L.digest(prereg)
    for probe, receipt, result_step in zip(expected_probes, receipts, result["probe_steps"], strict=True):
        request = probe_requests.get(receipt["request_id"])
        args_digest = CONTRACT.L.digest(CONTRACT.registry()["probes"][probe]["args"])
        if request is None or request["state"] != "done" or request["probe"] != probe \
                or request["prereg_digest"] != prereg_digest or request["args_digest"] != args_digest \
                or request["registry_digest"] != probe_ledger.registry_digest \
                or request["effect_key"] != probe_ledger.effect_key(prereg_digest, probe, args_digest) \
                or request["receipt"] != receipt or result_step != {**receipt, "probe": probe}:
            raise ValueError("probe request, registry binding, receipt, or result step differs")
    outcomes = [row["outcome"] for row in receipts]
    if outcomes != ["normal"] * 5 + ["abnormal"]:
        raise ValueError("probe outcomes differ")
    if len(corrections) != 4 or [row["event"] for row in corrections] != ["proposed", "approved", "consumed", "receipt"]:
        raise ValueError("correction approval lifecycle differs")
    if len({row["request_id"] for row in corrections}) != 1:
        raise ValueError("correction approval lifecycle crosses request ids")
    correction_request = correction_requests.get(corrections[-1]["request_id"])
    proposal = corrections[0]
    model_proposal = result["model"]["correction"]["review"]["decision"]["proposal"]
    registered = CONTRACT.correction_registry()["optimizer"]
    expected_correction_evidence = {
        "packet_digest": CONTRACT.L.digest(read(VALIDATION_INPUTS["correction_packet"])),
        "model_review_digest": CONTRACT.L.digest(result["model"]["correction"]),
        "protocol_sha256_lf": EXPECTED_SHA256["protocol"],
        "checkpoint_sha256": artifacts["R1"]["checkpoint"]["sha256"],
    }
    if correction_request is None or correction_request["state"] != "done" \
            or correction_request["receipt"] != corrections[-1] \
            or proposal["confirmed_mechanism"] != "optimizer" \
            or proposal["correction_id"] != registered["correction_id"] \
            or proposal["recovery_mode"] != "restart" or proposal["recovery_mode"] not in registered["recovery_modes"] \
            or proposal["protocol_case_arm"] != {"case": registered["protocol_case"], "arm": "corrected"} \
            or proposal["budget_gpu_s"] != 3600 \
            or model_proposal["correction_id"] != proposal["correction_id"] \
            or model_proposal["recovery_mode"] != proposal["recovery_mode"] \
            or proposal["loop_state_digest"] != CONTRACT.L.digest(result["loop"]) \
            or proposal["receipts_digest"] != CONTRACT.L.digest(result["probe_steps"]) \
            or proposal["registry_digest"] != correction_ledger.registry_digest \
            or proposal["effect_key"] != correction_ledger.effect_key(
                proposal["loop_state_digest"], proposal["correction_id"], proposal["recovery_mode"]) \
            or proposal["evidence"] != expected_correction_evidence \
            or corrections[-1]["tag"] != TAG or corrections[-1]["result_sha"] != EXPECTED_SHA256["recovery"] \
            or corrections[-1]["verdict"] != "recovered" or corrections[-1]["exit_code"] != 0 \
            or corrections[-1]["over_budget"] is not False \
            or not 0 < corrections[-1]["gpu_s"] <= proposal["budget_gpu_s"]:
        raise ValueError("correction receipt differs")
    public = []
    for index, (probe, receipt) in enumerate(zip(expected_probes, receipts, strict=True)):
        source = "model_selected" if index == 0 else "registered_new_probe" if index == 5 else "registered_next"
        public.append({"order": index + 1, "probe": probe, "selection_source": source,
                       "approval_events": ["proposed", "approved", "consumed", "receipt"],
                       "outcome": receipt["outcome"], "measurement": receipt["measurement"],
                       "gpu_s": receipt["gpu_s"], "exit_code": receipt["exit_code"],
                       "over_budget": receipt["over_budget"], "request_id": receipt["request_id"]})
    correction = {key: corrections[0][key] for key in
                  ("request_id", "correction_id", "recovery_mode", "confirmed_mechanism", "effect_key")}
    correction.update(approval_events=["proposed", "approved", "consumed", "receipt"],
                      verdict=corrections[-1]["verdict"], exit_code=corrections[-1]["exit_code"],
                      over_budget=corrections[-1]["over_budget"])
    return public, correction


def _model(kind: str, saved: dict, result: dict, packet_bytes: bytes, reservation: dict) -> dict:
    part = result["model"][kind]
    if saved["kind"] != kind or saved["final"] != {"status": "valid", "step": "first", "codes": []} \
            or saved["feedback"] is not None or saved["requests"] != 1 or saved["exit_code"] != 0:
        raise ValueError(f"{kind} feedback validation differs")
    recomputed = MODEL_CALL.review(saved, packet_bytes, kind, reservation)
    if recomputed != part["review"]:
        raise ValueError(f"{kind} stored review differs from semantic recomputation")
    if part["review"]["status"] != "valid" or part["review"]["requests"] != 1 or part["review"]["retried"]:
        raise ValueError(f"{kind} accepted review differs")
    if len(saved["calls"]) != 1 or part["attempts"][0]["calls"] != [{key: saved["calls"][0][key] for key in
            ("provider_model", "input_tokens", "output_tokens", "total_tokens", "wall_s")}]:
        raise ValueError(f"{kind} model accounting differs")
    call = part["attempts"][0]["calls"][0]
    if call["total_tokens"] != call["input_tokens"] + call["output_tokens"] or call["wall_s"] <= 0:
        raise ValueError(f"{kind} token/time accounting differs")
    return {"kind": kind, "accepted": True, "review_status": "valid", "first_status": "valid",
            "retried": False, "feedback": None, "decision": part["review"]["decision"],
            "call": call, "execution_wall_s": saved["execution_wall_s"],
            "source_record_sha256": sha256(INPUTS[f"model_{kind}"])}


def _walks(report: dict) -> bool:
    return (report["metrics"]["lin_vel_rmse_mps"] < G.WALK_FRACTION * report["standstill"]["lin_vel_rmse_mps"]
            and report["metrics"]["fall_rate"] <= G.REF_FALL_MAX)


def verify_public_payloads(payloads: dict[str, dict]) -> None:
    if set(payloads) != PUBLIC_FILES:
        raise ValueError("public artifact inventory differs")

    def visit(value: Any, label: str = "root") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in _FORBIDDEN_KEYS:
                    raise ValueError(f"forbidden public field: {label}.{key}")
                visit(child, f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{label}[{index}]")
        elif isinstance(value, str) and (_ABSOLUTE.search(value) or _PRIVATE.search(value)
                                         or _DANGEROUS_PATH.search(value)):
            raise ValueError(f"private or unsafe path in public value: {label}")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite public number: {label}")

    visit(payloads)


def verify_and_build() -> dict[str, dict]:
    if set(INPUTS) != set(EXPECTED_SHA256):
        raise ValueError("source artifact inventory differs")
    for logical_id, path in INPUTS.items():
        if not path.is_file() or sha256(path) != EXPECTED_SHA256[logical_id]:
            raise ValueError(f"source artifact differs: {logical_id}")
    if set(VALIDATION_INPUTS) != set(EXPECTED_VALIDATION_SHA256):
        raise ValueError("semantic validation artifact inventory differs")
    for logical_id, path in VALIDATION_INPUTS.items():
        if not path.is_file() or sha256(path) != EXPECTED_VALIDATION_SHA256[logical_id]:
            raise ValueError(f"semantic validation artifact differs: {logical_id}")

    protocol = read(INPUTS["protocol"])
    result = read(INPUTS["result"])
    if result["tag"] != TAG or result["seed"] != 2071 or result["restart_seed"] != 3071 \
            or result["outcome"] != "recovered" or result["demo_success"] is not True:
        raise ValueError("stored result differs")
    if set(result["conditions"]) != set(protocol["success_conditions"]) or not all(result["conditions"].values()):
        raise ValueError("thirteen completion conditions did not all pass")
    if result["protocol_sha256_lf"] != EXPECTED_SHA256["protocol"] or len(result["experiment_commit"]) != 40:
        raise ValueError("source/protocol binding differs")
    if result["design_coupling"] != protocol["design_coupling"]:
        raise ValueError("development interpretation binding differs")

    artifacts = {arm: read(INPUTS[f"artifact_{arm.lower()}"]) for arm in ("NONE", "R1", "CORRECTED")}
    readbacks = {arm: read(INPUTS[f"readback_{arm.lower()}"]) for arm in artifacts}
    _validate_readbacks(readbacks)
    reports = {arm: read(INPUTS[f"eval_{arm.lower()}"]) for arm in artifacts}
    checked = {}
    for arm, report in reports.items():
        checked[arm] = G.validate_fixed_report(report, {
            "name": report["name"], "checkpoint_file": "model_299.pt",
            "checkpoint_sha256": artifacts[arm]["checkpoint"]["sha256"], "action_scale": 0.25,
            "protocol_sha256_lf": EXPECTED_SHA256["fixed_eval_protocol"],
        })
    if len({json.dumps(row["versions"], sort_keys=True) for row in checked.values()}) != 1:
        raise ValueError("fixed evaluation runtimes differ")
    bands = read(INPUTS["bands"])["bands"]
    verdicts = {
        "NONE": REC.verdict(REC.compare(G._behavior(checked["NONE"]), G._behavior(checked["NONE"])), bands),
        "R1": REC.verdict(REC.compare(G._behavior(checked["R1"]), G._behavior(checked["NONE"])), bands),
        "CORRECTED": REC.verdict(REC.compare(G._behavior(checked["CORRECTED"]), G._behavior(checked["NONE"])), bands),
    }
    if not _walks(checked["NONE"]) or [verdicts[x]["label"] for x in ("NONE", "R1", "CORRECTED")] != ["healthy", "unhealthy", "healthy"]:
        raise ValueError("fixed evaluation recovery verdict differs")
    gate, recovery = read(INPUTS["source_gate"]), read(INPUTS["recovery"])
    if gate != result["source_gate"] or recovery != result["recovery"] or gate["fault_verdict"] != verdicts["R1"] \
            or recovery["verdict"] != verdicts["CORRECTED"]:
        raise ValueError("stored and recomputed verdicts differ")

    training = {}
    for arm in artifacts:
        records = read_jsonl(INPUTS[f"records_{arm.lower()}"])
        telemetry = read(VALIDATION_INPUTS[f"telemetry_{arm.lower()}"])
        artifact_telemetry = artifacts[arm]["telemetry"]
        if artifact_telemetry["sha256"] != EXPECTED_VALIDATION_SHA256[f"telemetry_{arm.lower()}"] \
                or artifact_telemetry["summary"] != telemetry["summary"] \
                or set(telemetry["summary"]) != set(telemetry["series"]) \
                or any(row["n"] != 300 or len(telemetry["series"][name]) != 300
                       for name, row in telemetry["summary"].items()):
            raise ValueError(f"{arm} telemetry file differs from artifact summary")
        training[arm] = _training(arm, artifacts[arm], records,
                                  EXPECTED_SHA256[f"readback_{arm.lower()}"], EXPECTED_SHA256[f"records_{arm.lower()}"])
        if artifacts[arm]["readback"] != readbacks[arm]:
            raise ValueError(f"{arm} readback file differs from artifact")
    probe_rows, correction = _validate_ledgers(result, artifacts)
    optimizer = probe_rows[-1]["measurement"]
    if optimizer["control"]["optimizer_steps"] != [20, 20, 20] \
            or optimizer["case"]["optimizer_steps"] != [4, 4, 4] or optimizer["checkpoint_loaded"] is not False:
        raise ValueError("new optimizer probe counts differ")
    raw_probe = read(INPUTS["probe_optimizer"])
    probe_fields = ("status", "optimizer_steps", "minibatches")
    raw_control = {key: raw_probe["NONE"]["measurement"][key] for key in probe_fields}
    raw_case = {key: raw_probe["R1"]["measurement"][key] for key in probe_fields}
    if raw_control != optimizer["control"] or raw_case != optimizer["case"]:
        raise ValueError("new optimizer probe raw evidence differs")

    saved_diagnosis = read(INPUTS["model_diagnosis"])
    saved_correction = read(INPUTS["model_correction"])
    diagnosis = _model("diagnosis", saved_diagnosis, result,
                       VALIDATION_INPUTS["diagnosis_packet"].read_bytes(),
                       read(VALIDATION_INPUTS["diagnosis_reservation"]))
    correction_model = _model("correction", saved_correction, result,
                              VALIDATION_INPUTS["correction_packet"].read_bytes(),
                              read(VALIDATION_INPUTS["correction_reservation"]))
    decision = diagnosis["decision"]
    if decision["ranking"][0] != "reward" or decision["ranking"].index("optimizer") != 3 \
            or decision["probe"] != "P_reward" or probe_rows[0]["probe"] != decision["probe"]:
        raise ValueError("raw model diagnosis sequence differs")
    if correction_model["decision"]["proposal"]["correction_id"] != "restore_configured_ppo_epochs":
        raise ValueError("raw model correction differs")
    raw_requests = saved_diagnosis["requests"] + saved_correction["requests"]
    if raw_requests != 2 or result["api_requests"] != raw_requests \
            or result["api_requests_in_saved_records"] != raw_requests \
            or result["pending_model_records"] != [] or result["transport_errors"] != []:
        raise ValueError("raw model request accounting or saved-record status differs")
    expected_costs = {"train_NONE", "train_R1", "source_evaluation", "reference",
                      *(f"probe_{index}_{probe}" for index, probe in enumerate(
                          ("P_reward", "P_value", "P_physics", "P_episode", "P_noise", "P_optimizer_updates"), 1)),
                      "correction_training", "correction_evaluation"}
    derived_conditions = {
        "source_failure_present": gate["outcome"] == "failure_present",
        "accepted_fresh_diagnosis": diagnosis["accepted"],
        "first_probe_from_model": probe_rows[0]["probe"] == diagnosis["decision"]["probe"],
        "approved_probe_receipts": all(row["approval_events"][-1] == "receipt" for row in probe_rows),
        "optimizer_update_count_confirmed": result["loop"]["status"] == "confirmed"
                                              and result["loop"]["hypotheses"] == ["optimizer_update_count"],
        "accepted_fresh_correction": correction_model["accepted"],
        "restored_update_count": training["CORRECTED"]["first_three_updates"]["optimizer_steps"] == [20, 20, 20],
        "approved_correction_consumed_once": correction["approval_events"] ==
                                               ["proposed", "approved", "consumed", "receipt"],
        "corrected_fixed_eval_healthy": verdicts["CORRECTED"]["label"] == "healthy",
        "costs_recorded": set(result["costs"]) == expected_costs
                          and all(isinstance(value, (int, float)) and value > 0 for value in result["costs"].values()),
    }
    if any(result["conditions"][key] is not value for key, value in derived_conditions.items()) \
            or not all(derived_conditions.values()):
        raise ValueError("derivable completion conditions differ from source evidence")

    evaluations = {arm: _public_fixed(arm, checked[arm], verdicts[arm], EXPECTED_SHA256[f"eval_{arm.lower()}"])
                   for arm in artifacts}
    calls = [diagnosis["call"], correction_model["call"]]
    summary = {
        "schema": "public_r1_repair_summary_v2", "status": "complete", "tag": TAG,
        "outcome": "recovered", "demo_success": True, "source_training_seed": 2071,
        "corrected_restart_seed": 3071, "completion_conditions": result["conditions"],
        "source_failure": {"none_walks": True, "r1_label": "unhealthy",
                           "failed_checks": verdicts["R1"]["failed"]},
        "recovery": {"corrected_label": "healthy", "failed_checks": [],
                     "checkpoint_sha256": recovery["checkpoint_sha256"]},
        "fixed_evaluation_metrics": {arm: checked[arm]["metrics"] for arm in artifacts},
        "diagnostic_sequence": [
            {"stage": "model", "raw_ranking_first": "reward", "raw_optimizer_rank": 4,
             "selected_probe": "P_reward"},
            {"stage": "registered_probe_loop", "normal_probes": [row["probe"] for row in probe_rows[:5]],
             "abnormal_probe": probe_rows[5]["probe"]},
            {"stage": "correction", "id": correction["correction_id"], "mode": correction["recovery_mode"],
             "epoch_transition": [5, 1, 5], "first_three_optimizer_steps": [20, 20, 20]},
        ],
        "model_accounting": {"api_requests": result["api_requests"],
                             "input_tokens": sum(x["input_tokens"] for x in calls),
                             "output_tokens": sum(x["output_tokens"] for x in calls),
                             "total_tokens": sum(x["total_tokens"] for x in calls),
                             "model_wall_s": sum(x["wall_s"] for x in calls)},
        "interpretation": protocol["interpretation"], "design_coupling": result["design_coupling"],
        "inspection_provenance": {
            "mode": "fresh_three_update_inspection", "checkpoint_loaded": False,
            "meaning": "학습 checkpoint를 불러 관측한 값이 아니라 같은 arm 설정으로 새로 실행한 3 update 계측이다.",
        },
        "source_commit_sha": result["experiment_commit"], "protocol_sha256_lf": result["protocol_sha256_lf"],
    }
    payloads = {
        "summary.json": summary,
        "evaluations/none.json": evaluations["NONE"],
        "evaluations/r1.json": evaluations["R1"],
        "evaluations/corrected.json": evaluations["CORRECTED"],
        "training_evidence.json": {"schema": "public_r1_repair_training_evidence_v1", "arms": training},
        "probe_and_correction.json": {
            "schema": "public_r1_repair_probe_and_correction_v1", "probe_count": len(probe_rows),
            "normal_probe_count": 5, "abnormal_probe_count": 1, "probes": probe_rows,
            "correction": correction,
            "source_artifact_sha256": {"probe_raw": EXPECTED_SHA256["probe_optimizer"],
                                       "probe_ledger": EXPECTED_SHA256["probe_ledger"],
                                       "correction_ledger": EXPECTED_SHA256["correction_ledger"]},
        },
        "model_evidence.json": {"schema": "public_r1_repair_model_evidence_v1",
                                "diagnosis": diagnosis, "correction": correction_model},
        "provenance.json": {
            "schema": "public_r1_repair_provenance_v1",
            "export_contract": "allowlisted_fields_only_no_private_or_user_paths",
            "source_commit_sha": result["experiment_commit"],
            "protocol_sha256_lf": EXPECTED_SHA256["protocol"],
            "fixed_eval_protocol_sha256_lf": EXPECTED_SHA256["fixed_eval_protocol"],
            "source_artifact_sha256": EXPECTED_SHA256,
        },
    }
    verify_public_payloads(payloads)
    return payloads


def write_new(payloads: dict[str, dict], output: Path = OUTPUT, audit: Path = AUDIT) -> None:
    verify_public_payloads(payloads)
    if output.exists() or audit.exists():
        raise FileExistsError("public output or private export audit already exists")
    output.mkdir(parents=True, exist_ok=False)
    for relative, payload in sorted(payloads.items()):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    public_hashes = {relative: sha256(output / relative) for relative in sorted(payloads)}
    audit.mkdir(parents=True, exist_ok=False)
    (audit / "receipt.json").write_text(json.dumps({
        "schema": "r1_repair_public_export_receipt_v1", "tag": TAG,
        "source_artifact_sha256": EXPECTED_SHA256, "public_artifact_sha256": public_hashes,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def verify_export(output: Path = OUTPUT) -> dict[str, str]:
    expected = verify_and_build()
    actual_files = {path.relative_to(output).as_posix() for path in output.rglob("*.json")} if output.is_dir() else set()
    if actual_files != PUBLIC_FILES:
        raise ValueError("exported public artifact inventory differs")
    for relative, payload in expected.items():
        if read(output / relative) != payload:
            raise ValueError(f"exported public artifact differs: {relative}")
    return {relative: sha256(output / relative) for relative in sorted(expected)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "export", "verify"))
    args = parser.parse_args()
    if args.command == "check":
        payloads = verify_and_build()
        print(json.dumps({"status": "pass", "public_files": sorted(payloads)}, ensure_ascii=False))
    elif args.command == "export":
        payloads = verify_and_build()
        write_new(payloads)
        print(json.dumps({"status": "exported", "public_files": sorted(payloads)}, ensure_ascii=False))
    else:
        print(json.dumps({"status": "pass", "public_sha256": verify_export()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
