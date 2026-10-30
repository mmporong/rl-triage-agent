"""실측 모델 선택·probe pair·원본 설정을 승인 원장과 실행 권한으로 결합한다."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from local_f1_restart_base_v2 import ROOT, binding, read, sha, write_new

sys.path.insert(0, str(ROOT / "src"))
from rl_triage import probe_loop as L
from rl_triage.correction_ledger import CorrectionLedger

SOURCE = "763e5181f53c29796d7499d96bee59d0b7ee8db3"
PROTOCOL = "0f644c87dd0f93b0cb9fdce7571004506d48808cc0e2f25cdcb83af4d6ce53d9"
CAPSULE = "0dbd270663db36e8699386720e5eb6390b276502f80691a06f1fe13da324932d"
RESPONSE = "fab2bf6297b2a821c41b7bb6a235ec3e8d96913bba3ce13019070029cdfbe3ab"
PAIR = "7a2a252b8951ed21b9f76047c3ee60830a900d6b05d290b275af295bf0cecea8"
LAB = "90b79bb2d44feb8d833f260f2bf37da3487180ba"


def lab_controls(lab: Path) -> dict[str, Any]:
    paths = {"num_steps_per_env": Path("source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py"),
             "action_scale": Path("source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py")}
    head = subprocess.run(["git", "-C", str(lab), "rev-parse", "HEAD"], capture_output=True, text=True,
                          encoding="utf-8", errors="strict", check=True).stdout.strip()
    if head != LAB:
        raise ValueError("Isaac Lab HEAD 불일치")
    values: dict[str, Any] = {"lab_head": head, "sources": {}}
    patterns = {"num_steps_per_env": r"num_steps_per_env\s*=\s*(\d+)",
                "action_scale": r"joint_pos\.scale\s*=\s*([0-9.]+)"}
    for key, relative in paths.items():
        target = lab / relative
        if subprocess.run(["git", "-C", str(lab), "diff", "--quiet", "HEAD", "--", relative.as_posix()],
                          check=False).returncode:
            raise ValueError(f"Isaac Lab {key} source가 dirty다")
        match = re.search(patterns[key], target.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"{key} source 값을 찾지 못했다")
        values[key] = int(match.group(1)) if key == "num_steps_per_env" else float(match.group(1))
        values["sources"][key] = {"path": relative.as_posix(), "sha256": sha(target)}
    if values["num_steps_per_env"] != 24 or values["action_scale"] != 0.25:
        raise ValueError("Go2 통제값 불일치")
    return values


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=False)
    result, response, capsule = read(args.model_result), read(args.response), read(args.capsule)
    engine = result.get("call", {}).get("journal", {}).get("engine_result", {})
    selection = engine.get("selection")
    response_selection = json.loads(response.get("safe_result", {}).get("content", "null"))
    if (result.get("status") != "choice_ready" or result.get("source_anchor") != SOURCE
            or result.get("protocol_sha256") != PROTOCOL or result.get("call", {}).get("capsule_sha256") != CAPSULE
            or sha(args.capsule) != CAPSULE or sha(args.response) != RESPONSE or not isinstance(capsule, dict)
            or engine.get("accepted") is not True or engine.get("model_status") != "valid"
            or selection != response_selection or selection.get("action") != "propose_correction"
            or selection.get("target") != "restore_body_frame_reward"
            or "observation:0:P_reward:outcome" not in selection.get("fact_ids", [])
            or result.get("approval_granted") is not False):
        raise ValueError("model proposal 결합 불일치")
    pair = read(args.pair)
    rows = pair.get("completed", [])
    if (sha(args.pair) != PAIR or pair.get("status") != "complete" or len(rows) != 2
            or {row.get("probe") for row in rows} != {"P_reward"}
            or {row.get("classification") for row in rows} != {"normal", "abnormal"}
            or {row.get("checkpoint_sha256") for row in rows} != {
                "b346a0c3e3ad6410ce93dcf73e6165ed69e2d6ab95b2c5398943b3a8874cf54d",
                "1d73d523c97c5c89c988febfbd3d5e533e90681cd0e6132d6ffe13888fa95731",
            }):
        raise ValueError("probe pair 불일치")
    manifest, readback, adapter = read(args.source_manifest), read(args.source_readback), read(args.source_adapter)
    expected = {"seed": 2221, "task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "num_envs": 4096,
                "max_iterations": 300, "configured_learning_epochs": 5, "configured_mini_batches": 4}
    if (any(manifest.get(k) != v for k, v in expected.items()) or manifest.get("original_faults") != []
            or manifest.get("applied_corrections") != [] or readback.get("remaining_faults") != []
            or adapter.get("source_checkpoint_sha256") is not None or LAB not in json.dumps(adapter)):
        raise ValueError("s2221 source controls 불일치")
    lab = lab_controls(args.lab)
    proposal = {"schema": "local_f1_model_proposal_v2", "status": "valid", "model_approved": False,
                "selection": selection, "source_anchor": SOURCE, "protocol_sha256": PROTOCOL,
                "capsule": binding(args.capsule), "response": binding(args.response),
                "result": binding(args.model_result), "pair": binding(args.pair)}
    proposal_path = args.out / "model-proposal.json"
    write_new(proposal_path, proposal)
    loop = L.digest({"pair": PAIR, "proposal": sha(proposal_path), "mechanism": "reward", "seed": 3221})
    ledger = CorrectionLedger(args.out / "correction-ledger.jsonl")
    event = ledger.propose(loop, "reward", L.digest({"pair": PAIR}), "restore_body_frame_reward", "restart",
                           {"case": "case_01", "arm": "corrected"}, 7200,
                           {"proposal_sha256": sha(proposal_path), "approval_is_model_output": False})
    ledger.approve(event["request_id"], "main:user-delegated")
    ledger.consume(event["request_id"], loop)
    authorization = {"schema": "local_f1_restart_authorization_v2", "status": "authorized",
                     "execution_authorized": True, "fresh_seed": 3221, "num_envs": 4096, "iterations": 300,
                     "num_steps_per_env": 24, "action_scale": 0.25,
                     "correction_id": "restore_body_frame_reward", "recovery_mode": "restart",
                     "approval": {"subject": "main:user-delegated", "automatic_model_approval": False,
                                  "request_id": event["request_id"], "ledger_state": "consumed",
                                  "ledger": binding(ledger.path)},
                     "model_proposal": binding(proposal_path), "pair": binding(args.pair),
                     "source_manifest": binding(args.source_manifest), "source_readback": binding(args.source_readback),
                     "source_adapter": binding(args.source_adapter), "source_lab": lab,
                     "none_checkpoint": binding(args.none_checkpoint), "fixed_eval": binding(args.fixed_eval)}
    write_new(args.out / "authorization.json", authorization)
    return authorization


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model_result", "capsule", "response", "pair", "source_manifest", "source_readback",
                 "source_adapter", "none_checkpoint", "fixed_eval"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--lab", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    prepare(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
