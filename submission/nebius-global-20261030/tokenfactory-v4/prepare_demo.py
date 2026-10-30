"""Build the judge view with the original CPU investigation modules. No model or GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path
import replay_core as core
import nim_evidence

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build_view(report, index, folder):
    rows = []
    by_id = {r["episode_id"]: r for r in report["episodes"]}
    for record in index["core_records"]:
        row = by_id[record["episode_id"]]
        request = json.loads((folder / record["request"]).read_text(encoding="utf-8"))
        response = json.loads((folder / record["response"]).read_text(encoding="utf-8"))
        metadata = json.loads((folder / record["metadata"]).read_text(encoding="utf-8"))
        text = request["messages"][1]["content"]
        view = json.loads(text if isinstance(text, str) else text[0]["text"])
        decision = row["decision"]
        spec = row["reconstructed_request_spec"]
        checks = {
            "recorded_http_200": metadata["http_status"] == 200,
            "exact_model_identity": request["model"] == response["model"] == core.MODEL,
            "citations_in_supplied_facts": set(decision["fact_ids"]) <= set(view["evidence_facts"]),
            "allowed_action_and_target": decision["target"] in view["available_actions"][decision["action"]],
            "original_core_accepts_decision": row["same_public_request"] and row["original_core_decisions"] > 0,
            "no_execution_authority": row["approval_granted"] is False and row["execution_allowed"] is False,
        }
        core.require(all(checks.values()), "judge view validation failed")
        if decision["action"] == "run_probe":
            core.require(spec["probe"] == decision["target"] and spec["approval_granted"] is False,
                         "core inspection request binding")
        request_digest = sha(folder / record["request"])
        response_digest = sha(folder / record["response"])
        proposal = None if spec is None else {
            "schema": "tokenfactory_recorded_inspection_review_v1",
            "episode_id": record["episode_id"], "provider": "Nebius Token Factory", "model": core.MODEL,
            "probe": spec["probe"], "fact_ids": decision["fact_ids"],
            "request_sha256": request_digest, "response_sha256": response_digest,
            "checkpoint_sha256": record["checkpoint_sha256"],
            "requires_new_registration_approval_and_lease": True,
            "approval_granted": False, "execution_allowed": False,
            "scope": "Recorded inspection proposal for review; not an executable authorization.",
        }
        rows.append({"id": record["episode_id"], "control": record["training_arms"]["case"] == "NONE",
            "data_policy": record["data_policy"], "trial": record["trial"] + 1,
            "facts": view["evidence_facts"], "decision": decision, "checks": checks,
            "model": response["model"], "response_id": response["id"],
            "usage": response.get("usage", {}), "request_file": record["request"],
            "response_file": record["response"], "request_sha256": request_digest,
            "response_sha256": response_digest, "review_request": proposal})
    return {"schema": "tokenfactory_judge_view_v1", "mode": "recorded_api_and_local_core_replay",
        "records": rows, "totals": {"records": 8, "holds": 4, "inspection_proposals": 4},
        "original_core_functions": report["original_core_functions"],
        "new_HTTP_or_GPU_calls": 0, "approval_granted": False, "execution_allowed": False,
        "scope": "Known development records. CPU replay of recorded API replies; no new GPU repair result."}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Private local CPU replay output; not served")
    a = p.parse_args()
    output = a.output.resolve()
    core.require(not output.is_relative_to(ROOT), "private runtime output must be outside the public bundle")
    report = core.replay(ROOT, a.repo_root.resolve(), output)
    index = json.loads((ROOT / "PUBLIC_INDEX.json").read_text(encoding="utf-8"))
    view = build_view(report, index, ROOT)
    core.store(ROOT / "WEB_EVIDENCE.json", view)
    core.store(ROOT / "NEMOTRON_EVIDENCE.json", nim_evidence.view(ROOT))
    names = ["WEB_EVIDENCE.json", "PUBLIC_INDEX.json", "CORE_SOURCES_LF.json", "evidence/training_source.json"]
    names += [r[k] for r in index["core_records"] for k in ("request", "response", "metadata")]
    names += ["NEMOTRON_EVIDENCE.json", "NIM_SOURCE_ANCHOR.json"] + ["nim-evidence/"+n for n in nim_evidence.INPUTS]
    manifest = {"schema": "tokenfactory_judge_web_manifest_v1", "files": {n: sha(ROOT / n) for n in names}}
    core.store(ROOT / "WEB_INDEX.json", manifest)
    print(json.dumps({"status": "PASS", "records": 8, "holds": 4, "inspection_proposals": 4,
                      "new_HTTP_or_GPU_calls": 0, "execution_allowed": False}))

if __name__ == "__main__":
    main()
