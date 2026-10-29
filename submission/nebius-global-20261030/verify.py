"""CPU-only byte integrity and recorded-result consistency checks; no live approval."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "source-map.json").read_text(encoding="utf-8"))
    objects = {}
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source path escapes package")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Source byte mismatch: " + entry["path"])
        if entry["path"].endswith(".json"):
            objects[entry["id"]] = json.loads(raw)
    s, p = objects["r1_summary"], objects["r1_probes"]
    m, t = objects["r1_model"], objects["r1_training"]
    ev = objects["r1_eval"]
    checks = {
        "r1_development_seeds": s["source_training_seed"] == 2071 and s["corrected_restart_seed"] == 3071,
        "model_first_then_registered_sequence": m["diagnosis"]["decision"]["probe"] == p["probes"][0]["probe"] == "P_reward" and p["probes"][0]["selection_source"] == "model_selected" and all(x["selection_source"] != "model_selected" for x in p["probes"][1:]),
        "five_normal_one_abnormal": [x["outcome"] for x in p["probes"]] == ["normal"] * 5 + ["abnormal"],
        "observed_optimizer_counts": p["probes"][5]["measurement"]["control"]["optimizer_steps"] == [20]*3 and p["probes"][5]["measurement"]["case"]["optimizer_steps"] == [4]*3,
        "approval_recorded_not_reissued": p["correction"]["approval_events"] == ["proposed", "approved", "consumed", "receipt"],
        "checkpoint_and_metrics_join": ev["checkpoint"]["sha256"] == s["recovery"]["checkpoint_sha256"] and ev["metrics"] == s["fixed_evaluation_metrics"]["CORRECTED"],
        "four_behavior_checks": set(ev["verdict"]["checks"]) == {"survival_ratio", "fall_frac", "err_xy_ratio", "err_yaw_ratio"} and all(ev["verdict"]["checks"].values()) and s["outcome"] == "recovered",
        "compound_two_corrections": all(c["repair_verdicts"] == ["not_recovered", "recovered"] for c in objects["residual"]["cases"]),
        "u1_scope_four_terms": objects["u1"]["scope"]["restored_terms"] == 4 and objects["u1"]["measurement_kind"] == "post_result_development_confirmation",
        "fresh_render_not_recovery": objects["fresh_render"]["new_training"] == 0 and objects["fresh_render"]["new_recovery_claim"] is False,
    }
    if not all(checks.values()):
        raise ValueError("Recorded consistency failed: " + repr(checks))
    return {"status": "PASS", "files_verified": len(manifest["files"]), "checks": checks,
            "scope": "Public recorded evidence bytes and joins, not live provenance/approval reauthentication"}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).parent), indent=2))
