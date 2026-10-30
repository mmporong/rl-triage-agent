"""Present already accepted NIM evidence. No model calls or new measurements."""
import hashlib,json
from pathlib import Path

INPUTS = ["latency_analysis.json","latency_pairs.csv","model_quality.csv","vl_observations_18.json",
          "vl_aux_observations_3.json","vl_selection_and_render.json","summary.json",
          "architecture_comparison.json","api_reference_evidence.json","sources.json","recovery_summary.json"]
SOURCE_COMMIT = "b29a791db9f6397c4b04698b349ef68606273ae5"
ANCHOR_SHA256 = "2ca099294618dcb50c10d444816433e0e46131fd419211b467ebaa4e5c07b7ee"

def view(folder):
    folder=Path(folder)
    raw=(folder/"NIM_SOURCE_ANCHOR.json").read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ANCHOR_SHA256: raise ValueError("NIM source anchor differs")
    anchor=json.loads(raw)
    for name in INPUTS:
        logical="nim-evidence/"+name
        if hashlib.sha256((folder/logical).read_bytes()).hexdigest()!=anchor["files"][logical]:
            raise ValueError("Accepted NIM input differs: "+name)
    read=lambda name:json.loads((folder/"nim-evidence"/name).read_text(encoding="utf-8"))
    latency=read("latency_analysis.json");summary=read("summary.json");recovery=read("recovery_summary.json")
    headline=summary["headline"]
    return {"schema":"nemotron_evidence_view_v1","provider":"NVIDIA hosted NIM",
        "phase":"historical_development_measurements","latency_measurement_date":"2026-10-16",
        "source_commit":SOURCE_COMMIT,"latency_unit":"API request wall-clock seconds",
        "latency":latency["stats"],"quality":latency["quality"],
        "vision":{"valid":headline["VL_valid_observations"],"classification":headline["VL_fixed_eval_class_correspondence"],
                  "auxiliary":headline["VL_s2080_aux_observations"]},
        "loop":headline["fresh_multimodal_diagnosis"],
        "evaluation":{"ratios":recovery["fixed_evaluation"]["ratios"],"criteria":recovery["fixed_evaluation"]["registered_criteria"],
                      "outcome":recovery["outcome"],"iterations":recovery["training"]["iterations"],
                      "conditions":recovery["fixed_evaluation"]["CORRECTED"]["condition_count"]},
        "boundaries":{"token_factory_latency_comparison":False,"whole_workflow_speedup":False,
                      "human_labeled_video_accuracy":False,"image_selection_benefit":False,
                      "new_3_5_recovery":False,"overall_quality_acceptance_met":False,
                      "grants_execution_authority":False},
        "input_sha256":{"nim-evidence/"+name:hashlib.sha256((folder/"nim-evidence"/name).read_bytes()).hexdigest() for name in INPUTS}}
