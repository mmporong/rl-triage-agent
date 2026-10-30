"""Offline core-module replay of recorded Token Factory requests and responses."""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

CORE_SOURCE_FILES = frozenset(['evals/agentic_investigation_call_v1.py', 'evals/agentic_investigation_call_v2.py', 'evals/agentic_investigation_contract_v1.py', 'evals/agentic_investigation_contract_v2.py', 'evals/agentic_investigation_evidence_v1.py', 'evals/agentic_investigation_loop_v1.py', 'evals/agentic_investigation_runtime_v1.py', 'evals/agentic_investigation_runtime_v2.py', 'evals/grounded_vision_observer_v1.py', 'evals/model_decision_journal_v1.py', 'evals/nemotron_endpoint_ledger_v3.py', 'evals/selection_comparison_controller_v1.py', 'evals/selection_comparison_view_v1.py'])
MODEL = "nvidia/Nemotron-3.5-Super-VL"
GOAL = ("Compare the candidate training record with its reference. Use supplied facts and registered tools only. "
        "Request remaining training evidence if needed. If all supplied candidate/reference measurements are "
        "indistinguishable and there is no supported anomaly, hold rather than invent a failure. "
        "Otherwise propose a registered inspection, never a confirmed cause or an execution approval.")


def require(value, reason):
    if not value:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def store(path, value):
    data = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, "existing replay output differs")
    else:
        with path.open("xb") as stream:stream.write(data)


def load_bundle(folder):
    spec = importlib.util.spec_from_file_location("recorded_bundle_verifier", folder / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    return module, module.load("PUBLIC_INDEX.json"), module.load("evidence/training_source.json")


def core_modules(repo, descriptor):
    require(repo.is_dir(), "project checkout unavailable")
    require(descriptor.get("schema") == "tokenfactory_public_core_replay_sources_v1"
            and descriptor.get("hash_algorithm") == "sha256_of_LF_bytes"
            and set(descriptor.get("files", {})) == CORE_SOURCE_FILES, "registered core source descriptor")
    for name, expected in descriptor["files"].items():
        path = (repo / name).resolve()
        require(path.is_relative_to(repo) and path.is_file(), "missing registered core source")
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        require(actual == expected, "core source LF SHA mismatch: " + name)
    sys.path[:0] = [str(repo / name) for name in ("evals", "bench", "src")]
    modules = {key:importlib.import_module(name) for key,name in
               (("A1","agentic_investigation_contract_v1"), ("A2","agentic_investigation_contract_v2"),
                ("CALL","agentic_investigation_call_v2"), ("LOOP","agentic_investigation_loop_v1"),
                ("RUNTIME","agentic_investigation_runtime_v2"))}
    for module in modules.values():
        require(Path(module.__file__).resolve() == repo / "evals" / (module.__name__ + ".py"),
                "core import came from another checkout")
    return modules


class PublishedTrainingBridge:
    def __init__(self, modules, training, training_sha, record, folder):
        self.modules, self.record = modules, record
        self.initial, self.bindings = {}, {}
        self.policy = record["data_policy"]
        require(self.policy in {"eager", "on_demand"} and set(record["training_arms"]) == {"case", "reference"},
                "registered case/policy fields")
        for role in ("case", "reference"):
            arm = record["training_arms"][role]
            row = training["arms"][arm]
            require(row["checkpoint"]["sha256"] == record["checkpoint_sha256"][role], "source checkpoint differs")
            projection = {"schema":"published_training_replay_projection_v1", "role":role, "source_arm":arm,
                          "training_source_sha256":training_sha, "checkpoint_sha256":row["checkpoint"]["sha256"],
                          "telemetry":row["telemetry_final_20pct"]}
            path = folder / (role + ".projection.json")
            store(path, projection)
            self.bindings[role] = {"checkpoint_sha256":row["checkpoint"]["sha256"],
                                   "telemetry_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            self.initial.update({f"telemetry:{role}:{key}":value for key,value in projection["telemetry"].items()})
        if self.policy == "on_demand":
            self.initial = {key:value for key,value in self.initial.items() if key.endswith(":Train/mean_reward")}
        self.evidence = {"evidence:training-01":"training"} if self.policy == "on_demand" else {}

    def start(self, folder, run_id):
        A, LOOP = self.modules["A1"], self.modules["LOOP"]
        C = A.C
        packet = C.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
                    facts=self.initial, input_bindings=self.bindings, execution_budget_s=1200)
        snapshot = C.V.build_snapshot(packet, run_id, "model_adaptive", C.V.file_digest(C.V.PROTOCOL), folder / "history.jsonl")
        return LOOP.Investigation(snapshot, goal=GOAL, evidence=self.evidence, max_decisions=2, allow_fixture=False)

    def messages(self, loop, *, visual_context=None):
        require(visual_context is None and loop.goal == GOAL and loop.evidence == self.evidence, "unregistered replay context")
        require(loop.snapshot["inputs"]["packet"]["facts"] == self.initial
                and loop.snapshot["inputs"]["packet"]["input_bindings"] == self.bindings, "replay source binding changed")
        context = loop.context()
        return {"context":context, "messages":self.modules["A1"].messages(context), "image_count":0}


def replay(bundle, repo, output):
    verifier, index, training = load_bundle(bundle)
    require(index["core_model"] == MODEL, "exact TF model identity required")
    descriptor = verifier.load("CORE_SOURCES_LF.json")
    modules = core_modules(repo, descriptor)
    training_sha = hashlib.sha256((bundle / "evidence/training_source.json").read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for record in index["core_records"]:
        episode = record["episode_id"]
        require(isinstance(episode, str) and re.fullmatch(r"ep[0-9]{2}",episode) and episode not in seen,
                "invalid or duplicate replay episode ID")
        seen.add(episode)
        case = output / episode
        bridge = PublishedTrainingBridge(modules, training, training_sha, record, case)
        loop = bridge.start(case, "recorded_replay_" + episode)
        current = modules["RUNTIME"].Runtime._request(loop, bridge, None)
        context, outgoing = current["context"], current["messages"]
        messages = (modules["CALL"]._messages(context, outgoing) if outgoing[0]["content"] == modules["A2"].SYSTEM
                    else modules["CALL"].from_v1(context, outgoing))
        response_format = copy.deepcopy(modules["A2"].response_format(context))
        response_format["json_schema"]["strict"] = True
        params = {"model":MODEL,"max_tokens":8192,"store":False,"stream":False,"response_format":response_format}
        actual_request = verifier.load(record["request"])
        require(canonical({"messages":messages,**params}) == canonical(actual_request),
                "reconstructed model-facing request differs from recorded actual request")
        response = verifier.load(record["response"])
        require(response["model"] == MODEL and response["choices"][0]["finish_reason"] == "stop", "recorded provider identity/completion")
        review = loop.propose(response["choices"][0]["message"]["content"])
        require(review["status"] == "valid" and canonical(review["decision"]) == canonical(record["decision"]), "original core rejected recorded decision")
        decision = review["decision"]
        spec = None
        if decision["action"] == "run_probe":
            spec = modules["A1"].C.request_spec(loop.snapshot, loop.probe_choice())
            require(spec["probe"] == decision["target"] and spec["approval_granted"] is False, "core request target/authority")
        else:
            require(decision["action"] == "hold" and loop.status == "hold", "unregistered recorded action")
        row = {"episode_id":episode,"model":MODEL,"decision":decision,"same_public_request":True,
               "actual_request_sha256":hashlib.sha256((bundle / record["request"]).read_bytes()).hexdigest(),
               "reconstructed_context_sha256":hashlib.sha256(canonical(context)).hexdigest(),
               "reconstructed_request_spec":spec,"original_core_decisions":len(loop.events),
               "source_scope":"published training summary; reconstructed local core context",
               "original_private_capsule_or_approval_reauthenticated":False,"approval_granted":False,
               "execution_allowed":False,"new_HTTP_or_GPU_calls":0}
        store(case / "result.json", row)
        rows.append(row)
    require(len(rows) == 8 and sum(x["decision"]["action"]=="hold" for x in rows)==4, "recorded replay totals differ")
    report = {"schema":"tokenfactory_public_core_replay_v1","status":"PASS","episodes":rows,
              "records_replayed":len(rows),"normal_control_holds":4,"pending_inspections":4,
              "original_core_functions":["Runtime._request","CALL2 message validation","Investigation.propose/probe_choice","C.request_spec"],
              "private_original_capsule_required":False,"posthoc_demonstration_only":True,
              "original_private_execution_source_reauthenticated":False,"new_HTTP_or_GPU_calls":0,
              "approval_granted":False,"execution_allowed":False}
    store(output / "CORE_REPLAY.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(Path(__file__).resolve().parent, args.repo_root.resolve(), args.output.resolve())
    print(json.dumps({"status":result["status"],"records_replayed":8,"normal_control_holds":4,
                      "pending_inspections":4,"new_HTTP_or_GPU_calls":0,"execution_allowed":False}))
