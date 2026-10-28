"""One consumed C reward repair: fresh restart300 and one fixed evaluation job."""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("evals", "bench", "src")]

import agentic_brev_repair_evidence_v1 as E
import agentic_play_job_v2 as JOB
import run_agentic_brev_probe_parent_v1 as P
import vision_brev_backend_v1 as OWNED
import vision_brev_training_backend_v2 as WORKER

SCHEMA, C = E.SCHEMA, E.C
SOURCES = E.SOURCES | P.SOURCES | set(JOB.REQUIRED_SOURCE_FILES)


def now():
    return datetime.now(timezone.utc)  # noqa: UP017 - registered Python 3.10


def admission_body(reference, operation, registered, current, *, entry):
    resolver = registered["resolver"]
    body = resolver.read(reference)
    E.require(set(body) == {"schema", "operation_id", "registration", "resource", "boot", "gpu_model",
                            "provider_observed_at_utc", "query_at_utc", "admission_at_utc",
                            "lease_acquired_at_utc", "lease_expires_at_utc"}
              and body["schema"] == SCHEMA + "_admission" and body["operation_id"] == operation,
              "admission schema/operation")
    provider, query, admitted, acquired, expires = (E.E._stamp(body[k]) for k in (
        "provider_observed_at_utc", "query_at_utc", "admission_at_utc", "lease_acquired_at_utc", "lease_expires_at_utc"))
    E.require(provider <= query <= admitted <= current and acquired <= admitted < expires
              and (admitted - provider).total_seconds() <= 30 and current < expires,
              "provider/query/admission/lease times stale/future")
    if entry:
        E.require((current - admitted).total_seconds() <= 30
                  and (expires - current).total_seconds() >= registered["contract"]["budget"]["wall_s"],
                  "admission freshness/remaining lease window")
    return body


class ActualServices:
    synthetic_cpu = False

    def clock(self):
        return now()

    def check(self, registered, admission, *, entry=False):
        E.require(os.name == "posix", "actual repair requires POSIX")
        resolver, layout = registered["resolver"], registered["layout"]
        E.require(E.J.file_sha(Path("/proc/sys/kernel/random/boot_id")) == admission["boot"]["sha256"], "current boot changed")
        resolver.file(admission["boot"])
        basis_path = resolver.file(layout["lease_basis"])
        with JOB.canonical_lease_scope():
            E.require(JOB.scope_active_root() == JOB.canonical_lease_root(), "canonical lease root")
            basis = JOB.validate_protocol(basis_path, repo=registered["repo"], now=self.clock())
            proof_path = resolver.file(admission["resource"])
            raw_proof = resolver.read(admission["resource"])
            # Native admission refs are required on producer. No rewritten proof.
            proof = JOB.validate_resource_proof(proof_path, admission["resource"]["sha256"], protocol=basis,
                protocol_sha=C.digest_lf_file(basis_path), now=self.clock(), require_fresh=entry)
            self.last_live = deepcopy(proof["_lease"])
            E.require(proof["lease"] == raw_proof["lease"], "lease projection not implicit")
            capture = {key: proof[key] for key in E.GPU.CAPTURE_KEYS}
            provider = E.GPU.verify_provider_inventory(Path(capture["provider_inventory_path"]),
                capture["provider_inventory_sha256"], capture["resource_id"])
            target = layout["provider_target"]
            E.require(provider["id"] == target["resource_id"] and provider["name"] == target["resource_name"]
                      and provider["instance_type"] == target["instance_type"], "provider target binding")
            lease = resolver.read(proof["lease"])
            E.require(lease["acquired_at_utc"] == admission["lease_acquired_at_utc"]
                      and lease["expires_at_utc"] == admission["lease_expires_at_utc"], "admission lease metadata")
            model = resolver.file(admission["gpu_model"]).read_text(encoding="utf-8").strip().split(",")
            E.require(len(model) == 2 and model[0].strip() == capture["gpu_uuid"]
                      and model[1].strip() in {"NVIDIA L40S", "L40S"}, "current GPU UUID/model")
            return capture

    def backend(self, registered, folder, capture, check, mode):
        layout = registered["layout"]
        reader = E.GPU.StrictRemoteGpuReader(capture, folder.parent / "gpu-observations")
        guarded = P.GuardedReader(reader, check)
        python, lab = Path(layout["python"]), Path(layout["lab"])
        expected_argv = [str(python), "-B", str(WORKER.WORKER), "--job", str(folder / "job.json"),
                         "--output", str(folder / "worker.json")]
        bounded = P.BoundedPopen(check, registered["contract"]["timeouts"][mode], expected_argv)
        protocol = {**registered["backend_protocol"],
            "protocol_path": str(registered["resolver"].file(layout["backend_protocol"])),
            "protocol_sha256_lf": C.digest_lf_file(registered["resolver"].file(layout["backend_protocol"])),
            "fixed_eval_protocol": str(registered["resolver"].file(layout["fixed_eval_protocol"]))}
        backend = OWNED.SafeRecoveryBackend(lab, python, protocol, layout["backend_anchor"],
            read_gpu=guarded, popen=WORKER.worker_popen(bounded))
        return backend, bounded

    def boot_after(self, folder):
        path = folder / "boot-after.txt"
        with path.open("xb") as stream:
            stream.write(Path("/proc/sys/kernel/random/boot_id").read_bytes())
        return E.ref(path)


def training_bundle(stage, artifact):
    artifact_ref = E.write(stage / "backend-result.json", artifact)
    manifest = artifact["manifest"]
    return E.write(stage / "bundle.json", {"artifact": artifact_ref,
        **{k: E.ref(stage / (k + ".json")) for k in ("job", "worker", "execution")},
        **{k: E.ref(stage / (k + ".log")) for k in ("stdout", "stderr")},
        "manifest": {"path": artifact["manifest_path"], "sha256": artifact["manifest_sha256"]},
        "readback": E.ref(Path(manifest["readback_path"])), "records": E.ref(Path(manifest["records_path"])),
        "checkpoint": artifact["checkpoint"],
        "telemetry": {k: artifact["telemetry"][k] for k in ("path", "sha256")}})


class RepairProvider:
    """Local POSIX executor; main may carry its raw refs through a registered IO service."""

    def __init__(self, registration_ref, root_map, *, services=None, allow_fixture=False):
        E.require(type(allow_fixture) is bool, "explicit fixture mode")
        self.registration_ref, self.root_map = deepcopy(registration_ref), deepcopy(root_map)
        self.allow_fixture = allow_fixture
        self.services = services if services is not None else ActualServices()
        E.require(getattr(self.services, "synthetic_cpu", None) is allow_fixture
                  and (allow_fixture or type(self.services) is ActualServices), "actual services cannot be replaced")
        self._registered()

    def _registered(self):
        registered = E.registration(self.registration_ref, self.root_map, allow_fixture=self.allow_fixture)
        E.require(SOURCES <= registered["protocol"]["sources_lf"].keys(), "provider full source closure")
        E.require(self.root_map == registered["layout"]["root_maps"]["provider"], "provider native root map required")
        return registered

    def _prepare(self, operation, request_ref):
        registered = self._registered()
        resolver = registered["resolver"]
        request = resolver.read(request_ref)
        E.require(request["schema"] == SCHEMA + "_request" and request["operation_id"] == operation
                  and request["registration"] == self.registration_ref
                  and len(operation) == 64 and all(x in "0123456789abcdef" for x in operation), "provider exact request")
        E.verify_model(request, registered)
        root = resolver.path(registered["layout"]["output_root"])
        E.require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), "owned output root")
        return registered, request, root / operation

    def lookup(self, operation_id, request_ref):
        _registered, _request, folder = self._prepare(operation_id, request_ref)
        for name in ("outcome.json", "failure.json"):
            path = folder / name
            if path.exists():
                raw = E.E.loads(path.read_text(encoding="utf-8"))
                E.require(raw["request"] == request_ref and raw["operation_id"] == operation_id, "lookup request changed")
                if name == "outcome.json":
                    E.verify_repair_evidence(E.ref(path), request_ref=request_ref,
                        registration_ref=self.registration_ref, root_map=self.root_map)
                return E.ref(path)
        return None

    def execute(self, operation_id, request_ref, authorization_ref, admission_ref):
        registered, request, folder = self._prepare(operation_id, request_ref)
        E.require(request["authorization"] == authorization_ref, "provider approval ref changed")
        E.verify_authorization(request, authorization_ref, registered)
        expected_claim = {"operation_id": operation_id, "request": request_ref,
                          "authorization": authorization_ref, "admission": admission_ref}
        if folder.exists():
            E.require(E.E.loads((folder / "intent.json").read_text(encoding="utf-8")) == expected_claim,
                      "repeated dispatch binding changed")
            value = self.lookup(operation_id, request_ref)
            if value is not None:
                return value
            raise ValueError("repair dispatch interrupted; lookup only, no restart")
        admission = admission_body(admission_ref, operation_id, registered, self.services.clock(), entry=True)
        E.require(admission["registration"] == self.registration_ref, "admission registration")
        root = folder.parent
        claim = root / (self.registration_ref["sha256"] + ".claim.json")
        if claim.exists():
            E.require(E.E.loads(claim.read_text(encoding="utf-8")) == expected_claim, "one repair per registration")
            raise ValueError("claimed repair requires lookup; no repeated dispatch")
        E.write(claim, expected_claim)
        folder.mkdir()
        E.write(folder / "intent.json", expected_claim)
        bounded = None
        started = self.services.clock()
        deadline = started + timedelta(seconds=registered["contract"]["budget"]["wall_s"])

        def check(*, entry=False):
            fresh = self._registered()
            E.verify_model(request, fresh)
            E.verify_authorization(request, authorization_ref, fresh)
            # The transported immutable prefix does not replace this live local C check.
            E.require(E.J.file_sha(fresh["resolver"].path(request["approval_ledger"]["path"]))
                      == request["approval_ledger"]["sha256"], "C live consumed ledger changed")
            current_admission = admission_body(admission_ref, operation_id, fresh, self.services.clock(), entry=entry)
            E.require(current_admission == admission, "admission raw changed")
            E.require(started <= self.services.clock() < deadline, "total registered repair wall window exceeded")
            return self.services.check(fresh, admission, entry=entry)

        try:
            capture = check(entry=True)
            entry_ref = E.write(folder / "admission-entry.json", {"admission": admission_ref,
                "checked_at_utc": self.services.clock().isoformat(), "capture": capture,
                "synthetic_cpu": self.allow_fixture, "live_lease": getattr(self.services, "last_live", {})})
            sources = {role: E.verify_training(registered["layout"]["sources"][role], role, registered)
                       for role in ("case", "reference")}
            train_stage = folder / "train"
            backend, bounded = self.services.backend(registered, train_stage, capture, check, "train")
            check()
            trained = backend.train(train_stage, "CORRECTED", 2221, "repair_" + operation_id[:20],
                corrections=[E.CORRECTION], iterations=300,
                source_checkpoint_sha256=registered["contract"]["source_inputs"]["case"]["checkpoint_sha256"])
            check()
            bounded.close()
            E.require(not bounded.groups_alive(), "training owned children remain")
            training_ref = training_bundle(train_stage, trained)
            verified_training = E.verify_training(training_ref, "output", registered)
            E.verify_execution(verified_training["execution"], verified_training["worker"], verified_training["job"],
                               "train", registered, capture, admission)
            # New fixed evaluation only; never restart training on evaluate failure.
            eval_stage = folder / "evaluate"
            backend, bounded = self.services.backend(registered, eval_stage, capture, check, "evaluate")
            check()
            none = deepcopy(sources["reference"]["artifact"])
            for field in ("checkpoint", "telemetry"):
                none[field]["path"] = str(registered["resolver"].path(none[field]["path"]))
            evaluated = backend.evaluate(eval_stage, {"NONE": none, "CORRECTED": trained}, "repair_eval_" + operation_id[:20])
            check()
            bounded.close()
            E.require(not bounded.groups_alive(), "evaluation owned children remain")
            evaluation_ref = E.write(eval_stage / "bundle.json", {
                **{k: E.ref(eval_stage / (k + ".json")) for k in ("job", "worker", "execution")},
                **{k: E.ref(eval_stage / (k + ".log")) for k in ("stdout", "stderr")},
                "jobs": E.ref(eval_stage / "fixed_eval.jobs.json"),
                **{arm: E.ref(eval_stage / "reports" / (arm + ".json")) for arm in ("NONE", "CORRECTED")}})
            elapsed = trained["execution"]["wall_s"] + evaluated["execution"]["wall_s"]
            exit_ref = E.write(folder / "admission-exit.json", {"admission": admission_ref,
                "checked_at_utc": self.services.clock().isoformat(), "capture": capture,
                "synthetic_cpu": self.allow_fixture, "live_lease": getattr(self.services, "last_live", {})})
            wall = (self.services.clock() - started).total_seconds()
            body = {"schema": SCHEMA + "_outcome", "synthetic_cpu": self.allow_fixture, "status": "completed",
                "operation_id": operation_id, "request": request_ref, "registration": self.registration_ref,
                "authorization": authorization_ref, "admission": admission_ref,
                "training": training_ref, "evaluation": evaluation_ref,
                "lifetime": {"entry": entry_ref, "exit": exit_ref},
                "boot_after": self.services.boot_after(folder), "gpu_s": elapsed, "wall_s_total": wall,
                "compute_usd": wall * registered["contract"]["budget"]["usd_per_gpu_s"]}
            # Keep an unaccepted candidate if verification fails; never overwrite it as success.
            candidate = E.write(folder / "candidate.json", body)
            E.verify_repair_evidence(candidate, request_ref=request_ref,
                registration_ref=self.registration_ref, root_map=self.root_map)
            check()
            return E.write(folder / "outcome.json", body)
        except BaseException as error:
            E.write(folder / "failure.json", {"schema": SCHEMA + "_failure", "synthetic_cpu": self.allow_fixture,
                "status": "hold", "operation_id": operation_id, "request": request_ref,
                "authorization": authorization_ref, "admission": admission_ref,
                "error_type": type(error).__name__, "error": str(error)})
            raise
        finally:
            if bounded is not None:
                bounded.close()
                E.require(not bounded.groups_alive(), "owned children survived final cleanup")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--lookup", action="store_true")
    args = parser.parse_args(argv)
    invocation = E.E.loads(E.regular(args.invocation.absolute()).read_text(encoding="utf-8"))
    E.require(set(invocation) == {"operation_id", "registration_ref", "root_map", "request_ref", "authorization_ref", "admission_ref"},
              "CLI invocation fields")
    provider = RepairProvider(invocation["registration_ref"], invocation["root_map"])
    result = provider.lookup(invocation["operation_id"], invocation["request_ref"]) if args.lookup else provider.execute(
        invocation["operation_id"], invocation["request_ref"], invocation["authorization_ref"], invocation["admission_ref"])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
