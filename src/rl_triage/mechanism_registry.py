"""명시적으로 전달된 검사 registry로 진단 고리와 승인 원장을 실행한다."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
from typing import Any

from rl_triage import probe_loop as legacy


REQUIRED_KEYS = {"mechanisms", "outcomes", "probes", "fixed_order"}


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("registry must be finite canonical JSON") from error


def normalize(registry: dict) -> dict:
    """검사 정의를 검증하고 호출자 변조와 분리된 canonical 사본을 반환한다."""
    if not isinstance(registry, dict) or set(registry) != REQUIRED_KEYS:
        raise ValueError("registry fields differ")
    value = _json_copy(registry)
    mechanisms, outcomes = value["mechanisms"], value["outcomes"]
    if (not isinstance(mechanisms, list) or not mechanisms
            or any(not isinstance(item, str) or not item for item in mechanisms)
            or len(set(mechanisms)) != len(mechanisms)):
        raise ValueError("registry mechanisms are invalid")
    if outcomes != list(legacy.OUTCOMES):
        raise ValueError("registry outcomes differ from the diagnostic loop contract")
    probes = value["probes"]
    if not isinstance(probes, dict) or not probes:
        raise ValueError("registry probes are missing")
    for name, spec in probes.items():
        if (not isinstance(name, str) or not name or not isinstance(spec, dict)
                or set(spec) != {"measures", "cost_gpu_s", "args", "expect"}
                or not isinstance(spec["measures"], str) or not spec["measures"]
                or isinstance(spec["cost_gpu_s"], bool) or not isinstance(spec["cost_gpu_s"], (int, float))
                or not math.isfinite(spec["cost_gpu_s"]) or spec["cost_gpu_s"] < 0
                or not isinstance(spec["args"], dict) or not isinstance(spec["expect"], dict)
                or set(spec["expect"]) != set(mechanisms)
                or any(item not in ("abnormal", "normal", None) for item in spec["expect"].values())):
            raise ValueError(f"invalid probe definition: {name!r}")
    if not math.isfinite(sum(spec["cost_gpu_s"] for spec in probes.values())):
        raise ValueError("registry total probe cost must be finite")
    if (not isinstance(value["fixed_order"], list) or not value["fixed_order"]
            or any(not isinstance(item, str) for item in value["fixed_order"])
            or len(value["fixed_order"]) != len(set(value["fixed_order"]))
            or not set(value["fixed_order"]) <= set(probes)):
        raise ValueError("fixed order must contain unique registered probes")
    return value


def registry_digest(registry: dict) -> str:
    return legacy.digest(normalize(registry))


def legacy_registry() -> dict:
    return normalize({"mechanisms": list(legacy.MECHANISMS), "outcomes": list(legacy.OUTCOMES),
                      "probes": deepcopy(legacy.PROBES), "fixed_order": list(legacy.FIXED_ORDER)})


def without_torque_signal(registry: dict | None = None) -> dict:
    """P_torque가 어떤 mechanism도 지지·기각하지 않는 명시적 정책을 만든다."""
    value = normalize(legacy_registry() if registry is None else registry)
    if "P_torque" not in value["probes"]:
        raise ValueError("P_torque is not registered")
    value["probes"]["P_torque"]["expect"] = {name: None for name in value["mechanisms"]}
    value["fixed_order"] = [name for name in value["fixed_order"] if name != "P_torque"]
    return normalize(value)


def _confirming(registry: dict, mechanism: str) -> str | None:
    for probe, spec in registry["probes"].items():
        if spec["expect"].get(mechanism) == "abnormal" and all(
                outcome != "abnormal" for name, outcome in spec["expect"].items() if name != mechanism):
            return probe
    return None


def _select(registry: dict, hypotheses: list[str], done: list[str]) -> str | None:
    candidates = []
    for probe, spec in registry["probes"].items():
        predictions = {spec["expect"].get(name) for name in hypotheses}
        if probe not in done and {"abnormal", "normal"} <= predictions:
            candidates.append(probe)
    return min(candidates, key=lambda probe: (registry["probes"][probe]["cost_gpu_s"], probe)) if candidates else None


def _next(registry: dict, hypotheses: list[str], observed: dict[str, str]) -> str | None:
    if len(hypotheses) == 1:
        probe = _confirming(registry, hypotheses[0])
        return probe if probe is not None and probe not in observed else None
    return _select(registry, hypotheses, list(observed))


def update(registry: dict, hypotheses: list[str], probe: str, outcome: str) -> tuple[list[str], list[str]]:
    registry = normalize(registry)
    if probe not in registry["probes"] or any(name not in registry["mechanisms"] for name in hypotheses):
        raise ValueError("unregistered probe or hypothesis")
    if outcome not in registry["outcomes"]:
        raise ValueError(f"unknown outcome: {outcome!r}")
    if outcome == "unknown":
        return list(hypotheses), []
    keep, dropped = [], []
    for name in hypotheses:
        prediction = registry["probes"][probe]["expect"].get(name)
        (dropped if prediction is not None and prediction != outcome else keep).append(name)
    return keep, dropped


def _status(registry: dict, hypotheses: list[str], observed: dict[str, str], budget_left: int) -> str:
    if not hypotheses:
        return "none_supported"
    if len(hypotheses) == 1:
        probe = _confirming(registry, hypotheses[0])
        if probe is not None and observed.get(probe) == "abnormal":
            return "confirmed"
    if budget_left <= 0 or (len(hypotheses) > 1 and _select(registry, hypotheses, list(observed)) is None):
        return "unidentifiable"
    return "open"


def simulate(registry: dict, strategy: str, outcomes: dict[str, str], ranking: list[str] | None = None,
             top_k: int = 3, budget: int | None = None, seed: int = 0) -> dict:
    registry = normalize(registry)
    probes, mechanisms = registry["probes"], registry["mechanisms"]
    budget = len(probes) if budget is None else budget
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    if strategy == "discriminate":
        if not ranking or any(name not in mechanisms for name in ranking):
            raise ValueError("discriminate requires a registered ranking")
        hypotheses = list(ranking[:top_k])
        order = None
    else:
        hypotheses = list(mechanisms)
        orders = {"exhaustive": sorted(probes), "fixed": list(registry["fixed_order"]),
                  "random": random.Random(seed).sample(sorted(probes), len(probes))}
        if strategy not in orders:
            raise ValueError(f"unknown strategy: {strategy!r}")
        order = orders[strategy]
    observed: dict[str, str] = {}
    steps = []
    while len(observed) < budget:
        if strategy != "exhaustive" and _status(registry, hypotheses, observed, budget - len(observed)) != "open":
            break
        probe = (_next(registry, hypotheses, observed) if strategy == "discriminate"
                 else next((name for name in order if name not in observed), None))
        if probe is None:
            break
        outcome = outcomes.get(probe, "unknown")
        observed[probe] = outcome
        hypotheses, dropped = update(registry, hypotheses, probe, outcome)
        steps.append({"probe": probe, "outcome": outcome, "dropped": dropped, "left": list(hypotheses)})
    final = _status(registry, hypotheses, observed, budget - len(observed))
    if final == "open":
        final = "unidentifiable"
    return {"strategy": strategy, "status": final, "conclusion": hypotheses[0] if final == "confirmed" else None,
            "probes_used": len(observed), "gpu_s": sum(probes[name]["cost_gpu_s"] for name in observed),
            "steps": steps}


class RegistryLedger(legacy.Ledger):
    """요청의 args와 registry digest를 함께 묶는 append-only 원장."""

    def __init__(self, path: Path, registry: dict):
        super().__init__(path)
        self._registry = normalize(registry)
        self.registry_digest = legacy.digest(self._registry)

    def effect_key(self, prereg_digest: str, probe: str, args_digest: str) -> str:
        return legacy.digest({"prereg": prereg_digest, "probe": probe, "args": args_digest,
                              "registry": self.registry_digest})[:16]

    def _bound_request(self, request_id: str, state: str) -> dict:
        request = self.requests().get(request_id)
        if request is None or request["state"] != state:
            raise legacy.LedgerError(f"{request_id}: request is not {state}")
        if request.get("registry_digest") != self.registry_digest:
            raise legacy.LedgerError(f"{request_id}: registry changed")
        if request.get("effect_key") != self.effect_key(
                request["prereg_digest"], request["probe"], request["args_digest"]):
            raise legacy.LedgerError(f"{request_id}: effect key differs")
        return request

    def propose(self, prereg: dict, probe: str, budget_gpu_s: int) -> dict:
        if probe not in self._registry["probes"]:
            raise legacy.LedgerError(f"unknown probe: {probe!r}")
        if type(budget_gpu_s) is not int or budget_gpu_s < 0:
            raise legacy.LedgerError("GPU budget must be a non-negative integer")
        args = self._registry["probes"][probe]["args"]
        prereg_digest, args_digest = legacy.digest(prereg), legacy.digest(args)
        effect_key = self.effect_key(prereg_digest, probe, args_digest)
        same = [row for row in self.requests().values() if row.get("effect_key") == effect_key]
        live = [row["request_id"] for row in same if row["state"] in ("pending", "approved", "running")]
        if live:
            raise legacy.LedgerError(f"same experiment is already active: {live}")
        done = [row["request_id"] for row in same if row["state"] == "done"]
        request_id = legacy.digest({"prereg": prereg_digest, "probe": probe, "args": args,
                                    "registry": self.registry_digest, "n": len(self.events())})[:12]
        event = {"event": "proposed", "request_id": request_id, "prereg_digest": prereg_digest,
                 "probe": probe, "args_digest": args_digest, "registry_digest": self.registry_digest,
                 "effect_key": effect_key, "budget_gpu_s": budget_gpu_s}
        if done:
            event["repeat_of"] = done[-1]
        return self._append(event)

    def approve(self, request_id: str, approver: str) -> dict:
        request = self._bound_request(request_id, "pending")
        return self._append({"event": "approved", "request_id": request_id, "approver": approver,
                             "registry_digest": self.registry_digest, "effect_key": request["effect_key"]})

    def consume(self, request_id: str, prereg: dict, probe: str) -> dict:
        request = self._bound_request(request_id, "approved")
        spec = self._registry["probes"].get(probe)
        if (spec is None or request["prereg_digest"] != legacy.digest(prereg) or request["probe"] != probe
                or request["args_digest"] != legacy.digest(spec["args"])):
            raise legacy.LedgerError(f"{request_id}: approved prereg, probe, args, or registry differs")
        return self._append({"event": "consumed", "request_id": request_id,
                             "registry_digest": self.registry_digest, "args_digest": request["args_digest"],
                             "effect_key": request["effect_key"]})

    def receipt(self, request_id: str, outcome: str, measurement: dict, gpu_s: float,
                exit_code: int | None, recovered_from: str | None = None,
                reason: str | None = None) -> dict:
        request = self._bound_request(request_id, "running")
        if (isinstance(gpu_s, bool) or not isinstance(gpu_s, (int, float))
                or not math.isfinite(gpu_s) or gpu_s < 0):
            raise legacy.LedgerError("receipt GPU duration must be finite and non-negative")
        if exit_code is not None and type(exit_code) is not int:
            raise legacy.LedgerError("receipt exit code must be an integer or null")
        if not isinstance(measurement, dict):
            raise legacy.LedgerError("receipt measurement must be a JSON object")
        try:
            measurement = _json_copy(measurement)
        except ValueError as error:
            raise legacy.LedgerError("receipt measurement must be finite JSON") from error
        over_budget = gpu_s > request["budget_gpu_s"]
        if exit_code != 0 or over_budget:
            outcome = "unknown"
        if outcome not in self._registry["outcomes"]:
            raise legacy.LedgerError(f"unknown outcome: {outcome!r}")
        event = {"event": "receipt", "request_id": request_id, "outcome": outcome,
                 "measurement": measurement, "gpu_s": gpu_s, "exit_code": exit_code,
                 "over_budget": over_budget, "registry_digest": self.registry_digest,
                 "args_digest": request["args_digest"], "effect_key": request["effect_key"]}
        if recovered_from is not None:
            event["recovered_from"] = recovered_from
        if reason is not None:
            event["reason"] = reason
        return self._append(event)

    def recover(self, request_id: str, artifact: dict | None, prereg: dict | None = None) -> dict:
        request = self._bound_request(request_id, "running")
        if artifact is None:
            return self.receipt(request_id, "unknown", {}, 0.0, None, reason="orphan_without_artifact")
        if prereg is None or legacy.digest(prereg) != request["prereg_digest"]:
            raise legacy.LedgerError(f"{request_id}: approved preregistration is required for recovery")
        required = {"outcome", "measurement", "gpu_s", "exit_code", "source"}
        if (not isinstance(artifact, dict) or not required <= set(artifact)
                or not isinstance(artifact["source"], str) or not artifact["source"]):
            raise legacy.LedgerError(f"{request_id}: recovery artifact execution fields are missing")
        bindings = {"probe": request["probe"], "args_digest": request["args_digest"],
                    "registry_digest": self.registry_digest, "effect_key": request["effect_key"]}
        checkpoint = prereg.get("checkpoint_sha256")
        if (not isinstance(checkpoint, str) or len(checkpoint) != 64
                or any(char not in "0123456789abcdef" for char in checkpoint)
                or artifact.get("checkpoint_sha256") != checkpoint):
            raise legacy.LedgerError(f"{request_id}: artifact checkpoint differs")
        if any(artifact.get(key) != value for key, value in bindings.items()):
            raise legacy.LedgerError(f"{request_id}: artifact probe, args, registry, or effect binding differs")
        return self.receipt(request_id, artifact["outcome"], artifact["measurement"],
                            artifact["gpu_s"], artifact["exit_code"],
                            recovered_from=artifact["source"])
