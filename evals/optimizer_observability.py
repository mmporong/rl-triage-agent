"""PPO 학습 루프를 구성 가능한 기본 단위로 계측하고 판정한다.

행동 probe와 독립적이며, manifest의 기대값을 런타임 PPO 속성에서 만들지 않는다.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from types import MethodType
from typing import Any, Callable, Iterable, Iterator, Mapping


SCHEMA = "optimizer_observability_record_v2"
MANIFEST_SCHEMA = "optimizer_observability_manifest_v1"
REQUIRED_BINDING = ("arm", "seed", "task", "num_envs", "max_iterations", "source_commit_sha",
                    "protocol_path", "protocol_sha256_lf", "source_sha256_lf")


def validate_manifest(data: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "configured_learning_epochs", "configured_mini_batches",
        "expected_optimizer_steps_per_update", "expected_minibatches_per_update",
        "runtime_updates_to_observe",
        *REQUIRED_BINDING,
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"manifest 필드 누락: {missing}")
    out = dict(data)
    if out["schema"] != MANIFEST_SCHEMA:
        raise ValueError("지원하지 않는 manifest schema")
    for key in required - {"schema"}:
        if key in REQUIRED_BINDING:
            continue
        if not isinstance(out[key], int) or isinstance(out[key], bool) or out[key] <= 0:
            raise ValueError(f"{key}는 양의 정수여야 한다")
    product = out["configured_learning_epochs"] * out["configured_mini_batches"]
    if out["expected_optimizer_steps_per_update"] != product:
        raise ValueError("expected optimizer step은 고정 epoch×mini-batch와 같아야 한다")
    if out["expected_minibatches_per_update"] != product:
        raise ValueError("expected mini-batch는 고정 epoch×mini-batch와 같아야 한다")
    if not isinstance(out["arm"], str) or not out["arm"]:
        raise ValueError("arm은 빈 문자열이 아니어야 한다")
    if not isinstance(out["seed"], int) or isinstance(out["seed"], bool) or out["seed"] < 0:
        raise ValueError("seed는 0 이상의 정수여야 한다")
    for key in ("task", "protocol_path"):
        if not isinstance(out[key], str) or not out[key]:
            raise ValueError(f"{key}가 비어 있다")
    for key in ("num_envs", "max_iterations"):
        if not isinstance(out[key], int) or isinstance(out[key], bool) or out[key] <= 0:
            raise ValueError(f"{key}는 양의 정수여야 한다")
    if out["runtime_updates_to_observe"] > out["max_iterations"]:
        raise ValueError("관측 update 수가 학습 예산보다 크다")
    def valid_hex(value: Any, length: int) -> bool:
        if not isinstance(value, str) or len(value) != length:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True

    if not valid_hex(out["source_commit_sha"], 40):
        raise ValueError("source commit SHA 형식이 잘못됐다")
    if not valid_hex(out["protocol_sha256_lf"], 64):
        raise ValueError("protocol SHA 형식이 잘못됐다")
    if (not isinstance(out["source_sha256_lf"], dict) or not out["source_sha256_lf"]
            or any(not isinstance(path, str) or not path or not valid_hex(digest, 64)
                   for path, digest in out["source_sha256_lf"].items())):
        raise ValueError("source SHA commitment가 비어 있다")
    primitives = out.get("inspection_primitives")
    if primitives is not None and (not isinstance(primitives, list) or not primitives
                                   or any(not isinstance(item, str) or not item for item in primitives)
                                   or len(primitives) != len(set(primitives))):
        raise ValueError("inspection_primitives는 중복 없는 문자열 목록이어야 한다")
    return out


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    fixed = validate_manifest(manifest)
    material = json.dumps(fixed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def manifest_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    fixed = validate_manifest(manifest)
    return {key: fixed[key] for key in REQUIRED_BINDING}


def _number(value: Any) -> float | None:
    try:
        if hasattr(value, "item"):
            value = value.item()
        number = float(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    return number if math.isfinite(number) else None


def _named_parameters(policy: Any) -> list[tuple[str, Any]]:
    if hasattr(policy, "named_parameters"):
        return list(policy.named_parameters())
    return [(f"parameter_{index}", item) for index, item in enumerate(policy.parameters())]


def _adam_steps(optimizer: Any, named: list[tuple[str, Any]]) -> dict[str, float | None]:
    out = {}
    for name, parameter in named:
        state = optimizer.state.get(parameter, {})
        out[name] = _number(state.get("step"))
    return out


def _gradient_snapshot(named: list[tuple[str, Any]]) -> tuple[float | None, set[str]]:
    squared, present = 0.0, set()
    for name, parameter in named:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        present.add(name)
        try:
            norm = _number(grad.detach().norm(2))
        except (AttributeError, RuntimeError):
            norm = None
        if norm is None:
            return None, present
        squared += norm * norm
    return (math.sqrt(squared) if present else None), present


@contextmanager
def _temporary_instance_attr(obj: Any, name: str, value: Any) -> Iterator[None]:
    namespace = getattr(obj, "__dict__", {})
    had = name in namespace
    previous = namespace.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, previous)
        elif name in getattr(obj, "__dict__", {}):
            delattr(obj, name)


@dataclass
class RuntimeObserver:
    ppo: Any
    manifest: dict[str, Any]
    sink: Callable[[dict[str, Any]], None]
    update_index: int = 0
    _installed: bool = False
    _had_update: bool = False
    _previous_update: Any = None

    def install(self) -> "RuntimeObserver":
        if self._installed:
            raise RuntimeError("observer가 이미 설치됨")
        namespace = getattr(self.ppo, "__dict__", {})
        self._had_update = "update" in namespace
        self._previous_update = namespace.get("update")
        original = self.ppo.update

        def observed(this):
            return self._observe_update(original)

        self.ppo.update = MethodType(observed, self.ppo)
        self._installed = True
        return self

    def remove(self) -> None:
        if not self._installed:
            return
        if self._had_update:
            self.ppo.update = self._previous_update
        elif "update" in getattr(self.ppo, "__dict__", {}):
            delattr(self.ppo, "update")
        self._installed = False

    def _observe_update(self, original: Callable[[], Any]) -> Any:
        index = self.update_index
        self.update_index += 1
        limit = self.manifest["runtime_updates_to_observe"]
        if index >= limit:
            return original()
        ppo, optimizer = self.ppo, self.ppo.optimizer
        storage = ppo.storage
        generator_name = "recurrent_mini_batch_generator" if ppo.policy.is_recurrent else "mini_batch_generator"
        generator = getattr(storage, generator_name)
        step = optimizer.step
        named = _named_parameters(ppo.policy)
        before = _adam_steps(optimizer, named)
        counts = {"minibatches": 0, "optimizer_steps": 0}
        gradient_norms: list[float | None] = []
        gradient_presence = {name: 0 for name, _ in named}
        learning_rates: list[list[float | None]] = []
        error = None

        def observed_generator(*args, **kwargs):
            for batch in generator(*args, **kwargs):
                counts["minibatches"] += 1
                yield batch

        def observed_step(*args, **kwargs):
            norm, present = _gradient_snapshot(named)
            gradient_norms.append(norm)
            for name in present:
                gradient_presence[name] += 1
            learning_rates.append([_number(group.get("lr")) for group in optimizer.param_groups])
            counts["optimizer_steps"] += 1
            return step(*args, **kwargs)

        try:
            with _temporary_instance_attr(storage, generator_name, observed_generator):
                with _temporary_instance_attr(optimizer, "step", observed_step):
                    result = original()
            return result
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            raise
        finally:
            after = _adam_steps(optimizer, named)
            delta = {
                name: None if before[name] is None or after[name] is None else after[name] - before[name]
                for name, _ in named
            }
            record = {
                "schema": SCHEMA,
                "manifest_sha256": manifest_fingerprint(self.manifest),
                "binding": manifest_binding(self.manifest),
                "update_index": index,
                "expected": {
                    "optimizer_steps": self.manifest["expected_optimizer_steps_per_update"],
                    "minibatches": self.manifest["expected_minibatches_per_update"],
                },
                "runtime_configuration": {
                    "learning_epochs": getattr(ppo, "num_learning_epochs", None),
                    "mini_batches": getattr(ppo, "num_mini_batches", None),
                },
                "observed": {
                    **counts,
                    "adam_step_before": before,
                    "adam_step_after": after,
                    "adam_step_delta": delta,
                    "gradient_present_steps": gradient_presence,
                    "pre_step_gradient_norm": gradient_norms,
                    "learning_rates": learning_rates,
                    "approx_kl": None,
                    "clip_fraction": None,
                },
                "unavailable": {
                    "approx_kl": "RSL-RL 2.3.3 PPO.update의 지역 변수이며 공개 hook 없음",
                    "clip_fraction": "RSL-RL 2.3.3이 계산 결과를 기록하지 않음",
                },
                "error": error,
            }
            self.sink(record)


def _count_primitive(records: list[Mapping[str, Any]], key: str, expected_key: str,
                     expected_records: int) -> dict[str, Any]:
    if not records:
        return {"status": "unknown", "first_mismatch_update": None}
    mismatches = [row["update_index"] for row in records
                  if row.get("observed", {}).get(key) != row.get("expected", {}).get(expected_key)]
    error = any(row.get("error") for row in records)
    incomplete = [row.get("update_index") for row in records] != list(range(expected_records))
    return {"status": "unknown" if error else ("abnormal" if mismatches else ("unknown" if incomplete else "normal")),
            "first_mismatch_update": min(mismatches) if mismatches else None}


@dataclass(frozen=True)
class InspectionContext:
    records: list[dict[str, Any]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    evaluate: Callable[[InspectionContext], dict[str, Any]]
    decisive: bool = False


class PrimitiveRegistry:
    def __init__(self, specs: Iterable[PrimitiveSpec] | None = None):
        self._specs: dict[str, PrimitiveSpec] = {}
        for spec in specs or ():
            self.register(spec)

    def register(self, spec: PrimitiveSpec) -> None:
        if not spec.name or spec.name in self._specs:
            raise ValueError(f"primitive 이름이 비었거나 중복됨: {spec.name!r}")
        self._specs[spec.name] = spec

    def resolve(self, names: list[str]) -> list[PrimitiveSpec]:
        missing = [name for name in names if name not in self._specs]
        if missing:
            raise ValueError(f"등록되지 않은 primitive: {missing}")
        return [self._specs[name] for name in names]

    def specs(self) -> tuple[PrimitiveSpec, ...]:
        return tuple(self._specs.values())


def _count_evaluator(observed_key: str, expected_key: str) -> Callable[[InspectionContext], dict[str, Any]]:
    return lambda ctx: _count_primitive(ctx.records, observed_key, expected_key,
                                        ctx.manifest["runtime_updates_to_observe"])


def _adam_evaluator(ctx: InspectionContext) -> dict[str, Any]:
    records, expected_records = ctx.records, ctx.manifest["runtime_updates_to_observe"]
    unknown = (not records or [row.get("update_index") for row in records] != list(range(expected_records))
               or any(row.get("error") for row in records)
               or any(any(delta is None for delta in row["observed"]["adam_step_delta"].values()) for row in records)
               or any(any(count != row["observed"]["optimizer_steps"]
                          for count in row["observed"]["gradient_present_steps"].values()) for row in records))
    mismatches = [] if unknown else [row["update_index"] for row in records
        if any(delta != row["observed"]["optimizer_steps"] for delta in row["observed"]["adam_step_delta"].values())]
    return {"status": "unknown" if unknown else ("abnormal" if mismatches else "normal"),
            "first_mismatch_update": min(mismatches) if mismatches else None}


def _gradient_evaluator(ctx: InspectionContext) -> dict[str, Any]:
    values = [value for row in ctx.records for value in row["observed"]["pre_step_gradient_norm"]]
    return {"status": "observed" if values else "unavailable", "values": values}


def _learning_rate_evaluator(ctx: InspectionContext) -> dict[str, Any]:
    values = [value for row in ctx.records for group in row["observed"]["learning_rates"] for value in group]
    return {"status": "observed" if values else "unavailable", "values": values}


def _unavailable(_ctx: InspectionContext) -> dict[str, Any]:
    return {"status": "unavailable"}


DEFAULT_REGISTRY = PrimitiveRegistry(iter((
    PrimitiveSpec("optimizer_step_count", _count_evaluator("optimizer_steps", "optimizer_steps"), decisive=True),
    PrimitiveSpec("minibatch_count", _count_evaluator("minibatches", "minibatches"), decisive=True),
    PrimitiveSpec("adam_step_delta", _adam_evaluator),
    PrimitiveSpec("pre_step_gradient_norm", _gradient_evaluator),
    PrimitiveSpec("learning_rate_trajectory", _learning_rate_evaluator),
    PrimitiveSpec("approx_kl", _unavailable),
    PrimitiveSpec("clip_fraction", _unavailable),
)))
PRIMITIVES = tuple(spec.name for spec in DEFAULT_REGISTRY.specs())


def _validate_record_commitments(records: list[Mapping[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    fingerprint = manifest_fingerprint(manifest)
    binding = manifest_binding(manifest)
    expected = {"optimizer_steps": manifest["expected_optimizer_steps_per_update"],
                "minibatches": manifest["expected_minibatches_per_update"]}
    out = []
    for index, row in enumerate(records):
        if row.get("schema") != SCHEMA:
            raise ValueError(f"record {index} schema가 다르다")
        if row.get("manifest_sha256") != fingerprint or row.get("binding") != binding:
            raise ValueError(f"record {index} manifest commitment가 다르다")
        if row.get("expected") != expected:
            raise ValueError(f"record {index} expected 값이 manifest와 다르다")
        if not isinstance(row.get("observed"), dict):
            raise ValueError(f"record {index} observed가 없다")
        out.append(dict(row))
    return out


def assemble_inspection(records: list[Mapping[str, Any]], manifest: Mapping[str, Any],
                        registry: PrimitiveRegistry = DEFAULT_REGISTRY) -> dict[str, Any]:
    """manifest가 고른 등록 기본 단위를 조립한다. record commitment 불일치는 거부한다."""
    fixed = validate_manifest(manifest)
    valid = _validate_record_commitments(records, fixed)
    names = fixed.get("inspection_primitives", list(PRIMITIVES))
    specs = registry.resolve(names)
    context = InspectionContext(valid, fixed)
    primitives = {spec.name: spec.evaluate(context) for spec in specs}
    statuses = [primitives[spec.name]["status"] for spec in specs if spec.decisive]
    overall = ("abnormal" if "abnormal" in statuses else
               "normal" if statuses and all(status == "normal" for status in statuses) else "unknown")
    return {
        "schema": "optimizer_observability_inspection_v1",
        "manifest": fixed,
        "records": len(valid),
        "status": overall,
        "primitives": primitives,
    }


def compare_checkpoint_adam_steps(before: Mapping[str, Any], after: Mapping[str, Any],
                                  metadata: Mapping[str, Any]) -> dict[str, Any]:
    """체크포인트 Adam step 차이. 상태 연속성 근거가 빠지면 정상/비정상을 만들지 않는다."""
    required_true = ("checkpoint_semantics_source_verified", "optimizer_state_continuous",
                     "frozen_parameters_known", "parameter_set_complete")
    completed = metadata.get("completed_update_delta")
    steps_per_update = metadata.get("expected_optimizer_steps_per_update")
    reasons = [name for name in required_true if metadata.get(name) is not True]
    if not isinstance(completed, int) or isinstance(completed, bool) or completed <= 0:
        reasons.append("completed_update_delta")
    if not isinstance(steps_per_update, int) or isinstance(steps_per_update, bool) or steps_per_update <= 0:
        reasons.append("expected_optimizer_steps_per_update")
    before_steps, after_steps = before.get("steps", {}), after.get("steps", {})
    if set(before_steps) != set(after_steps) or not before_steps:
        reasons.append("parameter_set")
    if reasons:
        return {"status": "unknown", "reasons": sorted(set(reasons)), "deltas": None}
    deltas = {name: after_steps[name] - before_steps[name] for name in before_steps}
    frozen = set(metadata.get("frozen_parameters", []))
    if frozen:
        return {"status": "unknown", "reasons": ["frozen_parameters"], "deltas": deltas}
    expected = completed * steps_per_update
    mismatches = {name: value for name, value in deltas.items() if value != expected}
    return {"status": "abnormal" if mismatches else "normal", "reasons": [], "deltas": deltas,
            "expected_adam_step_delta": expected, "mismatches": mismatches}
