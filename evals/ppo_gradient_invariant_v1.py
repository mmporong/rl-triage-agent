"""RSL-RL 2.3.3 PPO clipped surrogate의 log-prob gradient 불변식을 관찰한다."""
from __future__ import annotations

import math
from types import MethodType
from typing import Any, Callable, Mapping, Sequence

import optimizer_observability as O


GRADIENT_SCHEMA = "ppo_surrogate_gradient_v1"
PRIMITIVE = "ppo_surrogate_gradient"
ATOL = 1.0e-6
RTOL = 1.0e-5
SCALE = "batch_size"
CLIP_PARAM = 0.2
MAX_UPDATES = 3


def validate_manifest(data: Mapping[str, Any]) -> dict[str, Any]:
    """기존 manifest와 고정 gradient 수치 계약을 함께 검증한다."""
    out = O.validate_manifest(data)
    contract = out.get("gradient_invariant")
    expected = {
        "schema": GRADIENT_SCHEMA,
        "atol": ATOL,
        "rtol": RTOL,
        "scale": SCALE,
        "clip_param": CLIP_PARAM,
    }
    if contract != expected:
        raise ValueError(f"gradient_invariant는 고정 수치 계약과 같아야 한다: {expected}")
    if out["runtime_updates_to_observe"] > MAX_UPDATES:
        raise ValueError(f"gradient invariant 관측은 최대 {MAX_UPDATES} update다")
    names = out.get("inspection_primitives")
    required = {"optimizer_step_count", "minibatch_count", PRIMITIVE}
    if names is None:
        raise ValueError("inspection_primitives는 gradient invariant manifest에 필수다")
    if not required.issubset(names):
        raise ValueError(f"inspection_primitives에 필수 primitive가 없다: {sorted(required - set(names))}")
    return out


def _unsupported_reason(ppo: Any) -> str | None:
    if getattr(ppo.policy, "is_recurrent", False):
        return "recurrent_policy_unsupported"
    if getattr(ppo, "symmetry", None) is not None:
        return "symmetry_unsupported"
    if getattr(ppo, "rnd", None) is not None:
        return "rnd_unsupported"
    if getattr(ppo, "is_multi_gpu", False):
        return "multi_gpu_unsupported"
    if getattr(ppo, "normalize_advantage_per_mini_batch", False):
        return "per_minibatch_advantage_normalization_unsupported"
    return None


class GradientObserver(O.RuntimeObserver):
    """기존 RuntimeObserver의 count 증거에 surrogate gradient 증거를 더한다."""

    def __init__(self, ppo: Any, manifest: Mapping[str, Any], sink: Callable[[dict[str, Any]], None]):
        fixed = validate_manifest(manifest)
        runtime_clip = getattr(ppo, "clip_param", None)
        if (isinstance(runtime_clip, bool) or not isinstance(runtime_clip, (int, float))
                or float(runtime_clip) != CLIP_PARAM):
            raise ValueError(f"PPO clip_param은 manifest 고정값 {CLIP_PARAM}와 같아야 한다")
        self._external_sink = sink
        self._gradient_state: dict[str, Any] | None = None
        super().__init__(ppo, fixed, self._emit_record)

    def _observe_update(self, original: Callable[[], Any]) -> Any:
        index = self.update_index
        if index >= self.manifest["runtime_updates_to_observe"]:
            return super()._observe_update(original)

        reason = _unsupported_reason(self.ppo)
        state: dict[str, Any] = {
            "reason": reason,
            "yielded": 0,
            "registered": 0,
            "callbacks": 0,
            "batches": [],
            "pending": None,
            "handles": [],
        }
        self._gradient_state = state
        if reason is not None:
            try:
                return super()._observe_update(original)
            finally:
                self._gradient_state = None

        storage = self.ppo.storage
        generator_name = "mini_batch_generator"
        generator = getattr(storage, generator_name)
        policy = self.ppo.policy
        get_log_prob = policy.get_actions_log_prob

        def observed_generator(*args, **kwargs):
            for batch in generator(*args, **kwargs):
                state["yielded"] += 1
                if not isinstance(batch, (tuple, list)) or len(batch) <= 6:
                    state["reason"] = "batch_layout_missing"
                    state["pending"] = None
                else:
                    state["pending"] = {
                        "advantages": batch[4],
                        "old_log_prob": batch[6],
                        "hooked": False,
                    }
                yield batch
                state["pending"] = None

        def observed_get_log_prob(this, *args, **kwargs):
            value = get_log_prob(*args, **kwargs)
            pending = state.get("pending")
            if pending is None:
                return value
            if pending["hooked"]:
                state["reason"] = "multiple_log_prob_calls_per_batch"
                return value
            pending["hooked"] = True
            state["registered"] += 1
            self._register_gradient_hook(value, pending, state)
            return value

        try:
            with O._temporary_instance_attr(storage, generator_name, observed_generator):
                with O._temporary_instance_attr(policy, "get_actions_log_prob", MethodType(observed_get_log_prob, policy)):
                    return super()._observe_update(original)
        finally:
            for handle in state["handles"]:
                handle.remove()
            state["handles"].clear()
            state["pending"] = None
            self._gradient_state = None

    def _register_gradient_hook(self, value: Any, pending: dict[str, Any], state: dict[str, Any]) -> None:
        try:
            import torch
        except ImportError:
            state["reason"] = "torch_unavailable"
            return
        if not isinstance(value, torch.Tensor) or not value.requires_grad:
            state["reason"] = "log_prob_tensor_not_differentiable"
            return

        with torch.no_grad():
            new_snapshot = value.detach().clone()
            advantages = pending["advantages"].detach().clone()
            old_log_prob = pending["old_log_prob"].detach().clone()

        def inspect_gradient(gradient):
            state["callbacks"] += 1
            summary = self._summarize_batch(torch, new_snapshot, gradient, advantages, old_log_prob)
            state["batches"].append(summary)
            return None

        state["handles"].append(value.register_hook(inspect_gradient))

    def _summarize_batch(self, torch: Any, new_log_prob: Any, gradient: Any,
                         advantages: Any, old_log_prob: Any) -> dict[str, Any]:
        with torch.no_grad():
            new = torch.squeeze(new_log_prob.detach())
            actual = torch.squeeze(gradient.detach())
            advantage = torch.squeeze(advantages.detach())
            old = torch.squeeze(old_log_prob.detach())
            n = int(new.numel())
            finite = bool(
                new.dtype == torch.float32
                and actual.dtype == torch.float32
                and advantage.dtype == torch.float32
                and old.dtype == torch.float32
                and torch.isfinite(new).all().item()
                and torch.isfinite(actual).all().item()
                and torch.isfinite(advantage).all().item()
                and torch.isfinite(old).all().item()
            )
            shape_ok = new.shape == actual.shape == advantage.shape == old.shape and n > 0
            if not shape_ok:
                return {
                    "max_abs_error": None,
                    "mismatch_count": None,
                    "nonzero_expected_count": None,
                    "finite": finite,
                    "n": n,
                }
            ratio = torch.exp(new - old)
            active = ((advantage >= 0) & (ratio <= 1.0 + CLIP_PARAM)) | (
                (advantage < 0) & (ratio >= 1.0 - CLIP_PARAM)
            )
            expected = torch.where(active, -advantage * ratio, torch.zeros_like(ratio))
            scaled_actual = actual * n
            error = torch.abs(scaled_actual - expected)
            finite = bool(
                finite
                and torch.isfinite(ratio).all().item()
                and torch.isfinite(expected).all().item()
                and torch.isfinite(error).all().item()
            )
            close = torch.isclose(scaled_actual, expected, atol=ATOL, rtol=RTOL)
            return {
                "max_abs_error": float(error.max().item()) if finite else None,
                "mismatch_count": int((~close).sum().item()) if finite else n,
                "nonzero_expected_count": int((expected != 0).sum().item()),
                "finite": finite,
                "n": n,
                "ratio_min": float(ratio.min().item()) if torch.isfinite(ratio).all().item() else None,
                "ratio_max": float(ratio.max().item()) if torch.isfinite(ratio).all().item() else None,
                "clipped_gradient_fraction": float((~active).to(torch.float32).mean().item()),
            }

    def _emit_record(self, record: dict[str, Any]) -> None:
        state = self._gradient_state
        if state is None:
            evidence = {"status": "unknown", "batches": [], "reason": "observer_state_missing"}
        else:
            reason = state["reason"]
            summaries = list(state["batches"])
            complete = (
                reason is None
                and state["yielded"] > 0
                and state["registered"] == state["yielded"]
                and state["callbacks"] == state["yielded"]
                and len(summaries) == state["yielded"]
            )
            if record.get("error"):
                evidence = {"status": "unknown", "batches": summaries, "reason": "update_exception"}
            elif not complete:
                evidence = {
                    "status": "unknown",
                    "batches": summaries,
                    "reason": reason or "one_gradient_hook_per_batch_not_observed",
                }
            elif any(row["mismatch_count"] is None for row in summaries):
                evidence = {"status": "unknown", "batches": summaries, "reason": "tensor_shape_mismatch"}
            elif any(not row["finite"] for row in summaries):
                evidence = {"status": "unknown", "batches": summaries, "reason": "nonfinite_or_non_float32"}
            elif any(row["mismatch_count"] for row in summaries):
                evidence = {"status": "abnormal", "batches": summaries}
            elif not any(row["nonzero_expected_count"] for row in summaries):
                evidence = {"status": "unknown", "batches": summaries, "reason": "all_expected_gradients_zero"}
            else:
                evidence = {"status": "normal", "batches": summaries}
        record["observed"][PRIMITIVE] = evidence
        self._external_sink(record)


def _valid_summary(summary: Any) -> bool:
    if not isinstance(summary, dict):
        return False
    n = summary.get("n")
    mismatch = summary.get("mismatch_count")
    nonzero = summary.get("nonzero_expected_count")
    max_error = summary.get("max_abs_error")
    return (
        isinstance(summary.get("finite"), bool)
        and isinstance(n, int) and not isinstance(n, bool) and n > 0
        and isinstance(mismatch, int) and not isinstance(mismatch, bool) and 0 <= mismatch <= n
        and isinstance(nonzero, int) and not isinstance(nonzero, bool) and 0 <= nonzero <= n
        and isinstance(max_error, (int, float)) and not isinstance(max_error, bool)
        and math.isfinite(float(max_error)) and max_error >= 0
    )


def _recompute_evidence(row: Mapping[str, Any], evidence: Any) -> tuple[str, int]:
    if not isinstance(evidence, dict) or not isinstance(evidence.get("batches"), list):
        return "unknown", 0
    batches = evidence["batches"]
    reason = evidence.get("reason")
    if reason is not None:
        if not isinstance(reason, str) or not reason or evidence.get("status") != "unknown":
            raise ValueError(f"update {row.get('update_index')} gradient unknown 증거가 잘못됐다")
        return "unknown", len(batches)
    if any(not _valid_summary(summary) for summary in batches):
        raise ValueError(f"update {row.get('update_index')} gradient batch summary 형식이 잘못됐다")
    observed = row.get("observed", {})
    minibatches = observed.get("minibatches")
    optimizer_steps = observed.get("optimizer_steps")
    if (not isinstance(minibatches, int) or isinstance(minibatches, bool)
          or not isinstance(optimizer_steps, int) or isinstance(optimizer_steps, bool)
          or minibatches <= 0 or optimizer_steps != minibatches or len(batches) != minibatches):
        computed = "unknown"
    elif any(not summary["finite"] for summary in batches):
        computed = "unknown"
    elif any(summary["mismatch_count"] > 0 for summary in batches):
        computed = "abnormal"
    elif not any(summary["nonzero_expected_count"] > 0 for summary in batches):
        computed = "unknown"
    else:
        computed = "normal"
    declared = evidence.get("status")
    if declared != computed:
        raise ValueError(
            f"update {row.get('update_index')} gradient status가 batch 증거와 다르다: {declared!r} != {computed!r}"
        )
    return computed, len(batches)


def _gradient_evaluator(ctx: O.InspectionContext) -> dict[str, Any]:
    expected_records = ctx.manifest["runtime_updates_to_observe"]
    indices = [row.get("update_index") for row in ctx.records]
    computed = [
        _recompute_evidence(row, row.get("observed", {}).get(PRIMITIVE))
        for row in ctx.records
    ]
    incomplete = not ctx.records or indices != list(range(expected_records))
    unknown = incomplete or any(row.get("error") for row in ctx.records) or any(
        status == "unknown" for status, _ in computed
    )
    abnormal = [row["update_index"] for row, (status, _) in zip(ctx.records, computed)
                if status == "abnormal"]
    status = "unknown" if unknown else ("abnormal" if abnormal else "normal")
    return {
        "status": status,
        "first_mismatch_update": min(abnormal) if abnormal and not unknown else None,
        "batches": sum(count for _, count in computed),
    }


def registry() -> O.PrimitiveRegistry:
    result = O.PrimitiveRegistry(O.DEFAULT_REGISTRY.specs())
    result.register(O.PrimitiveSpec(PRIMITIVE, _gradient_evaluator, decisive=True))
    return result


def inspect_records(records: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    return O.assemble_inspection(list(records), validate_manifest(manifest), registry())
