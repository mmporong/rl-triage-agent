"""v1 수치·증거 계약을 유지하며 관측 텐서를 update 끝에 모아 읽는다."""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import math
from types import MethodType
from typing import Any, Callable

import optimizer_observability as O
import ppo_gradient_invariant_v1 as V1
from ppo_gradient_invariant_v1 import (ATOL, CLIP_PARAM, GRADIENT_SCHEMA, MAX_UPDATES,
    PRIMITIVE, RTOL, SCALE, inspect_records, registry, validate_manifest)


@dataclass(frozen=True)
class _Ref:
    index: int


class _Pack:
    """device별 한 번 CPU로 옮긴다. 정수 통계는 float64 정확 표현 범위다."""

    def __init__(self, torch):
        self.torch = torch
        self.tensors: list[Any] = []
        self.values: dict[int, list[float]] = {}

    def add(self, tensor) -> _Ref:
        ref = _Ref(len(self.tensors))
        self.tensors.append(tensor.detach().reshape(-1))
        return ref

    def number(self, value):
        if isinstance(value, self.torch.Tensor):
            return self.add(value.detach().clone()) if value.numel() == 1 else None
        return O._number(value)

    def materialize(self):
        devices: dict[Any, dict[Any, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, tensor in enumerate(self.tensors):
            devices[tensor.device][tensor.dtype].append(index)
        for groups in devices.values():
            chunks, ordered = [], []
            for indices in groups.values():
                chunks.append(self.torch.cat([self.tensors[i] for i in indices]).to(self.torch.float64))
                ordered.extend(indices)
            values = self.torch.cat(chunks).cpu().tolist()
            offset = 0
            for index in ordered:
                size = self.tensors[index].numel()
                self.values[index] = values[offset:offset + size]
                offset += size
        self.tensors.clear()

    def scalar(self, value):
        return O._number(self.values[value.index][0]) if isinstance(value, _Ref) else value


def _batch_summaries(torch, snapshots, pack):
    """동일 shape·dtype·device의 batch를 행으로 묶고 v1 공식을 적용한다."""
    groups = defaultdict(list)
    rows: list[Any] = [None] * len(snapshots)
    for index, snapshot in enumerate(snapshots):
        tensors = tuple(torch.squeeze(t.detach()) for t in snapshot)
        n = tensors[0].numel()
        typed = all(t.dtype == torch.float32 for t in tensors)
        if not n or any(t.shape != tensors[0].shape for t in tensors):
            finite = torch.stack([torch.isfinite(t).all() for t in tensors]).all()
            rows[index] = (n, typed, None, pack.add(finite))
            continue
        key = tuple((t.shape, t.dtype, t.device) for t in tensors)
        groups[key].append((index, n, typed, tensors))
    for members in groups.values():
        new, actual, advantage, old = [torch.stack([row[3][j].reshape(-1) for row in members]) for j in range(4)]
        n = members[0][1]
        ratio = torch.exp(new - old)
        active = ((advantage >= 0) & (ratio <= 1.0 + CLIP_PARAM)) | ((advantage < 0) & (ratio >= 1.0 - CLIP_PARAM))
        expected = torch.where(active, -advantage * ratio, torch.zeros_like(ratio))
        scaled_actual = actual * n
        error = torch.abs(scaled_actual - expected)
        finite = torch.stack([torch.isfinite(t).all(dim=1) for t in
            (new, actual, advantage, old, ratio, expected, error)]).all(dim=0)
        close = torch.isclose(scaled_actual, expected, atol=ATOL, rtol=RTOL)
        columns = [finite, error.amax(dim=1), (~close).sum(dim=1), (expected != 0).sum(dim=1),
            torch.isfinite(ratio).all(dim=1), ratio.amin(dim=1), ratio.amax(dim=1),
            (~active).to(torch.float32).mean(dim=1)]
        refs = [pack.add(column) for column in columns]
        for offset, (index, count, typed, _) in enumerate(members):
            rows[index] = (count, typed, offset, refs)
    return rows


def _decode_summaries(rows, pack):
    summaries = []
    for n, typed, offset, refs in rows:
        if offset is None:
            summaries.append({"max_abs_error": None, "mismatch_count": None,
                "nonzero_expected_count": None, "finite": typed and bool(pack.scalar(refs)), "n": n})
            continue
        finite, error, mismatch, nonzero, ratio_finite, low, high, clipped = [pack.values[r.index][offset] for r in refs]
        finite = typed and bool(finite)
        summaries.append({"max_abs_error": float(error) if finite else None,
            "mismatch_count": int(mismatch) if finite else n, "nonzero_expected_count": int(nonzero),
            "finite": finite, "n": n, "ratio_min": float(low) if ratio_finite else None,
            "ratio_max": float(high) if ratio_finite else None, "clipped_gradient_fraction": float(clipped)})
    return summaries


def _decode_norms(snapshots, pack, names):
    """v1처럼 첫 비유한 norm에서 멈추며 그 지점까지 presence를 기록한다."""
    values, presence = [], dict.fromkeys(names, 0)
    for row in snapshots:
        squared, present, valid = 0.0, False, True
        for name, ref in row:
            present = True
            presence[name] += 1
            norm = pack.scalar(ref)
            if norm is None:
                valid = False
                break
            squared += norm * norm
        values.append(math.sqrt(squared) if valid and present else None)
    return values, presence


class GradientObserver(V1.GradientObserver):
    """backward hook는 snapshot만 보관하고 증거 계산·직렬화는 update 안에서 끝낸다."""

    def _register_gradient_hook(self, value, pending, state):
        import torch
        if not isinstance(value, torch.Tensor) or not value.requires_grad:
            state["reason"] = "log_prob_tensor_not_differentiable"
            return
        new = value.detach().clone()
        advantage = pending["advantages"].detach().clone()
        old = pending["old_log_prob"].detach().clone()

        def inspect_gradient(gradient):
            state["callbacks"] += 1
            state["snapshots"].append((new, gradient.detach().clone(), advantage, old))
            return None

        state["handles"].append(value.register_hook(inspect_gradient))

    def _observe_update(self, original: Callable[[], Any]) -> Any:
        import torch
        index = self.update_index
        self.update_index += 1
        if index >= self.manifest["runtime_updates_to_observe"]:
            return original()
        ppo, optimizer = self.ppo, self.ppo.optimizer
        storage, policy = ppo.storage, ppo.policy
        state = {"reason": V1._unsupported_reason(ppo), "yielded": 0, "registered": 0,
            "callbacks": 0, "batches": [], "pending": None, "handles": [], "snapshots": []}
        self._gradient_state = state
        pack = _Pack(torch)
        named = O._named_parameters(policy)
        before_refs = {name: pack.number(optimizer.state.get(p, {}).get("step")) for name, p in named}
        generator_name = "recurrent_mini_batch_generator" if policy.is_recurrent else "mini_batch_generator"
        generator, step = getattr(storage, generator_name), optimizer.step
        counts = {"minibatches": 0, "optimizer_steps": 0}
        norm_snapshots, learning_rate_refs = [], []
        error = None
        supported = state["reason"] is None

        def observed_generator(*args, **kwargs):
            for batch in generator(*args, **kwargs):
                counts["minibatches"] += 1
                if supported:
                    state["yielded"] += 1
                    if not isinstance(batch, (tuple, list)) or len(batch) <= 6:
                        state["reason"] = "batch_layout_missing"
                        state["pending"] = None
                    else:
                        state["pending"] = {"advantages": batch[4], "old_log_prob": batch[6], "hooked": False}
                yield batch
                state["pending"] = None

        def observed_step(*args, **kwargs):
            row = []
            for name, parameter in named:
                grad = getattr(parameter, "grad", None)
                if grad is None:
                    continue
                try:
                    norm = pack.add(grad.detach().norm(2))
                except (AttributeError, RuntimeError):
                    norm = None
                row.append((name, norm))
            norm_snapshots.append(row)
            learning_rate_refs.append([pack.number(group.get("lr")) for group in optimizer.param_groups])
            counts["optimizer_steps"] += 1
            return step(*args, **kwargs)

        try:
            with ExitStack() as stack:
                stack.enter_context(O._temporary_instance_attr(storage, generator_name, observed_generator))
                stack.enter_context(O._temporary_instance_attr(optimizer, "step", observed_step))
                if supported:
                    get_log_prob = policy.get_actions_log_prob

                    def observed_get_log_prob(this, *args, **kwargs):
                        value = get_log_prob(*args, **kwargs)
                        pending = state["pending"]
                        if pending is None:
                            return value
                        if pending["hooked"]:
                            state["reason"] = "multiple_log_prob_calls_per_batch"
                            return value
                        pending["hooked"] = True
                        state["registered"] += 1
                        self._register_gradient_hook(value, pending, state)
                        return value

                    stack.enter_context(O._temporary_instance_attr(policy, "get_actions_log_prob", MethodType(observed_get_log_prob, policy)))
                return original()
        except BaseException as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            raise
        finally:
            try:
                with torch.no_grad():
                    after_refs = {name: pack.number(optimizer.state.get(p, {}).get("step")) for name, p in named}
                    rows = _batch_summaries(torch, state["snapshots"], pack)
                    pack.materialize()
                before = {name: pack.scalar(ref) for name, ref in before_refs.items()}
                after = {name: pack.scalar(ref) for name, ref in after_refs.items()}
                delta = {}
                for name, _ in named:
                    left, right = before[name], after[name]
                    delta[name] = None if left is None or right is None else right - left
                norms, presence = _decode_norms(norm_snapshots, pack, [name for name, _ in named])
                state["batches"] = _decode_summaries(rows, pack)
                self.sink({"schema": O.SCHEMA, "manifest_sha256": O.manifest_fingerprint(self.manifest),
                    "binding": O.manifest_binding(self.manifest), "update_index": index,
                    "expected": {"optimizer_steps": self.manifest["expected_optimizer_steps_per_update"],
                        "minibatches": self.manifest["expected_minibatches_per_update"]},
                    "runtime_configuration": {"learning_epochs": getattr(ppo, "num_learning_epochs", None),
                        "mini_batches": getattr(ppo, "num_mini_batches", None)},
                    "observed": {**counts, "adam_step_before": before, "adam_step_after": after,
                        "adam_step_delta": delta,
                        "gradient_present_steps": presence, "pre_step_gradient_norm": norms,
                        "learning_rates": [[pack.scalar(ref) for ref in row] for row in learning_rate_refs],
                        "approx_kl": None, "clip_fraction": None},
                    "unavailable": {"approx_kl": "RSL-RL 2.3.3 PPO.update의 지역 변수이며 공개 hook 없음",
                        "clip_fraction": "RSL-RL 2.3.3이 계산 결과를 기록하지 않음"}, "error": error})
            finally:
                for handle in state["handles"]:
                    handle.remove()
                state["handles"].clear()
                state["snapshots"].clear()
                state["pending"] = None
                self._gradient_state = None
