"""묶음 계측의 기존 증거 계약·예외 복원·CPU 읽기 경계를 검사한다."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "tests")]
import ppo_gradient_invariant_v1 as OLD
import ppo_gradient_invariant_v2 as NEW


def test_frozen_contract_and_evaluators_are_reused():
    for name in ("GRADIENT_SCHEMA", "PRIMITIVE", "ATOL", "RTOL", "SCALE", "CLIP_PARAM", "MAX_UPDATES"):
        assert getattr(NEW, name) == getattr(OLD, name)
    assert NEW.inspect_records is OLD.inspect_records
    assert NEW.validate_manifest is OLD.validate_manifest


def test_real_torch_cpu_evidence_equivalence_and_cleanup():
    python = Path.home() / "IsaacLab/_isaac_sim/kit/python/python.exe"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run([str(python), "-B", str(Path(__file__)), "--helper"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    value = json.loads(proc.stdout.splitlines()[-1])
    assert value["summary_cases"] >= 30
    assert value["exact"] and value["one_cpu_read"] and value["cleanup"]


def _helper():
    import torch
    from unittest.mock import patch
    import test_ppo_gradient_invariant_v1 as legacy
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    snapshots = []
    for count in (1, 4, 31, 64, 24576):
        for dtype in (torch.float32, torch.float64, torch.float16):
            new = torch.randn(count, dtype=dtype) * 0.1
            old = torch.zeros_like(new)
            adv = torch.randn_like(new)
            gradient = -adv * torch.exp(new - old) / count
            snapshots.append((new, gradient, adv, old))
            snapshots.append((new, gradient * -1, adv, old))
    for special in (float("nan"), float("inf"), float("-inf")):
        for position in range(4):
            items = [torch.zeros(4) for _ in range(4)]
            items[position][0] = special
            snapshots.append(tuple(items))
    snapshots += [(torch.zeros(0),) * 4, (torch.zeros(3), torch.zeros(2), torch.zeros(3), torch.zeros(3)),
        (torch.zeros(4),) * 4]
    observer = object.__new__(OLD.GradientObserver)
    expected = [observer._summarize_batch(torch, *row) for row in snapshots]
    pack = NEW._Pack(torch)
    # Tensor -> Python 읽기는 capture·계산 중 없어야 한다.
    with patch.object(torch.Tensor, "item", side_effect=AssertionError("early item")), \
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("early cpu")), \
            patch.object(torch.Tensor, "tolist", side_effect=AssertionError("early tolist")):
        rows = NEW._batch_summaries(torch, snapshots, pack)
        step = torch.tensor(7.0)
        ref = pack.number(step)
        step.add_(5)
    calls = []
    real_cpu = torch.Tensor.cpu
    def cpu(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return real_cpu(tensor, *args, **kwargs)
    with patch.object(torch.Tensor, "cpu", cpu):
        pack.materialize()
    actual = NEW._decode_summaries(rows, pack)
    assert actual == expected, [(i, a, b) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
    assert pack.scalar(ref) == 7.0
    assert len(calls) == 1 and not pack.tensors
    for gradients in ([None, None, None], [1.0, 2.0, 3.0],
                      [1.0, float("nan"), 3.0], [float("inf"), 2.0, None]):
        named = [(str(i), torch.nn.Parameter(torch.zeros(3))) for i in range(3)]
        for (_, parameter), value in zip(named, gradients):
            parameter.grad = None if value is None else torch.full_like(parameter, value)
        expected_norm, expected_present = NEW.O._gradient_snapshot(named)
        norms_pack = NEW._Pack(torch)
        refs = [(name, norms_pack.add(parameter.grad.detach().norm(2)))
                for name, parameter in named if parameter.grad is not None]
        norms_pack.materialize()
        norms, presence = NEW._decode_norms([refs], norms_pack, [name for name, _ in named])
        assert norms == [expected_norm]
        assert {name for name, count in presence.items() if count} == expected_present
    # v1 fixture가 검사하는 실패·재귀·다중 hook·비유한 값·상태 복원을 새 observer에도 적용한다.
    legacy.G = NEW
    legacy._torch_helper()
    print(json.dumps({"summary_cases": len(snapshots), "exact": True, "one_cpu_read": True, "cleanup": True}))


if __name__ == "__main__":
    _helper()
