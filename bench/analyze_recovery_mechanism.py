"""v4 저장 체크포인트·TensorBoard의 CPU 기술 통계. 시뮬레이터를 시작하지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "bench/protocols/recovery_mechanism_cpu_v1.json"
ARMS = ("clean_resume_reference", "resume_corrected", "resume_corrected_std_reset", "restart_corrected")
TAGS = ("Loss/value_function", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate",
        "Policy/mean_noise_std", "Train/mean_episode_length", "Train/mean_reward",
        "Episode_Reward/track_lin_vel_xy_exp", "Metrics/base_velocity/error_vel_xy")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective_rank(values):
    """특이값 분포의 exp(entropy); 입력 값이 모두 0이면 0."""
    total = sum(values)
    if not math.isfinite(total) or any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError("invalid singular values")
    return math.exp(-sum((v / total) * math.log(v / total) for v in values if v)) if total else 0.0


def summary(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("empty or nonfinite series")
    n = min(60, len(values))
    return {"n": len(values), "first_60_mean": sum(values[:n]) / n,
            "last_60_mean": sum(values[-n:]) / n, "min": min(values), "max": max(values)}


def checkpoint_metrics(path):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = payload["model_state_dict"], payload["optimizer_state_dict"]
    layers = {}
    for key, value in model.items():
        tensor = value.detach().double()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite tensor: {key}")
        row = {"shape": list(tensor.shape), "l2": tensor.norm().item(),
               "rms": tensor.square().mean().sqrt().item(), "min": tensor.min().item(), "max": tensor.max().item()}
        if tensor.ndim == 2:
            sv = torch.linalg.svdvals(tensor).tolist()
            row.update(weight_matrix_effective_rank=effective_rank(sv), singular_max=max(sv), singular_min=min(sv))
        layers[key] = row
    noise = model.get("std")
    if noise is None and "log_std" in model:
        noise = model["log_std"].exp()
    steps = [float(v["step"]) for v in optimizer["state"].values() if "step" in v]
    return payload, {"path": str(path), "sha256": sha(path), "iteration": payload["iter"],
                     "layers": layers, "policy_std": noise.tolist() if noise is not None else None,
                     "adam_steps": sorted(set(steps)), "adam_parameter_states": len(steps),
                     "optimizer_lrs": [g["lr"] for g in optimizer["param_groups"]],
                     "checkpoint_keys": sorted(payload.keys())}


def changes(before, after):
    output = {}
    for group in ("actor", "critic", "std"):
        keys = [k for k in before if k.startswith(group)]
        numerator = sum((after[k].double() - before[k].double()).square().sum().item() for k in keys)
        denominator = sum(before[k].double().square().sum().item() for k in keys)
        output[group] = math.sqrt(numerator / denominator) if denominator else None
    return output


def telemetry(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(logdir), size_guidance={"scalars": 0}).Reload()
    tags = acc.Tags()["scalars"]
    metrics, raw = {}, {}
    for key in TAGS:
        if key not in tags:
            metrics[key] = {"status": "unrecorded"}
            continue
        events = acc.Scalars(key)
        steps = [e.step for e in events]
        if len(set(steps)) != len(steps) or steps != sorted(steps):
            raise ValueError(f"duplicate or unordered steps: {logdir}/{key}")
        values = [e.value for e in events]
        metrics[key] = dict(summary(values), first_step=steps[0], last_step=steps[-1])
        raw[key] = [{"step": e.step, "value": e.value} for e in events]
    return {"metrics": metrics, "series": raw,
            "kl_clip_tags": [t for t in tags if "kl" in t.lower() or "clip" in t.lower()],
            "event_files": [{"path": str(p), "sha256": sha(p)} for p in sorted(logdir.glob("events.out.tfevents.*"))]}


def bindings(seed):
    rows = []
    fixed = read(ROOT / "bench/protocols/training_correction_v4.json")
    for role, run in (("original_none", f"baseline_p0c_s{seed}"), ("original_fault", f"h02_s{seed}")):
        meta = ROOT / f"bench/private/{run}.meta.json"
        metadata = read(meta)
        fault = "NONE" if role == "original_none" else "F1"
        if (metadata.get("seed") != seed or metadata.get("fault") != fault
                or metadata.get("train_exit_code") != 0 or metadata.get("max_iterations") != 300):
            raise ValueError("original run metadata differs")
        path = Path(metadata["run_dir"]) / "model_299.pt"
        if sha(path) != fixed["resume_inputs"]["seeds"][str(seed)]["runs"][fault]["hashes"]["checkpoint"]:
            raise ValueError("original checkpoint differs from v4 commitment")
        rows.append((role, path, None, {"path": str(meta), "sha256": sha(meta)}))
    for arm in ARMS:
        folder = ROOT / f"bench/private/training_correction_v4/reward_v4_s{seed}_20261013/{arm}"
        receipt = folder / "train/receipt.json"
        data = read(receipt)
        if not data["completed"] or data["seed"] != seed or data["arm"] != arm:
            raise ValueError("receipt mismatch")
        path = Path(data["checkpoint"]["path"])
        if sha(path) != data["checkpoint"]["sha256"]:
            raise ValueError("checkpoint SHA differs from receipt")
        rows.append((arm, path, read(folder / "parent.json")["train_parent_wall_s"],
                     {"path": str(receipt), "sha256": sha(receipt)}))
    return rows


def input_files():
    files = {Path(__file__).resolve(), PROTOCOL, ROOT / "bench/protocols/training_correction_v4.json"}
    for seed in (2026, 2028):
        for _, path, _, source in bindings(seed):
            files.update((path, Path(source["path"])))
            files.update(path.parent.glob("events.out.tfevents.*"))
            files.update((path.parent / "params").glob("*.yaml"))
        folder = ROOT / f"bench/private/training_correction_v4/reward_v4_s{seed}_20261013"
        for arm in ARMS:
            files.update((folder / arm / "parent.json", folder / arm / "eval/report.json"))
        files.add(ROOT / f"evals/results/reward_v4_s{seed}_20261013/result.json")
    return sorted(files, key=str)


def prepare_commitment(path):
    if path.exists():
        raise FileExistsError(path)
    document = {"schema": "recovery_cpu_inputs_v2", "scope": "verification_after_first_measurement",
                "note": "v1 측정 이후 입력 결합을 보강한 재검증이다. 처음 보는 자료의 사전등록으로 소급하지 않는다.",
                "protocol_sha256": sha(PROTOCOL),
                "inputs": {str(p.resolve()): sha(p) for p in input_files()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"input_files": len(document["inputs"]), "commitment": str(path)}))


def verify_commitment(path):
    document = read(path)
    if document.get("schema") != "recovery_cpu_inputs_v2" or document.get("protocol_sha256") != sha(PROTOCOL):
        raise ValueError("input commitment schema/protocol differs")
    current = {str(p.resolve()): sha(p) for p in input_files()}
    if document.get("inputs") != current:
        raise ValueError("bound input inventory or file contents changed")
    protocol = read(PROTOCOL)
    if protocol.get("schema") != "recovery_mechanism_cpu_v1" or protocol.get("seeds") != [2026, 2028]:
        raise ValueError("unsupported analysis protocol")
    return document


def run(output, commitment_path):
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only audit requires CUDA uninitialized")
    commitment = verify_commitment(commitment_path)
    protocol = read(PROTOCOL)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "recovery_mechanism_cpu_report_v1", "scope": protocol["scope"],
              "protocol_sha256": sha(PROTOCOL), "analysis_sha256": sha(__file__), "seeds": {},
              "input_commitment_sha256": sha(commitment_path), "verification_scope": commitment["scope"],
              "unmeasured": protocol["required_observation_metrics"],
              "interpretation": "가중치 행렬 rank는 특징 rank가 아니며 저장 체크포인트만으로 메커니즘을 확정하지 않는다."}
    for seed in protocol["seeds"]:
        models, rows = {}, {}
        for arm, path, wall, source in bindings(seed):
            payload, row = checkpoint_metrics(path)
            models[arm] = payload["model_state_dict"]
            row["source_binding"] = source
            row["train_parent_wall_s"] = wall
            row["telemetry"] = telemetry(path.parent)
            row["available_auxiliary_files"] = [p.name for p in path.parent.iterdir() if p.is_file() and not p.name.startswith(("model_", "events."))]
            rows[arm] = row
        for arm in ARMS:
            source = "original_none" if arm in ("clean_resume_reference", "restart_corrected") else "original_fault"
            rows[arm]["relative_parameter_change_from_" + source] = changes(models[source], models[arm])
        report["seeds"][str(seed)] = rows
    report["cuda_initialized"] = torch.cuda.is_initialized()
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "report.json"), "checkpoints": 12, "cuda_initialized": report["cuda_initialized"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_commitment(args.commitment)
    elif args.output:
        run(args.output, args.commitment)
    else:
        parser.error("measurement requires --output")
