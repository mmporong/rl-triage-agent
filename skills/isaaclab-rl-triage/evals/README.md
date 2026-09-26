# Evals

The skill is evaluated with the fault-injection benchmark in the repository root:

- cases: `bench/cases.json` (10 cases; each = 1 harmful + 2 benign hydra overrides, shuffled with a secret seed)
- telemetry: `bench/runs/*.telemetry.json` (Isaac-Velocity-Flat-Unitree-Go2-v0, 1024 envs, 100 iterations, seeds 42 and 7)
- runner: `uv run python evals/run_eval.py --seed 42 --mode both`
- answer key: published after evaluation in `bench/answer_key.json`
