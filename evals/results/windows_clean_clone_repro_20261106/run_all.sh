#!/usr/bin/env bash
# 사용법: clone 루트에서 `bash run_all.sh <tag>`. 로그는 ${REPRO_LOGS:-../repro-logs}/<tag>/에 남는다.
set -u
L="${REPRO_LOGS:-../repro-logs}/$1"; mkdir -p "$L"
export PYTHONUTF8=1
t(){ n=$1; shift; S=$(date +%s); "$@" > "$L/$n.log" 2>&1; rc=$?; echo "$n exit=$rc seconds=$(( $(date +%s)-S ))" | tee -a "$L/timings.txt"; }
echo "commit=$(git rev-parse HEAD)" > "$L/timings.txt"
t replay uv run --no-sync python evals/replay.py evals/results/blind_v1 evals/results/heldout_blind evals/results/heldout_changes --traces evals/results/traces
t prove uv run --no-sync python evals/prove_policies.py
git status --short > "$L/git_status_after_prove.txt"; git checkout -q -- policies/candidates
t correct_v2 uv run --no-sync python evals/correct_v2.py replay
t replay_page uv run --no-sync python evals/build_replay_page.py --check
t collect uv run --no-sync pytest --collect-only -q tests -p no:cacheprovider
t snapshot_tests uv run --no-sync pytest -q -p no:cacheprovider tests/test_replay_page.py::test_written_data_matches_current_approved_results tests/test_additional_checks_replay_v2.py::test_v2_snapshot_and_html_entrypoint_are_current tests/test_ppo_gradient_replay_v1.py::test_snapshot_html_and_fixed_viewport_are_current
[ -f tests/test_prove_policies.py ] && t prove_tests uv run --no-sync pytest -q -p no:cacheprovider tests/test_prove_policies.py || echo "prove_tests skipped=no_file" | tee -a "$L/timings.txt"
