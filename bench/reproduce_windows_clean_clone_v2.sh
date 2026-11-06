#!/usr/bin/env bash
# CPU clean-clone diagnostics; never restore or delete policy edits.
set -uo pipefail
if [[ $# != 1 || ! $1 =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "Usage: bash bench/reproduce_windows_clean_clone_v2.sh <new-tag>" >&2
  exit 2
fi
root=$(git rev-parse --show-toplevel) || exit 2
cd "$root" || exit 2
status=$(git status --porcelain --untracked-files=all -- policies/candidates) || exit 2
if [[ -n $status ]]; then
  echo "Refusing: policies/candidates has existing changes." >&2
  exit 2
fi
L="${REPRO_LOGS:-../repro-logs}/$1"
mkdir -p -- "$(dirname -- "$L")" || exit 2
mkdir -- "$L" || exit 2
export PYTHONUTF8=1
failed=0
t() {
  local name=$1 start rc
  shift
  start=$SECONDS
  "$@" > "$L/$name.log" 2>&1
  rc=$?
  printf '%s exit=%s seconds=%s\n' "$name" "$rc" "$((SECONDS-start))" | tee -a "$L/timings.txt"
  if (( rc != 0 )); then failed=1; fi
}
git rev-parse HEAD > "$L/timings.txt" || exit 2
t replay uv run --no-sync python evals/replay.py evals/results/blind_v1 evals/results/heldout_blind evals/results/heldout_changes --traces evals/results/traces
t prove uv run --no-sync python evals/prove_policies.py
t candidate_status git status --short --untracked-files=all -- policies/candidates
t correct_v2 uv run --no-sync python evals/correct_v2.py replay
t replay_page uv run --no-sync python evals/build_replay_page.py --check
t collect uv run --no-sync pytest --collect-only -q tests -p no:cacheprovider
t snapshot_tests uv run --no-sync pytest -q -p no:cacheprovider tests/test_replay_page.py::test_written_data_matches_current_approved_results tests/test_additional_checks_replay_v2.py::test_v2_snapshot_and_html_entrypoint_are_current tests/test_ppo_gradient_replay_v1.py::test_snapshot_html_and_fixed_viewport_are_current
t prove_tests uv run --no-sync pytest -q -p no:cacheprovider tests/test_prove_policies.py
exit "$failed"
