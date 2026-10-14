#!/usr/bin/env bash
# Frozen test protocol. Fault tests use a temporary root, never the held-out data.
set -Eeuo pipefail
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
  echo "Bash 5.1+ is required for PID-aware wait -n" >&2; exit 2
fi
ROOT="${ONEVOICE_PROJECT_ROOT:-$(cd -- "$(dirname -- "$0")/.." && pwd)}"
cd -- "$ROOT"
export PYTHONUTF8=1
PY="${PYTHON:-.venv/clean/Scripts/python.exe}"
[[ -e results/FINAL_TEST_DONE ]] && { echo "test already completed; refusing" >&2; exit 1; }
mkdir -p results logs
for output in results/nmt_cfg_final_test_test.jsonl results/safety_eval_test_final.json; do
  [[ ! -e "$output" ]] || { echo "existing final output requires review/archive before a fresh run" >&2; exit 1; }
done
LOCK=results/FINAL_TEST_RUNNING
mkdir "$LOCK" || { echo "another final test is running" >&2; exit 1; }
declare -a active=()
declare -A task_by_pid=()
cleanup() {
  local rc=$?
  trap - EXIT
  for pid in "${active[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${active[@]}"; do wait "$pid" 2>/dev/null || true; done
  rmdir "$LOCK" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
launch() {
  local name=$1; shift
  "$PY" "$@" > "logs/final_test_${name}.log" 2>&1 &
  local pid=$!
  active+=("$pid")
  task_by_pid[$pid]="$name"
}
: > results/final_test_processes.jsonl
for noise in demand babble alarm; do
  flags=()
  [[ "$noise" == demand ]] && flags+=(--with-clean)
  launch "m1_${noise}" tools/run_adr001_v2.py --split test --noise "$noise" "${flags[@]}" --arms off --out "results/final_test_m1_${noise}.json"
done
launch m2 tools/eval_nmt_config.py --split test --groups medication unit number intensity --bonus 5 --tag final_test --note "FINAL TEST, run once"
while (( ${#active[@]} )); do
  rc=0
  completed=""
  wait -n -p completed "${active[@]}" || rc=$?
  [[ -n "$completed" ]] || { echo "could not determine child status" >&2; exit 1; }
  job=${task_by_pid[$completed]}
  printf '{"task":"%s","pid":%s,"exit_code":%s}\n' "$job" "$completed" "$rc" >> results/final_test_processes.jsonl
  next=()
  for pid in "${active[@]}"; do [[ "$pid" == "$completed" ]] || next+=("$pid"); done
  active=("${next[@]}")
  if (( rc != 0 )); then
    echo "subprocess failed: task=$job pid=$completed exit=$rc" >&2; exit "$rc"
  fi
  if [[ "$job" == m2 ]]; then
    launch m3 tools/eval_safety.py --hyp results/nmt_cfg_final_test_test.jsonl --split test --out results/safety_eval_test_final.json
  fi
done
# Exit 0 is insufficient: reject missing/truncated/mismatched outputs as well.
"$PY" tools/check_final_outputs.py --root "$ROOT" --marker results/FINAL_TEST_DONE
