#!/bin/bash
# sequential NMT safety-set runs (memory-limited machine)
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe
while pgrep -f "run_nmt_safety.py --variant fp32" >/dev/null; do sleep 10; done
$PY tools/run_nmt_safety.py --variant int8enc_fp32dec --decode greedy > logs/nmt_safety_hybrid_greedy.log 2>&1
$PY tools/run_nmt_safety.py --variant int8 --decode beam4 > logs/nmt_safety_int8_beam4.log 2>&1
