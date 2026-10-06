#!/bin/bash
# FINAL TEST RUN: executed exactly once, after the configuration is locked (see results/optim_log.jsonl). M1 (ASR, denoise OFF = ADR-001 verdict),
# M2 (NMT slot preservation, glossary-constrained greedy int8 bonus 5), M3 (safety check on the M2 test outputs). Test IDs: configs/splits + templates.
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY=.venv/Scripts/python.exe
[ -e results/FINAL_TEST_DONE ] && { echo "test already run once; refusing"; exit 1; }
for n in demand babble alarm; do
  W=""; [ $n = demand ] && W="--with-clean"
  $PY tools/run_adr001_v2.py --split test --noise $n $W --arms off --out results/final_test_m1_$n.json > logs/final_test_m1_$n.log 2>&1 &
done
$PY tools/eval_nmt_config.py --split test --groups medication unit number intensity --bonus 5 --tag final_test --note "FINAL TEST, run once" > logs/final_test_m2.log 2>&1
$PY tools/eval_safety.py --hyp results/nmt_cfg_final_test_test.jsonl --split test --out results/safety_eval_test_final.json > logs/final_test_m3.log 2>&1
wait
date > results/FINAL_TEST_DONE
