"""Fault-test real Bash PID supervision in a temp root, never held-out runs."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from tonebridge.config import ROOT
from tonebridge.evalkit.metrics import ErrorCounts

sys.path.insert(0, str(ROOT / "tools"))
from benchmark_integrity import atomic_json, fingerprint, resume_checkpoint, validate_cell


def test_resume_checks_fingerprint_and_cell_coverage(tmp_path):
    path = tmp_path / "checkpoint.json"
    spec, ids, expected = {"seed": 7, "decoder": "int8", "bonus": 5}, ["a"], ["cell"]
    fresh = resume_checkpoint(path, spec, expected, ids)
    atomic_json(path, fresh)
    assert not resume_checkpoint(path, spec, expected, ids)["complete"]
    fresh["cells"] = {"cell": {"hyp": ["text"], "per_utt": [[1, 0, 4, 0]], "words": 1, "WER": 0., "CER": 0.}}
    atomic_json(path, fresh)
    assert resume_checkpoint(path, spec, expected, ids)["complete"]
    with pytest.raises(ValueError, match="mismatch"):
        resume_checkpoint(path, {**spec, "bonus": 6}, expected, ids)
    fresh["cells"]["cell"]["hyp"] = []
    atomic_json(path, fresh)
    with pytest.raises(ValueError, match="incomplete"):
        resume_checkpoint(path, spec, expected, ids)


def test_legacy_checkpoint_cannot_silently_resume(tmp_path):
    path = tmp_path / "legacy.json"
    atomic_json(path, {"cells": {"off": {}}, "ids": ["a"]})
    with pytest.raises(ValueError, match="mismatch"):
        resume_checkpoint(path, {"seed": 7}, ["off"], ["a"])


def test_empty_asr_hypothesis_counts_pure_deletions():
    counts = ErrorCounts()
    counts.add("xin chào", "")
    assert counts.word_errors == counts.words == 2 and counts.wer == 1.
    assert counts.char_errors == counts.chars and counts.cer == 1.


@pytest.mark.parametrize("change", ["rate", "counts"])
def test_invalid_metrics_cannot_be_resumed_as_complete(change):
    cell = {"hyp": ["text"], "per_utt": [[1, 0, 4, 0]], "words": 1, "WER": 0., "CER": 0.}
    if change == "rate": cell["WER"] = 1.
    else: cell["per_utt"][0][1] = -1
    with pytest.raises(ValueError): validate_cell(cell, 1)


@pytest.fixture
def batch_root(tmp_path):
    for directory in ("tools", "configs/splits", "configs/safety", "results"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/pipeline.json").write_text("{}")
    (tmp_path / "configs/splits/fleurs_vi_dev_test.json").write_text(json.dumps({"test": ["audio1", "audio2"]}))
    (tmp_path / "configs/safety/safety_set_v1.jsonl").write_text(json.dumps({"id": "s1", "split": "test"}) + "\n")
    for name in ("final_test.sh", "benchmark_integrity.py", "check_final_outputs.py"):
        shutil.copyfile(ROOT / "tools" / name, tmp_path / "tools" / name)
    # Actual OS subprocesses replace inference only in this fixture. They test
    # exit codes, dependency ordering, cleanup and output validation, not metrics.
    job = '''
import hashlib,json,os,sys,time
from pathlib import Path
from benchmark_integrity import atomic_json,fingerprint
root=Path.cwd();args=sys.argv
name=Path(__file__).name
if name=='run_adr001_v2.py':
    noise=args[args.index('--noise')+1]; task='m1_'+noise
else: task='m2' if name=='eval_nmt_config.py' else 'm3'
(root/('pid_'+task)).write_text(str(os.getpid()))
if os.environ.get('FAIL_TASK')==task: sys.exit(7)
time.sleep(5 if os.environ.get('SLOW_TASK')==task else .15)
if task.startswith('m1_'):
    spec={'split':'test','noise':noise,'arms':['off'],'config_file_sha256':hashlib.sha256((root/'configs/pipeline.json').read_bytes()).hexdigest()}
    cell={'WER':0.,'CER':0.,'words':2,'hyp':['ok','ok'],'per_utt':[[1,0,1,0],[1,0,1,0]]}
    data={'schema_version':'adr001-v3','complete':True,'ids':['audio1','audio2'],'run_spec':spec,'fingerprint':fingerprint(spec),'cells':{noise+'|'+snr+'|off':cell for snr in (['clean'] if noise=='demand' else [])+['10','5','0','-5']}}
    if os.environ.get('CORRUPT_TASK')==task: data={}
    atomic_json(root/args[args.index('--out')+1],data)
elif task=='m2':
    (root/'results/nmt_cfg_final_test_test.jsonl').write_text(json.dumps({'id':'s1','hyp':'ok'})+'\\n')
else:
    assert (root/'results/nmt_cfg_final_test_test.jsonl').exists()
    atomic_json(root/'results/safety_eval_test_final.json',{'real':{'n_items':1},'seeded':{}})
'''
    for name in ("run_adr001_v2.py", "eval_nmt_config.py", "eval_safety.py"):
        (tmp_path / "tools" / name).write_text(job, encoding="utf8")
    return tmp_path


def bash_path():
    git = shutil.which("git")
    candidate = Path(git).parent.parent / "bin/bash.exe" if git and os.name == "nt" else None
    path = str(candidate) if candidate and candidate.exists() else shutil.which("bash")
    if not path:
        pytest.skip("Bash 5.1+ unavailable for actual shell fault tests")
    return path


def run_batch(root, **settings):
    env = {**os.environ, "ONEVOICE_PROJECT_ROOT": str(root), "PYTHON": sys.executable, **settings}
    return subprocess.run([bash_path(), str(root / "tools/final_test.sh")], env=env, capture_output=True, text=True, timeout=20)


def test_batch_marks_done_only_after_all_five_jobs_and_valid_outputs(batch_root):
    result = run_batch(batch_root)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = json.loads((batch_root / "results/FINAL_TEST_DONE").read_text())
    assert marker["status"] == "COMPLETE" and len(marker["outputs_sha256"]) == 5
    assert len(marker["subprocesses"]) == 5 and all(row["exit_code"] == 0 for row in marker["subprocesses"])
    assert not (batch_root / "results/FINAL_TEST_RUNNING").exists()
    assert run_batch(batch_root).returncode != 0  # once-only protocol retained


@pytest.mark.parametrize("failed", ["m1_demand", "m1_babble", "m1_alarm", "m2", "m3"])
def test_every_background_failure_prevents_marker(batch_root, failed):
    result = run_batch(batch_root, FAIL_TASK=failed)
    assert result.returncode != 0 and "exit=7" in result.stderr
    assert not (batch_root / "results/FINAL_TEST_DONE").exists()
    assert not (batch_root / "results/FINAL_TEST_RUNNING").exists()
    if failed == "m2": assert not (batch_root / "pid_m3").exists()


def test_fail_fast_kills_long_running_sibling(batch_root):
    began = time.monotonic()
    result = run_batch(batch_root, FAIL_TASK="m1_alarm", SLOW_TASK="m1_demand")
    assert result.returncode != 0 and time.monotonic() - began < 4
    import psutil
    for path in batch_root.glob("pid_*"):
        assert not psutil.pid_exists(int(path.read_text()))


def test_zero_exit_with_corrupt_output_still_prevents_marker(batch_root):
    result = run_batch(batch_root, CORRUPT_TASK="m1_alarm")
    assert result.returncode != 0 and "validation failed" in result.stderr
    assert not (batch_root / "results/FINAL_TEST_DONE").exists()
    rows = [json.loads(line) for line in (batch_root / "results/final_test_processes.jsonl").read_text().splitlines()]
    assert len(rows) == 5 and all(row["exit_code"] == 0 for row in rows)
