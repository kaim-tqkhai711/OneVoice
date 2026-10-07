"""Offline proof: the --full pipeline runs end-to-end with every socket blocked, HF/Transformers offline env set, and all library caches redirected to
EMPTY temp dirs (so a model fetched from a library cache would fail, and anything written to a cache is detected). Pass = 0 connection attempts."""
import socket
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from tonebridge.config import PipelineConfig
from tonebridge.offline_guard import OfflineGuard, OfflineViolation

ROOT = Path(__file__).resolve().parents[1]
WAVS = [ROOT / "models/asr/zipformer-vi-int8/test_wavs" / f"{i}.wav" for i in range(3)]
needs_models = pytest.mark.skipif(not all(w.exists() for w in WAVS) or not (ROOT / "models/nmt/vi-en-int8-arm64").exists(), reason="models/ not present")


def test_guard_blocks_and_counts():
    with OfflineGuard() as g:
        with pytest.raises(OfflineViolation):
            socket.create_connection(("127.0.0.1", 9), timeout=0.2)
        with pytest.raises(OfflineViolation):
            socket.socket().connect(("10.255.255.1", 80))
        with pytest.raises(OfflineViolation):
            socket.getaddrinfo("huggingface.co", 443)
        with pytest.raises(Exception):
            urllib.request.urlopen("http://example.com", timeout=1)
    assert len(g.attempts) >= 4
    socket.getaddrinfo("localhost", 80)  # patches are removed again on exit


@needs_models
def test_full_pipeline_offline_three_wavs(tmp_path, monkeypatch):
    caches = {}
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME", "TORCH_HOME"):
        caches[var] = tmp_path / var
        caches[var].mkdir()
        monkeypatch.setenv(var, str(caches[var]))
    from tonebridge.pipeline import Pipeline
    from tonebridge.stages.factory import real_full

    cfg = PipelineConfig.load(ROOT / "configs/pipeline.json")
    with OfflineGuard() as g:
        pipe = Pipeline(cfg, real_full(cfg))
        for w in WAVS:
            sr, x = wavfile.read(w)
            assert sr == cfg.sample_rate
            res = pipe.run(x.astype(np.float32) / 32768.0, utt_id=w.stem)
            assert res.record.asr_text and res.record.nmt_text
            assert res.record.endpoint_to_text_ms is not None
    assert g.attempts == [], g.attempts
    assert g.os_conns_seen == set(), g.os_conns_seen
    for var, d in caches.items():
        assert list(d.iterdir()) == [], f"{var} was written to: {list(d.iterdir())}"


@needs_models
def test_no_network_capable_libs_imported_and_swiftf0_copy_identical():
    import hashlib
    import subprocess
    import sys

    code = ("import sys; from tonebridge.config import PipelineConfig; from tonebridge.stages.factory import real_full; "
            "real_full(PipelineConfig.load('configs/pipeline.json')); "
            "bad=[m for m in ('torch','transformers','huggingface_hub','requests','urllib3','librosa','optimum') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    import swift_f0
    wheel = Path(swift_f0.__file__).parent / "model.onnx"
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == hashlib.sha256((ROOT / "models/branch_b/swiftf0_model.onnx").read_bytes()).hexdigest()


@needs_models
def test_empty_segment_does_not_crash_asr():
    from tonebridge.stages.asr_sherpa import SherpaZipformerVi
    r = SherpaZipformerVi(decoder="decoder-epoch-12-avg-8.int8.onnx").transcribe(np.zeros(0, np.float32), "vi")
    assert r.text == "" and r.confidence == 0.0
