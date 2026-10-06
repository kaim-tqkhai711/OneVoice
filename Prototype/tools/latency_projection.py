"""E2E latency projection from MEASURED x86 stage times (results/latency_options.json, quiet run). Not a measurement on the phone.
E2E(PTT release -> first TTS sample) = VAD tail + ASR (+ NMT) + TTS first audio, each x86 time multiplied by k. Label: "x86 proxy x k (est.)".
Summing per-stage p95 values over-estimates the p95 of the sum (stages are not perfectly correlated): shown as a conservative bound, and also as p50-sum.
python tools/latency_projection.py -> results/latency_projection.md"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results/latency_options.json").read_text())["latency_ms_by_speech_bucket"]
VAD_MS = 0.0  # silero decision latency is inside the PTT-release flush here; not separately measured, stated as 0 (not zero in reality)
L = ["### Projected E2E (ms), x86 proxy x k (est.), vs Proposal p50 <= 1500 / p95 < 2000", "",
     "| speech s | config | k | p50 (est.) | p95 bound (est.) | vs 1.5 s | vs 2.0 s |", "|---|---|---|---|---|---|---|"]
cfgs = {"baseline (full ASR decode, full TTS)": ("asr_full", "tts_full"), "ASR tail only": ("asr_tail", "tts_full"),
        "TTS first clause only": ("asr_full", "tts_first_clause"), "both": ("asr_tail", "tts_first_clause"),
        "both + int8 voice": ("asr_tail", "tts8_first_clause")}
for b in ("2", "4", "6"):
    for name, (a, t) in cfgs.items():
        for k in (3, 4):
            p50 = (d[b][a]["p50"] + d[b]["nmt"]["p50"] + d[b][t]["p50"] + VAD_MS) * k
            p95 = (d[b][a]["p95"] + d[b]["nmt"]["p95"] + d[b][t]["p95"] + VAD_MS) * k
            L.append(f"| {b} | {name} | {k} | {p50:.0f} | {p95:.0f} | {'PASS' if p50 <= 1500 else 'FAIL'} | {'PASS' if p95 < 2000 else 'FAIL'} |")
out = "\n".join(L)
(ROOT / "results/latency_projection.md").write_text(out, encoding="utf8"); print(out)
