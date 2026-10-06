"""D: latency-reduction options, measured. Two modes.
  --wer      ASR decode per Silero-VAD segment vs whole-utterance decode on the dev set (WER difference, paired bootstrap CI). Not a timing run.
  --latency  per-stage timings by utterance length bucket (~2/4/6 s of speech): ASR full decode vs ASR tail (= decode time of the LAST VAD segment,
             the earlier segments are assumed decoded while the key is still held), NMT (locked glossary config), TTS full synthesis vs first clause,
             and the int8 Piper voice. RUN ON A QUIET MACHINE; the output file records the number of other python processes as a sanity flag.
python tools/bench_latency_options.py --wer --out results/asr_chunked_wer_dev.json
python tools/bench_latency_options.py --latency --out results/latency_options.json"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tonebridge.evalkit import datasets
from tonebridge.evalkit.metrics import ErrorCounts
from tonebridge.stages.asr_sherpa import SherpaZipformerVi
from tonebridge.stages.vad_silero import SileroVad

ap = argparse.ArgumentParser(); ap.add_argument("--wer", action="store_true"); ap.add_argument("--latency", action="store_true")
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=200); ap.add_argument("--per-bucket", type=int, default=20)
a = ap.parse_args()
split = json.loads((ROOT / "configs/splits/fleurs_vi_dev_test.json").read_text()); byid = {u: (p, r) for u, p, r, _ in datasets.fleurs_vi()}
vad = SileroVad(); asr = SherpaZipformerVi(decoder="decoder-epoch-12-avg-8.int8.onnx", threads=2)
SR = 16000


def seg_audio(x):
    sp = vad.segments(x)
    return [x[int(max(0, s - 0.1) * SR): int(min(len(x) / SR, e + 0.1) * SR)] for s, e in sp] or [x]


out = {"n_other_python_procs": sum(1 for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower().startswith("python")) - 1}
if a.wer:
    full, chunk, per = ErrorCounts(), ErrorCounts(), []
    nseg = []
    for u in split["dev"][: a.n]:
        x = sf.read(byid[u][0], dtype="float32")[0]; ref = byid[u][1]
        hf = asr.transcribe(x, "vi").text
        segs = seg_audio(x); nseg.append(len(segs))
        hc = " ".join(asr.transcribe(s, "vi").text for s in segs)
        bf, bc = (full.words, full.word_errors), (chunk.words, chunk.word_errors)
        full.add(ref, hf); chunk.add(ref, hc)
        per.append([full.word_errors - bf[1], chunk.word_errors - bc[1], full.words - bf[0]])
    per = np.array(per); rng = np.random.default_rng(0)
    d = [(per[i, 0].sum() - per[i, 1].sum()) / per[i, 2].sum() * 100 for i in (rng.integers(0, len(per), len(per)) for _ in range(1000))]
    out["wer"] = {"n_utts": len(per), "full_WER": round(full.wer * 100, 2), "chunked_WER": round(chunk.wer * 100, 2), "delta_pts_full_minus_chunked": round(float(np.mean(d)), 2),
                  "delta_ci95": [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)], "mean_segments": round(float(np.mean(nseg)), 2),
                  "utts_with_gt1_segment": int(sum(1 for n in nseg if n > 1))}
if a.latency:
    from tonebridge.nmt_constraints import GlossaryConstrainer
    from tonebridge.stages.nmt_ort import OrtMarianNmt
    from tonebridge.stages.tts_piper import PiperEn, split_clauses
    nmt = OrtMarianNmt(threads=2); gc = GlossaryConstrainer()
    tts = PiperEn(threads=2); tts8 = PiperEn(model_dir=ROOT / "models/tts/vits-piper-en_US-ljspeech-medium-int8", threads=2)
    pool = []
    for u in split["dev"]:
        x = sf.read(byid[u][0], dtype="float32")[0]; sp = vad.segments(x)
        if sp: pool.append((u, x, sp[-1][1] - sp[0][0], sp[0][0]))
    for t in (tts, tts8): t.synth("warm up the model please.")
    ms = lambda t0: (time.perf_counter() - t0) * 1000
    rows = {}
    for tgt in (2, 4, 6):
        # FLEURS speech spans are >= ~4.5 s, so the 2/4/6 s buckets are CROPS of the speech span (first tgt seconds after speech start); the VAD then runs on the crop
        cand = [(u, x[int(st * SR): int((st + tgt) * SR)], float(tgt)) for u, x, dur, st in pool if dur >= tgt + 0.3][: a.per_bucket]
        R = {k: [] for k in ["speech_s", "asr_full", "asr_tail", "n_seg", "nmt", "tts_full", "tts_first_clause", "tts8_full", "tts8_first_clause", "n_words_en"]}
        for u, x, dur in cand:
            asr.transcribe(x[: SR], "vi")
            t0 = time.perf_counter(); h = asr.transcribe(x, "vi"); R["asr_full"].append(ms(t0))
            segs = seg_audio(x); t0 = time.perf_counter()
            for i, s in enumerate(segs):
                t0 = time.perf_counter(); asr.transcribe(s, "vi"); last = ms(t0)
            R["asr_tail"].append(last); R["n_seg"].append(len(segs)); R["speech_s"].append(dur)
            t0 = time.perf_counter(); ids = nmt.encode_ids(h.text); o, _ = nmt.greedy(ids, gc(h.text, nmt.piece_ids), 5.0); en = nmt.decode_ids(o); R["nmt"].append(ms(t0))
            R["n_words_en"].append(len(en.split()))
            for name, t in (("tts", tts), ("tts8", tts8)):
                t0 = time.perf_counter(); t.synth(en); R[f"{name}_full"].append(ms(t0))
                t0 = time.perf_counter(); t.synth(split_clauses(en)[0]); R[f"{name}_first_clause"].append(ms(t0))
        q = lambda v, p: round(float(np.percentile(v, p)), 1)
        rows[str(tgt)] = {k: {"p50": q(v, 50), "p95": q(v, 95)} for k, v in R.items()}
        rows[str(tgt)]["n"] = len(cand)
        print(tgt, rows[str(tgt)], flush=True)
    out["latency_ms_by_speech_bucket"] = rows
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps(out)[:600])
