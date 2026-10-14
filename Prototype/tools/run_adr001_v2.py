"""ADR-001 grid: pinned runtime settings, fail-fast, fingerprinted atomic resume.

python tools/run_adr001_v2.py --split dev --noise demand --with-clean --out results/adr001_dev_demand_v3.json
Legacy results lack the new provenance contract and are refused, never silently
reused. Complete is set only after every requested cell succeeds.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_integrity import atomic_json, resume_checkpoint, validate_cell
from tonebridge.config import PipelineConfig, sha256_file
from tonebridge.assets import read_manifest, verify_files
from tonebridge.evalkit import datasets
from tonebridge.evalkit.metrics import ErrorCounts
from tonebridge.evalkit.noise import measured_snr_masked_db, mix_at_snr_masked, speech_mask
from tonebridge.evalkit.noisegen import make_alarm, make_babble


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--noise", choices=("demand", "babble", "alarm"), required=True)
    parser.add_argument("--snrs", nargs="+", default=["10", "5", "0", "-5"])
    parser.add_argument("--with-clean", action="store_true")
    parser.add_argument("--arms", nargs="+", default=["off", "on", "oa0.25", "oa0.5", "oa0.75", "cns"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--asr-method", choices=("greedy_search", "modified_beam_search"))
    parser.add_argument("--chunked", action="store_true")
    args = parser.parse_args(argv)
    snrs = [format(float(value), "g") for value in args.snrs]
    if any(not math.isfinite(float(v)) or not -40 <= float(v) <= 40 for v in snrs) or len(set(snrs)) != len(snrs):
        parser.error("SNRs must be distinct finite values in [-40,40]")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if len(set(args.arms)) != len(args.arms) or any(arm not in ("off", "on", "cns") and not (arm.startswith("oa") and 0 <= float(arm[2:]) <= 1) for arm in args.arms):
        parser.error("invalid/duplicate enhancement arms")
    cfg = PipelineConfig.load(args.config)
    if args.threads is not None: cfg.decode.threads = args.threads
    if args.asr_method is not None: cfg.decode.asr_method = args.asr_method
    cfg = cfg.prepare("branch-a-bench", verify_assets=False)
    enhancement = any(arm == "on" or arm.startswith("oa") for arm in args.arms)
    paths = [cfg.model_path("vad")] + [cfg.model_path("asr_vi") / name for name in
        (cfg.decode.asr_encoder, cfg.decode.asr_decoder, cfg.decode.asr_joiner, "tokens.txt")]
    if enhancement: paths.append(cfg.model_path("denoise"))
    manifest = read_manifest(cfg)
    if any(path not in manifest for path in paths):
        raise ValueError("ADR active assets not pinned")
    assets = [manifest[path] for path in paths]
    if verify_files(assets):
        raise ValueError("ADR active assets missing or invalid")
    from tonebridge.stages.asr_sherpa import SherpaZipformerVi
    from tonebridge.stages.vad_silero import SileroVad
    from tonebridge.stages.denoise import GtcrnDenoiser, oa_mix
    from tonebridge.stages.denoise_classic import ClassicalNs
    split_path = ROOT / "configs/splits/fleurs_vi_dev_test.json"
    split = json.loads(split_path.read_text(encoding="utf8"))
    byid = {uid: (path, ref) for uid, path, ref, _ in datasets.fleurs_vi()}
    ids = split[args.split][:args.limit]
    if not ids: raise ValueError("empty evaluation cohort")
    seed = split["seed"]
    def load(uid):
        audio, sr = sf.read(byid[uid][0], dtype="float32")
        if sr != 16000 or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError("invalid dataset audio")
        return audio
    tracks = []
    if args.noise == "demand":
        for archive in sorted((ROOT / "data/noise/demand").glob("*_16k.zip")):
            with zipfile.ZipFile(archive) as zipped:
                with zipped.open(next(name for name in zipped.namelist() if name.endswith("ch01.wav"))) as stream:
                    audio, sr = sf.read(stream, dtype="float32")
                    if sr != 16000: raise ValueError("noise sample rate mismatch")
                    tracks.append((archive.stem, audio))
    elif args.noise == "babble":
        tracks.append(("babble6", make_babble([load(uid) for uid in split["babble_pool"]], 90., 6, seed + 1)))
    else:
        tracks.append(("alarm", make_alarm(90., seed + 2)))
    if not tracks: raise ValueError("noise corpus absent")
    source_files = [Path(__file__), ROOT / "tools/benchmark_integrity.py", ROOT / "src/tonebridge/evalkit/metrics.py", ROOT / "src/tonebridge/evalkit/noise.py",
        ROOT / "src/tonebridge/evalkit/noisegen.py", ROOT / "src/tonebridge/evalkit/textnorm.py",
        ROOT / "src/tonebridge/stages/asr_sherpa.py", ROOT / "src/tonebridge/stages/vad_silero.py",
        ROOT / "src/tonebridge/stages/denoise.py", ROOT / "src/tonebridge/stages/denoise_classic.py"]
    spec = {"split": args.split, "noise": args.noise, "arms": args.arms, "snrs": snrs,
        "with_clean": args.with_clean, "threads": cfg.decode.threads, "asr_method": cfg.decode.asr_method,
        "asr_decoder": cfg.decode.asr_decoder, "chunked": args.chunked, "seed": seed,
        "runtime_config_hash": cfg.config_hash(), "config_file_sha256": sha256_file(args.config),
        "split_sha256": sha256_file(split_path), "assets": assets,
        "code_sha256": {path.name: sha256_file(path) for path in source_files},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "jiwer", "soundfile", "onnxruntime", "sherpa-onnx")},
        "inputs": [{"uid": uid, "audio_sha256": sha256_file(Path(byid[uid][0])),
                    "reference_sha256": hashlib.sha256(byid[uid][1].encode()).hexdigest()} for uid in ids],
        "noise_sha256": {name: hashlib.sha256(audio.tobytes()).hexdigest() for name, audio in tracks}}
    conditions = (["clean"] if args.with_clean else []) + snrs
    expected = [f"{args.noise}|{snr}|{arm}" for snr in conditions for arm in args.arms]
    result = resume_checkpoint(args.out, spec, expected, ids)
    if result["complete"]:
        print("COMPLETE (matching fingerprint, all cells already verified)", flush=True)
        return 0
    v = cfg.vad
    vad = SileroVad(model=cfg.model_path("vad"), min_silence_s=v.min_silence_s, min_speech_s=v.min_speech_s, threshold=v.threshold, pad_s=v.pad_s)
    d = cfg.decode
    asr = SherpaZipformerVi(model_dir=cfg.model_path("asr_vi"), encoder=d.asr_encoder, decoder=d.asr_decoder,
        joiner=d.asr_joiner, threads=d.threads, decoding_method=d.asr_method, num_active_paths=d.asr_active_paths)
    # OFF neither constructs nor requires GTCRN/classical denoise sessions.
    gt = GtcrnDenoiser(model=cfg.model_path("denoise"), threads=d.threads) if enhancement else None
    cns = ClassicalNs() if "cns" in args.arms else None
    def transcribe(audio):
        if not args.chunked: return asr.transcribe(audio, "vi").text
        spans = vad.segments(audio)
        return " ".join(asr.transcribe(audio[int(max(0, start - .1) * 16000):int(min(len(audio) / 16000, end + .1) * 16000)], "vi").text for start, end in spans) if spans else asr.transcribe(audio, "vi").text
    clips = []
    for uid in ids:
        audio = load(uid); start, end = vad.segment(audio)
        clips.append((uid, audio, byid[uid][1], speech_mask(len(audio), [(start, end)] if end > start else [])))
    atomic_json(args.out, result)
    began = time.perf_counter()
    for snr in conditions:
        if all(f"{args.noise}|{snr}|{arm}" in result["cells"] for arm in args.arms): continue
        rng = np.random.default_rng(seed + (0 if snr == "clean" else int(float(snr)) + 100) + {"demand": 0, "babble": 1000, "alarm": 2000}[args.noise])
        noisy, checks = [], []
        for index, (_, audio, _, mask) in enumerate(clips):
            mixed = audio if snr == "clean" else mix_at_snr_masked(audio, tracks[index % len(tracks)][1], float(snr), mask, rng)
            noisy.append(mixed)
            if snr != "clean": checks.append(measured_snr_masked_db(audio, mixed, mask))
        if checks:
            result["snr_check"][f"{args.noise}|{snr}"] = {"target": float(snr), "mean": float(np.mean(checks)), "max_abs_dev": float(np.max(np.abs(np.array(checks) - float(snr))))}
        enhanced = [gt.process(audio) for audio in noisy] if gt else None
        classical = [cns.process(audio) for audio in noisy] if cns else None
        for arm in args.arms:
            key = f"{args.noise}|{snr}|{arm}"
            if key in result["cells"]: continue
            counts, per, hypotheses = ErrorCounts(), [], []
            for index, (_, _, reference, _) in enumerate(clips):
                audio = noisy[index] if arm == "off" else enhanced[index] if arm == "on" else classical[index] if arm == "cns" else oa_mix(noisy[index], enhanced[index], float(arm[2:]))
                before = (counts.words, counts.word_errors, counts.chars, counts.char_errors)
                hyp = transcribe(audio); hypotheses.append(hyp); counts.add(reference, hyp)
                per.append([counts.words - before[0], counts.word_errors - before[1], counts.chars - before[2], counts.char_errors - before[3]])
            result["cells"][key] = {"WER": counts.wer, "CER": counts.cer, "words": counts.words, "per_utt": per, "hyp": hypotheses}
            validate_cell(result["cells"][key], len(ids))
            atomic_json(args.out, result)
            print(f"{key} WER={counts.wer:.4f} CER={counts.cer:.4f} elapsed_s={time.perf_counter()-began:.1f}", flush=True)
    if set(result["cells"]) != set(expected): raise ValueError("incomplete requested grid")
    result["complete"] = True
    atomic_json(args.out, result)
    print("COMPLETE: every requested cell succeeded", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ADR failed: {type(error).__name__}; no completion flag created", file=sys.stderr)
        raise SystemExit(1)
