# ToneBridge Prototype — Laptop runtime (Milestones A–C)

This directory contains the laptop reference engine. The real runtime is **VI→EN, CPU**, processing one endpointed 16-kHz mono WAV per invocation. Missing urgency weights/scaler are explicitly UNCONFIGURED and cues are disabled. Optional laptop playback streams approved chunks; Android capture/playback, NPU execution, bidirectional translation and prosody transfer remain outside this engine.

## Install and validate

Python **3.11–3.13** is supported by the config/preflight. The current validation target is Windows x64 / Python 3.13.9. Run the commands below from `OneVoice/Prototype/`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tools/check_runtime.py --mode stub --profile dev
.\.venv\Scripts\python.exe -m pytest tests/
```

On Linux/macOS use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

| Profile | Install file | Purpose |
|---|---|---|
| Runtime | `requirements-runtime.txt` | numpy/scipy, validated contracts, telemetry, ORT/sherpa/SentencePiece/SwiftF0 |
| Development | `requirements-dev.txt` | Runtime + pytest, ONNX graph references, librosa/soundfile/jiwer/soxr, urgency training self-checks |
| Benchmark/export | `requirements-bench.txt` | Development + Optimum/Transformers/export/tokenization tools |
| Optional playback | `requirements-playback.txt` | Runtime + pinned sounddevice/PortAudio; physical output only with `--playback` |

Direct dependencies have exact pins. Preflight checks installed public versions and actual imports (including DLL failures). `--allow-version-drift` is a diagnostic option for the preflight tool; the CLI requires the runtime pins. Transitive dependencies are resolved by pip; this is not a cross-platform hash-locked wheel repository.

## Explicit modes

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m tonebridge.cli --mode stub --wav input.wav --out results/output.wav --log results/turns.jsonl
.\.venv\Scripts\python.exe -m tonebridge.cli --mode full --wav input.wav --out results/full-output.wav --assert-offline
```

`--mode` is mandatory. The old `--real` and `--full` flags are rejected.

| Mode | Capabilities |
|---|---|
| `stub` | Existing skeleton: sidecar/fixed ASR, tagged NMT, always-pass checker, sine tone. This is only a control-flow demo. |
| `branch-a-bench` | Real Silero/Zipformer/Marian/Piper; existing skeleton safety/urgency/gate. Explicitly a benchmark, not a safe translation mode. |
| `full` | Real meaning/acoustic adapters, scoped safety and quality gate. Urgency UNCONFIGURED when either artifact path is null; a configured but untuned model cannot emit a cue. |

Startup writes structured mode/capabilities to stderr. Default JSONL/stdout contains only numeric metadata, mode, direction, effective config hash, gate action and safe error codes. Conversation text, identifiers, arbitrary gate reasons and network details are excluded. A passthrough denoiser when `denoise_mode=off` is the selected ADR-001 policy, not an implicit model fallback.

Config missing, malformed or containing unknown fields fails before inference. There is no default-config fallback. The default path is anchored to the project, independent of the shell working directory.

Exit codes:

- `0`: a completed turn, including a legitimate REPEAT/CONFIRM/ABSTAIN gate decision.
- `1`: invalid audio, stage error, process failure or timeout; no output audio is written.
- `2`: argument/configuration/dependency/asset-preflight error.

Use a fresh output path. Input and output must differ. If a blocked turn's requested output already exists, the CLI exits with an explicit error rather than making that old file appear to be the new result.

## Real model packs and preflight

```powershell
.\.venv\Scripts\python.exe tools/check_runtime.py --mode full
.\.venv\Scripts\python.exe tools/check_runtime.py --mode branch-a-bench
```

The repository intentionally ignores `models/` and `data/`. Run `tools/provision_models.py --mode branch-a-bench` or `--mode full` with benchmark/export dependencies to obtain the official pinned packs. See [MODEL_PROVISIONING.md](../docs/MODEL_PROVISIONING.md) for source revisions, integrity rules and original-pack import. Preflight checks **every selected file's size and SHA-256**, without creating/downloading/loading models.

The 14 October provisioning run verified 369/370 full assets (367/368 branch-a-bench). The locally reconstructed merged NMT decoder does not match the original pin and is deleted. Both preflights remain blocked by that one file; real inference has not been accepted. GTCRN is only required when denoise is ON/OA. The two Branch B graphs are only required in full mode. Test WAVs and evaluation corpora are separate inputs, not runtime model assets.

Custom model file names/paths must also be pinned in the asset manifest. Manifest duplicates, malformed hashes and paths escaping the project root fail. Model loading uses the same selected paths that preflight verified.

## Single effective config

Edit **`configs/pipeline.json`**:

- `thresholds`: ASR repeat/abstain thresholds, voicing quality, urgency score and translation length ratio.
- `branch_b`: acoustic evidence rules, `mlp_path`, `standardizer`, HIGH threshold and `tuned` flag; separate from fusion/quality thresholds.
- `quality`: independent acoustic rejection thresholds and explicit calibration provenance. These settings participate in the effective fingerprint.
- `decode`: threads, ASR files/method/path count, NMT files/token cap/beam settings, glossary groups/bonus/EOS penalty and TTS file/chunk size/`tts_first_clause`.
- `vad`: endpointing/padding settings.
- `models`: selected pack locations and provenance.
- `max_utterance_s`, `silence_peak`, `branch_timeout_s`, `turn_timeout_s`, `model_load_timeout_s`.
- `asset_manifest`, `safety_lexicon`: resource locations resolved relative to the project root.

For an alternate config under a `configs/` directory, its parent project directory is the asset root; otherwise the config's own directory is the asset root.

Legacy `configs/gate.json` and `configs/branch_b.json` are historical documentation only. Factories never load them. The effective snapshot hashes settings, resolved model paths, selected asset pins, actual lexicon bytes/version and mode. The exact lexicon snapshot is shared by constrained decoding and the checker. The factory's thread/glossary overrides are reflected in that snapshot.

Changing source config files during a turn cannot change the constructed settings. Mutating the Pipeline's effective settings is rejected; create a new Pipeline for a new config. Model pack pins are verified again inside the CLI worker and a changed real-mode fingerprint is rejected at load.

## Audio and failure policy

- Input must be **16 kHz, mono**, at most `max_utterance_s` (up to 15 s). Resample/downmix explicitly upstream.
- WAV PCM uint8 is centered at 128; signed int16/int32 are divided by their signed full scale. SciPy's left-aligned PCM24/int32 representation uses the same int32 scale.
- NaN/Inf, out-of-range IEEE floats, stereo, empty and overlong buffers are rejected before inference. Direct `Pipeline.run` expects normalized floating-point audio.
- Silence short-circuits before models. VAD's empty or too-short speech span returns REPEAT without calling ASR/NMT/TTS. Empty ASR does not call NMT. Token-cap truncation blocks TTS.
- A stage exception returns an ABSTAIN error record. Any partial TTS audio is discarded. Both branch waits share a deadline rather than receiving separate cumulative budgets.
- Thread timeout retires the library Pipeline permanently; do not reuse its native sessions. Python cannot forcibly cancel an inference thread.
- All CLI modes use a **spawned disposable process**. Separate load/turn watchdogs terminate/kill and reap the process, including native hangs/crashes. Python/native worker stdout/stderr is suppressed; results travel through structured local IPC. Model startup is outside the recorded endpoint latency; this is not a persistent warm-serving benchmark.

No new production mock stage was introduced. Tests use the existing skeleton and inject failures at its boundaries. Integration tests explicitly skip when their real model/data prerequisites are absent; a green basic suite does not establish full model quality.

## Scoped output safety and privacy (Milestone B)

Full mode compares VI/EN clinical relations per clause: subject, drug, Decimal amount, unit, frequency, action, negation and modality. Symptoms, history, condition, timing and administration site also participate in matching. Decimal comma/dot are equivalent. Unsupported critical syntax, changed/added entities and ambiguous bindings block TTS. Anchored unknown drug names may only be copied exactly; they are not clinically verified aliases. This bounded grammar is not a clinical certification or a general semantic parser.

Acoustic evidence uses raw PTT energy, 20 ms frame-energy contrast (a blind SNR proxy), real VAD speech coverage, SwiftF0 voicing quality, transcript rate and repetition. Token confidence alone cannot authorize speech. Defaults are **uncalibrated**: weak evidence gives REPEAT, missing evidence gives REPEAT/CONFIRM, and apparently strong evidence still gives CONFIRM in full mode. Automatic SPEAK requires an explicitly calibrated policy (`quality.calibrated=true`, `quality.calibration_id`) after evaluating recall, false rejects and coverage on representative audio. A provenance identifier records that choice; it does not itself prove clinical accuracy. The current checkout lacks the assets/corpora needed for that evaluation.

Default log: `results/turns.jsonl`, policy `metadata-v1`. Text remains in memory for safety/TTS but ordinary `TurnRecord.model_dump()` / `.model_dump_json()` exclude it. To explicitly retain transcripts for research:

```powershell
$env:PYTHONPATH = 'src'
.venv/clean/Scripts/python.exe -m tonebridge.cli --mode stub --eval-mode --wav input.wav --out results/eval-output.wav --log results/eval-turns.jsonl
```

`--eval-mode` enables `asr_text`, `nmt_text`, `gate.speak_text` and detailed reasons in stdout/JSONL, with policy `evaluation-v1`. Treat those artifacts as sensitive. Runtime and evaluation defaults use separate files; mixed-policy, legacy text-bearing or unverifiable existing logs are refused without deleting them. Choose a fresh log path. Argument/config/IO errors do not echo user values, filenames or exception messages.

Regression results and limits: [Milestone B acceptance report](../reports/MILESTONE_B_2026-10-14.md). Trained urgency and real-data quality calibration remain outstanding.

## Urgency artifact contract (Milestone C)

Configure `pipeline.json.branch_b.mlp_path` and `.standardizer`. `configs/branch_b.json` documents this contract; it is never an independent runtime config. Both artifacts must additionally be pinned by size/SHA-256 in `runtime_files.json`. Null either path means UNCONFIGURED, with `cue_enabled=false`; a bad configured artifact is a load failure, never an implicit fallback. Stub/branch-a-bench urgency is explicitly DISABLED.

The standardizer is JSON with `schema_version=tonebridge-urgency-v1`, the exact ordered `branch_b.FEATURE_NAMES` (21), `class_names=[LOW,HIGH]`, finite `mean[21]`, positive `scale[21]` and `model_sha256`. ONNX must expose float `z[B,21] -> logits[B,2]` and matching schema/feature/class metadata. The loader normalizes `(x-mean)/scale` and applies stable softmax to get P(HIGH), rather than treating logits as probabilities. `urgency_train.export_onnx` and `export_standardizer` export that contract; they do not manufacture or train real-data weights.

READY means artifacts loaded, not calibrated clinical performance. Real cues require `branch_b.tuned=true`, sufficient voicing/score and all safety/quality checks; leave tuning false until validated on real recordings. Weak acoustic input can still return UNKNOWN with specific reasons. Default runtime logs include enum classifier status/cue state without transcript content.

## First-clause delivery and optional playback

`decode.tts_first_clause=true` makes Piper synthesize/yield the first clause before synthesizing the remainder. Set false for an explicit whole-utterance baseline. Full translation must pass the gate before any TTS; this does not stream unvalidated NMT tokens. A Piper synthesis call itself remains buffered.

`Pipeline.run(..., on_audio=callback)` delivers approved chunks immediately. Default CLI remains buffered and writes the final WAV only on success. For physical streaming output:

```powershell
.venv/clean/Scripts/python.exe -m pip install -r requirements-playback.txt
.venv/clean/Scripts/python.exe tools/check_runtime.py --mode stub --profile playback
$env:PYTHONPATH = 'src'
.venv/clean/Scripts/python.exe -m tonebridge.cli --mode full --wav input.wav --out results/live-output.wav --playback
```

Playback uses a lazy mono sounddevice output stream inside the killable inference worker. Queued output is aborted on failure. Already played, fully approved chunks cannot be retracted after a later TTS/device fault; the buffered path still discards partial audio. `endpoint_to_first_audio_ms` measures buffer readiness; `endpoint_to_first_delivery_ms` measures callback completion. Neither is microphone-measured audible latency. Physical audible onset needs device/loopback measurements.

## Honest E2E and batch benchmarks

```powershell
.venv/clean/Scripts/python.exe tools/bench_e2e.py --mode stub --turns 100 --out results/bench_stub_new.json
.venv/clean/Scripts/python.exe tools/bench_e2e.py --mode branch-a-bench --wav speech1.wav speech2.wav --turns 100 --out results/bench_real_new.json
```

At least 50 turns; default 100. One persistent process loads stages once, records cold spawn/load/first inference separately, then measures warm turns with the same sessions. Reports min/max/p50/p95 with sample counts, audio/gate coverage, stage capabilities and metadata-only turn rows. Default stub inputs are four synthetic tones cycled across turns and only demonstrate control flow. Real modes require pinned packs and genuine speech WAVs; missing packs produce BLOCKED_PREFLIGHT with no fabricated latency/RSS. `--mode full` is supported for measuring resident full stages, including SwiftF0/band/safety/gate; its default quality policy may block every TTS turn, which is explicitly reported.

RSS uses **MiB = bytes / 2^20**, **GiB = bytes / 2^30**. The benchmark samples parent + descendant resident memory together every 10 ms and also records worker RSS/lifetime peak. Shared pages may count in multiple processes and sampling may miss short peaks. Legacy `peak_rss_mb` is an explicitly documented MiB API alias; new reports use `_mib`/`_gib` and `rss_unit`.

`tools/final_test.sh` requires Bash 5.1+, supervises each PID/exit code, launches safety evaluation only after successful NMT, and kills/reaps siblings after a failure. Completion requires all five jobs exit 0 **and** validated output coverage/provenance; the marker records output hashes. Existing final marker or stale NMT/safety outputs are refused for review/archive before another run. Do not rerun the frozen held-out experiment casually. The smoke/fault tests operate only in temporary roots.

ADR grid checkpoints use a fingerprint over central config, active asset pins, code, runtime versions, inputs/references, noise, seed and decode/grid settings. Legacy/mismatched/incomplete checkpoints are refused; outputs are atomic and `complete=true` is written only when all requested cells succeeded. OFF no longer loads GTCRN unnecessarily; empty ASR hypotheses count pure deletions under jiwer 4.

Measured results and outstanding real-model acceptance: [Milestone C report](../reports/MILESTONE_C_2026-10-14.md).

## Main integration

Registered laptop configs (`configs/laptop_*.json`) retain four-direction routing. They use the same bounded Pipeline, completion/evidence checks, acoustic confirmation and metadata-only serializer. `run_laptop.py` supplies explicit full mode and runs Windows workers in UTF-8. Direct CLI invocations require `--mode` and must use `-X utf8` for Marian laptop adapters on Windows. Historical Milestone config still authenticates `runtime_files.json`; the restoration registry has its own pinned laptop manifest, never an implicit replacement for those pins.
