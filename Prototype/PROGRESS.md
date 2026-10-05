# ToneBridge Prototype: PROGRESS

## Deviations from the Tech Proposal v3.1 (declared)
| # | Deviation | Reason | Status |
|---|---|---|---|
| 1 | **P0 = VI->EN. VI->KO is not demonstrated** (Proposal §2.4: "VI ↔ KO primary; VI ↔ EN secondary/supporting pivot"). KO is reported as "measured, did not meet budget/quality" | NMT bake-off: tc-big en->ko broken + 265 MB; SMaLL-100 vi->ko failed 3/3 pre-set thresholds (p50 218 ms > 150, INT8 581 MB > 350, 8/10 < 9/10 readable). See `docs/ADR-002-nmt-direction.md`. Original reason for a single direction: 60 h budget vs Timeline v1 234 h | ADR-002 revised, **Proposed**; Accepted only after owner confirms |
| 2 | No streaming Zipformer for VI in official k2-fsa lists; the Apache-2.0 VI checkpoint is offline. "<450 ms tail" (Proposal 4.2) must be re-measured as full-utterance decode | Verified on k2-fsa docs + HF cards | To measure D2 |
| 3 | Prosody transfer, dashboard, hash-chain audit (docs 01-07) dropped; Moonshine, NLLB dropped | Not in Proposal v3.1 / NLLB is CC-BY-NC | Proposal wins |
| 4 | **Vocal urgency outputs LOW / HIGH + UNKNOWN, not 3 levels.** Proposal §4.2 "Vocal urgency" row, verbatim: "3-level advisory posterior + UNKNOWN via gate" = three labelled levels + UNKNOWN | Recording kit has only NEUTRAL/URGENT; no middle level invented | Declared (owner approved approach) |
| 5 | **WebRTC APM Stage 0 removed** (laptop and phone). Proposal §4.1 "configurable WebRTC APM speech enhancement"; §4.2 "NS; optional HPF/AGC/AEC only where required" | PTT is half-duplex and input is file/segment-based, so AEC does not apply; AGC distorts the energy features of Branch B. Neural denoise (GTCRN) stays as the ADR-001 ON/OA arm. Optional classical-NS comparison arm, laptop only, 1 h timebox | Declared |

## Plan change (2026-10-06, owner decision): laptop-first, phone after gate G-L
Phone work starts only after gate G-L passes (see `docs/TIMELINE_v4.md` and the G-L definition below). Exception: k calibration (below) runs as soon as the SD712 is connected.

### Phone scope (owner decision, supersedes the earlier "port everything")
- **P1 (committed):** per-model measurements on SD712 via adb, no app. ASR: prebuilt sherpa-onnx, RTF + RSS + load time, 2 threads, shipped files. NMT: encoder and one decoder step measured separately with ORT's Android benchmark tool (availability to be verified, owner to be told if absent); E2E NMT = encoder + n_token x decoder_step, labelled "composed (est.)". TTS via sherpa-onnx. k calibrated from these.
- **P2 (conditional):** Kotlin app ASR->NMT->TTS only if P1 is done and D5 has >= 8 h free.
- **P3 (cut):** Branch B, safety check, gate on phone. Their numbers are laptop numbers; the report states "measured on x86, not on device".

## Risks
| # | Risk | Impact | Owner action | Deadline |
|---|---|---|---|---|
| R1 | **Qualcomm AI Hub token missing.** No compile/profile for QCS6490 yet. **Compliance risk against the challenge brief (Hardware 25%).** | Hardware score cannot cite a QCS6490 profile | Moved out of D2. Offline prep: small ONNX model exported fixed-shape + `tools/aihub_submit.py` compile/profile script ready, so it runs the moment the token exists | **Hard deadline: start of D5** |
| R2 | espeak-ng is GPL-3.0; Piper/sherpa-onnx VITS phonemize through it | Release license unresolved | Continue with Piper ljspeech, flagged "license chưa chốt" in `LICENSES.md`. Options in `docs/TTS_GPL_OPTIONS.md` | Must close before freeze (D6) |
| R3 | Latency proxy factor k = 4 is an estimate | G-L performance thresholds may be wrong | Calibrate with measured k when SD712 is connected | Immediately when device available |
| R4 | Earlier INT8 benches used the x86 `avx2` quantization config | Not the file that ships to ARM | Build ship artifacts with the `arm64` config; re-measure from those exact files | D2 |

## Portability law (in force from 2026-10-06)
- Inference path = ONNX graphs run by ORT or sherpa-onnx, plus pure logic (gate, safety check) that reads lexicon/config from JSON files.
- PyTorch, librosa, pyworld are NOT in the inference path. Allowed only for training, evaluation and as reference for equivalence tests.
- Measure with the exact INT8 files that will ship to the phone, `intra_op=2`, no x86-only library.
- Anything that does not comply goes into the port-debt table with an hour estimate.

## Port debt (hours are estimates, not measured)
| # | Item | Why it is debt | Est. hours to port |
|---|---|---|---|
| P1 | (needed only if scope P2) Marian/SentencePiece tokenizer for opus-mt-vi-en | Python `transformers` tokenizer on laptop; Android needs Kotlin/JNI sentencepiece | 3 |
| P2 | NMT greedy decode loop with KV cache around the merged decoder | `optimum` generate() on laptop; Kotlin loop over ORT-Android needed | 4 |
| P3 | ~~WebRTC APM Stage 0~~ | Removed from the pipeline (deviation #5); no debt | 0 |
| P4 | SwiftF0 post-processing glue (scope P3, cut) | numpy glue | 1.5 |
| P5 | Branch B stats + MLP on phone (scope P3, cut) | numpy; trivial in Kotlin but must match reference (`docs/BRANCH_B_DESIGN.md`) | 2.5 |
| P6 | Safety check + gate + glossary on phone (scope P3, cut) | Python logic; JSON-driven by design so the port is mechanical. Reported as "measured on x86, not on device" | 3 |
| P7 | (P2 only) 16 kHz resampling (soxr on laptop) | Android: Oboe at 16 kHz or Kotlin resampler | 1 |
| P8 | Audio-level SNR mixer, WER harness, LOSO | Eval-only, never ships | 0 |

## Gate G-L (laptop gate; all must pass before phone work)
Function:
- VI->EN wav->wav, real model at every stage, 0 stubs.
- Gate emits all 5 actions; there is an ABSTAIN case and a seeded-error case blocked from TTS.
- Branch B test: a denoised input is rejected/never reaches Branch B, test stays green. OA mixer has a sample-alignment test.

Quality (official numbers, laptop, shipping artifacts):
- WER/CER x SNR (clean/10/5/0) vs Proposal §4.5 targets (WER <=15/20/25/35 %), pass/fail per cell.
- ADR-001 table OFF/ON/OA(beta), conclusion by the rule >= 2 points absolute WER.
- Safety: detection recall + false-block rate, Clopper-Pearson 95% CI, with n.
- Urgency: LOSO mean +/- std vs z-score baseline (depends on team recordings).

Performance (PROXY, not a reported number; every laptop number labelled "x86 proxy"):
- E2E PTT release -> first TTS sample, 100 utterances, 2 threads: p50 <= 375 ms, p95 < 500 ms (= Proposal 1.5 s / 2.0 s divided by k = 4, **k is an estimate, not measured**).
- Peak RSS whole pipeline < 1.5 GB. Per-stage latency reported.
- A failing cell is reported as failed; thresholds are not edited.

## k calibration (as soon as the SD712 is connected, not waiting for G-L)
1. `winget install Google.PlatformTools` (owner), then `tools/device_probe.sh`.
2. Measure RTF of exactly one model: Zipformer VI INT8, same file, 2 threads, on phone and laptop.
3. k = RTF_phone / RTF_laptop. Recompute the G-L proxy thresholds: p50 <= 1500/k ms, p95 < 2000/k ms. Record k, both RTFs and n here.

## Session 1 (2026-10-06): Phase 0 fixes + D1

### Done
- D1 items 2-5: contracts/config/telemetry, walking skeleton (stubs, dual-branch fan-out, JSONL), license table v0, recording kit.
- NMT bake-off + ADR-002 draft. `tools/device_probe.sh` written (not yet run on a device).
- Reports: `reports/D1_nmt_bakeoff_va_skeleton.md`. Tests: 10 passed.

### Measurements (laptop x86, NOT SD712; ORT CPU, intra_op=2, INT8 dynamic, greedy, max_len 48, 90 runs)
| Model | Params | INT8 | p50 [95% CI] | p95 [95% CI] | RSS (inference) |
|---|---|---|---|---|---|
| opus-mt-tc-big-en-ko (broken output) | 209,158,401 | 265 MB | 171 ms [152, 196] | 608 ms [287, 664] | 966 MB |
| small100 vi->ko | 332.7M | 581 MB | 218 ms [203, 239] | 373 ms [336, 421] | 1596 MB |

### Blockers
- Q5 pending: SD712 specs. `adb` not found in PATH on this laptop.
- Q6: AI Hub token missing, now risk R1 (deadline start of D5).
- Recording kit: owner sends it to teammates 2026-10-06. Speaker codes stay spk01-spk06.

### Next (D2)
1. GPU/DSP timebox on SD712 (2 h). 2. Prebuilt sherpa-onnx APK on SD712. 3. AI Hub QCS6490 profile (token). 4. Zipformer VI RTF (2 threads) + real INT8 size, then WER harness. 5. opus-mt-vi-en tokenization check + bench + glossary v0. 6. ADR-001: noise mixer, GTCRN, OA convex mix with sample alignment test, WER table in background.

## Session 2 (2026-10-06): D1 review applied
- ADR-002 revised (Proposal section numbers quoted verbatim, KO re-entry criteria, Viet-Korean coverage line). Status Proposed.
- Added: risks R1-R4, portability law, port-debt table, G-L, k calibration, `docs/BRANCH_B_DESIGN.md` (awaiting owner approval, no Branch B code yet), `docs/TIMELINE_v3.md`, `docs/TTS_GPL_OPTIONS.md`.
- Discovered while designing Branch B: SwiftF0's ONNX contains its own STFT (Sin/Cos/MatMul, opset 18, input raw audio); its `core.py` depends only on numpy + onnxruntime.

## D2 (2026-10-06): started
### Zipformer VI INT8 (laptop x86 proxy; `tools/bench_asr_rtf.py`, `results/asr_vi_rtf.json`)
| Item | Value |
|---|---|
| Files shipped | encoder int8 67.59 MB + decoder **fp32** 4.93 MB + joiner int8 0.99 MB = **73.5 MB** (csukuangfj packaging; an int8 decoder, 1.25 MB, exists in zzasdf repo, not yet used) |
| Runtime | sherpa-onnx 1.13.8, offline transducer, greedy, 2 threads, CPU |
| RTF (pooled over 3 test wavs, 7.7 s audio, 10 repeats after 2 warm-up) | **0.040** (per-file p50 0.037-0.045). n is small: this is the k-calibration probe, not a quality number |
| Load time / RSS | 1.96 s / 231 MB after load+decode |
| Output sanity | 3 Vietnamese test wavs decoded to plausible text (uppercase, no diacritics loss). WER not measured yet (no reference transcripts, harness is the next item) |
Used later for k calibration: same file, same script on the phone.

### WER harness + ASR decoder choice (D2)
- Harness: `src/tonebridge/evalkit/` (normalization keeps diacritics, drops case/punctuation; corpus-level WER/CER with counts), `tools/run_asr_eval.py`, tests (13 pass). In-domain loader for `recording_kit` (`<spk>_<NEU|URG>_<sid>`), reports WER/CER split by NEU/URG; **waiting for recordings**.
- Generic set: **google/fleurs vi_vn test, CC-BY-4.0** (license read from the HF dataset API). 857 utts, 27,200 words, 10,814 s. Alternative rejected: VIVOS is CC-BY-NC-SA-4.0. **Overlap caveat:** the 70k h checkpoint is trained on pseudo-labelled data whose sources are not documented; overlap with FLEURS (read Wikipedia-style sentences) cannot be excluded, so treat this WER as optimistic and as a harness check only.
- Decoder INT8 vs fp32 (same encoder/joiner INT8, greedy, 2 threads, FLEURS test, x86): fp32 decoder WER **14.57 %** / CER 9.59 %, RTF 0.042; INT8 decoder WER **14.60 %** / CER 9.61 %, RTF 0.033 (RTF taken while other jobs ran, indicative). Difference 0.03 points WER. **Locked: INT8 decoder (`decoder-epoch-12-avg-8.int8.onnx`, 1.25 MB)**; ASR pack = 67.6 + 1.2 + 1.0 = ~69.8 MB.
- Real stage `stages/asr_sherpa.py` (confidence = geometric-mean token probability).

### opus-mt-vi-en (D2)
- Tokenization check (the tc-big failure mode): the model card has no usage example, so the check used the 30 clinical sentences: **0 unknown pieces / 287**, and the ORT-side tokenizer (spm piece -> vocab.json) matches `MarianTokenizer` ids on 30/30.
- Export: optimum ONNX, max logit diff vs torch 2.8e-5. Quantized with **`arm64` dynamic INT8** config (`tools/bench_hop.py --quant arm64`): encoder 44.8 MB + merged decoder 77.9 MB = **122.7 MB** (fp32 485.9 MB). License Apache-2.0, repo sha c8d2853e.
- Real stage `stages/nmt_ort.py`: ORT + sentencepiece + numpy only, own greedy KV-cache loop (no torch/transformers; matches the portability law). Lean bench (`tools/bench_nmt_ort.py`, 30 sentences x 3, 2 threads, x86 proxy): **p50 38.1 ms, p95 57.6 ms**, encoder p50 2.8 ms, decoder step p50 3.99 ms, 9.0 steps mean, **peak RSS 282 MB**. Composed estimate (encoder + n x step) = 38.7 ms vs measured 38.1 ms: the composition method used for the phone agrees with the whole-sentence number on laptop. The optimum/torch bench path gave p50 125 ms / RSS 1.78 GB: that was framework overhead, not the model.
- Quality (INT8 arm64, greedy, `results/nmt_vi_en_int8_vs_fp32.json`): output identical to fp32 on 24/30; differences are mixed (INT8 corrupts "ibuprofen" -> "medrine"; fp32 has "wine" for "rượu bia"). Errors that matter clinically in both: "năm miligam" -> "five millimetres" (unit), "Huyết áp rất cao" -> "High blood pressure" (drops "very"), "Uống một viên" -> "Take one" (drops "tablet"), "ibuprofen" -> "medrine". These are the classes the glossary and Semantic Safety Check must catch.

### Glossary, front-end arms, TTS, wiring (D2)
- Glossary v0: `configs/glossary_vi_en.json` (64 terms: symptoms, allergy, 15 medications, dose units, 4 negation markers; `strict` hard/soft) + `src/tonebridge/glossary.py` (longest-first, span-consuming match; `check(src, tgt)` reports hit/miss per term). It detects, it does not rewrite. Tests confirm it flags the two D2 NMT errors (miligam -> millimetres, ibuprofen -> medrine). "viên" is soft because "two paracetamols" is acceptable English for "hai viên paracetamol". Note: sentence-final "không?" is a question particle; handled in the D3 safety check.
- ADR-001 arms: `stages/denoise.py` (OFF / GTCRN via sherpa-onnx / `oa_mix`, y = beta*enh + (1-beta)*noisy), `evalkit/noise.py` (SNR mixer, whole-clip power). Tests: OA endpoints + midpoint, length-mismatch rejection, mixer hits 10/5/0 dB within 0.05 dB, **GTCRN output has lag 0 samples and equal length vs noisy input** (the alignment test). 25 tests pass.
- Classical-NS arm: not started (optional, 1 h timebox, laptop only).
- Piper EN (ljspeech-medium, fp32 60.6 MB, sherpa-onnx, 2 threads, 30 sentences x 3, x86 proxy): **full-synthesis time p50 165.8 ms, p95 246.0 ms** (sherpa generates the whole utterance before yielding, so this is first-audio latency), RTF 0.079, peak RSS 276 MB. Proposal target "<400 ms first-audio" (§4.2).
- Silero VAD through sherpa-onnx (`stages/vad_silero.py`), real-stage factory (`stages/factory.py`), `cli --real`. Smoke run, 2.4 s Vietnamese clip -> English wav: ASR "những nơi đã khống chế được căn bệnh" -> "Where the disease is controlled." (source is a noun-phrase fragment from the FLEURS-style test wav; translation not rated), gate SPEAK, endpoint -> first audio 385 ms (first call, cold). **Branch B, safety check and gate are still stubs until D3**, so G-L's "0 stubs" is not met yet.


### ADR-001 dev run (x86, `tools/run_adr001_grid.py`, `results/adr001_dev.json`, log `logs/adr001_dev.log`)
200 FLEURS test utterances (seed 0), noise = DEMAND OHALLWAY/OOFFICE/PCAFETER/PSTATION round-robin (CC-BY-4.0, **not hospital recordings**), measured SNR 10.0 / 5.0 / -0.0 dB, shipped INT8 ASR files, 2 threads. beta = 0.5 fixed, **not tuned**.

n_utts=200  noise=['OHALLWAY', 'OOFFICE', 'PCAFETER', 'PSTATION']  measured SNR={'10': 10.0, '5': 5.0, '0': -0.0}

| SNR | OFF WER / CER | OA0.5 WER / CER | ON WER / CER | best arm by 2-pt rule |
|---|---|---|---|---|
| clean | 13.08 / 8.62 | 13.04 / 8.59 (Δ +0.05 [-0.12, +0.23]) | 14.49 / 10.00 (Δ -1.41 [-1.93, -0.92]) | OFF |
| 10 | 13.29 / 8.82 | 13.24 / 8.86 (Δ +0.05 [-0.16, +0.24]) | 14.88 / 10.40 (Δ -1.59 [-2.18, -1.02]) | OFF |
| 5 | 13.40 / 9.14 | 13.27 / 9.05 (Δ +0.12 [-0.08, +0.33]) | 16.08 / 11.33 (Δ -2.69 [-3.48, -1.96]) | OFF |
| 0 | 15.44 / 11.14 | 15.63 / 11.24 (Δ -0.19 [-0.51, +0.12]) | 21.47 / 16.17 (Δ -6.03 [-7.37, -4.82]) | OFF |

Δ = WER(OFF) − WER(arm) in points, positive = arm better; paired bootstrap over utterances, 95% CI.

Reading, restricted to what was measured: under this noise set GTCRN-ON is worse than OFF in every cell (CI excludes 0 in all four) and the loss grows as SNR drops (+1.4 points clean, +6.0 points at 0 dB); OA(0.5) is indistinguishable from OFF (all CIs include 0). By the >= 2-point rule no arm beats OFF in any cell. Limits: the ASR shows little degradation under DEMAND noise (OFF 13.1 -> 15.4 % from clean to 0 dB), so this grid says little about harsher non-stationary hospital noise (alarms, overlapping speech); this is a dev run on a 200-utterance subset of a read-speech set, and the final ADR-001 table (D4) uses the in-domain recordings and the full protocol.
