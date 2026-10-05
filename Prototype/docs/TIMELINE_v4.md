# Timeline v4, 2026-10-06 (supersedes v3)

Hours are my estimates. Budget: D2-D6 <= 50 h (owner's cap; D1 already spent). Requirement: D6 keeps >= 4 h real buffer.
Scope (owner decisions 2026-10-06): laptop = full system + gate G-L. Phone P1 = per-model measurements via adb, no app. P2 = Kotlin ASR->NMT->TTS app, only if P1 is done and D5 has >= 8 h free. P3 = Branch B / safety / gate on phone is cut (port debt). WebRTC Stage 0 removed.

| Day | Item | Est. h |
|---|---|---|
| **D2** | WER harness + generic VI set + in-domain loader (recordings pending) | 2.0 |
| | ASR decoder INT8 vs fp32 (size, RTF, WER), lock shipped file | 0.5 |
| | opus-mt-vi-en: tokenization check, export, INT8 `arm64`, bench | 2.0 |
| | Glossary v0 | 1.5 |
| | ADR-001: SNR mixer, GTCRN ONNX, OA mix + alignment test, WER table in background | 3.0 |
| | Piper EN via sherpa-onnx | 1.0 |
| | Replace all stubs, VI->EN wav->wav smoke run | 1.5 |
| | Docs + commits | 0.5 |
| | **D2** | **12.0** (+2.0) |
| **D3** | Branch B (22 features, graph, normalization G/S, UNKNOWN rules, librosa equivalence tests) | 5.0 |
| | MLP + LOSO (G vs S) + z-score baseline (needs recordings) | 3.0 |
| | Semantic Safety Check + seeded error set | 3.0 |
| | Fusion + gate, 5 actions, ABSTAIN + TTS-block cases | 2.0 |
| | Provenance/alignment tests | 1.0 |
| | **D3** | **14.0** (+4.0) |
| **D4** | WER/CER x SNR grid (NEUTRAL/URGENT split) | 2.0 |
| | ADR-001 table (OFF / GTCRN / OA; classical-NS arm only if timebox 1 h, **optional, not counted**) | 1.0 |
| | Safety set authoring + recall / false-block + Clopper-Pearson | 3.0 |
| | Urgency LOSO report | 0.5 |
| | E2E 100 utt, RSS, per-stage latency ("x86 proxy") | 1.5 |
| | Fixes + freeze candidate artifacts | 3.0 |
| | AI Hub prep (fixed-shape ONNX + `tools/aihub_submit.py`) | 2.0 |
| | **D4** | **13.0** (+3.0) |
| **D5** | **Phone P1:** adb + `device_probe.sh` | 0.5 |
| | ASR via prebuilt sherpa-onnx: RTF, RSS, load, 2 threads, shipped files | 1.5 |
| | NMT: encoder + one decoder step with ORT Android benchmark tool (tool availability to be verified and reported) | 3.0 |
| | TTS via sherpa-onnx | 1.0 |
| | k from these numbers, update proxy thresholds | 0.5 |
| | AI Hub run (if token) | 1.0 |
| | **D5** | **7.5** (−2.5) |
| **D6** | Thermal / airplane / network check on P1 setup | 1.0 |
| | Close espeak-ng license, final report, freeze | 3.0 |
| | Buffer (real) | 4.0 (target) |
| | **D6** | **8.0** (−2.0) |

## Totals
- D2-D6 = 12 + 14 + 13 + 7.5 + 8 (incl. 4 h buffer) = **54.5 h** vs cap 50 h: **over by 4.5 h**. Without the buffer it is 50.5 h.
- Lines over their 10 h day: D2 (+2.0), D3 (+4.0), D4 (+3.0). Largest lines: Branch B 5.0, MLP/LOSO 3.0, safety check 3.0, ADR-001 3.0, safety eval 3.0, freeze/fixes 3.0.
- Not cut by me, per instruction. To fit 50 h with the 4 h buffer, 4.5 h must come from those lines or the cap must rise.
- P2 (Kotlin app, est. ~11 h: NMT tokenizer 3 + decode loop 4 + ASR/TTS glue 2.5 + build/debug 1.5) cannot start under this plan: D5 has 2.5 h free, condition is >= 8 h.
- Port debt when P3 is cut: SwiftF0 glue 1.5 + Branch B stats/MLP 2.5 + safety/gate/glossary 3.0 = 7 h (PROGRESS.md).
- External dependencies: team recordings (D3 LOSO, in-domain WER), AI Hub token (hard deadline start of D5), SD712 + adb (k, D5). k is measured earlier if the device appears earlier.
