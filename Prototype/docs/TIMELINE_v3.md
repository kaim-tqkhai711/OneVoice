# Timeline v3 (laptop-first), 2026-10-06

All hours are my estimates, not measurements. Budget assumption (**owner to correct**): 60 h total, D1 consumed ~10 h, so D2-D6 = 5 days x 10 h = 50 h available.
Gate G-L (PROGRESS.md) sits at the end of D4; phone work starts only after it passes. k calibration (0.5-1 h) is outside the plan and runs the moment the SD712 is connected.

| Day | Item | Est. h |
|---|---|---|
| **D2** Branch A real, laptop | Zipformer VI INT8: file size + RTF (2 threads) first | 1.0 |
| | WER/CER harness (VI text normalization, jiwer, public VI test set to validate the harness until team recordings exist) | 2.5 |
| | opus-mt-vi-en: model-card tokenization check, ONNX export, INT8 with **arm64** config, bench under D1 conditions | 2.0 |
| | Glossary v0 (JSON, ~60 terms) + hook into NMT | 1.5 |
| | ADR-001: noise mixer (SNR 10/5/0), GTCRN ONNX, OA convex mix + sample-alignment test, WER table launched in background | 3.0 |
| | Piper EN via sherpa-onnx, first-audio latency | 1.0 |
| | Replace all stubs in pipeline, VI->EN wav->wav smoke run | 1.5 |
| | Docs: PROGRESS, port debt, pin hashes | 0.5 |
| | **D2 total** | **13.0** (**+3.0 over 10**) |
| **D3** Branch B + safety + gate | Branch B per `BRANCH_B_DESIGN.md` (after approval): features, `band_feats.onnx`, reference tests | 4.5 |
| | MLP + LOSO + z-score baseline (needs recordings) | 3.0 |
| | Semantic Safety Check (negation, drug, dose/unit, allergy, symptom; JSON lexicon) + seeded error set | 3.0 |
| | Fusion + gate, all 5 actions, ABSTAIN case, TTS-block case | 2.0 |
| | Portability/provenance tests (denoised input rejected, mixer alignment) | 1.0 |
| | **D3 total** | **13.5** (**+3.5 over 10**) |
| **D4** G-L + AI Hub prep | WER/CER x SNR grid, pass/fail per cell | 2.0 |
| | ADR-001 OFF/ON/OA table + conclusion by the 2-point rule | 1.0 |
| | Safety set (needs authoring ~2 h) + recall / false-block with Clopper-Pearson CI | 3.0 |
| | Urgency LOSO report | 0.5 |
| | E2E 100 utt (2 threads), peak RSS, per-stage latency, labelled "x86 proxy" | 1.5 |
| | Bug fixes + freeze candidate artifacts | 3.0 |
| | AI Hub prep: fixed-shape ONNX + `tools/aihub_submit.py` | 2.0 |
| | **D4 total** | **13.0** (**+3.0 over 10**) |
| **D5** Phone port | Prebuilt sherpa-onnx APK as base, Zipformer + Piper + VAD | 1.5 |
| | ORT-Android NMT: Kotlin SentencePiece (P1) + KV-cache decode loop (P2) | 7.0 |
| | Branch B Kotlin: SwiftF0 glue + stats + MLP (P4, P5), golden-vector tests | 3.5 |
| | Safety/gate/glossary Kotlin from JSON (P6) | 3.0 |
| | Stage 0 NS decision (P3: 3 h or drop) + resampling (P7) | 1.0 to 4.0 |
| | AI Hub run if token exists | 1.5 |
| | **D5 total** | **18.5 to 21.5** (**+8.5 to +11.5 over 10**) |
| **D6** Phone measure + freeze | 100-utt run, RSS, thermal, airplane + network check | 4.0 |
| | License close (espeak-ng), final report, freeze | 4.0 |
| | **D6 total** | **8.0** (2.0 under 10) |

## Totals and buffer (honest)
- Planned D2-D6: 13.0 + 13.5 + 13.0 + (18.5..21.5) + 8.0 = **66.0 to 69.0 h** against **50 h** available: **deficit 16 to 19 h**.
- D6 buffer: nominally 2 h, but D2-D5 carry a cumulative overrun, so the **real D6 buffer is negative**. The plan does not fit as written.
- Lines over their day: every day D2-D5; the largest single lines are D5 NMT port (7 h), D5 Branch B Kotlin (3.5 h), D3 Branch B (4.5 h).
- Cuts that would restore fit, for you to choose (I will not apply any without your decision): drop WebRTC Stage 0 and use GTCRN only (-3 h, P3); defer the AI Hub run if still no token (-1.5 h); reduce glossary v0 to ~30 terms (-0.75 h); drop the OA mixer arm of ADR-001 (-1 h, but ADR-001 asks for OFF/ON/OA); cut Branch B to ~14 features (-1 h); accept NMT greedy decode with no beam in Kotlin (already assumed). Even with all cuts the deficit is ~10 h, so a scope decision (e.g. phone demo of ASR+NMT+TTS only, safety/Branch B measured on laptop) is likely required.
- Dependencies outside my control: team recordings (D3 LOSO, D4 WER on real speech), AI Hub token (D5 hard deadline), SD712 + adb (k calibration, D5-D6).
