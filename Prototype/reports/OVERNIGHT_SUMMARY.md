# Overnight summary (session 2026-10-07 01:42 -> ~05:10; stopped early: every item A-G and M finished)

All laptop numbers are **x86 proxy**, not SD712. Nothing touched the phone; nothing was pushed to a remote. Tests: 81 passed (`pytest`). Single commit trail: `git log` on `main` (local only).

## 1. Status A-G, M
| Item | Status | Wall-clock (approx.) | Where |
|---|---|---|---|
| A1 total RSS | done | 5 min | `results/rss_total*.json`, `results/rss_breakdown.json` |
| A2 SD712 specs applied (RSS line 1.0 GB, no energy; temps instead) | done | 5 min | `PROGRESS.md`, `docs/DEVIATIONS.md`, `tools/device_session.sh` |
| A3 DEVIATIONS | done | 5 min | `docs/DEVIATIONS.md` (9 rows incl. RSS, energy) |
| A4 Timeline v5 (factor 0.10, caveated) | done | 5 min | `docs/TIMELINE_v5.md` |
| B ADR-001 re-check | done | code 25 min; grid 2 h 10 min in background; report 10 min | `docs/ADR-001-denoise.md`, `results/adr001_v2_dev_table.md`, `results/listen/` |
| C NMT quality + safety set | done | ~1 h | `reports/NMT_QUALITY.md`, `results/nmt30_table.md`, `configs/safety/`, `docs/nmt_ref_template.csv` |
| D latency | done | ~1 h 15 min | `results/latency_projection.md`, `results/latency_options.json`, `results/asr_chunked_wer_dev.json` |
| E Branch B | done (no MLP, no F1, by instruction) | ~40 min + ablation in background | `src/tonebridge/branch_b*.py`, `tests/test_branch_b.py`, `results/branchb_ablation_table.md`, `docs/BRANCH_B_DESIGN.md` §8 |
| F Safety check v1 + fusion + gate | done | ~45 min | `src/tonebridge/safety.py`, `gate.py`, `tests/test_safety.py`, `tests/test_gate.py`, `tools/eval_safety.py` |
| G phone session prep | done (dry-run only) | ~20 min | `tools/device_session.sh`, `docs/DEVICE_SESSION.md` |
| M optimisation loop | done (M2, M3 reached >= 70 % in 1 round each; M1 stopped by the 2-round rule below 70 %) | M2/M3 ~10 min, M1 ~1 h | `results/optim_log.jsonl`, `reports/OVERNIGHT_RESEARCH.md`, `results/LOCKED_CONFIG.json`, final test 23 min |
Full pipeline with **0 stubs** runs wav -> wav: `python -m tonebridge.cli --wav X --out Y --full` (smoke: `results/full_smoke.jsonl`).

## 2. M1-M3 (stop line 70 % is a loop stop, not a claim; Proposal targets reported separately)
Single test run, locked config (`results/LOCKED_CONFIG.json`), done once (`results/FINAL_TEST_DONE`).

| Metric | dev before | dev after | test | >= 70 %? | Proposal target |
|---|---|---|---|---|---|
| **M1** ASR word accuracy per cell (OFF, greedy). Worst cell | babble -5 dB: 5.9 % | unchanged (no change kept) | babble -5 dB: 6.9 % | **No**: 3 of 13 cells below on dev AND test: demand -5 dB (dev 69.5 / test 61.6), babble 0 dB (46.1 / 39.6), babble -5 dB (5.9 / 6.9). Other 10 cells 73.7-90.8 | WER <= 15/20/25/35 % at clean/10/5/0: dev all pass except babble 0 dB (53.9); **test fails** clean (19.2), babble 10 (20.2), babble 5 (26.3), babble 0 (60.4); demand/alarm pass at 10/5/0 |
| **M2** NMT slot preservation, worst group | 57.9 % (severity; dose 58.7, medication 58.0) | 97.3 % (dose; med 98.0, severity 100, negation 99.4, allergy 100) | 90.7 % (dose [84.8, 94.8]); med 93.3, severity 97.4, negation 98.8, allergy 100 | Yes | No numeric target in the Proposal text |
| **M3** safety-check recall on the shipping pipeline's real NMT errors, with false-block <= 10 % | 60.0 % (n=45) | 83.9 % [66.3, 94.5] (n=31), false-block 0.0 % [0, 0.6] (n=581) | **90.7 % [77.9, 97.4] (n=43)**, false-block 3.7 % [2.3, 5.6] (n=569); dose group false-block 16.8 % (bug, below) | Yes (pooled) | No numeric target in the Proposal text |
Test details: M1 per cell in `docs/ADR-001-denoise.md`; M2/M3 in `logs/final_test_m2.log`, `results/safety_eval_test_final.json`.
Caveats that matter more than the numbers:
- **M2/M3 are in-lexicon upper bounds.** One author wrote the templates, the gold forms and the check lexicon, and the decoder constraint forces exactly those forms. They show "the required word is present", not that the translation is good (fluency/chrF unmeasured; owner reference file pending). The 50-58 % allergy "misses" are gold-list artefacts ("eggs" vs gold form "egg"), not unsafe outputs (D-8).
- **M3 test false-block is inflated by a lexicon bug I found after the test**: "không uống quá N" (a limit) was read as the intensifier "quá" -> 21 correct dose outputs got CONFIRM. Fixed as safety v1.2 *after* the run; the reported test number is v1.1. Post-hoc on the same test outputs (labelled, not the reported number): false-block 0.0 % [0, 0.6], recall unchanged (`results/safety_eval_test_v12_POSTHOC.json`). Seeded-error recall on test 97.7 % [96.7, 98.4] (n=1279; unit_dropped 83.3 %).
- M1: GTCRN/OA/classical NS/ASR beam search/chunked decoding did not help babble (dev: 53.9 % WER OFF; best alternative 54.1). Likely cause: babble = 6 Vietnamese talkers, same band and language as the target, at SNR <= 0 dB over speech frames; a single-channel front end cannot separate them and the ASR transcribes the babble. Two next directions: (a) multi-condition/noise-augmented fine-tuning of the Zipformer (needs GPU time and licensed data, 1-3 days) or a near-field/directional mic front end (hardware, not costed); (b) a **reliable "conditions too bad" detector** feeding REPEAT (see §7 item 3), ~0.5-1 day.

## 3. ADR-001 final (dev, WER %, SNR over speech mask; Delta = OFF - arm, rule >= 2 pts)
Table: `results/adr001_v2_dev_table.md` (13 cells x 6 arms, paired-bootstrap CIs). **No arm beats OFF by >= 2 points in any cell** (0 of 13). GTCRN ON is worse everywhere (clean +1.7, 0 dB demand +6.0, babble 0 dB +8.5, demand -5 dB +10.4 points WER). OA(0.25/0.5/0.75) and classical NS are within ~1 point (best +0.47 [-0.03, +0.97] at demand -5 dB, beta 0.25). **Conclusion: default denoise = OFF.** No dual-path WER gain is claimed. Limits: DEMAND/FLEURS-babble/synthetic alarm only, no hospital recordings, alarm barely degrades this ASR, read speech, 200 dev utterances.
Branch B ablation (raw vs GTCRN, dev n=100): median F0 differs by only 1.6-2.7 cents but GTCRN removes 7-16 points of voiced_frac and p90 deviation hits ~3400 cents under alarm; raw voiced_frac is inflated by babble (0.62-0.69) (`results/branchb_ablation_table.md`).

## 4. Latency projection and the two reduction options
Source: quiet x86 run on 20 FLEURS dev crops per bucket (`results/latency_options.json`); table `results/latency_projection.md`; label **"x86 proxy x k (est.)"**; p95 column is a conservative bound (sum of stage p95).
| speech | baseline p50 (k=3 / 4) | both options p50 (k=3 / 4) | p95 bound baseline / both (k=3) |
|---|---|---|---|
| 2 s | 1313 / 1751 ms | 1288 / 1718 | 2164 / 1980 |
| 4 s | 2490 / 3320 | 1831 / 2441 | 3406 / 3157 |
| 6 s | 3304 / 4406 | 2587 / 3449 | 4923 / 4730 |
vs Proposal 1.5 s p50 / 2.0 s p95: **only the 2 s utterance at k=3 passes p50**; 4 s and 6 s fail at either k even with both options. TTS first-audio dominates (x86 p50 288 / 558 / 707 ms for 2/4/6 s; ASR 89-233 ms; NMT 60-161 ms).
- ASR per-VAD-segment decode: tail p50 70 / 115 / 136 ms vs 89 / 161 / 233 for the full decode, but **WER 11.85 % vs 9.42 %: +2.42 points worse** (paired CI [1.75, 3.20], 200 dev utts, 126 with > 1 segment). Not adopted.
- TTS first clause: saves 0 ms (2 s, one clause), ~170 ms (4 s), ~140 ms (6 s) at p50. The smaller verified-license voice (ljspeech-medium-int8, 19 MB) is **3x slower on x86** (772 / 1438 / 2080 ms); kathleen/ryan/danny low were rejected on license (NC parent). ARM behaviour unmeasured.
k is still an estimate (3-4) until session 1 on the phone.

## 5. RSS against 1.0 GB (x86 proxy, ASR + NMT + TTS + VAD resident, 2 threads)
Peak **1021 MB** after 20 FLEURS utterances (mean ~12 s) = **over 1.0 GB**; 976-982 MB with GTCRN loaded; **760 MB with utterances <= 7 s** (PTT-like). Growth is activation arenas: ASR 195 -> 399 MB, TTS 162 -> 378 MB (alone), NMT flat (`results/rss_breakdown.json`). Phone RSS not measured.

## 6. Decisions I made that need your review (`reports/OVERNIGHT_DECISIONS.md`, D-1..D-15)
D-4 M2 = worst group; D-5 false-block counts CONFIRM; D-8 gold lists not edited after seeing results (slightly pessimistic for medication/allergy); D-9 circularity of the safety set; D-11 INT8 not pursued further (no systematic harm; hybrid worse); D-12 M3 defined on the shipping pipeline's errors; D-13 brand names removed from the lexicon; D-15 post-test safety fix; D-6 21 features (median-relative F0); D-7 gate semantics; D-1/D-2 split sizes and synthetic noise.

## 7. What I need from you (max 5, by how much it blocks)
1. **Team recordings** (`recording_kit/`): without them there is no urgency MLP, no LOSO F1, no tuned UNKNOWN thresholds, no in-domain WER.
2. **Gate signal for noisy input is broken**: ASR confidence barely separates wrong from right transcripts (babble 0 dB: mean 0.79 wrong vs 0.84 right; the 0.5 REPEAT rule catches 0 % of bad utterances; `results/asr_conf_gate_check_dev.json`); a blind SNR estimate catches only 17.5 % of bad utterances at 10 % false reject (AUC 0.75, `results/snr_gate_probe_dev.json`). Decide whether to spend ~1 day on a better quality signal (n-best disagreement / learned estimator) before any claim about noisy-condition safety.
3. **ORT Android benchmark tool is not published prebuilt**; session 2 (NMT on phone) needs `onnxruntime_perf_test` cross-compiled with the NDK (est. 1-2 h) or the P2 app. Session 1 (k) can run any time with the termux-static sherpa-onnx binary (first real test of whether it runs under `adb shell`).
4. **Latency**: projected E2E misses 1.5 s for utterances >= 4 s even with both options, TTS is the bottleneck; tell me whether to pursue sentence-level streaming TTS / a faster voice, or revise the claim after k is measured.
5. Review D-8/D-9 (reference set: fill `docs/nmt_ref_template.csv`) and decide the espeak-ng GPL question (R2) and the AI Hub token (R1, hard deadline start of D5).
