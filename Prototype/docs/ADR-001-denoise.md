# ADR-001 (final, dev): Branch A front-end = denoise OFF

Status: **Accepted on dev evidence, 2026-10-07 (overnight).** Supersedes the 2026-10-06 dev subset run (whole-file SNR, 200 utterances, DEMAND only), which used the wrong SNR definition.

## Protocol
- Data: FLEURS vi_vn, CC-BY-4.0. Fixed split by seed 20261007: dev 200 utterances (this table), test 300 (touched once, OFF arm only, see below), 357 babble-pool utterances (never evaluated). ID lists: `configs/splits/fleurs_vi_dev_test.json`.
- SNR is computed over the speech mask (Silero VAD span of the clean clip), noise power over the same mask (`evalkit/noise.mix_at_snr_masked`). The mix is exact by construction; test `tests/test_noise_v2.py` (<= 0.5 dB, synthetic and real clips) and `results/listen/` (clean/10/5/0/-5 for each noise type). The achieved-SNR column in the JSON files uses a projection estimator that is up to 1.6 dB off per utterance for babble (D-14).
- Noise types: DEMAND (hall, office, cafeteria, station; CC-BY-4.0, round-robin), babble (6 FLEURS talkers summed, seeded), alarm/beep (seeded synthetic). **Not recorded hospital noise.**
- Arms: OFF; GTCRN ON; OA(beta) beta in {0.25, 0.5, 0.75}; classical NS (numpy minimum-statistics Wiener, `stages/denoise_classic.py`, 1 h timebox, laptop only). ASR: shipped Zipformer VI INT8 files, greedy, 2 threads.
- Rule: an arm replaces OFF in a cell only if WER(OFF) - WER(arm) >= 2.0 points absolute. Paired bootstrap 95 % CI shown.

## Result (dev; WER %)
**demand**

| SNR | off | on | oa0.25 | oa0.5 | oa0.75 | cns | arm that beats OFF by >= 2 pts |
|---|---|---|---|---|---|---|---|
| clean | 9.42 | 11.09 (-1.67 [-2.20,-1.12]) | 9.37 (+0.05 [-0.07,+0.17]) | 9.29 (+0.13 [-0.05,+0.33]) | 9.37 (+0.05 [-0.13,+0.24]) | 9.17 (+0.25 [+0.05,+0.46]) | none (OFF) |
| 10 | 9.26 | 11.09 (-1.83 [-2.43,-1.31]) | 9.41 (-0.15 [-0.27,-0.05]) | 9.34 (-0.08 [-0.23,+0.07]) | 9.54 (-0.28 [-0.54,-0.05]) | 9.41 (-0.15 [-0.39,+0.11]) | none (OFF) |
| 5 | 10.22 | 12.70 (-2.48 [-3.32,-1.65]) | 10.26 (-0.03 [-0.21,+0.13]) | 10.26 (-0.03 [-0.28,+0.18]) | 10.69 (-0.47 [-1.03,-0.02]) | 10.46 (-0.23 [-0.67,+0.10]) | none (OFF) |
| 0 | 13.02 | 19.06 (-6.04 [-7.52,-4.72]) | 12.99 (+0.03 [-0.21,+0.26]) | 13.32 (-0.30 [-0.61,-0.03]) | 13.82 (-0.80 [-1.27,-0.38]) | 13.24 (-0.22 [-0.57,+0.15]) | none (OFF) |
| -5 | 30.54 | 40.89 (-10.36 [-12.44,-8.47]) | 30.07 (+0.47 [-0.03,+0.97]) | 30.12 (+0.42 [-0.29,+1.15]) | 31.49 (-0.95 [-1.86,-0.12]) | 31.60 (-1.07 [-1.87,-0.25]) | none (OFF) |

**babble**

| SNR | off | on | oa0.25 | oa0.5 | oa0.75 | cns | arm that beats OFF by >= 2 pts |
|---|---|---|---|---|---|---|---|
| 10 | 10.27 | 12.05 (-1.78 [-2.45,-1.14]) | 10.09 (+0.18 [-0.03,+0.42]) | 10.17 (+0.10 [-0.16,+0.37]) | 10.42 (-0.15 [-0.52,+0.25]) | 10.39 (-0.12 [-0.41,+0.22]) | none (OFF) |
| 5 | 17.37 | 21.21 (-3.85 [-4.91,-2.73]) | 17.53 (-0.17 [-0.64,+0.30]) | 17.95 (-0.58 [-1.18,-0.05]) | 18.66 (-1.30 [-2.05,-0.65]) | 17.82 (-0.45 [-1.23,+0.26]) | none (OFF) |
| 0 | 53.86 | 62.39 (-8.52 [-10.48,-6.64]) | 54.13 (-0.27 [-0.96,+0.39]) | 54.98 (-1.12 [-2.03,-0.19]) | 57.03 (-3.16 [-4.45,-1.92]) | 54.28 (-0.42 [-1.56,+0.74]) | none (OFF) |
| -5 | 94.12 | 94.47 (-0.35 [-1.60,+0.88]) | 93.89 (+0.23 [-0.30,+0.75]) | 94.24 (-0.12 [-0.81,+0.51]) | 94.01 (+0.12 [-0.79,+1.08]) | 94.72 (-0.60 [-1.66,+0.47]) | none (OFF) |

**alarm**

| SNR | off | on | oa0.25 | oa0.5 | oa0.75 | cns | arm that beats OFF by >= 2 pts |
|---|---|---|---|---|---|---|---|
| 10 | 9.46 | 10.92 (-1.47 [-2.08,-0.92]) | 9.37 (+0.08 [-0.07,+0.29]) | 9.42 (+0.03 [-0.11,+0.18]) | 9.46 (+0.00 [-0.17,+0.20]) | 9.39 (+0.07 [-0.13,+0.28]) | none (OFF) |
| 5 | 9.39 | 10.89 (-1.50 [-2.10,-0.97]) | 9.41 (-0.02 [-0.10,+0.07]) | 9.47 (-0.08 [-0.20,+0.03]) | 9.44 (-0.05 [-0.25,+0.15]) | 9.32 (+0.07 [-0.07,+0.21]) | none (OFF) |
| 0 | 9.47 | 10.89 (-1.42 [-2.03,-0.89]) | 9.46 (+0.02 [-0.10,+0.15]) | 9.46 (+0.02 [-0.18,+0.21]) | 9.54 (-0.07 [-0.33,+0.21]) | 9.32 (+0.15 [-0.03,+0.34]) | none (OFF) |
| -5 | 9.36 | 10.62 (-1.27 [-1.76,-0.78]) | 9.37 (-0.02 [-0.16,+0.13]) | 9.44 (-0.08 [-0.26,+0.10]) | 9.47 (-0.12 [-0.33,+0.07]) | 9.44 (-0.08 [-0.29,+0.08]) | none (OFF) |

Cells compared: 13. Cells where an arm beat OFF by >= 2 points: {'on': 0, 'oa0.25': 0, 'oa0.5': 0, 'oa0.75': 0, 'cns': 0}
Achieved SNR check (over mask): {"alarm|10": {"target": 10.0, "mean": 10.0, "max_abs_dev": 0.032}, "alarm|5": {"target": 5.0, "mean": 5.0, "max_abs_dev": 0.045}, "alarm|0": {"target": 0.0, "mean": -0.001, "max_abs_dev": 0.044}, "alarm|-5": {"target": -5.0, "mean": -4.999, "max_abs_dev": 0.066}, "babble|10": {"target": 10.0, "mean": 9.998, "max_abs_dev": 0.254}, "babble|5": {"target": 5.0, "mean": 4.992, "max_abs_dev": 0.82}, "babble|0": {"target": 0.0, "mean": 0.01, "max_abs_dev": 1.37}, "babble|-5": {"target": -5.0, "mean": -5.02, "max_abs_dev": 1.639}, "demand|10": {"target": 10.0, "mean": 10.0, "max_abs_dev": 0.111}, "demand|5": {"target": 5.0, "mean": 4.997, "max_abs_dev": 0.119}, "demand|0": {"target": 0.0, "mean": -0.004, "max_abs_dev": 0.593}, "demand|-5": {"target": -5.0, "mean": -4.994, "max_abs_dev": 0.328}}

## Verdict
No arm beats OFF by >= 2 points in any of the 13 cells. GTCRN ON is significantly worse everywhere (up to +10.4 points at demand -5 dB, +8.5 at babble 0 dB). OA and classical NS are within about 1 point of OFF (mostly slightly worse; best case +0.47 [-0.03, +0.97] at demand -5 dB for beta 0.25, below the 2-point rule). **Default denoise = OFF.** No dual-path WER improvement is claimed anywhere; the dual-path front end remains a requirement for Branch B isolation (never feed denoised audio), not a WER gain.
Why the earlier D2 table is not the final one: it used whole-file SNR, DEMAND only, 200 utterances from the test list.

## What the grid cannot say
Alarm/beep noise barely hurts this ASR (the SNR is defined over speech frames and beeps are sparse), so the alarm column is weak evidence. Babble at <= 0 dB destroys recognition with every arm (see summary M1). Real hospital noise, reverberation and the phone microphone are untested.

## Test split (one run, OFF arm only, 300 utterances; `results/final_test_m1_*.json`)
demand clean/10/5/0/-5 dB: 19.21 / 19.22 / 19.63 / 22.16 / 38.41; babble 10/5/0/-5: 20.20 / 26.28 / 60.36 / 93.08; alarm 10/5/0/-5: 19.22 / 19.23 / 19.30 / 19.34. The test subset is harder than dev even in clean speech (19.2 % [12.6, 26.0] vs 9.4 % [8.0, 11.0]); 7 of 300 utterances with WER > 60 % carry half of the clean-condition errors.
