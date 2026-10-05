# Branch B design: vocal urgency (for owner approval; no code written yet)

Status: **Approved 2026-10-06** (direction, 22 features, tolerance table); amendments below (section 5b, 6) added at owner request. Hour figures are estimates. Conforms to the portability law (PROGRESS.md).

## 1. Input contract
- Input: the PTT segment after capture -> one 16 kHz conversion -> VAD slice only. `wav [T_seg] float32 @ 16 kHz`, `T_seg <= 15 s` (config `max_utterance_s`), so SwiftF0 needs exactly one window (937 frames < its 1875-frame window) and its multi-window stitching is never exercised (port debt P4 shrinks).
- Branch B never receives denoiser output. Enforcement: `AudioBuffer` gets a `provenance` tag (`"raw16k"` | `"denoised"`); `BranchB.analyze` raises on `"denoised"`. Test: feeding a tagged-denoised buffer raises; feeding the pipeline's Branch-B tap returns the same array hash as the pre-denoiser segment. This test is part of G-L.

## 2. Features actually used: 22 (subset of eGeMAPS-style descriptors, not all 88)
Frame grid = SwiftF0 grid: 256-sample hop (16 ms). Voiced frame = SwiftF0 confidence >= 0.5. Statistics over voiced frames unless noted.

| # | Feature | Source | Computation |
|---|---|---|---|
| 1 | f0_mean_st | SwiftF0 pitch | mean of 12·log2(f0/27.5) |
| 2 | f0_std_norm | SwiftF0 pitch | std(f0 Hz) / mean(f0 Hz) |
| 3-5 | f0_p20_st, f0_p50_st, f0_p80_st | SwiftF0 pitch | percentiles (linear interpolation) of semitone values |
| 6 | f0_range_st | derived | p80 − p20 |
| 7-8 | f0_slope_rise, f0_slope_fall | SwiftF0 pitch | mean slope (st/s) of consecutive voiced-frame pairs, rising / falling separately |
| 9 | voiced_frac | SwiftF0 conf | voiced frames / all frames |
| 10 | voiced_seg_per_s | SwiftF0 conf | voiced runs per second of segment |
| 11-12 | voiced_len_mean_s, unvoiced_len_mean_s | SwiftF0 conf | mean run length (unvoiced = gaps between first and last voiced frame) |
| 13 | voicing_quality | SwiftF0 conf | mean confidence over voiced frames. Also feeds `UrgencyResult.voicing_quality` and the UNKNOWN rule |
| 14-18 | loud_mean, loud_std_norm, loud_p20, loud_p80, loud_range | loudness_db | `20·log10(sqrt((P[i-1]+P[i])/512))`, P[i] = sum of squares of hop i (SwiftF0's own definition) |
| 19 | loud_rise_slope | loudness_db | mean positive slope (dB/s) |
| 20 | alpha_ratio_db | spectral graph | mean over voiced frames of 10·log10(E[50-1000 Hz] / E[1-5 kHz]) |
| 21 | hammarberg_db | spectral graph | mean of 10·log10(Pmax[0-2 kHz] / Pmax[2-5 kHz]) |
| 22 | tilt_db | spectral graph | mean of 10·log10(E[0-500 Hz] / E[500-1500 Hz]) |

Not used (reason): jitter, shimmer, HNR, formants/bandwidths, MFCC, H1-H2, F1-F3 amplitudes (need pitch-synchronous analysis or LPC; no compact ONNX/Kotlin form; each would be port debt of several hours with unproven benefit). Final subset may shrink after LOSO ablation; it may not grow without a new port-cost line.

## 3. Computation and export
- F0 / voicing: SwiftF0 ONNX (`model.onnx`, 135 KB, opset 18, input raw `audio [1,samples]`, outputs `pitch`, `confidence`). Verified: it contains its own STFT (Sin/Cos/MatMul), and `swift_f0/core.py` depends only on numpy + onnxruntime. Pre/post glue (silence-peak gate `max|x| < 1e-3 -> conf 0`, loudness) is ~15 lines of numpy -> Kotlin.
- Spectral features: one small ONNX graph `band_feats.onnx` built with `onnx.helper` (same technique SwiftF0 uses, no torch in the artifact): `Pad(256 left zeros) -> Conv1d(kernel [514,1,512] = Hann·cos / −Hann·sin DFT basis, stride 256) -> re²+im² -> MatMul(band mask [257,4]) and ReduceMax(masked) -> out [N,6]` = E_50-1000, E_1000-5000, E_0-500, E_500-1500, Pmax_0-2000, Pmax_2000-5000. Frame i covers samples [256(i−1), 256(i+1)), the same alignment as the loudness window.
- Utterance statistics (percentiles, slopes, run lengths, dB ratios, z-score standardization) = pure logic over arrays; constants (band edges, thresholds, standardizer mean/std, MLP weights) read from `configs/branch_b.json`.
- MLP: 22 -> 32 -> 16 -> K (target < 1 MB); trained with torch, exported to ONNX (ORT runs it; torch not in the inference path).
- Fallback if the graph does not export/run on ORT-Android: the six band outputs are 40 lines of Kotlin (windowed DFT on 257 bins); cost 2 h, recorded in port debt P5.

## 4. Equivalence tests vs reference (torch/librosa allowed here only)
Reference: `librosa.stft(center=False, window="hann" periodic, n_fft=512, hop=256)` on the same left-padded signal, float64. Proposed tolerances (owner to confirm or change):

| Check | Tolerance |
|---|---|
| `band_feats.onnx` outputs vs librosa band sums, bins with power > 1e-10 | relative error <= 1e-4 |
| Same, in dB, frames above −80 dBFS | absolute <= 0.01 dB |
| Features 20-22 (graph + stats) vs full librosa/numpy reference | absolute <= 0.02 dB |
| Features 1-19 (numpy reference stats vs the shipped implementation, identical SwiftF0 outputs) | absolute <= 1e-4 (relative for Hz/dB values) |
| SwiftF0 sanity vs known tones (synthetic harmonic sweep 100-400 Hz) | median error <= 1 % |
| Golden vectors: Python writes `golden/branch_b_*.json` (20 clips: audio hash, 22 features); the future Kotlin test must reproduce them within the same tolerances | as above |

Also tested: silence -> UNKNOWN; denoised-provenance input rejected; utterance < 0.3 s or voiced_frac < config minimum -> UNKNOWN.

## 5. Model, validation, baseline
- Data: team recordings (spk01-spk06), instructed urgency labels, noise augmentation by SNR mixer (clean/10/5/0), augmentation applied only inside training folds.
- Validation: leave-one-speaker-out (6 folds). Standardizer fit on the training folds only (no leakage). Report mean ± std of Macro-F1 and high-urgency recall over folds.
- Baseline: per-speaker z-score of (loud_mean + f0_mean_st) with a threshold set on training folds. The MLP is reported against it, including when it loses. Result depends on the team's recordings; with 6 speakers the std will be large and is reported as is.
- UNKNOWN: voicing_quality < `voicing_quality_min` (0.5 in config) or voiced_frac too low, applied by the gate, not forced by the model.

## 5b. Normalization and UNKNOWN rules (owner-requested)
Normalization (all constants live in `configs/branch_b.json`; none computed at run time except the enrolment mean):
- **Default mode G (population z-score), used when the speaker has one utterance or none enrolled:** `z = (x − mu_pop) / sd_pop` with `mu_pop, sd_pop` = mean/std of each feature over the *training speakers only* (inside LOSO: the 5 training speakers; for the shipped model: all 6). A brand-new speaker's single utterance therefore uses only population statistics; nothing is taken from that speaker.
- **Optional mode S (enrolled), only when >= 5 earlier utterances of the same speaker exist (labels not used):** `z = (x − mu_spk) / sd_pop`, `mu_spk` = mean of that speaker's earlier utterances. `sd_pop` stays population-wide (5 utterances cannot give a stable std).
- LOSO reports both G and S (S uses 5 randomly chosen utterances of the held-out speaker, seed in config, excluded from that speaker's test set). Shipped default = G; S is shipped only if it beats G on LOSO mean by more than the std across folds.
- z-score baseline uses the same mode as the arm it is compared with.

UNKNOWN is returned (by the gate rule, before the MLP is consulted) when any of these holds. Thresholds are set a priori and checked against the recordings once; any change is logged in PROGRESS.md:
| Condition | Threshold |
|---|---|
| segment duration | < 0.3 s |
| voiced frames | < 20 (about 0.32 s) |
| voiced_frac (feature 9) | < 0.15 |
| voicing_quality (feature 13, mean F0 confidence over voiced frames) | < 0.70 |
| SwiftF0 confidence | frame voiced iff conf >= 0.5 (model-card calibration) |
| silence gate | `max|x| < 1e-3` on the whole segment |
`configs/pipeline.json` `voicing_quality_min` (currently 0.5) is replaced by these keys when the code lands.

## 6. Classes (resolved)
Proposal §4.2 module table, row "Vocal urgency", verbatim: "3-level advisory posterior + UNKNOWN via gate". That is three labelled levels plus UNKNOWN. The recording kit has two labels (NEUTRAL, URGENT) and no middle level is added to it. Therefore: **deviation #4** (PROGRESS.md): the prototype outputs LOW / HIGH + UNKNOWN. Nothing is invented for the middle level.

## 7. Estimate
Features + graph + reference tests: 4.5 h. MLP + LOSO + baseline: 3 h (blocked on recordings; synthetic placeholder only for pipeline plumbing). Included in `docs/TIMELINE_v3.md` D3.
