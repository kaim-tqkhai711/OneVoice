### Branch B ablation: raw noisy input vs GTCRN output (dev, n=100 utterances, SwiftF0). Cents = |1200*log2(f_a/f_b)| on frames voiced in both.

| noise | SNR | median cents raw vs denoised | p90 cents raw vs denoised | median cents raw vs clean | median cents denoised vs clean | voiced_frac raw | voiced_frac denoised | change (denoised - raw) |
|---|---|---|---|---|---|---|---|---|
| demand | 10 | 1.8 | 6.5 | 2.1 | 2.6 | 0.464 | 0.398 | -0.066 |
| demand | 5 | 2.0 | 7.5 | 3.0 | 3.2 | 0.455 | 0.376 | -0.079 |
| demand | 0 | 2.4 | 8.7 | 4.1 | 4.0 | 0.433 | 0.333 | -0.099 |
| demand | -5 | 2.7 | 9.8 | 5.0 | 4.3 | 0.388 | 0.275 | -0.113 |
| babble | 10 | 1.9 | 7.0 | 3.9 | 3.9 | 0.616 | 0.457 | -0.159 |
| babble | 5 | 2.1 | 7.8 | 5.8 | 5.7 | 0.642 | 0.49 | -0.152 |
| babble | 0 | 2.4 | 9.2 | 10.2 | 9.6 | 0.659 | 0.499 | -0.160 |
| babble | -5 | 2.5 | 9.8 | 49.7 | 55.1 | 0.687 | 0.529 | -0.158 |
| alarm | 10 | 1.6 | 1214.9 | 0.4 | 1.5 | 0.521 | 0.418 | -0.103 |
| alarm | 5 | 1.7 | 3281.5 | 0.7 | 1.5 | 0.522 | 0.417 | -0.104 |
| alarm | 0 | 1.7 | 3483.6 | 1.0 | 1.6 | 0.524 | 0.416 | -0.108 |
| alarm | -5 | 1.8 | 3452.2 | 1.5 | 1.8 | 0.525 | 0.413 | -0.111 |