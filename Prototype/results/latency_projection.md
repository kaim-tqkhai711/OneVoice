### Projected E2E (ms), x86 proxy x k (est.), vs Proposal p50 <= 1500 / p95 < 2000

| speech s | config | k | p50 (est.) | p95 bound (est.) | vs 1.5 s | vs 2.0 s |
|---|---|---|---|---|---|---|
| 2 | baseline (full ASR decode, full TTS) | 3 | 1313 | 2164 | PASS | FAIL |
| 2 | baseline (full ASR decode, full TTS) | 4 | 1751 | 2885 | FAIL | FAIL |
| 2 | ASR tail only | 3 | 1256 | 2081 | PASS | FAIL |
| 2 | ASR tail only | 4 | 1674 | 2775 | FAIL | FAIL |
| 2 | TTS first clause only | 3 | 1346 | 2062 | PASS | FAIL |
| 2 | TTS first clause only | 4 | 1794 | 2750 | FAIL | FAIL |
| 2 | both | 3 | 1288 | 1980 | PASS | PASS |
| 2 | both | 4 | 1718 | 2640 | FAIL | FAIL |
| 2 | both + int8 voice | 3 | 2756 | 3772 | FAIL | FAIL |
| 2 | both + int8 voice | 4 | 3675 | 5029 | FAIL | FAIL |
| 4 | baseline (full ASR decode, full TTS) | 3 | 2490 | 3406 | FAIL | FAIL |
| 4 | baseline (full ASR decode, full TTS) | 4 | 3320 | 4541 | FAIL | FAIL |
| 4 | ASR tail only | 3 | 2352 | 3361 | FAIL | FAIL |
| 4 | ASR tail only | 4 | 3136 | 4482 | FAIL | FAIL |
| 4 | TTS first clause only | 3 | 1969 | 3202 | FAIL | FAIL |
| 4 | TTS first clause only | 4 | 2625 | 4269 | FAIL | FAIL |
| 4 | both | 3 | 1831 | 3157 | FAIL | FAIL |
| 4 | both | 4 | 2441 | 4210 | FAIL | FAIL |
| 4 | both + int8 voice | 3 | 4246 | 7269 | FAIL | FAIL |
| 4 | both + int8 voice | 4 | 5661 | 9692 | FAIL | FAIL |
| 6 | baseline (full ASR decode, full TTS) | 3 | 3304 | 4923 | FAIL | FAIL |
| 6 | baseline (full ASR decode, full TTS) | 4 | 4406 | 6564 | FAIL | FAIL |
| 6 | ASR tail only | 3 | 3013 | 4860 | FAIL | FAIL |
| 6 | ASR tail only | 4 | 4018 | 6480 | FAIL | FAIL |
| 6 | TTS first clause only | 3 | 2878 | 4793 | FAIL | FAIL |
| 6 | TTS first clause only | 4 | 3837 | 6390 | FAIL | FAIL |
| 6 | both | 3 | 2587 | 4730 | FAIL | FAIL |
| 6 | both | 4 | 3449 | 6306 | FAIL | FAIL |
| 6 | both + int8 voice | 3 | 5654 | 11553 | FAIL | FAIL |
| 6 | both + int8 voice | 4 | 7538 | 15404 | FAIL | FAIL |