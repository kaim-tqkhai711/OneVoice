### ADR-001 v2 (dev); WER %, SNR over the VAD speech mask. Delta = WER(off) - WER(arm), positive = arm better; [95% paired bootstrap]. WIN = Delta >= 2.0 points.

**demand**

| SNR | off | on | oa0.25 | oa0.5 | oa0.75 | cns | arm that beats OFF by >= 2 pts |
|---|---|---|---|---|---|---|---|
| clean | 9.42 | 11.09 (-1.67 [-2.20,-1.12]) | 9.37 (+0.05 [-0.07,+0.17]) | 9.29 (+0.13 [-0.05,+0.33]) | 9.37 (+0.05 [-0.13,+0.24]) | 9.17 (+0.25 [+0.05,+0.46]) | none (OFF) |
| 10 | 9.26 | - | - | - | - | - | none (OFF) |
| 5 | (not run) |
| 0 | (not run) |
| -5 | (not run) |

**babble**

| SNR | off | on | oa0.25 | oa0.5 | oa0.75 | cns | arm that beats OFF by >= 2 pts |
|---|---|---|---|---|---|---|---|
| 10 | 10.27 | 12.05 (-1.78 [-2.46,-1.15]) | 10.09 (+0.18 [-0.02,+0.41]) | 10.17 (+0.10 [-0.15,+0.36]) | 10.42 (-0.15 [-0.49,+0.23]) | 10.39 (-0.12 [-0.45,+0.20]) | none (OFF) |
| 5 | (not run) |
| 0 | (not run) |
| -5 | (not run) |

**alarm**

| SNR | off | on | oa0.25 | oa0.5 | oa0.75 | cns | arm that beats OFF by >= 2 pts |
|---|---|---|---|---|---|---|---|
| 10 | 9.46 | 10.92 (-1.47 [-2.09,-0.93]) | 9.37 (+0.08 [-0.07,+0.26]) | 9.42 (+0.03 [-0.11,+0.19]) | 9.46 (+0.00 [-0.18,+0.19]) | 9.39 (+0.07 [-0.13,+0.30]) | none (OFF) |
| 5 | (not run) |
| 0 | (not run) |
| -5 | (not run) |

Cells compared: 4. Cells where an arm beat OFF by >= 2 points: {'on': 0, 'oa0.25': 0, 'oa0.5': 0, 'oa0.75': 0, 'cns': 0}
Achieved SNR check (over mask): {"alarm|10": {"target": 10.0, "mean": 10.0, "max_abs_dev": 0.032}, "babble|10": {"target": 10.0, "mean": 9.998, "max_abs_dev": 0.254}, "demand|10": {"target": 10.0, "mean": 10.0, "max_abs_dev": 0.111}}