# Timeline v5, 2026-10-07 (v4 re-timed with the measured estimate/actual factor; scope unchanged, nothing cut)

## Factor
D2 plan (v4 items, owner's figure): 10 h. Actual wall-clock: ~1 h of session time (compute was background). Ratio actual/estimate = **0.10**.
Caveat (stated, not hidden): the ratio holds for agent-coded + compute-bound items. It does NOT apply to items whose duration is set by people or hardware: team recordings, phone sessions, AI Hub token, owner review. Those keep factor 1.0 below. One sample (D2) only; treat 0.10 as optimistic and widen with the 0.25 column.

| Day | Item | v4 est. h | x0.10 | x0.25 | Scalable? |
|---|---|---|---|---|---|
| D3 | Branch B code + equivalence tests | 5.0 | 0.5 | 1.25 | yes (done overnight) |
| D3 | MLP + LOSO + baseline | 3.0 | 0.3 | 0.75 | code yes; **blocked on recordings** |
| D3 | Safety check + seeded error set | 3.0 | 0.3 | 0.75 | yes (done overnight) |
| D3 | Fusion + gate | 2.0 | 0.2 | 0.5 | yes (done overnight) |
| D3 | Provenance/alignment tests | 1.0 | 0.1 | 0.25 | yes |
| D4 | WER/CER x SNR grid | 2.0 | 0.2 | 0.5 | compute-bound |
| D4 | ADR-001 table | 1.0 | 0.1 | 0.25 | yes |
| D4 | Safety set + recall/false-block | 3.0 | 0.3 | 0.75 | yes |
| D4 | Urgency LOSO report | 0.5 | 0.05 | 0.125 | **blocked on recordings** |
| D4 | E2E 100 utt, RSS, per-stage | 1.5 | 0.15 | 0.4 | yes |
| D4 | Fixes + freeze candidate | 3.0 | 0.3 | 0.75 | mixed |
| D4 | AI Hub prep | 2.0 | 0.2 | 0.5 | **blocked on token** |
| D5 | Phone P1 (adb, ASR, NMT, TTS, k) | 7.5 | 7.5 | 7.5 | **no: hardware/owner-bound, factor 1.0** |
| D6 | Thermal/airplane/network check | 1.0 | 1.0 | 1.0 | no |
| D6 | espeak-ng license + final report + freeze | 3.0 | 0.5 | 1.0 | partly owner decision |
| D6 | Buffer | 4.0 | 4.0 | 4.0 | keep |

Sum of scalable items (D3+D4, excluding blocked): v4 21.0 h → 2.1 h at 0.10 / 5.25 h at 0.25. Fixed-duration items (phone 7.5 + thermal 1.0 + buffer 4.0) = 12.5 h. Realistic total D3-D6 = about 17-22 h of work plus waiting on recordings, token and phone, versus 41.5 h in v4 (v4 total 54.5 h incl. D2).
Consequence: the 50 h cap is no longer the binding constraint; the binding constraints are recordings (urgency F1 cannot be reported without them), SD712 sessions, and the AI Hub token (R1, deadline start of D5). Because time is freed, P2 (Kotlin app, est. 11 h) becomes feasible only if P1 finishes early: decide after the phone sessions.
