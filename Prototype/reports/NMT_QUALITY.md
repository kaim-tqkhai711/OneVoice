# NMT quality + safety set (C), VI->EN, opus-mt-vi-en, laptop x86 (2026-10-07)

Everything below is measured on the files in `results/` and `configs/safety/`; "(contended)" marks timings taken while background jobs ran and not used for projections.

## 1. 30 clinical sentences: fp32 | INT8 arm64 | error belongs to
Full table (source, 5 builds, check flags, attribution): `results/nmt30_table.md` (flags are automatic = Semantic Safety Check v1 critical/confirm; "SAFETY" = critical). Counts over the 30 sentences, greedy:
- fp32: 4 flagged, INT8: 5 flagged, INT8 encoder + fp32 decoder: 6 flagged, INT8 beam4: 3, fp32 beam4: 3.
- Attribution fp32 vs INT8 (greedy): both 4 (#6 "two paracetamols" unit lost, #18 "very" lost, #22 "five millimetres", #28 "one" without tablet), **INT8 only 1 (#17 ibuprofen -> "medrine")**, fp32 only 0, none 25.
- Errors my automatic check does not see (symptom / fluency, v1 scope): #14 "hurt by my chest", #20 "I've been diabetes", #25 "sick of stomach", #12 fp32 "Please contact your name", #7 fp32 "wine" for "rượu bia". Reference translations are still needed (`docs/nmt_ref_template.csv`, chrF later).

## 2. Is INT8 the cause? (1224 safety items, slot preserved = every PRIMARY gold slot present)
| build | ALL | dose | medication | severity | negation | allergy |
|---|---|---|---|---|---|---|
| fp32 greedy | 77.9 % | 56.0 | 65.3 | 79.6 | 98.8 | 100 |
| INT8 greedy | 77.5 % | 58.0 | 62.7 | 77.6 | 98.8 | 100 |
| INT8 enc + fp32 dec greedy | 76.3 % | 52.3 | 62.7 | 79.6 | 98.8 | 100 |
| INT8 beam4 | 79.7 % | 61.7 | 65.3 | 82.2 | 99.4 | 100 |
INT8-only errors 25 vs fp32-only 20 (net +5 of 1224). Conclusion: INT8 is not a systematic cause; the hybrid (fp32 decoder) is not better (size 352 MB vs 123 MB). Layer-exclusion search not run (D-11). Sizes: fp32 485.9 MB, INT8 122.7 MB, hybrid 352.5 MB.

## 3. Beam 4 vs greedy (INT8, quiet machine, 30 sentences x 3, 2 threads, `results/nmt30_quiet.json`)
| decode | p50 ms | p95 ms | flagged sentences (30) | items with a checkable-slot error (1224) |
|---|---|---|---|---|
| greedy | 67.0 | 97.4 | 5 | 441 |
| beam4 | 339.4 | 464.6 | 3 | 392 |
Beam4 is ~5x slower for 2 fewer flagged sentences / ~11 % fewer slot errors; greedy + glossary constraint (below) gives larger gains at no latency cost, so greedy is shipped.

## 4. Raw NMT slot preservation by group, Clopper-Pearson 95 % (all 1224; raw = no glossary)
Dev (n per group 76-160): allergy 100 [95.3, 100], dose 58.7 [50.3, 66.6], medication 58.0 [49.7, 66.0], negation 98.8 [95.6, 99.8], severity 57.9 [46.0, 69.1]. Files: `results/nmt_raw_slots_dev.txt`, `results/nmt_raw_slots_test.txt`. Real failures saved as the second evaluation set: `configs/safety/nmt_real_errors_raw_int8_greedy.jsonl` (441 items).
Typical raw failures: drug names rewritten ("ibuprofen" -> "Escondido", "paracetamol" -> "paracetamino", "heparin" -> "the gorilla"), "miligam" -> "millimetres", "viên" dropped, "rất/cực kỳ" dropped.

## 5. Glossary-constrained decoding (M2 lever, locked) - dev, per group
| | before (raw) | round 1 (bonus 5) | round "M3 r1" (lexicon v1.1, **locked**) |
|---|---|---|---|
| dose | 58.7 | 95.3 | **97.3** [93.3, 99.3] |
| medication | 58.0 | 93.3 | **98.0** [94.3, 99.6] |
| severity | 57.9 | 100 | **100** [95.3, 100] |
| negation | 98.8 | 99.4 | 99.4 |
| allergy | 100 | 98.7 | 100 |
Limit that matters: the constrained forms ARE the forms the gold list and the check lexicon accept (same author), so this shows the required word appears, not that the sentence is a good translation. Fluency/chrF unmeasured. A skim of 14 dev outputs: good ("The dose is 100 milligrams a day."), odd in places ("rất nặng hơn" -> "very severe", "vô cùng nặng hơn" -> "very, very severe").

## 6. Safety set
`configs/safety/safety_set_v1.jsonl`: 1224 items = negation 320 (incl. 20 question-particle controls), medication 300, dose 300, allergy 152, severity ("mức độ") 152. Split by template (dev 612 / test 612; no template on both sides; `configs/safety/split_templates.json`). Generator `tools/make_safety_set.py` (seed 20261008). Seeded errors: 1329 dev + 1279 test synthetic corruptions of the English reference, 11 error types.
