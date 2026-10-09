# NMT/safety code delivery — 2026-10-09

Branch: codex/nmt-safety-four-directions; isolated Python 3.12.13 CPU environment.
No pretrained weights downloaded. No factory/audio/runtime/shared-interface edits.

## Verified code behavior

- Four direction adapters invoke real local Marian generation on tiny randomly
  initialized test models. This proves local loading/routing/termination plumbing,
  not translation quality.
- EOS forced by a token budget is disabled, so exhausting the budget is truncated.
- VI-EN ONNX loop returns complete phrase coverage and accurate decode-step count.
- Known dose binding, negation scope, decimal/unit, added drug, stop/continue,
  uncertainty and comparator counterexamples are blocked by relational checks.
- Missing evidence, unknown tokens and unsupported critical content cannot receive PASS.
- Korean automatic speech remains blocked pending human review.
- Repaired prednisolone/prednisone alias, glossary substring matching and egg/eggs gold.

## Checker-only development measurement

80 draft references = 20 per direction. PASS 32, CONFIRM 48, FAIL 0.
Both Korean directions: no automatic PASS. File: NMT_CHECKER_DEV_DRAFT.json.
These counts are finite-grammar coverage on draft references, not NMT accuracy.
Holdout references were prepared and structurally checked, not used to tune outcomes.

## Validation boundary

Final scoped run: 105 passed, 1 failed, 1 test-model generation-config warning (21.58 s).
NMT/safety/glossary scope: 96 passed. Unchanged runtime gate compatibility: 9 passed,
1 fixture mismatch described below. This was not a full audio/model test suite.
The run includes number parsing,
EOS/truncation/constraints, tokenizer mapping checks, four tiny local model generations,
independent evaluation completeness and integration with unchanged VI-EN gate/pipeline.

Existing runtime-owned test_gate.test_correct_translation_reaches_tts still expects a
plain MtResult with no EOS evidence to speak. It now correctly gets CONFIRM with
translation_evidence_missing. That fixture needs EvidenceMtResult; this file was
left unchanged according to file ownership. New test_nmt_pipeline_bridge demonstrates
PASS reaches TTS with evidence and wrong/truncated outputs never reach TTS.
git diff --check passes. SHA-256 comparisons confirm contracts.py, stages/base.py,
factory.py, pipeline.py, config.py and cli.py are identical to main checkout.

No claim of four-direction pretrained-model smoke success, clinical validation,
NMT latency/RAM, E2E benchmark, Android or NPU performance is made.

## Next review gates

1. Runtime owner accepts additive subclass schema and updates its test fixture.
2. User approves checkpoint acquisition, separately from completed code/tests.
3. Audit actual tokenizers first (EN-KO candidate currently quarantined).
4. Review bilingual references and Korean lexicon.
5. Freeze model revisions/hashes; run standalone text benchmarks and human semantic labels.
6. Runtime owner integrates factory and runs audio E2E per direction.
