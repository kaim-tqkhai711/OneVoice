# NMT/safety handoff — laptop, four direct directions

Date: 2026-10-09. Owner: “Bạn của bạn”. Branch: codex/nmt-safety-four-directions.

## Scope and status

Implemented: four local-only Marian adapters; EOS/truncation/unknown/constraint evidence;
strict safety with bounded grammar; independent evaluation and text benchmark tools;
160 draft reference rows (40 per direction, 20 dev + 20 holdout).

Not validated: pretrained translation quality, actual candidate tokenizers, model latency/RAM.
No pretrained checkpoint was downloaded, per user's explicit “code/tests first” decision.
Korean references and lexicon are drafts; automatic speech remains blocked pending review.
Candidate EN-KO is quarantined pending an actual tokenizer/embedding mapping audit.

Shared contracts.py, stages/base.py and all audio/runtime/factory/config files remain unchanged.
The current main checkout is untouched. Integration is runtime owner's responsibility.

## Additive contract (no shared-interface changes needed)

- NMT signature stays translate(text, src, tgt).
- EvidenceMtResult subclasses existing MtResult and adds evidence.
- evidence: eos_reached, truncated, input_truncated, source/output token counts,
  source/output unknown counts, requested/satisfied/unsatisfied constraints,
  tokenizer issues, model ID, revision, backend.
- DetailedSafetyReport subclasses SafetyReport and adds status/direction/coverage/limitations.
- PASS => passed=True, confirm=False.
- CONFIRM => passed=True, confirm=True.
- FAIL => passed=False, confirm=False.
- Existing gate already prevents TTS for both CONFIRM and FAIL.
- Legacy MtResult without evidence is accepted as an input type but requires CONFIRM.
  This intentional safety behavior prevents an old adapter silently certifying truncation.
  Runtime test_gate._Nmt fixture needs evidence for its “correct translation speaks” test;
  that runtime-owned file was not edited here.
- An independently detectable semantic contradiction still returns FAIL even when
  evidence is absent; low-confidence ASR therefore retains the old ABSTAIN behavior.

## Constructors and assets

Existing VI-EN ONNX adapter remains OrtMarianNmt(model_dir, threads=2, ...).
It now returns evidence and checks vocabulary, input limits and decoder termination.
New HF adapters use unconstrained generation plus the strict checker; they do not
claim active medical constraints. VI-EN ONNX retains its existing lexical constraints.

New CPU reference adapters:

    from tonebridge.stages.nmt_marian import ViEnNmt, EnViNmt, EnKoNmt, KoEnNmt
    nmt = EnViNmt(model_dir=Path("models/nmt-laptop/en-vi"), threads=2)
    mt = nmt.translate("Hello.", "en", "vi")
    checker = ClinicalSafetyChecker("en", "vi")
    report = checker.check(mt)

Asset directories, relative to this checkout's Prototype:

- models/nmt-laptop/vi-en
- models/nmt-laptop/en-vi
- models/nmt-laptop/en-ko
- models/nmt-laptop/ko-en

Do not overwrite models/nmt/vi-en-int8-arm64 baseline.
Expected HF files: config.json, generation_config.json, tokenizer config/special tokens,
source.spm, target.spm, vocab.json, model.safetensors (or safetensors shards/index).
If config declares separate_vocabs, target_vocab.json is required.
Source SentencePiece pieces must map to published embedding IDs; never invent IDs.
After approved acquisition, inventory_nmt_assets.py pins revision and file hashes.
Adapter verifies manifest files. Missing provenance yields confirmation, not safe speech.
torch.set_num_threads is process-wide: runtime owner should coordinate its thread budget.

EN-KO upstream metadata at ae8606b7b29a495f31ce679cee2007f536a3a5ce:
config says shared embeddings/vocab_size 32001; files do not include target_vocab.json.
Historical repository reports source.spm/vocab mismatch. Metadata alone cannot repair this.
An adapter must reject mismatches; choose/convert another candidate only after validation.

## Safety limits and deliberate blocks

The checker compares entities, per-clause doses/units, action/administration method,
subject, question vs assertion, negation, uncertainty, comparator and supported frequency.
Decimal and mass conversions use Decimal, not float. Prednisone is not a prednisolone alias.
Additional/unparsed facts, ambiguous multi-entity scope, unsupported numbers/units and missing
translation evidence require confirmation. This finite grammar is deliberately conservative.
Conditions, temporal descriptions, many synonyms and unspecified medication names remain
outside current coverage. PASS is never an ASR correctness or diagnosis guarantee.
Glossary matching now uses word boundaries. The historical egg/eggs gold bug is
fixed without rewriting frozen old datasets; old published metric totals must not
be treated as fresh results under the repaired scorer.

Korean automatic PASS requires BOTH a reviewed lexicon and explicit checker opt-in;
these must be set only after bilingual human review. Draft mode still detects supported
contradictions but cannot be advertised as clinically validated.

## Reproducible commands

Use the isolated Python 3.12 environment .venv-nmt at checkout root.
Windows transformers 4.57.6 Marian JSON loader uses locale encoding; launch with -X utf8.

    $env:PYTHONPATH="Prototype/src"
    .venv-nmt/Scripts/python.exe -X utf8 -B -m pytest -q Prototype/tests/test_nmt_completion.py Prototype/tests/test_clinical_relations.py Prototype/tests/test_nmt_local_models.py Prototype/tests/test_nmt_evaluation.py Prototype/tests/test_nmt_pipeline_bridge.py

From Prototype, after model approval/acquisition:

    ../.venv-nmt/Scripts/python.exe -X utf8 -B tools/inventory_nmt_assets.py --model-dir models/nmt-laptop/en-vi --model-id Helsinki-NLP/opus-mt-en-vi --revision ACTUAL_40_CHARACTER_COMMIT
    ../.venv-nmt/Scripts/python.exe -X utf8 -B tools/run_text_nmt.py --direction en-vi --model-dir models/nmt-laptop/en-vi --text "Hello."
    ../.venv-nmt/Scripts/python.exe -X utf8 -B tools/benchmark_text_nmt.py --direction en-vi --model-dir models/nmt-laptop/en-vi --out results/nmt-laptop/en-vi.json --hyp-out results/nmt-laptop/en-vi-hyp.jsonl
    ../.venv-nmt/Scripts/python.exe -X utf8 -B tools/evaluate_text_nmt.py --direction en-vi --split dev --hyp results/nmt-laptop/en-vi-hyp.jsonl --out results/nmt-laptop/en-vi-eval.json

Text CLI: exit 0=PASS; exit 2=CONFIRM/FAIL; exit 1=model/input error. No audio is produced.
Benchmark each direction in a fresh process. Peak RSS includes startup/load; current RSS
and before/after load RSS are separate. These measurements exclude ASR, TTS and playback.
Run --split holdout only after development freezes; do not tune on its outcomes.

## Independent evaluation

references-only reports checker coverage on draft references, never NMT accuracy.
Hypothesis evaluation asserts every expected ID exactly once and verifies source/direction.
Optional external human labels: {id, reviewer, clinical_correct: bool, reason}.
False accept = human-incorrect output receiving PASS.
False block = human-correct output receiving CONFIRM/FAIL.
chrF is labeled draft-reference chrF and is not clinical correctness.
Draft reference/reviewer quality is a separate release requirement.
Legacy SemanticSafetyChecker.check_texts ablation switches remain available for
text-only research. Its runtime check(mt) always applies strict checks.

## Runtime-owner integration checklist

1. Approve additive evidence schema and import subclasses without editing shared files.
2. Use source/target languages from runtime config and select the matching adapter/checker.
3. Persist evidence in runtime telemetry; the old TurnRecord does not retain it.
4. Preserve safe failure: absent assets, invalid tokenizer, incomplete output => no TTS.
5. Keep UNKNOWN urgency; do not bypass checker to make Korean demos speak.
6. Update runtime test fixtures to include true EOS evidence, not fabricated production data.
7. Run four-direction E2E with real models after assets and Korean review are approved.

Suggested test-fixture update for runtime owner (not applied here):

    from tonebridge.nmt_evidence import EvidenceMtResult, TranslationEvidence
    # In test_gate._Nmt.translate, the deterministic fixture represents a complete
    # text output. This assertion belongs only to the test double, never production.
    return EvidenceMtResult(
        src_text=text, tgt_text=self.out, src_lang=src, tgt_lang=tgt, hops=["vi>en"],
        evidence=TranslationEvidence(
            eos_reached=True, truncated=False,
            source_tokens=len(text.split()), output_tokens=len(self.out.split())))

No commits, pushes, shared-interface edits, model downloads or factory integration were performed.
