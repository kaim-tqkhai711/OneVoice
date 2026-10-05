# ADR-002: NMT direction for the prototype (VI->EN primary, VI->KO not demonstrated)

- Status: **Accepted, 2026-10-06** (owner sign-off).
- Supersedes: ADR-002 "pivot vs NLLB" in `onevoice/02_TECHNICAL_REQUIREMENTS.md` (NLLB is CC-BY-NC and banned).
- Owner: Khai (solo prototype).

## Context

Tech Proposal v3.1 (`ToneBridge_TechProposal_v3_1_Revised_Template.docx`) names the language pairs in these places, verbatim:
- §1.2 Proposed Solution: "ToneBridge is an offline-first VI↔KO / VI↔EN medical communication prototype."
- §2.3 Target Users & Use Cases: Vietnamese clinicians / nurses, "VI ↔ KO; VI ↔ EN", priority Critical; Korean patients / families, "KO ↔ VI", priority Critical; Foreign trainees / staff, "EN ↔ VI", priority High.
- §2.4 Key Design Constraints, row "Language scope": "VI ↔ KO primary; VI ↔ EN secondary/supporting pivot".
- §3.3 Application Roadmap, Phase 1: "VI↔KO and VI↔EN short medical communication on an Android phone, measured under controlled noise."
- §4.2 module table: NMT "OPUS-MT / Marian pivot + glossary", "<600 ms total pivot target"; TTS "Pinned Piper VI/EN voice + MeloTTS KO".

The prototype has 60 working hours on one SD712 phone (2 big cores, ~3.5 GB RAM total) with a Proposal latency target of p50 <= 1.5 s / p95 < 2.0 s from PTT release to first TTS sample.

The hop en->ko is the problem:
- The only en->ko Opus-MT checkpoint is `Helsinki-NLP/opus-mt-tc-big-en-ko` (transformer-big). The other Helsinki-NLP Korean repos are ko->en only (`opus-mt-ko-en`, `opus-mt-tc-big-ko-en`, `opus-mt_tiny_kor-eng`).
- Timeline v1 budgeted ~110 MB and 350 ms for this hop.

## Measurements

Conditions (identical for every row): laptop (Intel Core Ultra, Family 6 Model 170, Windows 11), ONNX Runtime 1.23.2 CPU EP, `intra_op=2`, optimum ONNX export, dynamic INT8 (avx2 config), greedy, max_new_tokens 48, 30 clinical sentences x 3 repeats = 90 runs after 3 warm-up, `tools/bench_hop.py`. Raw rows: `results/*_runs.jsonl`, summaries: `results/*_summary.json`.
Laptop x86 numbers are not SD712 numbers. The 150 ms p50 acceptance threshold assumes SD712 is 3-4x slower than this laptop (est.; no SD712 measurement yet).

| Option | Checkpoint | Params | License | INT8 size | p50 ms [95% CI] | p95 ms [95% CI] | Peak RSS (inference) | Output quality |
|---|---|---|---|---|---|---|---|---|
| A. Pivot with big en->ko | opus-mt-tc-big-en-ko | 209,158,401 | CC-BY-4.0 | 265 MB | 171 [152, 196] | 608 [287, 664] | 966 MB | **Garbage**: even the model-card example fails (transformers 4.57.6, torch fp32 and ONNX fp32 alike). `source.spm` pieces such as `chest`, `pain` are missing from `vocab.json` -> `<unk>`. Model card reports BLEU 13.7 on flores101 |
| C. Direct vi->ko | alirezamsh/small100 | 332.7M | MIT | **581 MB** | **218 [203, 239]** | 373 [336, 421] | 1596 MB | Readable Korean, but errors that matter clinically (see pairs) |
| B. Drop KO from P0 | opus-mt-vi-en (single hop) | not yet measured | Apache-2.0 | not yet measured | not yet measured | not yet measured | not yet measured | Not yet measured |
| D. Another small en->ko model | none found | | | | | | | The Helsinki-NLP repos listed above are the only candidates found |

Note on the SMaLL-100 first run: p50 307 ms / p95 498 ms was measured while other jobs ran on the laptop and its peak RSS (6.05 GB) included the FP32 -> INT8 quantization step in the same process. The table shows the clean re-run (quantization skipped).

### SMaLL-100 bake-off: acceptance thresholds (set before measuring) vs result

| Threshold | Required | Measured | Result |
|---|---|---|---|
| Readable, on-topic Korean | >= 9/10 pairs | 8/10 (single non-native rater, see below) | **FAIL** |
| p50 latency (laptop) | <= 150 ms | 218 ms | **FAIL** |
| INT8 size | <= 350 MB | 581 MB | **FAIL** |

Per the pre-agreed rule, failing any threshold selects option B; no further candidates were tried.

### Ten source -> target pairs (SMaLL-100, vi->ko, greedy), verbatim

| # | Source (vi) | Output (ko) | Note |
|---|---|---|---|
| 1 | Bạn có bị đau ngực không? | 당신은 가슴에 고통을 느낀가요? | readable, slightly ungrammatical |
| 3 | Bạn bị đau này bao lâu rồi? | 여러분은 얼마나 오래 이 고통을 겪었습니까? | readable |
| 4 | Bạn có dị ứng với loại thuốc nào không? | 어떤 약에 알레르기가 있습니까? | correct |
| 5 | Tôi bị dị ứng với penicillin. | 나는 페니실린에 알레르기가있다. | correct |
| 6 | Uống hai viên paracetamol mỗi sáu giờ. | 6 시간마다 두 개의 파라세타мол을 마십시오. | **drug name corrupted with Cyrillic characters** |
| 10 | Tôi không thở được. | 난 숨을 수 없어. | **wrong meaning** ("I cannot hide"), loses "cannot breathe" |
| 14 | Tôi thấy chóng mặt và tức ngực. | 나는 가두르고 가슴을 느꼈습니다. | **garbled** |
| 22 | Chúng tôi sẽ tiêm cho bạn năm miligam. | 우리는 당신에게 5 밀리그램을 주입 할 것입니다. | correct |
| 27 | Cô ấy không có dị ứng thuốc nào được biết. | 그녀는 알레르기가있는 약은 없었습니다. | **negation scope distorted** |
| 30 | Tôi cần giúp đỡ ngay bây giờ. | 나는 지금 도움을 필요로. | incomplete but understandable |

Rating caveat: one rater who is not a native Korean speaker. The three safety-relevant failures (6, 10, 27) are exactly the classes the Semantic Safety Check exists to catch.

## Decision

**Option B.** P0 is VI->EN, fully on-device: ASR vi -> Opus-MT vi->en -> Semantic Safety Check -> Fusion Gate -> TTS en.
VI->KO is not demonstrated. The report states plainly: "measured, did not meet the budget/quality bar", with the numbers above attached.
Rejected: A (broken checkpoint, 265 MB, p95 above 500 ms on a fast laptop), C (581 MB INT8 and 1.6 GB RSS alone exceed the Proposal's 1.5 GB total target; clinically relevant errors), D (no candidate found).

TTS for B: Piper `en_US-ljspeech-medium` (public domain, verified from its MODEL_CARD). MeloTTS-Korean probing is deferred indefinitely.

## Consequences

- Demonstrated pair: VI->EN only (ASR vi -> opus-mt-vi-en -> Semantic Safety Check -> Fusion Gate -> Piper EN).
- Not demonstrated: VI->KO, KO->VI, EN->VI, EN->KO. No Korean ASR, NMT or TTS runs in the prototype.
- Attached measurements for the pairs not demonstrated: the two tables above (tc-big en->ko, SMaLL-100 vi->ko), `results/*_runs.jsonl`, `results/*_summary.json`, `tools/bench_hop.py`. Laptop x86 numbers, not SD712.
- The demo no longer covers the Viet-Korean setting of Proposal §2 Problem Definition (§2.3 row "Korean patients / families, KO <-> VI") and §3 Business Solution (§3.3 Phase 1 "VI<->KO").
- Proposal deviation #1 (PROGRESS.md): "VI <-> KO primary" is not demonstrated. Proposal deviation #2 (ASR): no streaming Zipformer for VI exists in the official k2-fsa lists; the Apache-2.0 VI checkpoint is offline, so the "<450 ms tail target" (§4.2) is re-measured as full-utterance decode time.
- Pivot code stays: `Nmt.translate` returns `hops`; an `en>ko` hop plugs in without pipeline changes.
- Semantic Safety Check lexicon is built for VI->EN (negation, drug, dose/unit, allergy, symptom). No Korean-side lexicon is written.
- Open measurement: `opus-mt-vi-en` size, latency, RSS and tokenization are not yet measured. D2 runs the same `bench_hop.py` conditions and checks the model-card example first (tc-big failed this check). If it fails the same way, stop and report.
- TTS: Piper `en_US-ljspeech-medium` (public domain per its MODEL_CARD). MeloTTS-Korean is not probed.

## Re-entry criteria for KO

KO comes back into the demo only when all four hold:
1. An en->ko or vi->ko model exists with a license that permits our use (checked from the primary source, recorded in `LICENSES.md`).
2. Its INT8 artifact is <= 350 MB.
3. It passes the latency threshold measured on the SD712 (threshold derived from the calibrated k, PROGRESS.md "k calibration").
4. Its output on the clinical sentence set is reviewed and approved by a reader who reads Korean.
