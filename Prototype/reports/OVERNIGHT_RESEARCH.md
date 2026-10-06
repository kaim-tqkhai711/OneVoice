# Overnight research notes (only for metrics below the 70 % stop line)

Venue/authors/years below were read from the arXiv abstract pages on 2026-10-07 (WebFetch). "Idea in my words" is my reading of the abstract, not of the full paper.

## M2: NMT slot preservation (dev, pre-optimisation: dose 58.7 %, medication 58.0 %, severity 57.9 %; negation 98.8 %, allergy 100 %)
Problem: Marian vi→en rewrites drug names into other words ("ibuprofen" → "Escondido"), drops dose units and intensifiers.

| Citation + link | Idea in my words | How applied here | Cost | Portability / license |
|---|---|---|---|---|
| Hokamp & Liu 2017, *Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search*, ACL 2017. https://arxiv.org/abs/1704.07138 | Extend beam search so that required words/phrases must appear in the output, without changing model weights | Required English forms from the glossary (drug, unit, number, intensifier) become constraints during decoding | Beam grows with number of constraints (cost grows linearly) | Pure decode logic around the ONNX decoder; no new model. OK |
| Post & Vilar 2018, *Fast Lexically Constrained Decoding with Dynamic Beam Allocation for NMT*, NAACL 2018. https://arxiv.org/abs/1804.06609 | Constraint-satisfying search with cost independent of the number of constraints by splitting a fixed beam into banks by constraints met | Informs a beam-4 variant; tonight's greedy variant is a simplified, soft form: add a bonus to the first token of each unmet constraint, force the rest of the word once started | Greedy: ~0 extra decoder steps; beam4 already measured ~6-12x slower than greedy here | Same as above. My simplification is NOT these algorithms and gives no guarantee |
| Dinu et al. 2019, *Training Neural MT to Apply Terminology Constraints*, ACL 2019. https://arxiv.org/abs/1906.01105 | Train the model to copy inline target terms given next to source words; avoids decode-time overhead | Would need fine-tuning opus-mt: **not allowed tonight** (no fine-tuning). Recorded as a next step | Needs training data + GPU hours | License of opus-mt (Apache-2.0) allows it; excluded by the session rules |

Levers allowed tonight: decoding (beam, length penalty), glossary/constrained decoding, quantization choice, safety-check rules/lexicon.
