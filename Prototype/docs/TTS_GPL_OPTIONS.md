# EN TTS without a GPL dependency: options (2026-10-06)

Problem: Piper-style VITS voices in sherpa-onnx phonemize with espeak-ng (GPL-3.0). Shipping it statically linked in the APK may make the whole app subject to GPL-3.0. Status: **license chưa chốt**; prototype continues with Piper `en_US-ljspeech-medium`.
Hour figures are estimates. Items marked **(U)** are from memory and not yet verified against the primary source.

| Option | What | Est. cost | Risks / unknowns |
|---|---|---|---|
| A. Keep Piper + espeak-ng, resolve the license | Read exactly how sherpa-onnx links espeak-ng (static vs shared lib / separate data), and what GPL-3.0 then requires of the app | 1.5 h reading + 1 h writing the decision | Might be unavoidable "whole app GPL"; outcome not known. 0 h of engineering if acceptable |
| B. sherpa-onnx VITS/other EN model with a lexicon front-end instead of espeak-ng | **(U)** some sherpa-onnx EN models (e.g. a VCTK VITS) use `lexicon.txt` rather than espeak-ng. Verify per model: front-end type, data license, size, latency | 2.5-3.5 h: verify front-end + license (1 h), swap, listen-check, re-bench latency/RSS (2 h) | Voice quality unknown; multi-speaker model needs a pinned speaker id; data license (VCTK) **(U)** |
| C. Android system TTS (`android.speech.tts`) | Phone's installed offline voice; no TTS asset shipped by us | ~2 h integration | Offline voice not guaranteed on SD712; voice not pinned (weakens the reproducibility claim); not measurable on laptop, so G-L cannot cover it; first-audio latency unknown |

Recommendation: do A's reading now (it is cheap and may close the question), and verify B's front-end/license claims in D2 if time allows; choose by D4 so that D5's phone port is built once. Decision belongs to the owner.
