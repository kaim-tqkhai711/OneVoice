# License table v0 (2026-10-06)

Status: **V** = verified from the primary source (HF model card/API or GitHub API, fetched 2026-10-06);
**U** = not verified, must be checked before release. Any NC/ND/GPL item is flagged.

| Asset | Role | License | Status | Source / note |
|---|---|---|---|---|
| zzasdf/viet_iter3_pseudo_label (`sherpa-onnx-zipformer-vi-int8-2025-04-20`) | ASR vi, **offline** transducer, ~70k h | Apache-2.0 | V (card) | Training-data terms (pseudo-label corpora) **U** |
| hynt/Zipformer-30M-RNNT-6000h (`...vi-30M-int8-2026-02-09`) | ASR vi | **CC-BY-NC-ND-4.0** | V | **REJECTED**: NC + ND |
| johnBamma/icefall-asr-ksponspeech-zipformer-2024-06-24 (`sherpa-onnx-zipformer-korean-2024-06-24`) | ASR ko, offline | Apache-2.0 (card) | V (card) | KsponSpeech data terms **U** (AI Hub Korea). Not needed for P0 |
| Helsinki-NLP/opus-mt-vi-en | NMT vi>en | Apache-2.0 | V (API) | |
| Helsinki-NLP/opus-mt-tc-big-en-ko | NMT en>ko, 209,158,401 params | CC-BY-4.0 | V (API) | **Rejected on engineering grounds** (265 MB INT8, p95 644 ms laptop, broken tokenization). Measurements kept as evidence |
| alirezamsh/small100 | NMT direct vi>ko candidate, 332.7M params | MIT | V (API + card) | Bake-off candidate |
| NLLB-200 | NMT | CC-BY-NC | known | **BANNED** by project rule |
| Silero VAD (snakers4/silero-vad) | VAD | MIT | V (GitHub API) | |
| WebRTC Audio Processing | Stage-0 NS/HPF | BSD-3-Clause | U | Pin exact build (python wheel / NDK build) on D2 |
| GTCRN (Xiaobin-Rong/gtcrn) | Neural denoise (ADR-001 ON/OA) | MIT | V (GitHub API) | Pretrained weights terms **U** |
| SwiftF0 (lars76/swift-f0) | F0 / voicing | MIT | V (GitHub API) | |
| eGeMAPS feature set | Acoustic descriptors | n/a (reimplemented with librosa/torchaudio) | n/a | openSMILE **BANNED** |
| sherpa-onnx (k2-fsa) | ASR/VAD/TTS runtime | Apache-2.0 | V (GitHub API) | Piper-style VITS models phonemize via espeak-ng (**GPL-3.0**): **U**, check how it is linked before release |
| Piper voice en_US-ljspeech-medium | TTS en (mode B) | Public domain (LJ Speech) | V (MODEL_CARD) | Chosen over lessac/amy. **Pipeline license chưa chốt**: phonemizer espeak-ng is GPL-3.0, see `docs/TTS_GPL_OPTIONS.md`; must close before freeze |
| Piper voice en_US-lessac-medium | TTS en | Blizzard 2013 licence (cstr.ed.ac.uk) | U | Terms not read; amy is fine-tuned from lessac, so same caveat |
| Piper runtime | TTS engine | rhasspy/piper MIT (archived) vs OHF piper1-gpl GPL-3.0 | per Proposal [10],[11] | Pin which artifact is used |
| MeloTTS-Korean | TTS ko | MIT (per Proposal [13]) | U | Probe deferred until the bake-off result |
| ONNX Runtime / ORT-Android | Inference | MIT | U | |
| Kaggle Hospital Ambient Noise (nafin59) | Noise eval set | unknown | U | Check dataset page before use |
| MUSAN, DNS noise | Noise tune sets | CC / research | U | Check per subset |
| espeak-ng | Phonemizer for Piper-style VITS | **GPL-3.0** | V (known) | **license chưa chốt**: how it is linked into the APK decides the consequence. Options: `docs/TTS_GPL_OPTIONS.md`. Deadline: before freeze |
