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
| google/fleurs (vi_vn) | Generic WER eval set | CC-BY-4.0 | V (HF dataset API, 2026-10-06) | Possible overlap with the 70k h pseudo-label training data cannot be excluded |
| AILAB-VNUHCM/vivos | VI eval set (not used) | CC-BY-NC-SA-4.0 | V (HF API) | **REJECTED**: NC |
| DEMAND noise (Zenodo 1227121) | Noise for the SNR grid (OHALLWAY, OOFFICE, PCAFETER, PSTATION used) | CC-BY-4.0 | V (Zenodo API) | Not hospital recordings; labelled "hospital-like at best". Hospital-specific noise (Kaggle) still **U** |
| m42-health/hospital_ambient_noise | Candidate | no license on card | U | Speech with hospital ambience, not used |
| GTCRN weights `gtcrn_simple.onnx` (sherpa-onnx release `speech-enhancement-models`) | Neural denoise arm | code MIT (GitHub API) | weights terms **U** | 535 KB; training data terms (DNS3) to check |
| silero_vad.onnx (sherpa-onnx `asr-models` release) | VAD | MIT (upstream) | V upstream, packaging U | |
| vits-piper-en_US-ljspeech-medium (sherpa-onnx `tts-models` release) | EN TTS | voice data public domain (MODEL_CARD, read) | V | 60.6 MB fp32; bundles espeak-ng-data (GPL-3.0), see docs/TTS_GPL_OPTIONS.md |
| csukuangfj/sherpa-onnx-zipformer-vi-int8-2025-04-20 | ASR packaging of zzasdf checkpoint | Apache-2.0 (README points to zzasdf) | V | decoder INT8 taken from zzasdf/viet_iter3_pseudo_label (Apache-2.0) |
| swift-f0 0.3.0 (PyPI wheel, `model.onnx` 135 KB) | F0 / voicing (Branch B), installed `--no-deps` | MIT (wheel METADATA `License-Expression: MIT`, LICENSE file in wheel) | V (2026-10-07) | Deps: numpy + onnxruntime only |
| librosa 0.11.0 | **Test reference only** (STFT equivalence for Branch B); not in the inference path | ISC | known (not re-read) | Portability law: reference allowed |
| torch (CPU) | Training/export only (urgency MLP code), never inference | BSD-3 | known | |
| sherpa-onnx v1.13.8 `android-aarch64-termux-static` CLI binaries | Phone session tooling | Apache-2.0 (repo) | V upstream, termux build packaging U | Not yet run on the phone |
| vits-piper-en_US-ljspeech-medium-int8 (sherpa-onnx `tts-models`) | Smaller EN voice tried for latency (same LJ Speech data, public domain) | PD data (MODEL_CARD in package, same voice as the shipped one) | V | **Slower on x86 (3x), not adopted**; espeak-ng GPL caveat unchanged |
| Piper en_US-kathleen-low | Candidate smaller voice | dataset CC0 BUT "finetuned from U.S. English Ryan voice (low)" and Ryan is CC BY-NC-SA 4.0 | V (HF MODEL_CARDs read 2026-10-07) | **REJECTED**: derived from NC weights |
| Piper en_US-ryan-low | Candidate | CC BY-NC-SA 4.0 | V | **REJECTED**: NC |
| Piper en_US-danny-low | Candidate | "See URL" (Mycroft mimic3-voices) + finetuned from Ryan | V | **REJECTED**: unresolved + NC parent |
| Synthetic noise (babble from FLEURS speakers, alarm generator) | ADR-001 noise types | FLEURS CC-BY-4.0 (speech) / generated | V | Not hospital recordings |
