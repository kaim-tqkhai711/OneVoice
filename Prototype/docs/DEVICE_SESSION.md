# Phone measurement sessions (SD712, Android 11) — prepared 2026-10-07, NOT yet run

Nothing here has touched the phone. Max 2 sessions, one script each: `tools/device_session.sh 1` and `tools/device_session.sh 2`.
Dry run (no phone): `DRY_RUN=1 bash tools/device_session.sh 2` (prints every adb call; the fake numbers it produces are deleted, never keep them).

## Official-source confirmations (checked 2026-10-07)
| Question | Finding | Source |
|---|---|---|
| Prebuilt sherpa-onnx Android arm64 CLI binary? | Yes: release asset `sherpa-onnx-v1.13.8-android-aarch64-termux-static.tar.bz2` (149.2 MB) contains `bin/sherpa-onnx-offline`, `bin/sherpa-onnx-offline-tts`, ... The extracted `sherpa-onnx-offline` is `ELF 64-bit ARM aarch64, interpreter /system/bin/linker64` (checked locally with `file`) = Android linker. It is the "termux" flavour; **running it from `adb shell` in /data/local/tmp is expected to work but is NOT verified until session 1.** Fallback: the 15.7 MB `...-termux-shared` tarball (needs its .so next to it, `LD_LIBRARY_PATH`) | https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.8 (GitHub API asset list) |
| Android 11 compatibility | Build script default `SHERPA_ONNX_ANDROID_PLATFORM=android-21` (Android 5.0) => Android 11 (API 30) is above the floor. No official statement for the termux binaries specifically | `build-android-arm64-v8a.sh` in k2-fsa/sherpa-onnx master |
| ORT benchmark tool for Android | `onnxruntime_perf_test` has an Android branch in `cmake/onnxruntime_unittests.cmake` (links `log`, `android`) so it **can be built** with the Android NDK (`build.py --android ...`). **No prebuilt binary is published; the ORT docs pages I fetched do not describe it** (docs fetch returned nothing about perf_test). Needs: Android NDK + ~1-2 h build (estimate). Fallback without NDK: a small Kotlin/C++ app with the ORT AAR (that is scope P2) | https://github.com/microsoft/onnxruntime (`onnxruntime/test/perftest/README.md`, cmake) |

**Owner to be told (requirement of the plan):** the ORT Android benchmark tool is *not* available prebuilt. Session 2 (NMT encoder + decoder step) is blocked until `onnxruntime_perf_test` is cross-compiled; session 1 (k) is not blocked.
`perf_test` on the merged decoder needs correct input shapes (past KV) — it generates random inputs from model metadata but dynamic dims may need `-I`/`-f` dim overrides; unverified, expect one iteration of debugging.

## Space needed on the phone (/data/local/tmp), checked by the script (needs 2x)
| Session | Files | MB (local sizes) |
|---|---|---|
| 1 | sherpa-onnx-offline (~few MB), ASR int8 enc 67.6 + dec 1.2 + joiner 1.0, 3 wavs | ~80 (script requires 2x120) |
| 2 | S1 + ORT perf_test + NMT enc 44.8 + dec 77.9, Piper TTS 79 (incl. espeak data), tts binary | ~300 (script requires 2x520) |
Laptop-side: `data/android_bin/` holds the 149 MB tarball and its extraction.

## Manual steps BEFORE a session (owner)
1. `winget install Google.PlatformTools`; on the phone: Settings > About > tap Build number 7x > Developer options > **USB debugging ON** (only for the session).
2. Accept the RSA fingerprint prompt; `adb devices` must show `device`.
3. **Airplane mode ON** (Wi-Fi/Bluetooth off), close all background apps, brightness minimum, screen kept on only if needed.
4. Charge level ≥ 50 % but **unplugged from a wall charger during runs if possible** (USB cable stays for adb; energy is not measured anyway). Let the phone cool: battery temperature within ~3 °C of its idle value (the script logs it; wait until two readings 2 min apart agree).
5. Note the room temperature and whether the phone is in a case.

## Run
Session 1 (≤ 30 min): `bash tools/device_session.sh 1`. Gives RTF of Zipformer VI INT8, 2 threads, 5 repeats, plus temperatures before/after each. Then compute `k = RTF_phone / 0.040` (laptop RTF from `results/asr_vi_rtf.json`, same files, 2 threads) and update the proxy thresholds in PROGRESS.md.
Session 2: needs the perf_test binary (`ORT_PERF=...`). Collects ASR RSS (VmHWM polling), NMT enc/dec-step, TTS, no-network counters, temperatures.

## AFTER a session
1. The script deletes `/data/local/tmp/tonebridge` on exit (also on error). Verify: `adb shell ls /data/local/tmp`.
2. **Turn USB debugging OFF**, revoke USB debugging authorizations if the phone is shared, airplane mode back as wanted.
3. Results: `results/device/session<N>_<timestamp>/` (JSONL + raw logs). Raw files are kept; any figure in a report must cite them.

## Limits to state in every report
Energy / battery **not measured** (battery cannot be read on this phone); battery temperature + thermal zones before/after instead. Free RAM on the phone is an estimate (≤ 1.5 GB) until the probe measures `MemAvailable`; the total-pipeline RSS threshold is 1.0 GB. Phone E2E for NMT is "composed (est.)" = encoder + n_tokens × decoder_step.
