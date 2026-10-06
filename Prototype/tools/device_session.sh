#!/usr/bin/env bash
# One script per phone session (SD712, Android 11, USB debugging on only during the session).
#   tools/device_session.sh 1   Session 1 (<= 30 min): device_probe + k calibration (Zipformer VI INT8 RTF, 2 threads)
#   tools/device_session.sh 2   Session 2: full P1 (ASR, NMT encoder + decoder step, TTS, RSS, load time, no-network check, temps before/after)
# Steps: adb devices -> free-space check -> push models + binaries to /data/local/tmp/tonebridge -> run -> pull JSONL -> delete device files.
# Idempotent (re-push skips files with equal size; the device dir is removed at the end, also on error via trap). Stops at the first failing step
# with a message. Energy/battery is NOT measured: battery temperature + thermal zones are logged before/after every run instead.
# Dry run on the laptop (no phone, nothing touched):  DRY_RUN=1 tools/device_session.sh 1
set -euo pipefail
SESSION="${1:?usage: device_session.sh 1|2}"
cd "$(dirname "$0")/.."
DEV=/data/local/tmp/tonebridge
OUTDIR="results/device/session${SESSION}_$(date +%Y%m%d_%H%M%S)"
JSONL="$OUTDIR/session${SESSION}.jsonl"
BIN_DIR="${BIN_DIR:-data/android_bin/sherpa}"        # extracted sherpa-onnx-v1.13.8-android-aarch64-termux-static (bin/ with sherpa-onnx-offline, sherpa-onnx-offline-tts)
ORT_PERF="${ORT_PERF:-data/android_bin/onnxruntime_perf_test}"  # must be built from source with the Android NDK (no prebuilt exists); session 2 only
ASR_DIR=models/asr/zipformer-vi-int8
TTS_DIR=models/tts/vits-piper-en_US-ljspeech-medium
NMT_DIR=models/nmt/vi-en-int8-arm64
NEED_MB_S1=120; NEED_MB_S2=520
DRY="${DRY_RUN:-0}"

die() { echo "FAILED at step: $*" >&2; exit 1; }
adb_() { if [ "$DRY" = 1 ]; then echo "[dry] adb $*" >&2; case "$*" in *"get-state"*) echo device;; *"df "*) echo "Filesystem 1K-blocks Used Available Use% Mounted"; echo "/data 100000000 1000 90000000 1% /data";; *"getprop ro.build.version.release"*) echo 11;; *"VmHWM"*) echo "VmHWM: 300000 kB";; *"RTF"*|*"offline"*) echo "Real time factor (RTF): 0.9 / 10.0 = 0.090";; *) echo 0;; esac; else adb ${SERIAL:+-s $SERIAL} "$@"; fi; }
sh_() { adb_ shell "$@" | tr -d '\r'; }
mb() { du -m "$@" 2>/dev/null | awk '{s+=$1} END{print s+0}'; }

# ---- step 0: preflight on the laptop ---------------------------------------------------------------------------------
[ "$SESSION" = 1 ] || [ "$SESSION" = 2 ] || die "session must be 1 or 2"
if [ "$DRY" != 1 ]; then
  command -v adb >/dev/null || die "adb not in PATH (winget install Google.PlatformTools)"
  [ -x "$BIN_DIR/bin/sherpa-onnx-offline" ] || die "missing $BIN_DIR/bin/sherpa-onnx-offline (see docs/DEVICE_SESSION.md, extract the termux-static tarball)"
  [ "$SESSION" = 1 ] || [ -x "$ORT_PERF" ] || die "missing $ORT_PERF: build onnxruntime_perf_test for Android arm64 first (docs/DEVICE_SESSION.md)"
fi
mkdir -p "$OUTDIR"; : > "$JSONL"

# ---- step 1: adb devices ---------------------------------------------------------------------------------------------
[ "$(adb_ get-state | tr -d '\r')" = "device" ] || die "adb devices: no authorized device (USB debugging on? accepted the RSA prompt?)"
trap 'echo "cleaning device dir"; adb_ shell rm -rf '"$DEV"' >/dev/null 2>&1 || true' EXIT
ANDROID="$(sh_ getprop ro.build.version.release | head -n1)"
echo "device android=$ANDROID"

# ---- step 2: free space ----------------------------------------------------------------------------------------------
NEED=$([ "$SESSION" = 1 ] && echo $NEED_MB_S1 || echo $NEED_MB_S2)
FREE_KB="$(sh_ df /data/local/tmp | awk 'NR==2{print $4}')"
[ "${FREE_KB:-0}" -gt $((NEED * 1024 * 2)) ] || die "free space check: need >= 2x${NEED} MB on /data/local/tmp, have ${FREE_KB:-?} kB"

# ---- helpers: temps, push, run ---------------------------------------------------------------------------------------
temps() {  # prints a JSON object: battery temp (tenth C) + first 8 thermal zones (raw)
  local bt tz="" z
  bt="$(sh_ dumpsys battery | sed -n 's/^ *temperature: *//p' | head -n1)"
  for z in $(sh_ 'ls -d /sys/class/thermal/thermal_zone* 2>/dev/null' | head -n 8); do
    tz+="${tz:+,}\"$(basename "$z")\":\"$(sh_ "cat $z/temp 2>/dev/null" | head -n1)\""
  done
  echo "{\"battery_temp_tenth_c\":\"${bt:-}\",\"zones\":{${tz}}}"
}
push_if_new() {  # src dst : skip when remote size equals local size
  local src="$1" dst="$2" ls rs
  [ "$DRY" = 1 ] && { echo "[dry] push_if_new $src -> $dst" >&2; return 0; }
  ls=$(wc -c < "$src" | tr -d ' '); rs=$(sh_ "stat -c %s $dst 2>/dev/null" | head -n1)
  [ "$ls" = "${rs:-x}" ] || adb_ push "$src" "$dst" >/dev/null || die "push $src"
}
log() { echo "$1" >> "$JSONL"; }

adb_ shell mkdir -p $DEV/bin $DEV/asr $DEV/wavs || die "mkdir device dir"
log "{\"event\":\"start\",\"session\":$SESSION,\"android\":\"$ANDROID\",\"mem_available_kb\":\"$(sh_ "grep -m1 MemAvailable /proc/meminfo" | tr -dc '0-9')\",\"airplane_mode_on\":\"$(sh_ settings get global airplane_mode_on | head -n1)\",\"temps\":$(temps)}"

# ---- step 3: push model + binary -------------------------------------------------------------------------------------
push_if_new "$BIN_DIR/bin/sherpa-onnx-offline" $DEV/bin/sherpa-onnx-offline
for f in encoder-epoch-12-avg-8.int8.onnx decoder-epoch-12-avg-8.int8.onnx joiner-epoch-12-avg-8.int8.onnx tokens.txt; do push_if_new "$ASR_DIR/$f" "$DEV/asr/$f"; done
WAVS=$(ls "$ASR_DIR"/test_wavs/*.wav 2>/dev/null | head -n 3)
[ -n "$WAVS" ] || [ "$DRY" = 1 ] || die "no test wavs under $ASR_DIR/test_wavs"
for w in $WAVS; do push_if_new "$w" "$DEV/wavs/$(basename "$w")"; done
adb_ shell chmod +x $DEV/bin/sherpa-onnx-offline || die "chmod"

# ---- step 4: run -----------------------------------------------------------------------------------------------------
asr_run() {  # repeat index
  local out rtf t0 t1 pid hwm=0 v
  t0="$(temps)"
  ( adb_ shell "cd $DEV && ./bin/sherpa-onnx-offline --encoder=asr/encoder-epoch-12-avg-8.int8.onnx --decoder=asr/decoder-epoch-12-avg-8.int8.onnx --joiner=asr/joiner-epoch-12-avg-8.int8.onnx --tokens=asr/tokens.txt --num-threads=2 --decoding-method=greedy_search $(for w in $WAVS; do printf 'wavs/%s ' "$(basename "$w")"; done) 2>&1" ) > "$OUTDIR/asr_run$1.txt" || die "asr run $1"
  t1="$(temps)"
  rtf="$(grep -o 'RTF): [0-9.]* / [0-9.]* = [0-9.]*' "$OUTDIR/asr_run$1.txt" | tail -n1 | awk '{print $NF}')"
  [ -n "$rtf" ] || die "asr run $1: RTF not found in output (see $OUTDIR/asr_run$1.txt)"
  log "{\"event\":\"asr_rtf\",\"run\":$1,\"threads\":2,\"rtf\":$rtf,\"temps_before\":$t0,\"temps_after\":$t1}"
}
for i in 1 2 3 4 5; do asr_run $i; done   # session 1: k = RTF_phone / RTF_laptop(0.040, results/asr_vi_rtf.json)

if [ "$SESSION" = 2 ]; then
  # network: byte counters on all interfaces before/after the whole session part (expect ~0 for airplane mode)
  NET0="$(sh_ cat /proc/net/dev | awk 'NR>2{rx+=$2;tx+=$10} END{print rx+0","tx+0}')"
  # load time + RSS (VmHWM polled from /proc/<pid>/status while a run is alive)
  adb_ shell "cd $DEV && (./bin/sherpa-onnx-offline --encoder=asr/encoder-epoch-12-avg-8.int8.onnx --decoder=asr/decoder-epoch-12-avg-8.int8.onnx --joiner=asr/joiner-epoch-12-avg-8.int8.onnx --tokens=asr/tokens.txt --num-threads=2 wavs/$(basename "$(echo $WAVS | cut -d' ' -f1)") & p=\$!; m=0; while kill -0 \$p 2>/dev/null; do v=\$(grep VmHWM /proc/\$p/status 2>/dev/null | tr -dc 0-9); [ -n \"\$v\" ] && [ \$v -gt \$m ] && m=\$v; done; echo PEAK_HWM_KB=\$m)" > "$OUTDIR/asr_rss.txt" || die "asr rss"
  log "{\"event\":\"asr_rss\",\"raw\":\"$(tr '\n' ' ' < "$OUTDIR/asr_rss.txt" | sed 's/"/\\"/g')\"}"
  # NMT: encoder and one decoder step, ORT perf_test, 2 threads (models: fixed random inputs; composed E2E = enc + n_tokens * step, label "composed (est.)")
  push_if_new "$ORT_PERF" $DEV/bin/onnxruntime_perf_test
  push_if_new "$NMT_DIR/encoder_model_quantized.onnx" $DEV/enc.onnx
  push_if_new "$NMT_DIR/decoder_model_merged_quantized.onnx" $DEV/dec.onnx
  adb_ shell chmod +x $DEV/bin/onnxruntime_perf_test || die "chmod perf_test"
  for m in enc dec; do
    t0="$(temps)"
    adb_ shell "cd $DEV && ./bin/onnxruntime_perf_test -e cpu -x 2 -S 1 -r 50 -I $m.onnx $m.csv 2>&1" > "$OUTDIR/nmt_$m.txt" || die "nmt $m perf_test (input shapes may need -I / dim overrides: see docs/DEVICE_SESSION.md)"
    log "{\"event\":\"nmt_$m\",\"raw_file\":\"nmt_$m.txt\",\"temps_before\":$t0,\"temps_after\":$(temps)}"
  done
  # TTS via sherpa-onnx-offline-tts
  push_if_new "$BIN_DIR/bin/sherpa-onnx-offline-tts" $DEV/bin/sherpa-onnx-offline-tts
  adb_ push "$TTS_DIR" $DEV/tts >/dev/null || die "push tts model"
  adb_ shell chmod +x $DEV/bin/sherpa-onnx-offline-tts
  for i in 1 2 3 4 5; do
    t0="$(temps)"
    adb_ shell "cd $DEV && ./bin/sherpa-onnx-offline-tts --vits-model=tts/en_US-ljspeech-medium.onnx --vits-tokens=tts/tokens.txt --vits-data-dir=tts/espeak-ng-data --num-threads=2 --output-filename=/data/local/tmp/tonebridge/o.wav 'Take two tablets of paracetamol every six hours.' 2>&1" > "$OUTDIR/tts_run$i.txt" || die "tts run $i"
    log "{\"event\":\"tts\",\"run\":$i,\"raw_file\":\"tts_run$i.txt\",\"temps_before\":$t0,\"temps_after\":$(temps)}"
  done
  NET1="$(sh_ cat /proc/net/dev | awk 'NR>2{rx+=$2;tx+=$10} END{print rx+0","tx+0}')"
  log "{\"event\":\"network_counters\",\"rx_tx_before\":\"$NET0\",\"rx_tx_after\":\"$NET1\",\"note\":\"all interfaces incl. loopback; compare deltas\"}"
fi

# ---- step 5: pull + cleanup ------------------------------------------------------------------------------------------
log "{\"event\":\"end\",\"temps\":$(temps)}"
echo "results in $OUTDIR (JSONL: $JSONL). Device dir is removed by the exit trap. Now turn USB debugging OFF."
