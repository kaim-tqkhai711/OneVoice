#!/usr/bin/env bash
# Probe an Android device over adb and write results/device_probe.json.
#
# Usage: tools/device_probe.sh [adb-serial]
# Pass: exit 0 and JSON file written. Fail: exit 1 (no device / adb missing); nothing is written.
# Unreadable values (no root) are recorded as null, never guessed.
set -euo pipefail

SERIAL="${1:-}"
OUT="results/device_probe.json"
ADB=(adb)
[ -n "$SERIAL" ] && ADB=(adb -s "$SERIAL")

command -v adb >/dev/null 2>&1 || { echo "adb not found in PATH" >&2; exit 1; }
[ "$("${ADB[@]}" get-state 2>/dev/null | tr -d '\r')" = "device" ] || { echo "no authorized device (check USB debugging)" >&2; exit 1; }

sh_() { "${ADB[@]}" shell "$@" 2>/dev/null | tr -d '\r'; }
esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
jstr() { if [ -z "$1" ]; then printf 'null'; else printf '"%s"' "$(esc "$1")"; fi; }
jnum() { if [[ "$1" =~ ^-?[0-9]+$ ]]; then printf '%s' "$1"; else printf 'null'; fi; }
prop() { sh_ getprop "$1" | head -n1; }

mkdir -p "$(dirname "$OUT")"

# CPU cores: max frequency per core in kHz (cpuinfo_max_freq, fallback scaling_max_freq).
N_CORES="$(sh_ 'ls -d /sys/devices/system/cpu/cpu[0-9]* 2>/dev/null | wc -l' | tr -d ' ')"
CORES=""
for ((i = 0; i < ${N_CORES:-0}; i++)); do
  f="$(sh_ "cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq 2>/dev/null" | head -n1)"
  [ -z "$f" ] && f="$(sh_ "cat /sys/devices/system/cpu/cpu$i/cpufreq/scaling_max_freq 2>/dev/null" | head -n1)"
  CORES+="${CORES:+,}{\"cpu\":$i,\"max_khz\":$(jnum "$f")}"
done

MEM_TOTAL="$(sh_ "grep -m1 MemTotal /proc/meminfo" | tr -dc '0-9')"
MEM_AVAIL="$(sh_ "grep -m1 MemAvailable /proc/meminfo" | tr -dc '0-9')"

BATT="$(sh_ dumpsys battery)"
bget() { printf '%s\n' "$BATT" | sed -n "s/^ *$1: *//p" | head -n1; }

# Thermal zones: first few readable temps (millidegrees C on most kernels).
TZ=""
for z in $(sh_ 'ls -d /sys/class/thermal/thermal_zone* 2>/dev/null' | head -n 8); do
  t="$(sh_ "cat $z/temp 2>/dev/null" | head -n1)"; ty="$(sh_ "cat $z/type 2>/dev/null" | head -n1)"
  TZ+="${TZ:+,}{\"zone\":$(jstr "$(basename "$z")"),\"type\":$(jstr "$ty"),\"temp_raw\":$(jnum "$t")}"
done

cat > "$OUT" <<EOF
{
  "probed_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "android_release": $(jstr "$(prop ro.build.version.release)"),
  "sdk_int": $(jnum "$(prop ro.build.version.sdk)"),
  "model": $(jstr "$(prop ro.product.model)"),
  "manufacturer": $(jstr "$(prop ro.product.manufacturer)"),
  "board_platform": $(jstr "$(prop ro.board.platform)"),
  "soc_model": $(jstr "$(prop ro.soc.model)"),
  "hardware": $(jstr "$(prop ro.hardware)"),
  "abi": $(jstr "$(prop ro.product.cpu.abi)"),
  "abilist": $(jstr "$(prop ro.product.cpu.abilist)"),
  "n_cores": $(jnum "${N_CORES:-}"),
  "cores": [${CORES}],
  "mem_total_kb": $(jnum "$MEM_TOTAL"),
  "mem_available_kb": $(jnum "$MEM_AVAIL"),
  "battery": {
    "level_pct": $(jnum "$(bget level)"),
    "status_code": $(jnum "$(bget status)"),
    "ac_powered": $(jstr "$(bget 'AC powered')"),
    "usb_powered": $(jstr "$(bget 'USB powered')"),
    "temperature_tenth_c": $(jnum "$(bget temperature)")
  },
  "thermal_zones": [${TZ}]
}
EOF
echo "wrote $OUT"
