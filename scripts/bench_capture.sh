#!/usr/bin/env bash
# The stage-14 live bench, as a script that RECORDS what it observed.
#
#     ./scripts/bench_capture.sh                       # all four runs
#     ./scripts/bench_capture.sh serial_logger         # one run by name
#     PORT=COM4 DURATION=60 ./scripts/bench_capture.sh
#
# NEEDS THE PHYSICAL BOARD. Every run below opens a serial port or a BLE
# connection to the insole. There is no simulated path here on purpose: this
# script exists to produce hardware evidence, and a fake would produce a log
# that looks like hardware evidence and is not. Without a board the runs fail
# at the transport and the script says so.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The 2026-09-03 bench was run as four commands typed by hand. Their logs end
# with read_serial's summary line and nothing else, so THE EXIT CODES WERE
# NEVER RECORDED. The "exit" column in docs/hardware_notes.md was DERIVED
# afterwards: the counters in each log were fed back through
# read_serial.exit_code, which returns 0 for all four. That derivation is
# sound -- exit_code is a pure function of exactly those counters -- but it is
# a reconstruction, not an observation, and a reconstruction cannot catch the
# case it most needs to: the process exiting nonzero for a reason the counters
# do not describe (an unhandled exception on the way out, a signal, a
# write that failed after the summary printed).
#
# So every run here captures `$?` immediately after the command and writes it
# into the log as its last line:
#
#     exit=0
#
# A future bench then has the code as an observation. Nothing else about the
# runs changes; the commands are the ones that were typed on the bench day.
#
# The stalled BLE attempts are worth capturing too -- a nonzero code recorded
# against a stall is the evidence that matters -- so a failing run writes its
# log and its code and the script keeps going to the next run.
set -uo pipefail                 # NOT -e: a nonzero exit is data here, not a stop

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
PORT=${PORT:-COM3}
DURATION=${DURATION:-60}
OUT=${OUT:-data/bench}

mkdir -p "$OUT"

# name | source | program | extra args
RUNS="
serial_logger|serial|read_serial|
ble_logger|ble|read_serial|
serial_preds|serial|infer_live|
ble_preds|ble|infer_live|
"

run_one() {
    local name=$1 source=$2 prog=$3
    local log="$OUT/$name.log" csv="$OUT/$name.csv"
    local cmd

    if [ "$prog" = "read_serial" ]; then
        cmd=("$PYTHON" -m insole.read_serial --source "$source" --port "$PORT"
             --duration "$DURATION" "$csv")
    else
        cmd=("$PYTHON" -m insole.infer_live --source "$source" --port "$PORT"
             --duration "$DURATION" --out "$csv")
    fi

    echo "== $name: ${cmd[*]}"
    # The command's own stdout is the log, as on the bench day. Its exit status
    # is captured the instant it returns, before anything else can clobber $?.
    "${cmd[@]}" > "$log" 2>&1
    local code=$?
    # THE POINT OF THIS SCRIPT: the observation, not a later derivation.
    echo "exit=$code" >> "$log"
    echo "   -> exit=$code  ($log)"
    return $code
}

want=${1:-}
failed=0
printf '%s\n' "$RUNS" | while IFS='|' read -r name source prog _rest; do
    [ -z "$name" ] && continue
    [ -n "$want" ] && [ "$want" != "$name" ] && continue
    run_one "$name" "$source" "$prog" || failed=$((failed + 1))
done

echo
echo "Each log's last line is now 'exit=<code>', recorded at the moment the"
echo "command returned. Cross-check it against read_serial.exit_code applied to"
echo "the summary line: if the two ever disagree, the recorded one is the fact"
echo "and the derivation is what needs explaining."
