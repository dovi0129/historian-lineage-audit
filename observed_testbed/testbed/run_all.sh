#!/usr/bin/env bash
# Launch all testbed runs (40 + 4 repeats) in parallel; each run has its own IoTDB instance, ports and data dir.
#   run_all.sh [-P N]            full design, N concurrent runs (default 8; choose from uptime/ps before launch)
#   run_all.sh --smoke [-P N]    three 300-tick runs + one repeat each (ids smoke_*)
#   --overwrite                  re-run ids whose run directories already exist (their old contents are deleted)
# Everything runs under taskset -c 0-15,32-47 (inherited by the JVMs). Per-run outputs: runs/<run_id>/ (INTERFACE.md);
# launcher logs: runs/_launcher/. Gates: (1) every run exits 0 and writes DONE, (2) check_run.py passes for every
# run, (3) compare_runs.py finds every repeat identical to its original. Marker runs/ALL_DONE (full) or
# runs/SMOKE_ALL_DONE (smoke) is written only if all three gates pass; otherwise the failed gates are logged and the
# launcher exits 1. A marker left by an earlier launch of the same mode is removed at start.
set -uo pipefail
TB="$(cd "$(dirname "$0")" && pwd)"
RUNS="$(cd "$TB/../runs" && pwd)"
PY="${TESTBED_PYTHON:-$TB/.venv/bin/python}"  # interpreter with asyncua and apache-iotdb
if [ "${RUN_ALL_PINNED:-}" != 1 ]; then
  RUN_ALL_PINNED=1 exec taskset -c 0-15,32-47 "$0" "$@"
fi
P=8
MODE=full
OVERWRITE=
while [ $# -gt 0 ]; do
  case "$1" in
    -P) P="$2"; shift 2 ;;
    --smoke) MODE=smoke; shift ;;
    --overwrite) OVERWRITE=--overwrite; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
L="$RUNS/_launcher"
mkdir -p "$L"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LIST="$L/${MODE}_runs_$STAMP.txt"
if [ "$MODE" = smoke ]; then
  (cd "$TB" && "$PY" run_list.py --smoke) > "$LIST"
  MARK="$RUNS/SMOKE_ALL_DONE"
else
  (cd "$TB" && "$PY" run_list.py) > "$LIST"
  MARK="$RUNS/ALL_DONE"
fi
LOG="$L/${MODE}_$STAMP.log"
STATUS="$L/${MODE}_status_$STAMP.txt"
: > "$STATUS"
rm -f "$MARK"
echo "$(date -u +%FT%TZ) start mode=$MODE parallel=$P runs=$(wc -l < "$LIST") list=$LIST" | tee -a "$LOG"

run_one() {  # run_id fault batch level F n_ticks slot
  local rid="$1"
  "$PY" "$TB/run_one.py" --run-id "$rid" --fault "$2" --batch "$3" --level "$4" --flush-every "$5" \
    --n-ticks "$6" --slot "$7" $OVERWRITE > "$L/$rid.out" 2>&1
  local rc=$?
  echo "$rid $rc" >> "$STATUS"
  echo "$(date -u +%FT%TZ) $rid exit=$rc done=$([ -f "$RUNS/$rid/DONE" ] && echo yes || echo no)"
  return 0
}
export -f run_one
export TB RUNS PY L STATUS OVERWRITE
xargs -P "$P" -L 1 bash -c 'run_one "$@"' _ < "$LIST" | tee -a "$LOG"

fail_run=0; fail_check=0; fail_rep=0
while read -r rid _; do
  rc="$(awk -v r="$rid" '$1 == r {print $2}' "$STATUS")"
  if [ "$rc" != 0 ] || [ ! -f "$RUNS/$rid/DONE" ]; then
    echo "gate run: $rid exit=${rc:-none} DONE=$([ -f "$RUNS/$rid/DONE" ] && echo yes || echo no)" | tee -a "$LOG"
    fail_run=$((fail_run + 1))
  fi
  if (cd "$TB" && "$PY" check_run.py "$RUNS/$rid") >> "$LOG" 2>&1; then :; else
    echo "gate check_run: $rid FAILED" | tee -a "$LOG"
    fail_check=$((fail_check + 1))
  fi
done < "$LIST"
while read -r rid _; do
  case "$rid" in *_rep1) ;; *) continue ;; esac
  orig="${rid%_rep1}"
  echo "compare $orig vs $rid" >> "$LOG"
  if (cd "$TB" && "$PY" compare_runs.py "$RUNS/$orig" "$RUNS/$rid") >> "$LOG" 2>&1; then
    echo "determinism $orig: identical" | tee -a "$LOG"
  else
    echo "gate determinism: $orig vs $rid DIFFERENT" | tee -a "$LOG"
    fail_rep=$((fail_rep + 1))
  fi
done < "$LIST"
n_runs="$(wc -l < "$LIST")"
echo "gates: run failures=$fail_run check failures=$fail_check determinism failures=$fail_rep (of $n_runs runs)" \
  | tee -a "$LOG"
if [ "$fail_run" -eq 0 ] && [ "$fail_check" -eq 0 ] && [ "$fail_rep" -eq 0 ]; then
  echo "$(date -u +%FT%TZ) $n_runs runs passed all gates; list $LIST; log $LOG" > "$MARK"
  echo "$(date -u +%FT%TZ) wrote $MARK" | tee -a "$LOG"
  exit 0
fi
echo "$(date -u +%FT%TZ) gates failed; no marker" | tee -a "$LOG"
exit 1
