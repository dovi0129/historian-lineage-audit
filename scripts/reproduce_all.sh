#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CPUS='0-15,32-47'
JOBS=16
SMOKE=0
while (($#)); do
  case "$1" in
    --cpus) CPUS="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --smoke) SMOKE=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'jobs must be positive' >&2; exit 2; }
export PYTHONPATH="$ROOT/src"
export REPRO_JOBS="$JOBS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$ROOT/artifacts/mpl-cache"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$ROOT/artifacts/logs" "$ROOT/tables/generated" "$ROOT/figures/generated"
cd "$ROOT"
run() {
  local name="$1"; shift
  echo "Running $name"
  taskset -c "$CPUS" "$PYTHON" "$@" > "artifacts/logs/$name.log" 2>&1
}
rm -f artifacts/REPRODUCTION_DONE artifacts/SMOKE_DONE
run example scripts/run_lineage_example.py --output artifacts/core_revision/example_audit.json
run example_missing scripts/run_lineage_example.py --without-clock --output artifacts/core_revision/example_missing_clock.json
if ((SMOKE)); then
  run fetch_smoke scripts/fetch_tep.py --smoke
  run parent_smoke scripts/101_historian_vintage_replay_kill_test.py --smoke --workers "$JOBS"
  run lineage_smoke scripts/102_historian_reconstruction_lineage_differentiation.py --smoke --workers "$JOBS"
  run compare_smoke scripts/compare_to_frozen.py --smoke
  touch artifacts/SMOKE_DONE
  echo 'SMOKE_DONE'
  exit 0
fi
run fetch_check scripts/fetch_tep.py
run calibration scripts/calibrate_tep.py
run parent_selected scripts/101_historian_vintage_replay_kill_test.py --workers "$JOBS"
run lineage_selected scripts/102_historian_reconstruction_lineage_differentiation.py --workers "$JOBS"
run parent_full scripts/101_historian_vintage_replay_kill_test.py --workers "$JOBS" --tep-panel full2x2 --out artifacts/historian_vintage_replay_tep_full_panel_breadth
run lineage_full scripts/102_historian_reconstruction_lineage_differentiation.py --workers "$JOBS" --tep-panel full2x2 --parent artifacts/historian_vintage_replay_tep_full_panel_breadth --out artifacts/historian_reconstruction_lineage_tep_full_panel_breadth
run full_panel_verify scripts/103_historian_tep_full_panel_verify.py
run strengthen scripts/build_stage1_strengthening.py
run astra scripts/build_astra_revision_evidence.py
run fallback scripts/run_fallback_sensitivity.py
run contribution scripts/summarize_contribution_scope.py
run lineage_export scripts/export_manuscript_evidence.py
run inventory scripts/build_tep_source_inventory.py --check-distinct
run pressure_scale scripts/build_pressure_scale_context.py
run trajectory_direction scripts/summarize_trajectory_direction.py
run compare scripts/compare_to_frozen.py
touch artifacts/REPRODUCTION_DONE
echo 'REPRODUCTION_DONE'
