# Lineage-aware availability audit: public reproduction

This repository reproduces the paper's public CSTR and Tennessee Eastman Process (TEP) lineage audit. It carries retained source event and availability times into feature reconstruction, then compares completed-export and origin-available views on the same forecasting cells. The numerical reference in `reference/` is a frozen public-only snapshot; `scripts/compare_to_frozen.py` checks regenerated cells and displayed tables at relative tolerance `1e-8`, with exact counts and labels.

```bash
git clone https://github.com/dovi0129/historian-lineage-audit.git
cd historian-lineage-audit
```

Accompanying manuscript (under review): M. Baek, G. Gil, Y. Jang, and S. Lee, "When Does a Reconstructed Historian Feature Become
Reproducible? A Lineage-Aware Availability Audit for Process Forecasting."

## Repository map

| Path | Purpose |
| --- | --- |
| `src/` | Retention, reconstruction, CSTR trajectory, and replay logic |
| `scripts/101_*`, `102_*`, `103_*` | Parent audit, lineage extension, and full TEP panel regression check |
| `scripts/build_*`, `run_fallback_sensitivity.py`, `summarize_*` | Public tables, figures, and sensitivity analyses |
| `scripts/fetch_tep.py`, `config/tep_sha256.csv` | Frozen commit retrieval and file verification |
| `config/operator_calibration.csv`, `scripts/calibrate_tep.py` | Published TEP tolerances and their public-data recomputation |
| `examples/` | Supplement Section S6 one-channel worked example and expected JSON |
| `reference/` | Frozen public CSVs, summaries, and displayed tables for comparison |
| `data/tep/`, `artifacts/` | Downloaded inputs and regenerated outputs; excluded from version control |

## Environment and data

Python 3.11.14 with the pinned packages in `requirements.txt` was used. `taskset` is needed on Linux. Dependencies are ordinary Python packages; no private data or service is needed. The 60 selected TEP Mode-1 files are fetched from `Blound04/tennessee-eastman-dataset` at commit `309b944f35ac440ff0c70616947ffe723c766e14`, using the Git LFS media endpoint. Their SHA-256 and byte counts are fixed in `config/tep_sha256.csv`. Upstream TEP data are MIT licensed, copyright 2025 Marcus Vinicius Pereira.

From this repository root:

```bash
python3 scripts/fetch_tep.py
# If the same 60 verified files are already available locally:
python3 scripts/fetch_tep.py --from-local /path/to/tep/files
PYTHON=python3 scripts/reproduce_all.sh --cpus 0-15,32-47 --jobs 16
```

The launcher sets `PYTHONPATH`, limits numerical-library threads to one per worker, logs each step at `artifacts/logs/<step>.log`, and writes `artifacts/REPRODUCTION_DONE` only after all steps and the frozen comparison pass. It also accepts `--smoke`, which runs the known-answer example and the bounded CSTR/TEP parent and lineage jobs; `artifacts/SMOKE_DONE` marks that narrower run. Run `mkdir -p artifacts/mpl-cache && MPLCONFIGDIR=artifacts/mpl-cache PYTHONPATH=src python3 -m pytest -q tests --basetemp=artifacts/pytest_tmp` for the unit checks. If `taskset` is unavailable, the individual Python entry points can be called with the same environment variables.

The default CPU list contains 32 logical CPUs. The parent and lineage scripts parallelize independent seed and operator/severity units. `--jobs` caps their processes. A complete TEP Swinging Door k=2 selected-panel unit took 201.95 s on the shared host: 101.45 s reading all 60 files and 100.50 s computing one unit (see `artifacts/full_unit_timing.json` after the local check). One channel of calibration took 109.59 s including the same 60-file load; the full calibration parallelizes channels. With 16 jobs on the default CPU set, the projected full run is roughly **1–3 hours** under comparable shared-host contention. This is an extrapolation from bounded units, not a measured end-to-end run. The full calculation was left for the author to run.

## Output map

| Paper location | Regenerated output |
| --- | --- |
| Figure 2, Table of worked values, Supplement S6 | `artifacts/stage1_strengthening/worked_example_*`, `figures/generated/lineage_worked_example.*`, `tables/generated/astra_worked_values.tex`, `examples/expected_output.json` |
| Main geometry and event/commit figures | `figures/generated/lineage_pit_horizon.*`, `lineage_event_commit_effect.*` |
| Main pressure decision figure and fault-exclusion table | `figures/generated/astra_pressure_decisions.*`, `tables/generated/core_fault_exclusion.tex` |
| Supplement model, geometry, event/commit, remedy and fault tables | `tables/generated/lineage_*.tex`, `astra_*.tex` |
| Supplement pressure fault map and remedy figures | `figures/generated/lineage_pressure_fault_map.*`, `lineage_remedy_effects.*`, `lineage_remedy_categories.*` |
| Supplement fallback sensitivity and trajectory direction | `artifacts/core_revision/fallback/`, `tables/generated/core_fallback.tex`, `artifacts/manuscript_export/trajectory_direction_summary.csv` |
| Supplement TEP source inventory and pressure scale | `tables/generated/tep_source_inventory.tex`, `pressure_scale_context.tex` |

The main numerical cells are in `artifacts/historian_vintage_replay_*`, `artifacts/historian_reconstruction_lineage_*`, `artifacts/astra_revision/`, and `artifacts/manuscript_export/`. The comparison log names every checked CSV column and table.

## Scope

The release omits private paper-machine records and all analyses that require them. The original manuscript also contains an older, final-export TEP selection-context table produced by an earlier public benchmark pipeline outside this release's frozen 101–103 runner set; that table is not claimed as regenerated here. The local Swinging Door decision clock is a declared algorithmic construction, not an observed historian commit clock. The TEP batches are replicate trajectories; the upstream repository does not document independent simulation seeds. Full comparison status is established only when `REPRODUCTION_DONE` exists.
