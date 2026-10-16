# Observed-availability testbed (IoTDB 1.3.7 + OPC UA)

Design and results: see `../README.md`.
(§8 binding, §9 decisions); output format: `../INTERFACE.md`.

Locations (environment variables; defaults assume this folder sits at `<repo root>/observed_testbed/testbed/`):

| Variable | Default | Used by |
|---|---|---|
| `TEP_DATA_DIR` | `<repo root>/data/tep` (written by `scripts/fetch_tep.py`) | `tep_data.py` |
| `TEP_SHA256_CSV` | `<repo root>/config/tep_sha256.csv` (columns `file`, `sha256`; `bytes` ignored) | `tep_data.py` |
| `TESTBED_IOTDB_HOME` | `testbed/vendor/iotdb/apache-iotdb-1.3.7-all-bin` | `iotdb_instance.py` |
| `TESTBED_PYTHON` | `testbed/.venv/bin/python` | `run_all.sh` |

| File | Purpose |
|---|---|
| `iotdb_instance.py` | isolated localhost IoTDB standalone (own conf/data/logs in `instances/<name>/`, own port block); flush/compaction settings |
| `historian_io.py` | IoTDB session helpers; tick ↔ time (T0 = 2023-08-31Z, one 7-day partition) |
| `tep_data.py` | loads a TEP file after checking its sha256 against the paper's source inventory |
| `opcua_source.py` | asyncua server replaying TEP; SourceTimestamp = process time, ServerTimestamp stamped at write |
| `collector.py` | OPC UA subscription (publishing interval 0, trigger StatusValueTimestamp) → IoTDB writes, FLUSH, membership query, digests |
| `query_client.py` | origin snapshots (window, anchor, commit_anchor) through its own IoTDB session |
| `run_one.py` | one run with the §8 barrier; writes `runs/<run_id>/` |
| `run_list.py`, `run_all.sh` | 40 runs + 4 repeats (or `--smoke`), parallel under taskset, markers `runs/ALL_DONE` / `runs/SMOKE_ALL_DONE` |
| `check_run.py` | format and internal-consistency check of a run directory (columns, rows per tick, digests from deltas) |
| `compare_runs.py` | determinism check of a repeat against its original (wall-clock fields dropped) |
| `calibrate_compdev.py` | CompDev calibration with IoTDB at F = 15 on batch 1; `config/compdev.json`, `config/achieved_retention.json` |
| `boundary_smoke.py`, `boundary_probe.py`, `boundary_multiseries_probe.py`, `boundary_summarize.py`, `boundary_report.py` | boundary smoke and probes; findings in `BOUNDARY_SMOKE.md` |
| `CHANNELS.md` | the five channels and why |
| `vendor/` | IoTDB 1.3.7 all-bin (sha512-verified download in `vendor/download/`) |
| `.venv/` | Python 3.12 venv: asyncua 2.1.0, apache-iotdb 1.3.4, pandas, openpyxl |
| `debug_runs/` | debugging and timing runs (not part of the design) |

`duplicate_rows_observed` (run_meta): per channel, rows in the final post-flush membership query minus distinct
(row_time, value) pairs, i.e. the extra stored copies from one-row SDT flush segments (BOUNDARY_SMOKE.md finding 5).
`tick_wall_ms` (run_meta): minimum wall time per tick; 0 = no pacing (ticks advance as fast as the barrier allows).
`memtable_flushes_logged` (run_meta): memtable flushes in the instance's datanode log ("The compression ratio of
tsfile ..."); run_one.py fails the run and check_run.py fails it unless this equals the files sealed by scheduled
FLUSHes (flushes.csv tsfiles_after − tsfiles_before, which must also be continuous and 0 or 1 per FLUSH).
`run_all.sh --overwrite` re-runs ids whose run directories already exist (old contents deleted); without it
run_one.py refuses to overwrite.
