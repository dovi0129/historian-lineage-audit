# Observed-availability testbed (Apache IoTDB with SDT)

This folder adds an *observed* availability clock to the lineage-aware audit. Instead of reconstructing retention
decisions from a declared rule, it runs a real open-source historian that compresses and stores data itself, records
when each value was written, retained or discarded, and what a query returned at every forecast origin, and then
checks whether the audit can rebuild those query-time inputs from the recorded events alone.

## Design

- Historian: Apache IoTDB 1.3.7 (standalone, localhost), time series with `DEADBAND=SDT`; timed flush and compaction
  off, memory-triggered flush made unreachable. Retention decisions take effect when a `FLUSH` is issued.
- Source: an OPC UA server (asyncua 2.1.0) replays public Tennessee Eastman Process values (faults 1 and 14,
  batches 9 and 10; channels XMEAS-7, XMEAS-9, XMV-10, XMEAS-4, XMEAS-21). SourceTimestamp = process time; the
  ServerTimestamp field is stamped by the source at write.
- Collector: OPC UA subscription → IoTDB write; `FLUSH` every F ticks (F ∈ {1, 5, 15, 60}); a full membership query
  right after each flush. Query client: a snapshot at every origin (lag window 0–8 min plus anchors).
- A deterministic barrier orders each tick: publish → write ack → (flush → membership query) → origin snapshot.
- Grid: SDT CompDev calibrated at F = 15 for retained fractions 1/2 and 1/5, plus a no-deadband control at F ∈ {1, 15};
  40 runs and 4 determinism repeats. File formats: `INTERFACE.md`.

Three states are recorded separately for every sample: written and queryable (write ack), retained (first
post-flush membership query that contains it) or discarded (first post-flush query that lacks it), and what each
origin's query returned.

## What the audit compares

Two independent paths: the *ledger* (write acks and post-flush membership only) and the *snapshot truth* (origin
query results and the flush schedule only). At every origin, channel and lag (0, 2, 4, 6, 8 min) it compares row sets
and reconstructed features: ledger replay of committed rows vs the committed part of the snapshot (primary), ledger
replay of visible rows vs the full snapshot, and three proxy clocks (event time, OPC UA ServerTimestamp, final
export) vs the committed snapshot. Negative checks plant a shifted commit time and a future row.

## Results (44 runs; `results/pooled/summary.md`, `results/facts.json`)

- All runs valid; launcher gates (run, record check, determinism) passed.
- Ledger replay vs snapshot, origins 8–3000 (origins before the first flush excluded: 480 over the 40 conditions):
  0 mismatching cells in every run (596,200 committed cells and 598,600 visible cells over the 40 conditions). Committed rows
  never changed or disappeared later. Negative checks were detected wherever applicable.
- Proxy clocks vs committed snapshot (feature level): 0 % at F = 1, 24.0 % at F = 5, 66.6 % at F = 15, 91.6 % at
  F = 60, with and without SDT. These shares equal the fraction of (origin, lag) pairs whose feature time lies after
  the last flush, i.e. they follow from the flush schedule. In this implementation the ServerTimestamp field is
  stamped by the source at write and the proxy equalled the event-time proxy in every cell; neither timestamp records
  the archive commit.
- Pressure (XMEAS-7) differences over the mismatching features only, event-time proxy vs committed snapshot:
  median 0.57, 0.67 and 1.02 kPa at F = 5, 15 and 60 (`results/pressure_magnitude.csv`). These are input
  differences, not forecast errors.
- With SDT and F > 1, 36–90 % of written rows were queryable before a flush and deleted at the flush (none at F = 1).
- IoTDB 1.3.7 behaviour differs from a whole-series Swinging Door rule: SDT restarts at every flush (each flushed
  segment keeps its first and last rows), so the retained set depends on F; a one-row flushed segment is stored and
  returned twice (`testbed/BOUNDARY_SMOKE.md`). Neither clock is uniformly earlier: at F = 1 IoTDB commits at the
  event tick, one tick before the declared rule's decision; at F > 1 commits wait for the next flush.

Scope: one historian version, a deterministic single-process barrier, no network delay. The agreement shows that the
recorded events reproduce what the historian returned; it is not a measurement of physical persistence or crash
recovery.

## Reproduce

```bash
python3 scripts/fetch_tep.py                     # repo root: data/tep/, SHA-256 verified
# Install Apache IoTDB 1.3.7 all-bin (apache.org, verify sha512) into observed_testbed/testbed/vendor/iotdb/
# and a Python 3.12 venv at observed_testbed/testbed/.venv with asyncua==2.1.0, apache-iotdb==1.3.4, pandas, openpyxl
bash observed_testbed/testbed/run_all.sh --smoke -P 3      # three 300-tick runs plus repeats
bash observed_testbed/testbed/run_all.sh -P 16             # 44 runs, ~20 min on 32 logical CPUs
python3 observed_testbed/audit/run_audit.py --all observed_testbed/runs --out observed_testbed/results_new --jobs 8
```

Environment variables (defaults in `testbed/README.md`): `TEP_DATA_DIR`, `TEP_SHA256_CSV`, `TESTBED_IOTDB_HOME`,
`TESTBED_PYTHON`, `LINEAGE_SRC`. Audit tests: `python3 -m pytest observed_testbed/audit/tests`.

## Provenance

The runs used testbed code digest `898e18db64e33621` (recorded in every `run_records/*/run_meta.json`). This public
copy differs from the code that ran only in path configuration (environment variables with repository-relative
defaults); a re-run smoke and a full re-audit with the configured paths reproduced the original outputs. Machine-local
paths in records are replaced by `<observed_testbed>`. Raw per-run logs (about 850 MB) are not included.
