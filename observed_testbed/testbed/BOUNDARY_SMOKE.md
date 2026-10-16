# Boundary smoke: Apache IoTDB 1.3.7 SDT around FLUSH

Date: 2026-10-16. Scripts: `boundary_smoke.py` (16 cells), `boundary_probe.py` (follow-up),
`boundary_multiseries_probe.py` (finding 7),
`boundary_summarize.py`, `boundary_report.py`. Raw outputs: `boundary_smoke/raw.json`, `boundary_smoke/raw_probe.json`;
every query result of every cell rendered in `boundary_smoke/raw_appendix.md`. Excerpts below are verbatim from it.

## Setup

- IoTDB 1.3.7 standalone (`apache-iotdb-1.3.7-all-bin.zip` from downloads.apache.org; published sha512
  `cbe2a823…6225cbf80` equals the computed sha512). OpenJDK 21.0.12 (system). Python client `apache-iotdb==1.3.4`
  (latest 1.3.x on PyPI; no 1.3.7 client exists), `asyncua==2.1.0`, venv `testbed/.venv` (system Python 3.12.3).
- Instance (`iotdb_instance.py`): own conf/data/logs under `testbed/instances/<name>/`, own port block,
  all addresses 127.0.0.1. Automatic flush off (`enable_timed_flush_seq_memtable=false`,
  `enable_timed_flush_unseq_memtable=false`, both intervals 86400000000 ms); compaction off (seq, unseq, cross,
  auto-repair); one data region per database; `time_partition_interval=604800000`, origin 0, `timestamp_precision=ms`,
  `wal_mode=ASYNC`. The memory-triggered flush was at its default (`flush_proportion=0.4`; a cell writes < 1 MB).
  Correction (2026-10-16, after the code audit): an earlier version of this line named
  `avg_series_point_number_threshold`, which does not exist in the 1.3.7 jars. Runs now set `flush_proportion=0.75`
  and check for unscheduled flushes (sealed-TsFile counts per FLUSH, IoTDB's memtable-flush log count).
- Time: tick t is stored at `T0 + 60000·t` ms, T0 = 2023-08-31T00:00:00Z = start of time partition 2800; the TsFiles
  land in `sequence/root.bs/1/2800/`, i.e. one partition.
- DDL: `CREATE TIMESERIES root.bs.<cell>.x WITH DATATYPE=DOUBLE, ENCODING=PLAIN, DEADBAND=SDT, COMPDEV=<c>` (SDT on) or
  without the last two clauses (SDT off). `SHOW TIMESERIES` reports `SDT | {compdev=0.01}`; COMPMINTIME/COMPMAXTIME were
  not set and are not listed.
- Cells: single point (n = 1); straight line 0.1·t (n = 20); kink, t for t ≤ 9 then 9 (n = 20); TEP fault 1 batch 1
  XMEAS-7 ticks 0–44 (n = 45, CompDev 0.5). CompDev 0.01 for the synthetic cells. Each with F = 1 and F = 15, SDT on
  and off. One write per tick, full-series query after each write; FLUSH every F ticks from tick F−1 and once after the
  last write; full-series query right after FLUSH returns and again 1 s later; TsFile listing after each FLUSH.
  All cells share database `root.bs` (one device per cell, run one after another); IoTDB refused a ninth database
  ("no available DataRegionGroup"), which does not affect runs (one database per instance).

## Findings

1. **Before FLUSH every written row is returned.** In all 16 cells, the query right after each write returned the
   committed rows plus every row written since the last FLUSH, uncompressed (kink F = 15 at tick 13: ticks 0–13).
2. **After FLUSH the discarded rows are gone, at once.** The query issued when FLUSH returned already showed the
   compressed set (kink F = 15: `[0..14]` before, `(0,0.0), (9,9.0), (14,9.0)` after), and the result 1 s later was
   identical at every FLUSH of every cell. Each FLUSH that had data sealed one new `.tsfile` + `.tsfile.resource` pair
   (FLUSH returned in 2–52 ms). No discarded row stayed visible after FLUSH. Durability against a crash was not tested.
3. **The last row of each flushed segment is stored** — contrary to the documented "does not store the last point".
   Line F = 15: segment 0–14 kept `[0, 14]`; perfectly linear, so 14 is not a corridor break.
4. **SDT state does not carry across flushes.** Each flushed segment starts a new corridor and keeps its first row:
   line F = 15 final rows `[0, 14, 15, 19]`; kink F = 15 `[0, 9, 14, 15, 19]`; TEP F = 15
   `[0, 11, 14 | 15, 21, 23, 26, 29 | 30, 33, 35, 40, 42, 44]`. The retained set therefore depends on the flush
   schedule, with a floor of two rows per segment of ≥ 2 rows.
5. **A one-row segment is returned twice.** With SDT on, a flushed segment holding a single row is stored as two
   identical rows: raw queries return `(t, v)` twice, `count()` counts both, window queries return both, and
   `ORDER BY TIME DESC LIMIT 1` returns one. Hence with F = 1 every row appears twice (line F = 1: 40 rows for 20
   writes), and with F = 3 or 15 the trailing one-row segment is doubled (probe: F = 3 → `…, 14, 15, 15`, count 12;
   F = 15 → `0, 14, 15, 15`, count 4). F = 2 gave 16 rows for 16 writes, no duplicate. SDT off never duplicated.
6. **Committed rows never disappeared or changed** in any later query of any cell (542 queries checked as
   multisets of (time, `repr(value)`), 0 violations).
7. **A multi-series query hangs once a series holds a doubled row** (found when the first calibration failed;
   `boundary_multiseries_probe.py`). Five SDT series (CompDev 1e-6, TEP fault 1 batch 1), F = 15. With 300 rows
   (20 full segments, no doubled row) every query returned in ≤ 0.06 s. With 301 rows (tick 300 alone in the last
   segment, so doubled) the single-series queries returned 302 rows in 0.06 s and 0.01 s, but
   `SELECT XMEAS_7, XMEAS_9, XMV_10, XMEAS_4, XMEAS_21 FROM root.q.d` (with or without `ORDER BY TIME`) failed after
   the 60 s query timeout: `301: DriverTask … is aborted by timeout`. The testbed therefore issues one query per
   series for membership, windows and anchors (`historian_io.py`); the per-series results are the same rows.

## Consequences for the spec (for the manager)

- The spec's documented-behaviour assumption (§2: last point not stored) does not hold for 1.3.7; the observed
  rule is "first and last row of every flushed segment plus SDT corridor breaks inside it".
- Achievable retention per F (floor 2/F): F = 1 → all rows (each twice); F = 5 → ≥ 2/5, so 1/5 is infeasible;
  F = 15 and F = 60 → both targets reachable. CompDev frozen at F = 15 will give different retention at F = 5 and 60.
- Duplicate rows will appear in membership queries and snapshots whenever a flushed segment has one row (all SDT
  F = 1 runs, and any trailing one-row segment). `membership_digest` lines would contain the duplicate line twice.

## Raw excerpts

```
kink_f15_sdt  (DDL ... DEADBAND=SDT, COMPDEV=0.01)
write t=13 value=9.0  query -> ticks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
write t=14 value=9.0  query -> ticks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
FLUSH t=14 returned in 0.0055 s; query -> (0,0.0), (9,9.0), (14,9.0)
         same result 1 s later: True; tsfile+resource files: 178 -> 178
write t=15 value=9.0  query -> ticks [0, 9, 14, 15]
write t=19 value=9.0  query -> ticks [0, 9, 14, 15, 16, 17, 18, 19]
FLUSH t=20 returned in 0.0056 s; query -> (0,0.0), (9,9.0), (14,9.0), (15,9.0), (19,9.0)

single_f1_sdt
write t= 0 value=1.0  query -> ticks [0]
FLUSH t= 0 returned in 0.0524 s; query -> (0,1.0), (0,1.0)

probe, line 0.1·t, n = 16, SDT on, F = 15
SELECT x         -> ['1693440000000\t\t0.0', '1693440840000\t\t1.4', '1693440900000\t\t1.5', '1693440900000\t\t1.5']
SELECT count(x)  -> 4.0
```
