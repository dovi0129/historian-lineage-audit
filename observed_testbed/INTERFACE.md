# Interface between the testbed  and the audit 

## Time
- 1 tick = 1 process minute. Tick t is TEP row t (0..3000). All CSV time columns below are integer ticks unless the
  column name ends in `_wall` (UTC ISO-8601 with microseconds) or `_ts` (OPC UA timestamp, UTC ISO-8601).
- The barrier order at tick t is fixed by spec §8: publish → receive → write ack → (flush → membership query) →
  origin snapshot for o = t. A final FLUSH and membership query are issued after the last tick; they are labelled
  tick = last_tick + 1 in `flushes.csv` and the membership files, and no snapshot exists at that tick.

## Files per run: `runs/<run_id>/`
1. `run_meta.json`: run_id, fault, batch, n_ticks (last tick = n_ticks − 1; smoke runs may be shorter than 3001),
   channels (list), deadband ("SDT" or "none"), compdev ({channel: float or
   null}), flush_every F, flush_ticks (list of ints, including the final flush), iotdb_version, iotdb_settings (dict,
   incl. automatic-flush settings), asyncua_version, tick_wall_ms, testbed_code_digest, created_wall.
2. `samples.csv`: run_id, channel, tick, value_sent, opc_source_ts, opc_server_ts, server_tick, recv_wall,
   write_req_wall, write_ack_wall, write_ack_tick. One row per channel per tick. `value_sent` is written with
   `repr(float)`. `server_tick` = the barrier tick during which the OPC UA server stamped the value.
3. `flushes.csv`: run_id, flush_id, tick, issue_wall, return_wall, tsfiles_before, tsfiles_after (number of sealed
   TsFiles in the instance's data directory immediately before issuing and after the return of this FLUSH). A run is
   invalid if tsfiles_before of a flush differs from tsfiles_after of the previous flush (an unscheduled flush
   happened) or if a FLUSH sealed more than one file.
4. `membership_delta.csv`: run_id, flush_id, tick, channel, row_time, value, change ("added" | "removed").
   The testbed queries the full series right after each FLUSH returns and writes only the difference from the
   previous membership query (the first one is all "added").
5. `membership_digest.csv`: run_id, flush_id, tick, channel, n_rows, digest. Digest convention (pinned): rows of the
   full post-flush query sorted numerically by row_time ascending; each line `f"{row_time},{repr(value)}"`; lines
   joined with "\n", no trailing newline; UTF-8; lowercase hex sha256. An empty series hashes the empty string.
6. `snapshots.csv`: run_id, origin, channel, row_time, value, kind, query_wall. For origin o and each channel:
   every row visible with o−8 ≤ time ≤ o (kind="window"); the latest visible row with time < o−8 (kind="anchor",
   omitted if none); and the latest visible row with time ≤ min(t_f(o), o−9) (kind="commit_anchor"), where t_f(o) is
   the tick of the last flush at or before o. Omit commit_anchor if no flush has occurred yet, if no such row exists,
   or if it is the same row as the anchor (then the anchor row serves both). The query client may use the flush
   schedule to compute t_f(o); it must not use membership results. Values as returned by IoTDB, written with `repr(float)`.
7. `events.log` (free text) and `DONE` marker written last.

Values are IoTDB DOUBLE. Nothing in the audit may read `snapshots.csv` when building the ledger, and nothing may read
`samples.csv` or membership files when building the snapshot truth (spec §8, two independent paths).

## Duplicate rows (IoTDB 1.3.7 one-row SDT segments; spec §9)
- Rows are written exactly as IoTDB returns them. `membership_delta.csv` has one line per stored copy (a doubled row
  is two "added" lines with the same row_time and value); the digest input contains the duplicate line twice
  (numeric sort, identical lines adjacent); `snapshots.csv` has one line per returned copy. Anchor queries
  (`ORDER BY TIME DESC LIMIT 1`) return one copy by construction.
- `run_meta.json` carries `duplicate_rows_observed` by channel.
- Audit: multiplicity is tracked and digests are verified over the multiset; identical duplicates collapse to one row
  for R and for row-set comparisons; same-time rows with different values are an error; duplicate counts are reported
  on both paths, and window-row multiplicities (not anchors) are compared descriptively.

## Changelog
- 2026-10-16: added commit_anchor rows, n_ticks, final-flush tick label, pinned digest convention.
- 2026-10-16: duplicate-row encoding and audit rule.
- 2026-10-16: flushes.csv gains tsfiles_before/tsfiles_after to detect unscheduled flushes (code audit finding 4).
