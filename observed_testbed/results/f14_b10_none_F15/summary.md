# Audit summary: f14_b10_none_F15

Status: **ok [valid; primary match 0/14935 cells; visible match 0/14965 cells]**

Validation errors 0, warnings 0; digests checked 1005.

Origins 8..3000 (2993); first flush tick 14; origins before first flush 6; truth-undetermined cells 0.

| comparison | cells | excluded (pre-flush / undet.) | row-set mm | feature mm cells | cell mm rate | max abs diff | longest run |
|---|---|---|---|---|---|---|---|
| Replay_commit_vs_S_commit | 14935 | 30 / 0 | 0 | 0 | 0.00% | – | – |
| Replay_visible_vs_S_all | 14965 | 0 / 0 | 0 | 0 | 0.00% | – | – |
| V3_vs_S_commit | 14935 | 30 / 0 | 13935 | 13935 | 93.30% | 5.412092548887813 | 14.0 |
| V_server_vs_S_commit | 14935 | 30 / 0 | 13935 | 13935 | 93.30% | 5.412092548887813 | 14.0 |
| V1_vs_S_commit | 14935 | 30 / 0 | 13935 | 13935 | 93.30% | 5.412092548887813 | 14.0 |
| S_all_vs_S_commit | 14935 | 30 / 0 | 13935 | 13935 | 93.30% | 5.412092548887813 | 14.0 |

Achieved retention (mean over channels): 1.0000.
Commit lag c_obs − τ: p50 7.0, p90 13.0, max 14.0 (n 15000; committed only at final flush 5).
Ledger violations 0, anomalies 0; snapshot violations 0.
V_server equals V3 everywhere: True.
Duplicate rows (identical copies, collapsed for R): membership rows stored more than once 0; window multiplicity agreement 14965/14965 cells.
Negative checks (a shift / b future-time / b future-commit): detected / detected / detected.
