# Audit summary: f14_b09_k5_F01

Status: **ok [valid; primary match 0/14965 cells; visible match 0/14965 cells]**

Validation errors 0, warnings 0; digests checked 15010.

Origins 8..3000 (2993); first flush tick 0; origins before first flush 0; truth-undetermined cells 0.

| comparison | cells | excluded (pre-flush / undet.) | row-set mm | feature mm cells | cell mm rate | max abs diff | longest run |
|---|---|---|---|---|---|---|---|
| Replay_commit_vs_S_commit | 14965 | 0 / 0 | 0 | 0 | 0.00% | – | – |
| Replay_visible_vs_S_all | 14965 | 0 / 0 | 0 | 0 | 0.00% | – | – |
| V3_vs_S_commit | 14965 | 0 / 0 | 0 | 0 | 0.00% | – | – |
| V_server_vs_S_commit | 14965 | 0 / 0 | 0 | 0 | 0.00% | – | – |
| V1_vs_S_commit | 14965 | 0 / 0 | 0 | 0 | 0.00% | – | – |
| S_all_vs_S_commit | 14965 | 0 / 0 | 0 | 0 | 0.00% | – | – |

Achieved retention (mean over channels): 1.0000.
Commit lag c_obs − τ: p50 0.0, p90 0.0, max 0.0 (n 15005; committed only at final flush 0).
Ledger violations 0, anomalies 0; snapshot violations 0.
V_server equals V3 everywhere: True.
Duplicate rows (identical copies, collapsed for R): membership rows stored more than once 15005; window multiplicity agreement 14965/14965 cells.
Negative checks (a shift / b future-time / b future-commit): detected / detected / not_applicable.
SDT overlap (Jaccard, IoTDB vs whole-series local rule; IoTDB resets SDT at every FLUSH): XMEAS-21 0.067, XMEAS-4 0.086, XMEAS-7 0.089, XMEAS-9 0.031, XMV-10 0.034.
