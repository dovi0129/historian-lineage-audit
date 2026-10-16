# Audit summary: f01_b09_k5_F60

Status: **ok [valid; primary match 0/14710 cells; visible match 0/14965 cells]**

Validation errors 0, warnings 0; digests checked 255.

Origins 8..3000 (2993); first flush tick 59; origins before first flush 51; truth-undetermined cells 0.

| comparison | cells | excluded (pre-flush / undet.) | row-set mm | feature mm cells | cell mm rate | max abs diff | longest run |
|---|---|---|---|---|---|---|---|
| Replay_commit_vs_S_commit | 14710 | 255 / 0 | 0 | 0 | 0.00% | – | – |
| Replay_visible_vs_S_all | 14965 | 0 / 0 | 0 | 0 | 0.00% | – | – |
| V3_vs_S_commit | 14710 | 255 / 0 | 14460 | 14460 | 98.30% | 30.56852272704691 | 59.0 |
| V_server_vs_S_commit | 14710 | 255 / 0 | 14460 | 14460 | 98.30% | 30.56852272704691 | 59.0 |
| V1_vs_S_commit | 14710 | 255 / 0 | 14460 | 14460 | 98.30% | 30.86848159326837 | 59.0 |
| S_all_vs_S_commit | 14710 | 255 / 0 | 14460 | 14460 | 98.30% | 31.732457121419884 | 59.0 |

Achieved retention (mean over channels): 0.1188.
Commit lag c_obs − τ: p50 29.0, p90 59.0, max 59.0 (n 1777; committed only at final flush 5).
Ledger violations 0, anomalies 0; snapshot violations 0.
V_server equals V3 everywhere: True.
Duplicate rows (identical copies, collapsed for R): membership rows stored more than once 5; window multiplicity agreement 14965/14965 cells.
Negative checks (a shift / b future-time / b future-commit): detected / detected / detected.
SDT overlap (Jaccard, IoTDB vs whole-series local rule; IoTDB resets SDT at every FLUSH): XMEAS-21 0.484, XMEAS-4 0.514, XMEAS-7 0.490, XMEAS-9 0.523, XMV-10 0.401.
