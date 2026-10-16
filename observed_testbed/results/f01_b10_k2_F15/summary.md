# Audit summary: f01_b10_k2_F15

Status: **ok [valid; primary match 0/14935 cells; visible match 0/14965 cells]**

Validation errors 0, warnings 0; digests checked 1005.

Origins 8..3000 (2993); first flush tick 14; origins before first flush 6; truth-undetermined cells 0.

| comparison | cells | excluded (pre-flush / undet.) | row-set mm | feature mm cells | cell mm rate | max abs diff | longest run |
|---|---|---|---|---|---|---|---|
| Replay_commit_vs_S_commit | 14935 | 30 / 0 | 0 | 0 | 0.00% | – | – |
| Replay_visible_vs_S_all | 14965 | 0 / 0 | 0 | 0 | 0.00% | – | – |
| V3_vs_S_commit | 14935 | 30 / 0 | 13935 | 13935 | 93.30% | 8.313396383584859 | 14.0 |
| V_server_vs_S_commit | 14935 | 30 / 0 | 13935 | 13935 | 93.30% | 8.313396383584859 | 14.0 |
| V1_vs_S_commit | 14935 | 30 / 0 | 13935 | 13935 | 93.30% | 8.313396383584859 | 14.0 |
| S_all_vs_S_commit | 14935 | 30 / 0 | 13935 | 13935 | 93.30% | 8.313396383584859 | 14.0 |

Achieved retention (mean over channels): 0.4947.
Commit lag c_obs − τ: p50 7.0, p90 14.0, max 14.0 (n 7418; committed only at final flush 5).
Ledger violations 0, anomalies 0; snapshot violations 0.
V_server equals V3 everywhere: True.
Duplicate rows (identical copies, collapsed for R): membership rows stored more than once 5; window multiplicity agreement 14965/14965 cells.
Negative checks (a shift / b future-time / b future-commit): detected / detected / detected.
SDT overlap (Jaccard, IoTDB vs whole-series local rule; IoTDB resets SDT at every FLUSH): XMEAS-21 0.796, XMEAS-4 0.797, XMEAS-7 0.786, XMEAS-9 0.804, XMV-10 0.796.
