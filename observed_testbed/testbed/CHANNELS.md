# Channels (fixed 2026-10-16, before CompDev calibration)

IoTDB series names replace `-` by `_` (`root.tep.d.XMEAS_7`, ...). Order below is the column order in every run.

| Channel | Role | Reason |
|---|---|---|
| XMEAS-7 | given by the brief | reactor pressure, the paper's pressure target |
| XMEAS-9 | given by the brief | reactor temperature, the paper's second target and second most frequent limiting channel |
| XMV-10 | given by the brief | reactor cooling water flow, the most frequent limiting channel in the paper |
| XMEAS-4 | added | A and C feed (stream 4), the measurement fault 1 (A/C feed ratio step) acts on directly |
| XMEAS-21 | added | reactor cooling water outlet temperature, the measurement fault 14 (cooling water valve sticking) acts on directly |

Both added channels belong to the paper's 34-channel semantic panel (XMEAS-1..22, XMV-1..12), and each fault gets one
channel it perturbs directly, so SDT sees a sustained change in every trajectory.
