# Retained-source worked example

`retained_sources.csv` contains source IDs, event times, reconstructed availability times, and values for one public TEP pressure channel. Run `python3 scripts/run_lineage_example.py`; the printed JSON is stored in `expected_output.json`. At origin 1498, the latest required source becomes available at 1499, so the 6 minute nominal horizon becomes 5 minutes and the requested feature is infeasible at that origin. The replayed current value is 2799.328221565883. The example uses retained records only; no raw trajectory is needed.
