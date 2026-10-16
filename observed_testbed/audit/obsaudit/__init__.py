"""Audit and comparison for the observed-availability testbed (spec §8).

Two independent paths:
- ledger (replay side): samples.csv, membership_*.csv, flushes.csv, run_meta.json
- truth (snapshot side): snapshots.csv, flushes.csv, run_meta.json
"""
