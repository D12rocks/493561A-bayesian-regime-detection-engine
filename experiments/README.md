# Experiments Registry & Artifact Catalog

> **CONFIDENTIAL** — Zetheta Algorithms Private Limited

This directory contains the immutable experiment records generated during model exploration, tuning, calibration, and backtesting.

- `registry/`: Contains individual `<experiment_id>.json` files and master index catalog `index.json`.
- `configs/`: Standalone configuration snapshots used for specific experimental trials.
- `results/`: Aggregated performance benchmark tables.

All experiments must be initiated and recorded using `src.utils.experiment_tracker.ExperimentTracker`. Untracked runs are forbidden.
