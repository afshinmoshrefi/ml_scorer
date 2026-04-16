# ML Pattern Scorer V2 -- Codex Training Review Results (Round 4)

## Findings

No new correctness findings identified in the current training-pipeline snapshot.

The Round 3 items appear to be fixed in the code now:

- `pat_hit_last_year` in production now matches training's forward trading-day lookup and Feb. 29 behavior.
- `cal_day_of_year` is normalized by the actual days in the year in both training and production.
- Walk-forward prediction parquet output now includes cohort metadata (`date`, `symbol`, `daysOut`, `direction`) when `--save-predictions` is enabled.

## Residual Risks / Gaps

- I did not benchmark the production-side `_compute_neighborhood_features()` latency impact; that remains a performance question, not a correctness bug.
- `config_ml.py` is still intentionally Windows-path-specific and not env-driven; the review brief treats that as context rather than a newly introduced defect.
- Calibration still relies on walk-forward predictions rather than final-model predictions by design, with the new overflow warning acting as a guardrail.

## Summary

Round 4 review did not uncover an additional bug to file. The previously reported training/serve skews and training-analysis gaps called out in R3 appear resolved in the current code.
