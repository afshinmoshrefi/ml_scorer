# ML Pattern Scorer V2 -- Codex Training Review Results (Round 3)

## Medium

### 1. `pat_hit_last_year` is still computed with different trading-day resolution in training vs production

- **Files:** `build_training_data.py`, `ml_scorer/feature_engine.py`
- **Lines:** `build_training_data.py:903-941`, `ml_scorer/feature_engine.py:145-153`, `ml_scorer/feature_engine.py:484-505`
- **Problem:** The Round 2 neighborhood-feature skew is fixed, but `pat_hit_last_year` still is not. Training builds the prior-year pattern from the replayed entry date and then searches **forward** to the next trading day for both entry and exit (`offset in range(0, 4)` / `range(0, 5)`). Production uses `_get_price_on_date()`, which searches **backward** up to 5 days on or before the requested date. That means a weekend/holiday pattern can train on one prior-year path and score live on a different one. Feb. 29 also diverges: production maps it to Feb. 28 in non-leap years, while training falls through to `NaN`.
- **Why it matters:** This is still train/serve skew on a deployed feature. It does not affect every sample, but when it does, the model sees a different historical success flag in production than it saw during training.
- **Suggested fix:** Make production `pat_hit_last_year` reuse the training convention exactly: construct the prior-year base date, search forward for the next trading day for entry and exit, and handle Feb. 29 the same way on both sides. If you prefer the production convention, then update training to match and retrain.

## Low

### 2. `cal_day_of_year` is misnormalized in leap years

- **Files:** `build_training_data.py`, `ml_scorer/feature_engine.py`
- **Lines:** `build_training_data.py:1050`, `ml_scorer/feature_engine.py:1306`
- **Problem:** Both pipelines compute `cal_day_of_year` as `dayofyear / 365.0`. In leap years, Dec. 31 becomes `366 / 365 = 1.0027`, so a feature that appears intended to be a normalized seasonal position can exceed 1.0 for the entire leap-year back half.
- **Why it matters:** This is not a train/serve mismatch, but it does inject a small calendar distortion into one of the 59 model inputs.
- **Suggested fix:** Normalize by the actual number of days in the year (`365` or `366`) or store raw ordinal day and let the model learn the scaling.

### 3. Walk-forward prediction files still drop cohort metadata needed for calibration review

- **Files:** `train_model.py`
- **Lines:** `train_model.py:1051-1060`, `train_model.py:1084-1093`, `train_model.py:1099-1151`
- **Problem:** The saved walk-forward parquet still contains only `val_year`, prediction, and target columns. It omits `date`, `symbol`, `daysOut`, and `direction`, so you cannot audit whether calibration drift or tail-range issues are concentrated in a particular side, holding-period cohort, or calendar regime.
- **Why it matters:** This is not breaking model training, but it weakens the validation loop. The new calibration range check tells you that overflow exists; the saved parquet still does not let you localize where it comes from.
- **Suggested fix:** When `save_predictions` is enabled, persist the corresponding metadata columns from the validation slice alongside the predictions. That keeps the current calibration flow intact while making post-hoc diagnostics possible.

## Summary

No new Critical or High findings remained after the Round 2 fixes. The main remaining issue is a narrower `pat_hit_last_year` train/serve skew; the other items are lower-risk validation and calendar correctness issues.
