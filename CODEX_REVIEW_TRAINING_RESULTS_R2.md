# ML Pattern Scorer V2 -- Codex Training Review Results (Round 2)

## Critical

### 1. Several shared V2 pattern features are still defined differently in training and production

- **Files:** `build_training_data.py`, `ml_scorer/feature_engine.py`
- **Lines:** `build_training_data.py:662-678`, `build_training_data.py:774-885`, `ml_scorer/feature_engine.py:460-515`
- **Problem:** Multiple shared model inputs are still computed from different definitions in the training pipeline versus the production scorer:
  - `pat_concurrent_count`: training uses the total number of replayed patterns on the date for the symbol, including self (`date_pattern_counter[entry_date]`), while production scans only the first combo lookup and counts same-date keys excluding the current `(date, daysOut, LorS)` key.
  - `pat_neighbor_avg_wr`: training averages prior-year realized win rates from replayed price history; production averages shifted-date combo win rates from the current opp dataset.
  - `pat_sharpness`: training uses `pat_wr_prior / neighbor_avg`; production uses `best_winrate / neighbor_avg`.
  - `pat_post_cliff`: training uses `pat_wr_prior - mean(post1w, post2w)`; production uses `best_winrate - post1w`.
  - `pat_pre_slope` uses the same algebraic form, but the underlying neighbor win rates come from different data sources, so it is also skewed.
- **Why it matters:** This is train/serve skew across several of the 59 deployed features. The model is being trained on one set of semantics and scored live on another, which can silently distort every production prediction.
- **Suggested fix:** Make these features share one source of truth. Either port the production logic into training exactly, or extract a shared implementation used by both pipelines. Retrain after the definitions are aligned.

## Medium

### 2. Calibration generation still has no explicit range check against final-model prediction output

- **Files:** `train_model.py`
- **Lines:** `1099-1149`, `1252-1286`
- **Problem:** `build_calibration_tables()` bins only walk-forward predictions via `pd.qcut`, but there is no subsequent check that the final production model’s prediction range is still covered by those calibration bins. If the final model produces values outside the WF range, production will silently clamp them into the last calibration bin.
- **Why it matters:** This can create overconfident or flattened calibrated probabilities at the extremes without any visibility during training.
- **Suggested fix:** After final-model holdout inference, compare `min/max(y_pred)` against the calibration table bounds and log or fail if the model materially exceeds them. Optionally persist those final-model range stats alongside the calibration JSON.
