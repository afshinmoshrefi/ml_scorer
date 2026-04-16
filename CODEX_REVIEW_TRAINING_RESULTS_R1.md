# ML Pattern Scorer V2 -- Codex Training Review Results (Round 1)

## Critical

### 1. Training computes `pat_depth_x_vix` with a different formula than production inference

- **File:** `build_training_data.py`
- **Lines:** 971-974
- **File:** `ml_scorer/feature_engine.py`
- **Lines:** 1429-1434
- **Problem:** The training pipeline defines `pat_depth_x_vix` as `pat_deepest_pass * (1 / vix)`, while production inference computes `pat_deepest_pass * (20.0 / vix)`. This is not just a constant offset in the code path; it means the deployed model is scored on a differently scaled feature than the one it was trained on.
- **Why it matters:** This is a silent train/serve skew on one of the 59 model inputs. Every live prediction is using a feature distribution different from training, which can materially distort model output.
- **Suggested fix:** Make the formula identical in one shared implementation or shared constant. Retrain after fixing so the saved models and production feature engineering are aligned.

## High

### 2. NaN `mfe_return` rows are kept in the dataset and are not filtered before MFE training

- **File:** `build_training_data.py`
- **Lines:** 827-859
- **File:** `train_model.py`
- **Lines:** 369-379, 948-999, 1177-1180
- **Problem:** `process_symbol()` excludes rows with invalid `actual_return`, but it does not exclude rows where `mfe_return` is `NaN`. Those rows are written to parquet and later loaded unchanged when `ACTIVE_TARGET == 'mfe'`. The MFE training path then uses `y_train = df[LABEL_COL]` without dropping NaN targets.
- **Why it matters:** The MFE ensemble can fail during training, silently skip rows in library-specific ways, or produce corrupted calibration and evaluation metrics from partially invalid targets.
- **Suggested fix:** Filter out rows with `NaN` `mfe_return` when building the parquet, or at minimum drop rows with `NaN` in `LABEL_COL` inside `load_training_data()` when `ACTIVE_TARGET == 'mfe'`.

### 3. Pattern lookup is hardcoded to year 2026 and will silently break after the next opp-file year rollover

- **File:** `build_training_data.py`
- **Lines:** 763-766
- **File:** `build_training_data.py`
- **Lines:** 458-465
- **Problem:** `process_symbol()` constructs `date_2026 = f"2026-{month_day}"` and uses that as the lookup key into `combo_data`. If the opportunity files are regenerated with 2027 dates, every lookup will miss and `compute_pattern_features_fast()` will return `None`, causing symbols to produce no samples with no explicit fatal error.
- **Why it matters:** This is a silent time-bomb. The next annual opp-file refresh can collapse training coverage without a clear failure mode.
- **Suggested fix:** Derive the opp-file year dynamically from the loaded combo data, or normalize combo-data keys to month-day instead of a hardcoded calendar year. Add an explicit sanity check that a symbol is producing pattern matches.

## Medium

### 4. Year-crossing neighborhood shifts use the wrong historical year for December patterns

- **File:** `build_training_data.py`
- **Lines:** 570-577
- **Problem:** `compute_neighborhood_features()` computes `shifted_md = entry_date + shift_days` using the sample year, then rewrites only the year portion for each prior year. For late-December entries, `+7` or `+14` can land in January of the next calendar year, but the code maps that to January of the same historical year `yr`, not `yr + 1`.
- **Why it matters:** For year-crossing patterns, the “post-pattern” neighbors are economically wrong dates, so `pat_neighbor_avg_wr`, `pat_pre_slope`, and `pat_post_cliff` are silently distorted in edge cases.
- **Suggested fix:** Construct shifted dates relative to each historical year's base pattern date, not by reusing the sample year's shifted month/day. In other words, build `{yr}-{month_day}` first, then add `shift_days`.

### 5. There is still no automated check that training `FEATURE_COLS` matches production `ml_scorer/config.py`

- **File:** `train_model.py`
- **Lines:** 74-109
- **File:** `ml_scorer/config.py`
- **Lines:** 83-117
- **Problem:** The feature list is duplicated in training and production. The training pipeline validates that the three model files agree on feature count, but it does not validate that training `FEATURE_COLS` matches the production service’s `FEATURE_COLS` by exact name and order.
- **Why it matters:** A future one-sided edit will deploy models that appear valid but are scored with the wrong positional feature mapping in production.
- **Suggested fix:** Move the feature list to a shared source of truth or add an explicit import-time/assertion check in training that compares exact feature names and order against `ml_scorer/config.py`.
