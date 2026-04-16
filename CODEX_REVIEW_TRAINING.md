# ML Pattern Scorer V2 -- Codex Training Pipeline Review (Round 4)

## Purpose

This document provides Codex with complete context to perform a thorough code review of the ML Pattern Scorer V2 **training pipeline**. This is separate from the production service review (CODEX_REVIEW.md). The goal is to identify any bugs, correctness issues, data integrity problems, or leakage risks before the next model retrain.

**Return findings as a prioritized list: Critical > High > Medium > Low.**
For each finding include: file, line(s), description, and suggested fix.

---

## Files to Review

| File | Role |
|------|------|
| `build_training_data.py` | Builds training parquets: loads price CSVs, opp files, computes all 59 features per sample |
| `train_model.py` | Walk-forward validation + Optuna tuning + final model training + calibration |
| `config_ml.py` | Training configuration: paths, TICKER_SECTOR, ETF_SECTOR, YEAR_COMBOS, PE_COMBOS |

The production service files (`ml_scorer/`) are covered in a separate review (CODEX_REVIEW.md). Focus here exclusively on the training pipeline.

---

## Architecture Overview

### Training Flow

```
build_training_data.py
  1. compute_market_regime_series()     -- VIX, yields, credit, breadth, fed rate (all dates, vectorized)
  2. compute_spx_seasonal_lookups()     -- SPX seasonal win rates per (week, PE-phase) per training year
  3. load_sp500_symbols()               -- 475 symbols from sp500_symbols.csv
  4. joblib.Parallel(n_jobs=24):
       process_symbol(symbol, ...)
         load_price_csv()              -- OHLCV CSV for symbol
         load_opp_patterns()           -- Read all combo .csv.gz files for symbol
         compute_all_ta_series()       -- Vectorized TA features (5 kept)
         compute_all_context_series()  -- Vectorized context features (2 kept)
         compute_pattern_features_fast()  -- Pattern-intrinsic features (22)
         compute_neighborhood_features()  -- Neighborhood/temporal features (4)
         [build labels: actual_return, mfe_return, hit_target]
         [merge market regime, calendar, interaction features]
  5. pd.concat() -> write training parquet

train_model.py
  1. load_training_data()       -- Read parquet, apply VIX filter, downcast dtypes
  2. run_optuna_tuning()        -- 75 trials on 2M subsample (train 2000-2022, val 2023)
  3. walk_forward_train()       -- 8 expanding windows (train 2000-201X, val 201X+1)
  4. build_calibration_tables() -- 20-bin empirical calibration from WF predictions
  5. train_final_model()        -- Train on all data 2000-2024, validate on 2025
  6. Copy models + calibration -> ml_scorer/models/ and ml_scorer/calibration/
```

### Data Sources

```
C:/seasonals/data/
  csv/US/          -- Stock OHLCV CSVs (~3500 files, back to 1981+)
  csv/ETF/         -- ETF CSVs (SPY, QQQ, sector ETFs, VIX-related, etc.)
  csv/INDX/        -- Index CSVs (VIX, VIX3M, US10Y, US2Y, IRX, ADVN, DECN, SPX)
  sp500/opp_by_symbol/{SYMBOL}/   -- 475 dirs, 116 gzip CSVs each
    {combo_name}.csv.gz           -- e.g. Monthly_Opp_March_10_8.csv.gz
  ETF/opp_by_symbol/{SYMBOL}/    -- 157 ETF dirs
  sp500_symbols.csv               -- 475-ticker S&P 500 list
```

### Opp File Format

```
LorS,date,daysOut,sym,sharpe_ratio,avg_profit,median_profit,sharpe_ratio2,avg_profit2
l,2026-03-15,20,AAPL,2.34,3.1,2.9,2.41,3.2
```

Columns 0-8 in order. `sharpe_ratio2` and `avg_profit2` are optional (may be absent in older files).

### Training Data Schema (output parquet)

66 columns: 59 features + `date`, `actual_return`, `hit_target`, `mfe_return`, `daysOut`, `symbol`, `direction`.

Tier files:
- `features/training_data_10_30.parquet` -- 34.7M samples
- `features/training_data_31_60.parquet` -- 54.4M samples
- `features/training_data_61_90.parquet` -- 59.0M samples

---

## The 59 Features (FEATURE_COLS)

Defined in `train_model.py` (training) and `ml_scorer/config.py` (production). Both lists must be identical in **count AND order**. LightGBM and CatBoost use positional numpy arrays at inference time, not name-based lookup.

| Group | Count | Features |
|-------|-------|---------|
| Pattern-Intrinsic | 23 | pat_sharpe_ratio, pat_avg_profit2, pat_direction, pat_data_years, pat_deepest_pass, pat_depth_utilization, pat_passes_recent_10, pat_recent_vs_deep_sharpe, pat_num_combos_qualifying, pat_pe_match, pat_pe_deepest, pat_pe_utilization, pat_best_winrate, pat_worst_winrate, pat_deepest_pass_capped30, pat_consistency_std, pat_concurrent_count, pat_neighbor_avg_wr, pat_sharpness, pat_pre_slope, pat_post_cliff, pat_hit_last_year, pat_daysOut |
| Technical | 5 | ta_trend_long, ta_price_vs_sma200, ta_sma50_vs_sma200, ta_trend_direction_match, ta_rvol_20 |
| Market Regime | 16 | mkt_vix_level, mkt_vix_percentile_60d, mkt_vix_5d_change, mkt_vix_term_structure, mkt_yield_curve_10y2y, mkt_yield_curve_slope, mkt_credit_spread, mkt_credit_spread_change_20d, mkt_spy_roc_20, mkt_spy_above_sma200, mkt_adv_decl_ratio_10d, mkt_sector_rotation, mkt_vix_regime_bucket, mkt_breadth_momentum, mkt_fed_rate_level, mkt_fed_rate_direction |
| SPX Seasonal | 4 | mkt_spx_seasonal_wr, mkt_spx_seasonal_ret, mkt_spx_seasonal_regime, mkt_spx_dir_alignment |
| Context | 2 | ctx_pct_from_52w_high, ctx_pct_from_52w_low |
| Calendar | 5 | cal_month, cal_day_of_year, cal_week_of_month, cal_is_opex_week, cal_pe_year |
| Interactions | 4 | pat_dir_x_mkt_trend, pat_dir_x_sector_trend, pat_depth_x_vix, pat_quality_x_regime |

---

## Fixes Already Applied (Pre-Round-1 and Round 1)

### Pre-Round-1 Fix 1 -- `load_opp_patterns`: Silent exception swallowing
`build_training_data.py`: `except Exception: continue` now emits `log.warning(f'Skipping combo file {symbol}/{fname}: {e}')`.

### Pre-Round-1 Fix 2 -- `load_opp_patterns`: Positional CSV parsing with no schema validation
`build_training_data.py`: Added header validation before fast positional parsing. Checks that the first line starts with the expected column prefix. If not, the combo file is skipped with a warning. Column index constants (`_COL_LORS`, `_COL_DATE`, etc.) are named rather than raw integers.

### Round 1 Fix 1 (Critical) -- `pat_depth_x_vix` train/serve skew
**File**: `ml_scorer/feature_engine.py` (production service)
**Problem**: Training computed `pat_depth_x_vix = pat_deepest_pass * (1 / vix)`. Production computed `pat_deepest_pass * (20.0 / vix)` -- 20x the training value.
**Fix**: Changed production formula to `deepest * (1.0 / vix)` to match training exactly. Also removed the misleading comment. Models remain valid without retraining since the fix corrects the serving path to match what the model was trained on.

### Round 1 Fix 2 (High) -- NaN `mfe_return` rows included in training data
**File**: `build_training_data.py`, `valid` mask in `process_symbol()`
**Problem**: `valid = ~np.isnan(actual_returns) & ~np.isnan(entry_prices) & (entry_prices != 0)` did not filter rows where `mfe_return` is NaN. Those rows were written to parquet and loaded unchanged for MFE training, potentially corrupting MFE ensemble calibration.
**Fix**: Added `& ~np.isnan(mfe_returns)` to the `valid` mask. Both SR and MFE targets are now always non-null for every surviving row.

### Round 1 Fix 3 (High) -- Hardcoded opp file year 2026 as pattern lookup key
**File**: `build_training_data.py`, `process_symbol()`
**Problem**: `date_2026 = f"2026-{month_day}"` was used as the combo_data lookup key. When opp files are regenerated for 2027, all lookups silently return None and no training samples are produced.
**Fix**: After loading `combo_data`, the opp year is derived dynamically from the first key found in any combo lookup dict: `opp_year = _key[0][:4]`. The pattern loop now builds `opp_date = f"{opp_year}-{month_day}"`.

### Round 1 Fix 4 (Medium) -- Year-crossing neighborhood shifts use wrong historical year
**File**: `build_training_data.py`, `compute_neighborhood_features()`
**Problem**: `shifted_md = entry_date + pd.Timedelta(days=shift_days)` computed the shift from the sample year, then rewrote only the year to `yr`. For Dec-28 + 14 days = Jan-11 of next year, the code mapped that to Jan-11 of year `yr` instead of `yr+1`.
**Fix**: Build `hist_base = pd.Timestamp(f"{yr}-{month_day}")` first, then add `shift_days`. Year-crossing is now handled correctly because the shift is applied to the historical base date.

### Round 1 Fix 5 (Medium) -- No automated check that training FEATURE_COLS matches production
**File**: `train_model.py`, `main_single()`
**Problem**: `FEATURE_COLS` was duplicated in training and production with no cross-file validation. A one-sided edit would deploy models scored with the wrong positional feature mapping.
**Fix**: Added assertion at the start of `main_single()` that imports `ml_scorer/config.py` via `importlib.util` and compares `FEATURE_COLS` by exact name and order. Training aborts with a detailed error if there is any mismatch. Exits cleanly if the production config file is not present (e.g., standalone training environment).

### Round 2 Fix 1 (Critical) -- Neighborhood features computed from different data sources in training vs production
**Files**: `ml_scorer/feature_engine.py`, `build_training_data.py`
**Problem**: Five features were using semantically different data at training vs inference time:
- `pat_neighbor_avg_wr`, `pat_sharpness`, `pat_pre_slope`, `pat_post_cliff`: training computed these from realized win rates using actual prior-year price history; production computed them from combo win rates (y2/y1) from the opp files.
- `pat_concurrent_count`: training counted all patterns (including self) across all dates from all combos; production only scanned the first combo and excluded self.
- `pat_post_cliff`: training used mean of post1w and post2w; production used only post1w.
**Fix (production service)**: Added `_compute_neighborhood_features()` method to `FeatureEngine` that mirrors `compute_neighborhood_features()` from `build_training_data.py` exactly: for each shift (+-7, +-14 days), replays the shifted pattern across prior 10 years using actual price data and computes realized win rates. Uses the same year-crossing-safe date construction (`hist_base + shift_days`) from Round 1 Fix 4. Fixed `pat_concurrent_count` to scan all combos and count all unique `(daysOut, direction)` pairs at the requested date (including self). **Requires retrain** to realign the model with these corrected feature definitions at inference time.

### Round 3 Fix 1 (Medium) -- `pat_hit_last_year` training-day resolution mismatch
**File**: `ml_scorer/feature_engine.py`, `_compute_hit_last_year()`
**Problem**: Training uses forward lookup (range 0-3 for entry, 0-4 for exit) and returns NaN for Feb 29 in non-leap prior years. Production used `_get_price_on_date` (backward lookup up to 5 days) and mapped Feb 29 → Feb 28 instead of NaN.
**Fix**: Rewrote `_compute_hit_last_year` to build `pd.Timestamp(f"{date.year-1}-{mm}-{dd}")`, return NaN on ValueError, then use explicit forward-scanning loops matching training exactly.

### Round 3 Fix 2 (Low) -- `cal_day_of_year` always divides by 365, distorts leap-year dates
**Files**: `build_training_data.py:1050`, `ml_scorer/feature_engine.py`
**Problem**: Both pipelines used `dayofyear / 365.0`. Dec 31 of a leap year gives 366/365 = 1.0027, breaking the 0-1 normalization assumption.
**Fix**: Training now uses `dates.dt.dayofyear / (365 + dates.dt.is_leap_year.astype(int))`. Production uses `date.timetuple().tm_yday / (366 if date.is_leap_year else 365)`. Both sides consistent, normalization correct.

### Round 3 Fix 3 (Low) -- Walk-forward parquet missing cohort metadata
**File**: `train_model.py`
**Problem**: Saved WF predictions contained only `val_year`, `predicted`, `actual_return`, `hit_target`. Made it impossible to localize calibration drift by symbol, daysOut tier, or direction.
**Fix**: Added `symbol`, `daysOut`, `direction`, `date` to `label_cols` in `load_training_data()` so they're always present in `df`. When `--save-predictions` is active, these are appended to `pred_df` alongside existing columns.

### Round 2 Fix 2 (Medium) -- No calibration range check against final model predictions
**File**: `train_model.py`, `train_final_model()`
**Problem**: Calibration bins are built from walk-forward predictions (2018-2025). The final model trains on more data and may produce predictions outside that range, silently falling back to the last calibration bin in production with no visibility during training.
**Fix**: Added range check at the end of `train_final_model()` that compares `min/max(y_pred)` on the 2025 holdout against the calibration bin bounds. Logs the percentage of holdout predictions outside the calibration range and emits a warning if either tail exceeds 5%.

---

## Known Design Decisions (Do NOT Flag These)

1. **Positional numpy arrays for model prediction**. LightGBM and CatBoost `predict()` use positional numpy arrays. Feature order in `FEATURE_COLS` must match training order. This is a known constraint, not a bug.
2. **SPX seasonal lookups use `year - 1` cutoff**. For a sample in year Y, the lookup uses SPX data from 1960 through Y-1. Intentional -- prevents leakage of current-year SPX data into the feature.
3. **Calibration built from walk-forward predictions, not final model**. Standard practice. The final model trains on more data, so its prediction scale may differ slightly from WF models. Accepted trade-off.
4. **MFE window extends 4 calendar days beyond exit date**. `close_values.loc[... <= ex + pd.Timedelta(days=4)]`. Intentional -- allows for execution delays on exit day. All MFE values are computed with this same convention so calibration is internally consistent.
5. **VIX > 35 filter applied as sample exclusion, not as a feature**. Samples during panic regimes are removed from training entirely, not included with VIX=35 as a special value. Production service refuses to score when VIX > 35.
6. **Sharpe ratio uses `np.sqrt(252)` annualization on per-trade returns**. Standard approximation for strategy performance metrics. Not a daily-returns Sharpe.
7. **`TIERS` in `train_model.py` includes `91_120`, `121_200`, `201_300`**. These are experimental tiers not deployed to production. They run correctly if passed via `--tier` but write model files that won't be used.
8. **`compute_pattern_features_fast` docstring says `pat_daysOut` was dropped**. Outdated docstring. `pat_daysOut` is included -- computed at DataFrame level (`sym_df['pat_daysOut'] = sym_df['daysOut']`) rather than inside this function.
9. **`pat_worst_winrate` returns `best_winrate` when no combo qualifies at `< 1.0`**. If only PE combos qualify (all 100% win rate), `worst_winrate` defaults to `best_winrate`. Deliberate fallback.
10. **Walk-forward uses expanding window starting from `TRAIN_START = 2000`**. All 8 windows include the full 2000-onward data. Pre-2018 data is weaker signal for 31-60/61-90 tiers but is not excluded -- the model learns to discount it naturally.
11. **Optuna tuning uses fixed 2022/2023 split regardless of tier**. The tuned params are applied to all subsequent training windows. Acceptable approximation.
12. **`compute_neighborhood_features` uses at most 10 prior years**. `prior_years = list(range(max(sample_year - 10, 2000), sample_year))`. Intentional limit for computational speed.
13. **`N_JOBS = 24`** in `config_ml.py`. Matches the 24-core dev machine. Override with `ML_SCORER_NJOBS` env var.
14. **Both SR and MFE targets are required to be non-NaN in every training row**. The `valid` mask in `process_symbol()` filters `~np.isnan(mfe_returns)` so the same parquet file is used cleanly for both SR and MFE training without per-target row differences.
15. **Opp file year is derived dynamically from combo_data keys**. `opp_year = _key[0][:4]` reads the year from the first combo_data key found. This is always the opp generation year (currently 2026) and will update automatically when opp files are regenerated.
16. **Neighborhood shift year-crossing is correct after Round 1 Fix 4**. `hist_base = pd.Timestamp(f"{yr}-{month_day}"); shifted_entry = hist_base + pd.Timedelta(days=shift_days)`. For Dec-28 + 14 days, `shifted_entry` is Jan-11 of `yr+1`, which is the economically correct neighbor date.
17. **FEATURE_COLS cross-file assertion fires at the start of `main_single()`**. Training aborts immediately if training and production feature lists diverge. The assertion is import-time for the production config module, so it does not depend on any runtime state.
18. **Neighborhood features use realized price history, not combo win rates**. After Round 2 Fix 1, both training and production compute `pat_neighbor_avg_wr`, `pat_sharpness`, `pat_pre_slope`, `pat_post_cliff` from actual prior-year price returns (prior 10 years, same shift logic). The production `_compute_neighborhood_features()` method mirrors `compute_neighborhood_features()` in `build_training_data.py` exactly.
19. **`pat_concurrent_count` counts all unique (daysOut, direction) at the date including self**. After Round 2 Fix 1, production scans all combos (not just the first) and counts unique `(daysOut, direction)` pairs at the requested date including the current pattern, matching training semantics.
21. **`pat_hit_last_year` uses forward trading-day lookup in both training and production**. After Round 3 Fix 1, both sides build `{year-1}-MM-DD`, return NaN on ValueError (Feb 29), then scan forward 0-3 days for entry and 0-4 days for exit. `_get_price_on_date` (backward lookup) is no longer used for this feature.
22. **`cal_day_of_year` is normalized by actual days in year**. After Round 3 Fix 2, the value is always in (0, 1] regardless of leap year. Same formula in both training and production.
23. **Walk-forward prediction parquet includes cohort metadata**. After Round 3 Fix 3, `--save-predictions` output includes `date`, `symbol`, `daysOut`, `direction` alongside predictions, enabling per-cohort calibration diagnostics.
20. **Calibration range check logs a warning at 5% tail overflow**. After Round 2 Fix 2, `train_final_model()` compares holdout prediction range against calibration bounds and warns if >5% fall outside either tail. This is advisory only -- no hard failure -- since some overflow is expected when the final model trains on more data than WF folds.

---

## Areas to Investigate

### 1. `process_symbol` -- label computation and look-ahead bias from `pat_daysOut` feature

`pat_daysOut` is the actual holding period in days for the specific opportunity. At training time, this value is set from the opp file (how many days the pattern specifies). The pattern was discovered using data up to the most recent available year (the opp file generation year). However, `pat_daysOut` itself is just an integer (e.g. 20) that doesn't encode future information -- it's a property of the pattern definition. Verify that no feature computation uses the actual exit price or forward return value to derive any input feature.

### 2. `compute_market_regime_series` -- VIX percentile uses full-history rolling rank

```python
mkt['mkt_vix_percentile_60d'] = vix_close.rolling(60).rank(pct=True)
```

`pandas rolling().rank()` computes rank within the rolling window, which is correct and leak-free (only uses the past 60 days). However, the entire regime DataFrame is built once using all available data and then reindexed per sample date. Verify that no market regime feature inadvertently uses future data (e.g., any `.fillna(method='ffill')` applied after the rolling computation that would pull future values backward).

### 3. `walk_forward_train` -- predictions saved per window but `val_year` is a scalar, not per-row

```python
pred_df = pd.DataFrame({
    'val_year': val_year,
    'predicted': y_pred,
    'actual_return': actual_return,
    'hit_target': y_val_binary,
})
```

`val_year` is a scalar assigned to all rows. If the validation data for a given window spans multiple calendar years (which should not happen given the window design, but verify), the `val_year` column would be wrong. Also, the predictions parquet saved to `wf_predictions_sr.parquet` does not include `symbol`, `date`, or `daysOut`. This means the calibration tables cannot be filtered by tier or direction, and any per-cohort analysis of calibration quality is impossible. Evaluate whether adding those columns would be useful for future calibration quality checks.

### 3. `_compute_neighborhood_features` in production -- performance impact on scoring latency

The new `_compute_neighborhood_features()` method in `ml_scorer/feature_engine.py` replays prior-year price history for each opportunity scored (up to 10 years x 4 shifts = 40 lookups per feature call, plus 10 for the pattern itself). For batch scoring (e.g. the `/select` route iterating over many candidates), this adds per-candidate computation that the old combo-lookup approach did not have. Verify whether the scoring latency increase is acceptable for the typical batch size. If it becomes a bottleneck, the prior-year win rates could be precomputed and cached per (symbol, month_day, daysOut, direction) at warmup time.

### 4. `train_model.py` global mutable state modified by CLI flags

```python
ACTIVE_TIER = '10_30'
ACTIVE_TARGET = 'sr'
DATA_PATH_OVERRIDE = None
FILTER_AGAINST_SEASON = False
```

These are module-level globals mutated by `main()` based on CLI args. `get_training_data_path()` and `TRAINING_DATA_PATH` are evaluated at import time (before CLI parsing), so `TRAINING_DATA_PATH` always reflects the default tier. The actual path used is resolved via `get_training_data_path(ACTIVE_TIER)` inside `load_training_data()`, which correctly uses the post-mutation value. Verify that no other code path uses the module-level `TRAINING_DATA_PATH` directly instead of calling `get_training_data_path()`.

### 5. Any new issues not covered above

Perform a fresh read of all three files. Focus on:
- Data leakage: does any feature at training time use information that would not be available at inference time on the trade date?
- Parallelism hazards: are any mutable objects shared across joblib workers?
- Label correctness: is `hit_target = (actual_return > 0)` consistent with the definition used in calibration and production?
- Edge cases in `compute_spx_seasonal_lookups` when SPX data is sparse (e.g. early years with < 5 samples per week/PE bucket)
- Whether `mkt_yield_curve_slope = yield_curve.diff(5)` is computed on calendar days or trading days
- Feb 29 handling in `compute_neighborhood_features` and `process_symbol`

---

## Data Flow (Training)

```
For each symbol (parallel):
  price_df = load_price_csv(symbol)            # full history back to 1981+
  patterns, combo_data = load_opp_patterns()   # all qualifying patterns from opp files
  ta_series = compute_all_ta_series(price_df)  # vectorized TA across full history
  ctx_series = compute_all_context_series()    # vectorized context across full history

  # Derive opp year from actual combo_data keys (not hardcoded)
  opp_year = first key's year from combo_data

  For each (month_day, daysOut, direction) in patterns:
    opp_date = f"{opp_year}-{month_day}"
    pat_feats = compute_pattern_features_fast(combo_data, opp_date, daysOut, direction)

    For each year in TRAIN_YEARS (2000-2025):
      entry_date = nearest trading day to {year}-MM-DD
      exit_date  = nearest trading day to entry_date + daysOut calendar days

      actual_return = (close[exit] - close[entry]) / close[entry] * 100 * direction_sign
      mfe_return    = max/min close in (entry, exit+4d] vs entry price * direction_sign
      hit_target    = 1 if actual_return > 0 else 0

      # Only rows where both actual_return AND mfe_return are non-NaN are kept
      valid = ~isnan(actual_return) & ~isnan(entry_price) & ~isnan(mfe_return)

      # Neighborhood shifts: build hist_base = {yr}-{month_day} first, then add shift_days
      nbr_feats = compute_neighborhood_features(entry_date, daysOut, prior years only)
      pat_hit_last_year = 1 if same pattern worked in year-1 else 0

      # Merge TA, context, market regime, SPX seasonal, calendar, interactions by date
      row = {pat_feats..., ta_feats..., mkt_feats..., spx_feats..., cal_feats..., interaction_feats...}
      append row to symbol samples

  return DataFrame of all samples for this symbol

concat all symbol DataFrames -> write to parquet
```

---

## Environment Context

- **Dev machine**: Windows 11, Python 3.12, 24-core / 64GB RAM
- **Training paths**: all hardcoded to `C:/seasonals/data` in `config_ml.py` (not env-var driven like the production service)
- **Output**: `C:/seasonals/ml_scorer/features/training_data_{tier}.parquet`
- **Models**: `C:/seasonals/ml_scorer/models/v2_*.{txt,json,cbm}`
- **Results**: `C:/seasonals/ml_scorer/results/`
- **Dependencies**: lightgbm, xgboost, catboost, optuna, joblib, pandas, numpy, scikit-learn, pyarrow, scipy

---

## What a Good Review Should Cover

1. **Data leakage**: any feature that uses future information at training time but not at inference
2. **Label correctness**: actual_return, mfe_return, hit_target computation correctness
3. **Silent data loss**: opp file parsing failures, symbol skipping, NaN propagation into labels
4. **Feature sync**: training `FEATURE_COLS` vs production `ml_scorer/config.py` FEATURE_COLS (now asserted at runtime)
5. **Calibration validity**: are WF prediction ranges representative of final model predictions
6. **Calendar edge cases**: Feb 29 handling in pattern replay and feature computation
7. **Parallelism**: shared mutable state in joblib workers
8. **Hardcoded paths**: `config_ml.py` paths are Windows-only, not configurable via env var
