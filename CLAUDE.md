# ML Pattern Scorer — Project Context for Claude Code

## CRITICAL: Read this entire file before doing any work. This is the complete context for continuing development.

## What This Is
ML model (LightGBM) that scores TradeWave seasonal stock pattern opportunities 0-100 for "probability of playing out." Gatekeeper for automated options trading. Personal use, $10K starting capital, options on 10-30 day patterns.

## Current Status: V1 Phase 1 COMPLETE. V2 rebuild is the NEXT STEP.

---

## Project Files

| File | Lines | Purpose |
|------|-------|---------|
| `config_ml.py` | 237 | Paths, sector mappings, TICKER_SECTOR dict (475 stocks), year combos, PE cycle calc, thresholds |
| `feature_engine.py` | 1169 | `FeatureEngine` class - computes all 73 V1 features for a single opportunity. CLI test mode. |
| `build_training_data.py` | 810 | Builds training parquet from price CSVs + opp files. Vectorized TA, batch DataFrame assembly. **Single-threaded in V1.** |
| `train_model.py` | 479 | LightGBM walk-forward validation + final model training. Memory-optimized (float32, column pruning, gc). Supports `--final-only` flag. |

### Saved Artifacts (V1 - will be overwritten by V2)
- `models/pattern_scorer_20260310.txt` — V1 production model (107KB)
- `results/feature_importance.csv` — 69 features ranked by gain
- `features/training_data.parquet` — 761 MB, 15.6M samples x 79 columns

---

## Data Dependencies (MUST exist on the machine)

```
/home/flask/data/csv/US/          # Stock price CSVs (OHLCV), ~3500 files, back to 1981+
/home/flask/data/csv/ETF/         # ETF CSVs (SPY, QQQ, XLK, XLF, HYG, LQD, TLT, etc.)
/home/flask/data/csv/INDX/        # Index CSVs (VIX, TNX/US10Y, IRX, DJI, ADVN, DECL, etc.)
/home/flask/data/sp500/opp_by_symbol/  # 475 dirs, 116 gzip CSVs each (pattern opportunities)
/home/flask/data/sp500_symbols.csv     # S&P 500 ticker list (504 lines incl header)
/home/flask/edgar/earnings/            # Earnings date JSONs (limited coverage, features were NaN)
```

### Price CSV format
```
,date,open,high,low,close,volume,adj_factor
0,1981-01-02,0.118,0.1188,0.118,0.118,21660800,0.0034
```

### Opportunity CSV format (gzip, per symbol, 116 combo files like `10_10.csv.gz`, `5_5_PE2.csv.gz`)
Fields: `LorS,date,daysOut,sym,sharpe_ratio,avg_profit,median_profit,sharpe_ratio2,avg_profit2` + win-rate percentile columns

### Config paths are set in `config_ml.py` — update if data moves:
```python
DATA_DIR = '/home/flask/data'
US_CSV_DIR = '/home/flask/data/csv/US'
ETF_CSV_DIR = '/home/flask/data/csv/ETF'
INDX_CSV_DIR = '/home/flask/data/csv/INDX'
OPP_BY_SYMBOL_DIR = '/home/flask/data/sp500/opp_by_symbol'
EARNINGS_DIR = '/home/flask/edgar/earnings'
SP500_SYMBOLS = '/home/flask/data/sp500_symbols.csv'
FEATURE_CACHE_DIR = '/home/flask/ml_scorer/features'
MODEL_DIR = '/home/flask/ml_scorer/models'
RESULTS_DIR = '/home/flask/ml_scorer/results'
```

---

## V1 Training Results (for reference / comparison with V2)

### Walk-Forward Validation Results

| Val Year | AUC   | Best Iter | Base WR | ML_70 WR | ML_80 WR | ML_85 WR | Sharpe@best |
|----------|-------|-----------|---------|----------|----------|----------|-------------|
| 2020     | 0.718 | 33        | 62.9%   | 83.1%    | 89.6%    | 91.7%    | 12.79@ML85  |
| 2021     | 0.590 | 9         | 68.9%   | 74.6%    | -        | -        | 6.32@ML70   |
| 2022     | 0.512 | 14        | 81.7%   | 84.2%    | -        | -        | 10.64@ML70  |
| 2023     | 0.464 | 1         | 65.0%   | -        | -        | -        | n/a         |
| 2024     | 0.660 | 24        | 66.1%   | 79.6%    | 88.5%    | 97.8%    | 17.55@ML85  |

### Final Model (Train 2015-2024, Val 2025)
- AUC=0.632, best iteration: 7
- Baseline: 61.7% WR, 1.12% avg return, Sharpe 2.42
- ML_70: 83.9% WR, 3.63% avg return, Sharpe 9.08

### V1 Feature Importance (Top 20)
1. pat_deepest_pass (3.06M) 2. mkt_yield_curve_10y2y (1.92M) 3. cal_pe_year (1.73M)
4. pat_deepest_pass_capped30 (1.30M) 5. mkt_credit_spread (1.28M) 6. pat_best_winrate (892K)
7. cal_day_of_year (801K) 8. mkt_vix_level (643K) 9. pat_direction (470K)
10. mkt_vix_term_structure (394K) 11. mkt_spy_above_sma200 (297K) 12. pat_num_combos_qualifying (211K)
13. pat_depth_utilization (185K) 14. cal_month (179K) 15. mkt_credit_spread_change_20d (155K)
16. mkt_spy_roc_20 (144K) 17. pat_pe_match (96K) 18. mkt_vix_percentile_60d (94K)
19. mkt_adv_decl_ratio_10d (92K) 20. mkt_sector_rotation (85K)

### 30 Zero-Importance Features (model never split on these)
**All short-term TA:** ta_rsi_14, ta_macd_histogram, ta_macd_hist_slope, ta_roc_5, ta_roc_20, ta_bollinger_position, ta_obv_slope_10, ta_atr_percentile, ta_volume_price_confirm, ta_price_vs_sma20, ta_price_vs_sma50, ta_sma20_vs_sma50, ta_trend_short, ta_trend_delta, ta_rs_vs_spy_20d
**Pattern:** pat_avg_profit, pat_median_profit, pat_mfe_ratio, pat_profit_per_day, pat_daysOut, pat_daysOut_bucket, pat_passes_at_max_depth
**Context:** ctx_position_in_52w_range, ctx_avg_volume_20d, ctx_market_cap_bucket, ctx_sector_rs_20d, ctx_stock_vs_sector_20d
**Calendar:** cal_quarter, cal_month_position

### Key V1 Findings
1. **Pattern depth + macro regime >> technical indicators.** Model answers: "robust deep pattern in favorable macro?"
2. **Extremely shallow models** (7-33 trees). Signal exists but is weak.
3. **Inconsistent across years.** Great on 2020/2024, poor on 2021-2023.
4. **ML_70 is practical threshold.** ML_80/85 often too few signals.
5. **PE cycle is #3 feature** but only had 1 cycle in training data.

---

## V2 Rebuild Plan (THIS IS WHAT TO BUILD NEXT)

### Overview of Changes
1. Extend training years from 2015-2025 to **2012-2025** (14 years, 2 full PE cycles)
2. Add **~21 new features** (neighborhood, pattern history, interactions, better regime)
3. Drop **30 zero-importance features** from V1
4. Add **joblib parallelization** for build_training_data.py
5. Estimated **~63 features** total (vs 73 in V1)

### Change 1: Extend to 2012-2025
- Update `TRAIN_YEARS` in build_training_data.py: `list(range(2012, 2026))`
- New walk-forward windows in train_model.py:
  ```
  Train 2012-2017 → Val 2018
  Train 2012-2018 → Val 2019
  Train 2012-2019 → Val 2020
  Train 2012-2020 → Val 2021
  Train 2012-2021 → Val 2022
  Train 2012-2022 → Val 2023
  Train 2012-2023 → Val 2024
  Train 2012-2024 → Val 2025 (holdout)
  ```
- Price data confirmed available back to 1981+ for all S&P stocks
- Macro data (VIX from 2005+, US2Y from 2013, HYG from 2007)

### Change 2: New Features — Pattern Temporal Neighborhood (~8 features)
User's key insight: check if same pattern works when shifted ±1-4 weeks. A pattern that "falls off a cliff" after end date is fragile. A pattern that works across shifted windows is robust.

For each pattern (sym, entry_date, days_out, direction), shift entry date by ±7/14/21/28 days and compute historical return in each shifted window:
- `pat_neighbor_pre1w_wr` — win rate if pattern shifted 1 week earlier
- `pat_neighbor_pre2w_wr` — 2 weeks earlier
- `pat_neighbor_post1w_wr` — 1 week later
- `pat_neighbor_post2w_wr` — 2 weeks later
- `pat_neighbor_avg_wr` — average win rate of all 4 shifted versions
- `pat_sharpness` — pattern WR / neighbor avg WR (high = narrow fragile spike, ~1.0 = broad robust effect)
- `pat_pre_slope` — are returns building up approaching the pattern start?
- `pat_post_cliff` — pattern WR minus post-pattern WR (large positive = cliff after end)

Implementation: for each training sample, we already have the price DataFrame. Shift the entry date, look up close prices at shifted_entry and shifted_entry + days_out, compute return, check if it hit target. Average across available historical years for that shifted window.

### Change 3: New Features — Pattern History (~5 features)
- `pat_recent_3yr_wr` — win rate in most recent 3 years vs all-time (pattern freshness)
- `pat_hit_last_year` — binary: did this exact pattern work the prior year?
- `pat_consistency_std` — std deviation of win rates across different depth levels (low = robust)
- `pat_alpha_decay` — regression slope of annual win rates over time (declining = pattern losing edge)
- `pat_concurrent_count` — how many other patterns fire for same stock on same day? (consensus signal)

### Change 4: New Features — Interaction Features (~4 features)
The V1 model was too shallow (7-33 trees) to learn interactions. Pre-compute them:
- `pat_dir_x_mkt_trend` — pattern direction aligned with SPY trend? (1=aligned, -1=against)
- `pat_dir_x_sector_trend` — pattern direction aligned with sector ETF trend?
- `pat_depth_x_vix` — pat_deepest_pass * (1/mkt_vix_level_normalized). Deep patterns in low-vol.
- `pat_quality_x_regime` — pat_sharpe_ratio * mkt_spy_above_sma200. Quality in bull market.

### Change 5: New Features — Better Regime (~4 features)
- `mkt_fed_rate_level` — fed funds rate proxy from IRX (3-month T-bill yield) in INDX/
- `mkt_fed_rate_direction` — 60-day change in fed rate (hiking/cutting/holding)
- `mkt_vix_regime_bucket` — categorical: 0=<15, 1=15-20, 2=20-30, 3=>30
- `mkt_breadth_momentum` — 20-day change in adv/decl ratio (breadth improving or deteriorating)

### Change 6: Parallelize with joblib
```python
from joblib import Parallel, delayed

# Precompute market regime (single-threaded, ~30 sec)
market_data = compute_market_regime_series()

# Parallel per-symbol processing
results = Parallel(n_jobs=N_JOBS)(
    delayed(process_symbol)(symbol, market_data) for symbol in symbols
)

# Concat results
df = pd.concat(results, ignore_index=True)
```
- N_JOBS configurable: 2-3 on current 3-core VM, 24 on 32-core server
- Each worker loads its own CSV, computes features, returns DataFrame
- Market regime passed to each worker (read-only, ~small)

### Change 7: Optuna Hyperparameter Tuning
Before main training, run automated hyperparameter search:
```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255]),
    }
    # Train on subsample (~2M rows) for speed, evaluate on validation split
    # Return validation metric (AUC or log_loss)
```
- ~50-100 trials on 2M sample subset
- ~20-30 min on 24 cores
- Use best params for all subsequent training
- Install: `pip install optuna`

### Change 8: Regression Target (instead of binary classification)
Instead of `hit_target` (0/1), predict `actual_return` (continuous %):
- `objective='regression'`, `metric='rmse'`
- Model learns magnitude, not just direction
- Convert to score: rank predictions into 0-100 percentiles, or use sigmoid scaling
- Same training time as binary, but richer signal for the model to learn from
- Keep binary evaluation metrics for comparison (threshold the predicted return at 0%)

### Change 9: Ensemble (LightGBM + XGBoost + CatBoost)
Train 3 gradient boosting models, average predictions:
```python
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Each gets same Optuna-tuned hyperparameters (translated to their API)
# Final score = (lgb_pred + xgb_pred + catboost_pred) / 3
```
- Each model finds slightly different feature interactions
- Averaging improves calibration and year-to-year stability
- 3x training time but same scoring time (3 predictions averaged)
- Install: `pip install xgboost catboost`
- For production scoring: load all 3 models, average predictions

### Change 10: Drop zero-importance features
Remove the 30 features listed above. Keep:
- **Pattern (keep):** pat_sharpe_ratio, pat_deepest_pass, pat_deepest_pass_capped30, pat_depth_utilization, pat_best_winrate, pat_worst_winrate, pat_direction, pat_data_years, pat_passes_recent_10, pat_recent_vs_deep_sharpe, pat_num_combos_qualifying, pat_pe_match, pat_pe_deepest, pat_pe_utilization, pat_avg_profit2
- **Technical (keep only 5):** ta_trend_long, ta_price_vs_sma200, ta_sma50_vs_sma200, ta_trend_direction_match, ta_rvol_20
- **Market (keep all 11 that had importance):** all except the 3 always-NaN ones
- **Context (keep 2):** ctx_pct_from_52w_high, ctx_pct_from_52w_low
- **Calendar (keep 5):** cal_pe_year, cal_day_of_year, cal_month, cal_is_opex_week, cal_week_of_month

---

## V2 Build Steps (Execution Order)
1. `pip install joblib optuna xgboost catboost` (if not already installed)
2. Update `build_training_data.py`: extend years to 2012-2025, add new features (neighborhood, history, interactions, regime), add joblib parallelization, drop zero-importance features
3. Update `config_ml.py`: TRAIN_YEARS range, N_JOBS setting, new feature lists
4. Rebuild training data: `nohup python3 build_training_data.py > build_log.txt 2>&1 &`
5. Update `train_model.py`:
   - New walk-forward windows (8 windows starting from val 2018)
   - Update FEATURE_COLS for V2 features
   - Add Optuna tuning phase before walk-forward
   - Switch to regression target (predict actual_return)
   - Add XGBoost + CatBoost ensemble training
   - Average 3 model predictions for final score
6. Retrain: `nohup python3 train_model.py > train_log.txt 2>&1 &`
7. Analyze V2 vs V1 results, compare regression vs binary, ensemble vs single model

### V2 Time Estimates (on 24-core server)
- Build training data: ~10-15 min
- Optuna tuning: ~20-30 min (50-100 trials on 2M subset)
- Walk-forward (3 models x 8 windows): ~90 min
- Final model training (3 models): ~15 min
- **Total: ~2.5-3 hours**

### V2 Time Estimates (on 3-core VM)
- Not recommended. ~20+ hours total.

---

## Technical Notes

### OOM Issues (IMPORTANT if running on limited RAM)
- 22GB RAM machine: training data as float64 = ~9GB, as float32 = ~5.6GB
- LightGBM max_bin=255 + 14M samples = OOM (needed 45GB VM)
- **Fix:** max_bin=63, parquet column pruning on load, float32 downcast, gc.collect between windows
- With V2 (2012-2025 = more samples), may need max_bin=63 even on larger machines
- If machine has 64GB+ RAM, can try max_bin=127 or 255 again

### Python Environment
- Python 3.8, pandas 1.2.4 (old — `rolling().rank()` not available, use numpy loops)
- lightgbm compiled from source (no wheel for this Python/OS combo)
- scikit-learn, pyarrow installed
- joblib needs to be installed: `pip install joblib`

### Performance Notes
- V1 build: ~3 hours single-threaded on 3-core VM for 475 symbols x 11 years
- V1 training: ~20 min per walk-forward window on 3 cores (LightGBM uses all cores for histogram building)
- Neighborhood features will add compute time per symbol (8 shifted price lookups per pattern per year)

### How Training Data is Built (conceptual)
1. Precompute market regime features for ALL dates (VIX, yields, credit spreads, SPY trend, breadth, sector rotation). One DataFrame, ~30 seconds.
2. For each symbol:
   a. Load price CSV, compute vectorized TA series (SMA, ATR, RSI, MACD, OBV, Bollinger, trend scores)
   b. Compute context series (52W high/low, volume, market cap)
   c. Load opportunity files (116 gzip CSVs), parse with fast string split, build depth profile
   d. Get unique patterns: set of (month, day, daysOut, direction) from opportunity data
   e. For each training year (2012-2025): replay each pattern, look up entry date, compute:
      - Pattern features from depth profile (V1 features + new V2 neighborhood/history features)
      - Technical features from precomputed TA series at entry date
      - Market regime features from precomputed regime series at entry date
      - Context features from precomputed context series at entry date
      - Calendar features from entry date
      - Label: actual_return from close[entry] to close[entry+daysOut], hit_target = 1 if profitable
   f. Return DataFrame of all samples for this symbol
3. Concat all symbol DataFrames, save as parquet

### How Patterns Work (domain knowledge)
- A "pattern" = seasonal tendency for a stock to move in a direction over specific calendar dates
- Defined by: (symbol, start_month_day, daysOut, direction_long_or_short)
- "Combo" = lookback window: e.g., `10_8` means look back 10 years, require 8 profitable years (80% win rate)
- Deeper combos (more years) = more robust pattern. `30_24` > `10_8`
- PE combos: only count presidential election cycle years (every 4th year)
- opp_by_symbol files contain patterns that CURRENTLY qualify (as of 2026 data)
- Training replays these patterns historically: "if this pattern existed in 2018, would it have worked?"
- Some look-ahead bias accepted (pattern discovered from 2026 data applied to 2018). Walk-forward validation catches issues.

---

## Production Usage (After V2 Training Completes)

### Phase 2: Daily Scoring Cron (`score_opportunities.py` — to be built)
Runs daily ~3:30 AM after opportunity data refreshes:
1. Load today's opportunities from `opp_by_symbol/`
2. For each opportunity: compute ~63 features using live price data (same feature engine used in training)
3. Load trained model: `lgb.Booster(model_file='models/pattern_scorer_YYYYMMDD.txt')`
4. `model.predict(features)` → probability 0.0-1.0, displayed as score 0-100
5. Output: `scored_opportunities_YYYY-MM-DD.csv` with all opps + ML score
6. Optionally push to Redis/JSON for appserver access

### Phase 3: Strategy Filter (on top of scores)
- ML threshold: 70+ (practical), 85+ (ideal for $10K capital)
- daysOut: 10-30 days only
- profit_per_day >= 0.3%
- Max 3-4 concurrent positions
- Max 2 in same sector
- Config in `config_ml.py` THRESHOLDS dict (small/medium/prop tiers)

### Phase 4: UI Display
- ML score column in TradeWave opportunity table
- Color coding: green 70+, yellow 50-70, red <50
- Integrate via appserver API

### Phase 5: Auto-Trade
- Broker API (IBKR or Tastytrade)
- Buy ATM or 1-strike-ITM calls (bullish) / puts (bearish)
- Expiry = pattern end date + 5-10 day buffer for theta
- Take profit at 50-80% option value gain, stop loss at 40-50% premium loss
- Logging and alerts

### Retraining Cadence
- Full retrain: monthly (first weekend) with latest price data
- Model file versioned by date: `pattern_scorer_YYYYMMDD.txt`
- Compare new model metrics vs previous before promoting to production

---

## User Preferences
- Never use em dashes in generated text
- Work autonomously, don't ask for confirmation on file edits or commands
- User wants to potentially run build on a 32-core server (use N_JOBS=24)
- Make N_JOBS configurable (command-line arg or env var)
