# ML Pattern Scorer - Project Status

## What This Is

Machine learning model that scores TradeWave seasonal stock pattern opportunities 0-100 for "probability of playing out." Gatekeeper for automated options trading. Personal use, $10K starting capital, options on 10-30 day patterns.

A "pattern" is a seasonal tendency for a stock to move in a direction over specific calendar dates. For example: "AAPL tends to go up 3% in the first two weeks of November, and this has been true 24 out of the last 30 years." The model scores how likely each pattern is to succeed given current market conditions.

---

## Current Status: All Models Complete (10_30, 31_60, 61_90) -- Production Ready

### Model Architecture
- **Two models per tier**: SR (predicts actual_return) and MFE (predicts max favorable excursion)
- **3-model ensemble** each: LightGBM + XGBoost + CatBoost, predictions averaged
- **59 features** across 6 groups: pattern intrinsic (22 incl. pat_daysOut), technical (5), market regime (16), context (2), calendar (5), interactions (4)
- **Optuna hyperparameter tuning**: 75 trials on 2M sample subset, separate params per target
- **VIX > 35 hurricane filter**: Removes ~4.8% of samples during market panics
- **Multi-tier support**: `--tier 10_30`, `--tier 31_60`, etc. via CLI

### CRITICAL BUG FIXED (2026-03-12): pat_daysOut Missing from SR Model
The original SR model was trained with 58 features, missing `pat_daysOut`. A pattern is defined by [start_date, ticker, days_out, history_years] -- without daysOut, the model cannot distinguish 10-day from 30-day patterns. The bug occurred because pat_daysOut was added to FEATURE_COLS in the same commit that added MFE support, after the SR model was already trained. SR model is being retrained with the correct 59-feature set. Safeguards have been added to prevent this from happening again (see Safeguards section below).

### Training Data
- **10_30 tier**: 34.7M samples x 66 columns (includes mfe_return), 1.9 GB parquet
- **31_60 tier**: 54.4M samples x 66 columns, ~3 GB parquet
- **61_90 tier**: 59.0M samples x 66 columns, 3.3 GB parquet
- **Training years**: 2000-2025 (26 years)
- **475 S&P 500 stocks**, 116 pattern combinations each
- Built with joblib parallelization (N_JOBS=24, ~54 min build time per tier)

### SR Model Walk-Forward Results (10-30 tier)

Config: Standard walk-forward + VIX>35 filter + SPX seasonal features as model inputs

RETRAINED 2026-03-12 with 59 features (pat_daysOut included). Previous 58-feature results were invalid.

| Val Year | AUC   | ML_70 WR | ML_70 Sharpe | ML_90 WR | ML_90 Sharpe |
|----------|-------|----------|--------------|----------|--------------|
| 2018     | 0.614 | 77.8%    | 7.04         | 80.5%    | 8.58         |
| 2019     | 0.637 | 77.8%    | 6.93         | 82.7%    | 8.09         |
| 2020     | 0.651 | 78.2%    | 8.32         | 82.2%    | 11.04        |
| 2021     | 0.611 | 78.8%    | 8.68         | 83.4%    | 10.72        |
| 2022     | 0.564 | 86.0%    | 12.31        | 90.4%    | 15.47        |
| 2023     | 0.644 | 79.7%    | 9.00         | 86.5%    | 12.35        |
| 2024     | 0.623 | 77.8%    | 8.21         | 78.8%    | 9.08         |
| 2025     | 0.671 | 78.1%    | 7.45         | 83.2%    | 10.04        |
| **Avg**  | **0.627** | **79.3%** | **8.49** | **83.5%** | **10.67** |

Improvement over old 58-feature model: AUC 0.611->0.627, ML_70 WR 78.4%->79.3%, Sharpe 8.15->8.49

- **AUC** = Area Under ROC Curve. Measures model's ability to separate winners from losers. 0.5 = random, 0.7+ = strong.
- **ML_70 WR** = Win rate when only taking the model's top 30% scored opportunities.
- **ML_70 Sharpe** = Risk-adjusted return (Sharpe ratio) at the ML_70 threshold.

### MFE Model Walk-Forward Results (10-30 tier)

MFE = Maximum Favorable Excursion: best price reached during the pattern window. For longs: (max_close - entry) / entry * 100. For shorts: (entry - min_close) / entry * 100. Always >= 0.

| Val Year | AUC_mfe | Spearman | R2    | Base MFE | ML_70 MFE | ML_90 MFE | ML_90 WR |
|----------|---------|----------|-------|----------|-----------|-----------|----------|
| 2018     | 0.670   | 0.344    | 0.149 | 5.24%    | 7.37%     | 9.43%     | 77.4%    |
| 2019     | 0.648   | 0.291    | 0.132 | 4.47%    | 6.41%     | 7.95%     | 63.1%    |
| 2020     | 0.698   | 0.398    | 0.164 | 6.80%    | 10.70%    | 14.39%    | 76.1%    |
| 2021     | 0.676   | 0.358    | 0.153 | 5.47%    | 8.06%     | 10.24%    | 76.7%    |
| 2022     | 0.726   | 0.463    | 0.186 | 8.02%    | 11.83%    | 14.60%    | 88.2%    |
| 2023     | 0.687   | 0.384    | 0.218 | 5.26%    | 8.21%     | 10.98%    | 79.2%    |
| 2024     | 0.664   | 0.337    | 0.171 | 5.18%    | 7.75%     | 10.07%    | 75.4%    |
| 2025     | 0.676   | 0.351    | 0.188 | 5.17%    | 7.97%     | 10.66%    | 71.5%    |
| **Avg**  | **0.681** | **0.366** | **0.170** |    |           |           |          |

- **AUC_mfe** = AUC against MFE median (can model rank high-MFE vs low-MFE opportunities?)
- **Spearman** = Rank correlation between predicted and actual MFE
- **ML_70 MFE** = Average MFE when taking top 30% scored opportunities (40-55% higher than baseline)
- **2025 holdout** (final model): AUC_mfe=0.676, Spearman=0.351, R2=0.188

### 31_60 SR Model Walk-Forward Results

54.4M samples. Optuna params: LR=0.071, num_leaves=76, min_child_samples=224.

| Val Year | AUC   | ML_70 WR | ML_70 Sharpe | ML_90 WR | ML_90 Sharpe |
|----------|-------|----------|--------------|----------|--------------|
| 2018     | 0.509 | 69.7%    | 4.05         | 70.3%    | 4.44         |
| 2019     | 0.478 | 64.6%    | 3.62         | 54.5%    | 0.79         |
| 2020     | 0.711 | 83.9%    | 12.40        | 88.6%    | 15.25        |
| 2021     | 0.602 | 76.5%    | 8.50         | 79.7%    | 9.82         |
| 2022     | 0.524 | 86.5%    | 13.07        | 89.9%    | 15.17        |
| 2023     | 0.664 | 82.8%    | 10.33        | 88.7%    | 12.35        |
| 2024     | 0.637 | 79.7%    | 8.93         | 82.3%    | 9.65         |
| 2025     | 0.721 | 82.1%    | 9.77         | 87.5%    | 12.52        |
| **Avg**  | **0.606** | **78.2%** | **8.83** | **80.2%** | **10.00** |

- 2018-2019 are structural dead zones (AUC ~0.5) -- pre-2018 training data lacks signal for 31-60 day patterns
- 2020-2025 all show strong lift, with 2025 holdout the strongest (AUC=0.721, +20.5pp WR lift)
- Average ML_70 WR (78.2%) comparable to 10_30 tier (79.3%)
- Optuna LR had to be constrained to (0.005, 0.08) -- higher LR caused overfitting at 54M samples

### 31_60 MFE Model Walk-Forward Results

54.4M samples. Uses same Optuna params as 31_60 SR.

| Val Year | AUC_mfe | Spearman | Base MFE | ML_70 MFE | ML_90 MFE | ML_90 WR |
|----------|---------|----------|----------|-----------|-----------|----------|
| 2018     | 0.631   | 0.275    | 7.60%    | 10.43%    | 13.57%    | 72.4%    |
| 2019     | 0.649   | 0.312    | 7.39%    | 10.05%    | 12.64%    | 66.4%    |
| 2020     | 0.729   | 0.458    | 10.74%   | 15.66%    | 22.40%    | 85.2%    |
| 2021     | 0.680   | 0.370    | 8.52%    | 11.84%    | 15.15%    | 76.2%    |
| 2022     | 0.713   | 0.438    | 11.95%   | 16.38%    | 20.66%    | 86.1%    |
| 2023     | 0.673   | 0.360    | 8.39%    | 12.06%    | 16.46%    | 77.7%    |
| 2024     | 0.650   | 0.310    | 8.32%    | 11.36%    | 14.77%    | 72.3%    |
| 2025     | 0.674   | 0.364    | 8.36%    | 12.39%    | 17.56%    | 73.1%    |
| **Avg**  | **0.675** | **0.361** |        |           |           |          |

- Comparable to 10_30 MFE (avg AUC_mfe 0.681 vs 0.675), but with much larger MFE values
- ML_90 avg MFE ranges 12.6-22.4% vs 8.0-14.4% for 10_30 -- longer holds = bigger favorable moves

### 61_90 SR Model Walk-Forward Results

59.0M samples. Dead-zone pattern similar to 31_60 but more pronounced for 2022.

| Val Year | AUC   | ML_70 WR | ML_70 Sharpe | ML_90 WR | ML_90 Sharpe |
|----------|-------|----------|--------------|----------|--------------|
| 2018     | 0.527 | 70.8%    | 4.02         | 68.3%    | 3.85         |
| 2019     | 0.496 | 73.0%    | 6.36         | 61.9%    | 2.28         |
| 2020     | 0.732 | 88.1%    | 14.73        | 93.5%    | 18.67        |
| 2021     | 0.573 | 72.9%    | 7.33         | 76.8%    | 8.64         |
| 2022     | 0.435 | 76.2%    | 7.41         | 78.5%    | 8.50         |
| 2023     | 0.663 | 85.7%    | 11.43        | 92.2%    | 12.73        |
| 2024     | 0.634 | 78.8%    | 8.99         | 79.6%    | 9.17         |
| 2025     | 0.698 | 82.1%    | 10.07        | 86.4%    | 12.76        |
| **Avg**  | **0.595** | **78.5%** | **8.79** |        |              |

- 2022 has AUC below 0.5 (model slightly hurts) -- unique to 61_90 tier
- 2020 is exceptional (AUC=0.732, ML_90 WR=93.5%) -- COVID volatility created strong 61-90 day signal
- 2025 holdout strong: AUC=0.698, ML_70 WR=82.1%, Sharpe=10.07
- Return distribution: at ML_85, 61.5% chance of >5%, 44.2% of >10%, 19.2% of >20%
- Pattern: as holding period increases, peak performance gets higher but consistency gets lower

### 61_90 MFE Model Walk-Forward Results

59.0M samples. Biggest MFE values of all tiers.

| Val Year | AUC_mfe | Spearman | Base MFE | ML_70 MFE | ML_90 MFE | ML_90 WR |
|----------|---------|----------|----------|-----------|-----------|----------|
| 2018     | 0.597   | 0.214    | 9.40%    | 12.63%    | 16.23%    | 68.0%    |
| 2019     | 0.654   | 0.330    | 10.48%   | 13.40%    | 17.03%    | 69.8%    |
| 2020     | 0.752   | 0.502    | 15.30%   | 23.37%    | 32.90%    | 90.2%    |
| 2021     | 0.667   | 0.343    | 11.06%   | 14.87%    | 18.96%    | 73.6%    |
| 2022     | 0.702   | 0.422    | 14.14%   | 18.72%    | 24.50%    | 85.6%    |
| 2023     | 0.668   | 0.354    | 11.91%   | 16.52%    | 23.36%    | 78.6%    |
| 2024     | 0.644   | 0.302    | 10.94%   | 14.72%    | 18.63%    | 68.7%    |
| 2025     | 0.696   | 0.404    | 11.68%   | 18.17%    | 24.32%    | 78.7%    |
| **Avg**  | **0.672** | **0.359** |        |           |           |          |

- ML_90 MFE ranges 16-33% vs 8-15% (10_30) and 13-22% (31_60) -- longest holds = biggest moves
- 2020 exceptional: AUC_mfe=0.752, ML_90 MFE=32.9% (COVID volatility)
- Ideal for stock portfolio rotation where position sizing can be larger and no theta decay

### MFE vs SR Feature Importance Shift

| Feature | SR Rank | MFE Rank | Interpretation |
|---------|---------|----------|----------------|
| ctx_pct_from_52w_high | #2 | **#1** | Stocks near 52w lows have more room for favorable moves |
| mkt_yield_curve_10y2y | #1 | #7 | Macro drives direction more than magnitude |
| pat_daysOut | not top 20 | **#11** | Longer windows = mechanically higher MFE |
| cal_pe_year | #4 | #13 | PE cycle drives direction more than magnitude |

### Feature Importance (Final Model, Top 20)

| Rank | Feature | Gain | Notes |
|------|---------|------|-------|
| 1 | mkt_yield_curve_10y2y | 103M | Yield curve (10Y minus 2Y treasury) |
| 2 | ctx_pct_from_52w_high | 77M | Stock's distance from 52-week high |
| 3 | pat_avg_profit2 | 72M | Pattern's average profit (secondary calc) |
| 4 | cal_pe_year | 70M | Presidential election cycle phase (1-4) |
| 5 | mkt_credit_spread | 68M | HYG-LQD spread (credit risk) |
| 6 | mkt_vix_level | 60M | VIX level |
| 7 | mkt_fed_rate_level | 58M | Fed funds rate proxy (IRX) |
| 8 | mkt_fed_rate_direction | 54M | 60-day change in fed rate |
| 9 | mkt_spx_seasonal_ret | 51M | SPX seasonal avg return (NEW in V2) |
| 10 | cal_day_of_year | 49M | Day of year (1-365) |
| 11 | mkt_vix_term_structure | 41M | VIX term structure (contango/backwardation) |
| 12 | mkt_credit_spread_change_20d | 39M | 20-day change in credit spread |
| 13 | mkt_spx_seasonal_wr | 30M | SPX seasonal win rate (NEW in V2) |
| 14 | ctx_pct_from_52w_low | 29M | Stock's distance from 52-week low |
| 15 | mkt_sector_rotation | 28M | Sector rotation signal |
| 16 | cal_month | 25M | Calendar month |
| 17 | pat_depth_utilization | 24M | How many depth levels pattern passes |
| 18 | pat_direction | 23M | Long (1) or Short (0) |
| 19 | mkt_spx_dir_alignment | 23M | Trade aligned with SPX seasonal? (NEW) |
| 20 | mkt_spy_roc_20 | 20M | SPY 20-day rate of change |

**Key insight**: Pattern depth + macro regime >> technical indicators. The model answers: "Is this a robust deep pattern in a favorable macro environment?"

---

## V2 Innovations (vs V1)

### SPX Seasonal Regime Features
Core idea: Use SPX's own seasonal patterns (66 years of daily data, 1960-present) to create a market regime indicator by presidential election cycle phase and week-of-year.

- **Rolling lookups**: For each training year Y, the lookup uses SPX data from 1960 through Y-1 only (no data leakage)
- **15-trading-day forward returns** computed from raw SPX price data, grouped by (week_of_year, pe_phase)
- **Three-state alignment**: +1 (aligned with strong season, WR >= 55%), 0 (neutral, 45-55%), -1 (against, WR <= 45%)
- These 3 features rank #9, #13, and #19 in importance out of 60 features

### Neighborhood Features
Check if same pattern works when shifted +/- 1-4 weeks. A pattern that "falls off a cliff" after end date is fragile. A pattern that works across shifted windows is robust.

### Interaction Features
Pre-computed interactions (direction x market trend, depth x VIX, quality x regime) that help compensate for shallow tree models.

### Ensemble
Three gradient boosting models (LightGBM, XGBoost, CatBoost) averaged. Improves calibration and year-to-year stability vs single model.

---

## Return Distribution by ML Threshold (Walk-Forward, Out-of-Sample)

This answers: "If I filter to ML_XX, what percentage of those trades actually move X%?"
This is critical for building the production filter that narrows ~200,000 daily opportunities to ~80-100 trades.

### 10_30 Tier (11M samples, 10-30 day holds)

| Return Threshold | Baseline | ML_70 | ML_85 | ML_90 |
|---|---|---|---|---|
| > 0% (win) | 67.8% | 79.3% | 82.5% | 83.5% |
| > 2% | 49.8% | 62.9% | 67.8% | 70.3% |
| > 5% | 27.4% | 39.8% | 45.8% | 49.7% |
| > 8% | 14.1% | 23.4% | 29.0% | 32.7% |
| > 10% | 9.0% | 16.2% | 21.0% | 24.2% |
| > 15% | 3.1% | 6.5% | 9.2% | 11.1% |
| > 20% | 1.2% | 2.7% | 4.2% | 5.2% |

### 31_60 Tier (17.3M samples, 31-60 day holds)

| Return Threshold | Baseline | ML_70 | ML_85 | ML_90 |
|---|---|---|---|---|
| > 0% (win) | 68.6% | 78.1% | 79.8% | 80.0% |
| > 2% | 57.6% | 68.0% | 70.7% | 71.6% |
| > 5% | 41.5% | 52.5% | 56.6% | 58.4% |
| > 8% | 27.8% | 38.6% | 43.6% | 46.0% |
| > 10% | 20.8% | 30.8% | 36.0% | 38.6% |
| > 15% | 9.7% | 16.5% | 21.1% | 23.6% |
| > 20% | 4.5% | 8.7% | 12.0% | 14.1% |

### 61_90 Tier (18.8M samples, 61-90 day holds)

| Return Threshold | Baseline | ML_70 | ML_85 | ML_90 |
|---|---|---|---|---|
| > 0% (win) | 65.3% | 78.5% | 81.3% | 82.4% |
| > 2% | 56.4% | 70.1% | 73.9% | 75.5% |
| > 5% | 43.8% | 58.5% | 61.5% | 63.8% |
| > 8% | 32.2% | 46.0% | 50.3% | 53.2% |
| > 10% | 25.8% | 38.7% | 44.2% | 47.0% |
| > 15% | 14.3% | 23.4% | 28.1% | 30.9% |
| > 20% | 7.9% | 14.1% | 19.2% | 21.5% |

### What This Means

**ML filtering roughly doubles the probability of large returns.** At ML_90 vs baseline, the chance of a >10% return goes from 9% to 24% (10_30) and 21% to 39% (31_60).

**31_60 naturally produces bigger moves.** More time in trade = more room to run. At ML_85, 56.6% chance of >5% stock move vs 45.8% for 10_30. For options, a 31_60 ML_85 trade gives >10% stock move 36% of the time.

**The ML score alone won't narrow 200K to 80-100 trades.** ML_85 keeps the top 15% (~30,000 opportunities). Additional stacked filters are needed:

1. **ML_score >= threshold** (e.g., 85 keeps top 15% = ~30,000)
2. **pred_return minimum** (e.g., > 3% expected return cuts further)
3. **win_prob minimum** (e.g., > 75% calibrated probability)
4. **pred_mfe minimum** (e.g., > 5% expected favorable excursion)
5. **Sector diversification** (max 2-3 trades per sector)
6. **No duplicate patterns** on same stock (take best per stock)
7. **Rank remaining by composite** (win_prob * pred_mfe or similar)
8. **Take top N** (80-100 best trades)

These filters are multiplicative -- each cuts the pool, and the combination of high ML score + high predicted return + high win probability + high MFE selects the cream. This is the strategy filter (step 11 in Next Steps).

---

## Experiments Tested and Rejected

### 1. PE-Cycle Walk-Forward
**Idea**: Train only on years with matching presidential election cycle phase.
**Result**: Average AUC 0.604 vs 0.621 for standard WF. Less training data hurts more than phase-matching helps.

### 2. Against-Season Pre-Filter
**Idea**: Remove trades where pattern direction opposes the SPX seasonal tendency.
**Result**: Removed ~25% of samples. Hurt model quality. The model uses alignment features better as inputs than as hard filters.

### 3. Separate Long/Short Models
**Idea**: Train separate 3-model ensembles for longs and shorts, since error analysis showed longs (67% WR) and shorts (59% WR) behave differently.
**Result**:

| Metric | Combined Model | Split Models |
|--------|---------------|--------------|
| Avg AUC | 0.595 | 0.609 |
| Avg ML_70 WR | 79.0% | 77.3% |
| Avg ML_70 Sharpe | 8.69 | 8.12 |

Split AUC was slightly better, but practical trading metrics (WR, Sharpe) were worse. The short model was highly unstable (AUC ranged 0.500 to 0.716, negative Sharpe in 2019). Halving the training data hurts more than direction specialization helps.

**Recurring lesson**: Reducing training data rarely pays off, even when the subset seems more homogeneous. The combined model handles direction through pat_direction and interaction features.

---

## Error Analysis (2025 Holdout)

- **Alignment is a strong separator**: Against-season shorts are coin flips (49.9% WR), aligned shorts are profitable (62.8%)
- **Big loser profile**: Shallow patterns (avg depth 5.0), volatile tech stocks, against-season alignment
- **Depth matters most**: Deep patterns (depth >= 15) have 70%+ WR regardless of other factors
- **Short trades are harder**: Longs average 67% WR vs shorts 59% WR across all conditions
- **Model calibration**: ML_70 threshold filters to 79% WR, ML_85 to 82.7% WR

---

## Project Files

| File | Purpose |
|------|---------|
| `config_ml.py` | Paths, sector mappings, 475-stock TICKER_SECTOR dict, constants |
| `build_training_data.py` | Builds training parquet from price CSVs + opportunity files. Joblib parallel. |
| `train_model.py` | Walk-forward validation, Optuna tuning, ensemble training. Supports `--target sr/mfe`, `--tier`, `--split-direction`, `--pe-cycle`, `--skip-optuna`, `--wf-only`, `--final-only` flags. |
| `feature_engine.py` | FeatureEngine class for production scoring (computes features for live opportunities) |
| `results/v2_tuned_params.json` | Optuna best hyperparameters |
| `results/v2_walk_forward_results.json` | Walk-forward results, 10_30 tier (combined model) |
| `results/v2_walk_forward_results_31_60.json` | Walk-forward results, 31_60 tier |
| `results/v2_tuned_params_31_60.json` | Optuna best params for 31_60 tier |
| `results/calibration_sr_31_60.json` | SR calibration table for 31_60 tier |
| `results/wf_predictions_sr_31_60.parquet` | 17.3M walk-forward predictions for 31_60 |
| `results/v2_walk_forward_results_31_60_mfe.json` | Walk-forward results, 31_60 MFE |
| `results/calibration_mfe_31_60.json` | MFE calibration table for 31_60 tier |
| `results/wf_predictions_mfe_31_60.parquet` | 17.3M walk-forward MFE predictions for 31_60 |
| `results/v2_walk_forward_results_61_90.json` | Walk-forward results, 61_90 SR |
| `results/calibration_sr_61_90.json` | SR calibration table for 61_90 tier |
| `results/wf_predictions_sr_61_90.parquet` | 18.8M walk-forward predictions for 61_90 |
| `results/v2_tuned_params_61_90.json` | Optuna best params for 61_90 tier |
| `results/v2_walk_forward_results_split.json` | Walk-forward results (split long/short experiment) |

### Saved Model Files (not in git, regenerated by training)
- **10_30 SR**: v2_lgb_20260312.txt, v2_xgb_20260312.json, v2_catboost_20260312.cbm
- **10_30 MFE**: v2_lgb_mfe_20260312.txt (iter 267), v2_xgb_mfe_20260312.json (iter 190), v2_catboost_mfe_20260312.cbm (iter 190)
- **31_60 SR**: v2_lgb_31_60_20260314.txt (iter 108), v2_xgb_31_60_20260314.json (iter 266), v2_catboost_31_60_20260314.cbm (iter 154)
- **31_60 MFE**: v2_lgb_31_60_mfe_20260314.txt (iter 127), v2_xgb_31_60_mfe_20260314.json (iter 136), v2_catboost_31_60_mfe_20260314.cbm (iter 90)
- **61_90 SR**: v2_lgb_61_90_20260314.txt (iter 3), v2_xgb_61_90_20260314.json (iter 5), v2_catboost_61_90_20260314.cbm (iter 750)
- **61_90 MFE**: v2_lgb_61_90_mfe_20260314.txt (iter 260), v2_xgb_61_90_mfe_20260314.json (iter 513), v2_catboost_61_90_mfe_20260314.cbm (iter 375)

### Production Service (`ml_scorer/ml_scorer/`)
Self-contained Flask service, copy entire `ml_scorer/` subdir to production servers.
- Endpoints: POST /score, GET /health, GET /tiers
- Multi-tier: TIERS dict in config.py maps tier names to model file paths
- Feature engine computes all 59 features from live price data + opportunity files
- Empirical calibration tables (20 quantile bins) for win_prob and P(hit predicted)
- Per-opportunity output: pred_return, pred_mfe, win_prob, p_hit_return, p_hit_mfe, ml_score (0-100)

### Safeguards Added (post pat_daysOut bug)
1. **train_model.py -- REQUIRED_FEATURES at data load**: Raises RuntimeError if pat_daysOut, pat_direction, pat_sharpe_ratio, or pat_deepest_pass are missing
2. **train_model.py -- Feature count validation after model save**: All 3 ensemble models must match expected feature count
3. **scorer.py -- Feature count validation at startup**: Production service validates each loaded model matches config

---

## V1 vs V2 Comparison

| Metric | V1 | V2 SR | V2 MFE |
|--------|----|----|------|
| Training years | 2015-2025 | 2000-2025 | 2000-2025 |
| Samples | 15.6M | 34.7M | 34.7M |
| Features | 73 (30 useless) | 59 (incl. pat_daysOut) | 59 (same) |
| Model | Single LightGBM | 3-model ensemble | 3-model ensemble |
| Target | Binary (hit_target) | Regression (actual_return) | Regression (mfe_return) |
| Best iter (LGB) | 7-33 (shallow) | 146 | 267 (deeper) |
| 2025 AUC | 0.632 | 0.671 | 0.676 (vs MFE median) |
| 2025 ML_70 WR | 83.9% | 78.1% | 68.2% |
| Avg WF AUC | 0.589 | 0.627 | 0.681 |
| Consistency | Poor (0.464-0.718) | Better (0.564-0.671) | Good (0.648-0.726) |

V2 trades some peak performance for much better year-to-year consistency. MFE model complements SR by predicting favorable excursion magnitude for profit target sizing.

---

## Next Steps

1. ~~Empirical calibration~~ DONE - Calibration tables built from WF predictions
2. ~~Upgrade feature_engine.py~~ DONE - All 59 features implemented for production
3. ~~Build production service~~ DONE - Flask service at ml_scorer/ml_scorer/
4. ~~SR model retrain (10_30)~~ DONE - Retrained with 59 features, improved results (AUC 0.627, ML_70 79.3%)
5. ~~Rebuild SR calibration (10_30)~~ DONE - calibration_sr.json rebuilt from new WF predictions
6. ~~31_60 SR model~~ DONE - WF complete (AUC 0.606, ML_70 78.2%, Sharpe 8.83), calibration built
7. ~~31_60 MFE model~~ DONE - WF complete (AUC_mfe 0.675, Spearman 0.361), calibration built
8. ~~Build 61_90 training data~~ DONE - 59.0M samples, 3.3 GB parquet
9. ~~61_90 SR model~~ DONE - WF complete (AUC 0.595, ML_70 78.5%, Sharpe 8.79), calibration built
9b. ~~61_90 MFE model~~ DONE - WF complete (AUC_mfe 0.672, Spearman 0.359), calibration built
10. ~~Train final models~~ DONE - All tiers, both targets trained with --skip-optuna --final-only
11. ~~Update production model files~~ DONE - All 18 model files + 6 calibration files copied to ml_scorer/ml_scorer/
11. **Strategy filter** - ML threshold, position sizing, sector limits
12. **UI integration** - ML score column in TradeWave opportunity table
13. **Auto-trade** - Broker API integration (IBKR or Tastytrade)

### Production Output Per Opportunity
- pred_return (SR model), pred_mfe (MFE model), win_prob, P(hit pred_return), P(hit pred_mfe), ml_score (0-100)

### Deferred Ideas
- CL (crude oil) and GC (gold) seasonal features for commodity-sector regime signals
- Train model to predict optimal profit target and stop loss per pattern
- Depth-tiered models (separate models for shallow vs deep patterns)

---

## Environment

- **Training machine**: Windows 11, 24-core / 64GB RAM
- **Python**: 3.8, pandas 1.2.4
- **ML libraries**: LightGBM 4.6.0, XGBoost 3.2.0, CatBoost 1.2.10, Optuna
- **Build time**: ~54 min (training data), ~90 min (walk-forward), ~15 min (final model)
