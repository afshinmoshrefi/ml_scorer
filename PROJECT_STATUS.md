# ML Pattern Scorer - Project Status

## What This Is

Machine learning model that scores TradeWave seasonal stock pattern opportunities 0-100 for "probability of playing out." Gatekeeper for automated options trading. Personal use, $10K starting capital, options on 10-30 day patterns.

A "pattern" is a seasonal tendency for a stock to move in a direction over specific calendar dates. For example: "AAPL tends to go up 3% in the first two weeks of November, and this has been true 24 out of the last 30 years." The model scores how likely each pattern is to succeed given current market conditions.

---

## Current Status: SR + MFE Models Complete, Calibration Next

### Model Architecture
- **Two models per tier**: SR (predicts actual_return) and MFE (predicts max favorable excursion)
- **3-model ensemble** each: LightGBM + XGBoost + CatBoost, predictions averaged
- **59 features** across 6 groups: pattern intrinsic, technical, market regime, context, calendar, interactions
- **Optuna hyperparameter tuning**: 75 trials on 2M sample subset, separate params per target
- **VIX > 35 hurricane filter**: Removes ~4.8% of samples during market panics
- **Multi-tier support**: `--tier 10_30`, `--tier 31_60`, etc. via CLI

### Training Data
- **34.7 million samples** x 66 columns (includes mfe_return), 1.9 GB parquet
- **Training years**: 2000-2025 (26 years)
- **475 S&P 500 stocks**, 116 pattern combinations each
- Built with joblib parallelization (N_JOBS=24, ~54 min build time)

### SR Model Walk-Forward Results (10-30 tier)

Config: Standard walk-forward + VIX>35 filter + SPX seasonal features as model inputs

| Val Year | AUC   | ML_70 WR | ML_70 Sharpe | ML_70 Trades |
|----------|-------|----------|--------------|--------------|
| 2018     | 0.581 | 76.5%    | 6.56         | 419K         |
| 2019     | 0.551 | 70.0%    | 4.03         | 425K         |
| 2020     | 0.635 | 76.5%    | 7.41         | 346K         |
| 2021     | 0.599 | 77.5%    | 8.20         | 424K         |
| 2022     | 0.544 | 86.4%    | 12.46        | 420K         |
| 2023     | 0.671 | 83.0%    | 10.61        | 424K         |
| 2024     | 0.627 | 78.0%    | 8.36         | 425K         |
| 2025     | 0.679 | 79.0%    | 7.60         | 418K         |
| **Avg**  | **0.611** | **78.4%** | **8.15** |          |

- **AUC** = Area Under ROC Curve. Measures model's ability to separate winners from losers. 0.5 = random, 0.7+ = strong.
- **ML_70 WR** = Win rate when only taking the model's top 30% scored opportunities.
- **ML_70 Sharpe** = Risk-adjusted return (Sharpe ratio) at the ML_70 threshold.
- **2025 holdout** (final model): AUC=0.679, ML_70 79.0% WR Sharpe 7.60, ML_85 82.7% WR Sharpe 9.41

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
| `results/v2_walk_forward_results.json` | Walk-forward results (combined model) |
| `results/v2_walk_forward_results_split.json` | Walk-forward results (split long/short experiment) |

### Saved Model Files (not in git, regenerated by training)
- SR: `models/v2_lgb_20260311.txt` (iter 146), `v2_xgb_20260311.json` (iter 152), `v2_catboost_20260311.cbm` (iter 20)
- MFE: `models/v2_lgb_mfe_20260312.txt` (iter 267), `v2_xgb_mfe_20260312.json` (iter 190), `v2_catboost_mfe_20260312.cbm` (iter 190)

---

## V1 vs V2 Comparison

| Metric | V1 | V2 SR | V2 MFE |
|--------|----|----|------|
| Training years | 2015-2025 | 2000-2025 | 2000-2025 |
| Samples | 15.6M | 34.7M | 34.7M |
| Features | 73 (30 useless) | 59 (all useful) | 59 (same) |
| Model | Single LightGBM | 3-model ensemble | 3-model ensemble |
| Target | Binary (hit_target) | Regression (actual_return) | Regression (mfe_return) |
| Best iter (LGB) | 7-33 (shallow) | 146 | 267 (deeper) |
| 2025 AUC | 0.632 | 0.679 | 0.676 (vs MFE median) |
| 2025 ML_70 WR | 83.9% | 79.0% | 68.2% |
| Consistency | Poor (0.464-0.718) | Better (0.544-0.679) | Good (0.648-0.726) |

V2 trades some peak performance for much better year-to-year consistency. MFE model complements SR by predicting favorable excursion magnitude for profit target sizing.

---

## Next Steps

1. **Empirical calibration** - Save per-sample WF predictions, build calibration tables for win_prob, P(hit pred_return), P(hit pred_mfe)
2. **Upgrade feature_engine.py** - Add 16 missing V2 features for production scoring
3. **Build production scorer** (`score_opportunities.py`) - Score daily opportunities with both models
4. **Strategy filter** - ML threshold, position sizing, sector limits
5. **UI integration** - ML score column in TradeWave opportunity table
6. **Auto-trade** - Broker API integration (IBKR or Tastytrade)

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
