# ML Pattern Scorer V2 -- Project Context for Claude Code

## CRITICAL: Read this entire file before doing any work.

## What This Is

ML scoring system that rates TradeWave seasonal stock pattern opportunities 0-100. Uses a 3-model ensemble (LightGBM + XGBoost + CatBoost) predicting continuous returns, with empirical calibration for win probabilities. Two models per tier: SR (stock return) predicts close-to-close return, MFE (max favorable excursion) predicts best possible exit price during the holding window.

Two-track trading strategy:
- **Options account** ($10K): 10-30 day patterns, ATM/ITM calls/puts
- **Stock portfolio** (larger capital): 31-90 day patterns, direct equity positions

## Current Status: V2 COMPLETE -- All 3 Tiers Production-Ready (2026-03-14)

All models trained, validated, deployed to production service. ETF inference supported without retraining.

---

## Repository Structure

```
ml_scorer/                          # Root -- training pipeline
  config_ml.py                      # Training config: paths, TICKER_SECTOR (475 stocks),
                                    #   ETF_SECTOR (157 ETFs), year/PE combos, thresholds
  build_training_data.py            # Builds training parquet from price CSVs + opp files
                                    #   joblib parallel (N_JOBS=24), supports --tier, --etf
  train_model.py                    # Walk-forward validation + final model training
                                    #   3-model ensemble, Optuna tuning, --save-predictions
  feature_engine.py                 # V1 feature engine (training-time, not used in prod)
  generate_report.py                # Generates ML_Scorer_V2_Analysis.docx
  train_overnight.sh                # Bash script for batch training all tiers

  features/                         # Training parquets (gitignored, ~8 GB total)
    training_data_10_30.parquet     #   34.7M samples, 66 cols, 2.0 GB
    training_data_31_60.parquet     #   54.4M samples, 66 cols, 2.9 GB
    training_data_61_90.parquet     #   59.0M samples, 66 cols, 3.1 GB

  results/                          # Walk-forward results, params, feature importance
    v2_walk_forward_results*.json   #   Per-window AUC, WR, Sharpe for each tier/target
    v2_tuned_params*.json           #   Optuna-tuned hyperparameters per tier
    v2_feature_importance*.csv      #   Feature importance rankings per tier
    calibration_*.json              #   Training-side calibration (copied to prod)
    wf_predictions_*.parquet        #   Per-sample WF predictions (gitignored)

  models/                           # V1 model (gitignored, superseded by prod models)

  ml_scorer/                        # PRODUCTION SERVICE (self-contained Flask package)
    __init__.py
    __main__.py                     # Entry: python -m ml_scorer
    app.py                          # Flask: POST /score, GET /health, GET /tiers
    config.py                       # Production config: env-var paths, TIERS dict, FEATURE_COLS
    feature_engine.py               # Production feature computation (59 features)
    scorer.py                       # ModelEnsemble: loads 3 models, averages predictions
    models/                         # 18 binary model files (3 algos x 2 targets x 3 tiers)
      v2_lgb_20260312.txt           #   LightGBM models (~100 KB, text format)
      v2_xgb_20260312.json          #   XGBoost models (~200 KB, JSON format)
      v2_catboost_20260312.cbm      #   CatBoost models (~50-800 KB, binary format)
      v2_lgb_mfe_20260312.txt       #   _mfe = MFE target model
      v2_lgb_31_60_20260314.txt     #   _31_60 = 31-60 day tier
      ...                           #   (18 files total, naming: v2_{algo}[_{tier}][_mfe]_{date}.{ext})
    calibration/                    # 6 JSON calibration files (SR + MFE x 3 tiers)
      calibration_sr.json           #   20-bin empirical lookup: predicted_return -> win_prob, P(hit)
      calibration_mfe.json
      calibration_sr_31_60.json
      calibration_mfe_31_60.json
      calibration_sr_61_90.json
      calibration_mfe_61_90.json
```

---

## Architecture Overview

### Training Pipeline (build_training_data.py -> train_model.py)

1. **Build training data**: For each of 475 S&P 500 symbols, replay seasonal patterns across 2000-2025 training years. Compute 59 features per sample from price data, opportunity files, and macro indicators. Output: parquet file per tier.

2. **Optuna tuning**: 75 trials on 2M subsample to find optimal LightGBM hyperparameters. Learning rate constrained to (0.005, 0.08) for larger tiers to prevent overfitting.

3. **Walk-forward validation**: 8 expanding windows (train 2000-2017 -> val 2018, ..., train 2000-2024 -> val 2025). Each window trains LGB + XGB + CatBoost ensemble. Predictions averaged. Evaluated with AUC, win rate at ML_70/85/90 percentile thresholds, Sharpe ratio.

4. **Final model**: Train on all data (2000-2025), save 3 model files per target per tier. Build empirical calibration tables from walk-forward predictions.

### Production Service (ml_scorer/ package)

Self-contained Flask app. Deploy by copying the `ml_scorer/` folder.

**Endpoints:**
- `POST /score` -- Score one or batch of opportunities. Input: symbol, date, daysOut, direction, tier. Output: pred_return, pred_mfe, win_prob, p_hit_return, p_hit_mfe, ml_score (0-100).
- `GET /health` -- Service health check
- `GET /tiers` -- List loaded tiers

**Deployment:**
```bash
# Only one env var needed -- all paths derive from it
export ML_SCORER_DATA_DIR=/home/flask/data
cd /path/to/ml_scorer
python -m ml_scorer
# Listens on 0.0.0.0:5090 (configurable via ML_SCORER_HOST/PORT)
```

**Path resolution:** Models and calibration are relative to the package dir (travel with the code). Data paths (CSVs, opportunity files) are relative to ML_SCORER_DATA_DIR. The feature engine searches US/ then ETF/ then INDX/ for price CSVs, and sp500/opp_by_symbol/ then ETF/opp_by_symbol/ for pattern files.

---

## The 59 Features (FEATURE_COLS)

All tiers and both SR/MFE models use the same 59 features. Defined in both `ml_scorer/config.py` (production) and `train_model.py` (training). Must always stay in sync.

**CRITICAL: pat_daysOut MUST be included.** A pattern is defined by [start_date, ticker, days_out, history_years]. The SR model was once accidentally trained without it (58 features) and couldn't distinguish 10-day from 30-day holds. Safeguards validate feature counts at training and serving time.

| Group | Count | Key Features |
|-------|-------|-------------|
| Pattern-Intrinsic | 22 | pat_deepest_pass, pat_sharpe_ratio, pat_direction, pat_daysOut, pat_neighbor_avg_wr, pat_sharpness, pat_consistency_std |
| Technical | 5 | ta_trend_long, ta_price_vs_sma200, ta_sma50_vs_sma200, ta_rvol_20 |
| Market Regime | 16 | mkt_vix_level, mkt_yield_curve_10y2y, mkt_credit_spread, mkt_fed_rate_level |
| SPX Seasonal | 4 | mkt_spx_seasonal_wr, mkt_spx_seasonal_ret, mkt_spx_seasonal_regime, mkt_spx_dir_alignment |
| Context | 2 | ctx_pct_from_52w_high, ctx_pct_from_52w_low |
| Calendar | 5 | cal_pe_year, cal_day_of_year, cal_month, cal_is_opex_week, cal_week_of_month |
| Interactions | 4 | pat_dir_x_mkt_trend, pat_dir_x_sector_trend, pat_depth_x_vix, pat_quality_x_regime |

---

## Model Performance (Walk-Forward Validation)

### 10-30 Day Tier (34.7M samples, 475 S&P 500 stocks)

| Val Year | AUC   | ML_70 WR | ML_70 Sharpe | ML_90 WR |
|----------|-------|----------|--------------|----------|
| 2018     | 0.614 | 77.8%    | 7.04         | 80.5%    |
| 2019     | 0.637 | 77.8%    | 6.93         | 82.7%    |
| 2020     | 0.651 | 78.2%    | 8.32         | 82.2%    |
| 2021     | 0.611 | 78.8%    | 8.68         | 83.4%    |
| 2022     | 0.564 | 86.0%    | 12.31        | 90.4%    |
| 2023     | 0.644 | 79.7%    | 9.00         | 86.5%    |
| 2024     | 0.623 | 77.8%    | 8.21         | 78.8%    |
| 2025     | 0.671 | 78.1%    | 7.45         | 83.2%    |
| **Avg**  | **0.627** | **79.3%** | **8.49** | **83.5%** |

### 31-60 Day Tier (54.4M samples) -- Avg AUC 0.606, ML_70 WR 78.2%, Sharpe 8.83
### 61-90 Day Tier (59.0M samples) -- Avg AUC 0.595, ML_70 WR 78.5%, Sharpe 8.79

2018-2019 are structural dead zones for 31-60 and 61-90 tiers (AUC ~0.5). Pre-2018 training data doesn't contain enough learnable signal for longer-hold patterns. 2020-2025 show strong signal across all tiers.

ML_70/ML_85/ML_90 = percentile thresholds. ML_70 means the model's top 30% of predictions by predicted return. At ML_70, all tiers achieve ~78-80% win rate with Sharpe ratios of 7-12.

---

## Key Design Decisions and Findings

### What Drives the Model
Pattern depth + macro regime >> technical indicators. The model answers: "is this a robust deep pattern in a favorable macro environment?" Top features: pat_deepest_pass, yield_curve_10y2y, cal_pe_year, credit_spread, VIX.

### VIX > 35 Hurricane Filter
Removes ~4.8% of training samples where VIX > 35. During market panics, seasonal patterns break down regardless of quality. Applied as sample filter in training, not as a feature. Production service refuses to score when VIX > 35.

### Shallow Models
Models are intentionally shallow (3-750 iterations, most under 200). The seasonal signal exists but is weak. Deeper models overfit. The 3-model ensemble compensates by capturing different feature interactions. Model files are tiny (~100-800 KB each).

### Regression Target
Models predict continuous actual_return (%), not binary win/lose. This gives richer signal -- the model learns magnitude, not just direction. Win probability is derived via empirical calibration tables from walk-forward predictions.

### ETF Experiment (Tested and Rejected)
Adding 157 ETFs to training data hurt model quality (AUC 0.627 -> 0.600, ML_70 WR 79.3% -> 77.9%). ETFs have different seasonal dynamics than individual stocks. However, the stocks-only model generalizes well to ETFs at inference time -- the production feature engine was patched to resolve ETF price CSVs and opportunity files automatically.

### Experiments That Did Not Help
- **PE-cycle walk-forward**: Training only on matching PE-phase years reduced data too much.
- **Separate long/short models**: Halving data hurt more than direction specialization helped.
- **Against-season pre-filter**: Removing against-season samples lost too much training data. The model uses alignment features better as inputs.

---

## Training Data Structure

Training parquets contain 66 columns: 59 features + date + actual_return + hit_target + mfe_return + daysOut + symbol + direction.

Built from:
- **Price CSVs**: OHLCV data back to 1981+ (US stocks), 1994+ (ETFs), various (indices)
- **Opportunity files**: 116 gzip CSVs per symbol in opp_by_symbol/. Each contains patterns that currently qualify at various depth/win-rate combos.
- **Pattern replay**: For each training year (2000-2025), replay each pattern historically. Compute features from price data at entry date, compute actual return from close[entry] to close[entry+daysOut].

### Data Dependencies (must exist on the machine)
```
{DATA_DIR}/csv/US/          # Stock price CSVs (~3500 files)
{DATA_DIR}/csv/ETF/         # ETF CSVs (SPY, QQQ, sector ETFs, etc.)
{DATA_DIR}/csv/INDX/        # Index CSVs (VIX, US10Y, IRX, ADVN, DECL, etc.)
{DATA_DIR}/sp500/opp_by_symbol/  # 475 dirs, 116 gzip CSVs each
{DATA_DIR}/ETF/opp_by_symbol/    # 157 dirs (for ETF inference)
{DATA_DIR}/sp500_symbols.csv     # S&P 500 ticker list
```

Dev machine: `DATA_DIR = C:/seasonals/data`
Production: Set `ML_SCORER_DATA_DIR=/home/flask/data` (env var)

---

## How to Retrain

### Full Rebuild (all 3 tiers, ~10-12 hours on 24-core CPU)
```bash
# 1. Rebuild training data (if price data updated)
python build_training_data.py --tier 10_30 --njobs 24
python build_training_data.py --tier 31_60 --njobs 24
python build_training_data.py --tier 61_90 --njobs 24

# 2. Train all models (Optuna + walk-forward + final)
python train_model.py --tier 10_30 --target sr --save-predictions
python train_model.py --tier 10_30 --target mfe --save-predictions
python train_model.py --tier 31_60 --target sr --save-predictions
python train_model.py --tier 31_60 --target mfe --save-predictions
python train_model.py --tier 61_90 --target sr --save-predictions
python train_model.py --tier 61_90 --target mfe --save-predictions

# 3. Copy model files to production
cp models/v2_*.{txt,json,cbm} ml_scorer/models/
cp results/calibration_*.json ml_scorer/calibration/
```

### Quick Experiment (skip Optuna, WF only, no final model)
```bash
python train_model.py --tier 10_30 --target sr --skip-optuna --wf-only
```

### Key CLI flags
- `--tier 10_30|31_60|61_90` -- which tier to train
- `--target sr|mfe` -- which target (default: sr)
- `--skip-optuna` -- reuse saved Optuna params
- `--wf-only` -- walk-forward only, skip final model training
- `--save-predictions` -- save per-sample predictions for calibration
- `--data-path <path>` -- override training data file (for experiments)
- `--etf` (build_training_data.py) -- build ETF data instead of stocks

---

## How Patterns Work (Domain Knowledge)

- A **pattern** = seasonal tendency for a stock to move in a direction over specific calendar dates
- Defined by: (symbol, start_month_day, daysOut, direction_long_or_short)
- **Combo** = lookback window depth: e.g., `10_8` means look back 10 years, require 8 profitable (80% win rate)
- Deeper combos = more robust pattern. `30_24` > `10_8`
- **PE combos**: only count presidential election cycle years (every 4th year)
- opp_by_symbol files contain patterns that currently qualify (as of latest data)
- Training **replays** patterns historically: "if this pattern existed in 2018, would it have worked?"
- Some look-ahead bias accepted (pattern discovered from 2026 data applied to 2018). Walk-forward validation mitigates this.

---

## Pending / Next Steps

- Strategy filter implementation (ML threshold + pred_return + win_prob + sector limits)
- Daily scoring cron (score_opportunities.py)
- UI integration (ML score column in TradeWave)
- Auto-trade (broker API integration)
- Monthly retraining cadence

---

## User Preferences

- Never use em dashes in generated text
- Work autonomously, don't ask for confirmation on file edits or commands
- N_JOBS=24 on the 24-core/64GB Windows dev machine - its only relevant to the computer for training not for production rollout
- Python 3.12 on Windows, dependencies: lightgbm, xgboost, catboost, optuna, joblib, pandas, numpy, scikit-learn, pyarrow, flask
