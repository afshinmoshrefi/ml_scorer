# ML Pattern Scorer -- Production Service

## What This Is

Self-contained Flask service that scores TradeWave seasonal stock/ETF pattern opportunities. Takes a symbol, date, daysOut, and direction, computes 59 features from live price data, runs them through a 3-model ensemble (LightGBM + XGBoost + CatBoost), and returns calibrated predictions.

**Two models per tier:**
- **SR model**: predicts actual_return (close-to-close return %). Used for win probability.
- **MFE model**: predicts mfe_return (max favorable excursion %). Used for profit target sizing.

**Three tiers:** 10_30 (10-30 day holds), 31_60 (31-60 days), 61_90 (61-90 days).

---

## Folder Structure

```
ml_scorer/
  opp_scorer.py           # Entry point: python opp_scorer.py
  app.py                  # Flask app: POST /score, GET /health, GET /tiers
  scorer_config.py        # ML config: tiers, features, models (paths from central config.py)
  feature_engine.py       # Computes all 59 features for a single opportunity
  scorer.py               # ModelEnsemble: loads models, averages predictions, calibrates
  opp_to_parquet.py       # Nightly cron: builds ML parquet caches from opportunity data
  requirements.txt        # pip install -r requirements.txt
  CLAUDE.md               # This file
  models/                 # 18 model files (3 algos x 2 targets x 3 tiers)
  calibration/            # 6 calibration JSONs (20-bin empirical lookup tables)
```

---

## Deployment

### Quick Start

```bash
cd /home/flask/ml_scorer
pip install -r requirements.txt
python opp_scorer.py
```

The service starts on `0.0.0.0:5090`. ML-specific configuration (tiers, features, models) is in `scorer_config.py`. Path configuration (DATA_DIR, CSV_DIR) is imported from the central `/home/flask/config.py` on Linux servers. On Windows dev, paths fall back to `C:/seasonals/data`. **IMPORTANT: Never create a file named `config.py` in this directory -- it would shadow the central TradeWave config and break deployments.**

### Production Deployment (systemd)

```ini
[Unit]
Description=ML Pattern Scorer
After=network.target

[Service]
Type=simple
User=flask
WorkingDirectory=/home/flask/ml_scorer
ExecStart=/usr/bin/python3 opp_scorer.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Production Deployment (gunicorn)

For production load, use gunicorn instead of the Flask dev server:

```bash
pip install gunicorn
cd /home/flask/ml_scorer
gunicorn -w 2 -b 0.0.0.0:5090 app:app
```

Note: `app:app` refers to the `app` object in `app.py`. Models load once at import time, so each gunicorn worker loads its own copy. 2 workers is recommended (models are small, ~5 MB total).

### Required Data on the Machine

```
/home/flask/data/
  csv/US/                    # Stock price CSVs (OHLCV, ~3500 files, back to 1981+)
  csv/ETF/                   # ETF CSVs (SPY, QQQ, sector ETFs, VIX-related, etc.)
  csv/INDX/                  # Index CSVs (VIX, VIX3M, US10Y, US2Y, IRX, ADVN, DECN, SPX)
  sp500/opp_by_symbol/       # 475 dirs (one per S&P stock), 116 gzip CSVs each
  ETF/opp_by_symbol/         # 157 dirs (for ETF inference)
  sp500_symbols.csv          # S&P 500 ticker list
```

### Parquet Cache (for fast batch scoring)

The ML scorer can read pre-built parquet caches instead of scanning hundreds of gzip files per symbol. These are generated nightly by `opp_to_parquet.py`.

```bash
# Cron: 0 1 * * * cd /home/flask/ml_scorer && python3 opp_to_parquet.py
python3 opp_to_parquet.py              # auto-detect dates (rest of week + next week on Sat)
python3 opp_to_parquet.py 2026-03-17   # specific date
python3 opp_to_parquet.py --dry-run    # preview without writing
```

Output: `data/<market>/ml_cache_<YYYY-MM-DD>.parquet` (one per market per date).
Covers 6 markets: DOW 30, NASDAQ 100, S&P 500, RUSSELL 1000, WILSHIRE 5000, ETFs.
Build time: ~4 min for one day across all 6 markets.
Self-cleaning: parquets live alongside the opportunity data, so they auto-delete when monthly opp refresh replaces the folder.

---

## API

### POST /score

Score one or multiple opportunities.

**Single request:**
```json
{
  "symbol": "AAPL",
  "date": "2026-03-15",
  "daysOut": 20,
  "direction": "l",
  "tier": "10_30"
}
```

**Batch request:**
```json
{
  "opportunities": [
    {"symbol": "AAPL", "date": "2026-03-15", "daysOut": 20, "direction": "l"},
    {"symbol": "XLE", "date": "2026-03-15", "daysOut": 20, "direction": "l"}
  ],
  "tier": "10_30"
}
```

**Response:**
```json
{
  "results": [
    {
      "symbol": "AAPL",
      "date": "2026-03-15",
      "daysOut": 20,
      "direction": "l",
      "pred_return": 2.34,
      "pred_mfe": 5.12,
      "win_prob": 0.78,
      "p_hit_return": 0.58,
      "p_hit_mfe": 0.47,
      "ml_score": 82.3
    }
  ],
  "tier": "10_30",
  "elapsed_ms": 145
}
```

**Fields:**
- `pred_return` -- predicted close-to-close return (%)
- `pred_mfe` -- predicted max favorable excursion (%), always >= 0
- `win_prob` -- P(actual_return > 0), calibrated from walk-forward data
- `p_hit_return` -- P(actual_return >= pred_return)
- `p_hit_mfe` -- P(mfe >= pred_mfe)
- `ml_score` -- 0-100 percentile rank (higher = stronger prediction)

**VIX blocking:** If current VIX > 35, the response includes `"vix_blocked": true` and no scores. During market panics, seasonal patterns break down.

### GET /health
Returns service status and loaded tiers.

### GET /tiers
Returns list of available tier names.

---

## How It Works Internally

### Scoring Pipeline (per opportunity)

1. **Feature Engine** (`feature_engine.py`) computes 59 features:
   - Loads symbol's price CSV (searches US/ -> ETF/ -> INDX/ directories)
   - Loads opportunity data from parquet cache (or falls back to gzip files)
   - Computes pattern features from depth profile (22 features)
   - Computes technical indicators from price data (5 features)
   - Looks up precomputed market regime (VIX, yields, credit, breadth -- 16 features)
   - Computes SPX seasonal regime from historical SPX data (4 features)
   - Computes context (52-week range -- 2 features)
   - Computes calendar features (5 features)
   - Computes interaction features (4 features)

2. **Model Ensemble** (`scorer.py`) predicts:
   - Runs features through 3 SR models (LGB + XGB + CatBoost), averages predictions -> pred_return
   - Runs features through 3 MFE models, averages -> pred_mfe
   - Looks up pred_return in SR calibration table (20 bins) -> win_prob, p_hit_return
   - Looks up pred_mfe in MFE calibration table -> p_hit_mfe
   - Computes ml_score as percentile rank from calibration bins

### Feature Validation

At startup, the scorer validates that every loaded model's feature count matches the configured FEATURE_COLS (59). This catches mismatches between model files and config after retraining. If validation fails, the service refuses to start.

---

## The 59 Features

Defined in `scorer_config.py` as `FEATURE_COLS`. All tiers and both SR/MFE models use the identical feature set.

**CRITICAL: pat_daysOut MUST always be included.** A pattern is defined by [start_date, ticker, days_out, history_years]. Without pat_daysOut, the model cannot distinguish 10-day from 30-day holds.

| Group | Count | Examples |
|-------|-------|---------|
| Pattern-Intrinsic | 22 | pat_deepest_pass, pat_sharpe_ratio, pat_direction, pat_daysOut, pat_neighbor_avg_wr, pat_sharpness, pat_consistency_std, pat_concurrent_count, pat_hit_last_year |
| Technical | 5 | ta_trend_long, ta_price_vs_sma200, ta_sma50_vs_sma200, ta_trend_direction_match, ta_rvol_20 |
| Market Regime | 16 | mkt_vix_level, mkt_vix_percentile_60d, mkt_vix_5d_change, mkt_vix_term_structure, mkt_yield_curve_10y2y, mkt_yield_curve_slope, mkt_credit_spread, mkt_credit_spread_change_20d, mkt_spy_roc_20, mkt_spy_above_sma200, mkt_adv_decl_ratio_10d, mkt_sector_rotation, mkt_vix_regime_bucket, mkt_breadth_momentum, mkt_fed_rate_level, mkt_fed_rate_direction |
| SPX Seasonal | 4 | mkt_spx_seasonal_wr, mkt_spx_seasonal_ret, mkt_spx_seasonal_regime, mkt_spx_dir_alignment |
| Context | 2 | ctx_pct_from_52w_high, ctx_pct_from_52w_low |
| Calendar | 5 | cal_month, cal_day_of_year, cal_week_of_month, cal_is_opex_week, cal_pe_year |
| Interactions | 4 | pat_dir_x_mkt_trend, pat_dir_x_sector_trend, pat_depth_x_vix, pat_quality_x_regime |

---

## Model Files

18 files in `models/`, naming convention: `v2_{algorithm}[_{tier}][_mfe]_{date}.{ext}`

| Algorithm | Extension | Loader | Typical Size |
|-----------|-----------|--------|-------------|
| LightGBM | `.txt` | `lgb.Booster(model_file=...)` | 50-200 KB |
| XGBoost | `.json` | `xgb.Booster(); m.load_model(...)` | 100-500 KB |
| CatBoost | `.cbm` | `CatBoostRegressor().load_model(...)` | 50-800 KB |

Models are intentionally shallow (3-750 boosting iterations). The seasonal signal is weak -- deeper models overfit. Model files are tiny because they store split rules, not data.

Tier suffix: no suffix = 10_30, `_31_60`, `_61_90`.
Target suffix: no suffix = SR (actual_return), `_mfe` = MFE (max favorable excursion).

---

## Calibration Files

6 JSON files in `calibration/`. Each contains 20 quantile bins mapping predicted return ranges to empirical probabilities, built from walk-forward validation predictions (no data leakage).

Structure:
```json
{
  "target": "sr",
  "n_samples": 11004843,
  "n_bins": 20,
  "bins": [
    {
      "bin": 0,
      "pred_min": -7.61,
      "pred_max": -0.51,
      "win_prob": 0.529,
      "p_hit": 0.601,
      "count": 550242
    },
    ...
  ]
}
```

- `win_prob`: P(actual_return > 0) for samples in this prediction bin
- `p_hit`: P(actual >= predicted) for SR, P(mfe >= predicted) for MFE

---

## ETF Support

The model was trained on S&P 500 stocks only. Adding ETFs to training data was tested and hurt model quality (ETFs have different seasonal dynamics). However, the stocks-only model generalizes well to ETFs at inference time.

The feature engine automatically resolves ETFs:
- Price CSVs: searches `csv/US/` first, then `csv/ETF/`
- Opportunity files: searches `sp500/opp_by_symbol/` first, then `ETF/opp_by_symbol/`
- Sector lookup: falls back from TICKER_SECTOR (stocks) to ETF_SECTOR (157 ETFs) to ETF_CATEGORY_SECTOR_ETF (category -> proxy ETF)

No special configuration needed. Just POST the ETF symbol to /score like any stock.

---

## Config Reference (scorer_config.py)

ML-specific configuration is in `scorer_config.py`. Paths are imported from the central `/home/flask/config.py` on Linux; on Windows dev they fall back to local defaults. No environment variables are used. **Never name any file in this directory `config.py`.**

| Setting | Value | Description |
|---------|-------|-------------|
| `DATA_DIR` | from central config | `/home/flask/config.py:ddir` on Linux, `C:/seasonals/data` on Windows fallback |
| `HOST` | `0.0.0.0` | Flask bind address |
| `PORT` | `5090` | Flask port |
| `VIX_CUTOFF` | `35` | Refuse to score when VIX exceeds this |
| `MAX_DEPTH_CAP` | `35` | Cap for depth_utilization denominator |
| `SPX_SEASONAL_FORWARD_DAYS` | `15` | Trading days for SPX seasonal return calc |
| `TIERS` | 3 tiers | Maps tier name to model/calibration file paths |
| `FEATURE_COLS` | 59 features | Feature list (must match trained models) |
| `ML_PARQUET_MARKETS` | 6 markets | Markets for daily parquet generation |
| `TICKER_SECTOR` | from config_ml.py | 475 stocks -> GICS sector mapping |
| `ETF_SECTOR` | from config_ml.py | 157 ETFs -> category mapping |
| `SECTOR_ETF` | 11 sectors | GICS sector -> SPDR ETF mapping |

All derived paths (CSV directories, opportunity directories, etc.) are computed from `DATA_DIR`, which comes from the central `/home/flask/config.py` on Linux servers.

---

## Dependencies

```
flask
lightgbm
xgboost
catboost
pandas
numpy
scikit-learn
pyarrow
```

Python 3.10+ recommended (tested on 3.12).

---

## Retraining

Model files are generated by the training pipeline in the parent directory. After retraining:

```bash
# From the root ml_scorer/ directory
cp models/v2_*.{txt,json,cbm} ml_scorer/models/
cp results/calibration_*.json ml_scorer/calibration/
```

Then restart the service. Feature validation at startup will catch any model/config mismatches.

Monthly retraining cadence recommended. Model files are versioned by date suffix (e.g., `_20260312`). Keep previous model files for rollback.
