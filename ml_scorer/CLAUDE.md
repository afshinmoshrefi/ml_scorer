# ML Pattern Scorer V3 -- Production Service

## What This Is

Self-contained Flask service that scores TradeWave seasonal stock/ETF pattern opportunities. Takes a symbol, date, daysOut, and direction, computes 62 features from live price data, runs them through a 3-model ensemble (LightGBM + XGBoost + CatBoost), and returns calibrated predictions.

**Two models per tier:**
- **SR model**: predicts actual_return (close-to-close return %). Used for win probability.
- **MFE model**: predicts mfe_return (max favorable excursion %). Used for profit target sizing.

**Three tiers:** 10_30 (10-30 day holds), 31_60 (31-60 days), 61_90 (61-90 days).

---

## Folder Structure

```
ml_scorer/
  app.py                    # Flask app: POST /score, POST /select, GET /health, GET /tiers
  config.py                 # Configuration: paths, tiers, 62 feature columns, ML_PARQUET_MARKETS
  feature_engine.py         # Computes all 62 features for a single opportunity
  scorer.py                 # ModelEnsemble: loads models, averages predictions, calibrates
  daily_opp_selection.py    # /select pipeline: parquet load, pre-filter, score, rank
  opp_to_parquet.py         # Nightly: build ml_cache_YYYY-MM-DD.parquet per market
  nightly.sh                # Cron script: runs opp_to_parquet.py for all markets
  warmup_cache.py           # Warms up price data cache after service restart
  requirements.txt          # pip install -r requirements.txt
  CLAUDE.md                 # This file
  models/                   # 18 model files (3 algos x 2 targets x 3 tiers)
  calibration/              # 6 calibration JSONs (20-bin empirical lookup tables)
```

---

## Deployment

### Quick Start

```bash
# 1. Copy this entire ml_scorer/ folder to the production machine
scp -r ml_scorer/ flask@104.238.214.253:/home/flask/ml_scorer/

# 2. Install dependencies
cd /home/flask/ml_scorer
pip install -r requirements.txt

# 3. Restart the systemd service
systemctl restart ml_scorer
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ML_SCORER_DATA_DIR` | **Yes** | `C:/seasonals/data` | Base path for all price/opp data |
| `ML_SCORER_HOST` | No | `0.0.0.0` | Flask bind address |
| `ML_SCORER_PORT` | No | `5090` | Flask port |
| `ML_SCORER_EARNINGS_DIR` | No | `{DATA_DIR}/../edgar/earnings` | Earnings date JSONs |

### Production Server

- Host: prodkeyprovider at 104.238.214.253, port 7675 (nginx reverse proxy)
- Ubuntu 20.04, Python 3.12, 4GB RAM, 2GB swap
- systemd service: `ml_scorer` -- `systemctl restart ml_scorer`
- gunicorn: **1 worker**, 300s timeout, Unix socket
- Working dir: `/home/flask/ml_scorer`
- Data dir: `ML_SCORER_DATA_DIR=/home/flask/data`

**CRITICAL: Run with 1 gunicorn worker only.** 4GB RAM cannot support 2 workers when price CSVs are loaded into memory per worker. Memory fix (2026-04-04): gzip fallback now uses targeted 5-date scan instead of full history load.

### Required Data on the Machine

```
{ML_SCORER_DATA_DIR}/
  csv/US/                    # Stock price CSVs (OHLCV, ~3500 files, back to 1981+)
  csv/ETF/                   # ETF CSVs (SPY, QQQ, sector ETFs, VIX-related, etc.)
  csv/INDX/                  # Index CSVs (VIX, VIX3M, US10Y, US2Y, IRX, ADVN, DECN, SPX, DXY)
  csv/COMM/                  # Commodity CSVs (CL crude oil, GC gold)
  sp500/opp_by_symbol/       # 475 dirs (one per S&P stock), 116 gzip CSVs each
  ETF/opp_by_symbol/         # 157 dirs (for ETF inference)
  sp500_symbols.csv          # S&P 500 ticker list
  sp500/ml_cache_YYYY-MM-DD.parquet    # Nightly parquet cache (optional, speeds up scoring)
```

---

## API

### POST /score

Score one or multiple opportunities. `tier` is optional -- auto-detected from `daysOut` if omitted.

**Single request:**
```json
{
  "symbol": "AAPL",
  "date": "2026-04-04",
  "daysOut": 20,
  "direction": "l"
}
```

**Batch request:**
```json
{
  "opportunities": [
    {"symbol": "AAPL", "date": "2026-04-04", "daysOut": 20, "direction": "l"},
    {"symbol": "XLE",  "date": "2026-04-04", "daysOut": 45, "direction": "l"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "symbol": "AAPL",
      "date": "2026-04-04",
      "daysOut": 20,
      "direction": "l",
      "tier": "10_30",
      "pred_return": 2.34,
      "pred_mfe": 5.12,
      "win_prob": 0.78,
      "p_hit_return": 0.58,
      "p_hit_mfe": 0.47,
      "ml_score": 82.3
    }
  ],
  "tiers_used": ["10_30"],
  "elapsed_ms": 145
}
```

**VIX blocking:** Per-opportunity. If VIX > 35, that symbol's result contains `"vix_blocked": true` and no scores. Other symbols in the same batch are still scored.

**Fields:**
- `pred_return` -- predicted close-to-close return (%)
- `pred_mfe` -- predicted max favorable excursion (%), always >= 0
- `win_prob` -- P(actual_return > 0), calibrated from walk-forward data
- `p_hit_return` -- P(actual_return >= pred_return)
- `p_hit_mfe` -- P(mfe >= pred_mfe)
- `ml_score` -- 0-100 percentile rank (higher = stronger prediction)
- `tier` -- which tier was used for this result

### POST /select

Find and score today's best opportunities from the nightly parquet cache.

**Request:**
```json
{
  "date": "2026-04-04",
  "resource_ids": ["2"],
  "num_picks": 10,
  "direction": "l",
  "days_out_min": 10,
  "days_out_max": 30,
  "min_avg_return": 3.0,
  "min_win_prob": 0.70,
  "exclude_symbols": ["AAPL"]
}
```

`resource_ids`: `"0"` DOW 30, `"1"` NASDAQ 100, `"2"` S&P 500, `"3"` RUSSELL 1000, `"4"` WILSHIRE 5000, `"11"` ETFs.

Requires nightly parquet cache. Returns error if parquet missing for the requested date.

### GET /health
Returns `status`, `tiers`, `feature_count` (62), `uptime_seconds`, `vix_cutoff`.

### GET /tiers
Returns list of available tier names.

---

## How It Works Internally

### Scoring Pipeline (per opportunity)

1. **Feature Engine** (`feature_engine.py`) computes 62 features:
   - Loads symbol's price CSV (searches US/ -> ETF/ -> INDX/ -> COMM/)
   - Loads opportunity data: parquet fast path first, targeted gzip scan fallback (5 dates only)
   - Pattern features from depth profile (23 features)
   - Technical indicators from price data (5 features)
   - Market regime indicators (VIX, yields, credit, breadth, DXY, CL, GC -- 19 features)
   - SPX seasonal regime (4 features)
   - Context: 52-week range (2 features)
   - Calendar features (5 features)
   - Interaction features (4 features)

2. **Model Ensemble** (`scorer.py`) predicts:
   - Runs through 3 SR models (LGB + XGB + CatBoost), averages -> pred_return
   - Runs through 3 MFE models, averages -> pred_mfe
   - Calibration lookup (20 bins) -> win_prob, p_hit_return, p_hit_mfe, ml_score

### Memory Architecture

**Caches (per gunicorn worker lifetime):**
- `_price_cache`: symbol -> DataFrame. Market symbols loaded eagerly; user symbols lazy. Grows with unique symbols scored -- largest memory consumer.
- `_opp_cache`: (symbol, date_str) -> combos dict. Targeted scan, 5 dates only. Small per entry.
- `_parquet_by_symbol`: symbol -> DataFrame. Reset on date change. One day's data only.
- `_market_cache`: regime indicators by date. Grows slowly.
- `_ta_cache`: technical indicators by (symbol, date).

### Feature Validation

At startup, validates every model's feature count matches FEATURE_COLS (62). Service refuses to start on mismatch.

---

## The 62 Features

Defined in `config.py` as `FEATURE_COLS`. All tiers and both SR/MFE models use the identical feature set.

**CRITICAL: pat_daysOut MUST always be included.** Without it the model cannot distinguish 10-day from 30-day holds.

| Group | Count | Examples |
|-------|-------|---------|
| Pattern-Intrinsic | 23 | pat_deepest_pass, pat_sharpe_ratio, pat_direction, pat_daysOut, pat_neighbor_avg_wr, pat_sharpness, pat_consistency_std, pat_concurrent_count, pat_hit_last_year |
| Technical | 5 | ta_trend_long, ta_price_vs_sma200, ta_sma50_vs_sma200, ta_trend_direction_match, ta_rvol_20 |
| Market Regime | 19 | mkt_vix_level, mkt_yield_curve_10y2y, mkt_credit_spread, mkt_spy_roc_20, mkt_fed_rate_level, mkt_dxy_roc_20, mkt_cl_roc_20, mkt_gc_roc_20 |
| SPX Seasonal | 4 | mkt_spx_seasonal_wr, mkt_spx_seasonal_ret, mkt_spx_seasonal_regime, mkt_spx_dir_alignment |
| Context | 2 | ctx_pct_from_52w_high, ctx_pct_from_52w_low |
| Calendar | 5 | cal_month, cal_day_of_year, cal_week_of_month, cal_is_opex_week, cal_pe_year |
| Interactions | 4 | pat_dir_x_mkt_trend, pat_dir_x_sector_trend, pat_depth_x_vix, pat_quality_x_regime |

---

## Model Files

18 files in `models/`, naming: `v2_{algorithm}[_{tier}][_mfe]_{date}.{ext}`

| Tier | Target | LightGBM | XGBoost | CatBoost |
|------|--------|----------|---------|----------|
| 10_30 | SR | v2_lgb_20260403.txt | v2_xgb_20260403.json | v2_catboost_20260403.cbm |
| 10_30 | MFE | v2_lgb_mfe_20260403.txt | v2_xgb_mfe_20260403.json | v2_catboost_mfe_20260403.cbm |
| 31_60 | SR | v2_lgb_31_60_20260403.txt | v2_xgb_31_60_20260403.json | v2_catboost_31_60_20260403.cbm |
| 31_60 | MFE | v2_lgb_31_60_mfe_20260404.txt | v2_xgb_31_60_mfe_20260404.json | v2_catboost_31_60_mfe_20260404.cbm |
| 61_90 | SR | v2_lgb_61_90_20260404.txt | v2_xgb_61_90_20260404.json | v2_catboost_61_90_20260404.cbm |
| 61_90 | MFE | v2_lgb_61_90_mfe_20260404.txt | v2_xgb_61_90_mfe_20260404.json | v2_catboost_61_90_mfe_20260404.cbm |

---

## Calibration Files

6 JSON files in `calibration/`. 20 quantile bins mapping predicted return -> empirical probabilities. Built from walk-forward predictions (no data leakage).

---

## Retraining

After retraining:
```bash
cp models/v2_*.{txt,json,cbm} ml_scorer/models/
cp results/calibration_*.json ml_scorer/calibration/
# Update TIERS in ml_scorer/config.py with new model file dates
systemctl restart ml_scorer
```
