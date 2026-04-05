# ML Scorer V3 Upgrade Guide

## Overview

The ML Pattern Scorer service has been upgraded from V1 to V3. This document captures everything an AI needs to understand when working with downstream consumers of the service.

**Production server:** `http://104.238.214.253:7675`

**No code changes required in any consumer.** All V3 changes are either internal to the scorer or additive to the API.

**Consumers:**
- `C:\seasonals\TradeWave_auto_trading\` -- no changes needed
- TradeWave appserver (4 ML display columns) -- no changes needed
- daily_ai_pick -- no changes needed

---

## What Did NOT Change

- Server URL and port (`http://104.238.214.253:7675`)
- `/score` request fields: `symbol`, `date`, `daysOut`, `direction`, `tier`
- `/score` response fields: `pred_return`, `pred_mfe`, `win_prob`, `p_hit_return`, `p_hit_mfe`, `ml_score`, `tiers_used`, `elapsed_ms`
- `/health` and `/tiers` endpoints
- All numeric field semantics and value ranges
- VIX blocking behavior: per-result in both V1 and V3, unchanged
- Support for single and batch requests to `/score`

---

## What Changed

### 1. Feature count: 59 -> 62

Three new commodity/FX regime features are computed internally: `mkt_dxy_roc_20`, `mkt_cl_roc_20`, `mkt_gc_roc_20`. The feature engine computes them automatically from data on the production server.

**Impact on callers:** None. Fully internal. No change required in any calling code.

---

### 2. `tier` is optional in `/score` requests

V3 auto-detects tier from `daysOut` when `tier` is not provided:

| `daysOut` | Auto-detected tier |
|-----------|-------------------|
| 1 - 30    | `10_30`           |
| 31 - 60   | `31_60`           |
| 61+       | `61_90`           |

Passing `tier` explicitly is still supported and still works. No change is required in existing callers.

---

### 3. `tier` field added to each `/score` result

Each result object now includes the tier that was used:

```json
{
  "symbol": "AAPL", "date": "2026-04-04", "daysOut": 20, "direction": "l",
  "tier": "10_30",
  "ml_score": 94.1, "win_prob": 0.83, "pred_return": 3.6, "pred_mfe": 8.2,
  "p_hit_return": 0.58, "p_hit_mfe": 0.47
}
```

V1 did not include `tier` in individual results. Additive only -- no existing code breaks.

---

### 4. New `/select` endpoint

V3 adds a `/select` endpoint that replaces the manual find-candidates + score pipeline with a single API call. It reads from a pre-built nightly parquet cache and returns pre-filtered, scored picks.

**Prerequisite:** Parquet cache files must exist at `{DATA_DIR}/{market_folder}/ml_cache_YYYY-MM-DD.parquet`, generated nightly by `opp_to_parquet.py`. If the cache is missing for the requested date, the endpoint returns an error.

**Request:**
```json
POST /select
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

**`resource_ids` values:**

| Value  | Universe      |
|--------|---------------|
| `"0"`  | DOW 30        |
| `"1"`  | NASDAQ 100    |
| `"2"`  | S&P 500       |
| `"3"`  | RUSSELL 1000  |
| `"4"`  | WILSHIRE 5000 |
| `"11"` | ETFs          |

**Response:**
```json
{
  "picks": [
    {
      "symbol": "AAPL", "date": "2026-04-04", "daysOut": 20, "direction": "l",
      "tier": "10_30", "ml_score": 94.1, "win_prob": 0.83,
      "pred_return": 3.6, "pred_mfe": 8.2,
      "p_hit_return": 0.58, "p_hit_mfe": 0.47
    }
  ],
  "candidates_after_prefilter": 120,
  "candidates_scored": 120,
  "elapsed_ms": 4500
}
```

**Optional future upgrade:** The `find_candidates` + `score_candidates` pipeline in `TradeWave_auto_trading/scorer.py` can be replaced with a single `/select` call once the nightly parquet cache generation (`opp_to_parquet.py`) is confirmed running on the production server.

---

## Validation

```bash
# Health -- confirm feature_count is 62 and all 3 tiers loaded
curl http://104.238.214.253:7675/health

# Single score (no tier -- tests auto-detection)
curl -X POST http://104.238.214.253:7675/score \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","date":"2026-04-04","daysOut":20,"direction":"l"}'

# Batch with mixed tiers
curl -X POST http://104.238.214.253:7675/score \
  -H "Content-Type: application/json" \
  -d '{"opportunities":[
    {"symbol":"AAPL","date":"2026-04-04","daysOut":20,"direction":"l"},
    {"symbol":"MSFT","date":"2026-04-04","daysOut":45,"direction":"l"},
    {"symbol":"SPY","date":"2026-04-04","daysOut":75,"direction":"l"}
  ]}'
```

Expected `/health` response:
```json
{
  "feature_count": 62,
  "status": "ok",
  "tiers": ["10_30", "31_60", "61_90"],
  "vix_cutoff": 35
}
```

---

## Model File Reference

18 model files in `ml_scorer/models/`, dated 2026-04-03 or 2026-04-04:

| Tier  | Target | LightGBM                          | XGBoost                          | CatBoost                           |
|-------|--------|-----------------------------------|----------------------------------|------------------------------------|
| 10_30 | SR     | v2_lgb_20260403.txt               | v2_xgb_20260403.json             | v2_catboost_20260403.cbm           |
| 10_30 | MFE    | v2_lgb_mfe_20260403.txt           | v2_xgb_mfe_20260403.json         | v2_catboost_mfe_20260403.cbm       |
| 31_60 | SR     | v2_lgb_31_60_20260403.txt         | v2_xgb_31_60_20260403.json       | v2_catboost_31_60_20260403.cbm     |
| 31_60 | MFE    | v2_lgb_31_60_mfe_20260404.txt     | v2_xgb_31_60_mfe_20260404.json   | v2_catboost_31_60_mfe_20260404.cbm |
| 61_90 | SR     | v2_lgb_61_90_20260404.txt         | v2_xgb_61_90_20260404.json       | v2_catboost_61_90_20260404.cbm     |
| 61_90 | MFE    | v2_lgb_61_90_mfe_20260404.txt     | v2_xgb_61_90_mfe_20260404.json   | v2_catboost_61_90_mfe_20260404.cbm |
