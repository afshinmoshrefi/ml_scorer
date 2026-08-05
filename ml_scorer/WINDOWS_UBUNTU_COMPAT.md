# ML Scorer: Windows/Ubuntu Compatibility Issue

## The Problem

The ml_scorer package runs in two environments:
- **Windows 11** (32-core dev machine): training, retraining, walk-forward validation
- **Ubuntu Ubuntu** (dev, staging, prod servers): inference service (Flask on port 7675)

The codebase is the same `ml_scorer/` folder in both places, but the two environments have different project structures and config systems. This has already caused a production outage (March 2026) when the folder was copied from dev to production and a local `config.py` shadowed the central TradeWave config.

## Current Architecture

### Ubuntu servers (dev/staging/prod)

```
/home/flask/
  config.py              <-- CENTRAL config (paths, API keys, URLs, all apps use this)
  config_ml.py           <-- ML-specific data (TICKER_SECTOR, YEAR_COMBOS, PE_COMBOS)
  ml_scorer/
    scorer_config.py     <-- ML service config (tiers, features, models)
                             imports DATA_DIR from central config.py
    app.py               <-- Flask service
    feature_engine.py
    scorer.py
    daily_opp_selection.py
    opp_to_parquet.py
    opp_scorer.py
    models/              <-- 18 model files
    calibration/         <-- 6 calibration JSONs
  data/
    csv/US/              <-- price CSVs
    csv/ETF/
    csv/INDX/
    sp500/opp_by_symbol/ <-- opportunity gzips
    ETF/opp_by_symbol/
```

Key point: ALL TradeWave apps (appserver, blog, edgar, realtime, ml_scorer) import paths from `/home/flask/config.py`. This is the single source of truth. When deploying an app, you copy its folder (e.g. `ml_scorer/`) to the server. The central `config.py` is never copied because it already exists on each server with server-specific values.

### Windows 11 (training machine)

```
C:/seasonals/
  ml_scorer/
    scorer_config.py
    app.py
    feature_engine.py
    ... (same files)
    models/
    calibration/
  data/
    csv/US/
    csv/ETF/
    csv/INDX/
    sp500/opp_by_symbol/
    ETF/opp_by_symbol/
```

Key point: There is NO central `config.py` on Windows. There is no `/home/flask/` directory. The training scripts and ml_scorer live under `C:/seasonals/`. The `config_ml.py` with TICKER_SECTOR etc. may or may not be present.

## What Went Wrong (March 2026 Incident)

1. AI (Claude) created a `config.py` inside `ml_scorer/` during development, with its own `DATA_DIR` auto-detection
2. This file was named `config.py`, same as the central TradeWave config
3. When `ml_scorer/` was copied to production, the local `config.py` got deployed too
4. Python's import resolution found the local `config.py` before the central one
5. The ML scorer service started fine (`/health` returned OK) but the `/select` endpoint silently returned 0 results because paths were wrong
6. The daily AI pick SVG on the homepage disappeared with no error

## Current Fix (Partial)

- Renamed `ml_scorer/config.py` to `scorer_config.py` so it can never shadow the central config
- `scorer_config.py` now does: `from config import ddir` on Ubuntu (gets central paths), falls back to hardcoded `C:/seasonals/data` on Windows
- All imports in ml_scorer files updated from `config` to `scorer_config`

## What Still Needs to Be Solved

### 1. Training scripts on Windows need scorer_config.py too

The training pipeline (walk-forward validation, feature engineering, model training) runs on Windows and imports from the ml_scorer package. These training scripts need to be checked and updated to use `scorer_config` instead of `config`.

Find all training scripts that import from ml_scorer or from config and update them.

### 2. config_ml.py dependency

`scorer_config.py` imports `TICKER_SECTOR`, `YEAR_COMBOS`, `PE_COMBOS`, `ETF_SECTOR`, `ETF_CATEGORY_SECTOR_ETF` from `config_ml.py` which lives in the parent directory on Ubuntu (`/home/flask/config_ml.py`). On Windows, this file may be in a different location or not exist. The current fallback generates minimal defaults but those may not match production behavior during training.

The training machine needs a copy of `config_ml.py` or the data it contains. Decide where it should live on Windows and make the import path work for both platforms.

### 3. Deployment workflow needs to be documented and safe

Currently deployment is manual: copy files via scp. This is error-prone. The deployment process should:

- Copy only the ml_scorer/ folder (never config.py from outside it)
- Verify that `scorer_config.py` correctly imports from the central config after deployment
- Restart the service
- Verify `/health` AND `/select` both return valid responses (not just health)
- Restore/verify the parquet cron job exists: `0 1 * * * cd /home/flask/ml_scorer && python3 opp_to_parquet.py`

### 4. After retraining on Windows, what files go to Ubuntu?

After a retrain, these files need to be copied from Windows to Ubuntu production:
- `models/*.txt`, `models/*.json`, `models/*.cbm` (new model files)
- `calibration/*.json` (new calibration tables)
- Updated `scorer_config.py` ONLY if TIERS or FEATURE_COLS changed

The scorer_config.py is the tricky one. If the retrain adds a new tier or changes feature columns, the config must be updated on production. But it must NOT overwrite the path-import logic at the top of the file. Consider splitting scorer_config.py into:
- `scorer_paths.py` -- path resolution (central config import + Windows fallback), NEVER changes during retrain
- `scorer_config.py` -- tiers, features, model filenames, safe to overwrite during retrain

### 5. Validation on startup

The service should validate on startup that:
- DATA_DIR exists and contains expected subdirectories (csv/US, sp500/opp_by_symbol, etc.)
- At least one parquet file exists for the current week
- Model files referenced in TIERS actually exist in models/
- Feature count matches loaded models

This would have caught the March 2026 issue immediately instead of silently returning empty results.

## Design Constraints

- Central config lives at `/home/flask/config.py` on all Ubuntu servers. This is not changing.
- Windows training machine has no central config and no `/home/flask/` path. This is not changing.
- The user deploys by copying the `ml_scorer/` folder. Keep this simple.
- No environment variables. Hardcode paths with platform auto-detection.
- No `Path(__file__).parent` constructs. Use explicit hardcoded path strings.
- The ml_scorer/ folder must work identically on Windows (training) and Ubuntu (inference) without manual edits after copying.

## Goal

Make it so that:
1. Retraining on Windows works without touching path config
2. Copying `ml_scorer/` to any Ubuntu server works without touching path config
3. If something IS wrong (bad paths, missing data, missing models), the service fails loud on startup, not silently on first request
