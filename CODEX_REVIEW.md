# ML Pattern Scorer V2 -- Codex Architecture & Code Review (Round 4)

## Purpose of This Document

This document provides Codex with complete context to perform a thorough architecture and code review of the ML Pattern Scorer V2 production service. Your job is to identify bugs, correctness issues, design problems, thread-safety risks, data integrity issues, and anything that could cause silent failures in a live trading system.

**Return your findings as a prioritized list: Critical > High > Medium > Low.**
For each finding include: file, line(s), description of the problem, and a concrete suggested fix.

This is Round 4. Round 1, Round 2, and Round 3 findings have been fixed (see sections below). Do not re-report them. Focus on new issues and areas not covered in prior rounds.

---

## Round 1 Fixes Applied (Do Not Re-Report)

All 7 Round 1 findings have been resolved. The following changes were made:

### feature_engine.py

**Fix 1 -- `_opp_cache` date-scope corruption (was Critical)**
- `_load_opp_files` no longer stores parquet-backed results in `_opp_cache`. Parquet results are returned directly without caching. The parquet file itself remains cached in `_parquet_cache` (keyed by file path, shared across all symbols).
- `_opp_cache` is now exclusively used for gzip-backed data (full history, date-safe).
- `_parquet_cache` moved from lazy `hasattr` initialization to `__init__`.

**Fix 2 -- `_load_opp_from_parquet` returns None when symbol missing (was part of High #4)**
- Changed `return {}` to `return None` when `len(sym_data) == 0`.
- `_load_opp_files` uses `if combos:` (falsy for both None and empty dict) to decide whether to fall through to gzip.

**Fix 3 -- ETF/INDX gzip fallback (was High #4)**
- `_load_opp_from_gzip` now searches `ETF`, `ETF_`, `INDX_COMMON`, `INDX_COMMON_` subdirectories in order after sp500 fails.

**Fix 4 -- `is not np.nan` identity bug (was Medium #5)**
- `ratio_now is not np.nan` replaced with `not np.isnan(ratio_now)` (and same for `ratio_then`) in `mkt_breadth_momentum` computation.

### app.py

**Fix 5 -- `/score` tier auto-detection (was High #3)**
- Added `_tier_for_days_out(days_out)` helper: `<=30` -> `10_30`, `<=60` -> `31_60`, else `61_90`.
- `/score` now resolves tier per opportunity in this order: (1) per-opportunity `tier` field, (2) top-level request `tier` default, (3) auto-detected from `daysOut`.
- Logs a WARNING when explicit tier and daysOut are inconsistent.
- Response field changed: `"tier": "10_30"` (single string) -> `"tiers_used": ["10_30", ...]` (array of all tiers used in the batch).

### scorer.py

**Fix 6 -- Missing model/calibration files silent failure (was High #2)**
- `_load_ensemble` now raises `RuntimeError` immediately if any model file is missing. No more silent skip.
- `_load_calibration` now raises `RuntimeError` if the calibration file is missing.
- Service startup will hard-fail rather than silently return `pred=0.0` / `win_prob=0.5`.

### daily_opp_selection.py

**Fix 7 -- Stale docstring (was Low #7)**
- `resource_ids` docstring example updated from `['2', '11']` to `['sp500', 'etf', 'indx']`.

---

## Round 2 Fixes Applied (Do Not Re-Report)

Three Round 2 issues were confirmed and fixed. The remaining 11 Round 2 items were either verified as non-issues, intentional design, or deferred. See "Areas to Investigate in Round 3" for the open items carried forward.

### config.py

**Fix 8 -- Deduplicate tier boundary logic**
- `tier_for_days_out(days_out)` added to `config.py` as the single source of truth for tier boundaries (`<=30` -> `10_30`, `<=60` -> `31_60`, `>60` -> `61_90`).
- Both `app.py` and `daily_opp_selection.py` now import and call it. No more inline copies.

### app.py + daily_opp_selection.py

**Fix 9 -- VIX NaN bypass (was Round 2 #12)**
- The check `if vix is not None and vix > VIX_CUTOFF` did not handle `np.nan`. `np.nan is not None` is True, but `np.nan > 35` is False, so the VIX block was silently bypassed when VIX data was unavailable.
- Fixed to an explicit three-way check using `vix != vix` as the NaN test (NaN is the only value not equal to itself):
  - `vix is None or vix != vix`: logs a WARNING and proceeds (can't enforce cutoff without data)
  - `vix > VIX_CUTOFF`: blocks with error response as before
  - else: proceeds normally
- Same fix applied in both `app.py` and `daily_opp_selection.py`.

### Round 2 Items Verified or Deferred (Do Not Re-Report)

- **R2 #4 (API breaking change `tier` -> `tiers_used`)**: Audited all callers in this repo. `warmup_cache.py` does not read any field from the `/score` response. No other internal callers. External consumers (TradeWave platform, auto-trading system) are in separate repos and will be updated when those systems are built.
- **R2 #3 (empty `{}` cached permanently from gzip)**: Intentional. Nightly `systemctl restart` clears in-memory caches. Acceptable behavior.
- **R2 #13 (no date input sanitization)**: Malformed dates raise `ValueError` caught by the outer `try/except`, returning a well-formed error response. Far-future dates producing NaN scores are a known edge case, not a priority.
- **R2 #14 (SPX seasonal latency spike at year rollover)**: Acknowledged operational behavior, not a bug.
- **R2 #10 (ML_PARQUET_MARKETS no startup log)**: Low priority, deferred.

---

## Round 3 Fixes Applied (Do Not Re-Report)

All 6 Round 3 findings have been resolved.

### app.py

**Fix 10 -- Regression: `_tier_for_days_out` NameError on mismatch warning path (was Round 3 High #1)**
- Line 177 (mismatch warning check) still called `_tier_for_days_out(days_out)` (the old local helper name) after Round 2 renamed it to the imported `tier_for_days_out`. Any `/score` request with an explicit `tier` field would raise `NameError` before entering the `try:` block, returning a 500 for the whole request.
- Fixed: `_tier_for_days_out(days_out)` -> `tier_for_days_out(days_out)` at line 177.

**Fix 11 -- `daysOut=0` or negative silently scored (was Round 3 Medium #6)**
- `int(days_out)` cast was followed by no validation. Values <= 0 were passed into feature generation and `tier_for_days_out`, which raises `ValueError` inside the `try:` block (caught and returned as error). However the error message was opaque and the validation belongs at the API boundary, not inside `tier_for_days_out`.
- Fixed: explicit `if days_out <= 0: append error, continue` guard added in `/score` before tier resolution.

### scorer.py

**Fix 12 -- Non-finite model prediction silently produces `ml_score=100` (was Round 3 Medium #5)**
- If any model's `predict()` returns `NaN` or `Inf` (rather than raising), `_predict_ensemble` returns that value, `_calibrate` falls through all bin comparisons to the last bin, and `_percentile_score` returns `100.0`. This produces a contradictory `win_prob` from the last calibration bin paired with `ml_score=100`.
- Fixed: `math.isfinite` guard added in `predict()` after `_predict_ensemble`. Raises `RuntimeError` with the non-finite values if either `pred_sr` or `pred_mfe` is non-finite.

### opp_to_parquet.py

**Fix 13 -- Silent schema drift drops entire date files (was Round 3 High #3)**
- `read_date_file()` subseted to available columns then assumed base columns were present when backfilling `avg_profit2`/`sharpe_ratio2`. Failures were only logged at `debug` level, invisible in normal nightly logs.
- Fixed: `REQUIRED_COLS = ['LorS', 'date', 'daysOut', 'sym', 'sharpe_ratio', 'avg_profit', 'median_profit']` added. `read_date_file()` validates required columns before processing, logs missing columns at `warning` level, and returns `None`. All other exceptions also upgraded from `debug` to `warning`.

### warmup_cache.py

**Fix 14 -- Warmup counted VIX-blocked and scoring-error results as successes (was Round 3 Medium #4)**
- `warmup()` only counted HTTP-level failures (exceptions) as `failed`. Any `/score` response with `"error"` in the result (including `vix_blocked`) was counted as `success += 1`.
- Fixed: inspects `result.get('results', [{}])[0]` for `error`/`vix_blocked` fields. Three outcome categories: `success` (no error), `blocked` (vix_blocked), `failed` (other error). Summary log now shows all three counts.

### feature_engine.py

**Fix 15 -- ETF parquet fast path never used in practice (was Round 3 High #2)**
- `_load_opp_from_parquet()` always built the path from `OPP_BY_SYMBOL_DIR` (hardcoded to the sp500 directory). If the sp500 parquet existed but didn't contain the ETF symbol, the function returned `None` immediately. The multi-market fallback only ran when the sp500 parquet file itself was absent -- which never happens on production after the nightly cron runs.
- Fixed: `_load_opp_from_parquet()` now iterates over all `ML_PARQUET_MARKETS` in config order (sp500 -> etf -> indx), building each market's `ml_cache_{date}.parquet` path. Returns combos from the first market whose parquet contains the symbol; `continue`s to the next market if the symbol is absent; returns `None` if symbol not found in any market's parquet.

### Round 3 Items Verified Not-Findings (Do Not Re-Report)

- **R3 Brief #6 (partial preds -> `np.mean([])` -> nan -> ml_score=100)**: Not the actual failure path. If a model's `predict()` raises, the exception propagates immediately out of the loop. The actual risk was non-finite model outputs, addressed by Fix 12.
- **R3 Brief #7 (compute_features returns without validating critical feature presence)**: The model handles NaN natively. VIX is separately validated in `app.py` before scoring. Minimal-data scoring producing a plausible score is acceptable behavior -- the caller can inspect individual feature values. Not flagged as a bug.
- **R3 Brief #8 `tier_for_days_out` guard for <= 0**: Now enforced at the API boundary (Fix 11). `tier_for_days_out` in config.py also raises `ValueError` for non-positive input as a belt-and-suspenders defense.

---

## System Overview

This is a Flask-based ML scoring service deployed on a Linux production server (Ubuntu 20.04, Python 3.12, gunicorn). It scores seasonal stock pattern opportunities using a 3-model ensemble (LightGBM + XGBoost + CatBoost) and returns calibrated win probabilities. The system is used to make real trading decisions on live capital.

### Request Flow

```
POST /score or POST /select
  -> app.py (Flask route)
  -> feature_engine.py (59 features computed from price/opp data)
  -> scorer.py (3-model ensemble, calibration lookup)
  -> JSON response
```

### Nightly Cron Flow

```
nightly.sh (1am ET)
  -> opp_to_parquet.py  (read opportunity CSVs, write ml_cache_YYYY-MM-DD.parquet per market)
  -> systemctl restart ml_scorer
  -> warmup_cache.py    (hit /score for all 475 symbols to pre-populate caches)
```

---

## Files to Review

All files are in `ml_scorer/` (the production service package):

| File | Role |
|------|------|
| `app.py` | Flask app, endpoints, startup |
| `config.py` | All configuration, paths, feature lists |
| `feature_engine.py` | Computes all 59 features per opportunity |
| `scorer.py` | Model loading, ensemble predict, calibration |
| `daily_opp_selection.py` | `/select` endpoint backend: load parquet, filter, score, rank |
| `opp_to_parquet.py` | Nightly: converts opportunity CSVs to parquet cache |
| `warmup_cache.py` | Post-restart cache warmup via HTTP calls to /score |
| `nightly.sh` | Orchestrates nightly cron: parquet -> restart -> warmup |

---

## Current Architecture (post Round 1 fixes)

### Opportunity Data Loading -- Two-Cache Design

`_load_opp_files(symbol, date_hint)` now uses two separate caches with different lifetimes:

```
_parquet_cache: cache_path -> full DataFrame
  - Keyed by file path (e.g., /home/flask/data/sp500/ml_cache_2026-04-03.parquet)
  - Loaded once per parquet file per service lifetime
  - All symbols share the same in-memory DataFrame

_opp_cache: symbol -> {combo -> {(date, daysOut, LorS) -> row_dict}}
  - Keyed by symbol only (gzip-backed data exclusively)
  - Gzip data is full history, so symbol-keyed cache is safe
  - Parquet results are NOT stored here
```

Decision flow:
1. If parquet exists and contains the symbol: filter from `_parquet_cache[path]`, return directly (no caching in `_opp_cache`)
2. If parquet not found OR symbol not in parquet: load from gzip, cache result in `_opp_cache[symbol]`

### Tier Resolution in `/score`

Per-opportunity, highest precedence first:
1. `tier` field on the individual opportunity object
2. Top-level `tier` field on the request
3. Auto-detected from `daysOut` via `tier_for_days_out()` (imported from `config.py`)

Tier boundaries (single source of truth in `config.tier_for_days_out`): `<=30` -> `10_30`, `<=60` -> `31_60`, `>60` -> `61_90`.

### Startup Failure Behavior

If any model file or calibration file is missing, `ModelEnsemble.__init__` raises `RuntimeError`. The service refuses to start.

---

## Known Design Decisions (Do NOT Flag These)

1. **Intentionally shallow models** (20-150 iterations per algo). The seasonal signal is weak; deep models overfit.
2. **Global mutable state** (`engine`, `scorer_mgr` in `app.py`). Write-once-per-symbol per service lifetime.
3. **`init_service()` at module level** in `app.py`. Causes models to load at gunicorn import time. Intentional.
4. **`warnings.filterwarnings('ignore')`** in feature_engine.py. Suppresses pandas/numpy warnings.
5. **VIX > 35 hard block**. Intentional -- models not trained on panic market data.
6. **NaN features passed to models**. Tree models handle NaN natively.
7. **`FEATURE_COLS_MFE = FEATURE_COLS`** alias for backward compatibility only. Both targets use 59 features.
8. **`opp_to_parquet.py` reads `opportunities/` directory**. This is a different directory structure than `opp_by_symbol/`. Both represent the same data in different organizations maintained by the TradeWave platform.
9. **Parquet results not cached in `_opp_cache`**. Intentional (Round 1 fix). The parquet file is cached in `_parquet_cache`; per-symbol filtering is done fresh each call.
10. **`/score` response uses `tiers_used` array** instead of single `tier` string. Intentional (Round 1 fix) to support mixed-horizon batch requests.
11. **Missing model files are fatal at startup**. Intentional (Round 1 fix). Degraded-mode scoring with 0 models was worse than refusing to start.
12. **`tier_for_days_out()` in `config.py` is the single source of truth** for tier boundaries. Both `app.py` and `daily_opp_selection.py` import it. (Round 2 fix.)
13. **VIX NaN logs a warning and proceeds**. When VIX CSV data is unavailable, the VIX > 35 block cannot be enforced. The system logs a warning and scores the opportunity rather than refusing. This is a deliberate trade-off: hard-blocking on missing VIX data would prevent all scoring during a data outage, which is worse than proceeding with a warning. (Round 2 fix.)
14. **`vix != vix` as NaN test**. Used in `app.py` and `daily_opp_selection.py` VIX checks. NaN is the only float where `x != x` is True. Avoids importing math/numpy just for `isnan`. (Round 2 fix.)
15. **`daysOut <= 0` rejected at API boundary**. `/score` now returns a 400-style error per-opportunity (not HTTP 400 but a well-formed error in the results array). `tier_for_days_out()` also raises `ValueError` for non-positive input as a second layer of defense. (Round 3 fix.)
16. **`math.isfinite` guard in `scorer.predict()`**. If the model ensemble returns NaN or Inf (not by raising), the guard raises `RuntimeError` before the result reaches calibration. (Round 3 fix.)
17. **`REQUIRED_COLS` validation in `opp_to_parquet.read_date_file()`**. Required columns are checked at read time and logged at WARNING level if missing. Schema drift now fails loudly. (Round 3 fix.)
18. **`_load_opp_from_parquet` iterates all markets**. Iterates `ML_PARQUET_MARKETS` in order, returns combos from the first market whose parquet contains the symbol. ETF and INDX symbols now use their own parquets. (Round 3 fix.)
19. **Warmup counts `vix_blocked` and `error` results separately**. `warmup()` inspects the per-opportunity result object. `success` only increments when no `error` field is present. (Round 3 fix.)

---

## Areas to Investigate in Round 4

Perform a fresh read of all files with fresh eyes. Focus on areas not previously covered.

### 1. `_load_opp_from_gzip`: PE Combo Directory Name Parsing

`find_combo_dirs` in `opp_to_parquet.py` produces `combo_suffix` by stripping `Monthly_Opp_{Month}_`:
```python
suffix = name[len(prefix):]   # e.g., '10_8' or '10_8_PE2'
```

`_parse_combo` (or equivalent inline code) in `feature_engine.py` then splits on `_` and casts the first two parts to `int`. Verify the actual on-disk directory name format for PE combos (e.g., is it `Monthly_Opp_March_4_4_PE2`?). If any directory name produces a non-numeric `parts[0]`, `int(parts[0])` raises `ValueError`, which is silently swallowed by `except Exception: continue` in `_load_opp_from_gzip`. This would silently drop entire combo files. Confirm whether this is handled correctly or needs a log.

### 2. INDX Directory Name Verification

`_load_opp_from_gzip` searches `INDX_COMMON` and `INDX_COMMON_` as fallback directories. `ML_PARQUET_MARKETS` in `config.py` also uses `'INDX_COMMON'` / `'INDX_COMMON_'`. Verify these match the actual directory names on both Linux production and Windows dev. A name mismatch means INDX symbols silently return empty without any error log.

### 3. `warmup_cache.py`: Only Warms `10_30` Tier

```python
result = _request('/score', {
    'symbol': sym, 'date': date_str,
    'daysOut': 20, 'direction': 'l', 'tier': '10_30'
})
```

This warms price and opp data caches for all symbols via the `10_30` tier path only. The `31_60` and `61_90` tiers share the same price cache (already warm), but will incur first-request latency for their opp data path on the first live `/score` call. Evaluate whether warmup should also issue `daysOut=45` and `daysOut=75` requests per symbol.

### 4. `daily_opp_selection.py`: Direct Column Access After Parquet Load

`load_candidates` uses direct key access (`df['sharpe_ratio']`, `df['LorS']`, etc.) on the parquet DataFrame. Despite `opp_to_parquet.py` now validating `REQUIRED_COLS` at write time, verify that `load_candidates` and `opp_to_parquet.REQUIRED_COLS` are in sync -- i.e., every column accessed by `load_candidates` is present in `REQUIRED_COLS`. A column used in `load_candidates` but absent from `REQUIRED_COLS` would still raise `KeyError` if the upstream CSV omits it.

### 5. Any New Issues Not Covered in Rounds 1-3

Perform a fresh read of all files with fresh eyes. Focus on:
- `nightly.sh`: error handling, restart race conditions, log rotation
- `daily_opp_selection.py`: filter logic, ranking correctness, edge cases when candidates list is empty
- Thread safety: `engine._parquet_cache` is a plain dict shared across gunicorn threads -- dict operations are GIL-protected in CPython, but confirm this is sufficient for the read/write pattern
- `config.py` `ML_PARQUET_MARKETS` Windows dev detection: the trailing-underscore check uses `os.path.isdir` at import time -- evaluate whether this could be wrong at import time on a machine where the dirs are created after startup

---

## Data Flow Summary (post Round 1 fixes)

```
/score request
  symbol="AAPL", date="2026-04-03", daysOut=20, direction="l"
  (tier omitted -> auto-detected as "10_30" from daysOut=20)
  |
  +--> engine.load_price_data(["AAPL"])
  |      loads AAPL.csv + all market CSVs (SPY, VIX, etc.) if not cached
  |
  +--> engine.compute_features("AAPL", "2026-04-03", 20, "l")
  |      |
  |      +--> compute_pattern_features()
  |      |      _load_opp_files("AAPL", "2026-04-03")
  |      |        -> _load_opp_from_parquet() -- filters _parquet_cache in memory
  |      |           if symbol found: return directly (NOT cached in _opp_cache)
  |      |           if symbol not found (None): fall through
  |      |        -> _load_opp_from_gzip() -- reads opp_by_symbol/*.csv.gz
  |      |           result cached in _opp_cache["AAPL"]
  |      |      scans all combos, builds depth profile -> 22 pat_* features
  |      |
  |      +--> compute_technical_features()      -> 20 ta_* features (5 in model)
  |      +--> compute_market_regime_features()  -> 16 mkt_* features
  |      +--> compute_stock_context_features()  -> 9 ctx_* features (2 in model)
  |      +--> compute_calendar_features()       -> 7 cal_* features (5 in model)
  |      +--> compute_spx_seasonal_features()   -> 4 mkt_spx_* features
  |      +--> compute_interaction_features()    -> 4 interaction features
  |      returns: merged dict ~80 features (scorer uses 59 from FEATURE_COLS)
  |
  +--> VIX check (three-way):
  |      vix is None or vix != vix  -> WARNING logged, proceed
  |      vix > VIX_CUTOFF           -> return error, vix_blocked=True
  |      else                       -> proceed
  |
  +--> tier = scorer_mgr.get_tier(tier_for_days_out(20))  # -> "10_30"
  +--> tier.predict(features)
         X_sr[1x59], X_mfe[1x59] built from FEATURE_COLS order
         avg(LGB, XGB, CatBoost) -> pred_sr, pred_mfe
         pred_sr -> win_prob, p_hit_return (via cal_sr 20-bin lookup)
         pred_mfe -> p_hit_mfe (via cal_mfe 20-bin lookup)
         ml_score = percentile of pred_sr within bins (0-100)
         returns: {pred_return, pred_mfe, win_prob, p_hit_return, p_hit_mfe, ml_score}
```

---

## Environment Context

- **Production**: Linux Ubuntu 20.04, Python 3.12, gunicorn 2 workers, nginx on port 7675, Flask on 5090
- **Dev**: Windows 11, Python 3.12, data dirs have trailing underscore (`sp500_`, `ETF_`)
- **Data dir**: Linux `/home/flask/data`, Windows `C:/seasonals/data`
- **Dependencies**: flask, lightgbm, xgboost, catboost, pandas, numpy, scikit-learn, pyarrow

---

## What a Good Review Should Cover

1. **Correctness issues introduced by Round 1 fixes** -- did any fix create new problems?
2. **Silent wrong scores** -- places where bad inputs or missing data produce plausible-looking but wrong output
3. **API contract breaks** -- the `tier` -> `tiers_used` change; any other consumers affected?
4. **Data integrity** -- schema mismatches between writer (opp_to_parquet) and reader (feature_engine, daily_opp_selection)
5. **Duplicated logic** -- e.g., tier boundary hardcoded in two places
6. **Edge cases** -- NaN propagation into calibration, year rollover, missing data files at inference time
7. **Operational issues** -- warmup coverage gaps, startup logging, deployment debugging

Please prioritize by severity: Critical = could cause wrong trades silently, High = runtime error or data loss, Medium = silent wrong behavior in edge case, Low = code quality or maintainability.
