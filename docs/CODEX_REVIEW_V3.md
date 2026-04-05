# Codex Review Brief: ML Scorer V3 Production Service

## Task

Review the ML Pattern Scorer V3 production service for bugs, performance issues, and memory problems. Read all five files listed below in full before writing findings. Fix any issues found directly in the code.

---

## What This Service Does

Flask service that scores seasonal stock/ETF pattern opportunities. Takes a symbol, date, daysOut (holding period), and direction (long/short). Computes 62 features from price CSVs and opportunity files, runs them through a 3-model ensemble (LightGBM + XGBoost + CatBoost), and returns calibrated predictions.

**Production server:** Ubuntu 20.04, 4GB RAM, gunicorn with 1 worker.

**Two models per tier:**
- SR model: predicts actual return (%). Used for win_prob and ml_score.
- MFE model: predicts max favorable excursion (%). Used for profit target sizing.

**Three tiers:** `10_30` (10-30 day holds), `31_60`, `61_90`.

---

## Files to Review

Review all five files. Do not review config.py or app.py unless a bug in another file traces back to them.

### 1. `feature_engine.py` (~1800 lines) -- HIGHEST PRIORITY

The most complex file. Computes all 62 features for a single opportunity.

**Recent change (memory fix):** The `_load_opp_from_gzip` method was patched from a full `pd.read_csv()` load (all history, all 116 combo files) to a targeted line-by-line scan that only extracts 5 dates (target +-7, +-14 days). Cache key was changed from `symbol` to `(symbol, date_str)`. This is the most recently modified area -- verify correctness.

**Known architecture:**
- `_price_cache`: symbol -> DataFrame. Market symbols (SPY, VIX, bonds, DXY, CL, GC, sector ETFs) loaded eagerly on first request. User symbol price CSVs lazy-loaded on demand.
- `_opp_cache`: (symbol, date_str) -> combos dict. Gzip-backed, date-specific.
- `_parquet_by_symbol`: symbol -> DataFrame, reset on date change. Parquet-backed fast path for opp data.
- `_market_cache`: precomputed regime indicators keyed by date.
- `_ta_cache`: technical indicators keyed by (symbol, date).

**Focus areas:**
- Correctness of the patched `_load_opp_from_gzip` and `_load_opp_files` methods
- Cache invalidation logic (does the parquet reset on date change work correctly alongside the gzip cache?)
- Memory leaks: caches that grow unbounded per worker lifetime
- `load_price_data()`: checks for both eager (market symbols) and lazy (user symbols) -- verify logic is correct
- Feature computation correctness: NaN handling, edge cases for missing price data, division by zero
- Any `iterrows()` usage that could be replaced (slow on large DataFrames)
- Exception handling: silent failures that return 0 or NaN instead of raising

### 2. `scorer.py` (~210 lines)

Model loading, ensemble prediction, calibration lookup.

**Focus areas:**
- `_calibrate()`: linear scan through 20 bins on every prediction -- called 3x per score (win_prob, p_hit_return, p_hit_mfe). Consider bisect for O(log n) lookup.
- `_percentile_score()`: same linear scan issue.
- `_predict_ensemble()`: creates a new `xgb.DMatrix` on every call -- check if this is avoidable.
- Feature array construction in `predict()`: uses `feature_dict.get(f, np.nan)` -- if a feature is missing entirely, the model receives NaN silently. Should this raise?
- `_validate_features()` at startup: validates count only, not names. A reordered feature list would pass validation but produce wrong predictions.

### 3. `daily_opp_selection.py` (~255 lines)

Called by `/select` endpoint. Loads parquet files, pre-filters candidates, scores them, ranks results.

**Focus areas:**
- `score_candidates()` uses `iterrows()` on the candidates DataFrame -- should use itertuples or convert to list of dicts first.
- `load_candidates()` calls `pd.concat` then immediately filters to `df['date'] == date` -- filtering before concat would reduce memory.
- VIX cutoff is hardcoded as `35` instead of using the `VIX_CUTOFF` config constant.
- `rank_and_select()`: the duplicate symbol check uses a set -- correct, but the pre-filter already limits to 3 candidates per symbol so duplicates are expected. Verify the dedup logic handles the 3-per-symbol correctly.
- Error path: if parquets don't exist for any requested market, returns empty picks with no indication of which markets were missing vs genuinely having no candidates.

### 4. `app.py` (~325 lines)

Flask routes: `/score`, `/select`, `/health`, `/tiers`.

**Focus areas:**
- `/score` calls `engine.load_price_data(symbols)` before scoring -- good for batches, but on single requests it's a no-op extra call.
- Tier mismatch warning log: logs a WARNING when `tier` is explicitly passed and differs from auto-detected -- verify this doesn't fire spuriously for legitimate mixed-tier batches.
- `/select` validates numeric params but doesn't validate `date` format -- an invalid date string propagates into parquet path construction.
- Exception handling in the scoring loop: catches all exceptions and appends an error result. Verify that a single bad symbol doesn't block the rest of the batch (it shouldn't, but confirm).

### 5. `scorer_config.py` (V1 entry point -- may not exist in V3)

V1 used `scorer_config.py` for HOST/PORT. V3 uses `config.py`. Check whether `scorer_config.py` still exists and if so whether it conflicts.

---

## Memory Constraints

**Critical:** 4GB RAM, 1 gunicorn worker. The worker holds all state for its lifetime.

Caches that grow without bound are dangerous:
- `_price_cache`: bounded by number of unique symbols ever scored in the worker's lifetime. Each symbol DataFrame is ~1-5MB. 500 symbols = ~500MB-2.5GB. This is the largest memory risk.
- `_opp_cache`: bounded by `(symbol, date_str)` pairs. With the targeted gzip scan, each entry is small (5 dates x ~116 combos). Low risk.
- `_parquet_by_symbol`: reset on date change. Bounded by one day's parquet size. Low risk.
- `_market_cache`: regime indicators, one entry per date. Grows slowly. Low risk.
- `_ta_cache`: technical indicators per `(symbol, date)`. Grows with unique (symbol, date) pairs. Medium risk.

**If you find that `_price_cache` or `_ta_cache` grow unboundedly, recommend a cap (e.g., evict oldest entries when > N symbols cached).**

---

## Known Good Behaviour (Do Not Break)

- 62 features in `FEATURE_COLS` -- count and order must match trained models exactly
- `pat_daysOut` must always be in the feature list
- VIX > 35 blocks scoring per-opportunity (not per-request)
- Parquet fast path for opp data, gzip fallback when parquet missing
- All model files validated at startup (count match); service refuses to start on mismatch

---

## Expected Output

For each issue found:
1. File and line number
2. Description of the bug, performance problem, or memory risk
3. Fix applied directly in the code

If no issues are found in a file, state that explicitly.
