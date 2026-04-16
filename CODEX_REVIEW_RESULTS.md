# ML Scorer Review Results

Reviewed against `CODEX_REVIEW.md`, with code inspection of the production package in `ml_scorer/`.

## Findings

### Critical

1. `ml_scorer/feature_engine.py:170-193`, `ml_scorer/feature_engine.py:195-246`, `ml_scorer/feature_engine.py:292-300`
   Problem: `_opp_cache` is keyed only by `symbol`, but the parquet fast path loads only a date-local 5-date slice (`target, +-7, +-14`). After the first request for a symbol, every later request for that symbol reuses the first date’s slice, even if the new request is for a different date. This silently corrupts pattern lookup, neighbor features, and depth features. In the common nightly flow, `warmup_cache.py` preloads one date for every symbol, so later ad hoc `/score` requests for other dates can return stale or all-NaN pattern features with no error.
   Suggested fix: Do not cache parquet-backed opp data by symbol alone. Either key `_opp_cache` by `(symbol, date_str)` or `(symbol, cache_path)`, or reserve `_opp_cache` for full-history gzip loads only and keep date-scoped parquet slices in a separate cache.

### High

2. `ml_scorer/scorer.py:51-81`, `ml_scorer/scorer.py:128-153`
   Problem: Missing model files or calibration files do not fail startup. `_load_ensemble()` silently skips absent files, `_predict_ensemble()` returns `0.0` when no models loaded, `_load_calibration()` returns `[]`, `_calibrate()` falls back to `0.5`, and `_percentile_score()` falls back to `50.0`. The service can therefore emit valid-looking scores from a broken deployment with no hard failure.
   Suggested fix: Treat missing model/calibration artifacts as fatal for each tier/target. Validate that all expected files exist and that each ensemble has at least one model before the service reports healthy. If degraded mode is ever allowed, surface it explicitly in `/health` and refuse `/score`.

3. `ml_scorer/app.py:118-125`, `ml_scorer/app.py:143-159`
   Problem: `/score` applies one top-level `tier` to the entire request and defaults it to `10_30`. Any request for `daysOut > 30` without an explicit tier is scored by the wrong models. Mixed-horizon batch requests are also impossible to score correctly through this API, because tiering is not computed per opportunity.
   Suggested fix: Auto-detect tier from each opportunity’s `daysOut` when `tier` is omitted, and validate explicit `tier` against `daysOut`. For batch requests, either route each row to the proper tier automatically or allow per-opportunity tier overrides.

4. `ml_scorer/feature_engine.py:170-188`, `ml_scorer/feature_engine.py:202-231`, `ml_scorer/feature_engine.py:251-260`, `ml_scorer/config.py:131-142`
   Problem: Opportunity loading is effectively hard-wired to S&P 500 on the parquet path. `_load_opp_from_parquet()` starts from `OPP_BY_SYMBOL_DIR` under `sp500`, and if that parquet exists but does not contain the symbol it returns `{}` rather than continuing to ETF/INDX data. `_load_opp_files()` treats `{}` as success and caches it. The gzip fallback only checks ETF, not indices. This makes the advertised multi-market path (`sp500`, `etf`, `indx`) silently unreliable.
   Suggested fix: Pass market/resource context into the feature engine and resolve parquet/gzip paths from that market directly. Treat an empty symbol slice in one parquet as a miss, not a successful load, so the loader can continue searching the correct market or fall back properly.

### Medium

5. `ml_scorer/feature_engine.py:1066-1073`
   Problem: `ratio_now is not np.nan` / `ratio_then is not np.nan` uses object identity rather than a NaN check. Distinct NaN objects pass this test, so `mkt_breadth_momentum` can be computed from NaNs and silently remain NaN through the subtraction path.
   Suggested fix: Replace the identity checks with `not np.isnan(ratio_now)` and `not np.isnan(ratio_then)`, or use `pd.notna(...)`.

### Low

6. `ml_scorer/feature_engine.py:1526-1574`, `ml_scorer/config.py:83-117`
   Problem: `get_feature_names()` is not the model feature list. It contains 33 extra computed features beyond `FEATURE_COLS`, while scoring uses `FEATURE_COLS` from config. This is not a runtime bug today, but it is a maintenance trap for debugging, documentation, and future validation work.
   Suggested fix: Either derive the scorer-facing list from `config.FEATURE_COLS` or rename `get_feature_names()` to make clear that it returns all computed features, not the trained-model feature schema.

7. `ml_scorer/daily_opp_selection.py:209-210`
   Problem: The docstring example for `resource_ids` still uses `['2', '11']`, but the actual supported values are `sp500`, `etf`, and `indx`.
   Suggested fix: Update the docstring to match the current API contract.

## Checked And Not Findings

- Calibration field naming is consistent in the shipped artifacts: the calibration JSON files use `p_hit_pred`, which matches `scorer.py`.
- `nightly.sh`'s socket existence check is not sufficient on its own, but `warmup_cache.py` immediately follows with a `/health` polling loop, so there is no readiness gap in the current flow.
- The 59 configured model features are present in the feature engine outputs; the main issue is stale/mis-scoped data feeding them, not missing assignments in the normal code paths.
