# ML Scorer Review Results - Round 3

Reviewed against the updated `CODEX_REVIEW.md` and current code in `ml_scorer/`.

## Findings

### High

1. `ml_scorer/app.py:160-172`
   Problem: the Round 2 tier refactor introduced a hard runtime regression. `app.py` imports `tier_for_days_out` from `config.py`, but the mismatch-check path still calls `_tier_for_days_out(days_out)`, which is undefined. This happens before the `try:` block, so any valid `/score` request raises `NameError` and can 500 the whole request instead of returning per-opportunity results.
   Suggested fix: replace `_tier_for_days_out(days_out)` with `tier_for_days_out(days_out)`, and keep the mismatch warning inside the main `try:` block if you want failures to be isolated per opportunity.

2. `ml_scorer/feature_engine.py:187-190`, `ml_scorer/feature_engine.py:209-235`
   Problem: the parquet fast path is still effectively S&P-500-only. `_load_opp_from_parquet()` always starts with the `sp500` parquet path, and if that file exists but the symbol is not present it returns `None` immediately. The multi-market fallback search only runs when the `sp500` parquet file itself does not exist. On a normal production day, that means ETF and index symbols never use their own parquets and always fall through to gzip.
   Suggested fix: iterate across all candidate market parquet paths in priority order and return the first one that actually contains the symbol. Do not stop after checking only `sp500`.

3. `ml_scorer/opp_to_parquet.py:153-169`
   Problem: upstream schema drift can silently drop entire date files during parquet generation. `read_date_file()` subsets to whatever columns exist, then immediately assumes base columns like `avg_profit` and `sharpe_ratio` are present when backfilling `avg_profit2` / `sharpe_ratio2`. If a required base column is renamed or missing, the broad `except` returns `None` and the file is skipped with only a debug log. That silently removes rows from the generated parquet and downstream scoring universe.
   Suggested fix: validate a required schema explicitly at read time, log missing required columns at warning or error level, and fail the generation for that market/date rather than silently skipping the file.

### Medium

4. `ml_scorer/warmup_cache.py:113-123`
   Problem: warmup treats any successful HTTP JSON response as a cache-warm success. It never inspects the `/score` payload for per-opportunity `error` results or `vix_blocked`. A deployment can therefore report `success += 1` for symbols that failed scoring, making the nightly warmup summary unreliable as an operational health signal.
   Suggested fix: inspect `result["results"]` and count a symbol as successful only when the first result contains a real score and no `error`. Track blocked/error cases separately in the warmup summary.

5. `ml_scorer/scorer.py:119-136`, `ml_scorer/scorer.py:139-164`
   Problem: `predict()` does not validate that model outputs are finite before calibration. If any model returns `NaN` rather than raising, `pred_sr`/`pred_mfe` become `NaN`; `_calibrate()` then falls through to the last bin, and `_percentile_score()` returns `100.0`. That can produce contradictory outputs like `win_prob` from the last calibration bin with `ml_score=100` from a non-finite prediction.
   Suggested fix: add a finite-value guard after `_predict_ensemble()` and raise a clear error if either prediction is not finite. Optionally log the feature summary for debugging.

6. `ml_scorer/app.py:158-161`, `ml_scorer/config.py:129-140`
   Problem: there is still no validation that `daysOut` is positive. `daysOut=0` or negative values are accepted, mapped into the `10_30` tier, and passed through feature generation even though several features and labels assume a forward holding period. That allows callers to get plausible-looking scores for nonsensical horizons.
   Suggested fix: reject non-positive `daysOut` at the API boundary with `400`, and consider asserting the same invariant inside `tier_for_days_out()`.

## Checked And Not Findings

- The `vix != vix` NaN check is unusual but correct in plain Python and NumPy scalar contexts.
- The Round 1 parquet-cache corruption issue appears fixed: parquet-backed results are no longer stored in `_opp_cache`.
- The Round 3 brief’s hypothetical “partial preds then `np.mean([])`” path is not how the current code fails. If a model `predict()` call raises, the exception propagates immediately. The remaining risk is non-finite model outputs, not partially discarded predictions.
