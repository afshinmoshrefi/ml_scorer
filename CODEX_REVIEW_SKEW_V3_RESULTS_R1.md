# Codex Review Results: Train/Serve Feature Parity Audit (V3 Round 1)

## Critical

### [Critical] Feature: `mkt_breadth_momentum`
**Training**: [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L325) computes `adl_ratio = adv_10 / dec_10` using 10-day rolling means, then `mkt_breadth_momentum = adl_ratio.diff(20)`.

**Production**: [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1182) builds a `tail(30)` slice, computes a 5-day mean ratio for the last 5 rows and another 5-day mean ratio for the first 5 rows of that 30-row tail, then subtracts them.

**Skew**: This is a real train/serve skew. Production differs from training in three ways:
- 5-day averaging instead of 10-day averaging
- approximately 25 trading days of separation instead of 20
- averaging numerator/denominator windows separately instead of diffing the already-formed ratio series

This changes both the smoothing behavior and the effective lookback, so the production model is receiving a materially different feature than it was trained on.

**Fix**: Recompute production breadth momentum exactly like training:
- build the 10-day rolling `adv/decl` ratio on the full history up to `date`
- take the current ratio value and subtract the ratio value 20 trading rows earlier
- return `NaN` if fewer than 30 rows are available to support the 10-day roll plus 20-row diff

### [Critical] Feature: `pat_dir_x_sector_trend`
**Training**: [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L983) defines sector trend as `sector ETF price > sector ETF SMA200`, then multiplies that `+1/-1` signal by pattern direction.

**Production**: [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1532) defines sector trend as the sign of the sector ETF 20-day return, then multiplies that by pattern direction.

**Skew**: This is a fundamental feature-definition mismatch. Training uses long-term trend regime; production uses short-term momentum. These signals will disagree frequently, especially during consolidations and countertrend rallies.

**Fix**: Change production to mirror training exactly:
- compute sector ETF price at `date`
- compute sector ETF SMA200 from the trailing 200 trading rows
- set sector trend to `+1` if `price > SMA200`, else `-1`
- multiply by `dir_sign`

## High

### [High] Feature: `mkt_credit_spread_change_20d`
**Training**: [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L301) computes `mkt_credit_spread = HYG / LQD` on aligned series, then `diff(20)`.

**Production**: [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1091) takes `tail(25)` on HYG and LQD separately and uses `iloc[0]` as the comparison baseline.

**Skew**: Production is not using the 20-trading-day-prior value. `tail(25).iloc[0]` is about 24 rows back, so the baseline is four trading days older than training intended. Production also takes the first row of each instrument independently, while training aligns the ratio series before differencing. If HYG and LQD calendars ever diverge, production can compare mismatched dates.

**Fix**: In production:
- align HYG and LQD to a shared index up to `date`
- form the ratio series first
- compute `current_ratio - ratio.iloc[-21]`
- require at least 21 aligned rows

### [High] Feature: `mkt_fed_rate_direction`
**Training**: [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L346) computes `irx_close.diff(60)`, which is exactly 60 trading rows.

**Production**: [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1210) looks up `IRX` at `date - 90 calendar days` via `_get_price_on_date()`.

**Skew**: Production is using a calendar anchor, not a 60-trading-day positional anchor. That usually means a 63-64 trading day lookback, plus a backward-search fallback. Training returns `NaN` if the 60th prior row does not exist; production can still return a nearby value.

**Fix**: In production, compute this from the trailing IRX series exactly as training does:
- slice `irx_df[irx_df.index <= date]`
- use `current - close.iloc[-61]`
- return `NaN` if fewer than 61 rows exist

## Medium

### [Medium] Feature: `mkt_yield_curve_slope`
**Training**: [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L292) computes `mkt_yield_curve_10y2y.diff(5)`, which is a 5-trading-day change in the spread series.

**Production**: [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1067) reconstructs the spread at `date`, then compares it with the spread at `date - 7 calendar days` using backward lookup.

**Skew**: This is a smaller version of the IRX issue. Seven calendar days is usually about five trading days, but on holiday weeks it can be only four. Production also uses fallback lookup instead of exact row distance.

**Fix**: Build the aligned yield-curve spread series up to `date` and compute an exact 5-row difference in production.

### [Medium] Feature: `ctx_pct_from_52w_high`, `ctx_pct_from_52w_low`
**Training**: [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L243) uses `rolling(252, min_periods=20).max()` and `.min()` on `high`/`low`.

**Production**: [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1260) uses `tail(260)` and then takes `max()` / `min()` across that larger window.

**Skew**: Production uses an 8-row longer lookback than training. That can pull in older extremes and shift both context ratios.

**Fix**: Change production to use `tail(252)` for these features, keeping the existing `len(sub) < 20` minimum gate.

### [Medium] Feature: `mkt_vix_percentile_60d`
**Training**: [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L276) uses `rolling(60).rank(pct=True)`.

**Production**: [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1026) computes `count(v <= current) / len(window)`.

**Skew**: For non-tied values these are equivalent. For tied VIX values, production behaves like a max-rank percentile, while pandas uses average rank by default. This is a minor numeric difference, not a structural skew.

**Fix**: Optional. If exact parity matters, compute the percentile rank using pandas over the 60-row trailing window in production.

## Low

### [Low] Feature: `mkt_vix_5d_change`
**Training**: [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L277) uses `pct_change(5)`.

**Production**: [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1032) manually computes `(vix - vix_5d_ago) / vix_5d_ago`, but falls back to `0` if the denominator is zero.

**Skew**: The main formula matches training. The only difference is the zero-denominator fallback. For VIX, a true zero close is not realistic, so this is effectively a theoretical edge case.

**Fix**: Return `np.nan` instead of `0` when the denominator is zero to match training semantics exactly.

## No Skew Found

### Neighborhood Features: `pat_neighbor_avg_wr`, `pat_sharpness`, `pat_pre_slope`, `pat_post_cliff`
No train/serve skew found in the rewritten neighborhood feature logic. The production `_compute_neighborhood_features()` in [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L508) now matches training [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L558) on:
- prior-year loop range
- year-crossing shift handling
- forward trading-day lookup ranges
- win-rate aggregation
- `pre1w - pre2w` ordering
- `pat_wr_prior / neighbor_avg` sharpness definition
- `pat_wr_prior - mean(post1w, post2w)` cliff definition

### Feature: `pat_hit_last_year`
No skew found. Training [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L913) and production [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L489) now both:
- return `NaN` on Feb. 29 prior-year invalid dates
- use forward 0-3 day entry lookup
- use forward 0-4 day exit lookup

### Feature: `cal_day_of_year`
No skew found. Training [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L1060) and production [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1333) both normalize by the actual number of days in the year.

### Feature: `mkt_spy_roc_20`
No skew found. Training `pct_change(20)` and production `iloc[-21]` on a trailing slice are semantically equivalent.

### Feature: `mkt_dxy_roc_20`, `mkt_cl_roc_20`, `mkt_gc_roc_20`
No code-level skew found in the ROC formula. Training [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L356) uses `pct_change(20)` on forward-filled series; production [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1224) uses `iloc[-21]` on trailing slices. These are equivalent when at least 21 rows exist.

I also verified locally that:
- `C:\seasonals\data\csv\COMM\CL.csv` exists
- `C:\seasonals\data\csv\COMM\GC.csv` exists
- `C:\seasonals\data\csv\INDX\DXY.csv` exists

So the specific "COMM files absent during training" failure mode described in the review brief is not supported by the local environment.

### Feature: `ta_trend_direction_match`
No skew found. Training [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L959) uses `ta_trend_long` for both directions with sign inversion. Production [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L1602) uses `ta_trend_short` for shorts, but `_compute_trend_scores()` returns complementary long/short scores whose component totals sum to 100, so the thresholds are equivalent.

### Feature: `mkt_vix_term_structure`
No meaningful skew found. Training forward-fills `VIX3M` onto the VIX calendar; production uses the latest available `VIX3M` value at or before `date`. These are equivalent unless the two series have missing dates on different calendars, which appears to be a low-risk edge case.

### Feature: `pat_concurrent_count`
No meaningful current-serving skew found. Training [build_training_data.py](C:\seasonals\ml_scorer\build_training_data.py#L784) counts unique pattern tuples active on the entry date for the symbol; production [feature_engine.py](C:\seasonals\ml_scorer\ml_scorer\feature_engine.py#L465) counts unique `(daysOut, direction)` pairs present on `date_str` across combos. For current live symbols with sufficient history, these scopes are effectively aligned. The remaining difference is limited to sparse early-history training rows and is not a current serve-path issue.

## Summary

The V3 neighborhood rewrite appears successful: I did not find remaining skew in the highest-risk neighborhood features.

The real remaining train/serve skews are concentrated in:
- `mkt_breadth_momentum`
- `pat_dir_x_sector_trend`
- `mkt_credit_spread_change_20d`
- `mkt_fed_rate_direction`
- `mkt_yield_curve_slope`
- `ctx_pct_from_52w_high` / `ctx_pct_from_52w_low`

Of these, the first two are the most important and should be fixed before relying on strict V3 train/serve parity.
