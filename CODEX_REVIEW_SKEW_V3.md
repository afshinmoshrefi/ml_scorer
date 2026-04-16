# Codex Review: Train/Serve Feature Parity Audit (V3)

## Context

This is a focused train/serve skew audit following the V3 retrain (2026-04-03/04). The production service is `ml_scorer/feature_engine.py`. Training features are computed in `build_training_data.py`. All prior Codex rounds found real bugs -- the most impactful were in feature computation differences between training and serving.

Two major changes were made in V3 that need close review:
1. `_compute_neighborhood_features()` was completely rewritten in production to match training logic (prior version used combo win rates instead of realized price returns)
2. Three new commodity/FX features were added: `mkt_dxy_roc_20`, `mkt_cl_roc_20`, `mkt_gc_roc_20`

Files under review:
- Training: `build_training_data.py` (repo root)
- Production: `ml_scorer/feature_engine.py`

---

## What to Review

### Priority 1 -- Neighborhood Features (highest risk, recently rewritten)

Compare production `_compute_neighborhood_features()` in feature_engine.py (lines 540-657) against `compute_neighborhood_features()` in build_training_data.py (lines 563-692) for every one of these features:
- `pat_neighbor_avg_wr`
- `pat_sharpness`
- `pat_pre_slope`
- `pat_post_cliff`
- `pat_concurrent_count`

For each feature, look for:
- Different loop ranges or iteration order
- Different handling of edge cases (empty price data, missing years, Feb 29)
- Different shift calculations (pre_wrs ordering, year-crossing logic)
- Different normalization or scaling
- Any feature that uses realized price returns in training but something else in production

### Priority 2 -- New Commodity/FX Features

Compare `mkt_dxy_roc_20`, `mkt_cl_roc_20`, `mkt_gc_roc_20` computation:
- Training: `build_training_data.py` `compute_market_regime_series()` lines 354-362 -- uses `.pct_change(20)` on trading-day-indexed Series after `.reindex(mkt.index, method='ffill')`
- Production: `feature_engine.py` `compute_market_regime_features()` lines 1224-1233 -- uses `.tail(30)` then `(iloc[-1] - iloc[-21]) / iloc[-21]`
- Are these semantically equivalent? (trading days vs calendar days, off-by-one risks, NaN handling)
- Is the file path resolution for DXY (INDX dir) vs CL/GC (COMM dir) consistent between training and production?

### Priority 3 -- All Other Features

Walk through every feature in FEATURE_COLS (59 total as used in training) and check for any remaining train/serve differences not caught in prior rounds:
- `pat_hit_last_year`: Feb 29 handling, trading-day forward lookup range
- `cal_day_of_year`: leap year normalization (divide by 366 vs 365)
- `mkt_vix_5d_change`: pct_change(5) in training vs manual (vix - vix_5d_ago) / vix_5d_ago in production
- `mkt_breadth_momentum`: `.diff(20)` on the ratio series in training vs 5-day average windows compared 20 rows apart in production
- `mkt_credit_spread_change_20d`: `.diff(20)` in training vs `iloc[0]` of a tail(25) window in production
- `mkt_fed_rate_direction`: `.diff(60)` (60 trading days) in training vs lookback of `date - 90 calendar days` in production
- `pat_dir_x_sector_trend`: sector ETF above SMA200 in training vs sector ETF 20d return sign in production
- `ctx_pct_from_52w_high` / `ctx_pct_from_52w_low`: rolling(252) vectorized in training vs tail(260) with `.max()/.min()` in production
- Any feature using `.ffill()` or reindex in training but a different method in production

---

## Severity Levels

- **Critical**: Feature computed from different source data or fundamentally different formula -- will cause systematic misprediction
- **High**: Edge case differences (NaN handling, boundary conditions) affecting >1% of scored opportunities
- **Medium**: Minor numerical differences unlikely to affect model decisions
- **Low**: Style/efficiency differences with no semantic impact

---

## Output Format

For each finding:
```
### [Severity] Feature: <feature_name>
**Training**: <how it's computed in build_training_data.py, with line reference>
**Production**: <how it's computed in feature_engine.py, with line reference>
**Skew**: <description of the difference and its impact>
**Fix**: <specific code change needed>
```

If no skew is found for a feature group, state explicitly: "No skew found in [group]."

---

## Pre-Loaded Findings (from human code review -- verify and expand each)

The following differences were identified during the human review that produced this document. Codex must verify each one, assess severity, and produce complete Fix instructions. Additional findings not listed here are expected -- enumerate them all.

---

### Finding A -- `mkt_vix_5d_change`: Different formula

**Training** (build_training_data.py line 277):
```python
mkt['mkt_vix_5d_change'] = vix_close.pct_change(5)
```
`pct_change(5)` computes `(close[t] - close[t-5]) / close[t-5]` over the contiguous trading-day-indexed VIX series.

**Production** (feature_engine.py lines 1031-1032):
```python
vix_5d_ago = vix_sub['close'].iloc[-6]
features['mkt_vix_5d_change'] = (vix - vix_5d_ago) / vix_5d_ago if vix_5d_ago != 0 else 0
```
`iloc[-6]` on a tail-sliced series reaches back 5 rows from the second-to-last position, which is row index -6 of the slice. The semantics are:
- Training: 5 VIX trading days ago
- Production: 5 rows ago in a slice ending at `date` (same intent, but the denominator differs: training normalizes by close[t-5], production normalizes by close[t-5] identically)

However there is a behavioral difference in the zero-value fallback: training returns `NaN` when `close[t-5] == 0` (via `pct_change` behavior), production returns `0`. Also training returns `NaN` when there are fewer than 5 prior observations; production returns `NaN` correctly via the `len(vix_sub) >= 6` guard.

Confirm whether the `0` fallback on zero denominator is a real edge case for VIX and whether it can produce a spurious 0.0 instead of NaN in the production path.

---

### Finding B -- `mkt_breadth_momentum`: Fundamentally different formula

**Training** (build_training_data.py lines 328-329):
```python
# V2: Breadth momentum - 20-day change in adv/decl ratio
mkt['mkt_breadth_momentum'] = adl_ratio.diff(20)
```
`adl_ratio` is the full vectorized 10-day rolling adv/decl ratio series. `.diff(20)` computes `ratio[t] - ratio[t-20]` where both values are 10-day rolling means themselves.

**Production** (feature_engine.py lines 1183-1196):
```python
advn_long = advn_df[advn_df.index <= date].tail(30)
decn_long = decn_df[decn_df.index <= date].tail(30)
# Current 5-day average ratio
adv_now = advn_long['close'].tail(5).mean()
dec_now = decn_long['close'].tail(5).mean()
ratio_now = adv_now / dec_now if dec_now > 0 else np.nan
# 20 days ago 5-day average ratio
adv_then = advn_long['close'].iloc[:5].mean()
dec_then = decn_long['close'].iloc[:5].mean()
ratio_then = adv_then / dec_then if dec_then > 0 else np.nan
features['mkt_breadth_momentum'] = ratio_now - ratio_then
```

This is a **Critical** difference:

1. **Window size mismatch**: Training uses a 10-day rolling mean for the ratio; production uses a 5-day mean.
2. **Lookback distance mismatch**: Training differences the ratio 20 trading days apart. Production takes a tail(30) window, then compares `iloc[:5]` (the earliest 5 rows in the 30-row window) to `tail(5)` (the most recent 5 rows). The distance between `iloc[:5].mean()` and `tail(5).mean()` within a 30-row tail is approximately 25 rows, not 20. Because `advn_long` is `tail(30)`, `iloc[:5]` is approximately 25 trading days ago, not 20.
3. **Ratio computation order**: Training computes `adv_10d / decl_10d` first, then diffs the ratio. Production computes `avg(adv_5d) / avg(decl_5d)` and diffs. These are not equivalent when individual day values vary significantly.

Verify exact row-count math in the production path and determine the true effective lookback. Provide a fix that matches the training formula exactly.

---

### Finding C -- `mkt_credit_spread_change_20d`: Different base date

**Training** (build_training_data.py line 307):
```python
mkt['mkt_credit_spread_change_20d'] = mkt['mkt_credit_spread'].diff(20)
```
`diff(20)` on the full trading-day-indexed credit spread series gives `spread[t] - spread[t-20]` where t-20 is exactly 20 trading days prior.

**Production** (feature_engine.py lines 1099-1101):
```python
hyg_sub = hyg_df[hyg_df.index <= date].tail(25)
lqd_sub = lqd_df[lqd_df.index <= date].tail(25)
spread_20d = hyg_sub['close'].iloc[0] / lqd_sub['close'].iloc[0]
features['mkt_credit_spread_change_20d'] = spread - spread_20d
```

`tail(25)` fetches the most recent 25 trading days. `iloc[0]` is therefore the value from approximately 24 trading days ago, not 20. The production baseline is 4 trading days older than training intended. When the credit spread is trending, this produces a systematically larger (or smaller) change value.

Additionally, `hyg_sub` and `lqd_sub` are fetched independently -- `iloc[0]` of each may be on different calendar dates if one has a missing data point, whereas training aligns both to the shared market index via `reindex(mkt.index, method='ffill')` before computing the ratio.

Verify whether HYG and LQD ever have misaligned trading calendars and whether the 4-day lookback difference is material.

---

### Finding D -- `mkt_fed_rate_direction`: Calendar days vs trading days

**Training** (build_training_data.py line 349):
```python
mkt['mkt_fed_rate_direction'] = irx_close.diff(60)
```
`.diff(60)` on the trading-day-indexed IRX series is exactly 60 trading days (~3 calendar months).

**Production** (feature_engine.py lines 1211-1213):
```python
irx_60d = self._get_price_on_date(irx_df, date - pd.Timedelta(days=90))  # ~60 trading days
features['mkt_fed_rate_direction'] = irx_price - irx_60d
```

`90 calendar days` is approximately 63-64 trading days (not 60). The comment says "~60 trading days" but 90 calendar days is consistently 3-4 trading days longer than 60 trading days. Over a period when the Fed is moving rates, this extra 3-4 days will produce a slightly larger magnitude change in production than training saw.

More importantly, `_get_price_on_date` uses a max_lookback=5 backward search -- it finds the closest available trading day on or before the computed anchor date. In training, `diff(60)` does an exact positional lookup with no fallback: if the 60th-prior row does not exist, the result is NaN, not a value from a nearby day.

---

### Finding E -- `mkt_dxy_roc_20` / `mkt_cl_roc_20` / `mkt_gc_roc_20`: COMM directory missing from training config

**Training** (build_training_data.py line 259, lines 358-362):
```python
def load_csv(subdir, name):
    path = os.path.join(CSV_DIR, subdir, f'{name}.csv')
    ...
cl = load_csv('COMM', 'CL')
gc = load_csv('COMM', 'GC')
```
`load_csv` is a local function inside `compute_market_regime_series()` that constructs paths as `{CSV_DIR}/{subdir}/{name}.csv`. `CSV_DIR` comes from `config_ml.py` which is hardcoded to `C:/seasonals/data/csv`.

**Production** (feature_engine.py lines 1224-1232):
```python
for _sym, _feat in [('DXY', 'mkt_dxy_roc_20'), ('CL', 'mkt_cl_roc_20'), ('GC', 'mkt_gc_roc_20')]:
    _df = self._get_price_df(_sym)
    if _df is not None:
        _sub = _df[_df.index <= date].tail(30)
        if len(_sub) >= 21:
            features[_feat] = (_sub['close'].iloc[-1] - _sub['close'].iloc[-21]) / _sub['close'].iloc[-21]
```

`_get_price_df` calls `_find_csv_path` (lines 97-103) which searches `US`, `ETF`, `INDX`, `COMM` subdirectories in that order. Production loads CL and GC from `COMM_CSV_DIR` (defined in `ml_scorer/config.py` line 11 as `{CSV_DIR}/COMM`).

The training `config_ml.py` has **no `COMM_CSV_DIR` defined** and no COMM subdirectory reference. The local `load_csv('COMM', 'CL')` in training constructs the same path `C:/seasonals/data/csv/COMM/CL.csv` directly from `CSV_DIR`. This means:

1. If `C:/seasonals/data/csv/COMM/CL.csv` does not exist on the training machine, `cl` returns `None` and `mkt_cl_roc_20` is `np.nan` for every training sample. The model was then trained with CL and GC always NaN, but production will have real values. This is a **Critical** skew if the COMM files were absent during training.
2. Confirm whether `C:/seasonals/data/csv/COMM/` existed and contained `CL.csv` and `GC.csv` at training time. If not, the V3 models have never seen real CL/GC values and will receive out-of-distribution inputs at serve time.

Additionally, the ROC-20 computation method differs:
- Training: `.pct_change(20)` on the full trading-day series -- `(close[t] - close[t-20]) / close[t-20]` with NaN for the first 20 rows
- Production: `(sub['close'].iloc[-1] - sub['close'].iloc[-21]) / sub['close'].iloc[-21]` on a tail(30) window

These are semantically equivalent when data is dense, but production's `tail(30)` fetches up to 30 calendar-proximate rows from a forward-filled series, so `iloc[-21]` is 20 rows back within that tail -- which is correct as long as the 30-row tail contains at least 21 rows. The `len(_sub) >= 21` guard handles the minimum length case. This part appears equivalent.

**Verify**: Were CL.csv and GC.csv present in `C:/seasonals/data/csv/COMM/` during the V3 training run?

---

### Finding F -- `pat_dir_x_sector_trend`: Different sector trend signal

**Training** (build_training_data.py lines 983-994):
```python
etf_sma200 = etf_c.rolling(200).mean().reindex(merged['date'], method='ffill')
sector_trend = np.where(etf_at_date > etf_sma200, 1, -1)
merged['pat_dir_x_sector_trend'] = pat_dir_sign * sector_trend
```
Training uses `sector_etf_price > sector_etf_SMA200` as the sector trend signal: a binary above/below long-term trend indicator. It never produces 0 (always +1 or -1).

**Production** (feature_engine.py lines 1540-1543):
```python
etf_ret = (etf_sub['close'].iloc[-1] - etf_sub['close'].iloc[-21]) / etf_sub['close'].iloc[-21]
sector_sign = 1.0 if etf_ret >= 0 else -1.0
result['pat_dir_x_sector_trend'] = dir_sign * sector_sign
```
Production uses the **sign of the sector ETF 20d return** as the sector trend signal. This is a **Critical** difference: a sector ETF could be above its 200-day SMA but have a negative 20d return (consolidation after a bull run), or below its 200-day SMA with a positive 20d return (dead-cat bounce). The two signals will disagree frequently and have different magnitudes of mean-reversion.

The training signal captures structural trend regime (bull/bear cycle). The production signal captures recent momentum. These are fundamentally different features under the same name.

---

### Finding G -- `mkt_spy_roc_20`: Slightly different denominator position

**Training** (build_training_data.py line 315):
```python
mkt['mkt_spy_roc_20'] = spy_close.pct_change(20)
```
`pct_change(20)` uses the value exactly 20 rows (20 trading days) prior as the denominator.

**Production** (feature_engine.py lines 1115-1117):
```python
spy_ret_20d = (spy_sub['close'].iloc[-1] - spy_sub['close'].iloc[-21]) / spy_sub['close'].iloc[-21]
```
`tail(250)` is fetched first, then `iloc[-21]` is the 21st-from-last element of that tail (i.e., 20 rows prior to the last element). This is semantically equivalent to `pct_change(20)`.

However, note that `spy_sub` is `tail(250)` with no `min_periods` constraint. If the SPY series has fewer than 250 rows before `date`, `tail(250)` returns whatever is available, and `iloc[-21]` correctly reaches 20 rows back. This matches training behavior. **No semantic skew found here** -- but Codex should confirm that `len(spy_sub) >= 21` is the correct guard (it is, matching `pct_change` which returns NaN for the first 20 rows).

---

### Finding H -- `ctx_pct_from_52w_high` / `ctx_pct_from_52w_low`: Different rolling window

**Training** (build_training_data.py lines 243-247):
```python
high_52w = highs.rolling(252, min_periods=20).max()
low_52w = lows.rolling(252, min_periods=20).min()
ctx['ctx_pct_from_52w_high'] = (closes - high_52w) / high_52w
ctx['ctx_pct_from_52w_low'] = (closes - low_52w) / low_52w.replace(0, np.nan)
```
Training uses `rolling(252)` (252 trading days = 1 year) with `min_periods=20`. It uses high-of-highs for the 52-week high and low-of-lows for the 52-week low.

**Production** (feature_engine.py lines 1259-1273):
```python
sub = df[df.index <= date].tail(260)  # ~1 year
high_52w = sub['high'].max()
low_52w = sub['low'].min()
features['ctx_pct_from_52w_high'] = (price - high_52w) / high_52w if high_52w > 0 else np.nan
features['ctx_pct_from_52w_low'] = (price - low_52w) / low_52w if low_52w > 0 else np.nan
```
Production fetches `tail(260)` rows. Differences:

1. **Window size**: Training uses 252 rows; production uses 260 rows. The extra 8 days means production may capture an extreme high or low that is ~1.5 months older than what training saw, potentially making `ctx_pct_from_52w_high` more negative (further from a higher historical peak).
2. **52w high source**: Training uses `high` column (intraday high). Production also uses `sub['high'].max()`. Same column -- no skew here.
3. **NaN handling**: Training uses `ctx['ctx_pct_from_52w_low'] = (closes - low_52w) / low_52w.replace(0, np.nan)` -- divides by NaN when low is 0. Production uses `if low_52w > 0 else np.nan` -- same intent but `low_52w` is a scalar from `.min()`, not a Series, so the `.replace()` pattern doesn't apply. Behavioral equivalence holds.
4. **min_periods**: Training has `min_periods=20` meaning the high/low is computed from as few as 20 bars. Production has no minimum check beyond `len(sub) < 20` at line 1260. These are equivalent.

The 252 vs 260 row difference is a **Medium** skew.

---

### Finding I -- `cal_day_of_year`: Different normalization for leap years

**Training** (build_training_data.py line 1060):
```python
merged['cal_day_of_year'] = dates.dt.dayofyear / (365 + dates.dt.is_leap_year.astype(int))
```
For leap years this divides by 366; for non-leap years by 365. This is correct and consistent.

**Production** (feature_engine.py line 1350):
```python
features['cal_day_of_year'] = date.timetuple().tm_yday / (366 if date.is_leap_year else 365)
```
Identical logic. No skew.

---

### Finding J -- `mkt_vix_percentile_60d`: Different computation method

**Training** (build_training_data.py line 276):
```python
mkt['mkt_vix_percentile_60d'] = vix_close.rolling(60).rank(pct=True)
```
`rolling(60).rank(pct=True)` is a proper rolling percentile rank using pandas' efficient rank implementation. For each date, it ranks the current VIX close within the prior 60-day window.

**Production** (feature_engine.py lines 1025-1026):
```python
vix_60 = vix_sub['close'].tail(60).values
features['mkt_vix_percentile_60d'] = sum(1 for v in vix_60 if v <= vix) / len(vix_60)
```
Production counts the number of values in the 60-day window that are `<= vix` (inclusive of today's value) divided by 60. This is equivalent to `rank(pct=True)` with `method='max'` for ties -- the current value is always counted in the numerator since `v <= vix` includes `v == vix`. Pandas `rank(pct=True)` defaults to `method='average'` for ties.

For non-tied VIX values (the common case), both formulas give identical results. For tied VIX values, the production result will be slightly higher (includes the current row in the count of values <= current). This is a **Low** skew under normal conditions.

---

### Finding K -- `ta_trend_direction_match`: Different short-side logic

**Training** (build_training_data.py lines 959-967):
```python
tl = merged['ta_trend_long']
is_long = merged['direction'] == 'l'
tdm = pd.Series(0, index=merged.index, dtype=float)
tdm[is_long & (tl >= 60)] = 1
tdm[is_long & (tl <= 40)] = -1
tdm[~is_long & (tl >= 60)] = -1
tdm[~is_long & (tl <= 40)] = 1
tdm[tl.isna()] = np.nan
merged['ta_trend_direction_match'] = tdm
```
Training computes this from `ta_trend_long` ONLY (not `ta_trend_short`). For shorts, high `trend_long` = -1 (against the short), low `trend_long` = +1 (favors the short).

**Production** (feature_engine.py lines 1602-1617):
```python
if dir_char == 'l':
    if ta['ta_trend_long'] >= 60:
        ta['ta_trend_direction_match'] = 1
    elif ta['ta_trend_long'] <= 40:
        ta['ta_trend_direction_match'] = -1
    else:
        ta['ta_trend_direction_match'] = 0
else:
    if ta['ta_trend_short'] >= 60:
        ta['ta_trend_direction_match'] = 1
    elif ta['ta_trend_short'] <= 40:
        ta['ta_trend_direction_match'] = -1
    else:
        ta['ta_trend_direction_match'] = 0
```
Production uses `ta_trend_short` for the short branch. Training uses `ta_trend_long` for both branches (with inverted sign for shorts). This is a **Critical** difference for short-direction patterns.

When a short pattern is scored:
- Training saw: `tdm = -1` when `trend_long >= 60`, `tdm = +1` when `trend_long <= 40`
- Production gives: `tdm = +1` when `trend_short >= 60`, `tdm = -1` when `trend_short <= 40`

These are logically opposite signals. `trend_short` is high when `trend_long` is low (they are complementary scores). So production's `trend_short >= 60` typically corresponds to training's `trend_long <= 40`, meaning production assigns `+1` where training would have assigned `+1` -- the signs may accidentally be equivalent in most cases because `trend_short = 100 - trend_long` (approximately). However, this requires careful verification since the production trend score does not enforce an exact complementary relationship. Verify by checking `_compute_trend_scores()` to determine whether `trend_long + trend_short == 100` always holds.

---

### Finding L -- `pat_concurrent_count`: Different counting scope

**Training** (build_training_data.py lines 785-821):
```python
date_pattern_counter = Counter()
for month_day, daysOut, direction in patterns:
    ...
    for year in TRAIN_YEARS:
        entry_date = pd.Timestamp(f"{year}-{month_day}")
        ...
        date_pattern_counter[entry_date] += 1
...
sym_df['pat_concurrent_count'] = [date_pattern_counter[entry_dates[i]] for i in valid_idx]
```
Training counts all patterns active on a given entry date for the current symbol, across all `(month_day, daysOut, direction)` pattern tuples that survive the `valid_idx` filter (i.e., have non-NaN labels). The counter increments once per `(year, pattern)` combination that has a valid entry date.

**Production** (feature_engine.py lines 467-473):
```python
active_pairs = set()
for _combo_lookup in combos.values():
    for _key in _combo_lookup:
        if _key[0] == date_str:
            active_pairs.add((_key[1], _key[2]))  # (daysOut, direction)
features['pat_concurrent_count'] = float(len(active_pairs))
```
Production counts unique `(daysOut, direction)` pairs appearing in any combo on `date_str`. This counts the number of distinct pattern types active on this date, regardless of combo depth or daysOut tier filter.

The key difference: training counts patterns that passed the `entry_date` near-trading-day lookup and price history filter (200+ rows). Production counts all patterns present in any combo file for this date, without the price/history filter. On sparse data (early years, newly listed stocks), production may count patterns that training would have excluded due to insufficient history. This is a **Medium** skew.

---

### Finding M -- Neighborhood features: `pat_pre_slope` ordering

**Training** (build_training_data.py lines 680-682):
```python
if len(pre_avg_rets) == 2:
    result['pat_pre_slope'] = pre_avg_rets[1] - pre_avg_rets[0]  # pre1w_wr - pre2w_wr
```
`pre_avg_rets` is accumulated in the order the shifts dictionary is iterated. The shifts are defined as:
```python
shifts = [(-14, 'pre2w'), (-7, 'pre1w'), (7, 'post1w'), (14, 'post2w')]
```
In the aggregation loop (lines 660-670), `pre_avg_rets` appends whenever `'pre' in label`. Since dict iteration is insertion-ordered in Python 3.7+, and the shifts are inserted with `pre2w` before `pre1w`, `pre_avg_rets[0]` = pre2w_wr and `pre_avg_rets[1]` = pre1w_wr. So `pat_pre_slope = pre1w_wr - pre2w_wr`. This captures whether win rates are increasing as we approach the pattern start (positive slope = momentum building toward pattern).

**Production** (feature_engine.py lines 626-645):
```python
shifts = [(-14, 'pre2w'), (-7, 'pre1w'), (7, 'post1w'), (14, 'post2w')]
...
pre_wrs = [shifted_wrs[l] for l in ('pre2w', 'pre1w') if l in shifted_wrs]
pre_slope = (pre_wrs[1] - pre_wrs[0]) if len(pre_wrs) == 2 else np.nan
```
Production explicitly orders `pre_wrs` as `[pre2w_wr, pre1w_wr]` by constructing the list from keys in that order. `pre_slope = pre_wrs[1] - pre_wrs[0] = pre1w_wr - pre2w_wr`.

Both formulas compute `pre1w_wr - pre2w_wr`. No skew -- but Codex should verify that the training loop iteration order is deterministic (Python dict insertion order is guaranteed in 3.7+, and the `shifted_wrs` dict in training is a plain dict populated in shifts-list order).

---

### Finding N -- Neighborhood features: `pat_concurrent_count` missing from `_compute_neighborhood_features()` return

**Production** (feature_engine.py lines 551-556):
```python
nan_result = {
    'pat_neighbor_avg_wr': np.nan,
    'pat_sharpness': np.nan,
    'pat_pre_slope': np.nan,
    'pat_post_cliff': np.nan,
}
```
`pat_concurrent_count` is NOT part of `_compute_neighborhood_features()`. It is computed separately at lines 468-473. This matches training where `pat_concurrent_count` is computed via `date_pattern_counter` (line 895) independently from `compute_neighborhood_features()`. No skew here, but Codex should confirm the production path assigns `pat_concurrent_count` before returning from `compute_pattern_features()` (it does, at line 473).

---

## Additional Items for Codex to Verify

These items were noted during code review but require active runtime analysis or additional context to confirm:

1. **`mkt_dxy_roc_20` path resolution on production server**: `load_price_data()` (lines 113-130) explicitly loads DXY from `INDX_CSV_DIR`. `_find_csv_path()` (lines 97-103) searches in order `US -> ETF -> INDX -> COMM`. If DXY exists in INDX on production but was loaded from INDX in training (`load_csv('INDX', 'DXY')`), paths are consistent. Confirm that DXY.csv exists in `{DATA_DIR}/csv/INDX/` on the production server at 104.238.214.253.

2. **`mkt_vix_term_structure` alignment**: Training (line 286) uses `vix3m_df['close'].reindex(vix_df.index, method='ffill')` -- aligns VIX3M to the VIX trading calendar. Production (lines 1040-1042) does a direct `iloc[-1]` on `vix3m_df` filtered to `<= date`. If VIX3M has missing dates that VIX does not, production will return the same most-recent value but training will have forward-filled the VIX3M to the VIX date. These can differ on days where VIX has a data point but VIX3M does not. Low risk but verify if VIX3M has sparse data.

3. **`pat_hit_last_year` Feb 29 handling**: Production (lines 499-502) wraps the prior-year base date in a try/except ValueError and returns NaN on Feb 29 inputs. Training (lines 922-923) does the same. Verify that both paths produce identical NaN behavior for a Feb 29 entry date in a leap year (prior year is non-leap, so `{year-1}-02-29` raises ValueError).

4. **SPX seasonal lookup end-year**: Training builds one lookup per year using data through `year-1` (line 140: `end_yr = year - 1`). Production (lines 1395-1401) builds a single lookup per scoring session using `end_year = date.year - 1`. For a sample dated 2024-06-15, training used data through 2023; production uses data through 2023. These are equivalent. **No skew**.

5. **`mkt_yield_curve_slope` lookback**: Training (line 296): `.diff(5)` on the trading-day spread series = 5 trading days. Production (lines 1071-1075): looks up yield curve at `date - 7 calendar days`. Seven calendar days is approximately 5 trading days, but on weeks with a holiday it could be only 4 trading days. The `_get_price_on_date` backward-search (max_lookback=5) will find a value even if the exact date is missing. Verify whether the 7-calendar-day lookback can produce a 4-trading-day difference and whether this is material.

6. **`mkt_adv_decl_ratio_10d` minimum sample size**: Training uses `adv_10d / dec_10d` where `adv_10d` is a `.rolling(10).mean()` -- first 9 rows are NaN. Production requires `len(advn_sub) >= 10` at line 1136. Both need at least 10 rows. Consistent.

7. **`pat_daysOut` feature in training vs production**: Training adds `pat_daysOut` at line 898 as a float32 copy of `daysOut`. Production sets `features['pat_daysOut'] = daysOut` at line 422 (inside `compute_pattern_features`). Both are the holding period in days. No skew -- but confirm that the model's FEATURE_COLS list in `config.py` includes `pat_daysOut` (it does, as the CRITICAL note in CLAUDE.md confirms).

---

## File References Summary

| File | Key Lines |
|------|-----------|
| `build_training_data.py` | `compute_market_regime_series()`: 252-364 |
| `build_training_data.py` | `compute_neighborhood_features()`: 563-692 |
| `build_training_data.py` | `compute_all_ta_series()`: 156-228 |
| `build_training_data.py` | `compute_all_context_series()`: 231-249 |
| `build_training_data.py` | `process_symbol()` -- interaction features: 978-1055 |
| `build_training_data.py` | `process_symbol()` -- calendar features: 1057-1075 |
| `build_training_data.py` | `compute_pattern_features_fast()`: 468-556 |
| `ml_scorer/feature_engine.py` | `_compute_neighborhood_features()`: 540-657 |
| `ml_scorer/feature_engine.py` | `compute_market_regime_features()`: 1007-1235 |
| `ml_scorer/feature_engine.py` | `compute_technical_features()`: 749-898 |
| `ml_scorer/feature_engine.py` | `compute_stock_context_features()`: 1241-1334 |
| `ml_scorer/feature_engine.py` | `compute_calendar_features()`: 1340-1381 |
| `ml_scorer/feature_engine.py` | `compute_interaction_features()`: 1504-1568 |
| `ml_scorer/feature_engine.py` | `compute_pattern_features()`: 309-487 |
