# L1 Direction Backtest -- Codex Review Brief

## Purpose

This is a fresh backtest script (`backtest_l1_directions.py`) written to test the S21 aggressive
strategy (WP/EP/T85/SK/3/C2) across three direction modes: Long-only, Short-only, and Long+Short
combined. This is new code -- the original `backtest_strategies.py` was long-only only.

The core question is whether the simulation correctly handles short positions. A bug in the
short-side P&L calculation, exit logic, or sizing would invalidate the short and combined results.

**Return findings as a prioritized list: Critical > High > Medium > Low.**
For each finding include: file, line(s), description, and suggested fix.

---

## Context

### The Original System (backtest_strategies.py)

- Long-only, day-by-day simulation using real price data
- EP exit: 3% trailing stop from high-water mark, triggers after day 2
- Kelly sizing (SK): quarter-Kelly, capped at 2x equal share
- C2 concentration: max 2 positions per sector
- WP ranking: sort by win_probability descending
- Hard stop: single-position loss > 10% of start-of-day equity
- Drawdown halt: 15% portfolio drawdown triggers 20-day halt
- Earnings filter: skip any trade with earnings during hold period
- `actual_return` in the parquet is **direction-adjusted** (positive = win for either direction)

### The New Script

`backtest_l1_directions.py` extends this with:

1. Both directions loaded from parquet (no longs-only filter)
2. `direction` stored in each position dict
3. For shorts: `cum_return = 1.0 - (price_today / entry_price)` instead of `price_today / entry_price - 1.0`
4. Kelly R computed separately per direction: `table[(dir_, thresh, year)] = R`
5. Direction filter applied per strategy: `"l"`, `"s"`, or `"both"`
6. SkipMonday filter: skip new entries on Mondays (weekday == 0)

### Results Produced

| Config | Sharpe | CAGR | DD | WR | Trades | All Yrs+ |
|--------|--------|------|-----|-----|--------|----------|
| L1_base_LS (both) | 4.09 | 64.1% | 6.7% | 58.0% | 750 | 8/8 |
| L1_skip_LS (both+SkipMon) | 3.92 | 64.0% | 7.1% | 58.2% | 740 | 8/8 |
| L1_skip_L (long+SkipMon) | 3.68 | 63.7% | 7.1% | 56.6% | 755 | 8/8 |
| L1_base_L (long-only) | 3.66 | 64.1% | 6.7% | 55.9% | 780 | 8/8 |
| L1_base_S (short-only) | 2.40 | 37.5% | 5.4% | 58.5% | 465 | 6/8 |
| L1_skip_S (short-only+SkipMon) | 2.26 | 35.2% | 5.9% | 59.0% | 424 | 6/8 |

The long-only baseline (Sharpe 3.66, CAGR 64.1%) matches the known S21 result from
`backtest_strategies.py`, which validates the data loading and simulation engine.

---

## File to Review

**`C:/seasonals/ml_scorer/backtest_l1_directions.py`** (complete file, ~450 lines)

Also reference:
- `backtest_strategies.py` lines 574-657 (original exit logic) and lines 664-884 (original simulation loop) for comparison
- `results/backtester_input_10_30.parquet` schema: `date, year, symbol, sector, direction, holding_days, ml_score, predicted_return, predicted_mfe, win_probability, p_hit_return, p_hit_mfe, actual_return, actual_mfe, stock_volatility_20d, atr_14d_pct`

---

## Key Areas to Investigate

### 1. Short P&L calculation (CRITICAL)

```python
# In run_strategy(), daily P&L computation:
price_ratio = price_series[today] / pos["entry_price"]
cr = price_ratio - 1.0 if pos["direction"] == "l" else 1.0 - price_ratio
```

Verify:
- Is `1.0 - price_ratio` the correct formula for a short position's daily P&L?
  - If stock goes from $100 to $95: `price_ratio = 0.95`, short `cr = 0.05`. Correct?
  - If stock goes from $100 to $105: `price_ratio = 1.05`, short `cr = -0.05`. Correct?
- Is this consistent with how `actual_return` is stored in the parquet for shorts?
  - The parquet `actual_return` for shorts should be `(entry - exit) / entry * 100` (positive = stock fell = short won)
  - Does the simulation's realized P&L match the parquet's `actual_return` sign convention?
- Is slippage applied correctly for shorts? `slipped = exit_ret - SLIPPAGE` -- for shorts, does slippage reduce gains the same way as longs?

### 2. EP exit logic with direction-adjusted returns (HIGH)

```python
def check_exit(pos, cum_return, trading_day_idx):
    if cum_return > pos["hwm"]:
        pos["hwm"] = cum_return
    stop = pos["hwm"] - EP_TRAIL_PCT  # 3%
    if cum_return <= stop and trading_day_idx >= 2:
        return ("exit", stop, "pct_trail")
    if is_last_day:
        return ("exit", cum_return, "hold_expiry")
```

For shorts, `cum_return` is now direction-adjusted (positive = short is winning = stock is falling).

Verify:
- Does HWM tracking work correctly for shorts? HWM should represent the best P&L achieved, regardless of direction.
- Does the trailing stop fire correctly? For a short that moves from 0% to +7% then back to +4%, stop = 7% - 3% = 4%, should trigger. Does it?
- Is there any edge case where a short position that has never been profitable (cum_return always negative) incorrectly triggers the EP trail? `stop = 0.0 - 0.03 = -0.03`. If `cum_return = -0.01`, is that >= -0.03, so no exit? Correct.
- When the EP trailing stop triggers, the exit return is `stop`, not `cum_return`. For a short, is `stop` the correct exit price? Or should the exit return be the actual current cum_return if it's worse than the stop?

### 3. Hard stop for shorts (HIGH)

```python
pos_loss = pos["allocation"] * max(0.0, -cum_return)
if pos_loss >= sod_equity * HARD_STOP_PCT:
    slipped = cum_return - SLIPPAGE
    cash += pos["allocation"] + pnl
```

For a short where `cum_return = -0.15` (stock rallied 15% against the short):
- `pos_loss = allocation * 0.15`. Triggers at 10% of equity. Correct.
- `slipped = -0.15 - 0.002 = -0.152`. Cash receives `allocation * (1 + (-0.152)) = allocation * 0.848`. Correct?

Verify the hard stop triggers and cash accounting is correct for short positions.

### 4. Kelly R direction-aware computation (MEDIUM)

```python
for dir_ in directions:
    dir_df = df[df["direction"] == dir_]
    for thresh in thresholds:
        filtered = dir_df[dir_df["ml_score"] >= thresh]
        for year in years:
            prior = filtered[filtered["year"] < year]
            wins = prior.loc[prior["actual_return"] > 0, "actual_return"]
            losses = prior.loc[prior["actual_return"] <= 0, "actual_return"].abs()
            R = wins.mean() / losses.mean()
            table[(dir_, thresh, year)] = R
```

Since `actual_return` is direction-adjusted (positive = win for either direction), `wins` and
`losses` should be correctly identified for both longs and shorts.

Verify:
- Is this the correct interpretation? For shorts, `actual_return > 0` means the stock fell (short won). Is this what the parquet stores?
- The Kelly R from the sample output shows: `('l', 85, 2018): 1.353, ('l', 85, 2019): 0.95`. R < 1 in 2019 means average losses exceeded average wins -- this would give very small Kelly fractions. Does this lead to near-zero position sizing in 2019, and is that reasonable?
- Is the Kelly R table keyed correctly? In `run_strategy()`: `kelly_r = kelly_r_table.get((dir_, THRESHOLD, year), 1.3)`. Matches the table keys `(dir_, thresh, year)`. Correct?

### 5. drawdown_halt cash accounting (MEDIUM)

```python
if drawdown >= DRAWDOWN_HALT_PCT and halt_days_remaining == 0:
    for pos in open_positions:
        cr = pos_returns.get(id(pos), 0.0)
        slipped = cr - SLIPPAGE
        pnl = pos["allocation"] * slipped
        cash += pos["allocation"] + pnl
```

For a short position being force-closed during a drawdown halt: `cr` is the direction-adjusted
return (already flipped for shorts). `pnl = allocation * slipped` is the dollar gain/loss.
`cash += allocation + pnl` returns the original allocation plus the P&L.

Verify this is correct for both long and short positions. Specifically: is `pos_returns[id(pos)]`
guaranteed to have been computed before the drawdown check? (It is computed in the first loop of
the day, before the drawdown check. Verify this ordering is preserved.)

### 6. Combined L+S portfolio dynamics (MEDIUM)

In the `"both"` direction mode, longs and shorts compete for the same 3 position slots.

Verify:
- Can the same symbol appear as both a long and a short on the same day? If so, what happens?
  The parquet has both `(AAPL, l)` and `(AAPL, s)` entries. If both pass filters and rank in
  the top 3, they'd both be entered. Is this intentional? Is the `symbol_open` deduplication
  present in the original code also needed here?
  - The original `backtest_strategies.py` does NOT have a symbol deduplication check.
  - The V3 script (`strategy_backtest_v3.py`) has `symbol_open: set[str]` to prevent same-symbol entries.
  - The new script does NOT have this check. Is having both a long and short on AAPL simultaneously correct trading behavior, or should same-symbol be deduplicated?

### 7. `pos_returns` dict keyed by `id(pos)` (LOW)

```python
pos_returns = {}
for pos in open_positions:
    ...
    pos_returns[id(pos)] = cr

# Later:
invested = sum(pos["allocation"] * (1 + pos_returns.get(id(pos), 0.0))
               for pos in open_positions)
```

`id(pos)` is the memory address of the dict object. Since `open_positions` is rebuilt each day
(`still_open` replaces it), the same dict objects persist with the same `id()`. However, if
Python garbage-collects a closed position and a new position happens to get the same `id()`,
there could be a key collision. In practice this is unlikely within a single simulation run, but
verify it cannot cause incorrect P&L lookups.

### 8. End-of-simulation position closing (LOW)

```python
if open_positions:
    last_day = trading_days[-1]
    for pos in open_positions:
        price_series = prices.get(pos["symbol"])
        if price_series is not None and last_day in price_series.index:
            price_ratio = price_series[last_day] / pos["entry_price"]
            cr = price_ratio - 1.0 if pos["direction"] == "l" else 1.0 - price_ratio
        else:
            cr = 0.0
```

Verify the end-of-sim close uses the same direction-aware formula as the daily loop. It does --
but confirm the `else: cr = 0.0` case (missing price data) is acceptable for short positions.

### 9. Equity metric computation in compute_metrics (LOW)

```python
total_return = (final_equity / INITIAL_CAPITAL) - 1
n_years = 8.0
cagr = (1 + total_return) ** (1 / n_years) - 1
```

The simulation runs 2018-2025 = 8 calendar years. `n_years = 8.0` is hardcoded.
- Verify this is the correct denominator for the annualization. The first trading day is
  2018-01-02 and last is 2025-12-31, which is approximately 8.0 years. Correct.
- Sharpe computation uses monthly returns. Verify `eq_df.groupby("month")["equity"].last()`
  correctly captures month-end equity. If the last trading day of a month is not in `equity_records`,
  the groupby will use the last available day of that month, which is correct.

### 10. SkipMonday filter correctness (LOW)

```python
is_monday = (today.weekday() == 0)
...
if skip_monday and is_monday:
    continue
```

`today` is a `datetime.date` object. `date.weekday()` returns 0 for Monday. Verify this is
correct -- Python's `date.weekday()` returns 0=Monday, 1=Tuesday, ..., 6=Sunday. The check
correctly skips entries (but not exits) on Mondays.

---

## Known Intentional Design Decisions (Do NOT Flag These)

1. **`actual_return` in the parquet is direction-adjusted.** For shorts, `actual_return > 0`
   means the trade won (stock fell). For longs, `actual_return > 0` means the stock rose.
   Kelly R computation uses this convention correctly.

2. **`THRESHOLD = 85` is hardcoded.** This is the S21 config parameter. Not configurable
   per-strategy intentionally -- all 6 strategies share the same ML threshold.

3. **Kelly R only computed for threshold=85.** All strategies use T85. No need for a full
   threshold table.

4. **No VIX block.** The original S21 system did not have a VIX block. This is intentional
   for fair comparison with the known S21 result.

5. **No symbol deduplication within a day (by design, for now).** The V3 system has it;
   the original doesn't. If Codex identifies this as a bug rather than a design gap, note it
   but do not assume it must match V3.

6. **Slippage applied as `exit_ret - SLIPPAGE` (0.2% round-trip).** Same as the original.
   For shorts, slippage is symmetric -- costs the same entering or exiting.

7. **Long-only baseline matches original S21 result: Sharpe 3.66, CAGR 64.1%.** This is the
   primary validation that the simulation engine is correct. Any review should note whether
   the short-side changes could have inadvertently affected the long-only path.

---

## Primary Question for Codex

**Is the short-side simulation mathematically correct?**

Specifically: given that `actual_return` is direction-adjusted in the parquet, and the simulation
flips the raw price return for short positions, do the two approaches stay consistent? Would a
simulation run where all actual_returns match perfectly (zero model error) produce a flat equity
curve for each trade, or would the sign convention cause systematic over/under-statement of
short P&L?

The long-only validation (Sharpe 3.66 matches known result) is strong evidence the simulation
engine is correct for longs. But it does NOT validate the short path independently.
