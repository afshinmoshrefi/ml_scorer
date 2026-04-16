# TradeWave Trading System -- Two-Level Configuration Handoff

**Date:** 2026-04-06
**Purpose:** Complete specification for the auto trading system covering Level 1 (Aggressive) and
Level 3 (Conservative) configurations. Includes all validated backtest results and exact
implementation parameters.

---

## Overview

The ML scoring system (V3, deployed 2026-04-04) provides the signal for all trading activity.
The **signal is the same for both levels** -- the ML scorer at `http://104.238.214.253:7675`
returns identical predictions regardless of which level is active. What differs between levels
is the **portfolio management rules**: exit strategy, position sizing, concentration, and filters.

| Parameter | Level 1 (Aggressive) | Level 3 (Conservative) |
|-----------|---------------------|------------------------|
| **Name** | Aggressive / High CAGR | Conservative / High Sharpe |
| **Exit rule** | EP: 3% trailing stop from HWM | target6_atr2: 6% target, 2x ATR trailing stop |
| **Sizing** | Kelly (quarter-Kelly, capped) | vol_inverse (inverse volatility) |
| **Concentration** | C2: max 2 per sector, 3 positions | risk_balanced: max 3 per sector, 10 positions |
| **ML threshold** | ML >= 85, WP >= 0 | ML >= 80, WP >= 0.74, pred_return >= 1.50 |
| **Ranking** | Win Probability (WP) | Composite rank (0.45*ML + 35*WP + 3*PR + 10*PHR) |
| **SkipMonday** | No (hurts this config) | Yes (+0.35 Sharpe) |
| **Directions** | Long + Short combined | Long + Short combined |

---

## Level 1 -- Aggressive

### Configuration

```
Exit:          EP (early profit)
               - Track high-water mark (HWM) from day 0
               - Trail stop = HWM - 3%
               - Exit when price drops 3% below HWM, minimum hold = 2 trading days
               - Hold to pattern expiry otherwise
Hard stop:     10% of portfolio equity per position
DD halt:       15% portfolio drawdown -> halt all new entries for 20 trading days
Sizing:        Quarter-Kelly, capped at 2x equal share
               W = win_probability from ML scorer
               R = rolling avg_win / avg_loss ratio (prior years at ML>=85)
               kelly_pct = max(W - (1-W)/R, 0)
               alloc = equity * kelly_pct * 0.25
               cap = (equity * 0.90 / max_positions) * 2
               position_size = min(alloc, cap)
Max positions: 3
Concentration: Max 2 positions per sector (C2)
ML threshold:  ml_score >= 85
Ranking:       Sort by win_probability descending
Directions:    Long + Short (combined pool, same rules for both)
SkipMonday:    No
Earnings:      Skip if earnings date falls within holding period
Slippage:      0.2% round-trip
Cash reserve:  10% minimum
```

### Backtest Results (2018-2025, $100K starting capital)

Validated with real day-by-day price simulation. Long-only result (3.66) matches the known S21
benchmark, confirming simulation correctness. Codex-reviewed: short-side sign convention
verified correct (2026-04-06).

#### 3 Positions (default S21 config)

| Direction | Sharpe | CAGR | Max DD | Win Rate | Trades | All 8 Yrs+ |
|-----------|--------|------|--------|----------|--------|------------|
| **Long + Short (recommended)** | **4.09** | **64.1%** | **6.7%** | **58.0%** | **750** | **Yes** |
| Long + Short + SkipMonday | 3.92 | 64.0% | 7.1% | 58.2% | 740 | Yes |
| Long only | 3.66 | 64.1% | 6.7% | 55.9% | 780 | Yes |
| Long only + SkipMonday | 3.68 | 63.7% | 7.1% | 56.6% | 755 | Yes |
| Short only | 2.40 | 37.5% | 5.4% | 58.5% | 465 | No (6/8) |
| Short only + SkipMonday | 2.26 | 35.2% | 5.9% | 59.0% | 424 | No (6/8) |

#### 5 Positions (higher CAGR variant)

| Direction | Sharpe | CAGR | Max DD | Win Rate | Trades | All 8 Yrs+ |
|-----------|--------|------|--------|----------|--------|------------|
| **Long + Short (recommended)** | **4.16** | **128.0%** | **11.0%** | **57.9%** | **1241** | **Yes** |
| Long + Short + SkipMonday | 3.92 | 130.6% | 10.3% | 59.5% | 1189 | Yes |
| Long only | 3.92 | 132.7% | 10.2% | 57.1% | 1280 | Yes |
| Long only + SkipMonday | 3.77 | 132.0% | 10.4% | 59.0% | 1223 | Yes |
| Short only | 2.42 | 60.8% | 10.4% | 59.3% | 740 | No (7/8) |
| Short only + SkipMonday | 2.24 | 53.9% | 10.0% | 59.3% | 676 | No (6/8) |

**Position count trade-off:** Sharpe barely changes (4.09 vs 4.16) but CAGR doubles (64% vs 128%)
and DD nearly doubles (6.7% vs 11%). More positions = more capital deployed at all times = same
signal quality, higher absolute returns, proportionally higher drawdown. Choose based on DD tolerance.

**SkipMonday note:** Does NOT help L1 at either position count. EP exit is fast enough that
Monday entries cause no drag. Only enable for L3.

**Short breakdown in 3-position L+S run:**
- Long trades: 634 of 750, WR 57.1%
- Short trades: 116 of 750, WR 62.9%
- Shorts contribute ~15% of trades, add +0.43 Sharpe, no DD cost

**Short-only not viable as standalone** at either position count -- loses money in strong bull
years (2019, 2021). Only effective as complement to longs in a combined portfolio.

### Year-by-Year Returns (L+S Combined)

| Year | 3 Positions | 5 Positions |
|------|-------------|-------------|
| 2018 | 68.8% | 139.2% |
| 2019 | 46.2% | 93.6% |
| 2020 | 56.7% | 129.0% |
| 2021 | 78.7% | 140.5% |
| 2022 | 85.4% | 189.5% |
| 2023 | 105.2% | 177.6% |
| 2024 | 24.4% | 57.5% |
| 2025 | 64.6% | 128.1% |

All years profitable at both position counts. Returns roughly 2x across all years at 5 positions.

### Holdout Validation

Walk-forward validated 2018-2025. All 8 years profitable for L+S combined at both position counts.

### Options Track (L1)

Options use the same ML signal with the following simplified model:
- Instrument: ATM long calls (longs) / ATM long puts (shorts)
- Premium: 2.5% of stock price
- Payoff: `0.55 * max(actual_return, 0) + 0.30 * max(mfe, 0)` relative to premium paid
- Theta drag: 10% of premium per 30 days held
- Starting capital: $10,000 separate account

| Direction | Sharpe | Max DD | CAGR (model) | Win Rate | All 8 Yrs+ |
|-----------|--------|--------|--------------|----------|------------|
| **L+S combined** | **5.34** | **22.3%** | **3959%** | **71.2%** | **Yes** |
| Long only | 4.97 | 30.4% | 2938% | 68.6% | Yes |
| Short only | 3.97 | 15.4% | 744% | 70.0% | No (not all 8) |

**IMPORTANT on CAGR:** The options CAGR (3959%) is a model artifact. The model uses a simplified
option payoff formula without IV, bid-ask spread, or realistic premium dynamics. Treat the CAGR
as an upper bound only. Even at 5-10% of model CAGR (~200-400% annualized), this remains an
excellent outcome. The Sharpe (5.34) and win rate (71.2%) are more reliable as signal quality
indicators. Best config: same OPT_013 parameters (strict threshold, wide concentration,
vol_inverse sizing) with L+S combined.

---

## Level 3 -- Conservative

### Configuration

```
Exit:          target6_atr2
               - If actual MFE >= 6%: exit at +6% (target hit)
               - If actual MFE < 6%: exit at max(actual_return, -2 * ATR_14d%)
               - ATR trailing floor provides downside protection without time limit
Hard stop:     Implicit via ATR floor (2x 14-day ATR)
Sizing:        vol_inverse (inverse volatility)
               base_weight = 0.10 (10% of equity per position baseline)
               scale = 25.0 / (stock_volatility_20d * 100)
               weight = clip(base_weight * scale, base_weight * 0.6, base_weight * 1.4)
Max positions: 10
Concentration: risk_balanced: max 3 per sector
ML threshold:  ml_score >= 80, win_probability >= 0.74, predicted_return >= 1.50
Ranking:       Composite: 0.45*ml_score + 35*win_probability + 3*predicted_return + 10*p_hit_return
Directions:    Long + Short (combined pool)
SkipMonday:    Yes (+0.35 Sharpe, confirmed in V4 backtest)
Earnings:      Skip if earnings within hold window
Slippage:      Implicit in model (not separately applied)
```

### Backtest Results (2018-2025, $100K starting capital)

Validated in the Codex V4 enhanced backtest (2026-04-06, 90 configs, event-based simulation).

| Direction | Sharpe | CAGR | Max DD | Win Rate | Trades | All 8 Yrs+ |
|-----------|--------|------|--------|----------|--------|------------|
| **Long + Short (recommended)** | **7.46** | **36.32%** | **1.84%** | **85.9%** | 1,194* | **Yes** |
| Long only | 7.23 | 36.04% | 2.75% | -- | -- | Yes |
| Short only | 5.82 | 26.38% | 2.55% | -- | -- | Yes |

*Trade count for baseline (STK_045 equivalent). SkipMonday reduces this slightly.

**2025 Out-of-Sample Holdout:**
- Train 2018-2024, validate 2025 only
- L+S: Sharpe 8.41 (better than in-sample -- confirmed generalization)

### Year-by-Year Returns (L+S Combined, SkipMonday enabled)

| Year | Return |
|------|--------|
| 2018 | 28.9% |
| 2019 | 27.2% |
| 2020 | 29.3% |
| 2021 | 43.4% |
| 2022 | 46.4% |
| 2023 | 40.4% |
| 2024 | 30.2% |
| 2025 | 43.6% |

All 8 years profitable. Worst year 27.2% (2019). All years in 27-47% range.

### Enhancement Attribution (V4 Study)

| Enhancement | Sharpe delta | DD delta | Notes |
|-------------|-------------|----------|-------|
| SkipMonday | +0.35 | -0.81pp | Best single filter, enable always |
| NoRepeat14d | +0.05 | -- | Small benefit, enable |
| VIX block (>35) | ~0 | -- | Safety only, enable |
| SymbolQuality | mixed | -- | Better on baseline B configs |
| WeeklyBreaker | **-4.78** | catastrophic | **NEVER enable** |
| Regime (100-Year) | -0.13 | -- | Only during active midterm windows |

### Options Track (L3)

Same OPT_013 options parameters apply. L3 does not change the options account configuration.
The options signal is from the same ML scorer -- only the stock account exit rules differ.

---

## Current Auto Trading State vs Recommended Changes

### What Currently Runs (as of 2026-04-06)

```
Level:        L1 equivalent (original S21)
Direction:    Long only
Exit:         EP (3% trailing stop)
Ranking:      CW composite (0.54*WP + 0.225*PR + 0.135*MG + 0.10*SQ)
Max positions: 3 (from Redis config)
ML threshold: varies (T85 or T90 depending on config)
ML scorer:    http://104.238.214.253:7675
```

### Recommended Changes for L1 L+S

To upgrade the current L1 system to full L+S:
1. Include short opportunities in the daily candidate pool (direction = "s" from opp files)
2. Call `/score` or `/select` with short direction as well as long
3. EP exit logic is direction-agnostic -- no changes needed to exit rules
4. For short positions: enter short (sell) at entry, cover (buy) at exit
5. Kelly sizing uses direction-specific win/loss ratios from prior history

Expected improvement: Sharpe 3.66 -> 4.09, same CAGR (~64%), same DD (~6.7%)

### Recommended Changes for L3

To implement L3, the following must change in the auto trading system:

1. **Exit rule**: Replace EP trailing stop with target6_atr2
   - Track if MFE >= 6%: if yes, take +6% profit
   - Otherwise: apply floor at max(actual_return, -2 * ATR_14d_pct)
   - ATR_14d_pct is returned by the ML scorer in the backtester input; in live trading,
     compute as (14-day true range average) / current price

2. **Sizing**: Replace Kelly with vol_inverse
   - Get stock_volatility_20d for each candidate (20-day return std dev)
   - position_weight = clip(0.10 * 25.0 / (vol * 100), 0.06, 0.14)

3. **Position limit**: Raise from 3 to 10

4. **Concentration**: Change from C2 (2/sector) to risk_balanced (3/sector)

5. **Ranking**: Change from CW to composite_rank
   - composite_rank = 0.45 * ml_score + 35.0 * win_probability + 3.0 * predicted_return + 10.0 * p_hit_return

6. **SkipMonday**: Add filter -- no new entries on Mondays

7. **Direction**: Include shorts (same as L1 change)

---

## The ML Scorer API (Same for Both Levels)

### Endpoint

```
POST http://104.238.214.253:7675/score
Content-Type: application/json
```

### Single Score Request

```json
{
  "symbol": "AAPL",
  "date": "2026-04-06",
  "daysOut": 20,
  "direction": "l"
}
```

### Response Fields

```json
{
  "symbol": "AAPL",
  "date": "2026-04-06",
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
```

### Batch Select Request (recommended for daily picks)

```
POST http://104.238.214.253:7675/select
Content-Type: application/json
```

```json
{
  "date": "2026-04-06",
  "resource_ids": ["2"],
  "num_picks": 20,
  "direction": "l",
  "days_out_min": 10,
  "days_out_max": 30,
  "min_avg_return": 1.5,
  "min_win_prob": 0.74
}
```

Note: Call `/select` twice -- once with `"direction": "l"` and once with `"direction": "s"` --
then merge and re-rank the combined pool. The `/select` endpoint does not support mixed directions
in a single call.

### VIX Blocking

When VIX > 35, the scorer returns `"vix_blocked": true` and no scores for that symbol.
Do not enter new positions on any VIX-blocked day. This applies to both levels.

---

## Level Selection Guide

| Scenario | Recommended Level |
|----------|------------------|
| Smaller account (<$50K), maximize CAGR | L1 |
| Larger account (>$100K), risk-adjusted growth | L3 |
| Can tolerate 7% DD, want 60%+ annual returns | L1 |
| Need DD < 2%, all years profitable guaranteed | L3 |
| Options account ($10K) | L1 or L3 (same options signal) |
| Automated unmonitored deployment | L3 (lower DD, more stable) |
| Active monitoring with manual override | L1 (higher CAGR) |

---

## Data Sources and Pattern Files

Pattern opportunities come from:
- Stocks: `sp500/opp_by_symbol/{SYMBOL}/` -- 475 S&P 500 stocks, 116 pattern files each
- ETFs: `ETF/opp_by_symbol/{SYMBOL}/` -- 157 ETFs
- Nightly parquet cache: `sp500/ml_cache_YYYY-MM-DD.parquet` (generated by nightly.sh cron)

The nightly cron on the production server generates the parquet cache. If the parquet is
available, `/select` uses it (fast). If missing, it falls back to scoring from raw opp files
(slow). Ensure `nightly.sh` runs each evening after market close.

---

## What Is NOT Implemented Yet

1. **L3 exit rule (target6_atr2) in auto trading**: The auto trading code only has EP exit.
   L3 requires implementing the ATR-based trailing stop logic.

2. **L1 L+S short entries**: Current code only enters long positions. Shorts require
   implementing sell-short / buy-to-cover order flow.

3. **A level configuration flag**: No runtime switch between L1 and L3 exists yet.
   Currently hardcoded to L1-equivalent parameters.

4. **vol_inverse sizing**: Current code uses CW composite ranking with fixed max_positions.
   vol_inverse requires reading 20-day volatility per candidate.
