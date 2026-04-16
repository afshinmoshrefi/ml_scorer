"""
L1 Aggressive Strategy -- Direction Study
Tests the S21 config (WP/EP/T85/SK/3/C2) in all three directions:
  - Long only
  - Short only
  - Long + Short combined

And with SkipMonday filter applied (best single enhancement from V4 study).

6 configs total. Uses the same day-by-day simulation engine as backtest_strategies.py
with direction-aware position tracking (shorts flip the price return sign).

Usage:
    python backtest_l1_directions.py
    python backtest_l1_directions.py --jobs 6
"""

import argparse
import json
import time
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Constants (same as backtest_strategies.py)
# ============================================================

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATA_DIR = Path("C:/seasonals/data")
OUT_DIR = RESULTS / "backtest_l1"

INITIAL_CAPITAL = 100_000
SLIPPAGE = 0.002
CASH_RESERVE = 0.10
HARD_STOP_PCT = 0.10
DRAWDOWN_HALT_PCT = 0.15
DRAWDOWN_HALT_DAYS = 20
EP_TRAIL_PCT = 0.03


# ============================================================
# Strategy Definitions
# S21 = WP / EP / T85 / SK / 3 positions / C2
# Three directions x two filter variants = 6 configs
# ============================================================

STRATEGIES = [
    {"id": "L1_base_L",    "direction_filter": "l",    "skip_monday": False, "max_positions": 3},
    {"id": "L1_base_S",    "direction_filter": "s",    "skip_monday": False, "max_positions": 3},
    {"id": "L1_base_LS",   "direction_filter": "both", "skip_monday": False, "max_positions": 3},
    {"id": "L1_skip_L",    "direction_filter": "l",    "skip_monday": True,  "max_positions": 3},
    {"id": "L1_skip_S",    "direction_filter": "s",    "skip_monday": True,  "max_positions": 3},
    {"id": "L1_skip_LS",   "direction_filter": "both", "skip_monday": True,  "max_positions": 3},
    {"id": "L1P5_base_L",  "direction_filter": "l",    "skip_monday": False, "max_positions": 5},
    {"id": "L1P5_base_S",  "direction_filter": "s",    "skip_monday": False, "max_positions": 5},
    {"id": "L1P5_base_LS", "direction_filter": "both", "skip_monday": False, "max_positions": 5},
    {"id": "L1P5_skip_L",  "direction_filter": "l",    "skip_monday": True,  "max_positions": 5},
    {"id": "L1P5_skip_S",  "direction_filter": "s",    "skip_monday": True,  "max_positions": 5},
    {"id": "L1P5_skip_LS", "direction_filter": "both", "skip_monday": True,  "max_positions": 5},
]

# Fixed S21 parameters
RANKING    = "WP"
THRESHOLD  = 85
SIZING     = "SK"
CONC_RULE  = "C2"
EXIT_RULE  = "EP"


# ============================================================
# Data Loading
# ============================================================

def load_data():
    path = RESULTS / "backtester_input_10_30.parquet"
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows -- longs: {(df['direction']=='l').sum():,}, "
          f"shorts: {(df['direction']=='s').sum():,}")

    for col in ["predicted_return", "predicted_mfe", "actual_return", "actual_mfe"]:
        df[col] = df[col] / 100.0

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_prices(symbols):
    print(f"Loading prices for {len(symbols)} symbols...")
    prices = {}
    missing = []
    for sym in symbols:
        csv_path = DATA_DIR / "csv" / "US" / f"{sym}.csv"
        if not csv_path.exists():
            missing.append(sym)
            continue
        try:
            p = pd.read_csv(csv_path, usecols=["date", "close"], parse_dates=["date"])
            p = p[(p["date"] >= "2017-01-01") & (p["date"] <= "2026-12-31")]
            p = p.set_index("date").sort_index()
            p.index = p.index.date
            prices[sym] = p["close"]
        except Exception:
            missing.append(sym)
    if missing:
        print(f"  WARNING: {len(missing)} symbols missing")
    print(f"  Loaded {len(prices)} symbols")
    return prices


def build_trading_days(prices):
    from datetime import date
    all_dates = set()
    for s in prices.values():
        all_dates.update(s.index)
    days = sorted(d for d in all_dates if date(2018, 1, 1) <= d <= date(2025, 12, 31))
    print(f"  {len(days)} trading days from {days[0]} to {days[-1]}")
    return days


def load_earnings():
    path = RESULTS / "earnings_dates.json"
    if not path.exists():
        return {}
    from datetime import date
    with open(path) as f:
        raw = json.load(f)
    earnings = {}
    for sym, dates in raw.items():
        earnings[sym] = set()
        for d in dates:
            try:
                y, m, day = d.split("-")
                earnings[sym].add(date(int(y), int(m), int(day)))
            except Exception:
                pass
    return earnings


def build_candidates_by_date(df, earnings_map):
    print("  Building candidates-by-date index...")

    if earnings_map:
        from datetime import date as _date
        earnings_ord = {}
        for sym, dates in earnings_map.items():
            if dates:
                earnings_ord[sym] = np.array(sorted(d.toordinal() for d in dates))

        keep_mask = np.ones(len(df), dtype=bool)
        for sym, grp in df.groupby("symbol"):
            earr = earnings_ord.get(sym)
            if earr is None or len(earr) == 0:
                continue
            entry_ords = np.array([d.toordinal() for d in grp["date"].values])
            exit_ords = entry_ords + grp["holding_days"].values.astype(int)
            idx = np.searchsorted(earr, entry_ords, side="left")
            conflict = np.zeros(len(grp), dtype=bool)
            valid = idx < len(earr)
            conflict[valid] = earr[idx[valid]] <= exit_ords[valid]
            valid2 = idx > 0
            conflict[valid2] |= earr[np.clip(idx[valid2] - 1, 0, len(earr) - 1)] >= entry_ords[valid2]
            keep_mask[grp.index.values[conflict]] = False

        n_filtered = (~keep_mask).sum()
        print(f"    Earnings filter removed {n_filtered:,} rows ({n_filtered/len(df):.1%})")
        df = df[keep_mask]

    cols = ["symbol", "sector", "direction", "holding_days", "ml_score",
            "predicted_return", "predicted_mfe", "win_probability",
            "actual_return", "actual_mfe", "stock_volatility_20d",
            "year", "atr_14d_pct"]
    candidates = {}
    for date_val, group in df.groupby("date"):
        candidates[date_val] = group[cols].to_dict("records")
    print(f"  {len(candidates)} candidate dates")
    return candidates


# ============================================================
# Kelly R (direction-aware)
# ============================================================

def precompute_kelly_r(df):
    """Compute rolling win/loss ratio per (direction, threshold, year).
    actual_return is direction-adjusted: positive = win for either direction.
    """
    print("Pre-computing Kelly R table (direction-aware)...")
    table = {}
    thresholds = [85]
    years = list(range(2018, 2026))
    directions = ["l", "s"]

    for dir_ in directions:
        dir_df = df[df["direction"] == dir_]
        for thresh in thresholds:
            filtered = dir_df[dir_df["ml_score"] >= thresh]
            for year in years:
                prior = filtered[filtered["year"] < year]
                if len(prior) < 100:
                    prior = filtered
                wins = prior.loc[prior["actual_return"] > 0, "actual_return"]
                losses = prior.loc[prior["actual_return"] <= 0, "actual_return"].abs()
                if len(wins) > 10 and len(losses) > 10:
                    R = float(wins.mean() / losses.mean())
                else:
                    R = 1.3
                table[(dir_, thresh, year)] = R

    sample = {k: round(v, 3) for k, v in list(table.items())[:4]}
    print(f"  Sample: {sample}")
    return table


# ============================================================
# Exit Rule
# ============================================================

def check_exit(pos, cum_return, trading_day_idx):
    """EP exit: 3% trailing stop from HWM. Exits on last day."""
    if cum_return > pos["hwm"]:
        pos["hwm"] = cum_return

    is_last_day = trading_day_idx >= pos["max_trading_days"]

    stop = pos["hwm"] - EP_TRAIL_PCT
    if cum_return <= stop and trading_day_idx >= 2:
        return ("exit", stop, "pct_trail")

    if is_last_day:
        return ("exit", cum_return, "hold_expiry")

    return ("hold", None, None)


# ============================================================
# Simulation Engine
# ============================================================

def run_strategy(strategy, candidates_by_date, prices, trading_days, kelly_r_table):
    sid = strategy["id"]
    dir_filter = strategy["direction_filter"]
    skip_monday = strategy["skip_monday"]
    max_pos = strategy["max_positions"]

    cash = float(INITIAL_CAPITAL)
    open_positions = []
    trades = []
    equity_records = []
    peak_equity = float(INITIAL_CAPITAL)
    halt_days_remaining = 0

    n_days = len(trading_days)
    day_to_idx = {d: i for i, d in enumerate(trading_days)}

    for day_idx in range(n_days):
        today = trading_days[day_idx]
        year = today.year

        # SkipMonday filter: no new entries on Mondays (weekday 0)
        # Note: we still process exits on Mondays
        is_monday = (today.weekday() == 0)

        # --- Compute start-of-day equity ---
        pos_returns = {}
        sod_invested = 0.0
        for pos in open_positions:
            sym = pos["symbol"]
            price_series = prices.get(sym)
            if price_series is not None and today in price_series.index:
                price_ratio = price_series[today] / pos["entry_price"]
                # Direction-adjusted return
                cr = price_ratio - 1.0 if pos["direction"] == "l" else 1.0 - price_ratio
            else:
                cr = 0.0
            pos_returns[id(pos)] = cr
            sod_invested += pos["allocation"] * (1 + cr)
        sod_equity = cash + sod_invested

        # --- Process exits ---
        still_open = []
        for pos in open_positions:
            cum_return = pos_returns[id(pos)]
            trading_days_held = day_idx - pos["entry_day_idx"]

            # Hard stop
            pos_loss = pos["allocation"] * max(0.0, -cum_return)
            if pos_loss >= sod_equity * HARD_STOP_PCT:
                slipped = cum_return - SLIPPAGE
                pnl = pos["allocation"] * slipped
                cash += pos["allocation"] + pnl
                trades.append(_make_trade(pos, today, slipped, trading_days_held, "hard_stop", sid))
                continue

            action, exit_ret, reason = check_exit(pos, cum_return, trading_days_held)
            if action == "exit":
                slipped = exit_ret - SLIPPAGE
                pnl = pos["allocation"] * slipped
                cash += pos["allocation"] + pnl
                trades.append(_make_trade(pos, today, slipped, trading_days_held, reason, sid))
            else:
                still_open.append(pos)

        open_positions = still_open

        # --- Post-exit equity and drawdown check ---
        invested = sum(pos["allocation"] * (1 + pos_returns.get(id(pos), 0.0))
                       for pos in open_positions)
        total_equity = cash + invested

        if total_equity > peak_equity:
            peak_equity = total_equity
        drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0

        if drawdown >= DRAWDOWN_HALT_PCT and halt_days_remaining == 0:
            for pos in open_positions:
                cr = pos_returns.get(id(pos), 0.0)
                slipped = cr - SLIPPAGE
                pnl = pos["allocation"] * slipped
                cash += pos["allocation"] + pnl
                td_held = day_idx - pos["entry_day_idx"]
                trades.append(_make_trade(pos, today, slipped, td_held, "drawdown_halt", sid))
            open_positions = []
            halt_days_remaining = DRAWDOWN_HALT_DAYS
            invested = 0.0
            total_equity = cash
            peak_equity = cash
            drawdown = 0.0  # recompute after liquidation (was stale pre-halt value)

        equity_records.append({
            "date": today,
            "equity": total_equity,
            "drawdown": drawdown,
        })

        if halt_days_remaining > 0:
            halt_days_remaining -= 1
            continue

        # SkipMonday: no new entries
        if skip_monday and is_monday:
            continue

        # --- Get candidates for today ---
        day_candidates = candidates_by_date.get(today, [])
        if not day_candidates:
            continue

        # Direction filter
        if dir_filter == "l":
            day_candidates = [c for c in day_candidates if c["direction"] == "l"]
        elif dir_filter == "s":
            day_candidates = [c for c in day_candidates if c["direction"] == "s"]
        # "both": keep all

        filtered = [c for c in day_candidates if c["ml_score"] >= THRESHOLD]
        if not filtered:
            continue

        available_slots = max_pos - len(open_positions)
        if available_slots <= 0:
            continue

        # --- Rank by win_probability ---
        ranked = sorted(filtered, key=lambda c: c["win_probability"], reverse=True)

        # --- Concentration (C2: max 2 per sector) ---
        sector_count = {}
        for pos in open_positions:
            s = pos["sector"]
            sector_count[s] = sector_count.get(s, 0) + 1

        selected = []
        for c in ranked:
            s = c["sector"]
            if sector_count.get(s, 0) < 2:
                selected.append(c)
                sector_count[s] = sector_count.get(s, 0) + 1
            if len(selected) + len(open_positions) >= max_pos:
                break

        selected = selected[:available_slots]
        if not selected:
            continue

        # --- Kelly sizing (SK: quarter Kelly, capped at 2x equal share) ---
        allocs = []
        for c in selected:
            dir_ = c["direction"]
            kelly_r = kelly_r_table.get((dir_, THRESHOLD, year), 1.3)
            W = c["win_probability"]
            kelly_pct = max(W - (1 - W) / kelly_r, 0.0)
            frac = kelly_pct * 0.25  # quarter Kelly
            alloc = total_equity * frac
            max_alloc = (total_equity * (1 - CASH_RESERVE) / max_pos) * 2
            allocs.append(min(alloc, max_alloc))

        # Cap total new allocation to available headroom
        cap = total_equity * (1 - CASH_RESERVE)
        headroom = cap - invested
        total_new = sum(allocs)
        if headroom <= 0:
            continue
        if total_new > headroom:
            scale = headroom / total_new
            allocs = [a * scale for a in allocs]

        # --- Enter positions ---
        for c, alloc in zip(selected, allocs):
            if alloc <= 0:
                continue
            sym = c["symbol"]
            price_series = prices.get(sym)
            if price_series is None or today not in price_series.index:
                continue
            entry_price = price_series[today]
            if entry_price <= 0:
                continue

            # Compute exit deadline in trading days
            exit_deadline = today + timedelta(days=c["holding_days"])
            exit_day_idx = day_idx
            for j in range(day_idx + 1, min(day_idx + c["holding_days"] + 10, n_days)):
                if trading_days[j] <= exit_deadline:
                    exit_day_idx = j
                else:
                    break
            max_td = max(exit_day_idx - day_idx, 1)

            cash -= alloc
            open_positions.append({
                "symbol": sym,
                "sector": c["sector"],
                "direction": c["direction"],
                "entry_date": today,
                "entry_day_idx": day_idx,
                "entry_price": entry_price,
                "holding_days": c["holding_days"],
                "max_trading_days": max_td,
                "predicted_return": c["predicted_return"],
                "predicted_mfe": c["predicted_mfe"],
                "win_probability": c["win_probability"],
                "ml_score": c["ml_score"],
                "actual_return": c["actual_return"],
                "allocation": alloc,
                "hwm": 0.0,
                "atr_14d_pct": c.get("atr_14d_pct", 0.02) or 0.02,
            })

    # Close remaining positions at end
    if open_positions:
        last_day = trading_days[-1]
        for pos in open_positions:
            price_series = prices.get(pos["symbol"])
            if price_series is not None and last_day in price_series.index:
                price_ratio = price_series[last_day] / pos["entry_price"]
                cr = price_ratio - 1.0 if pos["direction"] == "l" else 1.0 - price_ratio
            else:
                cr = 0.0
            slipped = cr - SLIPPAGE
            pnl = pos["allocation"] * slipped
            cash += pos["allocation"] + pnl
            td_held = len(trading_days) - 1 - pos["entry_day_idx"]
            trades.append(_make_trade(pos, last_day, slipped, td_held, "end_of_sim", sid))

    return trades, equity_records


def _make_trade(pos, exit_date, slipped_return, days_held, reason, strategy_id):
    return {
        "strategy_id": strategy_id,
        "symbol": pos["symbol"],
        "sector": pos["sector"],
        "direction": pos["direction"],
        "entry_date": pos["entry_date"],
        "exit_date": exit_date,
        "holding_days_cal": pos["holding_days"],
        "days_held": days_held,
        "ml_score": pos["ml_score"],
        "predicted_return": pos["predicted_return"],
        "predicted_mfe": pos["predicted_mfe"],
        "win_probability": pos["win_probability"],
        "allocation": pos["allocation"],
        "exit_return": slipped_return,
        "pnl_dollars": pos["allocation"] * slipped_return,
        "pnl_pct": slipped_return,
        "exit_reason": reason,
        "hwm": pos["hwm"],
    }


# ============================================================
# Metrics
# ============================================================

def compute_metrics(trades, equity_records, strategy):
    sid = strategy["id"]
    if not trades:
        return {"strategy_id": sid, "sharpe_ratio": 0, "annualized_return": 0,
                "max_drawdown": 0, "win_rate": 0, "total_trades": 0,
                "years_profitable": 0, "cagr": 0,
                **{f"year_{y}": 0 for y in range(2018, 2026)}}

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_records)

    total_trades = len(trades_df)
    win_rate = (trades_df["pnl_pct"] > 0).mean()

    final_equity = eq_df["equity"].iloc[-1]
    total_return = (final_equity / INITIAL_CAPITAL) - 1
    n_years = 8.0
    cagr = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 else -1

    eq_vals = eq_df["equity"].values
    peak = np.maximum.accumulate(eq_vals)
    dd = (peak - eq_vals) / peak
    max_dd = float(dd.max())

    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df["month"] = eq_df["date"].dt.to_period("M")
    monthly = eq_df.groupby("month")["equity"].last()
    monthly_returns = monthly.pct_change().dropna()
    if len(monthly_returns) > 1 and monthly_returns.std() > 0:
        sharpe = float((monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12))
    else:
        sharpe = 0.0

    year_returns = {}
    for yr in range(2018, 2026):
        yr_eq = eq_df[eq_df["date"].dt.year == yr]
        if len(yr_eq) >= 2:
            year_returns[yr] = float(yr_eq["equity"].iloc[-1] / yr_eq["equity"].iloc[0] - 1)
        else:
            year_returns[yr] = 0.0
    years_profitable = sum(1 for r in year_returns.values() if r > 0)

    # Direction breakdown
    long_trades = trades_df[trades_df["direction"] == "l"]
    short_trades = trades_df[trades_df["direction"] == "s"]

    return {
        "strategy_id": sid,
        "direction_filter": strategy["direction_filter"],
        "max_positions": strategy["max_positions"],
        "skip_monday": strategy["skip_monday"],
        "sharpe_ratio": sharpe,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_wr": float((long_trades["pnl_pct"] > 0).mean()) if len(long_trades) > 0 else 0,
        "short_wr": float((short_trades["pnl_pct"] > 0).mean()) if len(short_trades) > 0 else 0,
        "years_profitable": years_profitable,
        "final_equity": float(final_equity),
        **{f"year_{yr}": year_returns.get(yr, 0) for yr in range(2018, 2026)},
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated strategy IDs to run (default: all)")
    args = parser.parse_args()

    t0 = time.time()

    df = load_data()
    symbols = df["symbol"].unique().tolist()
    prices = load_prices(symbols)
    trading_days = build_trading_days(prices)
    earnings_map = load_earnings()
    candidates_by_date = build_candidates_by_date(df, earnings_map)
    kelly_r_table = precompute_kelly_r(df)

    print(f"\nData loaded in {time.time()-t0:.1f}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    strategies = STRATEGIES
    if args.ids:
        ids = set(args.ids.split(","))
        strategies = [s for s in STRATEGIES if s["id"] in ids]

    print(f"\nRunning {len(strategies)} strategies ({args.jobs} workers)...")
    t1 = time.time()

    def _run_one(strat):
        trades, eq = run_strategy(strat, candidates_by_date, prices, trading_days, kelly_r_table)
        metrics = compute_metrics(trades, eq, strat)
        return strat["id"], trades, eq, metrics

    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_run_one)(s) for s in strategies
    )

    print(f"\nCompleted in {time.time()-t1:.1f}s")

    all_trades = []
    all_equity = []
    all_metrics = []

    for sid, trades, eq, metrics in results:
        all_trades.extend(trades)
        for rec in eq:
            rec["strategy_id"] = sid
        all_equity.extend(eq)
        all_metrics.append(metrics)

    # Save outputs
    pd.DataFrame(all_trades).to_csv(OUT_DIR / "trades.csv", index=False)
    pd.DataFrame(all_equity).to_csv(OUT_DIR / "equity.csv", index=False)

    summary_df = pd.DataFrame(all_metrics).sort_values("sharpe_ratio", ascending=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    # ---- Print results ----
    print("\n" + "=" * 110)
    print("L1 AGGRESSIVE (S21: WP/EP/T85/SK/3/C2) -- DIRECTION STUDY")
    print("=" * 110)

    display_cols = ["strategy_id", "direction_filter", "max_positions", "skip_monday",
                    "sharpe_ratio", "cagr", "max_drawdown", "win_rate",
                    "total_trades", "long_trades", "short_trades",
                    "long_wr", "short_wr", "years_profitable"]
    disp = summary_df[display_cols].copy()
    disp["sharpe_ratio"] = disp["sharpe_ratio"].map(lambda x: f"{x:.2f}")
    disp["cagr"] = disp["cagr"].map(lambda x: f"{x:.1%}")
    disp["max_drawdown"] = disp["max_drawdown"].map(lambda x: f"{x:.1%}")
    disp["win_rate"] = disp["win_rate"].map(lambda x: f"{x:.1%}")
    disp["long_wr"] = disp["long_wr"].map(lambda x: f"{x:.1%}")
    disp["short_wr"] = disp["short_wr"].map(lambda x: f"{x:.1%}")
    print(disp.to_string(index=False))

    print("\n" + "=" * 110)
    print("YEAR-BY-YEAR RETURNS")
    print("=" * 110)
    yr_cols = [f"year_{y}" for y in range(2018, 2026)]
    yr_disp = summary_df[["strategy_id"] + yr_cols].copy()
    for c in yr_cols:
        yr_disp[c] = yr_disp[c].map(lambda x: f"{x:.1%}")
    print(yr_disp.to_string(index=False))

    print(f"\nResults saved to {OUT_DIR}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
