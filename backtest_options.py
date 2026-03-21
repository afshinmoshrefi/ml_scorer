"""
Options Strategy Playbook V2 Backtester

116 options strategy variants using synthetic options P&L model.
No historical options data required -- approximates P&L from stock returns.

Usage:
    python backtest_options.py                    # Run all 116 strategies
    python backtest_options.py --strategies 1,2,3 # Run specific strategies
    python backtest_options.py --jobs 12          # Parallel workers
"""

import argparse
import hashlib
import json
import math
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
# Constants
# ============================================================

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATA_DIR = Path("C:/seasonals/data")
BACKTEST_DIR = RESULTS / "backtest_options"

INITIAL_CAPITAL = 10_000  # $10K options account
SLIPPAGE_STOCK = 0.002    # Stock-level slippage for comparisons only
MAX_PREMIUM_PCT = 0.10    # Max 10% of capital per trade in premium
MAX_PREMIUM_HARD = 0.15   # Hard cap 15% per position
MAX_TOTAL_PREMIUM = 0.50  # Max 50% of capital in total premium
VIX_QUARTER_KELLY = 28    # VIX > 28: quarter-Kelly only
DRAWDOWN_HALT_PCT = 0.25  # 25% drawdown halts (more lenient for options)
DRAWDOWN_HALT_DAYS = 20

# ============================================================
# Synthetic Options Model Parameters
# ============================================================

STRIKE_PARAMS = {
    # code: (delta, leverage, time_value_pct, spread_cost, premium_pct_of_stock)
    "D30": (0.30, 12, 0.85, 0.025, 0.025),
    "D40": (0.40, 11, 0.70, 0.015, 0.036),
    "D50": (0.50, 11, 0.55, 0.015, 0.045),
    "D60": (0.60, 11, 0.40, 0.015, 0.055),
}

EXPIRY_PARAMS = {
    # code: (dte_func, premium_multiplier)
    # dte_func takes holding_days and returns total DTE
    "XS": (lambda h: h + 3, 1.0),
    "XN": (lambda h: h + 10, 1.3),
    "XB": (lambda h: h + 7, 1.5),
    "X2": (lambda h: h * 2, 1.8),
}

IV_PARAMS = {
    # code: (premium_adjustment, pass_probability)
    "IN": (1.10, 1.00),
    "IL": (1.00, 0.60),
    "IH": (0.90, 0.30),
}


# ============================================================
# Strategy Definitions (all 116)
# ============================================================

def _s(sid, cat, rank, thresh, exit_, size, pos, strike, expiry, iv):
    return {
        "id": sid, "category": cat, "ranking": rank,
        "threshold": thresh, "exit": exit_, "sizing": size,
        "max_positions": pos, "strike": strike, "expiry": expiry,
        "iv_filter": iv,
    }

STRATEGIES = [
    # Cash Machine (1-30)
    _s(1,  "Cash Machine", "WP", 90, "EC", "SK", 3, "D50", "XN", "IL"),
    _s(2,  "Cash Machine", "WP", 90, "EC", "SH", 3, "D50", "XN", "IL"),
    _s(3,  "Cash Machine", "WP", 90, "EC", "SA", 3, "D50", "XN", "IL"),
    _s(4,  "Cash Machine", "WP", 90, "EC", "SK", 3, "D60", "XN", "IL"),
    _s(5,  "Cash Machine", "WP", 90, "EC", "SH", 3, "D60", "XN", "IL"),
    _s(6,  "Cash Machine", "WP", 90, "EC", "SK", 3, "D50", "XB", "IL"),
    _s(7,  "Cash Machine", "WP", 90, "EC", "SH", 3, "D50", "XB", "IL"),
    _s(8,  "Cash Machine", "WP", 90, "EC", "SA", 3, "D50", "XB", "IL"),
    _s(9,  "Cash Machine", "WP", 85, "EC", "SK", 3, "D50", "XN", "IL"),
    _s(10, "Cash Machine", "WP", 85, "EC", "SH", 3, "D50", "XN", "IL"),
    _s(11, "Cash Machine", "WP", 85, "EC", "SA", 3, "D50", "XN", "IL"),
    _s(12, "Cash Machine", "WP", 85, "EC", "SK", 3, "D60", "XN", "IL"),
    _s(13, "Cash Machine", "WP", 85, "EC", "SH", 3, "D60", "XN", "IL"),
    _s(14, "Cash Machine", "WP", 85, "EC", "SK", 3, "D50", "XB", "IL"),
    _s(15, "Cash Machine", "WP", 85, "EC", "SH", 3, "D50", "XB", "IL"),
    _s(16, "Cash Machine", "CW", 90, "EC", "SK", 3, "D50", "XN", "IL"),
    _s(17, "Cash Machine", "CW", 90, "EC", "SH", 3, "D50", "XN", "IL"),
    _s(18, "Cash Machine", "CW", 90, "EC", "SA", 3, "D50", "XN", "IL"),
    _s(19, "Cash Machine", "CW", 90, "EC", "SK", 3, "D60", "XN", "IL"),
    _s(20, "Cash Machine", "CW", 90, "EC", "SH", 3, "D60", "XB", "IL"),
    _s(21, "Cash Machine", "CW", 85, "EC", "SK", 3, "D50", "XN", "IL"),
    _s(22, "Cash Machine", "CW", 85, "EC", "SH", 3, "D50", "XN", "IL"),
    _s(23, "Cash Machine", "CW", 85, "EC", "SA", 3, "D50", "XN", "IL"),
    _s(24, "Cash Machine", "WP", 90, "ES", "SK", 3, "D50", "XN", "IL"),
    _s(25, "Cash Machine", "WP", 90, "ES", "SH", 3, "D50", "XN", "IL"),
    _s(26, "Cash Machine", "WP", 90, "EC", "SK", 3, "D50", "XN", "IH"),
    _s(27, "Cash Machine", "WP", 90, "EC", "SH", 3, "D50", "XN", "IH"),
    _s(28, "Cash Machine", "CW", 90, "EC", "SK", 3, "D50", "XB", "IH"),
    _s(29, "Cash Machine", "WP", 85, "EC", "SK", 4, "D50", "XN", "IL"),
    _s(30, "Cash Machine", "WP", 85, "EC", "SH", 4, "D50", "XN", "IL"),
    # OTM User Preferred (31-55)
    _s(31, "OTM Preferred", "WP", 85, "EM", "SK", 3, "D40", "XS", "IN"),
    _s(32, "OTM Preferred", "WP", 85, "EM", "SH", 3, "D40", "XS", "IN"),
    _s(33, "OTM Preferred", "WP", 85, "EM", "SA", 3, "D40", "XS", "IN"),
    _s(34, "OTM Preferred", "WP", 85, "EM", "SK", 3, "D40", "XS", "IL"),
    _s(35, "OTM Preferred", "WP", 85, "EM", "SH", 3, "D40", "XS", "IL"),
    _s(36, "OTM Preferred", "WP", 85, "ET", "SK", 3, "D40", "XS", "IN"),
    _s(37, "OTM Preferred", "WP", 85, "ET", "SH", 3, "D40", "XS", "IN"),
    _s(38, "OTM Preferred", "WP", 85, "ET", "SA", 3, "D40", "XS", "IL"),
    _s(39, "OTM Preferred", "WP", 90, "EM", "SK", 3, "D40", "XS", "IN"),
    _s(40, "OTM Preferred", "WP", 90, "EM", "SH", 3, "D40", "XS", "IN"),
    _s(41, "OTM Preferred", "WP", 90, "EM", "SA", 3, "D40", "XS", "IL"),
    _s(42, "OTM Preferred", "WP", 90, "ET", "SK", 3, "D40", "XS", "IL"),
    _s(43, "OTM Preferred", "WP", 90, "ET", "SH", 3, "D40", "XS", "IL"),
    _s(44, "OTM Preferred", "CW", 85, "EM", "SK", 3, "D40", "XS", "IN"),
    _s(45, "OTM Preferred", "CW", 85, "EM", "SH", 3, "D40", "XS", "IN"),
    _s(46, "OTM Preferred", "CW", 85, "EM", "SA", 3, "D40", "XS", "IL"),
    _s(47, "OTM Preferred", "CW", 85, "ET", "SK", 3, "D40", "XS", "IN"),
    _s(48, "OTM Preferred", "CW", 85, "ET", "SH", 3, "D40", "XS", "IL"),
    _s(49, "OTM Preferred", "CW", 90, "EM", "SK", 3, "D40", "XS", "IL"),
    _s(50, "OTM Preferred", "CW", 90, "EM", "SH", 3, "D40", "XS", "IL"),
    _s(51, "OTM Preferred", "WP", 85, "EC", "SK", 3, "D40", "XS", "IN"),
    _s(52, "OTM Preferred", "WP", 85, "EC", "SH", 3, "D40", "XS", "IN"),
    _s(53, "OTM Preferred", "WP", 85, "EC", "SA", 3, "D40", "XS", "IL"),
    _s(54, "OTM Preferred", "WP", 90, "EC", "SK", 3, "D40", "XS", "IL"),
    _s(55, "OTM Preferred", "WP", 90, "EC", "SH", 3, "D40", "XS", "IL"),
    # Strike Study (56-75)
    _s(56, "Strike Study", "WP", 85, "EC", "SH", 3, "D30", "XS", "IL"),
    _s(57, "Strike Study", "WP", 85, "EC", "SH", 3, "D40", "XS", "IL"),
    _s(58, "Strike Study", "WP", 85, "EC", "SH", 3, "D50", "XS", "IL"),
    _s(59, "Strike Study", "WP", 85, "EC", "SH", 3, "D60", "XS", "IL"),
    _s(60, "Strike Study", "WP", 85, "EC", "SH", 3, "D30", "XN", "IL"),
    _s(61, "Strike Study", "WP", 85, "EC", "SH", 3, "D40", "XN", "IL"),
    _s(62, "Strike Study", "WP", 85, "EC", "SH", 3, "D50", "XN", "IL"),
    _s(63, "Strike Study", "WP", 85, "EC", "SH", 3, "D60", "XN", "IL"),
    _s(64, "Strike Study", "WP", 85, "EC", "SH", 3, "D30", "XB", "IL"),
    _s(65, "Strike Study", "WP", 85, "EC", "SH", 3, "D40", "XB", "IL"),
    _s(66, "Strike Study", "WP", 85, "EC", "SH", 3, "D50", "XB", "IL"),
    _s(67, "Strike Study", "WP", 85, "EC", "SH", 3, "D60", "XB", "IL"),
    _s(68, "Strike Study", "WP", 85, "EC", "SH", 3, "D30", "X2", "IL"),
    _s(69, "Strike Study", "WP", 85, "EC", "SH", 3, "D40", "X2", "IL"),
    _s(70, "Strike Study", "WP", 85, "EC", "SH", 3, "D50", "X2", "IL"),
    _s(71, "Strike Study", "WP", 85, "EC", "SH", 3, "D60", "X2", "IL"),
    _s(72, "Strike Study", "WP", 90, "EC", "SH", 3, "D30", "XN", "IL"),
    _s(73, "Strike Study", "WP", 90, "EC", "SH", 3, "D40", "XN", "IL"),
    _s(74, "Strike Study", "WP", 90, "EC", "SH", 3, "D50", "XN", "IL"),
    _s(75, "Strike Study", "WP", 90, "EC", "SH", 3, "D60", "XN", "IL"),
    # IV Filter Study (76-90)
    _s(76, "IV Filter", "WP", 85, "EC", "SH", 3, "D40", "XS", "IN"),
    _s(77, "IV Filter", "WP", 85, "EC", "SH", 3, "D40", "XS", "IL"),
    _s(78, "IV Filter", "WP", 85, "EC", "SH", 3, "D40", "XS", "IH"),
    _s(79, "IV Filter", "WP", 85, "EC", "SH", 3, "D50", "XN", "IN"),
    _s(80, "IV Filter", "WP", 85, "EC", "SH", 3, "D50", "XN", "IL"),
    _s(81, "IV Filter", "WP", 85, "EC", "SH", 3, "D50", "XN", "IH"),
    _s(82, "IV Filter", "WP", 90, "EC", "SH", 3, "D40", "XS", "IN"),
    _s(83, "IV Filter", "WP", 90, "EC", "SH", 3, "D40", "XS", "IL"),
    _s(84, "IV Filter", "WP", 90, "EC", "SH", 3, "D40", "XS", "IH"),
    _s(85, "IV Filter", "WP", 90, "EC", "SH", 3, "D50", "XN", "IN"),
    _s(86, "IV Filter", "WP", 90, "EC", "SH", 3, "D50", "XN", "IL"),
    _s(87, "IV Filter", "WP", 90, "EC", "SH", 3, "D50", "XN", "IH"),
    _s(88, "IV Filter", "CW", 85, "EC", "SA", 3, "D40", "XS", "IN"),
    _s(89, "IV Filter", "CW", 85, "EC", "SA", 3, "D40", "XS", "IL"),
    _s(90, "IV Filter", "CW", 85, "EC", "SA", 3, "D40", "XS", "IH"),
    # Kelly Sizing (91-105)
    _s(91,  "Kelly Sizing", "WP", 85, "EC", "SK", 3, "D40", "XS", "IL"),
    _s(92,  "Kelly Sizing", "WP", 85, "EC", "SH", 3, "D40", "XS", "IL"),
    _s(93,  "Kelly Sizing", "WP", 85, "EC", "SA", 3, "D40", "XS", "IL"),
    _s(94,  "Kelly Sizing", "WP", 85, "EC", "SF", 3, "D40", "XS", "IL"),
    _s(95,  "Kelly Sizing", "WP", 85, "EC", "SK", 3, "D50", "XN", "IL"),
    _s(96,  "Kelly Sizing", "WP", 85, "EC", "SH", 3, "D50", "XN", "IL"),
    _s(97,  "Kelly Sizing", "WP", 85, "EC", "SA", 3, "D50", "XN", "IL"),
    _s(98,  "Kelly Sizing", "WP", 85, "EC", "SF", 3, "D50", "XN", "IL"),
    _s(99,  "Kelly Sizing", "WP", 90, "EC", "SK", 3, "D40", "XS", "IL"),
    _s(100, "Kelly Sizing", "WP", 90, "EC", "SH", 3, "D40", "XS", "IL"),
    _s(101, "Kelly Sizing", "WP", 90, "EC", "SA", 3, "D40", "XS", "IL"),
    _s(102, "Kelly Sizing", "WP", 90, "EC", "SF", 3, "D40", "XS", "IL"),
    _s(103, "Kelly Sizing", "CW", 85, "EC", "SK", 4, "D40", "XS", "IL"),
    _s(104, "Kelly Sizing", "CW", 85, "EC", "SH", 4, "D40", "XS", "IL"),
    _s(105, "Kelly Sizing", "CW", 85, "EC", "SA", 4, "D40", "XS", "IL"),
    # Aggressive (106-116)
    _s(106, "Aggressive", "CR", 85, "EM", "SH", 4, "D30", "XS", "IN"),
    _s(107, "Aggressive", "CR", 85, "EM", "SA", 4, "D30", "XS", "IN"),
    _s(108, "Aggressive", "CR", 85, "EM", "SH", 4, "D40", "XS", "IN"),
    _s(109, "Aggressive", "MG", 85, "EM", "SH", 4, "D30", "XS", "IN"),
    _s(110, "Aggressive", "MG", 85, "EM", "SH", 4, "D40", "XS", "IN"),
    _s(111, "Aggressive", "CR", 85, "ET", "SH", 5, "D30", "XS", "IN"),
    _s(112, "Aggressive", "CR", 85, "ET", "SA", 5, "D40", "XS", "IN"),
    _s(113, "Aggressive", "MG", 85, "ET", "SH", 5, "D30", "XS", "IN"),
    _s(114, "Aggressive", "CR", 85, "EM", "SH", 4, "D30", "XS", "IL"),
    _s(115, "Aggressive", "MG", 85, "EM", "SA", 4, "D30", "XS", "IL"),
    _s(116, "Aggressive", "CR", 85, "ET", "SH", 5, "D40", "XS", "IL"),
]


# ============================================================
# Shared: Data Loading (same as stock backtester)
# ============================================================

def load_backtester_data():
    path = RESULTS / "backtester_input_10_30.parquet"
    print(f"Loading backtester input from {path}...")
    df = pd.read_parquet(path)
    df = df[df["direction"] == "l"].copy()
    print(f"  {len(df):,} long opportunities across {df['year'].nunique()} years")
    for col in ["predicted_return", "predicted_mfe", "actual_return", "actual_mfe"]:
        df[col] = df[col] / 100.0
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_prices(symbols):
    print(f"Loading price data for {len(symbols)} symbols...")
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
    print(f"  Loaded prices for {len(prices)} symbols")
    return prices


def build_trading_days(prices):
    from datetime import date
    all_dates = set()
    for series in prices.values():
        all_dates.update(series.index)
    days = sorted(d for d in all_dates
                  if date(2018, 1, 1) <= d <= date(2025, 12, 31))
    print(f"  {len(days)} trading days from {days[0]} to {days[-1]}")
    return days


def build_candidates_by_date(df):
    print("  Building candidates-by-date index...")
    candidates = {}
    cols = ["symbol", "sector", "holding_days", "ml_score",
            "predicted_return", "predicted_mfe", "win_probability",
            "actual_return", "actual_mfe", "stock_volatility_20d", "year"]
    for date_val, group in df.groupby("date"):
        candidates[date_val] = group[cols].to_dict("records")
    print(f"  {len(candidates)} unique dates with candidates")
    return candidates


def precompute_kelly_r(df):
    print("Pre-computing Kelly R table...")
    table = {}
    for thresh in [70, 80, 85, 90]:
        filtered = df[df["ml_score"] >= thresh]
        for year in range(2018, 2026):
            prior = filtered[filtered["year"] < year]
            if len(prior) < 100:
                prior = filtered
            wins = prior.loc[prior["actual_return"] > 0, "actual_return"]
            losses = prior.loc[prior["actual_return"] <= 0, "actual_return"].abs()
            if len(wins) > 10 and len(losses) > 10:
                R = float(wins.mean() / losses.mean())
            else:
                R = 1.3
            table[(thresh, year)] = R
    return table


# ============================================================
# Shared: Ranking (same as stock backtester)
# ============================================================

def rank_candidates(candidates, method):
    if not candidates:
        return []
    if method == "WP":
        return sorted(candidates, key=lambda c: c["win_probability"], reverse=True)
    elif method == "PR":
        return sorted(candidates, key=lambda c: c["predicted_return"], reverse=True)
    elif method == "MS":
        return sorted(candidates, key=lambda c: c["ml_score"], reverse=True)
    elif method == "MG":
        return sorted(candidates, key=lambda c: c["predicted_mfe"] - c["predicted_return"],
                       reverse=True)
    elif method in ("CW", "CR"):
        n = len(candidates)
        if n <= 1:
            return list(candidates)
        wp = np.array([c["win_probability"] for c in candidates])
        pr = np.array([c["predicted_return"] for c in candidates])
        mg = np.array([c["predicted_mfe"] - c["predicted_return"] for c in candidates])
        wp_r = rankdata(wp, method="average") / n
        pr_r = rankdata(pr, method="average") / n
        mg_r = rankdata(mg, method="average") / n
        if method == "CW":
            scores = 0.60 * wp_r + 0.25 * pr_r + 0.15 * mg_r
        else:
            scores = 0.30 * wp_r + 0.50 * pr_r + 0.20 * mg_r
        order = np.argsort(-scores)
        return [candidates[i] for i in order]
    return list(candidates)


# ============================================================
# IV Filter
# ============================================================

def passes_iv_filter(symbol, date, iv_code):
    """Deterministic IV filter using hash. Returns True if candidate passes."""
    if iv_code == "IN":
        return True
    _, pass_prob = IV_PARAMS[iv_code]
    seed = hashlib.md5(f"{symbol}_{date}_{iv_code}".encode()).hexdigest()
    # Use first 8 hex chars as a number [0, 1)
    val = int(seed[:8], 16) / 0xFFFFFFFF
    return val < pass_prob


# ============================================================
# Synthetic Options P&L Model
# ============================================================

def compute_synthetic_option_pnl(daily_stock_cum_returns, delta, leverage,
                                  time_value_pct, total_dte, spread_cost,
                                  premium_mult, iv_adj):
    """Compute daily synthetic option P&L as % of premium paid.

    Args:
        daily_stock_cum_returns: list of cumulative stock returns per trading day
        delta, leverage, time_value_pct: from STRIKE_PARAMS
        total_dte: from EXPIRY_PARAMS
        spread_cost: from STRIKE_PARAMS
        premium_mult: from EXPIRY_PARAMS
        iv_adj: from IV_PARAMS

    Returns:
        (final_option_pnl, daily_option_values): pnl as fraction of premium,
            and list of cumulative option P&L at each day
    """
    cum_opt_pnl = -spread_cost / 2  # Entry spread cost
    daily_values = []
    prev_stock_cum = 0.0

    for day_idx, stock_cum in enumerate(daily_stock_cum_returns):
        current_dte = total_dte - day_idx
        if current_dte <= 0:
            current_dte = 1

        # Stock move today
        stock_daily_move = stock_cum - prev_stock_cum
        prev_stock_cum = stock_cum

        # Delta P&L (as % of premium)
        delta_pnl = stock_daily_move * delta * leverage

        # Theta cost (sqrt model for acceleration)
        frac_now = math.sqrt(current_dte / total_dte) if total_dte > 0 else 0
        frac_tomorrow = math.sqrt(max(0, current_dte - 1) / total_dte) if total_dte > 0 else 0
        theta_cost = (frac_now - frac_tomorrow) * time_value_pct

        cum_opt_pnl += delta_pnl - theta_cost
        # Floor at -100%
        cum_opt_pnl = max(cum_opt_pnl, -1.0)
        daily_values.append(cum_opt_pnl)

    # Exit spread cost
    cum_opt_pnl -= spread_cost / 2
    cum_opt_pnl = max(cum_opt_pnl, -1.0)

    return cum_opt_pnl, daily_values


# ============================================================
# Options Position Sizing
# ============================================================

def compute_option_size(equity, total_premium_open, max_positions, method,
                        candidate, kelly_r):
    """Compute premium budget for a new options position."""
    # Max premium for this trade
    max_this = equity * MAX_PREMIUM_PCT  # 10% cap
    hard_cap = equity * MAX_PREMIUM_HARD  # 15% hard cap

    # Premium headroom
    headroom = equity * MAX_TOTAL_PREMIUM - total_premium_open
    if headroom <= 0:
        return 0.0

    if method == "SF":
        alloc = (equity * MAX_TOTAL_PREMIUM) / max_positions
    elif method in ("SK", "SH"):
        W = candidate["win_probability"]
        R = kelly_r
        kelly_pct = max(W - (1 - W) / R, 0.0)
        frac = kelly_pct * (0.25 if method == "SK" else 0.50)
        alloc = equity * frac
    elif method == "SA":
        W = candidate["win_probability"]
        R = kelly_r
        kelly_pct = max(W - (1 - W) / R, 0.0)
        ml = candidate["ml_score"]
        if ml >= 90:
            frac = kelly_pct * 0.50
        elif ml >= 80:
            frac = kelly_pct * 0.35
        else:
            frac = kelly_pct * 0.25
        alloc = equity * frac
    else:
        alloc = (equity * MAX_TOTAL_PREMIUM) / max_positions

    # Apply caps
    alloc = min(alloc, max_this, hard_cap, headroom)
    return max(alloc, 0.0)


# ============================================================
# Options Exit Rules
# ============================================================

def check_option_exit(pos, opt_pnl, stock_cum_return, trading_day_idx, exit_rule):
    """Check if options position should exit.

    Args:
        pos: position dict (mutated: stock_hwm, pred_reached, trail_stop)
        opt_pnl: current cumulative option P&L as fraction of premium
        stock_cum_return: current stock cumulative return
        trading_day_idx: trading days since entry
        exit_rule: 'ES', 'EC', 'EM', 'ET'

    Returns: (action, option_return, reason)
    """
    # Update stock high water mark
    if stock_cum_return > pos["stock_hwm"]:
        pos["stock_hwm"] = stock_cum_return

    is_last_day = trading_day_idx >= pos["max_trading_days"]

    if exit_rule == "ES":
        # Premium stop at -40%
        if opt_pnl <= -0.40:
            return ("exit", -0.40, "premium_stop")
        if is_last_day:
            return ("exit", opt_pnl, "expiry")
        return ("hold", None, None)

    elif exit_rule == "EC":
        # Combined: 50% premium stop + MFE trail on stock + theta-aware
        if opt_pnl <= -0.50:
            return ("exit", -0.50, "premium_stop")

        # MFE trailing stop on STOCK price
        pred_ret = pos["predicted_return"]
        if not pos["pred_reached"] and stock_cum_return >= pred_ret:
            pos["pred_reached"] = True
            pos["trail_stop"] = pred_ret
        if pos["pred_reached"]:
            excess = pos["stock_hwm"] - pred_ret
            pos["trail_stop"] = pred_ret + excess * 0.50
            if stock_cum_return <= pos["trail_stop"]:
                return ("exit", opt_pnl, "mfe_trail")

        # Theta-aware: 5 DTE, exit if not in profit
        if pos.get("current_dte", 999) <= 5 and opt_pnl < 0:
            return ("exit", opt_pnl, "theta_exit")

        if is_last_day:
            return ("exit", opt_pnl, "expiry")
        return ("hold", None, None)

    elif exit_rule == "EM":
        # MFE trailing stop on stock, return = option P&L
        pred_ret = pos["predicted_return"]
        if not pos["pred_reached"] and stock_cum_return >= pred_ret:
            pos["pred_reached"] = True
            pos["trail_stop"] = pred_ret
        if pos["pred_reached"]:
            excess = pos["stock_hwm"] - pred_ret
            pos["trail_stop"] = pred_ret + excess * 0.50
            if stock_cum_return <= pos["trail_stop"]:
                return ("exit", opt_pnl, "mfe_trail")
        if is_last_day:
            return ("exit", opt_pnl, "expiry")
        return ("hold", None, None)

    elif exit_rule == "ET":
        # Time + trailing
        time_limit = int(pos["max_trading_days"] * 0.60)
        if trading_day_idx >= time_limit and not pos["pred_reached"]:
            return ("exit", opt_pnl, "time_limit")
        # Same MFE trail as EM
        pred_ret = pos["predicted_return"]
        if not pos["pred_reached"] and stock_cum_return >= pred_ret:
            pos["pred_reached"] = True
            pos["trail_stop"] = pred_ret
        if pos["pred_reached"]:
            excess = pos["stock_hwm"] - pred_ret
            pos["trail_stop"] = pred_ret + excess * 0.50
            if stock_cum_return <= pos["trail_stop"]:
                return ("exit", opt_pnl, "mfe_trail")
        if is_last_day:
            return ("exit", opt_pnl, "expiry")
        return ("hold", None, None)

    return ("hold", None, None)


# ============================================================
# Simulation Engine
# ============================================================

def run_strategy(strategy, candidates_by_date, prices, trading_days, kelly_r_table):
    sid = strategy["id"]
    threshold = strategy["threshold"]
    exit_rule = strategy["exit"]
    sizing = strategy["sizing"]
    max_pos = strategy["max_positions"]
    ranking_method = strategy["ranking"]
    strike_code = strategy["strike"]
    expiry_code = strategy["expiry"]
    iv_code = strategy["iv_filter"]

    # Get synthetic model parameters
    delta, leverage, tv_pct, spread_cost, prem_pct = STRIKE_PARAMS[strike_code]
    dte_func, prem_mult = EXPIRY_PARAMS[expiry_code]
    iv_adj, _ = IV_PARAMS[iv_code]

    # Portfolio state
    cash = float(INITIAL_CAPITAL)
    open_positions = []
    trades = []
    equity_records = []
    peak_equity = float(INITIAL_CAPITAL)
    halt_days_remaining = 0

    n_days = len(trading_days)

    for day_idx in range(n_days):
        today = trading_days[day_idx]
        year = today.year
        kelly_r = kelly_r_table.get((threshold, year), 1.3)

        # --- Step 0: Compute option P&L for each open position ---
        pos_opt_pnl = {}
        pos_stock_cr = {}
        for pos in open_positions:
            sym = pos["symbol"]
            price_series = prices.get(sym)
            if price_series is not None and today in price_series.index:
                stock_cr = price_series[today] / pos["entry_price"] - 1.0
            else:
                stock_cr = pos.get("last_stock_cr", 0.0)
            pos["last_stock_cr"] = stock_cr
            pos_stock_cr[id(pos)] = stock_cr

            # Compute daily synthetic option P&L
            td_held = day_idx - pos["entry_day_idx"]
            if td_held > 0:
                stock_move = stock_cr - pos.get("prev_stock_cr", 0.0)
                current_dte = pos["total_dte"] - td_held
                if current_dte <= 0:
                    current_dte = 1
                pos["current_dte"] = current_dte

                delta_pnl = stock_move * delta * leverage
                frac_now = math.sqrt(current_dte / pos["total_dte"]) if pos["total_dte"] > 0 else 0
                frac_tom = math.sqrt(max(0, current_dte - 1) / pos["total_dte"]) if pos["total_dte"] > 0 else 0
                theta = (frac_now - frac_tom) * tv_pct

                pos["cum_opt_pnl"] += delta_pnl - theta
                pos["cum_opt_pnl"] = max(pos["cum_opt_pnl"], -1.0)
                pos["theta_total"] += theta

            pos["prev_stock_cr"] = stock_cr
            pos_opt_pnl[id(pos)] = pos["cum_opt_pnl"]

        # Compute total equity
        total_premium_open = sum(p["premium_paid"] for p in open_positions)
        invested = sum(p["premium_paid"] * (1 + pos_opt_pnl.get(id(p), 0)) for p in open_positions)
        total_equity = cash + invested

        if total_equity > peak_equity:
            peak_equity = total_equity

        # --- Step 1: Check exits ---
        still_open = []
        for pos in open_positions:
            opt_pnl = pos_opt_pnl[id(pos)]
            stock_cr = pos_stock_cr[id(pos)]
            td_held = day_idx - pos["entry_day_idx"]

            action, exit_ret, reason = check_option_exit(
                pos, opt_pnl, stock_cr, td_held, exit_rule)

            if action == "exit":
                exit_ret = max(exit_ret, -1.0)
                dollar_pnl = pos["premium_paid"] * exit_ret
                cash += pos["premium_paid"] + dollar_pnl
                trades.append({
                    "strategy_id": sid,
                    "symbol": pos["symbol"],
                    "sector": pos["sector"],
                    "entry_date": pos["entry_date"],
                    "exit_date": today,
                    "holding_days_cal": pos["holding_days"],
                    "days_held": td_held,
                    "ml_score": pos["ml_score"],
                    "predicted_return": pos["predicted_return"],
                    "predicted_mfe": pos["predicted_mfe"],
                    "win_probability": pos["win_probability"],
                    "premium_paid": pos["premium_paid"],
                    "option_return_pct": exit_ret,
                    "pnl_dollars": dollar_pnl,
                    "stock_return_at_exit": stock_cr,
                    "theta_cost_total": pos["theta_total"],
                    "exit_reason": reason,
                    "strike": strike_code,
                    "expiry": expiry_code,
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # Recompute equity after exits
        total_premium_open = sum(p["premium_paid"] for p in open_positions)
        invested = sum(p["premium_paid"] * (1 + p["cum_opt_pnl"]) for p in open_positions)
        total_equity = cash + invested

        # Drawdown check
        drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0
        if drawdown >= DRAWDOWN_HALT_PCT and halt_days_remaining == 0:
            for pos in open_positions:
                opt_ret = pos["cum_opt_pnl"]
                dollar_pnl = pos["premium_paid"] * opt_ret
                cash += pos["premium_paid"] + dollar_pnl
                trades.append({
                    "strategy_id": sid, "symbol": pos["symbol"],
                    "sector": pos["sector"], "entry_date": pos["entry_date"],
                    "exit_date": today, "holding_days_cal": pos["holding_days"],
                    "days_held": day_idx - pos["entry_day_idx"],
                    "ml_score": pos["ml_score"],
                    "predicted_return": pos["predicted_return"],
                    "predicted_mfe": pos["predicted_mfe"],
                    "win_probability": pos["win_probability"],
                    "premium_paid": pos["premium_paid"],
                    "option_return_pct": opt_ret,
                    "pnl_dollars": dollar_pnl,
                    "stock_return_at_exit": pos.get("last_stock_cr", 0),
                    "theta_cost_total": pos["theta_total"],
                    "exit_reason": "drawdown_halt",
                    "strike": strike_code, "expiry": expiry_code,
                })
            open_positions = []
            halt_days_remaining = DRAWDOWN_HALT_DAYS
            total_premium_open = 0
            invested = 0
            total_equity = cash
            peak_equity = cash

        equity_records.append({
            "date": today, "equity": total_equity, "cash": cash,
            "invested": invested, "open_positions": len(open_positions),
            "drawdown": drawdown, "total_premium": total_premium_open,
        })

        if halt_days_remaining > 0:
            halt_days_remaining -= 1
            continue

        # --- Step 2: New entries ---
        day_candidates = candidates_by_date.get(today, [])
        if not day_candidates:
            continue
        filtered = [c for c in day_candidates if c["ml_score"] >= threshold]
        if not filtered:
            continue

        # IV filter
        if iv_code != "IN":
            filtered = [c for c in filtered
                        if passes_iv_filter(c["symbol"], today, iv_code)]
        if not filtered:
            continue

        available_slots = max_pos - len(open_positions)
        if available_slots <= 0:
            continue

        ranked = rank_candidates(filtered, ranking_method)
        selected = ranked[:available_slots]

        for c in selected:
            sym = c["symbol"]
            price_series = prices.get(sym)
            if price_series is None or today not in price_series.index:
                continue
            entry_price = price_series[today]
            if entry_price <= 0:
                continue

            # Compute premium allocation
            total_prem_now = sum(p["premium_paid"] for p in open_positions)
            premium = compute_option_size(
                total_equity, total_prem_now, max_pos, sizing, c, kelly_r)
            if premium <= 0:
                continue

            # Compute DTE and max trading days
            holding_days = c["holding_days"]
            total_dte = dte_func(holding_days)
            exit_deadline = today + timedelta(days=holding_days)
            exit_day_idx = day_idx
            for j in range(day_idx + 1, min(day_idx + holding_days + 10, n_days)):
                if trading_days[j] <= exit_deadline:
                    exit_day_idx = j
                else:
                    break
            max_td = max(exit_day_idx - day_idx, 1)

            cash -= premium
            open_positions.append({
                "symbol": sym,
                "sector": c["sector"],
                "entry_date": today,
                "entry_day_idx": day_idx,
                "entry_price": entry_price,
                "holding_days": holding_days,
                "max_trading_days": max_td,
                "total_dte": total_dte,
                "predicted_return": c["predicted_return"],
                "predicted_mfe": c["predicted_mfe"],
                "win_probability": c["win_probability"],
                "ml_score": c["ml_score"],
                "premium_paid": premium,
                "cum_opt_pnl": -spread_cost / 2,  # Entry spread
                "prev_stock_cr": 0.0,
                "last_stock_cr": 0.0,
                "stock_hwm": 0.0,
                "pred_reached": False,
                "trail_stop": 0.0,
                "theta_total": 0.0,
                "current_dte": total_dte,
            })

    # Close remaining at end
    last_day = trading_days[-1]
    for pos in open_positions:
        opt_ret = pos["cum_opt_pnl"]
        dollar_pnl = pos["premium_paid"] * opt_ret
        cash += pos["premium_paid"] + dollar_pnl
        trades.append({
            "strategy_id": sid, "symbol": pos["symbol"],
            "sector": pos["sector"], "entry_date": pos["entry_date"],
            "exit_date": last_day, "holding_days_cal": pos["holding_days"],
            "days_held": len(trading_days) - 1 - pos["entry_day_idx"],
            "ml_score": pos["ml_score"],
            "predicted_return": pos["predicted_return"],
            "predicted_mfe": pos["predicted_mfe"],
            "win_probability": pos["win_probability"],
            "premium_paid": pos["premium_paid"],
            "option_return_pct": opt_ret,
            "pnl_dollars": dollar_pnl,
            "stock_return_at_exit": pos.get("last_stock_cr", 0),
            "theta_cost_total": pos["theta_total"],
            "exit_reason": "end_of_sim",
            "strike": strike_code, "expiry": expiry_code,
        })

    return trades, equity_records


# ============================================================
# Metrics
# ============================================================

def compute_strategy_metrics(trades, equity_records, strategy):
    sid = strategy["id"]
    if not trades:
        return _empty_metrics(strategy)

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity_records)

    total_trades = len(tdf)
    wins = tdf[tdf["option_return_pct"] > 0]
    losses = tdf[tdf["option_return_pct"] <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0

    avg_win = wins["option_return_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["option_return_pct"].mean() if len(losses) > 0 else 0
    gross_profit = wins["pnl_dollars"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total loss rate (option_return <= -90%)
    pct_total_loss = (tdf["option_return_pct"] <= -0.90).mean()

    # Avg theta cost
    avg_theta = tdf["theta_cost_total"].mean()

    # Options vs stock multiplier
    stock_sum = tdf["stock_return_at_exit"].sum()
    opt_sum = tdf["option_return_pct"].sum()
    multiplier = opt_sum / stock_sum if abs(stock_sum) > 0.001 else 0

    final_equity = edf["equity"].iloc[-1] if len(edf) > 0 else INITIAL_CAPITAL
    total_return = (final_equity / INITIAL_CAPITAL) - 1
    n_years = 8.0
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 else -1

    eq_vals = edf["equity"].values
    peak = np.maximum.accumulate(eq_vals)
    dd = (peak - eq_vals) / peak
    max_dd = dd.max() if len(dd) > 0 else 0

    edf["date"] = pd.to_datetime(edf["date"])
    edf["month"] = edf["date"].dt.to_period("M")
    monthly = edf.groupby("month")["equity"].last()
    monthly_returns = monthly.pct_change().dropna()
    if len(monthly_returns) > 1 and monthly_returns.std() > 0:
        sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
    else:
        sharpe = 0.0

    worst_trade = tdf["option_return_pct"].min()
    streak = max_streak = 0
    for ret in tdf["option_return_pct"]:
        if ret <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    year_returns = {}
    for yr in range(2018, 2026):
        yr_eq = edf[edf["date"].dt.year == yr]
        if len(yr_eq) >= 2:
            yr_ret = yr_eq["equity"].iloc[-1] / yr_eq["equity"].iloc[0] - 1
        else:
            yr_ret = 0.0
        year_returns[yr] = yr_ret

    years_profitable = sum(1 for r in year_returns.values() if r > 0)

    return {
        "strategy_id": sid, "category": strategy["category"],
        "ranking": strategy["ranking"], "threshold": strategy["threshold"],
        "exit": strategy["exit"], "sizing": strategy["sizing"],
        "max_positions": strategy["max_positions"],
        "strike": strategy["strike"], "expiry": strategy["expiry"],
        "iv_filter": strategy["iv_filter"],
        "total_return": total_return, "annualized_return": ann_return,
        "max_drawdown": max_dd, "sharpe_ratio": sharpe,
        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor, "total_trades": total_trades,
        "trades_per_year": total_trades / n_years,
        "worst_single_trade": worst_trade, "longest_losing_streak": max_streak,
        "years_profitable": years_profitable,
        "pct_total_loss": pct_total_loss,
        "avg_theta_cost": avg_theta,
        "options_vs_stock_mult": multiplier,
        "avg_premium_per_trade": tdf["premium_paid"].mean(),
        **{f"year_{yr}": year_returns.get(yr, 0) for yr in range(2018, 2026)},
    }


def _empty_metrics(strategy):
    return {
        "strategy_id": strategy["id"], "category": strategy["category"],
        "ranking": strategy["ranking"], "threshold": strategy["threshold"],
        "exit": strategy["exit"], "sizing": strategy["sizing"],
        "max_positions": strategy["max_positions"],
        "strike": strategy["strike"], "expiry": strategy["expiry"],
        "iv_filter": strategy["iv_filter"],
        "total_return": 0, "annualized_return": 0, "max_drawdown": 0,
        "sharpe_ratio": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
        "profit_factor": 0, "total_trades": 0, "trades_per_year": 0,
        "worst_single_trade": 0, "longest_losing_streak": 0,
        "years_profitable": 0, "pct_total_loss": 0, "avg_theta_cost": 0,
        "options_vs_stock_mult": 0, "avg_premium_per_trade": 0,
        **{f"year_{yr}": 0 for yr in range(2018, 2026)},
    }


def compute_weighted_scores(df):
    df["consistency_score"] = df["years_profitable"] / 8.0
    def normalize(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx - mn > 0 else pd.Series(0.5, index=s.index)
    df["sharpe_norm"] = normalize(df["sharpe_ratio"])
    df["dd_norm"] = 1 - normalize(df["max_drawdown"].abs())
    df["return_norm"] = normalize(df["total_return"])
    df["weighted_score"] = (
        0.40 * df["consistency_score"]
        + 0.25 * df["sharpe_norm"]
        + 0.20 * df["dd_norm"]
        + 0.15 * df["return_norm"]
    )
    return df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Options Strategy Playbook V2 Backtester")
    parser.add_argument("--strategies", type=str, default=None)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()

    t0 = time.time()

    if args.strategies:
        ids = set(int(x) for x in args.strategies.split(","))
        strategies = [s for s in STRATEGIES if s["id"] in ids]
    else:
        strategies = STRATEGIES
    print(f"Running {len(strategies)} options strategies with {args.jobs} workers\n")

    df = load_backtester_data()
    symbols = df["symbol"].unique().tolist()
    prices = load_prices(symbols)
    trading_days = build_trading_days(prices)
    candidates_by_date = build_candidates_by_date(df)
    kelly_r_table = precompute_kelly_r(df)

    print(f"\nData loaded in {time.time()-t0:.1f}s\n")

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Running backtests...")
    t1 = time.time()

    def _run_one(strat):
        trades, eq = run_strategy(strat, candidates_by_date, prices,
                                   trading_days, kelly_r_table)
        metrics = compute_strategy_metrics(trades, eq, strat)
        return strat["id"], trades, eq, metrics

    results = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(_run_one)(s) for s in strategies
    )

    print(f"\nAll backtests completed in {time.time()-t1:.1f}s")

    all_trades = []
    all_equity = []
    all_metrics = []

    for sid, trades, eq, metrics in results:
        all_trades.extend(trades)
        for rec in eq:
            rec["strategy_id"] = sid
        all_equity.extend(eq)
        all_metrics.append(metrics)

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(BACKTEST_DIR / "trades.csv", index=False)
    print(f"\nTrade log: {BACKTEST_DIR / 'trades.csv'} ({len(trades_df):,} trades)")

    equity_df = pd.DataFrame(all_equity)
    equity_df.to_csv(BACKTEST_DIR / "equity.csv", index=False)
    print(f"Equity curves: {BACKTEST_DIR / 'equity.csv'} ({len(equity_df):,} records)")

    summary_df = pd.DataFrame(all_metrics)
    summary_df = compute_weighted_scores(summary_df)
    summary_df = summary_df.sort_values("weighted_score", ascending=False)
    summary_df.to_csv(BACKTEST_DIR / "summary.csv", index=False)
    print(f"Strategy summary: {BACKTEST_DIR / 'summary.csv'}")

    # Print top 20
    print("\n" + "=" * 120)
    print("TOP 20 OPTIONS STRATEGIES BY WEIGHTED SCORE")
    print("=" * 120)
    cols = ["strategy_id", "category", "ranking", "threshold", "exit",
            "sizing", "max_positions", "strike", "expiry", "iv_filter",
            "total_return", "annualized_return", "max_drawdown",
            "sharpe_ratio", "win_rate", "total_trades", "years_profitable",
            "pct_total_loss", "avg_theta_cost", "weighted_score"]
    top20 = summary_df.head(20)[cols].copy()
    top20["total_return"] = top20["total_return"].map(lambda x: f"{x:.1%}")
    top20["annualized_return"] = top20["annualized_return"].map(lambda x: f"{x:.1%}")
    top20["max_drawdown"] = top20["max_drawdown"].map(lambda x: f"{x:.1%}")
    top20["sharpe_ratio"] = top20["sharpe_ratio"].map(lambda x: f"{x:.2f}")
    top20["win_rate"] = top20["win_rate"].map(lambda x: f"{x:.1%}")
    top20["pct_total_loss"] = top20["pct_total_loss"].map(lambda x: f"{x:.1%}")
    top20["avg_theta_cost"] = top20["avg_theta_cost"].map(lambda x: f"{x:.2%}")
    top20["weighted_score"] = top20["weighted_score"].map(lambda x: f"{x:.3f}")
    print(top20.to_string(index=False))

    # Year-by-year for top 10
    print("\n" + "=" * 120)
    print("YEAR-BY-YEAR RETURNS (TOP 10)")
    print("=" * 120)
    top10_ids = summary_df.head(10)["strategy_id"].tolist()
    yr_cols = [f"year_{y}" for y in range(2018, 2026)]
    hm = summary_df[summary_df["strategy_id"].isin(top10_ids)][
        ["strategy_id", "category", "strike", "expiry"] + yr_cols].copy()
    for c in yr_cols:
        hm[c] = hm[c].map(lambda x: f"{x:.1%}")
    print(hm.to_string(index=False))

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
