"""
Options Spread Strategy Backtester

Tests bull call debit spreads, bull put credit spreads, and deep ITM calls
against the same walk-forward validation data.

Synthetic spread P&L model -- no historical options data required.

Usage:
    python backtest_spreads.py [--jobs 12]
"""

import argparse
import hashlib
import time
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATA_DIR = Path("C:/seasonals/data")
BACKTEST_DIR = RESULTS / "backtest_spreads"

INITIAL_CAPITAL = 10_000
MAX_POSITION_PCT = 0.10     # Max 10% of capital per trade
MAX_TOTAL_ALLOC = 0.50      # Max 50% of capital deployed
DRAWDOWN_HALT_PCT = 0.30    # 30% drawdown halt
DRAWDOWN_HALT_DAYS = 20

# ============================================================
# Spread Model Parameters
# ============================================================

# Debit spread: cost_ratio = cost / spread_width
# Based on Black-Scholes for S&P 500 stocks at ~25% IV
DEBIT_PARAMS = {
    # (spread_width_pct, expiry_code): cost_ratio
    (0.03, "XN"): 0.40,  (0.05, "XN"): 0.37,
    (0.03, "XB"): 0.41,  (0.05, "XB"): 0.38,
    (0.03, "X2"): 0.39,  (0.05, "X2"): 0.36,
    (0.03, "XS"): 0.44,  (0.05, "XS"): 0.40,
}

# Credit spread: credit_ratio = credit / spread_width
# Short put OTM distance affects credit significantly
CREDIT_PARAMS = {
    # (short_otm_pct, spread_width_pct, expiry_code): credit_ratio
    (0.03, 0.03, "XS"): 0.28,  (0.03, 0.03, "XN"): 0.25,
    (0.03, 0.05, "XS"): 0.22,  (0.03, 0.05, "XN"): 0.20,
    (0.05, 0.03, "XS"): 0.18,  (0.05, 0.03, "XN"): 0.15,
    (0.05, 0.05, "XS"): 0.12,  (0.05, 0.05, "XN"): 0.10,
    (0.02, 0.03, "XS"): 0.33,  (0.02, 0.03, "XN"): 0.30,
    (0.02, 0.05, "XS"): 0.27,  (0.02, 0.05, "XN"): 0.24,
}

# Deep ITM single-leg (benchmark, same as #71 from options backtest)
DEEP_ITM_PARAMS = {
    "delta": 0.60, "leverage": 11, "tv_pct": 0.40,
    "spread_cost": 0.015, "prem_mult": 1.8,  # X2 expiry
}

EXPIRY_DTE = {"XS": lambda h: h + 3, "XN": lambda h: h + 10,
              "XB": lambda h: h + 7, "X2": lambda h: h * 2}

# ============================================================
# Strategy Definitions
# ============================================================

def _ds(sid, cat, rank, thresh, width, expiry, sizing, pos, exit_rule, iv="IL"):
    """Debit spread strategy."""
    return {"id": sid, "category": cat, "type": "DEBIT", "ranking": rank,
            "threshold": thresh, "spread_width": width, "expiry": expiry,
            "sizing": sizing, "max_positions": pos, "exit": exit_rule,
            "iv_filter": iv}

def _cs(sid, cat, rank, thresh, short_otm, width, expiry, sizing, pos, iv="IL"):
    """Credit spread strategy."""
    return {"id": sid, "category": cat, "type": "CREDIT", "ranking": rank,
            "threshold": thresh, "short_otm": short_otm,
            "spread_width": width, "expiry": expiry, "sizing": sizing,
            "max_positions": pos, "iv_filter": iv}

def _sl(sid, cat, rank, thresh, sizing, pos, iv="IL"):
    """Single-leg deep ITM (benchmark)."""
    return {"id": sid, "category": cat, "type": "SINGLE_LEG", "ranking": rank,
            "threshold": thresh, "sizing": sizing, "max_positions": pos,
            "iv_filter": iv}

STRATEGIES = [
    # === Debit Spreads: Bull Call Spreads ===
    # Tight (3%) spreads -- low breakeven (~1.2%)
    _ds(1,  "DS Tight",   "WP", 85, 0.03, "XN", "SK", 3, "EH"),
    _ds(2,  "DS Tight",   "WP", 85, 0.03, "XN", "SH", 3, "EH"),
    _ds(3,  "DS Tight",   "WP", 85, 0.03, "XB", "SK", 3, "EH"),
    _ds(4,  "DS Tight",   "WP", 85, 0.03, "XB", "SH", 3, "EH"),
    _ds(5,  "DS Tight",   "WP", 85, 0.03, "X2", "SK", 3, "EH"),
    _ds(6,  "DS Tight",   "WP", 85, 0.03, "X2", "SH", 3, "EH"),
    _ds(7,  "DS Tight",   "WP", 90, 0.03, "XN", "SK", 3, "EH"),
    _ds(8,  "DS Tight",   "WP", 90, 0.03, "XB", "SK", 3, "EH"),
    _ds(9,  "DS Tight",   "CW", 85, 0.03, "XN", "SK", 3, "EH"),
    _ds(10, "DS Tight",   "CW", 85, 0.03, "XB", "SH", 3, "EH"),
    _ds(11, "DS Tight",   "CR", 85, 0.03, "XN", "SK", 3, "EH"),
    _ds(12, "DS Tight",   "CR", 85, 0.03, "XN", "SH", 3, "EH"),
    # Wide (5%) spreads -- higher max profit
    _ds(13, "DS Wide",    "WP", 85, 0.05, "XN", "SK", 3, "EH"),
    _ds(14, "DS Wide",    "WP", 85, 0.05, "XN", "SH", 3, "EH"),
    _ds(15, "DS Wide",    "WP", 85, 0.05, "XB", "SK", 3, "EH"),
    _ds(16, "DS Wide",    "WP", 85, 0.05, "XB", "SH", 3, "EH"),
    _ds(17, "DS Wide",    "WP", 85, 0.05, "X2", "SK", 3, "EH"),
    _ds(18, "DS Wide",    "WP", 85, 0.05, "X2", "SH", 3, "EH"),
    _ds(19, "DS Wide",    "WP", 90, 0.05, "XN", "SK", 3, "EH"),
    _ds(20, "DS Wide",    "WP", 90, 0.05, "XB", "SK", 3, "EH"),
    _ds(21, "DS Wide",    "CW", 85, 0.05, "XN", "SK", 3, "EH"),
    _ds(22, "DS Wide",    "CW", 85, 0.05, "XN", "SH", 3, "EH"),
    _ds(23, "DS Wide",    "CR", 85, 0.05, "XN", "SH", 3, "EH"),
    _ds(24, "DS Wide",    "CR", 85, 0.05, "XB", "SH", 3, "EH"),
    # Early exit (exit when stock passes spread width)
    _ds(25, "DS EarlyExit","WP", 85, 0.03, "XN", "SK", 3, "EP"),
    _ds(26, "DS EarlyExit","WP", 85, 0.03, "XB", "SK", 3, "EP"),
    _ds(27, "DS EarlyExit","WP", 85, 0.05, "XN", "SK", 3, "EP"),
    _ds(28, "DS EarlyExit","WP", 85, 0.05, "XB", "SK", 3, "EP"),
    _ds(29, "DS EarlyExit","CW", 85, 0.03, "XN", "SK", 3, "EP"),
    _ds(30, "DS EarlyExit","CW", 85, 0.05, "XN", "SH", 3, "EP"),
    # Position count study
    _ds(31, "DS PosCount", "WP", 85, 0.03, "XN", "SK", 2, "EH"),
    _ds(32, "DS PosCount", "WP", 85, 0.03, "XN", "SK", 4, "EH"),
    _ds(33, "DS PosCount", "WP", 85, 0.03, "XN", "SH", 4, "EH"),
    _ds(34, "DS PosCount", "CW", 85, 0.03, "XN", "SK", 4, "EH"),
    # No IV filter
    _ds(35, "DS NoIV",     "WP", 85, 0.03, "XN", "SK", 3, "EH", iv="IN"),
    _ds(36, "DS NoIV",     "WP", 85, 0.05, "XN", "SK", 3, "EH", iv="IN"),
    # IH (strict IV filter)
    _ds(37, "DS StrictIV", "WP", 85, 0.03, "XN", "SK", 3, "EH", iv="IH"),
    _ds(38, "DS StrictIV", "WP", 85, 0.05, "XN", "SK", 3, "EH", iv="IH"),

    # === Credit Spreads: Bull Put Spreads ===
    # Tight OTM (2-3%), collect more credit
    _cs(39, "CS Tight",   "WP", 85, 0.02, 0.03, "XS", "SK", 3),
    _cs(40, "CS Tight",   "WP", 85, 0.02, 0.03, "XN", "SK", 3),
    _cs(41, "CS Tight",   "WP", 85, 0.02, 0.03, "XS", "SH", 3),
    _cs(42, "CS Tight",   "WP", 85, 0.03, 0.03, "XS", "SK", 3),
    _cs(43, "CS Tight",   "WP", 85, 0.03, 0.03, "XN", "SK", 3),
    _cs(44, "CS Tight",   "WP", 85, 0.03, 0.03, "XS", "SH", 3),
    _cs(45, "CS Tight",   "WP", 90, 0.02, 0.03, "XS", "SK", 3),
    _cs(46, "CS Tight",   "WP", 90, 0.03, 0.03, "XS", "SK", 3),
    _cs(47, "CS Tight",   "CW", 85, 0.02, 0.03, "XS", "SK", 3),
    _cs(48, "CS Tight",   "CW", 85, 0.03, 0.03, "XS", "SK", 3),
    _cs(49, "CS Tight",   "CR", 85, 0.02, 0.03, "XS", "SK", 3),
    _cs(50, "CS Tight",   "CR", 85, 0.03, 0.03, "XS", "SH", 3),
    # Wide OTM (5%), lower credit but higher WR
    _cs(51, "CS Wide",    "WP", 85, 0.05, 0.03, "XS", "SK", 3),
    _cs(52, "CS Wide",    "WP", 85, 0.05, 0.05, "XS", "SK", 3),
    _cs(53, "CS Wide",    "WP", 85, 0.05, 0.03, "XN", "SK", 3),
    _cs(54, "CS Wide",    "WP", 90, 0.05, 0.03, "XS", "SK", 3),
    _cs(55, "CS Wide",    "CW", 85, 0.05, 0.03, "XS", "SK", 3),
    # Credit spread position count
    _cs(56, "CS PosCount", "WP", 85, 0.03, 0.03, "XS", "SK", 4),
    _cs(57, "CS PosCount", "WP", 85, 0.02, 0.03, "XS", "SK", 4),
    _cs(58, "CS PosCount", "CW", 85, 0.03, 0.03, "XS", "SK", 4),
    # Credit spread no IV filter
    _cs(59, "CS NoIV",     "WP", 85, 0.03, 0.03, "XS", "SK", 3, iv="IN"),
    _cs(60, "CS NoIV",     "WP", 85, 0.02, 0.03, "XS", "SK", 3, iv="IN"),

    # === Single-Leg Deep ITM Benchmarks ===
    _sl(61, "Deep ITM",   "WP", 85, "SK", 3),
    _sl(62, "Deep ITM",   "WP", 85, "SH", 3),
    _sl(63, "Deep ITM",   "WP", 90, "SK", 3),
    _sl(64, "Deep ITM",   "CW", 85, "SK", 3),
    _sl(65, "Deep ITM",   "CW", 85, "SH", 3),
    # === ATR Stop Debit Spreads (3% tight) ===
    _ds(66, "DS ATR Stop", "WP", 85, 0.03, "XN", "SK", 3, "EA15"),
    _ds(67, "DS ATR Stop", "WP", 85, 0.03, "XN", "SK", 3, "EA20"),
    _ds(68, "DS ATR Stop", "WP", 85, 0.03, "XN", "SK", 3, "EA25"),
    _ds(69, "DS ATR Stop", "WP", 85, 0.03, "XN", "SK", 3, "EA30"),
    _ds(70, "DS ATR Stop", "WP", 85, 0.03, "XN", "SH", 3, "EA15"),
    _ds(71, "DS ATR Stop", "WP", 85, 0.03, "XN", "SH", 3, "EA20"),
    _ds(72, "DS ATR Stop", "WP", 85, 0.03, "XN", "SH", 3, "EA25"),
    _ds(73, "DS ATR Stop", "WP", 85, 0.03, "XN", "SH", 3, "EA30"),
    # === ATR Stop Debit Spreads (5% wide) ===
    _ds(74, "DS ATR Stop", "WP", 85, 0.05, "XN", "SK", 3, "EA15"),
    _ds(75, "DS ATR Stop", "WP", 85, 0.05, "XN", "SK", 3, "EA20"),
    _ds(76, "DS ATR Stop", "WP", 85, 0.05, "XN", "SK", 3, "EA25"),
    _ds(77, "DS ATR Stop", "WP", 85, 0.05, "XN", "SK", 3, "EA30"),
    _ds(78, "DS ATR Stop", "CW", 85, 0.03, "XN", "SK", 3, "EA15"),
    _ds(79, "DS ATR Stop", "CW", 85, 0.03, "XN", "SK", 3, "EA20"),
    _ds(80, "DS ATR Stop", "CW", 85, 0.03, "XN", "SK", 3, "EA25"),
    _ds(81, "DS ATR Stop", "CW", 85, 0.03, "XN", "SK", 3, "EA30"),
]


# ============================================================
# Data Loading (shared with other backtesters)
# ============================================================

def load_backtester_data():
    path = RESULTS / "backtester_input_10_30.parquet"
    df = pd.read_parquet(path)
    df = df[df["direction"] == "l"].reset_index(drop=True)
    for col in ["predicted_return", "predicted_mfe", "actual_return", "actual_mfe"]:
        df[col] = df[col] / 100.0
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"Loaded {len(df):,} long opportunities")
    return df


def load_prices(symbols):
    prices = {}
    for sym in symbols:
        csv_path = DATA_DIR / "csv" / "US" / f"{sym}.csv"
        if not csv_path.exists():
            continue
        try:
            p = pd.read_csv(csv_path, usecols=["date", "close"], parse_dates=["date"])
            p = p[(p["date"] >= "2017-01-01") & (p["date"] <= "2026-12-31")]
            p = p.set_index("date").sort_index()
            p.index = p.index.date
            prices[sym] = p["close"]
        except Exception:
            pass
    print(f"Loaded prices for {len(prices)} symbols")
    return prices


def build_trading_days(prices):
    from datetime import date
    all_dates = set()
    for s in prices.values():
        all_dates.update(s.index)
    days = sorted(d for d in all_dates
                  if date(2018, 1, 1) <= d <= date(2025, 12, 31))
    return days


def load_earnings():
    """Load earnings dates from cached JSON."""
    from datetime import date
    path = RESULTS / "earnings_dates.json"
    if not path.exists():
        print("  WARNING: earnings_dates.json not found")
        return {}
    import json as _json
    with open(path) as f:
        raw = _json.load(f)
    earnings = {}
    for sym, dates in raw.items():
        earnings[sym] = set()
        for d in dates:
            try:
                y, m, day = d.split("-")
                earnings[sym].add(date(int(y), int(m), int(day)))
            except Exception:
                pass
    print(f"  Loaded earnings for {len(earnings)} symbols")
    return earnings


def has_earnings_during_hold(symbol, entry_date, holding_days, earnings_map):
    dates = earnings_map.get(symbol)
    if not dates:
        return False
    exit_date = entry_date + timedelta(days=holding_days)
    for ed in dates:
        if entry_date <= ed <= exit_date:
            return True
    return False


def build_candidates_by_date(df, earnings_map):
    print("  Building candidates (with earnings filter)...")
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
        print(f"  {n_filtered:,}/{len(df):,} filtered by earnings ({n_filtered/len(df):.1%})")
        df_clean = df[keep_mask]
    else:
        df_clean = df

    candidates = {}
    cols = ["symbol", "sector", "holding_days", "ml_score",
            "predicted_return", "predicted_mfe", "win_probability",
            "actual_return", "actual_mfe", "stock_volatility_20d", "year",
            "atr_14d_pct"]
    for date_val, group in df_clean.groupby("date"):
        candidates[date_val] = group[cols].to_dict("records")
    print(f"  {len(candidates)} dates with candidates")
    return candidates


def precompute_kelly_r(df):
    table = {}
    for thresh in [70, 80, 85, 90]:
        filtered = df[df["ml_score"] >= thresh]
        for year in range(2018, 2026):
            prior = filtered[filtered["year"] < year]
            if len(prior) < 100:
                prior = filtered
            wins = prior.loc[prior["actual_return"] > 0, "actual_return"]
            losses = prior.loc[prior["actual_return"] <= 0, "actual_return"].abs()
            R = float(wins.mean() / losses.mean()) if len(wins) > 10 and len(losses) > 10 else 1.3
            table[(thresh, year)] = R
    return table


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
        return sorted(candidates, key=lambda c: c["predicted_mfe"] - c["predicted_return"], reverse=True)
    elif method in ("CW", "CR"):
        n = len(candidates)
        if n <= 1:
            return list(candidates)
        wp = np.array([c["win_probability"] for c in candidates])
        pr = np.array([c["predicted_return"] for c in candidates])
        mg = np.array([c["predicted_mfe"] - c["predicted_return"] for c in candidates])
        wp_r, pr_r, mg_r = rankdata(wp) / n, rankdata(pr) / n, rankdata(mg) / n
        if method == "CW":
            scores = 0.60 * wp_r + 0.25 * pr_r + 0.15 * mg_r
        else:
            scores = 0.30 * wp_r + 0.50 * pr_r + 0.20 * mg_r
        return [candidates[i] for i in np.argsort(-scores)]
    return list(candidates)


def passes_iv_filter(symbol, date, iv_code):
    if iv_code == "IN":
        return True
    prob = {"IL": 0.60, "IH": 0.30}[iv_code]
    val = int(hashlib.md5(f"{symbol}_{date}_{iv_code}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return val < prob


# ============================================================
# Spread P&L Models
# ============================================================

def debit_spread_pnl(stock_return, spread_width, cost_ratio):
    """Compute debit spread P&L as fraction of premium paid.

    At expiry: payoff = min(max(stock_return, 0), spread_width)
    Cost = cost_ratio * spread_width
    Return = payoff / cost - 1
    """
    payoff = min(max(stock_return, 0.0), spread_width)
    cost = cost_ratio * spread_width
    if cost <= 0:
        return 0.0
    return payoff / cost - 1.0


def credit_spread_pnl(stock_return, short_otm, spread_width, credit_ratio):
    """Compute credit spread P&L as fraction of collateral.

    Win if stock_return >= -short_otm (stock stays above short put).
    Max loss if stock_return <= -(short_otm + spread_width).
    Collateral = (1 - credit_ratio) * spread_width (in stock-price terms).
    """
    credit = credit_ratio * spread_width
    collateral = spread_width - credit
    if collateral <= 0:
        return 0.0

    if stock_return >= -short_otm:
        # Full win: keep credit
        return credit / collateral
    elif stock_return <= -(short_otm + spread_width):
        # Full loss
        return -1.0
    else:
        # Partial loss: stock between short and long strikes
        intrusion = -stock_return - short_otm
        loss = intrusion - credit
        return -min(loss / collateral, 1.0)


def single_leg_pnl(stock_return, holding_days, max_trading_days):
    """Deep ITM call synthetic P&L (simplified from options backtester)."""
    import math
    p = DEEP_ITM_PARAMS
    total_dte = EXPIRY_DTE["X2"](holding_days)

    # Approximate: assume linear stock path to final return
    cum_opt = -p["spread_cost"] / 2
    for d in range(1, max_trading_days + 1):
        frac = d / max_trading_days
        daily_move = stock_return / max_trading_days
        dte = total_dte - d
        if dte <= 0:
            dte = 1
        delta_pnl = daily_move * p["delta"] * p["leverage"]
        frac_now = math.sqrt(dte / total_dte) if total_dte > 0 else 0
        frac_tom = math.sqrt(max(0, dte - 1) / total_dte) if total_dte > 0 else 0
        theta = (frac_now - frac_tom) * p["tv_pct"]
        cum_opt += delta_pnl - theta
        cum_opt = max(cum_opt, -1.0)

    cum_opt -= p["spread_cost"] / 2
    return max(cum_opt, -1.0)


# ============================================================
# Position Sizing
# ============================================================

def compute_size(equity, total_deployed, max_positions, method, candidate,
                 kelly_r, risk_per_unit):
    """Compute dollar allocation (premium for debit, collateral for credit)."""
    max_this = equity * MAX_POSITION_PCT
    headroom = equity * MAX_TOTAL_ALLOC - total_deployed
    if headroom <= 0:
        return 0.0

    if method == "SF":
        alloc = (equity * MAX_TOTAL_ALLOC) / max_positions
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
        frac = kelly_pct * (0.50 if ml >= 90 else 0.35 if ml >= 80 else 0.25)
        alloc = equity * frac
    else:
        alloc = (equity * MAX_TOTAL_ALLOC) / max_positions

    return min(alloc, max_this, headroom)


# ============================================================
# Simulation Engine
# ============================================================

def run_strategy(strategy, candidates_by_date, prices, trading_days, kelly_r_table):
    sid = strategy["id"]
    stype = strategy["type"]
    threshold = strategy["threshold"]
    sizing = strategy["sizing"]
    max_pos = strategy["max_positions"]
    ranking_method = strategy["ranking"]
    iv_code = strategy.get("iv_filter", "IL")

    cash = float(INITIAL_CAPITAL)
    open_positions = []
    trades = []
    equity_records = []
    peak_equity = float(INITIAL_CAPITAL)
    halt_remaining = 0
    n_days = len(trading_days)

    for day_idx in range(n_days):
        today = trading_days[day_idx]
        year = today.year
        kelly_r = kelly_r_table.get((threshold, year), 1.3)

        # --- Update positions, check exits ---
        still_open = []
        for pos in open_positions:
            sym = pos["symbol"]
            ps = prices.get(sym)
            if ps is not None and today in ps.index:
                stock_cr = ps[today] / pos["entry_price"] - 1.0
            else:
                stock_cr = pos.get("last_cr", 0.0)
            pos["last_cr"] = stock_cr
            td_held = day_idx - pos["entry_day_idx"]
            is_last = td_held >= pos["max_td"]

            # Compute current P&L based on spread type
            should_exit = is_last
            exit_reason = "expiry" if is_last else None

            # Early exit for debit spreads: capture profit when stock passes spread width
            if stype == "DEBIT" and not should_exit:
                exit_code = strategy.get("exit", "EH")
                # Update stock HWM for ATR trailing stop
                if stock_cr > pos.get("stock_hwm", 0.0):
                    pos["stock_hwm"] = stock_cr

                if exit_code == "EP":
                    # Aggressive: exit at 90% of spread width
                    if stock_cr >= pos["spread_width"] * 0.9:
                        should_exit = True
                        exit_reason = "profit_target"
                elif exit_code.startswith("EA") and td_held >= 2:
                    # ATR trailing stop from stock HWM
                    mult_map = {"EA15": 1.5, "EA20": 2.0, "EA25": 2.5, "EA30": 3.0}
                    mult = mult_map.get(exit_code, 2.0)
                    atr = pos.get("atr_14d_pct", 0.02)
                    if atr <= 0:
                        atr = 0.02
                    atr_stop = pos.get("stock_hwm", 0.0) - mult * atr
                    if stock_cr <= atr_stop:
                        should_exit = True
                        exit_reason = "atr_trail"
                    # Also exit at max profit
                    elif stock_cr >= pos["spread_width"]:
                        should_exit = True
                        exit_reason = "max_profit"
                else:
                    # Hold to expiry: exit at 100% of spread width (at max profit)
                    if stock_cr >= pos["spread_width"]:
                        should_exit = True
                        exit_reason = "max_profit"

            # Early exit for credit spreads: stop-loss when stock breaches short put
            if stype == "CREDIT" and not should_exit:
                short_otm = pos.get("short_otm", 0.03)
                if stock_cr <= -(short_otm * 0.5):
                    # Stock is halfway to the short put -- exit early to limit damage
                    should_exit = True
                    exit_reason = "credit_stop"

            if should_exit:
                if stype == "DEBIT":
                    cost_ratio = DEBIT_PARAMS.get(
                        (pos["spread_width"], strategy["expiry"]), 0.40)
                    opt_ret = debit_spread_pnl(stock_cr, pos["spread_width"], cost_ratio)
                elif stype == "CREDIT":
                    credit_ratio = CREDIT_PARAMS.get(
                        (pos["short_otm"], pos["spread_width"], strategy["expiry"]), 0.20)
                    opt_ret = credit_spread_pnl(
                        stock_cr, pos["short_otm"], pos["spread_width"], credit_ratio)
                else:  # SINGLE_LEG
                    opt_ret = single_leg_pnl(stock_cr, pos["holding_days"], pos["max_td"])

                opt_ret = max(opt_ret, -1.0)
                dollar_pnl = pos["allocation"] * opt_ret
                cash += pos["allocation"] + dollar_pnl
                trades.append({
                    "strategy_id": sid, "type": stype,
                    "symbol": sym, "sector": pos["sector"],
                    "entry_date": pos["entry_date"], "exit_date": today,
                    "days_held": td_held, "ml_score": pos["ml_score"],
                    "predicted_return": pos["predicted_return"],
                    "win_probability": pos["win_probability"],
                    "allocation": pos["allocation"],
                    "stock_return": stock_cr,
                    "option_return_pct": opt_ret,
                    "pnl_dollars": dollar_pnl,
                    "exit_reason": exit_reason or "expiry",
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # Equity
        # For open positions, estimate current value
        total_deployed = sum(p["allocation"] for p in open_positions)
        # Simple mark-to-market: assume 50% of final P&L realized mid-hold
        invested = total_deployed  # approximate
        total_equity = cash + invested

        if total_equity > peak_equity:
            peak_equity = total_equity
        drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0

        if drawdown >= DRAWDOWN_HALT_PCT and halt_remaining == 0:
            for pos in open_positions:
                cash += pos["allocation"] * 0.5  # emergency exit at ~50% recovery
                trades.append({
                    "strategy_id": sid, "type": stype,
                    "symbol": pos["symbol"], "sector": pos["sector"],
                    "entry_date": pos["entry_date"], "exit_date": today,
                    "days_held": day_idx - pos["entry_day_idx"],
                    "ml_score": pos["ml_score"],
                    "predicted_return": pos["predicted_return"],
                    "win_probability": pos["win_probability"],
                    "allocation": pos["allocation"],
                    "stock_return": pos.get("last_cr", 0),
                    "option_return_pct": -0.50,
                    "pnl_dollars": -pos["allocation"] * 0.50,
                    "exit_reason": "drawdown_halt",
                })
            open_positions = []
            halt_remaining = DRAWDOWN_HALT_DAYS
            total_deployed = 0
            invested = 0
            total_equity = cash
            peak_equity = cash

        equity_records.append({
            "date": today, "equity": total_equity, "cash": cash,
            "invested": invested, "open_positions": len(open_positions),
            "drawdown": drawdown,
        })

        if halt_remaining > 0:
            halt_remaining -= 1
            continue

        # --- New entries ---
        day_candidates = candidates_by_date.get(today, [])
        if not day_candidates:
            continue
        filtered = [c for c in day_candidates if c["ml_score"] >= threshold]
        if not filtered:
            continue
        if iv_code != "IN":
            filtered = [c for c in filtered if passes_iv_filter(c["symbol"], today, iv_code)]
        if not filtered:
            continue

        avail = max_pos - len(open_positions)
        if avail <= 0:
            continue

        ranked = rank_candidates(filtered, ranking_method)[:avail]

        for c in ranked:
            sym = c["symbol"]
            ps = prices.get(sym)
            if ps is None or today not in ps.index:
                continue
            entry_price = ps[today]
            if entry_price <= 0:
                continue

            total_dep = sum(p["allocation"] for p in open_positions)
            alloc = compute_size(total_equity, total_dep, max_pos, sizing,
                                 c, kelly_r, 1.0)
            if alloc <= 0:
                continue

            # Exit deadline
            hd = c["holding_days"]
            exit_deadline = today + timedelta(days=hd)
            exit_day_idx = day_idx
            for j in range(day_idx + 1, min(day_idx + hd + 10, n_days)):
                if trading_days[j] <= exit_deadline:
                    exit_day_idx = j
                else:
                    break
            max_td = max(exit_day_idx - day_idx, 1)

            cash -= alloc
            pos_data = {
                "symbol": sym, "sector": c["sector"],
                "entry_date": today, "entry_day_idx": day_idx,
                "entry_price": entry_price,
                "holding_days": hd, "max_td": max_td,
                "predicted_return": c["predicted_return"],
                "predicted_mfe": c["predicted_mfe"],
                "win_probability": c["win_probability"],
                "ml_score": c["ml_score"],
                "allocation": alloc, "last_cr": 0.0,
                "atr_14d_pct": c.get("atr_14d_pct", 0.02) or 0.02,
                "stock_hwm": 0.0,
            }
            if stype == "DEBIT":
                pos_data["spread_width"] = strategy["spread_width"]
            elif stype == "CREDIT":
                pos_data["short_otm"] = strategy["short_otm"]
                pos_data["spread_width"] = strategy["spread_width"]

            open_positions.append(pos_data)

    # Close remaining
    last_day = trading_days[-1]
    for pos in open_positions:
        stock_cr = pos.get("last_cr", 0.0)
        if stype == "DEBIT":
            cr = DEBIT_PARAMS.get((pos["spread_width"], strategy["expiry"]), 0.40)
            opt_ret = debit_spread_pnl(stock_cr, pos["spread_width"], cr)
        elif stype == "CREDIT":
            cr = CREDIT_PARAMS.get(
                (pos["short_otm"], pos["spread_width"], strategy["expiry"]), 0.20)
            opt_ret = credit_spread_pnl(stock_cr, pos["short_otm"], pos["spread_width"], cr)
        else:
            opt_ret = single_leg_pnl(stock_cr, pos["holding_days"], pos["max_td"])
        opt_ret = max(opt_ret, -1.0)
        dollar_pnl = pos["allocation"] * opt_ret
        cash += pos["allocation"] + dollar_pnl
        trades.append({
            "strategy_id": sid, "type": stype,
            "symbol": pos["symbol"], "sector": pos["sector"],
            "entry_date": pos["entry_date"], "exit_date": last_day,
            "days_held": n_days - 1 - pos["entry_day_idx"],
            "ml_score": pos["ml_score"],
            "predicted_return": pos["predicted_return"],
            "win_probability": pos["win_probability"],
            "allocation": pos["allocation"],
            "stock_return": stock_cr,
            "option_return_pct": opt_ret,
            "pnl_dollars": dollar_pnl,
            "exit_reason": "end_of_sim",
        })

    return trades, equity_records


# ============================================================
# Metrics
# ============================================================

def compute_metrics(trades, equity_records, strategy):
    sid = strategy["id"]
    if not trades:
        return {"strategy_id": sid, "type": strategy["type"],
                "category": strategy["category"], **{k: 0 for k in [
                    "total_return", "annualized_return", "max_drawdown",
                    "sharpe_ratio", "win_rate", "avg_win", "avg_loss",
                    "profit_factor", "total_trades", "trades_per_year",
                    "worst_trade", "years_profitable", "weighted_score",
                ] + [f"year_{y}" for y in range(2018, 2026)]}}

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity_records)
    n = len(tdf)
    wins = tdf[tdf["option_return_pct"] > 0]
    losses = tdf[tdf["option_return_pct"] <= 0]

    final = edf["equity"].iloc[-1] if len(edf) > 0 else INITIAL_CAPITAL
    total_ret = final / INITIAL_CAPITAL - 1
    ann_ret = (1 + total_ret) ** (1 / 8.0) - 1 if total_ret > -1 else -1

    eq_vals = edf["equity"].values
    peak = np.maximum.accumulate(eq_vals)
    dd = (peak - eq_vals) / peak
    max_dd = dd.max()

    edf["date"] = pd.to_datetime(edf["date"])
    edf["month"] = edf["date"].dt.to_period("M")
    monthly = edf.groupby("month")["equity"].last().pct_change().dropna()
    sharpe = (monthly.mean() / monthly.std()) * np.sqrt(12) if len(monthly) > 1 and monthly.std() > 0 else 0

    gp = wins["pnl_dollars"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else 0
    pf = gp / gl if gl > 0 else float("inf")

    year_rets = {}
    for yr in range(2018, 2026):
        ye = edf[edf["date"].dt.year == yr]
        year_rets[yr] = ye["equity"].iloc[-1] / ye["equity"].iloc[0] - 1 if len(ye) >= 2 else 0

    config_parts = [strategy.get("ranking", ""), f"T{strategy.get('threshold', '')}",
                    strategy.get("sizing", ""), f"P{strategy.get('max_positions', '')}"]
    if strategy["type"] == "DEBIT":
        config_parts += [f"W{int(strategy['spread_width']*100)}%", strategy.get("expiry", ""),
                         strategy.get("exit", "")]
    elif strategy["type"] == "CREDIT":
        config_parts += [f"O{int(strategy['short_otm']*100)}%",
                         f"W{int(strategy['spread_width']*100)}%", strategy.get("expiry", "")]

    return {
        "strategy_id": sid, "type": strategy["type"],
        "category": strategy["category"],
        "config": "/".join(config_parts),
        "total_return": total_ret, "annualized_return": ann_ret,
        "max_drawdown": max_dd, "sharpe_ratio": sharpe,
        "win_rate": len(wins) / n if n > 0 else 0,
        "avg_win": wins["option_return_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["option_return_pct"].mean() if len(losses) > 0 else 0,
        "profit_factor": pf,
        "total_trades": n, "trades_per_year": n / 8.0,
        "worst_trade": tdf["option_return_pct"].min() if n > 0 else 0,
        "years_profitable": sum(1 for r in year_rets.values() if r > 0),
        **{f"year_{yr}": year_rets.get(yr, 0) for yr in range(2018, 2026)},
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--strategies", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()
    if args.strategies:
        ids = set(int(x) for x in args.strategies.split(","))
        strategies = [s for s in STRATEGIES if s["id"] in ids]
    else:
        strategies = STRATEGIES

    print(f"Running {len(strategies)} spread strategies with {args.jobs} workers\n")

    df = load_backtester_data()
    prices = load_prices(df["symbol"].unique().tolist())
    trading_days = build_trading_days(prices)
    earnings_map = load_earnings()
    candidates_by_date = build_candidates_by_date(df, earnings_map)
    kelly_r_table = precompute_kelly_r(df)

    print(f"Data loaded in {time.time()-t0:.1f}s\n")
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    print("Running backtests...")
    t1 = time.time()

    def _run(s):
        tr, eq = run_strategy(s, candidates_by_date, prices, trading_days, kelly_r_table)
        m = compute_metrics(tr, eq, s)
        return s["id"], tr, eq, m

    results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(_run)(s) for s in strategies)
    print(f"\nCompleted in {time.time()-t1:.1f}s")

    all_trades, all_eq, all_metrics = [], [], []
    for sid, tr, eq, m in results:
        all_trades.extend(tr)
        for r in eq:
            r["strategy_id"] = sid
        all_eq.extend(eq)
        all_metrics.append(m)

    tdf = pd.DataFrame(all_trades)
    tdf.to_csv(BACKTEST_DIR / "trades.csv", index=False)

    edf = pd.DataFrame(all_eq)
    edf.to_csv(BACKTEST_DIR / "equity.csv", index=False)

    sdf = pd.DataFrame(all_metrics)
    # Weighted score
    sdf["consistency"] = sdf["years_profitable"] / 8.0
    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx - mn > 0 else 0.5
    sdf["weighted_score"] = (0.40 * sdf["consistency"] + 0.25 * norm(sdf["sharpe_ratio"])
                              + 0.20 * (1 - norm(sdf["max_drawdown"].abs()))
                              + 0.15 * norm(sdf["total_return"]))
    sdf = sdf.sort_values("weighted_score", ascending=False)
    sdf.to_csv(BACKTEST_DIR / "summary.csv", index=False)

    print(f"\nTrades: {len(tdf):,}  Equity: {len(edf):,}")
    print(f"\n{'='*120}")
    print("TOP 20 SPREAD STRATEGIES")
    print(f"{'='*120}")
    for _, r in sdf.head(20).iterrows():
        print(f"S{int(r.strategy_id):>3d} {r.type:>10s} {r.category:>14s} | "
              f"Sharpe {r.sharpe_ratio:5.2f} | DD {r.max_drawdown:6.1%} | "
              f"WR {r.win_rate:5.1%} | Trades {int(r.total_trades):>5d} | "
              f"Ann {r.annualized_return:7.1%} | PF {r.profit_factor:5.2f} | "
              f"Yr+ {int(r.years_profitable)} | WtSc {r.weighted_score:.3f}")

    print(f"\n{'='*120}")
    print("BY STRUCTURE TYPE")
    print(f"{'='*120}")
    for t in ["DEBIT", "CREDIT", "SINGLE_LEG"]:
        sub = sdf[sdf["type"] == t]
        if len(sub) > 0:
            print(f"  {t:>10s}: n={len(sub):3d}  "
                  f"Sharpe={sub['sharpe_ratio'].mean():.2f}  "
                  f"DD={sub['max_drawdown'].mean():.1%}  "
                  f"WR={sub['win_rate'].mean():.1%}  "
                  f"Profitable={int((sub['total_return']>0).sum())}/{len(sub)}  "
                  f"8yr+={int((sub['years_profitable']==8).sum())}")

    yr_cols = [f"year_{y}" for y in range(2018, 2026)]
    print(f"\n{'='*120}")
    print("YEAR-BY-YEAR (TOP 10)")
    print(f"{'='*120}")
    for _, r in sdf.head(10).iterrows():
        yrs = " ".join(f"{r[c]:7.1%}" for c in yr_cols)
        print(f"S{int(r.strategy_id):>3d} {r.type:>10s} | {yrs} | {int(r.years_profitable)}yr+")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
