"""
Stock Strategy Playbook V2 Backtester

112 strategy variants against 8-year walk-forward validation data (2018-2025).
10-30 Day Tier | SR + MFE Ensemble Models

Usage:
    python backtest_strategies.py                    # Run all 112 strategies
    python backtest_strategies.py --strategies 1,2,3 # Run specific strategies
    python backtest_strategies.py --jobs 12          # Parallel workers (default: 12)
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
# Constants
# ============================================================

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATA_DIR = Path("C:/seasonals/data")
BACKTEST_DIR = RESULTS / "backtest"

INITIAL_CAPITAL = 100_000
SLIPPAGE = 0.002          # 0.2% round-trip (0.1% entry + 0.1% exit)
CASH_RESERVE = 0.10       # 10% minimum cash
HARD_STOP_PCT = 0.10      # 10% of equity single-position hard stop
DRAWDOWN_HALT_PCT = 0.15  # 15% drawdown halts trading
DRAWDOWN_HALT_DAYS = 20   # Trading days to halt after drawdown trigger
EARNINGS_BUFFER_DAYS = 1  # Skip if earnings within this many days of entry/exit
EP_TRAIL_PCT = 0.03       # 3% trailing stop for EP exit rule
ET_TIME_PCT = 0.60        # 60% of holding period time limit for ET rule
EM_TRAIL_FACTOR = 0.50    # Trail 50% of gains above predicted return


# ============================================================
# Strategy Definitions (all 112)
# ============================================================

def _s(sid, cat, rank, thresh, exit_, size, pos, conc):
    return {
        "id": sid, "category": cat, "ranking": rank,
        "threshold": thresh, "exit": exit_, "sizing": size,
        "max_positions": pos, "concentration": conc,
    }

STRATEGIES = [
    # Cash Machine (1-25)
    _s(1,   "Cash Machine", "WP", 85, "EM", "SF", 3, "C2"),
    _s(2,   "Cash Machine", "WP", 85, "EM", "SK", 3, "C2"),
    _s(3,   "Cash Machine", "WP", 85, "EM", "SH", 3, "C2"),
    _s(4,   "Cash Machine", "WP", 85, "EM", "SC", 3, "C2"),
    _s(5,   "Cash Machine", "WP", 85, "EM", "SV", 3, "C2"),
    _s(6,   "Cash Machine", "WP", 90, "EM", "SK", 3, "C1"),
    _s(7,   "Cash Machine", "WP", 90, "EM", "SH", 3, "C1"),
    _s(8,   "Cash Machine", "WP", 90, "ET", "SK", 3, "C1"),
    _s(9,   "Cash Machine", "WP", 90, "ET", "SH", 4, "C1"),
    _s(10,  "Cash Machine", "WP", 85, "ET", "SK", 4, "C2"),
    _s(11,  "Cash Machine", "WP", 85, "ET", "SH", 4, "C2"),
    _s(12,  "Cash Machine", "WP", 85, "ET", "SA", 4, "C2"),
    _s(13,  "Cash Machine", "CW", 85, "EM", "SK", 3, "C2"),
    _s(14,  "Cash Machine", "CW", 85, "EM", "SH", 3, "C2"),
    _s(15,  "Cash Machine", "CW", 85, "EM", "SA", 3, "C2"),
    _s(16,  "Cash Machine", "CW", 90, "EM", "SK", 3, "C1"),
    _s(17,  "Cash Machine", "CW", 90, "ET", "SH", 3, "C1"),
    _s(18,  "Cash Machine", "CW", 85, "ET", "SK", 4, "C2"),
    _s(19,  "Cash Machine", "CW", 85, "ET", "SH", 4, "C2"),
    _s(20,  "Cash Machine", "CW", 85, "ET", "SA", 4, "C1"),
    _s(21,  "Cash Machine", "WP", 85, "EP", "SK", 3, "C2"),
    _s(22,  "Cash Machine", "WP", 85, "EP", "SH", 3, "C2"),
    _s(23,  "Cash Machine", "WP", 90, "EP", "SK", 3, "C1"),
    _s(24,  "Cash Machine", "CW", 85, "EP", "SA", 4, "C2"),
    _s(25,  "Cash Machine", "WP", 85, "EM", "SA", 3, "C1"),
    # Growth (26-50)
    _s(26,  "Growth", "PR", 85, "EM", "SH", 4, "C2"),
    _s(27,  "Growth", "PR", 85, "EM", "SK", 4, "C2"),
    _s(28,  "Growth", "PR", 80, "EM", "SH", 5, "C2"),
    _s(29,  "Growth", "PR", 80, "EM", "SA", 5, "C2"),
    _s(30,  "Growth", "PR", 85, "ET", "SH", 4, "C2"),
    _s(31,  "Growth", "PR", 85, "ET", "SA", 4, "C2"),
    _s(32,  "Growth", "CR", 85, "EM", "SH", 4, "C2"),
    _s(33,  "Growth", "CR", 85, "EM", "SA", 4, "C2"),
    _s(34,  "Growth", "CR", 80, "EM", "SH", 5, "C2"),
    _s(35,  "Growth", "CR", 80, "ET", "SH", 5, "C2"),
    _s(36,  "Growth", "CR", 85, "ET", "SA", 4, "C1"),
    _s(37,  "Growth", "MS", 85, "EM", "SH", 4, "C2"),
    _s(38,  "Growth", "MS", 85, "EM", "SA", 4, "C2"),
    _s(39,  "Growth", "MS", 80, "EM", "SH", 5, "C2"),
    _s(40,  "Growth", "MS", 80, "ET", "SH", 5, "CN"),
    _s(41,  "Growth", "PR", 90, "EM", "SH", 3, "C1"),
    _s(42,  "Growth", "PR", 90, "ET", "SA", 3, "C1"),
    _s(43,  "Growth", "CR", 90, "EM", "SH", 3, "C1"),
    _s(44,  "Growth", "CR", 90, "ET", "SH", 3, "C1"),
    _s(45,  "Growth", "MG", 85, "EM", "SH", 4, "C2"),
    _s(46,  "Growth", "MG", 85, "EM", "SA", 4, "C2"),
    _s(47,  "Growth", "MG", 85, "ET", "SH", 4, "C2"),
    _s(48,  "Growth", "MG", 80, "EM", "SH", 5, "C2"),
    _s(49,  "Growth", "PR", 80, "EM", "SH", 6, "CN"),
    _s(50,  "Growth", "CR", 80, "EM", "SA", 6, "C2"),
    # Baseline (51-65)
    _s(51,  "Baseline", "WP", 70, "EH", "SF", 3, "CN"),
    _s(52,  "Baseline", "WP", 80, "EH", "SF", 3, "CN"),
    _s(53,  "Baseline", "WP", 85, "EH", "SF", 3, "CN"),
    _s(54,  "Baseline", "WP", 90, "EH", "SF", 3, "CN"),
    _s(55,  "Baseline", "PR", 70, "EH", "SF", 3, "CN"),
    _s(56,  "Baseline", "PR", 80, "EH", "SF", 3, "CN"),
    _s(57,  "Baseline", "PR", 85, "EH", "SF", 3, "CN"),
    _s(58,  "Baseline", "PR", 90, "EH", "SF", 3, "CN"),
    _s(59,  "Baseline", "MS", 70, "EH", "SF", 3, "CN"),
    _s(60,  "Baseline", "MS", 80, "EH", "SF", 3, "CN"),
    _s(61,  "Baseline", "MS", 85, "EH", "SF", 3, "CN"),
    _s(62,  "Baseline", "MS", 90, "EH", "SF", 3, "CN"),
    _s(63,  "Baseline", "WP", 85, "EH", "SF", 4, "CN"),
    _s(64,  "Baseline", "WP", 85, "EH", "SF", 5, "CN"),
    _s(65,  "Baseline", "WP", 85, "EH", "SF", 6, "CN"),
    # Position Count Study (66-85)
    _s(66,  "Position Count", "WP", 85, "EM", "SK", 2, "C1"),
    _s(67,  "Position Count", "WP", 85, "EM", "SH", 2, "C1"),
    _s(68,  "Position Count", "WP", 85, "EM", "SK", 4, "C2"),
    _s(69,  "Position Count", "WP", 85, "EM", "SH", 4, "C2"),
    _s(70,  "Position Count", "WP", 85, "EM", "SK", 5, "C2"),
    _s(71,  "Position Count", "WP", 85, "EM", "SH", 5, "C2"),
    _s(72,  "Position Count", "WP", 85, "EM", "SK", 6, "C2"),
    _s(73,  "Position Count", "WP", 85, "EM", "SH", 6, "C2"),
    _s(74,  "Position Count", "WP", 85, "EM", "SK", 8, "C2"),
    _s(75,  "Position Count", "WP", 85, "EM", "SH", 8, "C2"),
    _s(76,  "Position Count", "CW", 85, "EM", "SK", 2, "C1"),
    _s(77,  "Position Count", "CW", 85, "EM", "SH", 2, "C1"),
    _s(78,  "Position Count", "CW", 85, "EM", "SK", 4, "C2"),
    _s(79,  "Position Count", "CW", 85, "EM", "SH", 4, "C2"),
    _s(80,  "Position Count", "CW", 85, "EM", "SK", 6, "C2"),
    _s(81,  "Position Count", "CW", 85, "EM", "SH", 6, "C2"),
    _s(82,  "Position Count", "CW", 85, "EM", "SK", 8, "C2"),
    _s(83,  "Position Count", "CW", 85, "EM", "SH", 8, "C2"),
    _s(84,  "Position Count", "CW", 85, "ET", "SA", 5, "C2"),
    _s(85,  "Position Count", "CW", 85, "ET", "SA", 8, "C2"),
    # Kelly Deep Dive (86-100)
    _s(86,  "Kelly Dive", "WP", 85, "EM", "SK", 3, "C2"),
    _s(87,  "Kelly Dive", "WP", 85, "EM", "SH", 3, "C2"),
    _s(88,  "Kelly Dive", "WP", 85, "EM", "SA", 3, "C2"),
    _s(89,  "Kelly Dive", "WP", 85, "ET", "SK", 3, "C2"),
    _s(90,  "Kelly Dive", "WP", 85, "ET", "SH", 3, "C2"),
    _s(91,  "Kelly Dive", "WP", 85, "ET", "SA", 3, "C2"),
    _s(92,  "Kelly Dive", "CW", 85, "EM", "SK", 4, "C2"),
    _s(93,  "Kelly Dive", "CW", 85, "EM", "SH", 4, "C2"),
    _s(94,  "Kelly Dive", "CW", 85, "EM", "SA", 4, "C2"),
    _s(95,  "Kelly Dive", "CW", 85, "ET", "SK", 4, "C2"),
    _s(96,  "Kelly Dive", "CW", 85, "ET", "SH", 4, "C2"),
    _s(97,  "Kelly Dive", "CW", 85, "ET", "SA", 4, "C2"),
    _s(98,  "Kelly Dive", "PR", 85, "EM", "SK", 4, "C2"),
    _s(99,  "Kelly Dive", "PR", 85, "EM", "SH", 4, "C2"),
    _s(100, "Kelly Dive", "PR", 85, "EM", "SA", 4, "C2"),
    # Aggressive (101-112)
    _s(101, "Aggressive", "PR", 70, "EM", "SH", 6, "CN"),
    _s(102, "Aggressive", "PR", 70, "ET", "SH", 8, "CN"),
    _s(103, "Aggressive", "CR", 70, "EM", "SH", 6, "CN"),
    _s(104, "Aggressive", "CR", 70, "ET", "SH", 8, "CN"),
    _s(105, "Aggressive", "MG", 70, "EM", "SH", 6, "CN"),
    _s(106, "Aggressive", "MS", 70, "EM", "SH", 6, "CN"),
    _s(107, "Aggressive", "PR", 80, "EM", "SH", 8, "CN"),
    _s(108, "Aggressive", "CR", 80, "EM", "SH", 8, "CN"),
    _s(109, "Aggressive", "PR", 70, "EM", "SA", 6, "C2"),
    _s(110, "Aggressive", "CR", 70, "EM", "SA", 6, "C2"),
    _s(111, "Aggressive", "MG", 70, "EM", "SA", 6, "C2"),
    _s(112, "Aggressive", "MS", 70, "EM", "SA", 8, "C2"),
]


# ============================================================
# Data Loading
# ============================================================

def load_backtester_data():
    """Load the merged backtester input, filter to longs, convert to decimal returns."""
    path = RESULTS / "backtester_input_10_30.parquet"
    print(f"Loading backtester input from {path}...")
    df = pd.read_parquet(path)

    # Longs only (playbook Section 4.3)
    df = df[df["direction"] == "l"].reset_index(drop=True)
    print(f"  {len(df):,} long opportunities across {df['year'].nunique()} years")

    # Convert percentage-point returns to decimal
    for col in ["predicted_return", "predicted_mfe", "actual_return", "actual_mfe"]:
        df[col] = df[col] / 100.0

    # Ensure date is a python date for fast comparison
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def load_prices(symbols):
    """Load close prices and volumes for all symbols from CSVs.

    Returns:
        prices: dict of {symbol: pd.Series(date -> close_price)}
        volumes: dict of {symbol: pd.Series(date -> volume)}
    """
    print(f"Loading price data for {len(symbols)} symbols...")
    prices = {}
    volumes = {}
    missing = []

    for sym in symbols:
        csv_path = DATA_DIR / "csv" / "US" / f"{sym}.csv"
        if not csv_path.exists():
            missing.append(sym)
            continue
        try:
            p = pd.read_csv(csv_path, usecols=["date", "close", "volume"],
                            parse_dates=["date"])
            # Filter to relevant range (2017-2026)
            p = p[(p["date"] >= "2017-01-01") & (p["date"] <= "2026-12-31")]
            p = p.set_index("date").sort_index()
            # Convert index to python date
            p.index = p.index.date
            prices[sym] = p["close"]
            volumes[sym] = p["volume"]
        except Exception as e:
            missing.append(sym)

    if missing:
        print(f"  WARNING: {len(missing)} symbols missing price data: {missing[:10]}...")
    print(f"  Loaded prices for {len(prices)} symbols")
    return prices, volumes


def build_trading_days(prices):
    """Get sorted array of all trading days from price data, filtered to 2018-2025."""
    from datetime import date
    all_dates = set()
    for series in prices.values():
        all_dates.update(series.index)
    days = sorted(d for d in all_dates
                  if date(2018, 1, 1) <= d <= date(2025, 12, 31))
    print(f"  {len(days)} trading days from {days[0]} to {days[-1]}")
    return days


def load_earnings():
    """Load earnings dates from cached JSON. Returns {symbol: set of date objects}."""
    path = RESULTS / "earnings_dates.json"
    if not path.exists():
        print("  WARNING: earnings_dates.json not found, skipping earnings filter")
        return {}
    from datetime import date
    with open(path) as f:
        raw = json.load(f)
    earnings = {}
    total = 0
    for sym, dates in raw.items():
        earnings[sym] = set()
        for d in dates:
            try:
                y, m, day = d.split("-")
                earnings[sym].add(date(int(y), int(m), int(day)))
                total += 1
            except Exception:
                pass
    print(f"  Loaded {total} earnings dates for {len(earnings)} symbols")
    return earnings


def has_earnings_during_hold(symbol, entry_date, holding_days, earnings_map):
    """Check if any earnings date falls within the holding period."""
    dates = earnings_map.get(symbol)
    if not dates:
        return False
    exit_date = entry_date + timedelta(days=holding_days)
    for ed in dates:
        if entry_date <= ed <= exit_date:
            return True
    return False


def build_candidates_by_date(df, earnings_map):
    """Group candidates by date as list of dicts, filtering out earnings conflicts."""
    print("  Building candidates-by-date index (with earnings filter)...")

    if earnings_map:
        # Convert earnings to {symbol: sorted numpy array of ordinals} for fast lookup
        from datetime import date as _date
        earnings_ord = {}
        for sym, dates in earnings_map.items():
            if dates:
                earnings_ord[sym] = np.array(sorted(d.toordinal() for d in dates))

        # Vectorized check per symbol group using searchsorted
        print("    Checking earnings overlap per symbol...")
        keep_mask = np.ones(len(df), dtype=bool)
        df_idx = df.index.values

        for sym, grp in df.groupby("symbol"):
            earr = earnings_ord.get(sym)
            if earr is None or len(earr) == 0:
                continue
            entry_ords = np.array([d.toordinal() for d in grp["date"].values])
            exit_ords = entry_ords + grp["holding_days"].values.astype(int)
            # For each row, find first earnings >= entry. If that earnings <= exit, conflict.
            idx = np.searchsorted(earr, entry_ords, side="left")
            conflict = np.zeros(len(grp), dtype=bool)
            # Check the earnings date at idx (first >= entry)
            valid = idx < len(earr)
            conflict[valid] = earr[idx[valid]] <= exit_ords[valid]
            # Also check idx-1 (last < entry) in case exit extends past it
            valid2 = idx > 0
            conflict[valid2] |= earr[np.clip(idx[valid2] - 1, 0, len(earr) - 1)] >= entry_ords[valid2]
            keep_mask[grp.index.values[conflict]] = False

        n_filtered = (~keep_mask).sum()
        print(f"    {n_filtered:,} of {len(df):,} filtered by earnings ({n_filtered/len(df):.1%})")
        df_clean = df[keep_mask]
    else:
        df_clean = df

    candidates = {}
    cols = ["symbol", "sector", "holding_days", "ml_score",
            "predicted_return", "predicted_mfe", "win_probability",
            "actual_return", "actual_mfe", "stock_volatility_20d", "year"]
    for date_val, group in df_clean.groupby("date"):
        candidates[date_val] = group[cols].to_dict("records")
    print(f"  {len(candidates)} dates with candidates")
    return candidates


# ============================================================
# Kelly R Pre-computation
# ============================================================

def precompute_kelly_r(df):
    """Compute rolling win/loss ratio for Kelly sizing.

    For each (threshold, year), use prior years' validation data.
    Returns: dict of {(threshold, year): R_value}
    """
    print("Pre-computing Kelly R table...")
    table = {}
    thresholds = [70, 80, 85, 90]
    years = list(range(2018, 2026))

    for thresh in thresholds:
        filtered = df[df["ml_score"] >= thresh]
        for year in years:
            prior = filtered[filtered["year"] < year]
            if len(prior) < 100:
                # First year or insufficient data: use all available
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
# Ranking Functions
# ============================================================

def rank_candidates(candidates, method):
    """Rank candidates by the given method. Returns sorted list (best first)."""
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
# Concentration Filter
# ============================================================

def apply_concentration(ranked, open_positions, rule, max_positions):
    """Filter candidates by sector concentration limits."""
    if rule == "CN":
        return ranked

    max_per_sector = 2 if rule == "C2" else 1

    # Count sectors in current open positions
    sector_count = {}
    for pos in open_positions:
        s = pos["sector"]
        sector_count[s] = sector_count.get(s, 0) + 1

    selected = []
    for c in ranked:
        s = c["sector"]
        if sector_count.get(s, 0) < max_per_sector:
            selected.append(c)
            sector_count[s] = sector_count.get(s, 0) + 1
        if len(selected) + len(open_positions) >= max_positions:
            break
    return selected


# ============================================================
# Position Sizing
# ============================================================

def compute_size(equity, open_positions, max_positions, method, candidate,
                 kelly_r, strategy_threshold):
    """Compute dollar allocation for a new position."""
    available = equity * (1 - CASH_RESERVE)

    if method == "SF":
        return available / max_positions

    elif method in ("SK", "SH"):
        W = candidate["win_probability"]
        R = kelly_r
        kelly_pct = max(W - (1 - W) / R, 0.0)
        frac = kelly_pct * (0.25 if method == "SK" else 0.50)
        alloc = equity * frac
        max_alloc = available / max_positions * 2
        return min(alloc, max_alloc)

    elif method == "SC":
        base = available / max_positions
        scale = (candidate["win_probability"] - 0.50) * 4
        scale = max(0.5, min(1.5, scale))
        return base * scale

    elif method == "SV":
        base = available / max_positions
        vol = candidate.get("stock_volatility_20d", 0.25)
        if vol <= 0:
            vol = 0.25
        vol_ratio = 0.25 / vol
        vol_ratio = max(0.5, min(2.0, vol_ratio))
        return base * vol_ratio

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
        max_alloc = available / max_positions * 2
        return min(alloc, max_alloc)

    return available / max_positions


def enforce_allocation_cap(new_allocs, current_invested, equity):
    """Scale down new allocations if total would exceed 90% of equity.

    Args:
        new_allocs: list of proposed $ allocations for new positions
        current_invested: current mark-to-market value of ALL open positions
        equity: current total equity
    """
    total_new = sum(new_allocs)
    cap = equity * (1 - CASH_RESERVE)
    headroom = cap - current_invested
    if headroom <= 0:
        return [0.0] * len(new_allocs)
    if total_new <= headroom:
        return new_allocs
    scale = headroom / total_new
    return [a * scale for a in new_allocs]


# ============================================================
# Exit Rules
# ============================================================

def check_exit(pos, cum_return, trading_day_idx, exit_rule):
    """Check if position should exit.

    Args:
        pos: position dict (mutated: hwm, pred_reached, trail_stop)
        cum_return: current cumulative return (decimal)
        trading_day_idx: number of trading days since entry (1-based)
        exit_rule: 'EH', 'EM', 'ET', 'EP'

    Returns: (action, exit_return, reason) where action is 'hold' or 'exit'
    """
    # Update high water mark
    if cum_return > pos["hwm"]:
        pos["hwm"] = cum_return

    is_last_day = trading_day_idx >= pos["max_trading_days"]

    if exit_rule == "EH":
        if is_last_day:
            return ("exit", cum_return, "hold_expiry")
        return ("hold", None, None)

    elif exit_rule == "EM":
        pred_ret = pos["predicted_return"]
        if not pos["pred_reached"] and cum_return >= pred_ret:
            pos["pred_reached"] = True
            pos["trail_stop"] = pred_ret

        if pos["pred_reached"]:
            excess = pos["hwm"] - pred_ret
            pos["trail_stop"] = pred_ret + excess * EM_TRAIL_FACTOR
            if cum_return <= pos["trail_stop"]:
                return ("exit", pos["trail_stop"], "trailing_stop")

        if is_last_day:
            return ("exit", cum_return, "hold_expiry")
        return ("hold", None, None)

    elif exit_rule == "ET":
        # Time limit check
        time_limit = int(pos["max_trading_days"] * ET_TIME_PCT)
        if trading_day_idx >= time_limit and not pos["pred_reached"]:
            return ("exit", cum_return, "time_limit")

        # Same trailing logic as EM
        pred_ret = pos["predicted_return"]
        if not pos["pred_reached"] and cum_return >= pred_ret:
            pos["pred_reached"] = True
            pos["trail_stop"] = pred_ret

        if pos["pred_reached"]:
            excess = pos["hwm"] - pred_ret
            pos["trail_stop"] = pred_ret + excess * EM_TRAIL_FACTOR
            if cum_return <= pos["trail_stop"]:
                return ("exit", pos["trail_stop"], "trailing_stop")

        if is_last_day:
            return ("exit", cum_return, "hold_expiry")
        return ("hold", None, None)

    elif exit_rule == "EP":
        stop = pos["hwm"] - EP_TRAIL_PCT
        if cum_return <= stop and trading_day_idx >= 2:
            return ("exit", stop, "pct_trail")

        if is_last_day:
            return ("exit", cum_return, "hold_expiry")
        return ("hold", None, None)

    return ("hold", None, None)


# ============================================================
# Simulation Engine
# ============================================================

def run_strategy(strategy, candidates_by_date, prices, trading_days,
                 trading_day_set, kelly_r_table):
    """Run a single strategy simulation across all trading days.

    Returns: (trades_list, equity_records)
    """
    sid = strategy["id"]
    threshold = strategy["threshold"]
    exit_rule = strategy["exit"]
    sizing = strategy["sizing"]
    max_pos = strategy["max_positions"]
    conc_rule = strategy["concentration"]
    ranking = strategy["ranking"]

    # Portfolio state
    cash = float(INITIAL_CAPITAL)
    open_positions = []
    trades = []
    equity_records = []
    peak_equity = float(INITIAL_CAPITAL)
    halt_days_remaining = 0

    n_days = len(trading_days)
    # Pre-build a day -> index lookup for fast exit deadline computation
    day_to_idx = {d: i for i, d in enumerate(trading_days)}

    for day_idx in range(n_days):
        today = trading_days[day_idx]
        year = today.year

        # Kelly R for this threshold/year
        kelly_r = kelly_r_table.get((threshold, year), 1.3)

        # --- Step 0: Compute start-of-day equity (before exits) ---
        sod_invested = 0.0
        pos_returns = {}  # cache cumulative returns for this day
        for pos in open_positions:
            sym = pos["symbol"]
            price_series = prices.get(sym)
            if price_series is not None and today in price_series.index:
                cr = price_series[today] / pos["entry_price"] - 1.0
            else:
                cr = 0.0
            pos_returns[id(pos)] = cr
            sod_invested += pos["allocation"] * (1 + cr)
        sod_equity = cash + sod_invested

        # --- Step 1: Update open positions, check exits ---
        still_open = []

        for pos in open_positions:
            cum_return = pos_returns[id(pos)]
            trading_days_held = day_idx - pos["entry_day_idx"]

            # Hard stop: single position loss > 10% of start-of-day equity
            pos_loss = pos["allocation"] * max(0.0, -cum_return)
            if pos_loss >= sod_equity * HARD_STOP_PCT:
                slipped = cum_return - SLIPPAGE
                pnl = pos["allocation"] * slipped
                cash += pos["allocation"] + pnl
                trades.append(_make_trade(pos, today, slipped, trading_days_held,
                                          "hard_stop", sid))
                continue

            # Normal exit check
            action, exit_ret, reason = check_exit(
                pos, cum_return, trading_days_held, exit_rule)

            if action == "exit":
                slipped = exit_ret - SLIPPAGE
                pnl = pos["allocation"] * slipped
                cash += pos["allocation"] + pnl
                trades.append(_make_trade(pos, today, slipped, trading_days_held,
                                          reason, sid))
            else:
                still_open.append(pos)

        open_positions = still_open

        # --- Step 2: Compute post-exit equity and check drawdown ---
        invested = 0.0
        for pos in open_positions:
            cr = pos_returns.get(id(pos), 0.0)
            invested += pos["allocation"] * (1 + cr)
        total_equity = cash + invested

        if total_equity > peak_equity:
            peak_equity = total_equity
        drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0

        if drawdown >= DRAWDOWN_HALT_PCT and halt_days_remaining == 0:
            # Close all positions
            for pos in open_positions:
                price_series = prices.get(pos["symbol"])
                if price_series is not None and today in price_series.index:
                    cr = price_series[today] / pos["entry_price"] - 1.0
                else:
                    cr = 0.0
                slipped = cr - SLIPPAGE
                pnl = pos["allocation"] * slipped
                cash += pos["allocation"] + pnl
                td_held = day_idx - pos["entry_day_idx"]
                trades.append(_make_trade(pos, today, slipped, td_held,
                                          "drawdown_halt", sid))
            open_positions = []
            halt_days_remaining = DRAWDOWN_HALT_DAYS
            invested = 0.0
            total_equity = cash
            peak_equity = cash  # Reset peak after halt

        # Record daily equity
        equity_records.append({
            "date": today,
            "equity": total_equity,
            "cash": cash,
            "invested": invested,
            "open_positions": len(open_positions),
            "drawdown": drawdown,
        })

        # --- Step 3: Skip new entries if halted ---
        if halt_days_remaining > 0:
            halt_days_remaining -= 1
            continue

        # --- Step 4: Get and filter new candidates ---
        day_candidates = candidates_by_date.get(today, [])
        if not day_candidates:
            continue

        filtered = [c for c in day_candidates if c["ml_score"] >= threshold]
        if not filtered:
            continue

        available_slots = max_pos - len(open_positions)
        if available_slots <= 0:
            continue

        # --- Step 5: Rank ---
        ranked = rank_candidates(filtered, ranking)

        # --- Step 6: Apply concentration limits ---
        selected = apply_concentration(ranked, open_positions, conc_rule, max_pos)
        selected = selected[:available_slots]
        if not selected:
            continue

        # --- Step 7: Size positions ---
        allocs = []
        for c in selected:
            alloc = compute_size(total_equity, open_positions, max_pos,
                                 sizing, c, kelly_r, threshold)
            allocs.append(alloc)

        # Enforce 90% cap using current mark-to-market invested value
        allocs = enforce_allocation_cap(allocs, invested, total_equity)

        # --- Step 8: Enter positions ---
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

            # Compute exit deadline and max trading days
            exit_deadline = today + timedelta(days=c["holding_days"])
            # Find last trading day <= exit_deadline
            exit_day_idx = day_idx
            for j in range(day_idx + 1, min(day_idx + c["holding_days"] + 10, n_days)):
                if trading_days[j] <= exit_deadline:
                    exit_day_idx = j
                else:
                    break
            max_td = exit_day_idx - day_idx
            if max_td < 1:
                max_td = 1

            cash -= alloc
            open_positions.append({
                "symbol": sym,
                "sector": c["sector"],
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
                "pred_reached": False,
                "trail_stop": 0.0,
            })

    # Close any remaining open positions at end of simulation
    if open_positions:
        last_day = trading_days[-1]
        for pos in open_positions:
            price_series = prices.get(pos["symbol"])
            if price_series is not None and last_day in price_series.index:
                cr = price_series[last_day] / pos["entry_price"] - 1.0
            else:
                cr = 0.0
            slipped = cr - SLIPPAGE
            pnl = pos["allocation"] * slipped
            cash += pos["allocation"] + pnl
            td_held = len(trading_days) - 1 - pos["entry_day_idx"]
            trades.append(_make_trade(pos, last_day, slipped, td_held,
                                      "end_of_sim", sid))

    return trades, equity_records


def _make_trade(pos, exit_date, slipped_return, days_held, reason, strategy_id):
    """Create a trade record dict."""
    return {
        "strategy_id": strategy_id,
        "symbol": pos["symbol"],
        "sector": pos["sector"],
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
# Metrics Computation
# ============================================================

def compute_strategy_metrics(trades, equity_records, strategy):
    """Compute all metrics for one strategy."""
    sid = strategy["id"]

    if not trades:
        return _empty_metrics(strategy)

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_records)

    total_trades = len(trades_df)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0

    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    gross_profit = wins["pnl_dollars"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Total and annualized return
    final_equity = eq_df["equity"].iloc[-1] if len(eq_df) > 0 else INITIAL_CAPITAL
    total_return = (final_equity / INITIAL_CAPITAL) - 1
    n_years = 8.0
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if total_return > -1 else -1

    # Max drawdown
    eq_vals = eq_df["equity"].values
    peak = np.maximum.accumulate(eq_vals)
    dd = (peak - eq_vals) / peak
    max_dd = dd.max() if len(dd) > 0 else 0

    # Sharpe ratio (annualized, from monthly returns)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df["month"] = eq_df["date"].dt.to_period("M")
    monthly = eq_df.groupby("month")["equity"].last()
    monthly_returns = monthly.pct_change().dropna()
    if len(monthly_returns) > 1 and monthly_returns.std() > 0:
        sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
    else:
        sharpe = 0.0

    # Worst single trade
    worst_trade = trades_df["pnl_pct"].min() if total_trades > 0 else 0

    # Longest losing streak
    streak = 0
    max_streak = 0
    for ret in trades_df["pnl_pct"]:
        if ret <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Trades per year
    trades_per_year = total_trades / n_years

    # Year-by-year returns
    year_returns = {}
    for yr in range(2018, 2026):
        yr_eq = eq_df[eq_df["date"].dt.year == yr]
        if len(yr_eq) >= 2:
            yr_ret = yr_eq["equity"].iloc[-1] / yr_eq["equity"].iloc[0] - 1
        else:
            yr_ret = 0.0
        year_returns[yr] = yr_ret

    years_profitable = sum(1 for r in year_returns.values() if r > 0)

    return {
        "strategy_id": sid,
        "category": strategy["category"],
        "ranking": strategy["ranking"],
        "threshold": strategy["threshold"],
        "exit": strategy["exit"],
        "sizing": strategy["sizing"],
        "max_positions": strategy["max_positions"],
        "concentration": strategy["concentration"],
        "total_return": total_return,
        "annualized_return": ann_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "trades_per_year": trades_per_year,
        "worst_single_trade": worst_trade,
        "longest_losing_streak": max_streak,
        "years_profitable": years_profitable,
        **{f"year_{yr}": year_returns.get(yr, 0) for yr in range(2018, 2026)},
    }


def _empty_metrics(strategy):
    """Return empty metrics for a strategy with no trades."""
    return {
        "strategy_id": strategy["id"],
        "category": strategy["category"],
        "ranking": strategy["ranking"],
        "threshold": strategy["threshold"],
        "exit": strategy["exit"],
        "sizing": strategy["sizing"],
        "max_positions": strategy["max_positions"],
        "concentration": strategy["concentration"],
        "total_return": 0, "annualized_return": 0, "max_drawdown": 0,
        "sharpe_ratio": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
        "profit_factor": 0, "total_trades": 0, "trades_per_year": 0,
        "worst_single_trade": 0, "longest_losing_streak": 0,
        "years_profitable": 0,
        **{f"year_{yr}": 0 for yr in range(2018, 2026)},
    }


def compute_weighted_scores(summary_df):
    """Compute the weighted selection score across all strategies."""
    # Consistency: years_profitable / 8
    summary_df["consistency_score"] = summary_df["years_profitable"] / 8.0

    # Normalize Sharpe, drawdown, return across strategies
    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx - mn == 0:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    summary_df["sharpe_norm"] = normalize(summary_df["sharpe_ratio"])
    summary_df["dd_norm"] = 1 - normalize(summary_df["max_drawdown"].abs())
    summary_df["return_norm"] = normalize(summary_df["total_return"])

    summary_df["weighted_score"] = (
        0.40 * summary_df["consistency_score"]
        + 0.25 * summary_df["sharpe_norm"]
        + 0.20 * summary_df["dd_norm"]
        + 0.15 * summary_df["return_norm"]
    )
    return summary_df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Stock Strategy Playbook V2 Backtester")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategy IDs to run (default: all)")
    parser.add_argument("--jobs", type=int, default=12,
                        help="Number of parallel workers (default: 12)")
    args = parser.parse_args()

    t0 = time.time()

    # Select strategies
    if args.strategies:
        ids = set(int(x) for x in args.strategies.split(","))
        strategies = [s for s in STRATEGIES if s["id"] in ids]
    else:
        strategies = STRATEGIES
    print(f"Running {len(strategies)} strategies with {args.jobs} workers\n")

    # Load data
    df = load_backtester_data()
    symbols = df["symbol"].unique().tolist()
    prices, volumes = load_prices(symbols)
    trading_days = build_trading_days(prices)
    trading_day_set = set(trading_days)
    earnings_map = load_earnings()
    candidates_by_date = build_candidates_by_date(df, earnings_map)
    kelly_r_table = precompute_kelly_r(df)

    print(f"\nData loaded in {time.time()-t0:.1f}s")
    print(f"Kelly R samples: {dict(list(kelly_r_table.items())[:4])}\n")

    # Create output directory
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    # Run strategies in parallel
    print("Running backtests...")
    t1 = time.time()

    def _run_one(strat):
        trades, eq = run_strategy(
            strat, candidates_by_date, prices, trading_days,
            trading_day_set, kelly_r_table)
        metrics = compute_strategy_metrics(trades, eq, strat)
        return strat["id"], trades, eq, metrics

    results = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(_run_one)(s) for s in strategies
    )

    print(f"\nAll backtests completed in {time.time()-t1:.1f}s")

    # Collect results
    all_trades = []
    all_equity = []
    all_metrics = []

    for sid, trades, eq, metrics in results:
        all_trades.extend(trades)
        for rec in eq:
            rec["strategy_id"] = sid
        all_equity.extend(eq)
        all_metrics.append(metrics)

    # Save trade log
    trades_df = pd.DataFrame(all_trades)
    trades_path = BACKTEST_DIR / "trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"\nTrade log: {trades_path} ({len(trades_df):,} trades)")

    # Save equity curves
    equity_df = pd.DataFrame(all_equity)
    equity_path = BACKTEST_DIR / "equity.csv"
    equity_df.to_csv(equity_path, index=False)
    print(f"Equity curves: {equity_path} ({len(equity_df):,} records)")

    # Save strategy summary
    summary_df = pd.DataFrame(all_metrics)
    summary_df = compute_weighted_scores(summary_df)
    summary_df = summary_df.sort_values("weighted_score", ascending=False)
    summary_path = BACKTEST_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Strategy summary: {summary_path}")

    # Print top 20
    print("\n" + "=" * 100)
    print("TOP 20 STRATEGIES BY WEIGHTED SCORE")
    print("=" * 100)
    cols = ["strategy_id", "category", "ranking", "threshold", "exit",
            "sizing", "max_positions", "concentration",
            "total_return", "annualized_return", "max_drawdown",
            "sharpe_ratio", "win_rate", "total_trades", "years_profitable",
            "weighted_score"]
    top20 = summary_df.head(20)[cols].copy()
    top20["total_return"] = top20["total_return"].map(lambda x: f"{x:.1%}")
    top20["annualized_return"] = top20["annualized_return"].map(lambda x: f"{x:.1%}")
    top20["max_drawdown"] = top20["max_drawdown"].map(lambda x: f"{x:.1%}")
    top20["sharpe_ratio"] = top20["sharpe_ratio"].map(lambda x: f"{x:.2f}")
    top20["win_rate"] = top20["win_rate"].map(lambda x: f"{x:.1%}")
    top20["weighted_score"] = top20["weighted_score"].map(lambda x: f"{x:.3f}")
    print(top20.to_string(index=False))

    # Year-by-year heatmap for top 10
    print("\n" + "=" * 100)
    print("YEAR-BY-YEAR RETURNS (TOP 10 STRATEGIES)")
    print("=" * 100)
    top10_ids = summary_df.head(10)["strategy_id"].tolist()
    yr_cols = [f"year_{y}" for y in range(2018, 2026)]
    heatmap = summary_df[summary_df["strategy_id"].isin(top10_ids)][
        ["strategy_id", "category"] + yr_cols].copy()
    for c in yr_cols:
        heatmap[c] = heatmap[c].map(lambda x: f"{x:.1%}")
    print(heatmap.to_string(index=False))

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
