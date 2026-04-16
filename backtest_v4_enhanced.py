"""
V4 Enhanced Backtest
====================
Extends the Codex V3 best configurations with 5 new enhancement dimensions:

  1. VIX hard block    -- no entries when VIX >= 35 (matches production service)
  2. Best-4 filters    -- SymbolQuality, NoRepeat14d, SkipMonday, WeeklyBreaker
  3. 100-Year Pattern  -- regime switch during midterm windows (Sep 27 - Jul 18+1yr)
  4. Multi-tier        -- combine 10-30 and 31-60 day parquets into one daily pool
  5. VIX-scaled sizing -- shrink positions when VIX is elevated (below 35 hard block)

Base configs: the 2 best combined L+S strategies from Codex V3 rerun:
  - BASE_A: win_probability / strict / risk_balanced / target6_atr2 / vol_inverse  (STK_045, Sharpe 7.11)
  - BASE_B: combo_rank / balanced / risk_balanced / target6_atr2 / vol_inverse     (STK_063, Sharpe 7.06)

Test matrix: 12 enhancement combos x 2 base configs = 24 stock runs.
Also runs options and spreads with baseline + best enhancement combo.

Usage:
    python backtest_v4_enhanced.py [--jobs N]

Outputs:
    results/backtest_v4/summary.csv
    results/backtest_v4/yearly.csv
    results/backtest_v4/report.md
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_10_30 = ROOT / "results" / "backtester_input_10_30.parquet"
DATA_31_60 = ROOT / "results" / "backtester_input_31_60.parquet"
EARNINGS_PATH = ROOT / "results" / "earnings_dates.json"
VIX_PATH = Path("C:/seasonals/data/csv/INDX/VIX.csv")
OUT_DIR = ROOT / "results" / "backtest_v4"
OUT_DIR.mkdir(exist_ok=True)

BUSINESS_DAYS = 252
STOCK_CAPITAL = 100_000.0
OPTIONS_CAPITAL = 10_000.0
SPREAD_CAPITAL = 25_000.0

# 100-Year Pattern midterm windows in backtest data
MIDTERM_WINDOWS = [
    (pd.Timestamp("2018-09-27"), pd.Timestamp("2019-07-18")),
    (pd.Timestamp("2022-09-27"), pd.Timestamp("2023-07-18")),
]
PATTERN_ML_FLOOR = 70.0          # minimum ML during regime window
PATTERN_EXCLUDE_SECTORS = frozenset({"Energy"})
PATTERN_EXTRA_POSITIONS = 2      # additional max_positions during regime window
VIX_HARD_BLOCK = 35.0
VIX_SCALE_REF = 20.0             # target VIX for 100% position size


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V4Config:
    strategy_id: str
    label: str
    # Core strategy params
    rank_key: str               # ml_score | win_probability | combo_rank
    min_ml_score: float
    min_win_probability: float
    min_predicted_return: float
    max_positions: int
    sector_cap: int
    base_weight: float
    size_mode: str              # vol_inverse | equal | confidence
    exit_profile: str           # target6_atr2 | target4_trail2 | hold
    asset_class: str            # stock | option | spread
    # Enhancement flags
    vix_hard_block: bool = False
    vix_scaled_sizing: bool = False
    skip_monday: bool = False
    no_repeat_days: int = 0
    weekly_breaker_n: int = 0   # pause if N losses in last 5 completed trades
    symbol_quality: bool = False
    pattern_regime: bool = False
    multi_tier: bool = False
    direction: str = "both"     # both | long | short
    # Options / spread params
    option_premium_pct: float = 0.025
    option_theta_mult: float = 0.10
    spread_family: str = "bull_put"
    spread_otm: float = 2.0
    spread_width: float = 8.0


# ---------------------------------------------------------------------------
# VIX data
# ---------------------------------------------------------------------------

def load_vix() -> dict[pd.Timestamp, float]:
    df = pd.read_csv(VIX_PATH, parse_dates=["date"])
    df = df[["date", "close"]].dropna()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return dict(zip(df["date"], df["close"]))


# ---------------------------------------------------------------------------
# Earnings filter (same as V3)
# ---------------------------------------------------------------------------

def load_earnings_calendar() -> dict[str, np.ndarray]:
    import json
    raw = json.loads(EARNINGS_PATH.read_text())
    out: dict[str, np.ndarray] = {}
    for symbol, dates in raw.items():
        arr = (
            pd.to_datetime(pd.Series(dates), errors="coerce")
            .dropna().sort_values().to_numpy(dtype="datetime64[D]")
        )
        if len(arr):
            out[symbol] = arr
    return out


def build_earnings_flag(df: pd.DataFrame, calendar: dict[str, np.ndarray]) -> np.ndarray:
    flags = np.zeros(len(df), dtype=bool)
    for symbol, idx in df.groupby("symbol", sort=False, observed=True).groups.items():
        earnings = calendar.get(str(symbol))
        if earnings is None:
            continue
        positions = np.asarray(idx, dtype=np.int64)
        starts = df.loc[positions, "date"].to_numpy(dtype="datetime64[D]")
        ends = df.loc[positions, "exit_date"].to_numpy(dtype="datetime64[D]")
        left = np.searchsorted(earnings, starts, side="left")
        right = np.searchsorted(earnings, ends, side="right")
        flags[positions] = right > left
    return flags


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquet(path: Path) -> pd.DataFrame:
    cols = [
        "date", "year", "symbol", "sector", "direction", "holding_days",
        "ml_score", "predicted_return", "predicted_mfe", "win_probability",
        "p_hit_return", "p_hit_mfe", "actual_return", "actual_mfe",
        "stock_volatility_20d", "atr_14d_pct",
    ]
    df = pd.read_parquet(path, columns=cols).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["exit_date"] = df["date"] + pd.to_timedelta(df["holding_days"], unit="D")
    for c in ["ml_score", "predicted_return", "predicted_mfe", "win_probability",
              "p_hit_return", "p_hit_mfe", "actual_return", "actual_mfe",
              "stock_volatility_20d", "atr_14d_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["holding_days"] = pd.to_numeric(df["holding_days"], errors="coerce").fillna(0).astype("int16")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype("int16")
    df["direction"] = df["direction"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["sector"] = df["sector"].astype(str)
    df = df.dropna(subset=["date", "symbol", "sector", "direction", "ml_score",
                            "predicted_return", "win_probability", "actual_return",
                            "actual_mfe", "atr_14d_pct"]).reset_index(drop=True)
    return df


def load_data(multi_tier: bool, earnings_calendar: dict) -> pd.DataFrame:
    df = load_parquet(DATA_10_30)
    if multi_tier:
        df31 = load_parquet(DATA_31_60)
        df31["tier"] = "31_60"
        df["tier"] = "10_30"
        df = pd.concat([df, df31], ignore_index=True)
    else:
        df["tier"] = "10_30"
    df["has_earnings"] = build_earnings_flag(df, earnings_calendar)
    df["eligible"] = ~df["has_earnings"]
    df["atr_pct_points"] = (df["atr_14d_pct"] * 100.0).astype("float32")
    # Combo rank (same formula as V3)
    df["combo_rank"] = (
        0.45 * df["ml_score"]
        + 35.0 * df["win_probability"]
        + 3.0 * df["predicted_return"]
        + 10.0 * df["p_hit_return"]
    ).astype("float32")
    return df


# ---------------------------------------------------------------------------
# Symbol quality
# ---------------------------------------------------------------------------

def add_symbol_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (symbol, year), compute average actual_return from strictly prior years.
    Normalized to 0-1 percentile rank, then scaled 0-10 to add to ranking signals.
    """
    yearly = (
        df.groupby(["symbol", "year"])["actual_return"]
        .mean()
        .reset_index()
        .rename(columns={"actual_return": "yr_ret"})
        .sort_values(["symbol", "year"])
    )

    def expanding_prior(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("year").reset_index(drop=True)
        cumrets = []
        for i, row in g.iterrows():
            prior = g.loc[g["year"] < row["year"], "yr_ret"]
            cumrets.append(prior.mean() if len(prior) > 0 else 0.0)
        g["prior_ret"] = cumrets
        return g

    yearly = yearly.groupby("symbol", group_keys=False).apply(expanding_prior)
    yearly["prior_ret"] = yearly["prior_ret"].fillna(0.0)

    # Percentile rank per year
    yearly["sym_quality_score"] = yearly.groupby("year")["prior_ret"].rank(pct=True) * 10.0

    df = df.merge(
        yearly[["symbol", "year", "sym_quality_score"]],
        on=["symbol", "year"], how="left"
    )
    df["sym_quality_score"] = df["sym_quality_score"].fillna(5.0).astype("float32")
    return df


# ---------------------------------------------------------------------------
# Return builders (same logic as V3)
# ---------------------------------------------------------------------------

def build_stock_return(df: pd.DataFrame, profile: str) -> np.ndarray:
    actual = df["actual_return"].to_numpy(dtype=float)
    mfe = df["actual_mfe"].to_numpy(dtype=float)
    atr = df["atr_pct_points"].to_numpy(dtype=float)
    if profile == "hold":
        return actual
    if profile == "target4_trail2":
        target_hit = mfe >= 4.0
        trail_hit = (mfe >= 2.5) & ((mfe - actual) >= 2.0)
        return np.where(target_hit, 4.0, np.where(trail_hit, np.maximum(actual, mfe - 2.0), actual))
    if profile == "target6_atr2":
        target_hit = mfe >= 6.0
        stop_level = -2.0 * atr
        return np.where(target_hit, 6.0, np.maximum(actual, stop_level))
    raise ValueError(profile)


def build_option_return(df: pd.DataFrame, premium_pct: float, theta_mult: float) -> np.ndarray:
    actual = df["actual_return"].to_numpy(dtype=float) / 100.0
    mfe = df["actual_mfe"].to_numpy(dtype=float) / 100.0
    holding = df["holding_days"].to_numpy(dtype=float)
    gross = 0.55 * np.maximum(actual, 0.0) + 0.30 * np.maximum(mfe, 0.0)
    pnl = (gross - premium_pct) / premium_pct
    theta_drag = theta_mult * (holding / 30.0)
    return np.clip((pnl - theta_drag) * 100.0, -100.0, 300.0)


def build_spread_return(df: pd.DataFrame, family: str, otm: float, width: float) -> np.ndarray:
    actual = df["actual_return"].to_numpy(dtype=float)
    mfe = df["actual_mfe"].to_numpy(dtype=float)
    if family == "bull_call":
        debit = 0.40 * width
        terminal = np.clip(actual, 0.0, width)
        early = np.where(mfe >= (otm + 0.5 * width), 0.60 * width, terminal)
        return np.clip(((early - debit) / debit) * 100.0, -100.0, 150.0)
    if family == "bull_put":
        credit = 0.32 * width
        risk = width - credit
        intrinsic_loss = np.clip((-actual) - otm, 0.0, width)
        return np.clip(((credit - intrinsic_loss) / risk) * 100.0, -100.0, 60.0)
    raise ValueError(family)


def compute_trade_return(df: pd.DataFrame, config: V4Config) -> np.ndarray:
    if config.asset_class == "stock":
        return build_stock_return(df, config.exit_profile)
    if config.asset_class == "option":
        return build_option_return(df, config.option_premium_pct, config.option_theta_mult)
    if config.asset_class == "spread":
        return build_spread_return(df, config.spread_family, config.spread_otm, config.spread_width)
    raise ValueError(config.asset_class)


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def is_in_pattern_window(date: pd.Timestamp) -> bool:
    return any(s <= date <= e for s, e in MIDTERM_WINDOWS)


def pick_candidates(
    df: pd.DataFrame,
    config: V4Config,
    vix_map: dict[pd.Timestamp, float],
) -> pd.DataFrame:
    work = df.copy()
    work["in_regime"] = work["date"].apply(is_in_pattern_window) if config.pattern_regime else False

    # Determine per-row thresholds (regime switch lowers ML floor during window)
    if config.pattern_regime:
        effective_ml = np.where(
            work["in_regime"].to_numpy(),
            np.minimum(config.min_ml_score, PATTERN_ML_FLOOR),
            config.min_ml_score,
        )
    else:
        effective_ml = config.min_ml_score

    # Direction filter
    if config.direction == "long":
        dir_mask = work["direction"] == "l"
    elif config.direction == "short":
        dir_mask = work["direction"] == "s"
    else:
        dir_mask = pd.Series(True, index=work.index)

    # Pattern regime: long-only + no Energy during window
    if config.pattern_regime:
        regime_dir_ok = np.where(
            work["in_regime"].to_numpy(),
            work["direction"].to_numpy() == "l",
            dir_mask.to_numpy(),
        )
        regime_sector_ok = np.where(
            work["in_regime"].to_numpy(),
            ~work["sector"].isin(PATTERN_EXCLUDE_SECTORS).to_numpy(),
            True,
        )
    else:
        regime_dir_ok = dir_mask.to_numpy()
        regime_sector_ok = np.ones(len(work), dtype=bool)

    # VIX hard block
    if config.vix_hard_block:
        vix_ok = work["date"].map(lambda d: vix_map.get(d, 0.0) < VIX_HARD_BLOCK).to_numpy()
    else:
        vix_ok = np.ones(len(work), dtype=bool)

    # Skip Monday
    if config.skip_monday:
        monday_ok = (work["date"].dt.dayofweek != 0).to_numpy()
    else:
        monday_ok = np.ones(len(work), dtype=bool)

    ml_ok = work["ml_score"].to_numpy() >= effective_ml
    wp_ok = work["win_probability"].to_numpy() >= config.min_win_probability
    pr_ok = work["predicted_return"].to_numpy() >= config.min_predicted_return

    mask = (
        work["eligible"].to_numpy()
        & ml_ok & wp_ok & pr_ok
        & regime_dir_ok & regime_sector_ok
        & vix_ok & monday_ok
    )

    cols = [
        "date", "exit_date", "year", "symbol", "sector", "direction",
        "holding_days", "win_probability", "stock_volatility_20d",
        "trade_return_pct", "in_regime",
    ]
    if config.symbol_quality:
        cols.append("sym_quality_score")

    selected = work.loc[mask, cols].copy()
    if selected.empty:
        return selected

    # Build rank value
    rank_col = work.loc[mask, config.rank_key].to_numpy()
    if config.symbol_quality:
        rank_col = rank_col + 0.5 * selected["sym_quality_score"].to_numpy()
    selected["rank_value"] = rank_col

    selected = selected.sort_values(["date", "rank_value"], ascending=[True, False])
    selected = selected.drop_duplicates(["date", "symbol", "direction"], keep="first")
    return selected


# ---------------------------------------------------------------------------
# Portfolio simulation with V4 enhancements
# ---------------------------------------------------------------------------

def position_weight(row: pd.Series, config: V4Config, vix: float) -> float:
    w = config.base_weight
    if config.size_mode == "vol_inverse":
        vol = max(15.0, float(row["stock_volatility_20d"]) * 100.0)
        w = min(max(w * (25.0 / vol), w * 0.6), w * 1.4)
    elif config.size_mode == "confidence":
        scale = max(0.75, min(1.35, float(row["win_probability"]) / config.min_win_probability))
        w = min(w * scale, 0.24)
    if config.vix_scaled_sizing and vix > 0:
        vix_scale = min(1.0, VIX_SCALE_REF / max(vix, 10.0))
        w = w * vix_scale
    return w


def simulate_portfolio(
    candidates: pd.DataFrame,
    config: V4Config,
    starting_capital: float,
    vix_map: dict[pd.Timestamp, float],
) -> dict:
    if candidates.empty:
        return _empty_result(starting_capital)

    by_date = {pd.Timestamp(date): frame for date, frame in candidates.groupby("date", sort=True)}
    event_dates = sorted({pd.Timestamp(d) for d in candidates["date"]}
                         | {pd.Timestamp(d) for d in candidates["exit_date"]})

    cash = starting_capital
    open_positions: list[dict] = []
    equity_points: list[tuple] = []
    trade_returns: list[float] = []
    symbol_open: set[str] = set()
    last_exit_date: dict[str, pd.Timestamp] = {}   # for no-repeat filter
    recent_outcomes: list[bool] = []               # True=win, for weekly breaker
    breaker_pause_until: pd.Timestamp | None = None

    for current_date in event_dates:
        # Close expired positions
        survivors = []
        for pos in open_positions:
            if pos["exit_date"] <= current_date:
                cash += pos["allocation"] * (1.0 + pos["return_pct"] / 100.0)
                ret = pos["return_pct"]
                trade_returns.append(ret)
                symbol_open.discard(pos["symbol"])
                last_exit_date[pos["symbol"]] = current_date
                recent_outcomes.append(ret > 0)
                if len(recent_outcomes) > 10:
                    recent_outcomes.pop(0)
            else:
                survivors.append(pos)
        open_positions = survivors

        equity_now = cash + sum(p["allocation"] for p in open_positions)

        # Weekly breaker: pause if last 5 completed trades have N losses
        if config.weekly_breaker_n > 0 and len(recent_outcomes) >= 5:
            last5 = recent_outcomes[-5:]
            losses = sum(1 for win in last5 if not win)
            if losses >= config.weekly_breaker_n:
                breaker_pause_until = current_date + pd.Timedelta(days=4)

        paused = breaker_pause_until is not None and current_date <= breaker_pause_until

        # Regime: increase max_positions during pattern window
        max_pos = config.max_positions
        if config.pattern_regime and is_in_pattern_window(current_date):
            max_pos += PATTERN_EXTRA_POSITIONS

        if current_date in by_date and len(open_positions) < max_pos and not paused:
            vix_today = vix_map.get(current_date, 20.0)
            sector_counts: dict[str, int] = {}
            for pos in open_positions:
                sector_counts[pos["sector"]] = sector_counts.get(pos["sector"], 0) + 1
            slots = max_pos - len(open_positions)

            for _, row in by_date[current_date].iterrows():
                if slots <= 0:
                    break
                sym = str(row["symbol"])
                sec = str(row["sector"])
                if sym in symbol_open:
                    continue
                if sector_counts.get(sec, 0) >= config.sector_cap:
                    continue
                # No-repeat filter
                if config.no_repeat_days > 0 and sym in last_exit_date:
                    days_since = (current_date - last_exit_date[sym]).days
                    if days_since < config.no_repeat_days:
                        continue
                alloc = min(equity_now * position_weight(row, config, vix_today), cash)
                if alloc < equity_now * 0.02:
                    continue
                open_positions.append({
                    "exit_date": row["exit_date"],
                    "allocation": alloc,
                    "return_pct": float(row["trade_return_pct"]),
                    "symbol": sym,
                    "sector": sec,
                })
                cash -= alloc
                symbol_open.add(sym)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
                slots -= 1

        equity_points.append((pd.Timestamp(current_date),
                               cash + sum(p["allocation"] for p in open_positions)))

    # Flush remaining positions
    if open_positions:
        final_date = max(p["exit_date"] for p in open_positions)
        for pos in open_positions:
            cash += pos["allocation"] * (1.0 + pos["return_pct"] / 100.0)
            trade_returns.append(pos["return_pct"])
        equity_points.append((pd.Timestamp(final_date), cash))

    return _compute_metrics(equity_points, trade_returns, starting_capital, candidates)


def _empty_result(capital: float) -> dict:
    return {
        "ending_capital": capital, "annualized_return": 0.0, "sharpe": 0.0,
        "max_drawdown": 0.0, "win_rate": 0.0, "trade_count": 0,
        "avg_trade_return": 0.0, "all_years_profitable": False,
        "worst_year": None, "worst_year_return": None, "yearly_returns": {},
    }


def _compute_metrics(equity_points, trade_returns, starting_capital, candidates) -> dict:
    equity = pd.Series({d: v for d, v in equity_points}).sort_index().groupby(level=0).last()
    bidx = pd.date_range(equity.index.min(), equity.index.max(), freq="B")
    equity = equity.reindex(bidx).ffill().fillna(starting_capital)
    daily_ret = equity.pct_change().fillna(0.0)
    sharpe = float(np.sqrt(BUSINESS_DAYS) * daily_ret.mean() / daily_ret.std(ddof=0)) \
        if daily_ret.std(ddof=0) > 0 else 0.0
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min() * 100.0)
    total_days = max((equity.index.max() - equity.index.min()).days, 1)
    ending = float(equity.iloc[-1])
    ann_ret = float(((ending / starting_capital) ** (365.25 / total_days) - 1.0) * 100.0)
    # Yearly returns
    yearly: dict[int, float] = {}
    for year in sorted(candidates["year"].dropna().unique()):
        ye = equity[equity.index.year == int(year)]
        if len(ye) >= 2:
            yearly[int(year)] = float((ye.iloc[-1] / ye.iloc[0] - 1.0) * 100.0)
    positives = [v > 0 for v in yearly.values()]
    tr_arr = np.array(trade_returns)
    return {
        "ending_capital": ending,
        "annualized_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": float((tr_arr > 0).mean() * 100.0) if len(tr_arr) else 0.0,
        "trade_count": len(trade_returns),
        "avg_trade_return": float(tr_arr.mean()) if len(tr_arr) else 0.0,
        "all_years_profitable": bool(positives and all(positives) and len(yearly) >= 7),
        "worst_year": min(yearly, key=yearly.get) if yearly else None,
        "worst_year_return": min(yearly.values()) if yearly else None,
        "yearly_returns": yearly,
    }


# ---------------------------------------------------------------------------
# Run single config
# ---------------------------------------------------------------------------

def run_config(config: V4Config, earnings_calendar: dict, vix_map: dict) -> dict:
    try:
        df = load_data(config.multi_tier, earnings_calendar)
        df["trade_return_pct"] = compute_trade_return(df, config)
        if config.symbol_quality:
            df = add_symbol_quality(df)
        capital = STOCK_CAPITAL if config.asset_class == "stock" else \
                  OPTIONS_CAPITAL if config.asset_class == "option" else SPREAD_CAPITAL
        candidates = pick_candidates(df, config, vix_map)
        result = simulate_portfolio(candidates, config, capital, vix_map)
        yr = result["yearly_returns"]
        row = {
            "strategy_id": config.strategy_id,
            "label": config.label,
            "asset_class": config.asset_class,
            "direction": config.direction,
            "exit": config.exit_profile,
            "sharpe": round(result["sharpe"], 4),
            "max_drawdown": round(result["max_drawdown"], 4),
            "win_rate": round(result["win_rate"], 2),
            "annualized_return": round(result["annualized_return"], 4),
            "trade_count": result["trade_count"],
            "avg_trade_return": round(result["avg_trade_return"], 4),
            "all_years_profitable": result["all_years_profitable"],
            "worst_year": result["worst_year"],
            "worst_year_return": round(result["worst_year_return"], 2) if result["worst_year_return"] else None,
            "vix_hard_block": config.vix_hard_block,
            "vix_scaled_sizing": config.vix_scaled_sizing,
            "skip_monday": config.skip_monday,
            "no_repeat_days": config.no_repeat_days,
            "weekly_breaker": config.weekly_breaker_n,
            "symbol_quality": config.symbol_quality,
            "pattern_regime": config.pattern_regime,
            "multi_tier": config.multi_tier,
        }
        # Yearly columns
        for y in range(2018, 2026):
            row[f"y{y}"] = round(yr.get(y, float("nan")), 2)
        return row
    except Exception as exc:
        return {"strategy_id": config.strategy_id, "label": config.label,
                "error": str(exc), "sharpe": float("nan")}


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

# Best Codex V3 base params
BASE_CONFIGS = {
    "A": dict(  # STK_045 equivalent
        rank_key="win_probability",
        min_ml_score=80.0, min_win_probability=0.74, min_predicted_return=2.0,
        max_positions=10, sector_cap=3, base_weight=0.10,
        size_mode="vol_inverse", exit_profile="target6_atr2",
    ),
    "B": dict(  # STK_063 equivalent
        rank_key="combo_rank",
        min_ml_score=70.0, min_win_probability=0.68, min_predicted_return=1.5,
        max_positions=10, sector_cap=3, base_weight=0.10,
        size_mode="vol_inverse", exit_profile="target6_atr2",
    ),
}

# Enhancement combos to test
ENHANCEMENTS = [
    ("01_baseline",       dict()),
    ("02_vix_block",      dict(vix_hard_block=True)),
    ("03_sym_quality",    dict(symbol_quality=True)),
    ("04_no_repeat14",    dict(no_repeat_days=14)),
    ("05_skip_monday",    dict(skip_monday=True)),
    ("06_wkly_breaker",   dict(weekly_breaker_n=3)),
    ("07_best4",          dict(symbol_quality=True, no_repeat_days=14,
                               skip_monday=True, weekly_breaker_n=3)),
    ("08_vix_best4",      dict(vix_hard_block=True, symbol_quality=True,
                               no_repeat_days=14, skip_monday=True, weekly_breaker_n=3)),
    ("09_regime",         dict(pattern_regime=True)),
    ("10_vix_regime",     dict(vix_hard_block=True, pattern_regime=True)),
    ("11_vix_best4_reg",  dict(vix_hard_block=True, symbol_quality=True, no_repeat_days=14,
                               skip_monday=True, weekly_breaker_n=3, pattern_regime=True)),
    ("12_full",           dict(vix_hard_block=True, symbol_quality=True, no_repeat_days=14,
                               skip_monday=True, weekly_breaker_n=3, pattern_regime=True,
                               multi_tier=True)),
]

# Direction variants for baseline
DIRECTIONS = [("both", "both"), ("long", "long"), ("short", "short")]


def make_stock_configs() -> list[V4Config]:
    configs = []
    for base_name, base_params in BASE_CONFIGS.items():
        for enh_name, enh_params in ENHANCEMENTS:
            for dir_label, dir_val in DIRECTIONS:
                sid = f"V4_STK_{base_name}_{enh_name}_{dir_label[:1].upper()}"
                label = f"Base{base_name} | {enh_name} | {dir_label}"
                merged = {**base_params, **enh_params, "direction": dir_val}
                configs.append(V4Config(
                    strategy_id=sid,
                    label=label,
                    asset_class="stock",
                    **merged,
                ))
    return configs


def make_options_configs() -> list[V4Config]:
    """Top options config from Codex V3 + best enhancement combo."""
    base_opt = dict(
        rank_key="combo_rank",
        min_ml_score=80.0, min_win_probability=0.74, min_predicted_return=2.0,
        max_positions=6, sector_cap=3, base_weight=0.08,
        size_mode="vol_inverse", exit_profile="option",
        asset_class="option",
        option_premium_pct=0.025, option_theta_mult=0.10,
    )
    configs = []
    for enh_name, enh_params in [
        ("01_baseline", {}),
        ("08_vix_best4", dict(vix_hard_block=True, symbol_quality=True,
                              no_repeat_days=14, skip_monday=True, weekly_breaker_n=3)),
        ("11_vix_best4_reg", dict(vix_hard_block=True, symbol_quality=True, no_repeat_days=14,
                                  skip_monday=True, weekly_breaker_n=3, pattern_regime=True)),
    ]:
        for dir_label, dir_val in DIRECTIONS:
            sid = f"V4_OPT_{enh_name}_{dir_label[:1].upper()}"
            configs.append(V4Config(
                strategy_id=sid,
                label=f"Options | {enh_name} | {dir_label}",
                direction=dir_val,
                **{**base_opt, **enh_params},
            ))
    return configs


def make_spread_configs() -> list[V4Config]:
    """Top spread config from Codex V3 (bull_put_2.0_8.0) + best enhancement combo."""
    base_spr = dict(
        rank_key="combo_rank",
        min_ml_score=80.0, min_win_probability=0.74, min_predicted_return=2.0,
        max_positions=8, sector_cap=3, base_weight=0.10,
        size_mode="vol_inverse", exit_profile="spread",
        asset_class="spread",
        spread_family="bull_put", spread_otm=2.0, spread_width=8.0,
    )
    configs = []
    for enh_name, enh_params in [
        ("01_baseline", {}),
        ("08_vix_best4", dict(vix_hard_block=True, symbol_quality=True,
                              no_repeat_days=14, skip_monday=True, weekly_breaker_n=3)),
        ("11_vix_best4_reg", dict(vix_hard_block=True, symbol_quality=True, no_repeat_days=14,
                                  skip_monday=True, weekly_breaker_n=3, pattern_regime=True)),
    ]:
        for dir_label, dir_val in DIRECTIONS:
            sid = f"V4_SPR_{enh_name}_{dir_label[:1].upper()}"
            configs.append(V4Config(
                strategy_id=sid,
                label=f"Spread | {enh_name} | {dir_label}",
                direction=dir_val,
                **{**base_spr, **enh_params},
            ))
    return configs


# ---------------------------------------------------------------------------
# Holdout check
# ---------------------------------------------------------------------------

def run_holdout(config: V4Config, earnings_calendar: dict, vix_map: dict) -> list[dict]:
    rows = []
    for period, yr_filter in [("2018_2024", lambda y: y <= 2024), ("2025", lambda y: y == 2025)]:
        df = load_data(config.multi_tier, earnings_calendar)
        df = df[df["year"].apply(yr_filter)].reset_index(drop=True)
        df["trade_return_pct"] = compute_trade_return(df, config)
        if config.symbol_quality:
            df = add_symbol_quality(df)
        capital = STOCK_CAPITAL if config.asset_class == "stock" else \
                  OPTIONS_CAPITAL if config.asset_class == "option" else SPREAD_CAPITAL
        candidates = pick_candidates(df, config, vix_map)
        result = simulate_portfolio(candidates, config, capital, vix_map)
        rows.append({
            "strategy_id": config.strategy_id,
            "label": config.label,
            "period": period,
            "sharpe": round(result["sharpe"], 4),
            "max_drawdown": round(result["max_drawdown"], 4),
            "annualized_return": round(result["annualized_return"], 4),
            "win_rate": round(result["win_rate"], 2),
            "trade_count": result["trade_count"],
        })
    return rows


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def render_table(df: pd.DataFrame, cols: list[str] | None = None, n: int | None = None) -> str:
    work = df[cols].copy() if cols else df.copy()
    if n:
        work = work.head(n)
    hdrs = list(work.columns)
    lines = ["| " + " | ".join(hdrs) + " |",
             "| " + " | ".join(["---"] * len(hdrs)) + " |"]
    for _, row in work.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, yearly: pd.DataFrame, holdout: pd.DataFrame) -> None:
    stock = summary[summary["asset_class"] == "stock"].copy()
    opts = summary[summary["asset_class"] == "option"].copy()
    sprs = summary[summary["asset_class"] == "spread"].copy()

    year_cols = [c for c in yearly.columns if c.startswith("y")]
    key_cols = ["strategy_id", "label", "direction", "sharpe", "max_drawdown",
                "win_rate", "annualized_return", "trade_count", "all_years_profitable",
                "worst_year_return"]

    lines = [
        "# V4 Enhanced Backtest Results\n",
        "Objective: highest return at highest stability and safety.\n",
        "Base: Codex V3 best configs (Sharpe 7.11). Enhancements: VIX block, Best-4 filters, ",
        "100-Year Pattern regime, multi-tier, VIX-scaled sizing.\n",
        f"Run date: {pd.Timestamp.now().date()}\n\n",
        "---\n",
        "## Stock Strategy Results\n",
        "### All Configs Ranked by Sharpe\n",
        render_table(stock.sort_values("sharpe", ascending=False), key_cols, 40),
        "\n\n### Top 10 Combined L+S\n",
        render_table(stock[stock["direction"] == "both"].sort_values("sharpe", ascending=False),
                     key_cols, 10),
        "\n\n### Top 10 Long-Only\n",
        render_table(stock[stock["direction"] == "long"].sort_values("sharpe", ascending=False),
                     key_cols, 10),
        "\n\n### Top 10 Short-Only\n",
        render_table(stock[stock["direction"] == "short"].sort_values("sharpe", ascending=False),
                     key_cols, 10),
        "\n\n### Enhancement Impact on BaseA (Combined L+S)\n",
    ]

    base_a = stock[(stock["strategy_id"].str.contains("_A_")) & (stock["direction"] == "both")]
    lines.append(render_table(base_a.sort_values("sharpe", ascending=False),
                              ["label", "sharpe", "max_drawdown", "win_rate",
                               "annualized_return", "trade_count", "all_years_profitable"]))

    lines += [
        "\n\n### Enhancement Impact on BaseB (Combined L+S)\n",
    ]
    base_b = stock[(stock["strategy_id"].str.contains("_B_")) & (stock["direction"] == "both")]
    lines.append(render_table(base_b.sort_values("sharpe", ascending=False),
                              ["label", "sharpe", "max_drawdown", "win_rate",
                               "annualized_return", "trade_count", "all_years_profitable"]))

    lines += [
        "\n\n### Year-by-Year Returns (Top 5 Combined)\n",
    ]
    top5 = stock[stock["direction"] == "both"].nlargest(5, "sharpe")["strategy_id"].tolist()
    yr_top = yearly[yearly["strategy_id"].isin(top5)][["strategy_id", "label"] + year_cols]
    lines.append(render_table(yr_top))

    lines += [
        "\n\n### Holdout Check (Top 5: 2018-2024 train vs 2025 out-of-sample)\n",
    ]
    top5_holdout = holdout[holdout["strategy_id"].isin(top5)]
    lines.append(render_table(top5_holdout))

    lines += [
        "\n\n---\n",
        "## Options Strategy Results\n",
        render_table(opts.sort_values("sharpe", ascending=False), key_cols),
        "\n\n---\n",
        "## Spread Strategy Results\n",
        render_table(sprs.sort_values("sharpe", ascending=False), key_cols),
        "\n\n---\n",
        "## Key Findings\n",
    ]

    if not stock.empty:
        best = stock[stock["direction"] == "both"].sort_values("sharpe", ascending=False).iloc[0]
        lines.append(f"- Best combined stock: `{best['strategy_id']}` | "
                     f"Sharpe {best['sharpe']:.2f} | DD {best['max_drawdown']:.2f}% | "
                     f"CAGR {best['annualized_return']:.2f}% | WR {best['win_rate']:.1f}%\n")
        baseline = stock[(stock["strategy_id"].str.contains("01_baseline")) &
                         (stock["direction"] == "both")]
        if not baseline.empty:
            bl = baseline.iloc[0]
            lines.append(f"- Baseline (no enhancements): Sharpe {bl['sharpe']:.2f} | "
                         f"DD {bl['max_drawdown']:.2f}%\n")
            lines.append(f"- Enhancement lift: +{best['sharpe'] - bl['sharpe']:.2f} Sharpe, "
                         f"{best['max_drawdown'] - bl['max_drawdown']:.2f}pp DD change\n")

    report = "\n".join(lines)
    out_path = OUT_DIR / "report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    print("Loading VIX and earnings...")
    vix_map = load_vix()
    earnings_calendar = load_earnings_calendar()

    all_configs = make_stock_configs() + make_options_configs() + make_spread_configs()
    print(f"Running {len(all_configs)} configs on {args.jobs} workers...")

    t0 = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_config)(cfg, earnings_calendar, vix_map)
        for cfg in all_configs
    )
    print(f"Completed in {(time.time() - t0)/60:.1f} min")

    summary = pd.DataFrame([r for r in results if "error" not in r])
    year_cols = [c for c in summary.columns if c.startswith("y")]
    yearly = summary[["strategy_id", "label"] + year_cols].copy()
    summary_out = summary.drop(columns=year_cols)

    summary_out.to_csv(OUT_DIR / "summary.csv", index=False)
    yearly.to_csv(OUT_DIR / "yearly.csv", index=False)

    # Holdout check for top 10 combined stock configs
    print("Running holdout checks...")
    top_ids = set(
        summary_out[(summary_out["asset_class"] == "stock") &
                    (summary_out["direction"] == "both")]
        .nlargest(10, "sharpe")["strategy_id"].tolist()
    )
    top_configs = {c.strategy_id: c for c in all_configs}
    holdout_rows = []
    for sid in top_ids:
        cfg = top_configs[sid]
        holdout_rows.extend(run_holdout(cfg, earnings_calendar, vix_map))
    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(OUT_DIR / "holdout.csv", index=False)

    write_report(summary_out, yearly, holdout)
    print(f"\nDone. Results in {OUT_DIR}")

    # Print quick summary
    stk = summary_out[(summary_out["asset_class"] == "stock") &
                      (summary_out["direction"] == "both")].nlargest(5, "sharpe")
    print("\n=== TOP 5 COMBINED STOCK CONFIGS ===")
    print(stk[["strategy_id", "label", "sharpe", "max_drawdown",
               "win_rate", "annualized_return", "all_years_profitable"]].to_string(index=False))


if __name__ == "__main__":
    main()
