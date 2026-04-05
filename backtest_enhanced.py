"""
Enhanced Stock Strategy Backtester with 7 Improvements

Tests each improvement independently and in combination against the baseline.
Improvements: MFE-adaptive spreads, no-repeat rule, Monday skip, VIX-scaled sizing,
weekly loss breaker, multi-tier, symbol quality scores.

Usage: python backtest_enhanced.py [--jobs 12]
"""

import argparse
import hashlib
import json
import math
import time
import warnings
from datetime import timedelta, date as _date
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATA_DIR = Path("C:/seasonals/data")
BACKTEST_DIR = RESULTS / "backtest_enhanced"

INITIAL_CAPITAL = 100_000
SLIPPAGE = 0.002
CASH_RESERVE = 0.10
EP_TRAIL_PCT = 0.03
EM_TRAIL_FACTOR = 0.50
HARD_STOP_PCT = 0.10
DRAWDOWN_HALT_PCT = 0.15
DRAWDOWN_HALT_DAYS = 20

# ============================================================
# Improvement flags (each can be toggled independently)
# ============================================================

DEFAULT_IMPROVEMENTS = {
    "mfe_adaptive_spreads": False,  # #1: Use predicted_mfe for debit spread width
    "no_repeat_14d": False,         # #2: No same-symbol within 14 days
    "skip_monday": False,           # #3: Skip Monday entries
    "vix_scaled_sizing": False,     # #4: Scale size by VIX level
    "weekly_loss_breaker": False,   # #5: Pause after 3 losses in 5 days
    "symbol_quality": False,        # #7: Penalize low-quality symbols in ranking
}


# ============================================================
# Data Loading
# ============================================================

def load_data(tier="10_30"):
    path = RESULTS / f"backtester_input_{tier}.parquet"
    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    df = df[df["direction"] == "l"].reset_index(drop=True)
    for col in ["predicted_return", "predicted_mfe", "actual_return", "actual_mfe"]:
        df[col] = df[col] / 100.0
    df["date"] = pd.to_datetime(df["date"]).dt.date
    print(f"  {len(df):,} long opportunities")
    return df


def load_prices(symbols):
    prices = {}
    for sym in symbols:
        p_path = DATA_DIR / "csv" / "US" / f"{sym}.csv"
        if not p_path.exists():
            continue
        try:
            p = pd.read_csv(p_path, usecols=["date", "close", "volume"],
                            parse_dates=["date"])
            p = p[(p["date"] >= "2017-01-01") & (p["date"] <= "2026-12-31")]
            p = p.set_index("date").sort_index()
            p.index = p.index.date
            prices[sym] = p["close"]
        except Exception:
            pass
    return prices


def load_earnings():
    path = RESULTS / "earnings_dates.json"
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    earnings = {}
    for sym, dates in raw.items():
        earnings[sym] = set()
        for d in dates:
            try:
                y, m, day = d.split("-")
                earnings[sym].add(_date(int(y), int(m), int(day)))
            except Exception:
                pass
    return earnings


def load_vix():
    """Load VIX daily closes for VIX-scaled sizing."""
    vix_path = DATA_DIR / "csv" / "INDX" / "VIX.csv"
    if not vix_path.exists():
        return {}
    v = pd.read_csv(vix_path, usecols=["date", "close"], parse_dates=["date"])
    v = v.set_index("date").sort_index()
    v.index = v.index.date
    return v["close"].to_dict()


def filter_earnings(df, earnings_map):
    if not earnings_map:
        return df
    print("  Filtering earnings...")
    earnings_ord = {}
    for sym, dates in earnings_map.items():
        if dates:
            earnings_ord[sym] = np.array(sorted(d.toordinal() for d in dates))

    keep = np.ones(len(df), dtype=bool)
    for sym, grp in df.groupby("symbol"):
        earr = earnings_ord.get(sym)
        if earr is None:
            continue
        entry_ords = np.array([d.toordinal() for d in grp["date"].values])
        exit_ords = entry_ords + grp["holding_days"].values.astype(int)
        idx = np.searchsorted(earr, entry_ords, side="left")
        conflict = np.zeros(len(grp), dtype=bool)
        v = idx < len(earr)
        conflict[v] = earr[idx[v]] <= exit_ords[v]
        v2 = idx > 0
        conflict[v2] |= earr[np.clip(idx[v2] - 1, 0, len(earr) - 1)] >= entry_ords[v2]
        keep[grp.index.values[conflict]] = False

    n_filt = (~keep).sum()
    print(f"  {n_filt:,}/{len(df):,} filtered by earnings ({n_filt/len(df):.1%})")
    return df[keep].reset_index(drop=True)


def compute_symbol_quality(df):
    """Compute rolling symbol quality scores from prior-year data."""
    print("  Computing symbol quality scores...")
    quality = {}  # {(symbol, year): score}
    for year in range(2018, 2026):
        prior = df[df["year"] < year]
        if len(prior) < 1000:
            prior = df
        sym_stats = prior.groupby("symbol")["actual_return"].agg(["mean", "count"])
        sym_stats = sym_stats[sym_stats["count"] >= 10]
        # Score = mean return (winsorized)
        scores = sym_stats["mean"].clip(-0.10, 0.10)
        for sym, score in scores.items():
            quality[(sym, year)] = score
    return quality


def build_candidates(df):
    candidates = {}
    cols = ["symbol", "sector", "holding_days", "ml_score",
            "predicted_return", "predicted_mfe", "win_probability",
            "actual_return", "actual_mfe", "stock_volatility_20d", "year",
            "atr_14d_pct"]
    for date_val, group in df.groupby("date"):
        candidates[date_val] = group[cols].to_dict("records")
    return candidates


def precompute_kelly_r(df):
    table = {}
    for thresh in [70, 80, 85, 90]:
        filt = df[df["ml_score"] >= thresh]
        for year in range(2018, 2026):
            prior = filt[filt["year"] < year]
            if len(prior) < 100:
                prior = filt
            w = prior.loc[prior["actual_return"] > 0, "actual_return"]
            l = prior.loc[prior["actual_return"] <= 0, "actual_return"].abs()
            table[(thresh, year)] = float(w.mean() / l.mean()) if len(w) > 10 and len(l) > 10 else 1.3
    return table


# ============================================================
# Ranking with symbol quality
# ============================================================

def rank_candidates(candidates, method, symbol_quality=None, year=None):
    if not candidates:
        return []

    # Apply symbol quality adjustment if enabled
    if symbol_quality and year:
        for c in candidates:
            sq = symbol_quality.get((c["symbol"], year), 0.0)
            c["_sq"] = sq
    else:
        for c in candidates:
            c["_sq"] = 0.0

    if method == "WP":
        key = lambda c: c["win_probability"] + c["_sq"] * 2
        return sorted(candidates, key=key, reverse=True)
    elif method == "CW":
        n = len(candidates)
        if n <= 1:
            return list(candidates)
        wp = np.array([c["win_probability"] for c in candidates])
        pr = np.array([c["predicted_return"] for c in candidates])
        mg = np.array([c["predicted_mfe"] - c["predicted_return"] for c in candidates])
        sq = np.array([c["_sq"] for c in candidates])
        wp_r = rankdata(wp) / n
        pr_r = rankdata(pr) / n
        mg_r = rankdata(mg) / n
        sq_r = rankdata(sq) / n if sq.std() > 0 else np.full(n, 0.5)
        # Add symbol quality as 10% weight, reduce others proportionally
        scores = 0.54 * wp_r + 0.225 * pr_r + 0.135 * mg_r + 0.10 * sq_r
        return [candidates[i] for i in np.argsort(-scores)]
    elif method == "CR":
        n = len(candidates)
        if n <= 1:
            return list(candidates)
        wp = np.array([c["win_probability"] for c in candidates])
        pr = np.array([c["predicted_return"] for c in candidates])
        mg = np.array([c["predicted_mfe"] - c["predicted_return"] for c in candidates])
        sq = np.array([c["_sq"] for c in candidates])
        wp_r = rankdata(wp) / n
        pr_r = rankdata(pr) / n
        mg_r = rankdata(mg) / n
        sq_r = rankdata(sq) / n if sq.std() > 0 else np.full(n, 0.5)
        scores = 0.27 * wp_r + 0.45 * pr_r + 0.18 * mg_r + 0.10 * sq_r
        return [candidates[i] for i in np.argsort(-scores)]
    else:
        return sorted(candidates, key=lambda c: c["win_probability"], reverse=True)


# ============================================================
# Exit rules (same as stock backtester)
# ============================================================

def check_exit(pos, cum_return, td_idx, exit_rule):
    if cum_return > pos["hwm"]:
        pos["hwm"] = cum_return
    is_last = td_idx >= pos["max_td"]

    if exit_rule == "EP":
        stop = pos["hwm"] - EP_TRAIL_PCT
        if cum_return <= stop and td_idx >= 2:
            return ("exit", stop, "pct_trail")
        if is_last:
            return ("exit", cum_return, "hold_expiry")
        return ("hold", None, None)

    elif exit_rule == "EM":
        pr = pos["predicted_return"]
        if not pos["pred_reached"] and cum_return >= pr:
            pos["pred_reached"] = True
            pos["trail_stop"] = pr
        if pos["pred_reached"]:
            pos["trail_stop"] = pr + (pos["hwm"] - pr) * EM_TRAIL_FACTOR
            if cum_return <= pos["trail_stop"]:
                return ("exit", pos["trail_stop"], "trailing_stop")
        if is_last:
            return ("exit", cum_return, "hold_expiry")
        return ("hold", None, None)

    elif exit_rule == "ET":
        tl = int(pos["max_td"] * 0.60)
        if td_idx >= tl and not pos["pred_reached"]:
            return ("exit", cum_return, "time_limit")
        pr = pos["predicted_return"]
        if not pos["pred_reached"] and cum_return >= pr:
            pos["pred_reached"] = True
            pos["trail_stop"] = pr
        if pos["pred_reached"]:
            pos["trail_stop"] = pr + (pos["hwm"] - pr) * EM_TRAIL_FACTOR
            if cum_return <= pos["trail_stop"]:
                return ("exit", pos["trail_stop"], "trailing_stop")
        if is_last:
            return ("exit", cum_return, "hold_expiry")
        return ("hold", None, None)

    return ("hold", None, None)


# ============================================================
# Simulation Engine
# ============================================================

def run_strategy(config, candidates_by_date, prices, trading_days, kelly_r_table,
                 improvements, vix_daily, symbol_quality):
    sid = config["id"]
    threshold = config["threshold"]
    exit_rule = config["exit"]
    sizing = config["sizing"]
    max_pos = config["max_positions"]
    ranking = config["ranking"]
    conc = config.get("concentration", "C1")

    cash = float(INITIAL_CAPITAL)
    open_positions = []
    trades = []
    equity_records = []
    peak_equity = float(INITIAL_CAPITAL)
    halt_remaining = 0
    recent_losses = []  # (day_idx,) for weekly breaker
    recent_symbols = {}  # {symbol: last_exit_day_idx} for no-repeat
    n_days = len(trading_days)

    for day_idx in range(n_days):
        today = trading_days[day_idx]
        year = today.year
        kelly_r = kelly_r_table.get((threshold, year), 1.3)

        # --- Step 0: Compute positions ---
        pos_returns = {}
        sod_invested = 0.0
        for pos in open_positions:
            ps = prices.get(pos["symbol"])
            cr = (ps[today] / pos["entry_price"] - 1.0) if (ps is not None and today in ps.index) else 0.0
            pos_returns[id(pos)] = cr
            sod_invested += pos["allocation"] * (1 + cr)
        sod_equity = cash + sod_invested

        # --- Step 1: Check exits ---
        still_open = []
        for pos in open_positions:
            cr = pos_returns[id(pos)]
            td_held = day_idx - pos["entry_day_idx"]

            # Hard stop
            if pos["allocation"] * max(0, -cr) >= sod_equity * HARD_STOP_PCT:
                slipped = cr - SLIPPAGE
                cash += pos["allocation"] * (1 + slipped)
                trades.append(_trade(pos, today, slipped, td_held, "hard_stop", sid))
                recent_losses.append(day_idx)
                recent_symbols[pos["symbol"]] = day_idx
                continue

            action, ret, reason = check_exit(pos, cr, td_held, exit_rule)

            # ATR trailing stop improvement (checked after regular exit logic)
            if action != "exit" and improvements.get("atr_stop_mult", 0) > 0:
                mult = improvements["atr_stop_mult"]
                atr = pos.get("atr_14d_pct", 0.02)
                if atr <= 0:
                    atr = 0.02
                atr_stop = pos["hwm"] - mult * atr
                if cr <= atr_stop and td_held >= 2:
                    action, ret, reason = "exit", atr_stop, "atr_trail"

            if action == "exit":
                slipped = ret - SLIPPAGE
                cash += pos["allocation"] * (1 + slipped)
                trades.append(_trade(pos, today, slipped, td_held, reason, sid))
                if slipped < 0:
                    recent_losses.append(day_idx)
                recent_symbols[pos["symbol"]] = day_idx
            else:
                still_open.append(pos)

        open_positions = still_open

        # Recompute equity
        invested = sum(pos["allocation"] * (1 + pos_returns.get(id(pos), 0)) for pos in open_positions)
        total_equity = cash + invested

        if total_equity > peak_equity:
            peak_equity = total_equity
        dd = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0

        # Drawdown halt
        if dd >= DRAWDOWN_HALT_PCT and halt_remaining == 0:
            for pos in open_positions:
                cr = pos_returns.get(id(pos), 0)
                slipped = cr - SLIPPAGE
                cash += pos["allocation"] * (1 + slipped)
                trades.append(_trade(pos, today, slipped, day_idx - pos["entry_day_idx"],
                                     "drawdown_halt", sid))
            open_positions = []
            halt_remaining = DRAWDOWN_HALT_DAYS
            invested = 0
            total_equity = cash
            peak_equity = cash

        equity_records.append({"date": today, "equity": total_equity, "cash": cash,
                               "invested": invested, "open_positions": len(open_positions),
                               "drawdown": dd})

        if halt_remaining > 0:
            halt_remaining -= 1
            continue

        # --- Improvement #3: Skip Monday ---
        if improvements.get("skip_monday") and today.weekday() == 0:
            continue

        # --- Improvement #5: Weekly loss breaker ---
        if improvements.get("weekly_loss_breaker"):
            recent_losses = [d for d in recent_losses if day_idx - d <= 5]
            if len(recent_losses) >= 3:
                continue

        # --- Step 2: New candidates ---
        day_cands = candidates_by_date.get(today, [])
        if not day_cands:
            continue
        filtered = [c for c in day_cands if c["ml_score"] >= threshold]
        if not filtered:
            continue

        avail = max_pos - len(open_positions)
        if avail <= 0:
            continue

        # --- Improvement #7: Rank with symbol quality ---
        sq = symbol_quality if improvements.get("symbol_quality") else None
        ranked = rank_candidates(filtered, ranking, sq, year)

        # --- Improvement #2: No-repeat 14d ---
        if improvements.get("no_repeat_14d"):
            ranked = [c for c in ranked
                      if day_idx - recent_symbols.get(c["symbol"], -999) > 10]

        # Concentration filter (C1)
        if conc != "CN":
            max_per = 2 if conc == "C2" else 1
            sec_count = {}
            for p in open_positions:
                sec_count[p["sector"]] = sec_count.get(p["sector"], 0) + 1
            sel = []
            for c in ranked:
                if sec_count.get(c["sector"], 0) < max_per:
                    sel.append(c)
                    sec_count[c["sector"]] = sec_count.get(c["sector"], 0) + 1
                if len(sel) >= avail:
                    break
            ranked = sel
        else:
            ranked = ranked[:avail]

        # --- Enter positions ---
        for c in ranked:
            sym = c["symbol"]
            ps = prices.get(sym)
            if ps is None or today not in ps.index:
                continue
            entry_price = ps[today]
            if entry_price <= 0:
                continue

            # Sizing
            W = c["win_probability"]
            R = kelly_r
            kelly_pct = max(W - (1 - W) / R, 0.0)
            if sizing == "SK":
                frac = kelly_pct * 0.25
            elif sizing == "SH":
                frac = kelly_pct * 0.50
            elif sizing == "SA":
                ml = c["ml_score"]
                frac = kelly_pct * (0.50 if ml >= 90 else 0.35 if ml >= 80 else 0.25)
            else:
                frac = (1 - CASH_RESERVE) / max_pos

            alloc = total_equity * frac

            # --- Improvement #4: VIX-scaled sizing ---
            if improvements.get("vix_scaled_sizing") and vix_daily:
                vix = vix_daily.get(today, 15)
                if vix > 15:
                    vix_scale = min(1.0, (35 - vix) / 20)
                    alloc *= max(vix_scale, 0.1)

            # Caps
            max_alloc = total_equity * (1 - CASH_RESERVE) / max_pos * 2
            headroom = total_equity * (1 - CASH_RESERVE) - invested
            alloc = min(alloc, max_alloc, headroom)
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
            invested += alloc
            open_positions.append({
                "symbol": sym, "sector": c["sector"],
                "entry_date": today, "entry_day_idx": day_idx,
                "entry_price": entry_price,
                "holding_days": hd, "max_td": max_td,
                "predicted_return": c["predicted_return"],
                "predicted_mfe": c["predicted_mfe"],
                "win_probability": c["win_probability"],
                "ml_score": c["ml_score"],
                "allocation": alloc,
                "hwm": 0.0, "pred_reached": False, "trail_stop": 0.0,
                "atr_14d_pct": c.get("atr_14d_pct", 0.02) or 0.02,
            })

    # Close remaining
    last = trading_days[-1]
    for pos in open_positions:
        ps = prices.get(pos["symbol"])
        cr = (ps[last] / pos["entry_price"] - 1.0) if (ps is not None and last in ps.index) else 0.0
        slipped = cr - SLIPPAGE
        cash += pos["allocation"] * (1 + slipped)
        trades.append(_trade(pos, last, slipped, n_days - 1 - pos["entry_day_idx"],
                             "end_of_sim", sid))

    return trades, equity_records


def _trade(pos, exit_date, slipped, days_held, reason, sid):
    return {
        "strategy_id": sid, "symbol": pos["symbol"], "sector": pos["sector"],
        "entry_date": pos["entry_date"], "exit_date": exit_date,
        "days_held": days_held, "ml_score": pos["ml_score"],
        "predicted_return": pos["predicted_return"],
        "predicted_mfe": pos["predicted_mfe"],
        "win_probability": pos["win_probability"],
        "allocation": pos["allocation"],
        "pnl_pct": slipped, "pnl_dollars": pos["allocation"] * slipped,
        "exit_reason": reason, "hwm": pos["hwm"],
    }


# ============================================================
# Metrics
# ============================================================

def compute_metrics(trades, equity_records, config):
    sid = config["id"]
    if not trades:
        return {"strategy_id": sid, "label": config.get("label", ""), "sharpe_ratio": 0,
                "max_drawdown": 0, "win_rate": 0, "total_trades": 0,
                "annualized_return": 0, "years_profitable": 0, "total_return": 0,
                "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
                **{f"year_{y}": 0 for y in range(2018, 2026)}}

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity_records)
    n = len(tdf)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]

    final = edf["equity"].iloc[-1]
    tot_ret = final / INITIAL_CAPITAL - 1
    ann = (1 + tot_ret) ** (1/8) - 1 if tot_ret > -1 else -1

    peak = np.maximum.accumulate(edf["equity"].values)
    dd = ((peak - edf["equity"].values) / peak).max()

    edf["date"] = pd.to_datetime(edf["date"])
    monthly = edf.set_index("date").resample("ME")["equity"].last().pct_change().dropna()
    sharpe = (monthly.mean() / monthly.std()) * np.sqrt(12) if len(monthly) > 1 and monthly.std() > 0 else 0

    gp = wins["pnl_dollars"].sum() if len(wins) > 0 else 0
    gl = abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else 0

    yr_ret = {}
    for yr in range(2018, 2026):
        ye = edf[edf["date"].dt.year == yr]
        yr_ret[yr] = ye["equity"].iloc[-1] / ye["equity"].iloc[0] - 1 if len(ye) >= 2 else 0

    return {
        "strategy_id": sid, "label": config.get("label", ""),
        "total_return": tot_ret, "annualized_return": ann,
        "max_drawdown": dd, "sharpe_ratio": sharpe,
        "win_rate": len(wins) / n if n > 0 else 0,
        "avg_win": wins["pnl_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["pnl_pct"].mean() if len(losses) > 0 else 0,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "total_trades": n, "trades_per_year": n / 8,
        "years_profitable": sum(1 for r in yr_ret.values() if r > 0),
        **{f"year_{yr}": yr_ret.get(yr, 0) for yr in range(2018, 2026)},
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()

    t0 = time.time()

    # Load data
    df_10_30 = load_data("10_30")
    df_31_60 = load_data("31_60")
    earnings = load_earnings()
    df_10_30 = filter_earnings(df_10_30, earnings)
    df_31_60 = filter_earnings(df_31_60, earnings)

    symbols = list(set(df_10_30["symbol"].unique()) | set(df_31_60["symbol"].unique()))
    prices = load_prices(symbols)
    print(f"  Loaded prices for {len(prices)} symbols")

    trading_days = sorted(d for d in set().union(*(s.index for s in prices.values()))
                          if _date(2018, 1, 1) <= d <= _date(2025, 12, 31))
    print(f"  {len(trading_days)} trading days")

    vix_daily = load_vix()
    cands_10_30 = build_candidates(df_10_30)
    cands_31_60 = build_candidates(df_31_60)
    kelly_r = precompute_kelly_r(df_10_30)
    kelly_r_31 = precompute_kelly_r(df_31_60)
    sym_quality = compute_symbol_quality(df_10_30)

    print(f"\nData loaded in {time.time()-t0:.1f}s\n")
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Define test configurations
    # ============================================================

    # Baseline: Strategy #23 equivalent (EP/T90/SK/P3/C1)
    base = {"ranking": "WP", "threshold": 90, "exit": "EP", "sizing": "SK",
            "max_positions": 3, "concentration": "C1"}

    tests = []
    test_id = 0

    # 0: Baseline (no improvements)
    test_id += 1
    tests.append({"id": test_id, "label": "Baseline", **base,
                   "improvements": {}, "tier": "10_30"})

    # Individual improvements
    for name, flag in [
        ("NoRepeat14d", "no_repeat_14d"),
        ("SkipMonday", "skip_monday"),
        ("VIXScaled", "vix_scaled_sizing"),
        ("WeeklyBreaker", "weekly_loss_breaker"),
        ("SymbolQuality", "symbol_quality"),
    ]:
        test_id += 1
        tests.append({"id": test_id, "label": name, **base,
                       "improvements": {flag: True}, "tier": "10_30"})

    # Multi-tier (31-60 standalone with same params)
    test_id += 1
    tests.append({"id": test_id, "label": "Tier31_60", **base,
                   "improvements": {}, "tier": "31_60"})

    # All improvements combined on 10-30
    test_id += 1
    all_imp = {k: True for k in DEFAULT_IMPROVEMENTS}
    tests.append({"id": test_id, "label": "AllCombined_10_30", **base,
                   "improvements": all_imp, "tier": "10_30"})

    # All improvements on 31-60
    test_id += 1
    tests.append({"id": test_id, "label": "AllCombined_31_60", **base,
                   "improvements": all_imp, "tier": "31_60"})

    # All improvements with different configs
    for lbl, cfg in [
        ("AllComb_EM_T90", {**base, "exit": "EM"}),
        ("AllComb_CW_T85", {**base, "ranking": "CW", "threshold": 85}),
        ("AllComb_CR_T85", {**base, "ranking": "CR", "threshold": 85}),
        ("AllComb_SH_T85", {**base, "sizing": "SH", "threshold": 85}),
        ("AllComb_P4_C2", {**base, "max_positions": 4, "concentration": "C2"}),
    ]:
        test_id += 1
        tests.append({"id": test_id, "label": lbl, **cfg,
                       "improvements": all_imp, "tier": "10_30"})

    # ATR Stop -- standalone (each multiplier on base config, EP exit replaced by ATR trail)
    for mult, label in [(1.5, "ATR_1.5x"), (2.0, "ATR_2.0x"), (2.5, "ATR_2.5x"), (3.0, "ATR_3.0x")]:
        test_id += 1
        tests.append({"id": test_id, "label": label, **base,
                       "improvements": {"atr_stop_mult": mult}, "tier": "10_30"})

    # ATR Stop -- combined with all other improvements
    for mult, label in [(1.5, "AllComb_ATR_1.5x"), (2.0, "AllComb_ATR_2.0x"),
                        (2.5, "AllComb_ATR_2.5x"), (3.0, "AllComb_ATR_3.0x")]:
        test_id += 1
        tests.append({"id": test_id, "label": label, **base,
                       "improvements": {**all_imp, "atr_stop_mult": mult}, "tier": "10_30"})

    # ATR Stop -- on 31-60 tier with best standalone multipliers
    for mult, label in [(2.0, "ATR31_60_2.0x"), (2.5, "ATR31_60_2.5x")]:
        test_id += 1
        tests.append({"id": test_id, "label": label, **base,
                       "improvements": {"atr_stop_mult": mult}, "tier": "31_60"})

    print(f"Running {len(tests)} test configurations...")

    def _run(t):
        tier = t.pop("tier", "10_30")
        imp = t.pop("improvements", {})
        cands = cands_10_30 if tier == "10_30" else cands_31_60
        kr = kelly_r if tier == "10_30" else kelly_r_31
        tr, eq = run_strategy(t, cands, prices, trading_days, kr,
                              imp, vix_daily, sym_quality)
        m = compute_metrics(tr, eq, t)
        t["tier"] = tier
        t["improvements"] = imp
        m["tier"] = tier
        m["improvements"] = str(imp)
        return t["id"], tr, eq, m

    results = Parallel(n_jobs=args.jobs, verbose=5)(delayed(_run)(dict(t)) for t in tests)

    all_trades, all_eq, all_metrics = [], [], []
    for sid, tr, eq, m in results:
        all_trades.extend(tr)
        for r in eq:
            r["strategy_id"] = sid
        all_eq.extend(eq)
        all_metrics.append(m)

    pd.DataFrame(all_trades).to_csv(BACKTEST_DIR / "trades.csv", index=False)
    pd.DataFrame(all_eq).to_csv(BACKTEST_DIR / "equity.csv", index=False)
    sdf = pd.DataFrame(all_metrics).sort_values("sharpe_ratio", ascending=False)
    sdf.to_csv(BACKTEST_DIR / "summary.csv", index=False)

    print(f"\n{'='*120}")
    print("IMPROVEMENT IMPACT (each tested independently vs baseline)")
    print(f"{'='*120}")
    for _, r in sdf.iterrows():
        yr_cols = " ".join(f"{r.get(f'year_{y}', 0):+.0%}" for y in range(2018, 2026))
        print(f"  {r['label']:<25s} | Sharpe {r['sharpe_ratio']:5.2f} | "
              f"DD {r['max_drawdown']:5.1%} | WR {r['win_rate']:5.1%} | "
              f"Trades {int(r['total_trades']):>5d} | Ann {r['annualized_return']:7.1%} | "
              f"PF {r['profit_factor']:5.2f} | Yr+ {int(r['years_profitable'])} | "
              f"{yr_cols}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
