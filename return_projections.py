"""Compute realistic return projections from backtest data."""
import pandas as pd
import numpy as np

stock_trades = pd.read_csv("results/backtest/trades.csv")
spread_trades = pd.read_csv("results/backtest_spreads/trades.csv")

print("=" * 100)
print("REALISTIC RETURN PROJECTIONS (simple, non-compounding)")
print("=" * 100)

# ============================================================
# PER-TRADE ECONOMICS
# ============================================================

strategies = [
    ("Stock #23 (EP/T90/SK/C1)", stock_trades[stock_trades["strategy_id"] == 23], "pnl_pct"),
    ("Stock #6 (EM/T90/SK/C1)", stock_trades[stock_trades["strategy_id"] == 6], "pnl_pct"),
    ("Debit Spread #7 (3%/T90/XN)", spread_trades[spread_trades["strategy_id"] == 7], "option_return_pct"),
    ("Credit Spread #47 (2%OTM/3%W)", spread_trades[spread_trades["strategy_id"] == 47], "option_return_pct"),
]

print("\nPER-TRADE ECONOMICS:")
print("-" * 100)
print(f"{'Strategy':<35s} {'WR':>5s} {'AvgWin':>8s} {'AvgLoss':>8s} {'EV/Trade':>9s} {'Trades/Yr':>10s}")
print("-" * 100)
for name, t, col in strategies:
    wr = (t[col] > 0).mean()
    avg_w = t[t[col] > 0][col].mean()
    avg_l = t[t[col] <= 0][col].mean()
    ev = t[col].mean()
    tpy = len(t) / 8
    print(f"{name:<35s} {wr:>5.1%} {avg_w:>+7.1%} {avg_l:>+7.1%} {ev:>+8.2%} {tpy:>10.0f}")

# ============================================================
# DOLLAR PROJECTIONS
# ============================================================

print("\n" + "=" * 100)
print("DOLLAR PROJECTIONS BY ACCOUNT SIZE")
print("(Simple annual return = EV/trade x trades/yr x allocation%)")
print("=" * 100)

# Stock strategy projections
print("\nSTOCK STRATEGY #23 (3 positions, ~17% each via Quarter Kelly):")
t23 = stock_trades[stock_trades["strategy_id"] == 23]
ev = t23["pnl_pct"].mean()
tpy = len(t23) / 8
alloc = 0.17
for cap in [100_000, 250_000, 500_000, 1_000_000, 5_000_000]:
    annual = ev * tpy * alloc * cap
    pct = ev * tpy * alloc
    print(f"  ${cap:>10,} -> ${annual:>12,.0f}/yr  ({pct:.0%} simple return)")

print("\nSTOCK STRATEGY #6 (3 positions, ~17% each):")
t6 = stock_trades[stock_trades["strategy_id"] == 6]
ev = t6["pnl_pct"].mean()
tpy = len(t6) / 8
for cap in [100_000, 250_000, 500_000, 1_000_000, 5_000_000]:
    annual = ev * tpy * alloc * cap
    pct = ev * tpy * alloc
    print(f"  ${cap:>10,} -> ${annual:>12,.0f}/yr  ({pct:.0%} simple return)")

print("\nDEBIT SPREAD #7 (3 positions, 10% premium per trade):")
ds7 = spread_trades[spread_trades["strategy_id"] == 7]
ev = ds7["option_return_pct"].mean()
tpy = len(ds7) / 8
alloc = 0.10
for cap in [10_000, 25_000, 50_000, 100_000]:
    annual = ev * alloc * cap * tpy
    pct = ev * tpy * alloc
    print(f"  ${cap:>10,} -> ${annual:>12,.0f}/yr  ({pct:.0%} return on capital)")

print("\nCREDIT SPREAD #47 (3 positions, 10% collateral per trade):")
cs47 = spread_trades[spread_trades["strategy_id"] == 47]
ev = cs47["option_return_pct"].mean()
tpy = len(cs47) / 8
for cap in [10_000, 25_000, 50_000, 100_000]:
    annual = ev * alloc * cap * tpy
    pct = ev * tpy * alloc
    print(f"  ${cap:>10,} -> ${annual:>12,.0f}/yr  ({pct:.0%} return on capital)")

# ============================================================
# YEAR-BY-YEAR (non-compounding, fixed capital)
# ============================================================

print("\n" + "=" * 100)
print("YEAR-BY-YEAR SIMPLE RETURNS")
print("=" * 100)

configs = [
    ("Stock #23", stock_trades[stock_trades["strategy_id"] == 23], "pnl_pct", 100_000, 0.17),
    ("Stock #6", stock_trades[stock_trades["strategy_id"] == 6], "pnl_pct", 100_000, 0.17),
    ("DS #7", spread_trades[spread_trades["strategy_id"] == 7], "option_return_pct", 10_000, 0.10),
    ("CS #47", spread_trades[spread_trades["strategy_id"] == 47], "option_return_pct", 10_000, 0.10),
]

for name, tdf, col, cap, alloc in configs:
    tdf = tdf.copy()
    tdf["year"] = pd.to_datetime(tdf["entry_date"]).dt.year
    print(f"\n{name} (${cap:,} account):")
    total = 0
    for yr in range(2018, 2026):
        yt = tdf[tdf["year"] == yr]
        if len(yt) == 0:
            print(f"  {yr}:   0 trades")
            continue
        ev = yt[col].mean()
        n = len(yt)
        dollar = ev * n * alloc * cap
        total += dollar
        print(f"  {yr}: {n:3d} trades, EV {ev:+.1%}/trade -> ${dollar:>+10,.0f}  ({ev*n*alloc:+.0%})")
    avg = total / 8
    print(f"  AVERAGE: ${avg:>+10,.0f}/yr  ({avg/cap:+.0%} annual)")

# ============================================================
# WORST CASE SCENARIOS
# ============================================================

print("\n" + "=" * 100)
print("WORST CASE & RISK METRICS")
print("=" * 100)

for name, tdf, col in [
    ("Stock #23", stock_trades[stock_trades["strategy_id"] == 23], "pnl_pct"),
    ("Stock #6", stock_trades[stock_trades["strategy_id"] == 6], "pnl_pct"),
    ("DS #7", spread_trades[spread_trades["strategy_id"] == 7], "option_return_pct"),
    ("CS #47", spread_trades[spread_trades["strategy_id"] == 47], "option_return_pct"),
]:
    print(f"\n{name}:")
    print(f"  Worst single trade:     {tdf[col].min():+.1%}")
    print(f"  P5 (5th percentile):    {tdf[col].quantile(0.05):+.1%}")
    print(f"  P10 (10th percentile):  {tdf[col].quantile(0.10):+.1%}")
    # Worst month (group by month)
    tdf2 = tdf.copy()
    tdf2["month"] = pd.to_datetime(tdf2["entry_date"]).dt.to_period("M")
    monthly = tdf2.groupby("month")[col].sum()
    print(f"  Worst month (sum):      {monthly.min():+.1%} ({monthly.idxmin()})")
    # Max consecutive losses
    streak = max_streak = 0
    for r in tdf[col]:
        if r <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"  Max losing streak:      {max_streak} trades")
    # Probability of 3+ consecutive losses
    n_streaks_3 = 0
    streak = 0
    for r in tdf[col]:
        if r <= 0:
            streak += 1
            if streak == 3:
                n_streaks_3 += 1
        else:
            streak = 0
    print(f"  Streaks of 3+ losses:   {n_streaks_3} times in 8 years")

# ============================================================
# COMBINED PORTFOLIO EXAMPLE
# ============================================================

print("\n" + "=" * 100)
print("COMBINED PORTFOLIO EXAMPLE")
print("=" * 100)

# Stock: $500K, Strategy #23
s23 = stock_trades[stock_trades["strategy_id"] == 23]
stock_ev = s23["pnl_pct"].mean()
stock_tpy = len(s23) / 8

# Options: $25K, Debit Spread #7
ds7 = spread_trades[spread_trades["strategy_id"] == 7]
opt_ev = ds7["option_return_pct"].mean()
opt_tpy = len(ds7) / 8

stock_annual = stock_ev * stock_tpy * 0.17 * 500_000
opt_annual = opt_ev * opt_tpy * 0.10 * 25_000

print(f"\nBacktest estimates:")
print(f"  Stock:   $500K x Strategy #23 -> ${stock_annual:>+12,.0f}/yr  ({stock_annual/500_000:+.0%})")
print(f"  Options: $25K  x Debit Spreads -> ${opt_annual:>+12,.0f}/yr  ({opt_annual/25_000:+.0%})")
print(f"  Total:   $525K combined       -> ${stock_annual+opt_annual:>+12,.0f}/yr  ({(stock_annual+opt_annual)/525_000:+.0%})")

print(f"\nConservative (30% haircut for execution, timing, model decay):")
d = 0.70
print(f"  Stock:   ${stock_annual*d:>+12,.0f}/yr  ({stock_annual*d/500_000:+.0%})")
print(f"  Options: ${opt_annual*d:>+12,.0f}/yr  ({opt_annual*d/25_000:+.0%})")
print(f"  Total:   ${(stock_annual+opt_annual)*d:>+12,.0f}/yr  ({(stock_annual+opt_annual)*d/525_000:+.0%})")

print(f"\nPessimistic (50% haircut):")
d = 0.50
print(f"  Stock:   ${stock_annual*d:>+12,.0f}/yr  ({stock_annual*d/500_000:+.0%})")
print(f"  Options: ${opt_annual*d:>+12,.0f}/yr  ({opt_annual*d/25_000:+.0%})")
print(f"  Total:   ${(stock_annual+opt_annual)*d:>+12,.0f}/yr  ({(stock_annual+opt_annual)*d/525_000:+.0%})")
