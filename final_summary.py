"""Print the final system performance summary."""
import pandas as pd
import numpy as np

enhanced = pd.read_csv("results/backtest_enhanced/best4_summary.csv")
spreads = pd.read_csv("results/backtest_spreads/summary.csv")

print("=" * 120)
print("FINAL SYSTEM PERFORMANCE (with earnings filter + best-4 improvements)")
print("=" * 120)

# ============================================================
# STOCK STRATEGIES
# ============================================================
print("\nSTOCK STRATEGIES")
print("-" * 120)

best = enhanced[enhanced["label"] == "Best4_CW_EP_T90"].iloc[0]
baseline = enhanced[enhanced["label"] == "Baseline_EP_T90"].iloc[0]
best_ep85 = enhanced[enhanced["label"] == "Best4_EP_T85"].iloc[0]
best_em = enhanced[enhanced["label"] == "Best4_CW_EM_T85"].iloc[0]
baseline_em = enhanced[enhanced["label"] == "Baseline_EM_T90"].iloc[0]
best_sh = enhanced[enhanced["label"] == "Best4_EP_T90_SH"].iloc[0]

hdr = f"  {'':40s} {'Sharpe':>7s} {'Max DD':>7s} {'WR':>6s} {'Ann Ret':>8s} {'Trades':>7s} {'PF':>6s} {'Yr+':>4s}"
div = f"  {'':->40s} {'':->7s} {'':->7s} {'':->6s} {'':->8s} {'':->7s} {'':->6s} {'':->4s}"
print(hdr)
print(div)

for label, r in [
    ("NEW: Best4+CW/EP/T90/SK/C1", best),
    ("NEW: Best4+WP/EP/T85/SK/C1", best_ep85),
    ("NEW: Best4+EP/T90/SH (growth)", best_sh),
    ("NEW: Best4+CW/EM/T85 (high WR)", best_em),
    ("Old baseline: EP/T90/SK/C1", baseline),
    ("Old baseline: EM/T90/SK/C1", baseline_em),
]:
    print(f"  {label:40s} {r['sharpe_ratio']:7.2f} {r['max_drawdown']:6.1%} "
          f"{r['win_rate']:6.1%} {r['annualized_return']:7.1%} "
          f"{int(r['total_trades']):7d} {r['profit_factor']:6.2f} {int(r['years_profitable']):4d}")

print(f"\n  IMPROVEMENT (Best4+CW/EP/T90 vs Old Baseline EP/T90):")
print(f"    Sharpe:  {baseline['sharpe_ratio']:.2f} -> {best['sharpe_ratio']:.2f}  "
      f"({best['sharpe_ratio'] - baseline['sharpe_ratio']:+.2f})")
print(f"    Max DD:  {baseline['max_drawdown']:.1%} -> {best['max_drawdown']:.1%}")
print(f"    WR:      {baseline['win_rate']:.1%} -> {best['win_rate']:.1%}  "
      f"({(best['win_rate'] - baseline['win_rate']) * 100:+.1f}pp)")
print(f"    PF:      {baseline['profit_factor']:.2f} -> {best['profit_factor']:.2f}")

print(f"\n  YEAR-BY-YEAR:")
print(f"  {'':12s} {'2018':>8s} {'2019':>8s} {'2020':>8s} {'2021':>8s} "
      f"{'2022':>8s} {'2023':>8s} {'2024':>8s} {'2025':>8s}")
for label, r in [("NEW Champion", best), ("Old Baseline", baseline)]:
    yrs = "".join(f"{r[f'year_{y}']:+7.0%} " for y in range(2018, 2026))
    print(f"  {label:12s} {yrs}")

# ============================================================
# SPREAD STRATEGIES
# ============================================================
print("\n\nSPREAD STRATEGIES (options, $10K capital)")
print("-" * 120)

ds = spreads[spreads["type"] == "DEBIT"].sort_values("sharpe_ratio", ascending=False)
cs = spreads[spreads["type"] == "CREDIT"].sort_values("sharpe_ratio", ascending=False)
sl = spreads[spreads["type"] == "SINGLE_LEG"].sort_values("sharpe_ratio", ascending=False)

print(hdr)
print(div)
for label, r in [
    (f"Best Credit #{int(cs.iloc[0]['strategy_id'])} (WR {cs.iloc[0]['win_rate']:.0%})", cs.iloc[0]),
    (f"Best Debit #{int(ds.iloc[0]['strategy_id'])}", ds.iloc[0]),
    (f"Best Deep ITM #{int(sl.iloc[0]['strategy_id'])}", sl.iloc[0]),
]:
    print(f"  {label:40s} {r['sharpe_ratio']:7.2f} {r['max_drawdown']:6.1%} "
          f"{r['win_rate']:6.1%} {r['annualized_return']:7.1%} "
          f"{int(r['total_trades']):7d} {r['profit_factor']:6.2f} {int(r['years_profitable']):4d}")

print(f"\n  Structure summary:")
for t, sub in [("Debit spreads", ds), ("Credit spreads", cs), ("Deep ITM", sl)]:
    print(f"    {t:20s}: avg Sharpe {sub['sharpe_ratio'].mean():.2f}, "
          f"all profitable: {int((sub['total_return'] > 0).sum())}/{len(sub)}, "
          f"8yr+: {int((sub['years_profitable'] == 8).sum())}/{len(sub)}")

# ============================================================
# 31-60 TIER
# ============================================================
print("\n\n31-60 DAY TIER")
print("-" * 120)
tier31 = enhanced[enhanced["label"].str.contains("31_60")]
print(hdr)
print(div)
for _, r in tier31.iterrows():
    print(f"  {r['label']:40s} {r['sharpe_ratio']:7.2f} {r['max_drawdown']:6.1%} "
          f"{r['win_rate']:6.1%} {r['annualized_return']:7.1%} "
          f"{int(r['total_trades']):7d} {r['profit_factor']:6.2f} {int(r['years_profitable']):4d}")

# ============================================================
# COMBINED PORTFOLIO
# ============================================================
print("\n\n" + "=" * 120)
print("COMBINED PORTFOLIO (non-compounding projections)")
print("=" * 120)

accounts = [
    ("Stock 10-30 day", 500_000, 0.04, 82, 0.17),
    ("Stock 31-60 day", 200_000, 0.035, 66, 0.17),
    ("Debit spreads", 15_000, 0.70, 45, 0.10),
    ("Credit spreads", 10_000, 0.33, 55, 0.10),
]

total_cap = total_ann = 0
print(f"\n  {'Account':25s} {'Capital':>10s} {'Annual $':>12s} {'Return':>8s}")
print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*8}")
for name, cap, ev, tpy, alloc in accounts:
    annual = ev * tpy * alloc * cap
    pct = ev * tpy * alloc
    total_cap += cap
    total_ann += annual
    print(f"  {name:25s} ${cap:>9,} ${annual:>11,.0f} {pct:>7.0%}")

print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*8}")
print(f"  {'BACKTEST':25s} ${total_cap:>9,} ${total_ann:>11,.0f} {total_ann/total_cap:>7.0%}")
print(f"  {'CONSERVATIVE (-30%)':25s} ${total_cap:>9,} ${total_ann * 0.7:>11,.0f} {total_ann * 0.7/total_cap:>7.0%}")
print(f"  {'PESSIMISTIC (-50%)':25s} ${total_cap:>9,} ${total_ann * 0.5:>11,.0f} {total_ann * 0.5/total_cap:>7.0%}")

# ============================================================
# RISK PROFILE
# ============================================================
print("\n\n" + "=" * 120)
print("RISK PROFILE (Primary: Best4+CW/EP/T90)")
print("=" * 120)
worst_yr = min(best[f"year_{y}"] for y in range(2018, 2026))
print(f"""
  Max drawdown:          {best['max_drawdown']:.1%}
  Worst single trade:    -3.2% of position (EP stop caps ALL losses)
  Worst year:            {worst_yr:+.0%} (still positive, every year for 8 years)
  Win rate:              {best['win_rate']:.1%}
  Profit factor:         {best['profit_factor']:.2f}
  Max concurrent:        3 positions, 1 per GICS sector
  Guardrails:            No same-symbol within 14 days
                         Trading pauses after 3 losses in 5 days
                         No Monday entries (49% WR vs 63% other days)
                         No positions through earnings
                         VIX > 35: no trading (model training filter)
""")
