"""Test the best-4 improvements combo across multiple strategy configurations."""

import sys
import time
from datetime import date as _date
from pathlib import Path

# Patch __file__ for imports
sys.path.insert(0, str(Path(__file__).parent))
from backtest_enhanced import (
    load_data, load_earnings, filter_earnings, load_prices, load_vix,
    build_candidates, precompute_kelly_r, compute_symbol_quality,
    run_strategy, compute_metrics, BACKTEST_DIR
)

import pandas as pd
from joblib import Parallel, delayed

t0 = time.time()

# Load all data
df_10 = load_data("10_30")
df_31 = load_data("31_60")
earnings = load_earnings()
df_10 = filter_earnings(df_10, earnings)
df_31 = filter_earnings(df_31, earnings)
symbols = list(set(df_10["symbol"].unique()) | set(df_31["symbol"].unique()))
prices = load_prices(symbols)
print(f"  {len(prices)} symbols")
trading_days = sorted(d for d in set().union(*(s.index for s in prices.values()))
                      if _date(2018, 1, 1) <= d <= _date(2025, 12, 31))
vix_daily = load_vix()
cands_10 = build_candidates(df_10)
cands_31 = build_candidates(df_31)
kr_10 = precompute_kelly_r(df_10)
kr_31 = precompute_kelly_r(df_31)
sq_10 = compute_symbol_quality(df_10)
sq_31 = compute_symbol_quality(df_31)
print(f"Data loaded in {time.time()-t0:.1f}s\n")

BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# Best 4 improvements (no VIX scaling)
best4 = {"symbol_quality": True, "no_repeat_14d": True,
         "weekly_loss_breaker": True, "skip_monday": True}
best4_vix = {**best4, "vix_scaled_sizing": True}

base_ep = {"ranking": "WP", "threshold": 90, "exit": "EP", "sizing": "SK",
           "max_positions": 3, "concentration": "C1"}
base_em = {"ranking": "WP", "threshold": 90, "exit": "EM", "sizing": "SK",
           "max_positions": 3, "concentration": "C1"}

tests = []
tid = 0

def add(label, tier="10_30", imp=None, **overrides):
    global tid
    tid += 1
    cfg = {**base_ep, **overrides}
    tests.append({"id": tid, "label": label, **cfg,
                   "improvements": imp or {}, "tier": tier})

# Baselines
add("Baseline_EP_T90")
add("Baseline_EM_T90", exit="EM")

# Best 4 on primary configs
add("Best4_EP_T90", imp=best4)
add("Best4_EM_T90", imp=best4, exit="EM")
add("Best4_EP_T85", imp=best4, threshold=85)
add("Best4_EM_T85", imp=best4, exit="EM", threshold=85)

# Best 4 + ranking variations
add("Best4_CW_EP_T85", imp=best4, ranking="CW", threshold=85)
add("Best4_CW_EP_T90", imp=best4, ranking="CW")
add("Best4_CR_EP_T85", imp=best4, ranking="CR", threshold=85)
add("Best4_CW_EM_T85", imp=best4, ranking="CW", exit="EM", threshold=85)

# Best 4 + position count
add("Best4_EP_T90_P2", imp=best4, max_positions=2)
add("Best4_EP_T85_P4_C2", imp=best4, threshold=85, max_positions=4, concentration="C2")

# Best 4 + sizing
add("Best4_EP_T90_SH", imp=best4, sizing="SH")
add("Best4_EP_T85_SA", imp=best4, sizing="SA", threshold=85)

# Best 4 on 31-60 tier
add("Best4_EP_T90_31_60", tier="31_60", imp=best4)
add("Best4_EM_T90_31_60", tier="31_60", imp=best4, exit="EM")
add("Best4_EP_T85_31_60", tier="31_60", imp=best4, threshold=85)
add("Best4_CW_EP_T85_31_60", tier="31_60", imp=best4, ranking="CW", threshold=85)

# Best4 + VIX for comparison
add("Best4+VIX_EP_T90", imp=best4_vix)

# ET exit with best4
add("Best4_ET_T90", imp=best4, exit="ET")

print(f"Running {len(tests)} configs...")


def _run(t):
    t = dict(t)
    tier = t.pop("tier", "10_30")
    imp = t.pop("improvements", {})
    cands = cands_10 if tier == "10_30" else cands_31
    kr = kr_10 if tier == "10_30" else kr_31
    sqq = sq_10 if tier == "10_30" else sq_31
    tr, eq = run_strategy(t, cands, prices, trading_days, kr, imp, vix_daily, sqq)
    m = compute_metrics(tr, eq, t)
    m["tier"] = tier
    return t["id"], tr, eq, m


results = Parallel(n_jobs=12, verbose=5)(delayed(_run)(t) for t in tests)

mdf = pd.DataFrame([m for _, _, _, m in results]).sort_values("sharpe_ratio", ascending=False)
mdf.to_csv(BACKTEST_DIR / "best4_summary.csv", index=False)

# Save all trades
all_trades = []
for _, tr, _, _ in results:
    all_trades.extend(tr)
pd.DataFrame(all_trades).to_csv(BACKTEST_DIR / "best4_trades.csv", index=False)

print(f"\n{'='*140}")
print("BEST-4 IMPROVEMENTS: FULL RESULTS")
print(f"{'='*140}")
for _, r in mdf.iterrows():
    yr = " ".join(f"{r.get(f'year_{y}', 0):+.0%}" for y in range(2018, 2026))
    print(f"  {r['label']:<28s} | Sharpe {r['sharpe_ratio']:5.2f} | "
          f"DD {r['max_drawdown']:5.1%} | WR {r['win_rate']:5.1%} | "
          f"Tr {int(r['total_trades']):>4d} | Ann {r['annualized_return']:7.1%} | "
          f"PF {r['profit_factor']:5.2f} | Yr+ {int(r['years_profitable'])} | {yr}")

print(f"\nRuntime: {time.time()-t0:.1f}s")
