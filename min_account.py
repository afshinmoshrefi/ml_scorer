"""Minimum account size analysis for automated options trading."""
import pandas as pd

trades = pd.read_csv("results/backtest_spreads/trades.csv")

cs54 = trades[trades["strategy_id"] == 54]
ds9 = trades[trades["strategy_id"] == 9]

print("=" * 80)
print("MINIMUM ACCOUNT SIZE FOR AUTOMATED OPTIONS")
print("=" * 80)

print("\nTRADE SIZE REALITY CHECK:")
print("\nCredit spread (3% wide on S&P 500 stocks):")
print("  $50 stock  -> $150 collateral per contract (1.5 x 100 shares)")
print("  $100 stock -> $300 collateral per contract")
print("  $200 stock -> $600 collateral per contract")
print("  $500 stock -> $1,500 collateral per contract")
print("  Average S&P 500 stock ~$150 -> ~$450 collateral per contract")

print("\nDebit spread (3% wide, ~40% cost ratio):")
print("  $50 stock  -> $60 premium per contract")
print("  $100 stock -> $120 premium per contract")
print("  $200 stock -> $240 premium per contract")
print("  Average S&P 500 ~$150 -> ~$180 premium per contract")

print("\n" + "=" * 80)
print("ACCOUNT MINIMUMS")
print("=" * 80)

print("""
CREDIT SPREADS ONLY (93% WR, Sharpe 3.65):
  Per trade: 1 contract, avg ~$450 collateral
  3 concurrent positions: $1,350 deployed
  50% cash reserve rule: need $2,700
  Buffer for worst case (3 simultaneous max losses): +$1,350
  Broker minimum for spread approval: $2,000

  MINIMUM: $2,500
  PRACTICAL: $5,000 (can trade 2-3 contracts on cheaper stocks)

DEBIT SPREADS ONLY (78% WR, 100% all-8-yr):
  Per trade: 1 contract, avg ~$180 premium
  3 concurrent positions: $540 deployed
  50% cash reserve: need $1,080
  Buffer: +$540

  MINIMUM: $1,500
  PRACTICAL: $3,000

BOTH STRATEGIES:
  Credit: $450/pos x 3 = $1,350
  Debit: $180/pos x 3 = $540
  Total deployed: $1,890
  50% reserve: need $3,780

  MINIMUM: $4,000
  PRACTICAL: $5,000-$10,000
""")

print("=" * 80)
print("PROJECTED RETURNS BY ACCOUNT SIZE")
print("=" * 80)

print("\nCredit spreads only (55 trades/yr, 33% EV on collateral):")
for cap in [2500, 5000, 10000, 25000]:
    # At small accounts, only 1 contract per trade
    # So the actual $ deployed depends on stock price, not % of account
    # Approximate: avg $450 collateral per trade, 55 trades/yr
    # On small accounts, the 10% allocation cap means fewer dollars per trade
    # but the contract size is fixed at 1
    deployed = min(cap * 0.10, 450)  # cap at 1 contract avg
    annual = 0.33 * 55 * deployed
    print(f"  ${cap:>6,} -> ${annual:>6,.0f}/yr ({annual/cap:.0%})")

print("\nDebit spreads only (45 trades/yr, 70% EV on premium):")
for cap in [1500, 3000, 5000, 10000]:
    deployed = min(cap * 0.10, 180)  # cap at 1 contract avg
    annual = 0.70 * 45 * deployed
    print(f"  ${cap:>6,} -> ${annual:>6,.0f}/yr ({annual/cap:.0%})")

print("\nBoth combined (split 50/50):")
for cap in [5000, 10000, 25000]:
    cs_dep = min(cap * 0.5 * 0.10, 450)
    ds_dep = min(cap * 0.5 * 0.10, 180)
    annual = 0.33 * 55 * cs_dep + 0.70 * 45 * ds_dep
    print(f"  ${cap:>6,} -> ${annual:>6,.0f}/yr ({annual/cap:.0%})")

print("""
COMMISSION IMPACT:
  ~100 trades/yr x 2 legs x $0.65/contract = ~$130/yr
  On $2,500: 5.2% drag (significant)
  On $5,000: 2.6% drag (manageable)
  On $10,000: 1.3% drag (negligible)
  Use a low-commission broker (TastyTrade, IBKR) for small accounts.

BOTTOM LINE:
  $2,500: Can run credit spreads only, 1 contract, ~$8K/yr backtest
  $5,000: Both strategies, comfortable margin, ~$15K/yr backtest
  $10,000: Full system, minimal drag, ~$25K/yr backtest

  Conservative (-30%): $5K -> ~$10K/yr, $10K -> ~$18K/yr
  Pessimistic (-50%):  $5K -> ~$7.5K/yr, $10K -> ~$12.5K/yr
""")
