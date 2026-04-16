# Codex V4 Enhanced Backtest Brief

## Mission

Run `backtest_v4_enhanced.py` and produce a comprehensive analysis of the results.
Goal: identify the highest-return, highest-stability, lowest-drawdown strategy configuration achievable from the TradeWave ML seasonal pattern system.

---

## Context

The V3 Codex rerun found:
- Best combined L+S stock: Sharpe 7.11, DD -2.65%, CAGR 35.67% (STK_045)
- Best long-only: Sharpe 6.86 (STK_063)
- Best short-only: Sharpe 5.51 (STK_009)
- All top configs use: `target6_atr2` exit + `risk_balanced` concentration + `vol_inverse` sizing

The V4 script adds 5 new enhancement dimensions on top of those best configs:
1. **VIX hard block** -- no entries when VIX >= 35 (matches production service behavior)
2. **Best-4 filters** -- SymbolQuality, NoRepeat14d, SkipMonday, WeeklyBreaker
3. **100-Year Pattern regime** -- lower ML threshold to 70, long-only, exclude Energy during Sep 27-Jul 18 in midterm election years (2018+2022 in backtest data)
4. **Multi-tier** -- combine 10-30 day and 31-60 day opportunity pools
5. **VIX-scaled sizing** -- shrink position sizes proportionally when VIX is elevated

---

## How to Run

```bash
cd C:\seasonals\ml_scorer
python backtest_v4_enhanced.py --jobs 8
```

Runtime estimate: 15-30 minutes depending on CPU.

Outputs in `results/backtest_v4/`:
- `summary.csv` -- all configs ranked by Sharpe
- `yearly.csv` -- year-by-year returns per config
- `holdout.csv` -- 2018-2024 train vs 2025 out-of-sample for top 10
- `report.md` -- auto-generated markdown report

---

## What to Analyze

### 1. Enhancement attribution (stocks, combined L+S)

For each of the 12 enhancement combos tested on BaseA and BaseB:
- Does each individual enhancement improve or hurt Sharpe vs baseline?
- Does each individual enhancement improve or hurt max drawdown?
- Which enhancements stack cleanly (additive) vs conflict?
- What is the lift from the full enhancement stack (config 12_full) vs baseline?

Expected pattern based on prior research:
- VIX block: small Sharpe lift, meaningful DD reduction
- SymbolQuality: +15-20% Sharpe lift
- NoRepeat14d: best DD reduction of the four
- SkipMonday: small WR improvement
- WeeklyBreaker: reduces trade count but smooths equity curve
- PatternRegime: should lift CAGR during 2018-2019 and 2022-2023 periods
- MultiTier: temporal diversification, smooths Sharpe

### 2. Direction analysis

- How much does combined L+S beat long-only Sharpe?
- Does adding shorts increase or decrease max drawdown?
- Is there a configuration where short-only exceeds long-only Sharpe?

### 3. Robustness: 2025 holdout

For the top 10 combined L+S stock configs:
- Does 2025 performance exceed or trail 2018-2024 in-sample?
- Are there configs that degrade significantly in 2025?
- The V3 top configs showed BETTER 2025 performance than in-sample -- does that hold?

### 4. Year-by-year stability

For the top 5 combined configs:
- Are all years profitable?
- What is the worst single year?
- Is 2019 (partial midterm window year) notably better than surrounding years?
- Is 2022-2023 (the stronger 100-Year window) noticeably better?
- Is 2025 the weakest or strongest year?

### 5. Options and spreads

Run with:
- Baseline (replicates V3 result as sanity check)
- VIX + Best-4 enhancement
- VIX + Best-4 + PatternRegime

Report whether enhancements improve options and spread Sharpe similarly to stocks,
or whether the benefit is smaller (expected: smaller, since options/spreads are already
capped-payoff instruments that benefit less from SymbolQuality).

### 6. Final recommended configuration

Based on the results, recommend:
- **Primary combined stock config**: the single config ID that best balances Sharpe, DD, and stability
- **Long-only sleeve**: for capital that cannot go short
- **Short-only sleeve**: for hedging or standalone
- **Options config**: best options enhancement combo
- **Spread config**: best spread enhancement combo
- **What NOT to enable**: any enhancement that consistently hurts performance

---

## Scoring Criteria

Rank configurations by a composite score:
```
score = (sharpe * 0.50) + (DD_score * 0.30) + (consistency_score * 0.20)
```
where:
- `DD_score = 1 - abs(max_drawdown) / 20` (normalized, 0% DD = 1.0, 20% DD = 0.0)
- `consistency_score = (years_profitable / total_years) * 0.7 + (1 - worst_year_return/-30) * 0.3` (clipped)

A configuration that achieves all-years-profitable with DD < 3% and Sharpe > 7 is the target.

---

## Error Handling

If any config fails (error in output):
- Note which config failed and why
- Skip it and continue analysis
- The `error` key will be present in the summary row if a config crashed

---

## Output Format

Produce:
1. Updated `results/backtest_v4/report.md` with your analysis appended (add a "## Codex Analysis" section at the bottom)
2. Updated `docs/TradeWave_V4_Strategy_Assessment.docx` mirroring the markdown

The auto-generated `report.md` contains the raw tables. Your job is to interpret them:
- Write the "Codex Analysis" section with findings, enhancement rankings, and the final recommended strategy set
- Include exact config IDs for each recommendation
- Include a "What Changed vs V3" summary comparing best V4 to best V3 Sharpe/DD

---

## Important Notes

1. **The VIX hard block was NOT applied in the V3 backtest.** It only exists in production. This means V3 results during VIX>35 periods (mainly Q4 2018, Mar 2020) include entries that would be blocked in production. V4 with VIX block is therefore more realistic.

2. **The 100-Year Pattern** is the user's own discovery (named in their book). It refers to the historical finding that SPX has never been down from Sep 27 to ~Jul 18+1yr in midterm election years since 1930. The backtest data contains 2 such windows (2018 and 2022 midterms). The regime switch in V4 exploits this by relaxing entry criteria during these windows.

3. **Multi-tier** combines the 10-30 day parquet with the 31-60 day parquet. On any given day, candidates from both tiers compete in the same daily selection pool. This adds temporal diversification but may also dilute the strongest 10-30 day signals if 31-60 day signals are weaker. Report whether the combined tier beats 10-30 alone.

4. **Symbol quality** is computed from prior-year returns within the backtest data itself -- this is valid as long as we only use years strictly prior to the current trade date (no look-ahead). The computation is validated in the code.

5. **WeeklyBreaker** pauses entries after 3+ losses in last 5 completed trades. It reduces trade frequency but smooths drawdowns. If it conflicts with other signals, it will reduce CAGR more than it helps Sharpe.

6. **Target Sharpe > 8, DD < 2%, CAGR > 35%, all years profitable** -- if any config achieves all four simultaneously, flag it prominently.
