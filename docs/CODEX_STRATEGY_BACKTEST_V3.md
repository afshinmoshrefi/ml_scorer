# Independent Strategy Backtest Task -- ML Pattern Scorer V3

## Your Mission

You are an independent quantitative analyst with no prior knowledge of this system. Your job is to design, build, and run a comprehensive strategy backtest using the ML scorer output data provided. You have full freedom to decide what strategy variations to test. Do not look for existing backtest code -- write your own from scratch.

At the end, produce two output files:
1. `results/project_v3_codex_strategy_final.md` -- machine-readable results and analysis
2. `docs/TradeWave_V3_Codex_Strategy_Assessment.docx` -- human-readable report (see format spec at the bottom)

---

## The Data

### Primary Dataset

**File:** `results/backtester_input_10_30.parquet`
**Rows:** ~11 million (~8.0M long, ~3.0M short)
**Period:** 2018-2025 (8 years of walk-forward out-of-sample predictions)
**Universe:** 475 S&P 500 stocks, both long and short directions, 10-30 day holding periods

This is walk-forward validation data -- each row is a real out-of-sample prediction made by the ML ensemble at the time of entry. There is no look-ahead bias. The ML models were trained on pre-2018 data before predicting 2018, trained on pre-2019 data before predicting 2019, and so on.

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Pattern entry date |
| `year` | int | Calendar year (2018-2025) |
| `symbol` | str | Stock ticker |
| `sector` | str | GICS sector (11 sectors) |
| `direction` | str | 'l' = long, 's' = short (~73% long, ~27% short) |
| `holding_days` | int | Intended holding period in days (10-30) |
| `ml_score` | float | ML model confidence score, 0-100 (higher = model more confident) |
| `predicted_return` | float | Model's predicted close-to-close return (%) |
| `predicted_mfe` | float | Model's predicted max favorable excursion (%) -- best exit achievable in window |
| `win_probability` | float | Calibrated probability of positive return (0-1) |
| `p_hit_return` | float | Probability actual return >= predicted_return (0-1) |
| `p_hit_mfe` | float | Probability actual MFE >= predicted_mfe (0-1) |
| `actual_return` | float | What actually happened: close-to-close return (%) |
| `actual_mfe` | float | Actual max favorable excursion achieved (%) |
| `stock_volatility_20d` | float | 20-day realized volatility of stock at entry (annualized %) |
| `atr_14d_pct` | float | 14-day ATR as % of stock price at entry (mean ~2.5%) |

### Understanding the ML Outputs

The ML scorer produces four prediction fields per opportunity:

- **ml_score (0-100)**: Percentile rank of predicted_return within the calibration distribution. Score 90 means the model predicts this is in the top 10% of all opportunities it has ever seen.
- **win_probability**: Empirically calibrated probability of a positive return, derived from walk-forward predictions. Not a model output directly -- it comes from a lookup table built from out-of-sample predictions.
- **predicted_return**: Raw model prediction of actual close-to-close return. The model is a regression, not classification.
- **predicted_mfe**: Model's estimate of the best possible exit price during the holding window. Always positive by design. Useful for estimating profit targets and options strike selection.

---

## The Trading Context

These are **seasonal stock patterns** -- tendencies for specific stocks to move in a direction during specific calendar periods, based on 20+ years of historical data. Each row represents one pattern activation: a stock that has historically shown a tendency to go in a specific direction during this time of year.

**Key characteristics:**
- Holding periods: 10-30 calendar days
- Both longs and shorts: `direction='l'` means enter long (buy), `direction='s'` means enter short (sell)
- For short trades: `actual_return` is already sign-corrected -- positive means the short was profitable (stock fell)
- `actual_mfe` for shorts: the max favorable excursion for a short (i.e., max downward move in the stock during the window)
- Multiple patterns can activate for the same stock on the same day; a long and a short for the same stock on the same day is unusual but possible -- treat them as independent signals
- Multiple patterns can activate across different stocks on the same day
- Earnings announcements can destroy patterns -- assume you have an earnings filter available that flags any trade where an earnings date falls between entry and exit (consider this when designing strategy rules)
- In live trading, only a subset of patterns are entered -- this is the core strategy design problem

---

## What You Need to Build

### Step 1: Understand the Data

Before designing strategies, explore the data. Answer:
- What is the base win rate (actual_return > 0) across all rows?
- How does win rate vary with ml_score decile?
- How does win rate vary with win_probability decile?
- What is the correlation between predicted_return and actual_return?
- What is the distribution of actual_return and actual_mfe?
- How does performance vary by year, sector, holding_days?

### Step 2: Define a Backtesting Framework

Build a backtester that simulates a trading account with:
- A starting capital amount (your choice, but document it)
- A mechanism to select which patterns to enter each day
- Position sizing logic
- Exit rules
- Concentration limits (optional)
- Tracking of portfolio equity over time

The backtester should compute per-trade outcomes and aggregate statistics.

### Step 3: Design Strategy Variations

**IMPORTANT: Test long and short strategies separately first, then combined.** The long and short universes have different characteristics (different opportunity counts, different volatility profiles, potentially different ML signal quality). Report them independently before combining.

Design at least 50 distinct strategy configurations that vary along the following dimensions. You choose the specific parameters and ranges to test -- these are suggestions, not requirements:

**Selection / Ranking:** How do you decide which patterns to enter on a given day? Consider what signal or combination of signals best predicts which patterns will succeed. Think about whether ranking by one ML output is better than another, or whether combinations work better.

**Threshold filtering:** At what confidence level do you start entering trades? Consider the tradeoff between selectivity (fewer but better trades) and diversification (more trades, smoother equity curve).

**Position sizing:** How much capital to allocate per trade? Consider fixed sizing, volatility-adjusted sizing, or model-confidence-based sizing.

**Concentration:** How many concurrent positions? How many per sector? What happens when many patterns activate simultaneously?

**Exit rules:** The dataset provides `actual_return` which assumes hold-to-close at `holding_days`. But in reality you can exit earlier. Design exit rules using `actual_mfe` and `atr_14d_pct` to model:
- Trailing stops (flat % from high water mark)
- ATR-based stops (stop distance = N * daily ATR)
- Profit targets (exit when cumulative return reaches X%)
- Time-based exits (exit early if position is losing after N days)

**Stock selection variations:** Should you weight by sector? By volatility? By symbol history?

**Direction-aware design:** When running short strategies, consider that short patterns may activate more in certain market regimes (high VIX, bear markets). Think about whether the ML signals (ml_score, win_probability) have equal predictive power for shorts vs longs -- explore this empirically before assuming symmetry.

### Step 3.5: Direction Analysis (Long vs Short vs Combined vs Regime)

This step is required. Run it after your core strategy sweep.

**Part A: Long-only strategies**
Apply your best strategy configurations to `direction=='l'` rows only. Report all metrics.

**Part B: Short-only strategies**
Apply the same (or equivalent) configurations to `direction=='s'` rows only. Key questions:
- Is the ML signal equally predictive for shorts?
- Do the same ml_score / win_probability thresholds work, or do you need different cutoffs?
- What is the base win rate for shorts vs longs?
- Are short patterns more concentrated in specific years or sectors?

**Part C: Combined long+short portfolio**
Run the strategy simultaneously on both directions. On any given day you may have both long and short positions active. A long and a short on the same symbol on the same day should both be entered (they represent independent seasonal patterns). Report the combined portfolio statistics and compare to long-only and short-only.

**Part D: Regime-switch analysis (if the data supports it)**
Test whether a simple market regime rule improves the combined portfolio:
- Candidate regime signals from the data: year (2018-2022 included bear periods), sector trends, implied by VIX proxies (the dataset does not include VIX directly but you can infer regimes from annual return distributions)
- Example rule: "favor shorts in years where SPX is down" or "run more shorts when short win rates are higher than long win rates"
- Do NOT force a regime-switch if the data does not support it. Report whether regime switching adds value or is noise. A combined long+short without regime rules is a valid conclusion if the improvement is marginal.
- If you find a regime switch that genuinely helps, define it clearly (what signal, what threshold, how to evaluate it in live trading)

**What to report for Part D:**
- Regime-switch Sharpe vs Combined Sharpe vs Long-only Sharpe
- How often the switch fires (what % of time are you in long-only vs short-only vs combined)
- Any evidence of overfitting (does it help every year or just specific years?)

### Step 4: Evaluate Strategies

For each strategy configuration compute:
- **Sharpe Ratio** (annualized, use 0% risk-free rate)
- **Maximum Drawdown** (peak-to-trough on equity curve)
- **Win Rate** (% of trades with positive return)
- **Annualized Return** (%)
- **Number of trades**
- **Average trade return**
- **Worst single year** (Sharpe or return -- identify if any year was unprofitable)
- **All 8 years profitable?** (yes/no -- important robustness check)

Sort all strategies by Sharpe Ratio. Identify the Pareto frontier of Sharpe vs Drawdown.

### Step 5: Separate Analysis for Options

Using the same ML scorer signals, model an **options strategy** for a $10,000 account:
- Buy ATM or near-ATM calls on the selected stocks
- Use `predicted_mfe` to help size the expected move
- Model option premium as approximately 2-3% of stock price for ATM calls with 30 DTE (simplified)
- P&L = change in option value based on stock movement during holding period
- Design exit rules appropriate for options (time decay, stop on premium, etc.)
- Target: strategies that work within the $10K account size constraint

Report the same statistics as stock strategies.

### Step 6: Separate Analysis for Spreads

Using the same signals, model **vertical spread strategies**:
- **Bull put spreads (credit):** Sell a put at X% OTM, buy a put at (X + W)% OTM. Collect premium = spread width * some fraction. Win if stock stays above short put at expiry.
- **Bull call spreads (debit):** Buy ATM call, sell call at W% OTM. Max profit = spread width. Max loss = premium paid.
- Test variations on OTM distance (X%), spread width (W%), and exit timing.
- Report Sharpe, drawdown, win rate, annualized return for top configurations.

---

## Performance Benchmarks

A strategy is worth considering if it achieves:
- Sharpe > 1.5 (stocks or spreads), Sharpe > 0.8 (options)
- Maximum drawdown < 30% (stocks), < 70% (options/spreads)
- Profitable in at least 6 of 8 years

A strategy is excellent if it achieves:
- Sharpe > 3.0 (stocks or spreads), Sharpe > 1.3 (options)
- Maximum drawdown < 15% (stocks)
- Profitable all 8 years

---

## Output File 1: `results/project_v3_codex_strategy_final.md`

Structure this file as follows:

```
# V3 Strategy Backtest Results -- Independent Assessment

## Data Exploration Summary
[Key statistics from Step 1]

## Backtesting Methodology
[Describe your framework, assumptions, how exits are modeled]

## Stock Strategy Results

### Long-Only Strategies
#### Full Rankings (all N long strategies, sorted by Sharpe)
[Table: ID, Config, Sharpe, MaxDD, WinRate, AnnReturn, Trades, All8yrs]
#### Top 10 Detailed
[For each: per-year breakdown, equity curve stats, parameter interpretation]
#### Key Findings (long)
[What parameters matter most?]

### Short-Only Strategies
#### Full Rankings (all N short strategies, sorted by Sharpe)
[Table: same columns]
#### Top 10 Detailed
#### Key Findings (short)
[How does signal quality compare to longs? What changed?]

### Combined Long+Short Portfolio
#### Top 10 Combined Configurations
[Table: ID, Config, Sharpe, MaxDD, WinRate, AnnReturn, LongTrades, ShortTrades, All8yrs]
#### Per-year Breakdown for Best Combined Strategy
#### Key Findings (combined)
[Does combining add diversification benefit or just dilute the long edge?]

### Regime-Switch Analysis
[If applicable: describe the regime rule, show Sharpe improvement, year-by-year breakdown]
[If not applicable: state why regime switching did not add value]

### Overall Key Findings
[What parameters matter most? What surprised you?]

## Options Strategy Results
[Same structure]

## Spread Strategy Results
[Same structure]

## Recommended Strategy Set
[Your top picks organized as:]
- **Long-only stock portfolio**: best config with rationale
- **Short-only stock portfolio**: best config with rationale (or "not viable" with explanation)
- **Combined long+short portfolio**: recommended config and regime rule (if any)
- **Options account**: best config (long calls, or consider puts for short patterns)
- **Spreads account**: best config

[For each: state which direction(s) it trades, what the regime rule is if any, and why you chose it over alternatives]

## Independent Observations
[What does the ML scorer's output actually predict well vs poorly?]
[Where does it add value? Where is it noise?]
[Any patterns in when it fails?]
```

---

## Output File 2: `docs/TradeWave_V3_Codex_Strategy_Assessment.docx`

Write a professional Word document (.docx using python-docx) with the following sections:

1. **Executive Summary** (1 page): What is this system, what did you find, what is the headline result?

2. **ML Model Signal Quality** (1-2 pages): How predictive are the ML outputs? Include charts or tables showing win rate vs ml_score decile, predicted vs actual return correlation, year-by-year signal quality.

3. **Stock Strategy Results** (3-4 pages): Separate subsections for long-only, short-only, and combined long+short. Include a table of top 20 strategies for each direction. Highlight the best risk-adjusted and best absolute Sharpe strategy for each. Include a year-by-year performance breakdown for the top combined strategy. If a regime-switch improves the combined portfolio, describe the rule and show the before/after comparison.

4. **Options Strategy Results** (1-2 pages): Same format. Note the inherent high-drawdown nature of leveraged options.

5. **Spread Strategy Results** (1-2 pages): Same format. Distinguish credit vs debit spread performance.

6. **Risk Analysis** (1 page): Where does this strategy fail? What market conditions hurt it? What is the realistic worst-case scenario?

7. **Recommendations** (1 page): Your independent recommendation for how to deploy these strategies. What account sizes, what capital allocation, what to watch out for.

**Formatting requirements:**
- Use python-docx to generate the file
- Proper headings (Heading 1, Heading 2)
- Tables for all numerical results
- Bold key numbers
- Professional tone -- this is a strategy assessment report, not a sales document
- Do not overstate results. Note where backtest numbers may differ from live trading.

---

## Important Notes

- This is 8 years of out-of-sample data -- the ML models never saw these years during training. Results are meaningful but not a guarantee of future performance.
- The dataset does not contain bid-ask spreads or slippage. Reduce annualized return estimates by 20-30% for realistic live trading expectations.
- Patterns with earnings dates between entry and exit should be excluded. Assume ~25% of 10-30 day patterns have earnings during the window.
- Do not data mine: if you test 100+ configurations, use a hold-out year (2025) to validate your top 5 picks. Report both in-sample (2018-2024) and out-of-sample (2025) results for top strategies.
- Work with what the data tells you. Your goal is an honest independent assessment, not to find the best-possible number.
