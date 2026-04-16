# ML Pattern Scorer V2 -- Codex Review Results (Round 4)

## High

### 1. `/score` can still 500 the entire batch on malformed opportunity items before per-item error handling

- **File:** `ml_scorer/app.py`
- **Lines:** 140-158
- **Problem:** The batch pre-load step assumes every element in `opportunities` is a dict (`o.get(...)`), and `daysOut` is cast with `int(days_out)` before entering the per-opportunity `try/except`. A malformed item such as a string/list instead of a dict, or a non-integer `daysOut` like `"abc"`, raises immediately and aborts the whole request with a 500 instead of returning a well-formed per-opportunity error.
- **Why it matters:** One bad entry can drop an entire mixed batch in production.
- **Suggested fix:** Validate each list element is a dict before preloading symbols, and move `daysOut` coercion/validation inside guarded per-item logic that appends an error result and continues.

### 2. `/select` accepts unvalidated request parameter types and can fail with runtime errors

- **File:** `ml_scorer/app.py`
- **Lines:** 258-267
- **File:** `ml_scorer/daily_opp_selection.py`
- **Lines:** 81, 89, 177, 193, 251
- **Problem:** `/select` forwards raw JSON values for `num_picks`, `days_out_min`, `days_out_max`, `min_avg_return`, and `min_win_prob` directly into numeric comparisons. If the caller sends strings, nulls, or other invalid types, downstream comparisons like `df['daysOut'] >= days_out_min` or `num_picks > 0` can raise and return a 500.
- **Why it matters:** This is an externally reachable API boundary issue; bad inputs are not contained and can crash the request.
- **Suggested fix:** Parse and validate all `/select` inputs at the route boundary and return a 400-style JSON error for invalid types or ranges.

### 3. gzip fallback still silently drops broken combo files

- **File:** `ml_scorer/feature_engine.py`
- **Lines:** 273-292
- **Problem:** `_load_opp_from_gzip()` swallows every exception with `except Exception: continue` while reading combo CSVs. A corrupt gzip, schema issue, or parsing error silently removes that combo from the symbol’s depth profile with no log.
- **Why it matters:** This path is still used when parquet is absent or missing the symbol, so live scoring can produce plausible but wrong pattern features and wrong scores without any operational signal.
- **Suggested fix:** Log a warning including symbol, combo filename, and exception for every skipped file. Consider hard-failing when all combo files for a symbol fail.

### 4. `nightly.sh` does not fail fast and can restart/warm against stale parquet state

- **File:** `ml_scorer/nightly.sh`
- **Lines:** 6-22
- **Problem:** The script does not use `set -euo pipefail` and does not check step exit codes. If `opp_to_parquet.py` fails, the script still restarts the service and runs warmup. That can leave production serving stale or incomplete data while cron appears to have run.
- **Why it matters:** This is an operational integrity issue in a live trading pipeline.
- **Suggested fix:** Add `set -euo pipefail` and stop the script immediately when parquet generation, restart, or warmup fails. Log explicit failure points.

## Medium

### 5. `/select` deduplicates by raw sharpe before ML scoring, which can discard the best ML-ranked opportunity for a symbol

- **File:** `ml_scorer/daily_opp_selection.py`
- **Lines:** 99-103
- **Problem:** The selection pipeline keeps only one row per symbol based on highest raw `sharpe_ratio` before ML scoring. Since final ranking is based on `win_prob` and `pred_return`, a lower-sharpe candidate for the same symbol can be discarded even if the model would have ranked it higher.
- **Why it matters:** This can silently miss the best trade candidate for a symbol.
- **Suggested fix:** Deduplicate after ML scoring, or keep the top N raw candidates per symbol for scoring instead of only one.

## Checked, Not Findings

- PE combo parsing is consistent with current combo naming from `config_ml.PE_COMBOS` (`4_4_PE2`, etc.), so `_parse_combo()` matches the current on-disk format.
- `daily_opp_selection.load_candidates()` is in sync with `opp_to_parquet.REQUIRED_COLS` for its direct required columns; `sharpe_ratio2` is already treated as optional.
