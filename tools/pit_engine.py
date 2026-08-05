"""
Point-in-time (PIT) seasonal pattern engine + honest-mining premise test.

WHY THIS EXISTS
---------------
The live system went flat and the leading diagnosis (see memory:
project_live_overfit_investigation) is candidate-set selection look-ahead:
opp files are mined on FULL price history (through 2026), so a pattern's
presence in any training year, and its headline features (deepest_pass,
best_winrate, avg_profit2), encode outcomes from years the walk-forward
later "validates" on. Walk-forward holds out the model fit but not the
pattern selection.

This module computes everything from price CSVs alone, strictly as-of:

1. outcome_series(): realized return of entering (symbol, month-day) and
   holding daysOut calendar days, for every available year. Entry/exit
   conventions EXACTLY mirror build_training_data.py:
     - entry  = Timestamp(f"{year}-{month_day}"), Feb-29 in non-leap year
       -> year skipped (ValueError path, matches training);
       forward trading-day search offsets 0..3 calendar days
     - exit   = entry + daysOut calendar days; forward search offsets 0..4
     - return = (close_exit - close_entry) / close_entry  (long);
       short return = -long return; win = return > 0 STRICTLY
2. pit_stats(): qualification stats for year Y using ONLY years < Y:
     - pit_deepest: max y1 in 5..40 s.t. the last y1 years are ALL present
       and wins >= floor(0.8 * y1)   (mirrors the miner's YEAR_COMBOS rule:
       combos are y1_y2 with y2 >= floor(0.8*y1); passing the loosest y2 at
       depth y1 == "qualifies at depth y1")
     - pit_wr5 / pit_wr10 / pit_wr20: win rate over last 5/10/20 years
       (gaps allowed; requires >= 60% of window present)
     - pit_q10_8: 1 if last 10 years all present and wins >= 8 (the classic
       10_8 combo, headline qualification)
3. The premise test: a hindsight-free universe (calendar grid of month-days
   x daysOut grid x both directions x all S&P symbols -- NO opp files) is
   scored per year: does PIT qualification predict NEXT-year performance?
   That is the foundational premise of the whole product, tested honestly.

USAGE
-----
  # smoke
  python tools/pit_engine.py --symbols AAPL MSFT XOM --md-step 15
  # full premise test (background, ~30-60 min on 24 cores)
  python tools/pit_engine.py --njobs 24 --out results/experiments/pit_premise

Outputs: <out>_rows.parquet (per pattern-year), <out>_summary.csv,
<out>_by_year.csv, <out>_verdict.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from config_ml import US_CSV_DIR, SP500_SYMBOLS  # noqa: E402

FWD_YEARS = list(range(2000, 2026))   # years we test forward performance on
MAX_DEPTH = 40                        # miner scans y1 in 5..40
MIN_DEPTH = 5
DAYS_OUT_GRID = [10, 20, 30, 60, 90]  # spans all three production tiers


# ----------------------------------------------------------------------
# Price loading + outcome series (conventions mirror build_training_data)
# ----------------------------------------------------------------------

def load_close(symbol):
    """Return (close_dict {Timestamp: float}, trading_days set) or None."""
    path = os.path.join(US_CSV_DIR, f'{symbol}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    close = df['close']
    return dict(zip(close.index, close.values)), set(close.index)


def outcome_series(close, tdays, month_day, days_out, year_lo, year_hi):
    """LONG return per year for entering at month_day, holding days_out
    calendar days. Returns {year: (ret, entry_timestamp)}. Missing years
    absent. Mirrors build_training_data.py entry range(0,4) / exit range(0,5)."""
    out = {}
    for yr in range(year_lo, year_hi + 1):
        try:
            base = pd.Timestamp(f'{yr}-{month_day}')
        except ValueError:
            continue  # Feb-29 in non-leap year: training skips too
        entry = None
        for off in range(0, 4):
            cand = base + pd.Timedelta(days=off)
            if cand in tdays:
                entry = cand
                break
        if entry is None:
            continue
        exit_raw = entry + pd.Timedelta(days=days_out)
        exit_d = None
        for off in range(0, 5):
            cand = exit_raw + pd.Timedelta(days=off)
            if cand in tdays:
                exit_d = cand
                break
        if exit_d is None:
            continue
        p_in = close.get(entry)
        p_out = close.get(exit_d)
        if p_in is None or p_out is None or p_in == 0 or np.isnan(p_in) or np.isnan(p_out):
            continue
        out[yr] = ((p_out - p_in) / p_in, entry)
    return out


# ----------------------------------------------------------------------
# PIT qualification stats (strictly prior years only)
# ----------------------------------------------------------------------

def pit_stats_for_years(rets_by_year, fwd_years, direction):
    """For each Y in fwd_years (where a forward outcome exists), compute PIT
    stats from years < Y only. direction: 'l' or 's'.

    Returns list of dict rows. Uses prefix sums over a continuous year axis
    so deepest-pass scan is O(depths) per year."""
    if not rets_by_year:
        return []
    y0 = min(rets_by_year)
    y1 = max(max(rets_by_year), max(fwd_years))
    n = y1 - y0 + 1
    present = np.zeros(n, dtype=np.int32)
    win = np.zeros(n, dtype=np.int32)
    ret_arr = np.full(n, np.nan)
    for yr, r in rets_by_year.items():
        i = yr - y0
        present[i] = 1
        ret_arr[i] = r if direction == 'l' else -r
        win[i] = 1 if ret_arr[i] > 0 else 0  # strict >0, matches training
    cp = np.concatenate([[0], np.cumsum(present)])
    cw = np.concatenate([[0], np.cumsum(win)])

    def window(iY, length):
        """(#present, #wins) over the `length` years ending at iY-1.

        lo is clamped to 0 so partial early-history windows return their
        partial counts (audit fix: the previous hard (0,0) made wr5/10/20
        silently stricter than the documented gaps-allowed semantics; the
        full-window checks np_ == d / p10 == 10 are unaffected because a
        clamped window can never reach the required count)."""
        lo = max(iY - length, 0)
        return cp[iY] - cp[lo], cw[iY] - cw[lo]

    rows = []
    for Y in fwd_years:
        iY = Y - y0
        if iY < 0 or iY >= n or not present[iY]:
            continue  # no forward outcome to test
        fwd_ret = ret_arr[iY]
        # deepest pass: max y1 with FULL window present and wins >= floor(0.8*y1)
        deepest = 0
        max_scan = min(MAX_DEPTH, iY)
        for d in range(max_scan, MIN_DEPTH - 1, -1):
            np_, nw = window(iY, d)
            if np_ == d and nw >= int(0.8 * d):
                deepest = d
                break
        p10, w10 = window(iY, 10)
        p5, w5 = window(iY, 5)
        p20, w20 = window(iY, 20)
        n_prior = cp[iY]
        rows.append({
            'year': Y,
            'fwd_ret': float(fwd_ret),
            'fwd_win': int(fwd_ret > 0),
            'pit_deepest': deepest,
            'pit_q10_8': int(p10 == 10 and w10 >= 8),
            'pit_wr5': (w5 / p5) if p5 >= 3 else np.nan,
            'pit_wr10': (w10 / p10) if p10 >= 6 else np.nan,
            'pit_wr20': (w20 / p20) if p20 >= 12 else np.nan,
            'pit_n_prior': int(n_prior),
            # presence counts (audit fix): enable the matched-stratum baseline
            # "could have qualified but did not" (p10 == 10 and q10_8 == 0),
            # removing the firm-age/data-continuity confound from the lift.
            'p5': int(p5), 'p10': int(p10), 'p20': int(p20),
        })
    return rows


# ----------------------------------------------------------------------
# Premise test worker (one symbol)
# ----------------------------------------------------------------------

def premise_worker(symbol, month_days, days_out_grid, vix_close=None):
    loaded = load_close(symbol)
    if loaded is None:
        return None
    close, tdays = loaded
    if len(close) < 252 * 12:   # need enough history for 10y windows by 2000
        return None
    first_year = min(tdays).year + 1
    rows = []
    for md in month_days:
        for do in days_out_grid:
            full = outcome_series(close, tdays, md, do, first_year, 2025)
            if len(full) < 12:
                continue
            rby = {yr: v[0] for yr, v in full.items()}
            # entry-date VIX (audit fix): enables the VIX<=35 hurricane-filter
            # sensitivity so the comparison to training-side numbers (which
            # apply that filter) is like-for-like.
            vix_by_year = {}
            if vix_close is not None:
                for yr, (_r, entry) in full.items():
                    v = np.nan
                    for off in range(0, 6):
                        cand = entry - pd.Timedelta(days=off)
                        if cand in vix_close:
                            v = vix_close[cand]
                            break
                    vix_by_year[yr] = v
            for direction in ('l', 's'):
                for r in pit_stats_for_years(rby, FWD_YEARS, direction):
                    r['symbol'] = symbol
                    r['month_day'] = md
                    r['daysOut'] = do
                    r['direction'] = direction
                    r['vix_entry'] = vix_by_year.get(r['year'], np.nan)
                    rows.append(r)
    if not rows:
        return None
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Aggregation / verdict
# ----------------------------------------------------------------------

DEPTH_BUCKETS = [(0, 0, 'never_qualified'), (5, 9, 'depth_5_9'),
                 (10, 14, 'depth_10_14'), (15, 19, 'depth_15_19'),
                 (20, 29, 'depth_20_29'), (30, 99, 'depth_30plus')]


def bucket_depth(d):
    for lo, hi, name in DEPTH_BUCKETS:
        if lo <= d <= hi:
            return name
    return 'never_qualified'


def aggregate(df, out_prefix):
    df = df[df['pit_n_prior'] >= 10].copy()
    df['depth_bucket'] = df['pit_deepest'].map(bucket_depth)

    def block(g):
        n = len(g)
        wr = g['fwd_win'].mean()
        mr = g['fwd_ret'].mean()
        sem = g['fwd_ret'].std() / np.sqrt(n) if n > 1 else np.nan
        return pd.Series({'n': n, 'fwd_wr': wr, 'fwd_avg_ret_pct': mr * 100,
                          'ret_t_stat': (mr / sem) if sem and sem > 0 else np.nan})

    summary = (df.groupby(['direction', 'depth_bucket'])
                 .apply(block, include_groups=False).reset_index())
    order = {name: i for i, (_, _, name) in enumerate(DEPTH_BUCKETS)}
    summary['o'] = summary['depth_bucket'].map(order)
    summary = summary.sort_values(['direction', 'o']).drop(columns='o')
    summary.to_csv(f'{out_prefix}_summary.csv', index=False)

    by_year = (df.assign(q=df['pit_q10_8'])
                 .groupby(['direction', 'q', 'year'])
                 .agg(n=('fwd_win', 'size'), fwd_wr=('fwd_win', 'mean'),
                      fwd_avg_ret_pct=('fwd_ret', lambda s: s.mean() * 100))
                 .reset_index())
    by_year.to_csv(f'{out_prefix}_by_year.csv', index=False)

    # headline verdict numbers
    verdict = {}
    for direction in ('l', 's'):
        d = df[df['direction'] == direction]
        unq = d[d['pit_deepest'] == 0]
        q108 = d[d['pit_q10_8'] == 1]
        deep = d[d['pit_deepest'] >= 15]
        verdict[direction] = {
            'n_unqualified': int(len(unq)),
            'wr_unqualified': float(unq['fwd_win'].mean()) if len(unq) else None,
            'n_q10_8': int(len(q108)),
            'wr_q10_8': float(q108['fwd_win'].mean()) if len(q108) else None,
            'lift_q10_8_pp': float((q108['fwd_win'].mean() - unq['fwd_win'].mean()) * 100)
                             if len(q108) and len(unq) else None,
            'n_deep15plus': int(len(deep)),
            'wr_deep15plus': float(deep['fwd_win'].mean()) if len(deep) else None,
            'lift_deep15plus_pp': float((deep['fwd_win'].mean() - unq['fwd_win'].mean()) * 100)
                                  if len(deep) and len(unq) else None,
            'avg_ret_q10_8_pct': float(q108['fwd_ret'].mean() * 100) if len(q108) else None,
            'avg_ret_unqualified_pct': float(unq['fwd_ret'].mean() * 100) if len(unq) else None,
        }
    verdict['context'] = {
        'training_base_wr_full_history_selected': 0.678,
        'training_selection_gap_claimed_pp': 13.0,
        'note': ('lift_*_pp is the HONEST forward edge of PIT qualification vs '
                 'never-qualified patterns in the same hindsight-free universe. '
                 'Compare against the 13pp qualify/not gap measured on the '
                 'full-history-mined training data: the difference is the '
                 'look-ahead component.'),
    }
    with open(f'{out_prefix}_verdict.json', 'w') as f:
        json.dump(verdict, f, indent=2)

    print('\n=== PIT PREMISE SUMMARY (per direction x depth bucket) ===')
    print(summary.to_string(index=False))
    print('\n=== VERDICT ===')
    print(json.dumps(verdict, indent=2))
    return summary, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='*', default=None,
                    help='explicit symbols (default: per --universe)')
    ap.add_argument('--universe', choices=['sp500', 'all'], default='sp500',
                    help='sp500 = current S&P list (survivor-biased; audit '
                         'CRITICAL); all = every CSV in csv/US (~3500 incl. '
                         'non-members, weakens 2026-membership conditioning)')
    ap.add_argument('--symbols-limit', type=int, default=None)
    ap.add_argument('--md-step', type=int, default=3,
                    help='sample every Nth day of year as entry month-day')
    ap.add_argument('--njobs', type=int, default=24)
    ap.add_argument('--out', default='results/experiments/pit_premise')
    args = ap.parse_args()

    # month-day grid from a NON-leap reference year (never yields 02-29)
    ref = pd.Timestamp('2001-01-01')
    month_days = [(ref + pd.Timedelta(days=k)).strftime('%m-%d')
                  for k in range(0, 365, args.md_step)]

    if args.symbols:
        symbols = args.symbols
    elif args.universe == 'all':
        symbols = sorted(f[:-4] for f in os.listdir(US_CSV_DIR) if f.endswith('.csv'))
    else:
        symbols = pd.read_csv(SP500_SYMBOLS)['symbols'].tolist()
    if args.symbols_limit:
        symbols = symbols[:args.symbols_limit]

    # VIX closes for the entry-date hurricane-filter sensitivity column
    vix_close = None
    vix_path = os.path.join(os.path.dirname(US_CSV_DIR), 'INDX', 'VIX.csv')
    if os.path.exists(vix_path):
        vdf = pd.read_csv(vix_path, index_col=0)
        vdf['date'] = pd.to_datetime(vdf['date'])
        vix_close = dict(zip(vdf['date'], vdf['close']))

    print(f'PIT premise test ({args.universe}): {len(symbols)} symbols x '
          f'{len(month_days)} month-days x {len(DAYS_OUT_GRID)} daysOut x 2 '
          f'directions, fwd years {FWD_YEARS[0]}-{FWD_YEARS[-1]}, njobs={args.njobs}')
    t0 = time.time()
    results = Parallel(n_jobs=args.njobs, verbose=5)(
        delayed(premise_worker)(s, month_days, DAYS_OUT_GRID, vix_close)
        for s in symbols)
    frames = [r for r in results if r is not None]
    if not frames:
        raise SystemExit('No symbols produced rows -- check universe/paths')
    df = pd.concat(frames, ignore_index=True)
    print(f'{len(df):,} pattern-year rows from {len(frames)} symbols '
          f'in {time.time()-t0:.0f}s')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(f'{args.out}_rows.parquet', index=False)
    aggregate(df, args.out)


if __name__ == '__main__':
    main()
