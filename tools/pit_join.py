"""
Join point-in-time (PIT) pattern features onto a training parquet.

For every unique (symbol, entry month-day, daysOut) pattern in the training
data, computes the per-year PIT qualification stats from price CSVs alone
(tools/pit_engine.py machinery -- strictly prior years, verified leak-free by
the pit-leak-audit workflow) and merges them onto the training rows:

  pit_deepest, pit_q10_8, pit_wr5, pit_wr10, pit_wr20, pit_n_prior,
  pit_depth_x_vix (honest replacement for the leaky pat_depth_x_vix)

These are the honest replacements for the opp-file-derived pattern features
(pat_deepest_pass, pat_sharpe_ratio, pat_best_winrate, ...), which encode
full-history selection look-ahead.

Note: the training row 'date' is the forward-SNAPPED entry trading day, so the
month-day key carries up to 3 days of calendar jitter vs the pattern's base
month-day for rows that snapped. Prior-year PIT windows inherit that jitter;
the premise test's neighbor analysis shows +-3d windows are near-identical, so
the effect is noise, and it is identical across A/B arms.

Usage:
  python tools/pit_join.py --tier 10_30 --njobs 24
  -> features/training_data_10_30_pit.parquet
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TOOLS)
sys.path.insert(0, _TOOLS)
sys.path.insert(0, _REPO)

from pit_engine import load_close, outcome_series, pit_stats_for_years, FWD_YEARS  # noqa: E402

PIT_COLS = ['pit_deepest', 'pit_q10_8', 'pit_wr5', 'pit_wr10', 'pit_wr20', 'pit_n_prior']


def join_worker(symbol, patterns):
    """patterns: list of (month_day, daysOut). Returns DataFrame of PIT stats
    keyed by (symbol, month_day, daysOut, direction, year)."""
    loaded = load_close(symbol)
    if loaded is None:
        return None
    close, tdays = loaded
    first_year = min(tdays).year + 1
    rows = []
    for md, do in patterns:
        full = outcome_series(close, tdays, md, int(do), first_year, 2025)
        if not full:
            continue
        rby = {yr: v[0] for yr, v in full.items()}
        for direction in ('l', 's'):
            for r in pit_stats_for_years(rby, FWD_YEARS, direction):
                rows.append({
                    'symbol': symbol, 'month_day': md, 'daysOut': int(do),
                    'direction': direction, 'year': r['year'],
                    'pit_deepest': r['pit_deepest'], 'pit_q10_8': r['pit_q10_8'],
                    'pit_wr5': r['pit_wr5'], 'pit_wr10': r['pit_wr10'],
                    'pit_wr20': r['pit_wr20'], 'pit_n_prior': r['pit_n_prior'],
                })
    return pd.DataFrame(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default='10_30')
    ap.add_argument('--njobs', type=int, default=24)
    ap.add_argument('--symbols-limit', type=int, default=None)
    args = ap.parse_args()

    src = os.path.join(_REPO, 'features', f'training_data_{args.tier}.parquet')
    dst = os.path.join(_REPO, 'features', f'training_data_{args.tier}_pit.parquet')

    print(f'Loading {src} ...')
    df = pd.read_parquet(src)
    df['date'] = pd.to_datetime(df['date'])
    df['month_day'] = df['date'].dt.strftime('%m-%d')
    df['year'] = df['date'].dt.year.astype(np.int32)
    print(f'{len(df):,} rows, {df["symbol"].nunique()} symbols')

    pat = df[['symbol', 'month_day', 'daysOut']].drop_duplicates()
    by_sym = {s: list(zip(g['month_day'], g['daysOut']))
              for s, g in pat.groupby('symbol')}
    symbols = sorted(by_sym)
    if args.symbols_limit:
        symbols = symbols[:args.symbols_limit]
        df = df[df['symbol'].isin(symbols)].copy()
    print(f'{len(pat):,} unique (symbol, month-day, daysOut) patterns '
          f'across {len(symbols)} symbols')

    t0 = time.time()
    results = Parallel(n_jobs=args.njobs, verbose=5)(
        delayed(join_worker)(s, by_sym[s]) for s in symbols)
    pit = pd.concat([r for r in results if r is not None], ignore_index=True)
    print(f'PIT stats: {len(pit):,} pattern-direction-years in {time.time()-t0:.0f}s')

    merged = df.merge(pit, on=['symbol', 'month_day', 'daysOut', 'direction', 'year'],
                      how='left')
    assert len(merged) == len(df), 'merge fan-out: duplicate PIT keys'
    cov = merged['pit_deepest'].notna().mean()
    print(f'PIT coverage: {cov*100:.2f}% of training rows')

    # honest replacement for the leaky pat_depth_x_vix interaction
    vix = merged.get('mkt_vix_level')
    merged['pit_depth_x_vix'] = np.where(
        (vix > 0) & merged['pit_deepest'].notna(),
        merged['pit_deepest'] / vix, np.nan)

    for c in PIT_COLS + ['pit_depth_x_vix']:
        merged[c] = merged[c].astype(np.float32)
    merged = merged.drop(columns=['month_day'])

    merged.to_parquet(dst, index=False)
    print(f'Saved {dst} ({os.path.getsize(dst)/1e9:.2f} GB)')

    # sanity: PIT vs leaky depth correlation (expect positive but far from 1)
    both = merged[['pit_deepest', 'pat_deepest_pass']].dropna()
    if len(both):
        print(f'corr(pit_deepest, pat_deepest_pass) = '
              f'{both["pit_deepest"].corr(both["pat_deepest_pass"]):.3f} '
              f'(n={len(both):,})')


if __name__ == '__main__':
    main()
