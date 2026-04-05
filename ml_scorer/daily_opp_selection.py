"""
Daily Opportunity Selection
============================

Reads parquet caches, pre-filters candidates, scores them through the ML
ensemble, and returns the top N diversified picks ranked by win_prob.

This module is called by the /select route in app.py. It does not make
HTTP calls -- it uses the scorer and feature engine directly since it runs
inside the ML scorer service on keyprovider.

Selection pipeline:
  1. Load parquet files for the requested markets and date
  2. Filter to target date, direction, daysOut range
  3. Pre-filter: sharpe_ratio > 1 OR sharpe_ratio2 > 1
  4. Pre-filter: avg_profit >= min_avg_return
  5. Remove excluded symbols
  6. Deduplicate: keep top 3 candidates per symbol (by sharpe_ratio) for scoring
  7. Score through ML ensemble
  8. Post-filter: win_prob >= min_win_prob
  9. Rank by win_prob (primary), pred_return (tiebreaker)
  10. Pick top N with no duplicate symbols
"""

import os
import time
import logging

import pandas as pd
import numpy as np

try:
    from .config import DATA_DIR, ML_PARQUET_MARKETS, TIERS, tier_for_days_out
except ImportError:
    from config import DATA_DIR, ML_PARQUET_MARKETS, TIERS, tier_for_days_out

log = logging.getLogger('ml_scorer')


def load_candidates(date, resource_ids, direction, days_out_min, days_out_max,
                    min_avg_return, exclude_symbols=None):
    """Load and pre-filter candidates from parquet files.

    Returns a DataFrame of candidates passing all pre-filters, or empty DataFrame.
    """
    all_dfs = []

    for rid in resource_ids:
        rid = str(rid)
        if rid not in ML_PARQUET_MARKETS:
            log.warning(f'Unknown resource_id: {rid}')
            continue

        display_name, folder = ML_PARQUET_MARKETS[rid]
        path = os.path.join(DATA_DIR, folder, f'ml_cache_{date}.parquet')
        if not os.path.exists(path):
            log.warning(f'No parquet for {display_name} on {date}: {path}')
            continue

        try:
            df = pd.read_parquet(path)
            all_dfs.append(df)
            log.info(f'Loaded {display_name}: {len(df):,} rows')
        except Exception as e:
            log.warning(f'Failed to read {path}: {e}')

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Filter to target date only (parquet has 5 dates for neighbor features)
    df = df[df['date'] == date]

    # Direction filter
    if direction in ('l', 's'):
        df = df[df['LorS'] == direction]
    # 'both' keeps everything

    # DaysOut range
    df = df[(df['daysOut'] >= days_out_min) & (df['daysOut'] <= days_out_max)]

    # Sharpe filter: SR > 1 OR SR2 > 1
    sr_ok = df['sharpe_ratio'] > 1.0
    sr2_ok = df['sharpe_ratio2'] > 1.0 if 'sharpe_ratio2' in df.columns else sr_ok
    df = df[sr_ok | sr2_ok]

    # Minimum average return
    df = df[df['avg_profit'] >= min_avg_return]

    # Exclude symbols
    if exclude_symbols:
        exclude_set = {s.upper() for s in exclude_symbols}
        df = df[~df['sym'].str.upper().isin(exclude_set)]

    if df.empty:
        return df

    # Keep top 3 candidates per symbol by sharpe_ratio so the ML model can
    # choose the best one. Full dedup to 1-per-symbol happens after scoring
    # in rank_and_select. Cap at 3 to limit scoring overhead.
    df = df.sort_values('sharpe_ratio', ascending=False)
    df = df.groupby('sym', sort=False).head(3).reset_index(drop=True)

    log.info(f'Pre-filter: {len(df)} candidates ({df["sym"].nunique()} unique symbols) after filtering')
    return df


def score_candidates(candidates, engine, scorer_mgr, date):
    """Score pre-filtered candidates through the ML ensemble.

    Determines the appropriate tier based on each candidate's daysOut and scores
    using the corresponding models.

    Returns list of dicts with all metadata + ML scores.
    """
    results = []

    # Pre-load price data for all symbols at once
    symbols = candidates['sym'].unique().tolist()
    engine.load_price_data(symbols)

    for _, row in candidates.iterrows():
        symbol = row['sym']
        days_out = int(row['daysOut'])
        direction = row['LorS']

        tier_name = tier_for_days_out(days_out)

        tier = scorer_mgr.get_tier(tier_name)
        if tier is None:
            continue

        try:
            features = engine.compute_features(symbol, date, days_out, direction)

            # VIX check
            vix = features.get('mkt_vix_level')
            if vix is None or vix != vix:  # None or NaN (NaN != NaN is True)
                log.warning(f'VIX data unavailable for {symbol} -- VIX block bypassed')
            elif vix > 35:
                continue

            scores = tier.predict(features)

            results.append({
                'symbol': symbol,
                'date': date,
                'daysOut': days_out,
                'direction': direction,
                'tier': tier_name,
                'sharpe_ratio': float(row['sharpe_ratio']),
                'sharpe_ratio2': float(row.get('sharpe_ratio2', row['sharpe_ratio'])),
                'avg_profit': float(row['avg_profit']),
                'median_profit': float(row['median_profit']),
                'avg_profit2': float(row.get('avg_profit2', row['avg_profit'])),
                'ml_score': scores['ml_score'],
                'win_prob': scores['win_prob'],
                'pred_return': scores['pred_return'],
                'pred_mfe': scores['pred_mfe'],
                'p_hit_return': scores['p_hit_return'],
                'p_hit_mfe': scores['p_hit_mfe'],
            })
        except Exception as e:
            log.warning(f'Error scoring {symbol}: {e}')

    return results


def rank_and_select(scored, num_picks, min_win_prob):
    """Rank scored candidates and select top N diversified picks.

    Ranking: win_prob (descending), then pred_return (descending) as tiebreaker.
    Diversification: no duplicate symbols (already deduped, but safety check).
    """
    # Post-filter: minimum win probability
    filtered = [r for r in scored if r['win_prob'] >= min_win_prob]

    if not filtered:
        return []

    # Sort by win_prob desc, then pred_return desc
    filtered.sort(key=lambda r: (-r['win_prob'], -r['pred_return']))

    # Pick top N, skipping duplicate symbols. 0 = return all qualifying.
    picks = []
    seen_symbols = set()
    for r in filtered:
        if r['symbol'] in seen_symbols:
            continue
        picks.append(r)
        seen_symbols.add(r['symbol'])
        if num_picks > 0 and len(picks) >= num_picks:
            break

    return picks


def select_daily_opps(date, resource_ids, num_picks, direction, days_out_min,
                      days_out_max, min_avg_return, min_win_prob,
                      exclude_symbols, engine, scorer_mgr):
    """Main entry point for daily opportunity selection.

    Args:
        date: target date string 'YYYY-MM-DD'
        resource_ids: list of market resource IDs to search (e.g. ['sp500', 'etf', 'indx'])
        num_picks: number of patterns to return
        direction: 'l' for long, 's' for short, 'both'
        days_out_min: minimum holding period in days
        days_out_max: maximum holding period in days
        min_avg_return: minimum avg_profit percentage (e.g. 5.0)
        min_win_prob: minimum win probability after ML scoring (e.g. 0.80)
        exclude_symbols: list of symbols to skip (no-repeat rule)
        engine: FeatureEngine instance (shared with app.py)
        scorer_mgr: ScorerManager instance (shared with app.py)

    Returns:
        dict with 'picks' (list of selected opportunities) and metadata
    """
    t0 = time.time()

    # Step 1-6: Load and pre-filter
    candidates = load_candidates(
        date, resource_ids, direction, days_out_min, days_out_max,
        min_avg_return, exclude_symbols
    )

    if candidates.empty:
        return {
            'picks': [],
            'candidates_after_prefilter': 0,
            'candidates_scored': 0,
            'elapsed_ms': round((time.time() - t0) * 1000, 1),
            'message': 'No candidates passed pre-filters',
        }

    candidates_count = len(candidates)

    # Step 7: Score through ML ensemble
    scored = score_candidates(candidates, engine, scorer_mgr, date)

    # Step 8-10: Filter by win_prob, rank, and select
    picks = rank_and_select(scored, num_picks, min_win_prob)

    elapsed = (time.time() - t0) * 1000

    return {
        'picks': picks,
        'candidates_after_prefilter': candidates_count,
        'candidates_scored': len(scored),
        'candidates_passing_win_prob': len([r for r in scored if r['win_prob'] >= min_win_prob]),
        'elapsed_ms': round(elapsed, 1),
    }
