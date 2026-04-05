"""
Build training dataset for ML Pattern Scorer.

Approach:
1. Precompute market regime features for all dates (one-time)
2. For each S&P 500 symbol:
   a. Precompute technical + context features (vectorized across full time series)
   b. Load opportunity data, get unique (month-day, daysOut, direction) patterns
   c. Replay each pattern across training years (2015-2025)
   d. Look up precomputed features + compute label from actual prices
3. Output: parquet file with all samples

Filters:
- daysOut: 10-30 (options sweet spot, configurable)
- Only patterns that exist in at least one combo file
- Only dates with sufficient price history (200+ trading days)
"""

import os
import sys
import gzip
import csv
import time
import json
import math
import logging
import argparse
import calendar as cal_module
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config_ml import (
    US_CSV_DIR, ETF_CSV_DIR, INDX_CSV_DIR, OPP_BY_SYMBOL_DIR, EARNINGS_DIR,
    FEATURE_CACHE_DIR, TICKER_SECTOR, SECTOR_ETF, MAX_DEPTH_CAP, get_pe_year,
    N_JOBS, CSV_DIR, SPX_SEASONAL_CUTOFF_YEAR, SPX_SEASONAL_FORWARD_DAYS,
    ETF_OPP_DIR, ETF_SECTOR, ETF_CATEGORY_SECTOR_ETF
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ======================================================================
# Configuration
# ======================================================================

TRAIN_YEARS = list(range(2000, 2026))  # 2000-2025 (V2b: extended for PE-cycle depth)
DAYS_OUT_MIN = 10
DAYS_OUT_MAX = 30

# Tier definitions for multi-model training
TIERS = {
    '10_30':  (10, 30),
    '31_60':  (31, 60),
    '61_90':  (61, 90),
    '91_120': (91, 120),
    '121_200': (121, 200),
    '201_300': (201, 300),
}

def get_output_path(days_min, days_max):
    """Return parquet path for a given daysOut tier."""
    return os.path.join(FEATURE_CACHE_DIR, f'training_data_{days_min}_{days_max}.parquet')
from config_ml import SP500_SYMBOLS as SP500_SYMBOLS_PATH

# ======================================================================
# SPX Seasonal Regime Lookup (rolling per year, no leakage)
# ======================================================================

SPX_START_YEAR = 1960  # drop pre-1960 data (different market structure)

def _load_spx_data():
    """Load SPX price data once, return DataFrame."""
    path = os.path.join(CSV_DIR, 'INDX', 'SPX.csv')
    if not os.path.exists(path):
        log.warning(f"SPX.csv not found at {path}, SPX seasonal features will be NaN")
        return None
    df = pd.read_csv(path, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df


def _build_single_lookup(spx_df, start_year, end_year):
    """
    Build one SPX seasonal lookup from start_year through end_year (inclusive).
    Returns dict of (week_of_year, pe_phase) -> {'wr': float, 'ret': float}
    """
    fwd_days = SPX_SEASONAL_FORWARD_DAYS
    df = spx_df[(spx_df.index.year >= start_year) & (spx_df.index.year <= end_year)]
    if len(df) < 500:
        return {}

    closes = df['close'].values
    dates = df.index

    fwd_returns = np.full(len(closes), np.nan)
    for i in range(len(closes) - fwd_days):
        if closes[i] != 0:
            fwd_returns[i] = (closes[i + fwd_days] - closes[i]) / closes[i] * 100.0

    weeks = np.array([d.isocalendar()[1] for d in dates])
    pe_phases = np.array([get_pe_year(d.year) for d in dates])

    lookup = {}
    for wk in range(1, 54):
        for pe in range(1, 5):
            mask = (weeks == wk) & (pe_phases == pe) & ~np.isnan(fwd_returns)
            if mask.sum() < 5:
                continue
            rets = fwd_returns[mask]
            wr = (rets > 0).sum() / len(rets)
            avg_ret = rets.mean()
            lookup[(wk, pe)] = {'wr': wr, 'ret': avg_ret}

    return lookup


def compute_spx_seasonal_lookups(train_years):
    """
    Build rolling SPX seasonal lookups: one per training year.

    For each year Y in train_years, builds a lookup using SPX data from
    SPX_START_YEAR (1960) through Y-1. This prevents data leakage while
    including the most recent available data for each sample.

    Returns: dict of {year: lookup_dict}
    """
    spx_df = _load_spx_data()
    if spx_df is None:
        return {}

    log.info(f"SPX data: {len(spx_df)} rows from {spx_df.index[0].date()} to {spx_df.index[-1].date()}")

    lookups = {}
    for year in train_years:
        end_yr = year - 1
        lk = _build_single_lookup(spx_df, SPX_START_YEAR, end_yr)
        lookups[year] = lk

    # Log summary
    sizes = [len(v) for v in lookups.values()]
    years_range = f"{train_years[0]}-{train_years[-1]}"
    spans = f"{SPX_START_YEAR} to (year-1)"
    log.info(f"SPX seasonal lookups: {len(lookups)} years ({years_range}), "
             f"data span {spans}, entries per year: {min(sizes)}-{max(sizes)}")
    return lookups


# ======================================================================
# Vectorized Technical Indicator Computation
# ======================================================================

def compute_all_ta_series(df):
    """
    Precompute technical indicators for an entire price series.
    V2: Only keep 5 TA features that had importance in V1 (dropped 15 zero-importance).
    Kept: ta_trend_long, ta_price_vs_sma200, ta_sma50_vs_sma200, ta_trend_direction_match, ta_rvol_20
    """
    closes = df['close']
    highs = df['high']
    lows = df['low']
    volumes = df['volume']

    ta = pd.DataFrame(index=df.index)

    # SMAs (needed for kept features + trend score)
    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()

    # ATR (needed for normalization)
    tr = pd.concat([
        highs - lows,
        (highs - closes.shift(1)).abs(),
        (lows - closes.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    atr_safe = atr14.replace(0, np.nan)

    # Kept TA features
    ta['ta_price_vs_sma200'] = (closes - sma200) / atr_safe
    ta['ta_sma50_vs_sma200'] = (sma50 - sma200) / atr_safe

    # Relative volume
    avg_vol_20 = volumes.rolling(20).mean()
    ta['ta_rvol_20'] = volumes / avg_vol_20.replace(0, np.nan)

    # Trend long score (composite, was #11 in V1 importance)
    sma20 = closes.rolling(20).mean()
    sma100 = closes.rolling(100).mean()

    # MACD components for trend score
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    histogram = ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    ema5 = closes.ewm(span=5, adjust=False).mean()
    wma_weights = np.arange(1, 14, dtype=float)
    wma13 = closes.rolling(13).apply(lambda x: np.average(x, weights=wma_weights), raw=True)

    # RSI for trend score
    delta = closes.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))

    tl = pd.Series(0.0, index=df.index)
    for sma_vals in [sma20, sma50, sma100, sma200]:
        diff = (closes - sma_vals) / atr_safe
        tl += (5 + diff).clip(0, 10)

    macd_pos = (histogram > 0)
    tl += np.where(macd_pos, 7, 3)
    macd_norm = (histogram.abs() / atr_safe).clip(0, 5) / 5 * 10
    tl += np.where(macd_pos, macd_norm, 10 - macd_norm)
    tl += np.where(ema5 > wma13, 7, 3)
    tl += np.where(closes > ema5, 7, 3)
    tl += (rsi_14 / 10).clip(0, 10)
    roc20_abs = closes.pct_change(20).abs()
    adx_proxy = (roc20_abs * 200).clip(0, 10)
    tl += adx_proxy / 2 + 2.5

    ta['ta_trend_long'] = tl

    return ta


def compute_all_context_series(df):
    """
    Precompute stock-specific context features for entire series.
    V2: Only keep 2 context features that had importance (dropped 5 zero-importance).
    """
    closes = df['close']
    highs = df['high']
    lows = df['low']

    ctx = pd.DataFrame(index=df.index)

    # 52-week (252 trading days) high/low
    high_52w = highs.rolling(252, min_periods=20).max()
    low_52w = lows.rolling(252, min_periods=20).min()

    ctx['ctx_pct_from_52w_high'] = (closes - high_52w) / high_52w
    ctx['ctx_pct_from_52w_low'] = (closes - low_52w) / low_52w.replace(0, np.nan)

    return ctx


def compute_market_regime_series():
    """
    Precompute market regime features for all dates.
    V2: Added fed rate, VIX regime bucket, breadth momentum.
    Dropped always-NaN columns (spy_breadth_approx, new_highs_lows_10d).
    Returns DataFrame indexed by date.
    """
    def load_csv(subdir, name):
        path = os.path.join(CSV_DIR, subdir, f'{name}.csv')
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df

    mkt = pd.DataFrame()

    # VIX
    vix_df = load_csv('INDX', 'VIX')
    if vix_df is not None:
        mkt['mkt_vix_level'] = vix_df['close']
        # Vectorized percentile rank using pandas rolling
        vix_close = vix_df['close']
        mkt['mkt_vix_percentile_60d'] = vix_close.rolling(60).rank(pct=True)
        mkt['mkt_vix_5d_change'] = vix_close.pct_change(5)

        # V2: VIX regime bucket (0=<15, 1=15-20, 2=20-30, 3=>30)
        mkt['mkt_vix_regime_bucket'] = pd.cut(
            vix_close, bins=[0, 15, 20, 30, 999], labels=[0, 1, 2, 3]
        ).astype(float).reindex(mkt.index)

        vix3m_df = load_csv('INDX', 'VIX3M')
        if vix3m_df is not None:
            mkt['mkt_vix_term_structure'] = vix_close - vix3m_df['close'].reindex(vix_df.index, method='ffill')
        else:
            mkt['mkt_vix_term_structure'] = np.nan

    # Yield curve
    us10y = load_csv('INDX', 'US10Y')
    us2y = load_csv('INDX', 'US2Y')
    if us10y is not None and us2y is not None:
        spread = us10y['close'] - us2y['close'].reindex(us10y.index, method='ffill')
        mkt['mkt_yield_curve_10y2y'] = spread.reindex(mkt.index, method='ffill') if len(mkt) > 0 else spread
        mkt['mkt_yield_curve_slope'] = mkt['mkt_yield_curve_10y2y'].diff(5)

    # Credit spread (HYG/LQD ratio)
    hyg = load_csv('ETF', 'HYG')
    lqd = load_csv('ETF', 'LQD')
    if hyg is not None and lqd is not None:
        ratio = hyg['close'] / lqd['close'].reindex(hyg.index, method='ffill')
        if len(mkt) > 0:
            mkt['mkt_credit_spread'] = ratio.reindex(mkt.index, method='ffill')
        else:
            mkt['mkt_credit_spread'] = ratio
        mkt['mkt_credit_spread_change_20d'] = mkt['mkt_credit_spread'].diff(20)

    # SPY
    spy = load_csv('ETF', 'SPY')
    if spy is not None:
        if len(mkt) == 0:
            mkt = pd.DataFrame(index=spy.index)
        spy_close = spy['close'].reindex(mkt.index, method='ffill')
        mkt['mkt_spy_roc_20'] = spy_close.pct_change(20)
        sma200 = spy_close.rolling(200).mean()
        mkt['mkt_spy_above_sma200'] = (spy_close > sma200).astype(int)

    # Breadth
    advn = load_csv('INDX', 'ADVN')
    decn = load_csv('INDX', 'DECN')
    if advn is not None and decn is not None:
        adv_10 = advn['close'].rolling(10).mean()
        dec_10 = decn['close'].rolling(10).mean()
        ratio = adv_10 / dec_10.replace(0, np.nan)
        adl_ratio = ratio.reindex(mkt.index, method='ffill')
        mkt['mkt_adv_decl_ratio_10d'] = adl_ratio
        # V2: Breadth momentum - 20-day change in adv/decl ratio
        mkt['mkt_breadth_momentum'] = adl_ratio.diff(20)
    else:
        mkt['mkt_adv_decl_ratio_10d'] = np.nan
        mkt['mkt_breadth_momentum'] = np.nan

    # Sector rotation: XLK vs XLU 20d
    xlk = load_csv('ETF', 'XLK')
    xlu = load_csv('ETF', 'XLU')
    if xlk is not None and xlu is not None:
        xlk_ret = xlk['close'].pct_change(20)
        xlu_ret = xlu['close'].pct_change(20)
        mkt['mkt_sector_rotation'] = (xlk_ret - xlu_ret).reindex(mkt.index, method='ffill')
    else:
        mkt['mkt_sector_rotation'] = np.nan

    # V2: Fed rate proxy from IRX (3-month T-bill yield)
    irx = load_csv('INDX', 'IRX')
    if irx is not None:
        irx_close = irx['close'].reindex(mkt.index, method='ffill')
        mkt['mkt_fed_rate_level'] = irx_close
        mkt['mkt_fed_rate_direction'] = irx_close.diff(60)
    else:
        mkt['mkt_fed_rate_level'] = np.nan
        mkt['mkt_fed_rate_direction'] = np.nan

    # V3: Commodity / FX regime -- 20-day ROC for DXY, CL, GC
    dxy = load_csv('INDX', 'DXY')
    mkt['mkt_dxy_roc_20'] = dxy['close'].reindex(mkt.index, method='ffill').pct_change(20) if dxy is not None else np.nan

    cl = load_csv('COMM', 'CL')
    mkt['mkt_cl_roc_20'] = cl['close'].reindex(mkt.index, method='ffill').pct_change(20) if cl is not None else np.nan

    gc = load_csv('COMM', 'GC')
    mkt['mkt_gc_roc_20'] = gc['close'].reindex(mkt.index, method='ffill').pct_change(20) if gc is not None else np.nan

    return mkt


def compute_rs_vs_spy(stock_close, spy_close):
    """Compute 20-day relative strength vs SPY."""
    stock_ret = stock_close.pct_change(20)
    spy_ret = spy_close.pct_change(20)
    return stock_ret - spy_ret.reindex(stock_close.index, method='ffill')


def compute_sector_rs(stock_close, sector_etf_sym, spy_close, etf_data):
    """Compute sector and stock-vs-sector relative strength."""
    etf_close = etf_data.get(sector_etf_sym)
    if etf_close is None:
        return pd.Series(np.nan, index=stock_close.index), pd.Series(np.nan, index=stock_close.index)

    etf_c = etf_close.reindex(stock_close.index, method='ffill')
    spy_c = spy_close.reindex(stock_close.index, method='ffill')

    etf_ret = etf_c.pct_change(20)
    spy_ret = spy_c.pct_change(20)
    stock_ret = stock_close.pct_change(20)

    sector_rs = etf_ret - spy_ret
    stock_vs_sector = stock_ret - etf_ret

    return sector_rs, stock_vs_sector


# ======================================================================
# Pattern Feature Extraction
# ======================================================================

def load_opp_patterns(symbol, days_min, days_max, opp_base_dir=None):
    """
    Load opportunity data for a symbol and extract unique patterns.
    Uses raw string splitting for speed (3-4x faster than csv.DictReader).
    Returns:
        patterns: set of (month_day_str, daysOut, direction) e.g. ('03-01', 12, 'l')
        combo_data: dict of combo_name -> {(date_str, daysOut, dir) -> row_dict}
    """
    if opp_base_dir is None:
        opp_base_dir = OPP_BY_SYMBOL_DIR
    opp_dir = os.path.join(opp_base_dir, symbol)
    if not os.path.isdir(opp_dir):
        return set(), {}

    patterns = set()
    combo_data = {}

    # Expected column positions in the opp CSV (fast positional parse).
    # If the upstream format changes, the header check below will catch it.
    _EXPECTED_HEADER = 'LorS,date,daysOut,sym,sharpe_ratio,avg_profit,median_profit'
    _COL_LORS = 0
    _COL_DATE = 1
    _COL_DAYS_OUT = 2
    _COL_SHARPE = 4
    _COL_AVG_PROFIT = 5
    _COL_MEDIAN_PROFIT = 6
    _COL_AVG_PROFIT2 = 8  # optional, added later

    for fname in os.listdir(opp_dir):
        if not fname.endswith('.csv.gz'):
            continue
        combo_name = fname.replace('.csv.gz', '')
        path = os.path.join(opp_dir, fname)
        lookup = {}
        try:
            with gzip.open(path, 'rt') as fh:
                header = next(fh).rstrip()
                # Validate header matches expected column layout before positional parsing
                if not header.startswith(_EXPECTED_HEADER):
                    log.warning(
                        f'Schema mismatch in {symbol}/{fname}: '
                        f'expected header starting with "{_EXPECTED_HEADER}", '
                        f'got "{header[:80]}" -- skipping combo'
                    )
                    continue
                for line in fh:
                    parts = line.rstrip().split(',')
                    d = int(parts[_COL_DAYS_OUT])
                    if d < days_min or d > days_max:
                        continue
                    direction = parts[_COL_LORS]
                    date_str = parts[_COL_DATE]
                    month_day = date_str[5:]  # MM-DD

                    patterns.add((month_day, d, direction))
                    avg_p = float(parts[_COL_AVG_PROFIT])
                    lookup[(date_str, d, direction)] = {
                        'sharpe_ratio': float(parts[_COL_SHARPE]),
                        'avg_profit': avg_p,
                        'median_profit': float(parts[_COL_MEDIAN_PROFIT]),
                        'avg_profit2': float(parts[_COL_AVG_PROFIT2]) if len(parts) > _COL_AVG_PROFIT2 and parts[_COL_AVG_PROFIT2] else avg_p,
                    }
        except Exception as e:
            log.warning(f'Skipping combo file {symbol}/{fname}: {e}')
            continue
        if lookup:
            combo_data[combo_name] = lookup

    return patterns, combo_data


def compute_pattern_features_fast(combo_data, date_2026, daysOut, direction, data_years):
    """
    Compute Group 1 pattern features using preloaded combo data.
    V2: Dropped zero-importance features (pat_avg_profit, pat_median_profit, pat_mfe_ratio,
    pat_profit_per_day, pat_daysOut, pat_daysOut_bucket, pat_passes_at_max_depth).
    Added: pat_consistency_std.
    """
    lookup_key = (date_2026, daysOut, direction)

    best_sharpe = None
    best_row = None
    deepest_pass = 0
    pe_deepest = 0
    num_combos_qualifying = 0
    passes_recent_10 = 0
    sharpe_at_10 = None
    sharpe_at_deepest = None
    best_winrate = 0.0
    worst_winrate = 1.0
    deepest_pass_capped30 = 0
    max_possible_depth = 0
    winrates_by_depth = []  # V2: for consistency_std

    for combo_name, lookup in combo_data.items():
        is_pe = combo_name.endswith('_PE2')
        parts = combo_name.split('_')
        year1 = int(parts[0])
        year2 = int(parts[1])

        if not is_pe and year1 > max_possible_depth:
            max_possible_depth = year1

        row = lookup.get(lookup_key)
        if row is None:
            continue

        winrate = year2 / year1 if year1 > 0 else 0
        num_combos_qualifying += 1

        if is_pe:
            if year1 > pe_deepest:
                pe_deepest = year1
        else:
            winrates_by_depth.append(winrate)
            if year1 > deepest_pass:
                deepest_pass = year1
                sharpe_at_deepest = row['sharpe_ratio']
            if year1 == 10 and year2 == 10:
                passes_recent_10 = 1
                sharpe_at_10 = row['sharpe_ratio']
            if min(year1, 30) > deepest_pass_capped30:
                deepest_pass_capped30 = min(year1, 30)

        if winrate > best_winrate:
            best_winrate = winrate
        if winrate < worst_winrate:
            worst_winrate = winrate
        if best_sharpe is None or row['sharpe_ratio'] > best_sharpe:
            best_sharpe = row['sharpe_ratio']
            best_row = row

    if best_row is None:
        return None

    avg_profit2 = best_row['avg_profit2']
    depth_denom = min(data_years, MAX_DEPTH_CAP)
    pe_denom = depth_denom / 4.0

    # V2: consistency_std - low std across depth levels = robust
    pat_consistency_std = float(np.std(winrates_by_depth)) if len(winrates_by_depth) > 1 else 0.0

    return {
        'pat_sharpe_ratio': best_row['sharpe_ratio'],
        'pat_avg_profit2': avg_profit2,
        'pat_direction': 1 if direction == 'l' else 0,
        'pat_data_years': data_years,
        'pat_deepest_pass': deepest_pass,
        'pat_depth_utilization': deepest_pass / depth_denom if depth_denom > 0 else 0,
        'pat_passes_recent_10': passes_recent_10,
        'pat_recent_vs_deep_sharpe': sharpe_at_10 / sharpe_at_deepest if sharpe_at_10 and sharpe_at_deepest else np.nan,
        'pat_num_combos_qualifying': num_combos_qualifying,
        'pat_pe_match': 1 if pe_deepest > 0 else 0,
        'pat_pe_deepest': pe_deepest,
        'pat_pe_utilization': pe_deepest / pe_denom if pe_denom > 0 else 0,
        'pat_best_winrate': best_winrate,
        'pat_worst_winrate': worst_winrate if worst_winrate < 1.0 else best_winrate,
        'pat_deepest_pass_capped30': deepest_pass_capped30,
        'pat_consistency_std': pat_consistency_std,
    }


# ======================================================================
# V2 Neighborhood Features (leak-free: uses only PRIOR-year data)
# ======================================================================

def compute_neighborhood_features(close_values, trading_days_set, price_index,
                                  entry_date, daysOut, direction):
    """
    Compute pattern temporal neighborhood features using ONLY prior-year data.
    For each shifted window (entry +/- 7/14 days), compute average return
    across years BEFORE the sample's year. This avoids data leakage.
    """
    dir_mult = 1 if direction == 'l' else -1
    sample_year = entry_date.year
    month_day = f"{entry_date.month:02d}-{entry_date.day:02d}"
    shifts = [(-14, 'pre2w'), (-7, 'pre1w'), (7, 'post1w'), (14, 'post2w')]

    # Compute shifted returns across prior years only
    shifted_wrs = {}  # label -> list of wins (1/0) across prior years
    pat_wrs_prior = []  # pattern itself across prior years

    # Use up to 10 prior years for stability
    prior_years = list(range(max(sample_year - 10, 2000), sample_year))

    for shift_days, label in shifts:
        wins = []
        for yr in prior_years:
            try:
                # Build historical base date first, then shift so that year-crossing
                # is handled correctly (e.g. Dec-28 + 7 days -> Jan-04 of yr+1,
                # not Jan-04 of yr as the old sample-year-rewrite approach did).
                hist_base = pd.Timestamp(f"{yr}-{month_day}")
                shifted_entry = hist_base + pd.Timedelta(days=shift_days)
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                continue

            # Find nearest trading day
            actual_entry = None
            for offset in range(0, 4):
                candidate = shifted_entry + pd.Timedelta(days=offset)
                if candidate in trading_days_set:
                    actual_entry = candidate
                    break
            if actual_entry is None:
                continue

            shifted_exit = actual_entry + pd.Timedelta(days=daysOut)
            actual_exit = None
            for offset in range(0, 5):
                candidate = shifted_exit + pd.Timedelta(days=offset)
                if candidate in trading_days_set:
                    actual_exit = candidate
                    break
            if actual_exit is None:
                continue

            try:
                p_in = close_values[actual_entry]
                p_out = close_values[actual_exit]
                if p_in == 0 or np.isnan(p_in) or np.isnan(p_out):
                    continue
                ret = (p_out - p_in) / p_in * dir_mult
                wins.append(1.0 if ret > 0 else 0.0)
            except (KeyError, IndexError):
                continue

        shifted_wrs[label] = wins

    # Also compute pattern's own prior-year win rate for comparison
    for yr in prior_years:
        try:
            pat_entry = pd.Timestamp(f"{yr}-{month_day}")
        except (ValueError, pd.errors.OutOfBoundsDatetime):
            continue
        actual_entry = None
        for offset in range(0, 4):
            candidate = pat_entry + pd.Timedelta(days=offset)
            if candidate in trading_days_set:
                actual_entry = candidate
                break
        if actual_entry is None:
            continue
        pat_exit = actual_entry + pd.Timedelta(days=daysOut)
        actual_exit = None
        for offset in range(0, 5):
            candidate = pat_exit + pd.Timedelta(days=offset)
            if candidate in trading_days_set:
                actual_exit = candidate
                break
        if actual_exit is None:
            continue
        try:
            p_in = close_values[actual_entry]
            p_out = close_values[actual_exit]
            if p_in == 0 or np.isnan(p_in) or np.isnan(p_out):
                continue
            ret = (p_out - p_in) / p_in * dir_mult
            pat_wrs_prior.append(1.0 if ret > 0 else 0.0)
        except (KeyError, IndexError):
            continue

    # Aggregate across shifts
    all_shifted_wrs = []
    pre_avg_rets = []
    post_avg_rets = []
    for label, wins in shifted_wrs.items():
        if wins:
            wr = float(np.mean(wins))
            all_shifted_wrs.append(wr)
            if 'pre' in label:
                pre_avg_rets.append(wr)
            else:
                post_avg_rets.append(wr)

    neighbor_avg = float(np.mean(all_shifted_wrs)) if all_shifted_wrs else np.nan
    pat_wr_prior = float(np.mean(pat_wrs_prior)) if pat_wrs_prior else np.nan

    result = {
        'pat_neighbor_avg_wr': neighbor_avg,
        'pat_sharpness': pat_wr_prior / neighbor_avg if neighbor_avg and neighbor_avg > 0 and not np.isnan(pat_wr_prior) else np.nan,
    }

    # Pre-slope: are shifted win rates building up approaching the pattern?
    if len(pre_avg_rets) == 2:
        result['pat_pre_slope'] = pre_avg_rets[1] - pre_avg_rets[0]  # pre1w_wr - pre2w_wr
    else:
        result['pat_pre_slope'] = np.nan

    # Post-cliff: pattern WR minus post-pattern WR (large positive = cliff after end)
    if post_avg_rets and not np.isnan(pat_wr_prior):
        result['pat_post_cliff'] = pat_wr_prior - float(np.mean(post_avg_rets))
    else:
        result['pat_post_cliff'] = np.nan

    return result


# ======================================================================
# Main Training Data Builder
# ======================================================================

def load_sp500_symbols():
    """Load S&P 500 ticker list."""
    df = pd.read_csv(SP500_SYMBOLS_PATH)
    return df['symbols'].tolist()


def load_price_csv(symbol, csv_dir=None):
    """Load a stock/ETF price CSV."""
    if csv_dir is None:
        csv_dir = US_CSV_DIR
    path = os.path.join(csv_dir, f'{symbol}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df


def process_symbol(symbol, mkt_regime, spy_close, etf_closes, spx_lookups=None,
                   days_min=None, days_max=None, csv_dir=None, opp_base_dir=None,
                   sector_lookup=None):
    """
    Process a single symbol: compute all features for all training years.
    V2: Parallelized via joblib. Each worker runs this independently.
    spx_lookups: dict of {year: spx_seasonal_lookup} for rolling lookups.
    days_min/days_max: daysOut filter range (defaults to module-level DAYS_OUT_MIN/MAX).
    csv_dir/opp_base_dir: override directories for ETF support.
    sector_lookup: dict mapping symbol->sector (defaults to TICKER_SECTOR).
    Returns DataFrame or None.
    """
    if days_min is None:
        days_min = DAYS_OUT_MIN
    if days_max is None:
        days_max = DAYS_OUT_MAX
    if sector_lookup is None:
        sector_lookup = TICKER_SECTOR

    # Load price data
    price_df = load_price_csv(symbol, csv_dir=csv_dir)
    if price_df is None or len(price_df) < 252:
        return None

    data_years = len(price_df) / 252.0

    # Load opportunity patterns
    patterns, combo_data = load_opp_patterns(symbol, days_min, days_max, opp_base_dir=opp_base_dir)
    if not patterns:
        return None

    # Derive opp file year dynamically from actual combo_data keys so this code
    # survives the annual opp-file year rollover (2026 -> 2027 -> ...).
    opp_year = None
    for _lookup in combo_data.values():
        for _key in _lookup:
            opp_year = _key[0][:4]  # 'YYYY' from 'YYYY-MM-DD'
            break
        if opp_year is not None:
            break
    if opp_year is None:
        return None

    # Precompute technical indicators (vectorized)
    ta_series = compute_all_ta_series(price_df)

    # Precompute context features
    ctx_series = compute_all_context_series(price_df)

    # Build trading day set for fast lookup
    trading_days = set(price_df.index)
    price_index = price_df.index
    close_values = price_df['close']

    # Combine all precomputed series into one DataFrame for fast merge
    combined = ta_series.copy()
    for col in ctx_series.columns:
        combined[col] = ctx_series[col]

    # Precompute calendar features for all dates
    cal_obj = cal_module.Calendar()

    # Step 1: Build list of (entry_date, daysOut, direction, pat_feats_idx)
    rows_entry = []
    pat_list = []

    # V2: Count patterns per (date, symbol) for pat_concurrent_count
    from collections import Counter
    date_pattern_counter = Counter()

    for month_day, daysOut, direction in patterns:
        opp_date = f"{opp_year}-{month_day}"
        pat_feats = compute_pattern_features_fast(combo_data, opp_date, daysOut, direction, data_years)
        if pat_feats is None:
            continue

        pat_idx = len(pat_list)
        pat_list.append(pat_feats)

        for year in TRAIN_YEARS:
            try:
                entry_date = pd.Timestamp(f"{year}-{month_day}")
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                continue

            # Find nearest trading day (forward up to 3 days)
            if entry_date not in trading_days:
                found = False
                for offset in range(1, 4):
                    alt = entry_date + pd.Timedelta(days=offset)
                    if alt in trading_days:
                        entry_date = alt
                        found = True
                        break
                if not found:
                    continue

            # Need enough history
            entry_loc = price_index.get_loc(entry_date)
            if entry_loc < 200:
                continue

            rows_entry.append((entry_date, daysOut, direction, pat_idx))
            date_pattern_counter[entry_date] += 1

    if not rows_entry:
        return None

    # Step 2: Build samples DataFrame in batch
    entry_dates = [r[0] for r in rows_entry]
    days_outs = [r[1] for r in rows_entry]
    directions = [r[2] for r in rows_entry]
    pat_idxs = [r[3] for r in rows_entry]

    # Compute labels in batch
    entry_prices = close_values.reindex(entry_dates).values
    exit_dates_raw = [ed + pd.Timedelta(days=do) for ed, do in zip(entry_dates, days_outs)]

    # Find nearest trading day for each exit
    exit_prices = np.full(len(rows_entry), np.nan)
    for i, exit_d in enumerate(exit_dates_raw):
        for offset in range(0, 5):
            alt = exit_d + pd.Timedelta(days=offset)
            if alt in trading_days:
                exit_prices[i] = close_values[alt]
                break

    actual_returns = (exit_prices - entry_prices) / entry_prices * 100
    # Flip sign for short
    dir_multiplier = np.array([1 if d == 'l' else -1 for d in directions])
    actual_returns = actual_returns * dir_multiplier

    # Compute MFE (maximum favorable excursion) return per sample
    # For longs: max close in window. For shorts: min close in window.
    mfe_returns = np.full(len(rows_entry), np.nan)
    for i in range(len(rows_entry)):
        if np.isnan(entry_prices[i]) or entry_prices[i] == 0:
            continue
        ed = entry_dates[i]
        ex = exit_dates_raw[i]
        # Get all closes between entry (exclusive) and exit (inclusive)
        window = close_values.loc[(close_values.index > ed) & (close_values.index <= ex + pd.Timedelta(days=4))]
        if len(window) == 0:
            continue
        if directions[i] == 'l':
            best_price = window.max()
            mfe_returns[i] = (best_price - entry_prices[i]) / entry_prices[i] * 100
        else:
            best_price = window.min()
            mfe_returns[i] = (entry_prices[i] - best_price) / entry_prices[i] * 100

    # Filter valid: exclude NaN actual_return AND NaN mfe_return so both SR and
    # MFE training targets are always non-null for every surviving row.
    valid = (~np.isnan(actual_returns) & ~np.isnan(entry_prices) & (entry_prices != 0)
             & ~np.isnan(mfe_returns))
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return None

    sample_dates = [entry_dates[i] for i in valid_idx]

    sym_df = pd.DataFrame({
        'symbol': symbol,
        'date': sample_dates,
        'daysOut': [days_outs[i] for i in valid_idx],
        'direction': [directions[i] for i in valid_idx],
        'hit_target': (actual_returns[valid_idx] > 0).astype(int),
        'actual_return': actual_returns[valid_idx],
        'mfe_return': mfe_returns[valid_idx],
    })

    # Merge pattern features
    pat_rows = [pat_list[pat_idxs[i]] for i in valid_idx]
    pat_df = pd.DataFrame(pat_rows, index=sym_df.index)
    sym_df = pd.concat([sym_df, pat_df], axis=1)

    # V2: Concurrent pattern count
    sym_df['pat_concurrent_count'] = [date_pattern_counter[entry_dates[i]] for i in valid_idx]

    # V2b: Pattern holding period as feature (useful for multi-tier models)
    sym_df['pat_daysOut'] = sym_df['daysOut'].astype(np.float32)

    # V2: Neighborhood features (per sample)
    nbr_rows = []
    for i in valid_idx:
        nbr = compute_neighborhood_features(
            close_values, trading_days, price_index,
            entry_dates[i], days_outs[i], directions[i]
        )
        nbr_rows.append(nbr)
    if nbr_rows:
        nbr_df = pd.DataFrame(nbr_rows, index=sym_df.index)
        sym_df = pd.concat([sym_df, nbr_df], axis=1)

    # V2: Pattern history features
    # pat_hit_last_year: did this exact pattern work the prior year?
    # pat_recent_3yr_wr: win rate in most recent 3 years vs all-time
    hit_last_year = np.full(len(sym_df), np.nan)
    for idx_row, i in enumerate(valid_idx):
        ed = entry_dates[i]
        do = days_outs[i]
        d = directions[i]
        # Look for same pattern 1 year ago
        try:
            prev_entry = pd.Timestamp(f"{ed.year - 1}-{ed.month:02d}-{ed.day:02d}")
        except ValueError:
            continue
        # Find nearest trading day
        actual_prev = None
        for offset in range(0, 4):
            candidate = prev_entry + pd.Timedelta(days=offset)
            if candidate in trading_days:
                actual_prev = candidate
                break
        if actual_prev is None:
            continue
        prev_exit = actual_prev + pd.Timedelta(days=do)
        actual_prev_exit = None
        for offset in range(0, 5):
            candidate = prev_exit + pd.Timedelta(days=offset)
            if candidate in trading_days:
                actual_prev_exit = candidate
                break
        if actual_prev_exit is None:
            continue
        try:
            p_in = close_values[actual_prev]
            p_out = close_values[actual_prev_exit]
            mult = 1 if d == 'l' else -1
            ret = (p_out - p_in) / p_in * mult
            hit_last_year[idx_row] = 1 if ret > 0 else 0
        except (KeyError, IndexError):
            pass
    sym_df['pat_hit_last_year'] = hit_last_year

    # Merge precomputed TA + context features by date
    sym_df = sym_df.set_index('date')
    merged = sym_df.join(combined, how='left')
    merged = merged.reset_index()

    # Trend direction match (V2: kept from V1, uses ta_trend_long)
    tl = merged['ta_trend_long']
    is_long = merged['direction'] == 'l'
    tdm = pd.Series(0, index=merged.index, dtype=float)
    tdm[is_long & (tl >= 60)] = 1
    tdm[is_long & (tl <= 40)] = -1
    tdm[~is_long & (tl >= 60)] = -1
    tdm[~is_long & (tl <= 40)] = 1
    tdm[tl.isna()] = np.nan
    merged['ta_trend_direction_match'] = tdm

    # Market regime (merge by date, forward-fill)
    mkt_reindexed = mkt_regime.reindex(merged['date'], method='ffill')
    for col in mkt_regime.columns:
        merged[col] = mkt_reindexed[col].values

    # V2: Interaction features
    spy_close_at_date = spy_close.reindex(merged['date'], method='ffill')
    spy_sma200 = spy_close.rolling(200).mean().reindex(merged['date'], method='ffill')

    # pat_dir_x_mkt_trend: pattern direction aligned with SPY trend?
    spy_trend = np.where(spy_close_at_date > spy_sma200, 1, -1)
    pat_dir_sign = np.where(merged['pat_direction'] == 1, 1, -1)
    merged['pat_dir_x_mkt_trend'] = pat_dir_sign * spy_trend

    # pat_dir_x_sector_trend: pattern direction aligned with sector ETF trend?
    sector = sector_lookup.get(symbol)
    etf_sym = SECTOR_ETF.get(sector, '') if sector else ''
    # For ETF categories not in SECTOR_ETF, try ETF_CATEGORY_SECTOR_ETF
    if not etf_sym and sector:
        etf_sym = ETF_CATEGORY_SECTOR_ETF.get(sector, '')
    if etf_sym and etf_sym in etf_closes:
        etf_c = etf_closes[etf_sym]
        etf_at_date = etf_c.reindex(merged['date'], method='ffill')
        etf_sma200 = etf_c.rolling(200).mean().reindex(merged['date'], method='ffill')
        sector_trend = np.where(etf_at_date > etf_sma200, 1, -1)
        merged['pat_dir_x_sector_trend'] = pat_dir_sign * sector_trend
    else:
        merged['pat_dir_x_sector_trend'] = np.nan

    # pat_depth_x_vix: deep patterns in low-vol environments
    vix_norm = merged.get('mkt_vix_level', pd.Series(np.nan, index=merged.index))
    vix_inv = 1.0 / vix_norm.replace(0, np.nan)
    merged['pat_depth_x_vix'] = merged['pat_deepest_pass'] * vix_inv

    # pat_quality_x_regime: quality in bull market
    merged['pat_quality_x_regime'] = merged['pat_sharpe_ratio'] * merged.get('mkt_spy_above_sma200', np.nan)

    # SPX seasonal regime features (rolling per-year lookups, no leakage)
    if spx_lookups:
        dates_tmp = pd.to_datetime(merged['date'])
        years_arr = dates_tmp.dt.year.values
        wk_arr = np.array([d.isocalendar()[1] for d in dates_tmp])
        pe_arr = np.array([get_pe_year(d.year) for d in dates_tmp])

        spx_wr = np.full(len(merged), np.nan)
        spx_ret = np.full(len(merged), np.nan)
        for i in range(len(merged)):
            yr_lookup = spx_lookups.get(int(years_arr[i]))
            if yr_lookup is None:
                continue
            key = (int(wk_arr[i]), int(pe_arr[i]))
            entry = yr_lookup.get(key)
            if entry is not None:
                spx_wr[i] = entry['wr']
                spx_ret[i] = entry['ret']

        merged['mkt_spx_seasonal_wr'] = spx_wr
        merged['mkt_spx_seasonal_ret'] = spx_ret

        # Regime bucket: -2 (<40%), -1 (40-50%), 0 (50-60%), +1 (60-70%), +2 (>70%)
        regime = np.full(len(merged), np.nan)
        regime[spx_wr < 0.40] = -2
        regime[(spx_wr >= 0.40) & (spx_wr < 0.50)] = -1
        regime[(spx_wr >= 0.50) & (spx_wr < 0.60)] = 0
        regime[(spx_wr >= 0.60) & (spx_wr < 0.70)] = 1
        regime[spx_wr >= 0.70] = 2
        merged['mkt_spx_seasonal_regime'] = regime

        # Direction alignment: +1 (aligned with strong season), 0 (neutral), -1 (against)
        pat_dir = np.where(merged['pat_direction'] == 1, 1, -1)
        alignment = np.full(len(merged), np.nan)
        valid = ~np.isnan(spx_wr)
        # Neutral zone: 45-55% WR (no directional edge)
        bullish = valid & (spx_wr >= 0.55)
        bearish = valid & (spx_wr <= 0.45)
        neutral = valid & (spx_wr > 0.45) & (spx_wr < 0.55)
        season_dir = np.zeros(len(merged))
        season_dir[bullish] = 1
        season_dir[bearish] = -1
        alignment[neutral] = 0
        alignment[bullish | bearish] = pat_dir[bullish | bearish] * season_dir[bullish | bearish]
        merged['mkt_spx_dir_alignment'] = alignment
    else:
        merged['mkt_spx_seasonal_wr'] = np.nan
        merged['mkt_spx_seasonal_ret'] = np.nan
        merged['mkt_spx_seasonal_regime'] = np.nan
        merged['mkt_spx_dir_alignment'] = np.nan

    # Calendar features (vectorized) - V2: dropped cal_quarter, cal_month_position
    dates = pd.to_datetime(merged['date'])
    merged['cal_month'] = dates.dt.month
    merged['cal_day_of_year'] = dates.dt.dayofyear / (365 + dates.dt.is_leap_year.astype(int))
    merged['cal_week_of_month'] = (dates.dt.day - 1) // 7 + 1
    merged['cal_pe_year'] = dates.dt.year.map(get_pe_year)

    # OpEx week (batch by year-month)
    is_opex = np.zeros(len(merged), dtype=int)
    for (yr, mo), grp in merged.groupby([dates.dt.year, dates.dt.month]):
        fridays = [d for d in cal_obj.itermonthdays2(yr, mo) if d[0] != 0 and d[1] == 4]
        if len(fridays) >= 3:
            opex_day = fridays[2][0]
            opex_dt = pd.Timestamp(yr, mo, opex_day)
            opex_start = opex_dt - pd.Timedelta(days=opex_dt.weekday())
            opex_end = opex_start + pd.Timedelta(days=4)
            mask = (dates[grp.index] >= opex_start) & (dates[grp.index] <= opex_end)
            is_opex[grp.index[mask]] = 1
    merged['cal_is_opex_week'] = is_opex

    return merged


def build_training_data(n_jobs=None, days_min=None, days_max=None, etf_mode=False):
    """Main entry point: build the full training dataset. V2 with joblib parallelization.
    etf_mode: if True, build ETF training data instead of stocks.
    """
    if n_jobs is None:
        n_jobs = N_JOBS
    if days_min is None:
        days_min = DAYS_OUT_MIN
    if days_max is None:
        days_max = DAYS_OUT_MAX

    if etf_mode:
        tier_tag = f'{days_min}_{days_max}'
        output_path = os.path.join(FEATURE_CACHE_DIR, f'training_data_{tier_tag}_etf.parquet')
        log.info(f"Starting ETF training data build (years {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}, "
                 f"daysOut {days_min}-{days_max}, n_jobs={n_jobs})")
    else:
        output_path = get_output_path(days_min, days_max)
        log.info(f"Starting V2 training data build (years {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}, "
                 f"daysOut {days_min}-{days_max}, n_jobs={n_jobs})")

    # Step 1: Precompute market regime features (single-threaded, shared)
    log.info("Precomputing market regime features...")
    t0 = time.time()
    mkt_regime = compute_market_regime_series()
    log.info(f"Market regime: {len(mkt_regime)} dates, {len(mkt_regime.columns)} features in {time.time()-t0:.1f}s")
    log.info(f"Market features: {list(mkt_regime.columns)}")

    # Load SPY close for interaction features
    spy_df = load_price_csv('SPY')
    if spy_df is None:
        spy_path = os.path.join(ETF_CSV_DIR, 'SPY.csv')
        spy_df = pd.read_csv(spy_path, index_col=0)
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        spy_df = spy_df.set_index('date').sort_index()
    spy_close = spy_df['close']

    # Load sector ETF closes
    etf_closes = {}
    for etf_sym in set(SECTOR_ETF.values()):
        path = os.path.join(ETF_CSV_DIR, f'{etf_sym}.csv')
        if os.path.exists(path):
            edf = pd.read_csv(path, index_col=0)
            edf['date'] = pd.to_datetime(edf['date'])
            edf = edf.set_index('date').sort_index()
            etf_closes[etf_sym] = edf['close']

    # Step 1b: Precompute SPX seasonal lookups (rolling per year, no leakage)
    log.info("Computing SPX seasonal regime lookups (rolling per year)...")
    t0 = time.time()
    spx_lookups = compute_spx_seasonal_lookups(TRAIN_YEARS)
    log.info(f"SPX seasonal lookups computed in {time.time()-t0:.1f}s")

    # Step 2: Get symbol list
    if etf_mode:
        # Use ETF symbols from config, filtered to those with opportunity dirs
        symbols = sorted(ETF_SECTOR.keys())
        symbols = [s for s in symbols if os.path.isdir(os.path.join(ETF_OPP_DIR, s))]
        csv_dir = ETF_CSV_DIR
        opp_base_dir = ETF_OPP_DIR
        sector_lookup = ETF_SECTOR
        log.info(f"Processing {len(symbols)} ETF symbols with {n_jobs} workers")
    else:
        symbols = load_sp500_symbols()
        symbols = [s for s in symbols if os.path.isdir(os.path.join(OPP_BY_SYMBOL_DIR, s))]
        csv_dir = None  # defaults to US_CSV_DIR
        opp_base_dir = None  # defaults to OPP_BY_SYMBOL_DIR
        sector_lookup = None  # defaults to TICKER_SECTOR
        log.info(f"Processing {len(symbols)} symbols with {n_jobs} workers")

    # Step 3: Parallel per-symbol processing
    t_start = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_symbol)(symbol, mkt_regime, spy_close, etf_closes, spx_lookups,
                                days_min, days_max, csv_dir, opp_base_dir, sector_lookup)
        for symbol in symbols
    )

    # Filter None results
    all_samples = [r for r in results if r is not None]
    skipped = len(results) - len(all_samples)
    total_samples = sum(len(r) for r in all_samples)
    elapsed = time.time() - t_start
    log.info(f"Processed {len(symbols)} symbols in {elapsed:.1f}s ({skipped} skipped)")
    log.info(f"Total samples: {total_samples:,}")

    # Concatenate
    log.info("Concatenating all symbols...")
    df = pd.concat(all_samples, ignore_index=True)
    log.info(f"DataFrame shape: {df.shape}")
    log.info(f"Columns ({len(df.columns)}): {list(df.columns)}")

    # Save
    log.info(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info(f"Saved {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")

    # Stats
    log.info(f"\nLabel distribution:")
    log.info(f"  hit_target=1: {(df['hit_target']==1).sum():,} ({(df['hit_target']==1).mean()*100:.1f}%)")
    log.info(f"  hit_target=0: {(df['hit_target']==0).sum():,} ({(df['hit_target']==0).mean()*100:.1f}%)")
    log.info(f"  avg actual_return: {df['actual_return'].mean():.2f}%")
    log.info(f"\nYear distribution:")
    log.info(df.groupby(df['date'].dt.year)['hit_target'].agg(['count', 'mean']).to_string())

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--njobs', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--days-min', type=int, default=None, help='Minimum daysOut (default: 10)')
    parser.add_argument('--days-max', type=int, default=None, help='Maximum daysOut (default: 30)')
    parser.add_argument('--tier', type=str, default=None,
                        choices=list(TIERS.keys()),
                        help='Named tier (e.g. 31_60, 91_120). Overrides --days-min/max.')
    parser.add_argument('--all-tiers', action='store_true',
                        help='Build training data for all 6 tiers sequentially')
    parser.add_argument('--etf', action='store_true',
                        help='Build ETF training data (uses ETF CSVs and opportunity dirs)')
    parser.add_argument('--append-etf', action='store_true',
                        help='After building ETF data, append to the main stock parquet')
    args = parser.parse_args()

    if args.all_tiers:
        for tier_name, (dmin, dmax) in TIERS.items():
            log.info(f"\n{'='*60}")
            log.info(f"BUILDING TIER: {tier_name} (daysOut {dmin}-{dmax})")
            log.info(f"{'='*60}")
            build_training_data(n_jobs=args.njobs, days_min=dmin, days_max=dmax,
                                etf_mode=args.etf)
    else:
        days_min = args.days_min
        days_max = args.days_max
        if args.tier:
            days_min, days_max = TIERS[args.tier]
        df = build_training_data(n_jobs=args.njobs, days_min=days_min, days_max=days_max,
                                 etf_mode=args.etf)

        # Optionally append ETF data to the main stock parquet
        if args.append_etf and args.etf:
            tier_tag = f'{days_min}_{days_max}'
            stock_path = get_output_path(days_min, days_max)
            etf_path = os.path.join(FEATURE_CACHE_DIR, f'training_data_{tier_tag}_etf.parquet')
            combined_path = os.path.join(FEATURE_CACHE_DIR, f'training_data_{tier_tag}_combined.parquet')
            if os.path.exists(stock_path) and os.path.exists(etf_path):
                log.info(f"Appending ETF data to stock data...")
                stock_df = pd.read_parquet(stock_path)
                etf_df = pd.read_parquet(etf_path)
                log.info(f"  Stock samples: {len(stock_df):,}")
                log.info(f"  ETF samples: {len(etf_df):,}")
                combined = pd.concat([stock_df, etf_df], ignore_index=True)
                combined.to_parquet(combined_path, index=False)
                log.info(f"  Combined samples: {len(combined):,}")
                log.info(f"  Saved to {combined_path} ({os.path.getsize(combined_path) / 1e6:.1f} MB)")
            else:
                log.error(f"Cannot append: stock={os.path.exists(stock_path)}, etf={os.path.exists(etf_path)}")
