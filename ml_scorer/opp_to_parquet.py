#!/usr/bin/env python3
"""
Daily ML Parquet Cache Generator
=================================

Reads opportunity data from the opportunities/ directories for all configured
markets and writes compact parquet files for the ML scorer to load instantly
at inference time.

HOW IT WORKS:
  The TradeWave platform stores pattern opportunities in gzip CSV files organized
  as: data/<market>/opportunities/Monthly_Opp_<Month>_<combo>/<date>.csv.gz

  Each date file contains all symbols for that market/combo/date. For ML scoring,
  we need data for 5 dates per target date: the target date itself plus +-7 and
  +-14 day neighbors (used for pattern neighborhood features).

  This script reads those small date files directly (no scanning of huge files),
  combines them per market, and writes one parquet per market per target date.

OUTPUT:
  data/<market>/ml_cache_<YYYY-MM-DD>.parquet

  Example: /home/flask/data/sp500/ml_cache_2026-03-17.parquet

  The parquet lives alongside opp_by_symbol/ and opportunities/ so it auto-deletes
  when opportunity data is refreshed monthly (the whole market folder gets replaced).

SCHEDULE:
  Cron: 0 1 * * * (1am ET every night)
  - Generates parquets for current week's remaining weekdays
  - On Saturday, generates the full following week (Mon-Fri)
  - Skips dates that already have a parquet (idempotent)

RUNTIME:
  ~15 seconds per date across all 6 markets. Full week = ~75 seconds.

USAGE:
  # Normal nightly run (auto-determines dates)
  python3 opp_to_parquet.py

  # Generate for specific dates
  python3 opp_to_parquet.py 2026-03-17 2026-03-18

  # Dry run (show what would be generated, don't write)
  python3 opp_to_parquet.py --dry-run
"""

import os
import sys
import gzip
import logging
from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd

# Allow running standalone or as part of the ml_scorer package
try:
    from .config import DATA_DIR, ML_PARQUET_MARKETS
except ImportError:
    try:
        from config import DATA_DIR, ML_PARQUET_MARKETS
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from config import DATA_DIR, ML_PARQUET_MARKETS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('opp_to_parquet')

# Columns we need from the opportunity CSVs.
# Used by the ML scorer's feature_engine and by daily_opp_selection for pre-filtering.
KEEP_COLS = [
    'LorS', 'date', 'daysOut', 'sym',
    'sharpe_ratio', 'avg_profit', 'median_profit', 'sharpe_ratio2', 'avg_profit2',
]

# Columns that must be present for the parquet to be usable.
# sharpe_ratio2 / avg_profit2 are optional (backfilled from base columns if absent).
REQUIRED_COLS = ['LorS', 'date', 'daysOut', 'sym', 'sharpe_ratio', 'avg_profit', 'median_profit']

# Month name lookup for directory names (Monthly_Opp_March_10_10)
MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December',
}


def get_target_dates():
    """Determine which dates need parquets based on current day of week.

    - Mon-Fri: generate for remaining weekdays this week (today through Friday)
    - Saturday: generate for next Mon-Fri
    - Sunday: generate for next Mon-Fri
    """
    today = datetime.now().date()
    weekday = today.weekday()  # 0=Mon, 6=Sun

    dates = []
    if weekday <= 4:  # Mon-Fri: rest of this week
        d = today
        while d.weekday() <= 4:
            dates.append(d)
            d += timedelta(days=1)
    else:  # Sat/Sun: next full week
        # Jump to next Monday
        next_monday = today + timedelta(days=(7 - weekday))
        for i in range(5):
            dates.append(next_monday + timedelta(days=i))

    return dates


def get_neighbor_dates(target_date):
    """Return the 5 dates needed for ML scoring: target + neighbors at +-7, +-14 days."""
    return [
        target_date + timedelta(days=shift)
        for shift in [-14, -7, 0, 7, 14]
    ]


def get_months_needed(dates):
    """Return set of (year, month) tuples covering all dates."""
    return {(d.year, d.month) for d in dates}


def find_combo_dirs(opp_base, month_name):
    """Find all combo directories for a given month.

    Returns list of (combo_dir_path, combo_suffix) tuples.
    combo_suffix is the part after Monthly_Opp_<Month>_, e.g. '10_10' or '10_8_PE2'.
    """
    if not os.path.isdir(opp_base):
        return []

    prefix = f'Monthly_Opp_{month_name}_'
    results = []
    for name in os.listdir(opp_base):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            results.append((os.path.join(opp_base, name), suffix))
    return results


def read_date_file(path):
    """Read a single date gzip CSV, keeping only the columns the ML scorer needs.

    Returns a DataFrame or None if the file doesn't exist, is empty, or fails
    schema validation. Logs a warning (not just debug) on any failure so that
    missing files are visible in nightly logs.
    """
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, 'rb') as gz:
            raw = gz.read()
        if not raw:
            return None
        df = pd.read_csv(BytesIO(raw))

        # Validate required columns are present before proceeding.
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            log.warning(f'Schema mismatch in {path}: missing required columns {missing} -- skipping file')
            return None

        # Keep only the columns the scorer needs. avg_profit2/sharpe_ratio2 are optional.
        available = [c for c in KEEP_COLS if c in df.columns]
        df = df[available]
        if 'avg_profit2' not in df.columns:
            df['avg_profit2'] = df['avg_profit']
        if 'sharpe_ratio2' not in df.columns:
            df['sharpe_ratio2'] = df['sharpe_ratio']
        return df
    except Exception as e:
        log.warning(f'Failed to read {path}: {e}')
        return None


def build_parquet_for_date(target_date, market_folder, opp_base):
    """Build the parquet for one market and one target date.

    Reads all combo directories for the months needed (target + neighbor dates),
    extracts only rows matching the 5 required dates, adds a 'combo' column,
    and returns a single DataFrame ready to write.
    """
    neighbor_dates = get_neighbor_dates(target_date)
    date_strs = {d.strftime('%Y-%m-%d') for d in neighbor_dates}
    date_files = {d.strftime('%Y-%m-%d') + '.csv.gz' for d in neighbor_dates}
    months_needed = get_months_needed(neighbor_dates)

    all_dfs = []
    files_read = 0

    for year, month in months_needed:
        month_name = MONTH_NAMES[month]
        combo_dirs = find_combo_dirs(opp_base, month_name)

        for combo_dir, combo_suffix in combo_dirs:
            # Only read the specific date files we need
            for date_fname in date_files:
                path = os.path.join(combo_dir, date_fname)
                df = read_date_file(path)
                if df is not None and len(df) > 0:
                    df['combo'] = combo_suffix
                    all_dfs.append(df)
                    files_read += 1

    if not all_dfs:
        return None, 0, 0

    combined = pd.concat(all_dfs, ignore_index=True)

    # Downcast to save memory and disk
    for col in ['sharpe_ratio', 'avg_profit', 'median_profit', 'sharpe_ratio2', 'avg_profit2']:
        if col in combined.columns:
            combined[col] = combined[col].astype('float32')
    if 'daysOut' in combined.columns:
        combined['daysOut'] = combined['daysOut'].astype('int16')

    return combined, files_read, len(combined)


def parquet_path_for(market_folder, target_date):
    """Return the parquet file path for a market and date."""
    date_str = target_date.strftime('%Y-%m-%d')
    return os.path.join(DATA_DIR, market_folder, f'ml_cache_{date_str}.parquet')


def generate_all(target_dates, dry_run=False):
    """Generate parquets for all markets and all target dates.

    Skips any (market, date) combination where the parquet already exists.
    """
    total_written = 0
    total_skipped = 0

    for target_date in target_dates:
        date_str = target_date.strftime('%Y-%m-%d')

        for rid, (display_name, folder) in ML_PARQUET_MARKETS.items():
            out_path = parquet_path_for(folder, target_date)

            if os.path.exists(out_path):
                log.debug(f'SKIP {display_name} {date_str} (exists)')
                total_skipped += 1
                continue

            if dry_run:
                log.info(f'WOULD generate {display_name} {date_str} -> {out_path}')
                total_skipped += 1
                continue

            opp_base = os.path.join(DATA_DIR, folder, 'opportunities')
            if not os.path.isdir(opp_base):
                log.warning(f'No opportunities dir for {display_name}: {opp_base}')
                continue

            df, files_read, row_count = build_parquet_for_date(
                target_date, folder, opp_base
            )

            if df is None or len(df) == 0:
                log.warning(f'No data for {display_name} {date_str} (0 rows from {opp_base})')
                continue

            # Write parquet
            market_dir = os.path.join(DATA_DIR, folder)
            if not os.path.isdir(market_dir):
                log.error(f'Market dir missing: {market_dir}')
                continue

            df.to_parquet(out_path, index=False, compression='snappy')
            size_mb = os.path.getsize(out_path) / 1024 / 1024
            log.info(
                f'{display_name} {date_str}: {row_count:,} rows from {files_read} files '
                f'-> {size_mb:.1f} MB ({out_path})'
            )
            total_written += 1

    log.info(f'Done. Written: {total_written}, skipped: {total_skipped}')


def main():
    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    if dry_run:
        args.remove('--dry-run')

    if args:
        # Explicit dates provided
        target_dates = []
        for arg in args:
            try:
                target_dates.append(datetime.strptime(arg, '%Y-%m-%d').date())
            except ValueError:
                log.error(f'Invalid date format: {arg} (expected YYYY-MM-DD)')
                sys.exit(1)
    else:
        target_dates = get_target_dates()

    if not target_dates:
        log.info('No dates to generate.')
        return

    log.info(f'Target dates: {", ".join(d.strftime("%Y-%m-%d (%a)") for d in target_dates)}')
    log.info(f'Markets: {", ".join(name for name, _ in ML_PARQUET_MARKETS.values())}')

    if dry_run:
        log.info('DRY RUN - no files will be written')

    generate_all(target_dates, dry_run=dry_run)


if __name__ == '__main__':
    main()
