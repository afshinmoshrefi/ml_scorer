"""
Production Feature Engine for ML Pattern Scorer (V2).

Computes all features for any (symbol, date, daysOut, direction) opportunity.
Adapted from the V1 training feature_engine.py with V2 features added:
  - Pattern V2: consistency_std, concurrent_count, neighbor_avg_wr, sharpness,
    pre_slope, post_cliff, hit_last_year, daysOut
  - Market Regime V2: vix_regime_bucket, breadth_momentum, fed_rate_level, fed_rate_direction
  - Interaction: pat_dir_x_mkt_trend, pat_dir_x_sector_trend, pat_depth_x_vix, pat_quality_x_regime

6 feature groups:
  1. Pattern-Intrinsic (23 V1 + 8 V2)
  2. Technical Momentum (20)
  3. Market Regime (14 V1 + 4 V2)
  4. Stock-Specific Context (9)
  5. Calendar/Cycle (7)
  6. SPX Seasonal Regime (4)
  + Interaction features (4)

Usage:
  engine = FeatureEngine()
  engine.load_price_data(['AAPL', 'MSFT'])
  features = engine.compute_features('AAPL', '2024-03-15', 20, 'l')
"""

import os
import gzip
import json
import hashlib
import math
import warnings
from collections import OrderedDict
from datetime import timedelta

import numpy as np
import pandas as pd

try:
    from .config import (
        DATA_DIR, ML_PARQUET_MARKETS,
        US_CSV_DIR, ETF_CSV_DIR, INDX_CSV_DIR, COMM_CSV_DIR, OPP_BY_SYMBOL_DIR, EARNINGS_DIR,
        YEAR_COMBOS, PE_COMBOS, MAX_DEPTH_CAP, TICKER_SECTOR, SECTOR_ETF,
        get_pe_year, SPX_SEASONAL_FORWARD_DAYS, CSV_DIR,
        ETF_SECTOR, ETF_CATEGORY_SECTOR_ETF, PATTERN_RISK_FREE_RETURN,
        CONTEXT_PROFILE_CACHE_MAX, CONTEXT_SNAPSHOT_CACHE_MAX,
        CONTEXT_TARGET_CACHE_MAX,
    )
except ImportError:
    from config import (
        DATA_DIR, ML_PARQUET_MARKETS,
        US_CSV_DIR, ETF_CSV_DIR, INDX_CSV_DIR, COMM_CSV_DIR, OPP_BY_SYMBOL_DIR, EARNINGS_DIR,
        YEAR_COMBOS, PE_COMBOS, MAX_DEPTH_CAP, TICKER_SECTOR, SECTOR_ETF,
        get_pe_year, SPX_SEASONAL_FORWARD_DAYS, CSV_DIR,
        ETF_SECTOR, ETF_CATEGORY_SECTOR_ETF, PATTERN_RISK_FREE_RETURN,
        CONTEXT_PROFILE_CACHE_MAX, CONTEXT_SNAPSHOT_CACHE_MAX,
        CONTEXT_TARGET_CACHE_MAX,
    )

# SPX_SEASONAL_CUTOFF_YEAR is not in the package config; define here.
# Training used 1960-1999 to avoid leakage into 2000+ training data.
SPX_SEASONAL_CUTOFF_YEAR = 1999

warnings.filterwarnings('ignore')

# Aggregate pattern fields that are actually consumed by the 62-feature V3
# model.  Legacy diagnostic fields are still populated, but a harmless median
# rounding difference must not invalidate an otherwise exact model profile.
_V3_AGGREGATE_PROFILE_KEYS = (
    'pat_sharpe_ratio', 'pat_avg_profit2', 'pat_direction',
    'pat_data_years', 'pat_deepest_pass', 'pat_depth_utilization',
    'pat_passes_recent_10', 'pat_recent_vs_deep_sharpe',
    'pat_num_combos_qualifying', 'pat_pe_match', 'pat_pe_deepest',
    'pat_pe_utilization', 'pat_best_winrate', 'pat_worst_winrate',
    'pat_deepest_pass_capped30', 'pat_consistency_std', 'pat_daysOut',
)

# Opportunity rows are the authoritative inputs used to train V3.  The raw-price
# rebuild must independently reproduce the qualifying combo set and the consumed
# aggregate profile before those rows can be served.  Two upstream, two-decimal
# statistics can differ slightly after later OHLC corrections: Sharpe at the last
# displayed decimal, and favorable excursion because it depends on intrawindow
# highs/lows.  Keep those reconciliations tightly bounded instead of rejecting a
# model-faithful prebuilt profile or accepting an arbitrary mismatch.
_V3_PROFILE_VALIDATION_TOLERANCES = {
    'pat_sharpe_ratio': (0.011, 0.01),
    'pat_avg_profit2': (0.011, 0.05),
}

_PINNED_PRICE_SYMBOLS = frozenset({
    'SPY', 'HYG', 'LQD', 'XLK', 'XLU', 'XLF', 'XLE', 'XLV', 'XLY',
    'XLC', 'XLI', 'XLP', 'XLRE', 'XLB', 'VIX', 'VIX3M', 'US10Y',
    'US2Y', 'ADVN', 'DECN', 'IRX', 'DXY', 'TLT',
})


class PatternProfileUnavailable(RuntimeError):
    """Raised when an exact checkpoint has no trustworthy V3 pattern profile."""

    def __init__(self, reason, details=None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


class FeatureEngine:
    """Computes all features for pattern opportunities (V1 + V2)."""

    def __init__(self):
        # Price data cache: symbol -> DataFrame (or str path for lazy-loaded user symbols)
        self._price_cache = {}
        # Opportunity depth profile cache: symbol -> dict (gzip-backed only -- full history)
        # Parquet-backed results are NOT stored here because they are date-scoped slices.
        self._opp_cache = {}
        # Parquet by-symbol cache: pre-split at load time for O(1) symbol lookup
        # Reset when date changes -- parquets are date-scoped so stale splits are invalid.
        self._parquet_date = None           # date string of currently loaded parquets
        self._parquet_by_symbol = {}        # symbol -> DataFrame (pre-split from parquet)
        self._parquet_loaded_markets = set()  # market folders already loaded for this date
        # Market data (SPY, VIX, bonds, credit, sector ETFs)
        self._market_cache = {}
        # Commodity regime inputs live in a separate namespace.  This prevents
        # the crude-oil context ticker CL from shadowing NYSE equity CL.
        self._commodity_price_cache = {}
        # Precomputed technical indicators cache
        self._ta_cache = {}
        # Breadth data
        self._breadth_cache = None
        # SPX seasonal lookup (lazy-loaded)
        self._spx_seasonal_lookup = None
        # Resource-aware snapshots and dynamically rebuilt checkpoint profiles.
        self._context_opp_snapshot_cache = OrderedDict()
        self._recalculated_profile_cache = OrderedDict()
        self._selected_recurrence_cache = OrderedDict()
        self._context_target_cache = OrderedDict()
        self._price_source_versions = {}

    @staticmethod
    def _lru_get(cache, key):
        try:
            value = cache.pop(key)
        except KeyError:
            return None
        cache[key] = value
        return value

    @staticmethod
    def _lru_put(cache, key, value, maximum):
        cache.pop(key, None)
        cache[key] = value
        evicted = None
        if len(cache) > maximum:
            evicted = cache.popitem(last=False)
        return evicted

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load_csv(self, path):
        """Load a price CSV into a DataFrame indexed by date."""
        df = pd.read_csv(path, index_col=0)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        # Use adjusted close if adj_factor exists
        if 'adj_factor' in df.columns:
            df['close_raw'] = df['close']
            # adj_factor * close = adjusted close (but we keep raw for volume calcs)
        return df

    def _find_csv_path(self, symbol):
        """Resolve symbol to CSV file path."""
        # Try US first (SP500 stocks), then ETF, then INDX, then COMM
        for subdir in ['US', 'ETF', 'INDX', 'COMM']:
            path = os.path.join(CSV_DIR, subdir, f'{symbol}.csv')
            if os.path.exists(path):
                return path
        return None

    def load_price_data(self, symbols):
        """Load price data for a list of symbols."""
        # Always load market data
        market_symbols = {
            'SPY': ETF_CSV_DIR, 'HYG': ETF_CSV_DIR, 'LQD': ETF_CSV_DIR,
            'XLK': ETF_CSV_DIR, 'XLU': ETF_CSV_DIR, 'XLF': ETF_CSV_DIR,
            'XLE': ETF_CSV_DIR, 'XLV': ETF_CSV_DIR, 'XLY': ETF_CSV_DIR,
            'XLC': ETF_CSV_DIR, 'XLI': ETF_CSV_DIR, 'XLP': ETF_CSV_DIR,
            'XLRE': ETF_CSV_DIR, 'XLB': ETF_CSV_DIR,
            # ETF_CATEGORY_SECTOR_ETF maps fixed-income ETFs to TLT. Keep it
            # in the version-aware shared-source cache just like sector ETFs.
            'TLT': ETF_CSV_DIR,
        }
        indx_symbols = {
            'VIX': INDX_CSV_DIR, 'VIX3M': INDX_CSV_DIR,
            'US10Y': INDX_CSV_DIR, 'US2Y': INDX_CSV_DIR,
            'ADVN': INDX_CSV_DIR, 'DECN': INDX_CSV_DIR,
            'IRX': INDX_CSV_DIR,
            'DXY': INDX_CSV_DIR,
        }
        comm_symbols = {
            'CL': COMM_CSV_DIR,
            'GC': COMM_CSV_DIR,
        }

        for sym, directory in {**market_symbols, **indx_symbols}.items():
            path = os.path.join(directory, f'{sym}.csv')
            if os.path.exists(path):
                stat = os.stat(path)
                version = (
                    path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
                version_key = ('market', sym)
                if self._price_source_versions.get(version_key) != version:
                    self._price_cache[sym] = self._load_csv(path)
                    self._price_source_versions[version_key] = version

        # Commodity tickers are model context, not target-security aliases.
        # In particular, CL is both crude oil and a US equity ticker.
        for sym, directory in comm_symbols.items():
            path = os.path.join(directory, f'{sym}.csv')
            if os.path.exists(path):
                stat = os.stat(path)
                version = (
                    path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
                version_key = ('commodity', sym)
                if self._price_source_versions.get(version_key) != version:
                    self._commodity_price_cache[sym] = self._load_csv(path)
                    self._price_source_versions[version_key] = version

        for sym in symbols:
            if sym not in self._price_cache:
                path = self._find_csv_path(sym)
                if path:
                    self._price_cache[sym] = path  # lazy load

    def _get_price_df(self, symbol):
        """Get price DataFrame for symbol, loading lazily if needed."""
        if symbol not in self._price_cache:
            path = self._find_csv_path(symbol)
            if path is None:
                return None
            self._price_cache[symbol] = self._load_csv(path)
        elif isinstance(self._price_cache[symbol], str):
            self._price_cache[symbol] = self._load_csv(self._price_cache[symbol])
        return self._price_cache[symbol]

    def _get_commodity_price_df(self, symbol):
        """Return a commodity context series without entering equity namespace."""
        if symbol not in self._commodity_price_cache:
            path = os.path.join(COMM_CSV_DIR, f'{symbol}.csv')
            if not os.path.exists(path):
                return None
            self._commodity_price_cache[symbol] = self._load_csv(path)
        return self._commodity_price_cache[symbol]

    def _get_price_on_date(self, df, date, max_lookback=5):
        """Get the close price on or just before a date."""
        if df is None:
            return None
        if isinstance(date, str):
            date = pd.Timestamp(date)
        # Look back up to max_lookback days for a trading day
        for i in range(max_lookback + 1):
            d = date - pd.Timedelta(days=i)
            if d in df.index:
                return df.loc[d, 'close']
        return None

    def _get_slice(self, df, end_date, lookback_days):
        """Get a slice of price data ending on/before end_date."""
        if df is None or df.empty:
            return None
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        mask = df.index <= end_date
        subset = df[mask]
        if len(subset) < 2:
            return None
        start = end_date - pd.Timedelta(days=lookback_days * 2)  # extra buffer for trading days
        return subset[subset.index >= start].tail(lookback_days)

    # ------------------------------------------------------------------
    # Group 1: Pattern-Intrinsic Features (23 V1 + 8 V2)
    # ------------------------------------------------------------------

    def _load_opp_files(self, symbol, date_hint=None):
        """Load opportunity combo data for a symbol.

        Returns dict of combo -> indexed lookup dict.
        The lookup is keyed by (date_str, daysOut, LorS) -> row dict for O(1) access.

        Data source priority:
          1. ml_cache_YYYY-MM-DD.parquet (nightly cron, fastest)
          2. Targeted gzip scan -- streams each file, extracts only 5 target dates.
             Far lower memory than loading full history. Cached by (symbol, date_str).
        """
        date_str = date_hint[:10] if isinstance(date_hint, str) else (
            str(date_hint)[:10] if date_hint is not None else None)

        # Fast path: parquet
        if date_str is not None:
            combos = self._load_opp_from_parquet(symbol, date_str)
            if combos:
                return combos

        # Slow path: targeted gzip scan (5 dates only, line-by-line, no DataFrame)
        cache_key = (symbol, date_str)
        if cache_key in self._opp_cache:
            return self._opp_cache[cache_key]
        combos = self._load_opp_from_gzip(symbol, date_str)
        self._opp_cache[cache_key] = combos
        return combos

    def _load_opp_from_parquet(self, symbol, date_hint):
        """Load opp data from ml_cache parquet. Returns combos dict or None.

        Parquets are pre-split by symbol at load time (groupby once, O(1) per-symbol
        lookup thereafter). Resets when date changes since parquets are date-scoped.
        Markets are loaded lazily -- stops as soon as symbol is found.
        """
        if isinstance(date_hint, str):
            date_str = date_hint[:10]
        else:
            date_str = str(date_hint)[:10]

        # Reset on date change -- pre-split dicts are tied to a specific date's parquet
        if self._parquet_date != date_str:
            self._parquet_date = date_str
            self._parquet_by_symbol = {}
            self._parquet_loaded_markets = set()

        # O(1) hit: already split from a previously loaded market
        if symbol in self._parquet_by_symbol:
            return self._build_combos_from_df(self._parquet_by_symbol[symbol])

        # Load markets lazily until symbol is found or all exhausted
        for _rid, (_display, folder) in ML_PARQUET_MARKETS.items():
            if folder in self._parquet_loaded_markets:
                continue
            path = os.path.join(DATA_DIR, folder, f'ml_cache_{date_str}.parquet')
            if not os.path.exists(path):
                self._parquet_loaded_markets.add(folder)
                continue
            try:
                df = pd.read_parquet(path)
                # Pre-split entire parquet by symbol once -- O(n) now, O(1) per lookup
                by_sym = {sym: grp.reset_index(drop=True) for sym, grp in df.groupby('sym')}
                self._parquet_by_symbol.update(by_sym)
                self._parquet_loaded_markets.add(folder)
                if symbol in by_sym:
                    return self._build_combos_from_df(by_sym[symbol])
            except Exception as e:
                warnings.warn(f'ml_cache parquet read failed ({_display}): {e}')
                self._parquet_loaded_markets.add(folder)

        return None  # symbol not found in any market parquet

    def _build_combos_from_df(self, sym_df):
        """Build combo lookup dict from a symbol's DataFrame using vectorized array access.

        Returns dict of {combo -> {(date_str, daysOut, LorS) -> {sharpe_ratio, ...}}}
        Same structure as the gzip path. Uses numpy arrays instead of iterrows() for speed.
        """
        combos = {}
        c_vals = sym_df['combo'].values
        d_vals = [str(d)[:10] for d in sym_df['date'].values]
        do_vals = sym_df['daysOut'].values.astype(int)
        dr_vals = sym_df['LorS'].values
        sr_vals = sym_df['sharpe_ratio'].values
        ap_vals = sym_df['avg_profit'].values
        mp_vals = sym_df['median_profit'].values
        a2_vals = (sym_df['avg_profit2'].values if 'avg_profit2' in sym_df.columns
                   else ap_vals)
        for i in range(len(sym_df)):
            c = c_vals[i]
            if c not in combos:
                combos[c] = {}
            combos[c][(d_vals[i], int(do_vals[i]), dr_vals[i])] = {
                'sharpe_ratio': float(sr_vals[i]),
                'avg_profit': float(ap_vals[i]),
                'median_profit': float(mp_vals[i]),
                'avg_profit2': float(a2_vals[i]),
            }
        return combos

    def _load_opp_from_gzip(self, symbol, date_str=None):
        """Targeted gzip scan: stream each file, extract only 5 dates around target.

        Searches sp500 first, then ETF, then INDX markets (plain and _ variants).
        Never loads a full DataFrame -- reads line by line to minimise memory.
        """
        opp_dir = os.path.join(OPP_BY_SYMBOL_DIR, symbol)
        if not os.path.isdir(opp_dir):
            data_root = os.path.dirname(os.path.dirname(OPP_BY_SYMBOL_DIR))
            found = False
            for market_subdir in ['ETF', 'ETF_', 'INDX_COMMON', 'INDX_COMMON_']:
                alt = os.path.join(data_root, market_subdir, 'opp_by_symbol', symbol)
                if os.path.isdir(alt):
                    opp_dir = alt
                    found = True
                    break
            if not found:
                return {}

        # Build set of 5 target dates: target + neighbours at +-7, +-14 days
        if date_str is not None:
            target = pd.Timestamp(date_str)
            search_dates = {
                (target + timedelta(days=s)).strftime('%Y-%m-%d')
                for s in (-14, -7, 0, 7, 14)
            }
        else:
            search_dates = None  # load all dates (only used when date unknown)

        combos = {}
        for fname in os.listdir(opp_dir):
            if not fname.endswith('.csv.gz'):
                continue
            combo_name = fname.replace('.csv.gz', '')
            path = os.path.join(opp_dir, fname)
            try:
                with gzip.open(path, 'rt') as gz:
                    header = gz.readline().strip().split(',')
                    date_idx = header.index('date')
                    days_idx = header.index('daysOut')
                    dir_idx = header.index('LorS')
                    sr_idx = header.index('sharpe_ratio')
                    ap_idx = header.index('avg_profit')
                    mp_idx = header.index('median_profit')
                    ap2_idx = header.index('avg_profit2') if 'avg_profit2' in header else ap_idx
                    lookup = {}
                    for line in gz:
                        fields = line.strip().split(',')
                        d = fields[date_idx][:10]
                        if search_dates is not None and d not in search_dates:
                            continue
                        lookup[(d, int(fields[days_idx]), fields[dir_idx])] = {
                            'sharpe_ratio': float(fields[sr_idx]),
                            'avg_profit': float(fields[ap_idx]),
                            'median_profit': float(fields[mp_idx]),
                            'avg_profit2': float(fields[ap2_idx]),
                        }
                if lookup:
                    combos[combo_name] = lookup
            except Exception as e:
                warnings.warn(f'Skipping combo file {symbol}/{fname}: {e}')
                continue
        return combos

    def _parse_combo(self, combo_name):
        """Parse combo name like '10_8' or '10_8_PE2' into (year1, year2, is_pe)."""
        parts = combo_name.split('_')
        is_pe = combo_name.endswith('_PE2')
        year1 = int(parts[0])
        year2 = int(parts[1])
        return year1, year2, is_pe

    # ------------------------------------------------------------------
    # Exact-horizon V3 pattern profile for POST /score/context
    # ------------------------------------------------------------------

    def _context_paths(self, resource_id, symbol):
        """Resolve target price and opportunity paths in the requested universe."""
        resource_id = str(resource_id)
        market = ML_PARQUET_MARKETS.get(resource_id)
        if market is None:
            raise PatternProfileUnavailable(
                'unsupported_resource', {'resource_id': resource_id})

        target_dir = ETF_CSV_DIR if resource_id == '11' else US_CSV_DIR
        price_path = os.path.join(target_dir, f'{symbol}.csv')
        if not os.path.isfile(price_path):
            raise PatternProfileUnavailable(
                'target_price_unavailable',
                {'resource_id': resource_id, 'symbol': symbol},
            )

        _display, folder = market
        opp_dir = os.path.join(DATA_DIR, folder, 'opp_by_symbol', symbol)
        if not os.path.isdir(opp_dir):
            raise PatternProfileUnavailable(
                'pattern_definitions_unavailable',
                {'resource_id': resource_id, 'symbol': symbol},
            )
        return price_path, opp_dir

    def _prepare_context_target(self, resource_id, symbol):
        """Load a resource-qualified target without ticker namespace ambiguity."""
        price_path, opp_dir = self._context_paths(resource_id, symbol)
        stat = os.stat(price_path)
        file_version = (
            price_path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        target_key = (str(resource_id), symbol)

        # Load regime inputs first, then explicitly install the target in the
        # equity/ETF namespace.  Commodity CL is held in its own cache.
        self.load_price_data([])
        cached = self._lru_get(self._context_target_cache, target_key)
        if cached is None or cached[3] != file_version:
            frame = self._load_csv(price_path)
            cached = (frame, price_path, opp_dir, file_version)
            evicted = self._lru_put(
                self._context_target_cache,
                target_key,
                cached,
                CONTEXT_TARGET_CACHE_MAX,
            )
            if evicted is not None:
                evicted_key, evicted_value = evicted
                evicted_symbol = evicted_key[1]
                still_cached = any(
                    key[1] == evicted_symbol for key in self._context_target_cache
                )
                if (
                    not still_cached
                    and evicted_symbol not in _PINNED_PRICE_SYMBOLS
                    and self._price_cache.get(evicted_symbol) is evicted_value[0]
                ):
                    self._price_cache.pop(evicted_symbol, None)

        # Downstream V3 feature groups use the legacy symbol lookup. Install the
        # current resource-qualified frame for only this calculation; LRU
        # eviction above prevents target frames from accumulating forever.
        self._price_cache[symbol] = cached[0]
        return cached

    def _combo_definitions(self, opp_dir):
        """Return real per-symbol combo definitions in on-disk generator order."""
        definitions = []
        for fname in os.listdir(opp_dir):
            if not fname.endswith('.csv.gz'):
                continue
            combo_name = fname[:-7]
            try:
                year1, year2, is_pe = self._parse_combo(combo_name)
            except (ValueError, IndexError):
                continue
            if year1 <= 0 or year2 <= 0 or year2 > year1:
                continue
            path = os.path.join(opp_dir, fname)
            stat = os.stat(path)
            definitions.append({
                'name': combo_name,
                'year1': year1,
                'year2': year2,
                'is_pe': is_pe,
                'path': path,
                'mtime_ns': stat.st_mtime_ns,
                'ctime_ns': stat.st_ctime_ns,
                'bytes': stat.st_size,
            })
        # Preserve the same combo insertion order used by the V3 training
        # builder and the legacy /score path. Rounded-Sharpe ties select the
        # first row, so changing this order would change a trained feature.
        return definitions

    @staticmethod
    def _definitions_version(definitions):
        material = '|'.join(
            f"{item['name']}:{item['mtime_ns']}:{item['ctime_ns']}:{item['bytes']}"
            for item in definitions
        )
        return hashlib.sha256(material.encode('utf-8')).hexdigest()

    def _context_opp_snapshot(self, opp_dir, date_str, definitions,
                              active_date_strs=None):
        """Read all prebuilt rows for one exact date once.

        Besides exact-horizon validation, the complete active-pair set is used
        for the V3 concurrent-pattern feature.  This snapshot is resource-aware;
        unlike the legacy scorer it never silently falls back to another market.
        """
        active_date_strs = frozenset(active_date_strs or (date_str,))
        active_date_strs = frozenset((*active_date_strs, date_str))
        key = (
            opp_dir,
            date_str,
            tuple(sorted(active_date_strs)),
            self._definitions_version(definitions),
        )
        cached = self._lru_get(self._context_opp_snapshot_cache, key)
        if cached is not None:
            return cached

        rows = {}
        active_pairs_by_date = {
            active_date: set() for active_date in active_date_strs
        }
        for definition in definitions:
            combo_name = definition['name']
            combo_rows = {}
            try:
                with gzip.open(definition['path'], 'rt') as gz:
                    header = gz.readline().strip().split(',')
                    date_idx = header.index('date')
                    days_idx = header.index('daysOut')
                    dir_idx = header.index('LorS')
                    sr_idx = header.index('sharpe_ratio')
                    ap_idx = header.index('avg_profit')
                    mp_idx = header.index('median_profit')
                    ap2_idx = header.index('avg_profit2') if 'avg_profit2' in header else ap_idx
                    for line in gz:
                        fields = line.rstrip().split(',')
                        if len(fields) <= max(date_idx, days_idx, dir_idx, sr_idx, ap_idx, mp_idx, ap2_idx):
                            continue
                        row_date = fields[date_idx][:10]
                        if row_date not in active_date_strs:
                            continue
                        pair = (int(fields[days_idx]), fields[dir_idx])
                        active_pairs_by_date[row_date].add(pair)
                        if row_date == date_str:
                            combo_rows[pair] = {
                                'sharpe_ratio': float(fields[sr_idx]),
                                'avg_profit': float(fields[ap_idx]),
                                'median_profit': float(fields[mp_idx]),
                                'avg_profit2': float(fields[ap2_idx]),
                            }
            except Exception as exc:
                warnings.warn(
                    f'Skipping context combo file {definition["path"]}: {exc}')
                continue
            if combo_rows:
                rows[combo_name] = combo_rows

        snapshot = {
            'rows': rows,
            'active_pairs': active_pairs_by_date[date_str],
            'active_pairs_by_date': active_pairs_by_date,
        }
        self._lru_put(
            self._context_opp_snapshot_cache,
            key,
            snapshot,
            CONTEXT_SNAPSHOT_CACHE_MAX,
        )
        return snapshot

    @staticmethod
    def _context_entry_date(price_index, nominal_date):
        """Return the 0..3-day forward-snapped entry used by V3 training.

        Future opportunity dates can be newer than the latest EOD price row. In
        that case only the deterministic weekend part of the snap is knowable,
        so use the next weekday instead of rejecting an otherwise scoreable row.
        Once EOD data arrives, the price-index branch becomes authoritative and
        the scorer process/data generation changes before caches are warmed.
        """
        nominal_date = pd.Timestamp(nominal_date)
        last_price_date = price_index[-1] if len(price_index) else None
        for offset in range(0, 4):
            candidate = nominal_date + pd.Timedelta(days=offset)
            if candidate in price_index:
                return candidate
            if last_price_date is not None and candidate > last_price_date:
                if candidate.weekday() < 5:
                    return candidate
        return None

    def _context_concurrent_dates(self, price_index, nominal_date):
        """Nominal dates that V3 training maps to the same actual entry."""
        actual_entry = self._context_entry_date(price_index, nominal_date)
        if actual_entry is None:
            return None, ()
        nominal_dates = []
        for offset in range(0, 4):
            candidate = actual_entry - pd.Timedelta(days=offset)
            if self._context_entry_date(price_index, candidate) == actual_entry:
                nominal_dates.append(candidate.strftime('%Y-%m-%d'))
        return actual_entry, tuple(sorted(nominal_dates))

    @staticmethod
    def _context_concurrent_count(active_pairs_by_date, days_out):
        """Count same-entry patterns within the model's training tier."""
        days_out = int(days_out)
        if days_out <= 30:
            tier_min, tier_max = 10, 30
        elif days_out <= 60:
            tier_min, tier_max = 31, 60
        else:
            tier_min, tier_max = 61, 90
        return float(sum(
            1
            for pairs in active_pairs_by_date.values()
            for pair_days_out, _direction in pairs
            if tier_min <= int(pair_days_out) <= tier_max
        ))

    def _compute_historical_observation(self, price_df, month, day, year,
                                        days_out, dir_char):
        """Recalculate one completed historical observation.

        Ground truth: TradeWave windows use CALENDAR days.  ``days_out`` is the
        legacy raw offset, so the nominal end is exactly ``start + days_out``.
        No leap-day increment is ever added.  Entry and end independently snap
        forward to available price rows using the generator's 0-3/0-4 limits.
        """
        try:
            nominal_start = pd.Timestamp(year=year, month=month, day=day)
        except (ValueError, TypeError):
            return None

        index = price_df.index
        entry = None
        for offset in range(0, 4):
            candidate = nominal_start + pd.Timedelta(days=offset)
            if candidate in index:
                entry = candidate
                break
        if entry is None:
            return None

        nominal_end = nominal_start + pd.Timedelta(days=int(days_out))
        if nominal_end > index[-1]:
            return None

        exit_date = None
        for offset in range(0, 5):
            candidate = nominal_end + pd.Timedelta(days=offset)
            if candidate in index:
                exit_date = candidate
                break
        if exit_date is None:
            return None

        try:
            entry_price = float(price_df.loc[entry, 'close'])
            exit_price = float(price_df.loc[exit_date, 'close'])
        except (KeyError, TypeError, ValueError):
            return None
        if (not math.isfinite(entry_price) or not math.isfinite(exit_price)
                or entry_price == 0):
            return None

        window = price_df.loc[
            (price_df.index > entry) & (price_df.index <= exit_date)]
        if window.empty:
            return None

        close_return = (exit_price - entry_price) / entry_price * 100.0
        if dir_char == 's':
            close_return = -close_return
            favorable_price = float(window['low'].min())
            favorable_return = (entry_price - favorable_price) / entry_price * 100.0
        else:
            favorable_price = float(window['high'].max())
            favorable_return = (favorable_price - entry_price) / entry_price * 100.0

        if not (math.isfinite(close_return) and math.isfinite(favorable_return)):
            return None
        # Opportunity files store annual percentages to two decimals before
        # aggregating.  Matching that ordering reproduces their rows exactly.
        return {
            'year': year,
            'return': round(close_return, 2),
            'favorable_return': round(favorable_return, 2),
            'nominal_start': nominal_start,
            'nominal_end': nominal_end,
            'entry_date': entry,
            'exit_date': exit_date,
        }

    @staticmethod
    def _selected_winning_years(partial):
        """Read TradeWave's selected recurrence threshold from request context.

        The context endpoint remains compatible with older callers that used
        ``partial`` only as opaque provenance.  A threshold is enforced only
        when one of the known statistical fields is present.
        """
        value = partial
        for _depth in range(3):
            if not isinstance(value, dict):
                break
            selected = None
            for field in ('min_winning_years', 'partialYears', 'year2'):
                if value.get(field) not in (None, ''):
                    selected = value[field]
                    break
            if selected is not None:
                value = selected
                break
            if 'selection' not in value:
                return None
            value = value.get('selection')
        if value is None or isinstance(value, bool) or value == '':
            return None
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise PatternProfileUnavailable(
                'invalid_selected_recurrence',
                {'field': 'min_winning_years'},
            ) from exc
        if not math.isfinite(number) or not number.is_integer() or number <= 0:
            raise PatternProfileUnavailable(
                'invalid_selected_recurrence',
                {'field': 'min_winning_years'},
            )
        return int(number)

    @staticmethod
    def _selected_year_spec(years):
        """Return (mode, PE phase, requested observations) for a years string."""
        value = str(years or '').strip().lower()
        pe_phase = None
        if value.startswith('pe') and '-' in value:
            phase_text, count_text = value[2:].split('-', 1)
            try:
                phase_token = int(phase_text)
                requested = int(count_text)
            except ValueError as exc:
                raise PatternProfileUnavailable(
                    'invalid_selected_recurrence', {'field': 'years'}
                ) from exc
            # TradeWave's public PE labels use 0..3 while the scorer's
            # get_pe_year helper uses 1..4.  In both systems ``pe2`` is the
            # midterm phase; pe0 maps to the scorer's phase 4.
            if phase_token not in (0, 1, 2, 3, 4):
                raise PatternProfileUnavailable(
                    'invalid_selected_recurrence', {'field': 'years'}
                )
            pe_phase = 4 if phase_token == 0 else phase_token
            mode = 'pe'
        else:
            try:
                requested = int(value)
            except ValueError as exc:
                raise PatternProfileUnavailable(
                    'invalid_selected_recurrence', {'field': 'years'}
                ) from exc
            mode = 'consecutive'
        if requested <= 0 or requested > 100:
            raise PatternProfileUnavailable(
                'invalid_selected_recurrence', {'field': 'years'}
            )
        return mode, pe_phase, requested

    def compute_selected_recurrence_summary(self, resource_id, symbol, date,
                                            days_out, direction, years, partial):
        """Recalculate the caller-selected historical cohort at one horizon.

        This summary is explanatory screen evidence, not a new V3 feature
        definition or an inference gate. V3 consumes its trained all-combo
        profile whenever that profile can be validated, whether or not the
        selected cohort still meets its original recurrence rule.
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)
        dir_char = direction[0].lower()
        mode, pe_phase, requested = self._selected_year_spec(years)
        required = self._selected_winning_years(partial)
        if required is not None and required > requested:
            raise PatternProfileUnavailable(
                'invalid_selected_recurrence',
                {
                    'required_positive_years': required,
                    'requested_observations': requested,
                },
            )

        price_df, price_path, _opp_dir, file_version = self._prepare_context_target(
            resource_id, symbol)
        cache_key = (
            str(resource_id), symbol, str(date)[:10], int(days_out), dir_char,
            str(years).strip().lower(), required, file_version,
        )
        cached = self._lru_get(self._selected_recurrence_cache, cache_key)
        if cached is not None:
            return cached

        observations = []
        first_year = int(price_df.index[0].year)
        for year in range(first_year, int(date.year)):
            if mode == 'pe' and get_pe_year(year) != pe_phase:
                continue
            observation = self._compute_historical_observation(
                price_df, date.month, date.day, year, int(days_out), dir_char)
            if observation is not None:
                observations.append(observation)

        selected = observations[-requested:]
        sample_size = len(selected)
        positive_years = sum(row['return'] >= 0.0 for row in selected)
        complete = sample_size == requested
        if required is None:
            status = 'not_enforced'
        elif not complete:
            status = 'insufficient_history'
        elif positive_years < required:
            status = 'below_threshold'
        else:
            status = 'qualified'

        returns = np.asarray([row['return'] for row in selected], dtype=float)
        favorable = np.asarray(
            [row['favorable_return'] for row in selected], dtype=float)
        summary = {
            'status': status,
            'mode': mode,
            'years': str(years).strip().lower(),
            'requested_observations': requested,
            'sample_size': sample_size,
            'positive_years': int(positive_years),
            'required_positive_years': required,
            'win_rate': (
                round(float(positive_years) / sample_size, 6)
                if sample_size else None
            ),
            'average_return_pct': (
                round(float(np.mean(returns)), 4) if sample_size else None
            ),
            'median_return_pct': (
                round(float(np.median(returns)), 4) if sample_size else None
            ),
            'average_favorable_excursion_pct': (
                round(float(np.mean(favorable)), 4) if sample_size else None
            ),
            'complete': complete,
            'data_as_of': str(price_df.index[-1])[:10],
            'price_source': os.path.basename(price_path),
        }
        if pe_phase is not None:
            summary['pe_phase'] = pe_phase
        self._lru_put(
            self._selected_recurrence_cache,
            cache_key,
            summary,
            CONTEXT_PROFILE_CACHE_MAX,
        )
        return summary

    def _combo_row_from_observations(self, observations, year1, year2, days_out):
        """Build one opportunity row if its real combo threshold qualifies."""
        if len(observations) < year1:
            return None
        selected = observations[-year1:]
        returns = np.asarray([row['return'] for row in selected], dtype=float)
        favorable = np.asarray(
            [row['favorable_return'] for row in selected], dtype=float)
        if len(returns) != year1 or not np.isfinite(returns).all() or not np.isfinite(favorable).all():
            return None
        if int(np.sum(returns >= 0.0)) < year2:
            return None

        avg_profit = float(np.mean(returns))
        median_profit = float(np.median(returns))
        avg_profit2 = float(np.mean(favorable))
        profit_std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        profit_std2 = float(np.std(favorable, ddof=1)) if len(favorable) > 1 else 0.0
        hurdle = PATTERN_RISK_FREE_RETURN * (float(days_out) / 365.0)
        sharpe = (avg_profit - hurdle) / profit_std if profit_std > 0 else 0.0
        sharpe2 = (avg_profit2 - hurdle) / profit_std2 if profit_std2 > 0 else 0.0
        return {
            'sharpe_ratio': round(sharpe, 2),
            'avg_profit': round(avg_profit, 2),
            'median_profit': round(median_profit, 2),
            'avg_profit2': round(avg_profit2, 2),
            'sharpe_ratio2': round(sharpe2, 2),
        }

    def _aggregate_pattern_profile(self, qualifying_rows, definitions,
                                   data_years, dir_char, days_out):
        """Aggregate qualifying combo rows exactly as V3 training did."""
        best_sharpe = None
        best_row = None
        best_combo = None
        deepest_pass = 0
        pe_deepest = 0
        num_combos_qualifying = 0
        passes_recent_10 = 0
        sharpe_at_10 = None
        sharpe_at_deepest = None
        best_winrate = 0.0
        worst_winrate = 1.0
        deepest_pass_capped30 = 0
        non_pe_winrates = []

        for definition in definitions:
            combo_name = definition['name']
            row = qualifying_rows.get(combo_name)
            if row is None:
                continue
            year1 = definition['year1']
            year2 = definition['year2']
            is_pe = definition['is_pe']
            winrate = year2 / year1
            num_combos_qualifying += 1

            if is_pe:
                pe_deepest = max(pe_deepest, year1)
            else:
                non_pe_winrates.append(winrate)
                if year1 > deepest_pass:
                    deepest_pass = year1
                    sharpe_at_deepest = row['sharpe_ratio']
                if year1 == 10 and year2 == 10:
                    passes_recent_10 = 1
                    sharpe_at_10 = row['sharpe_ratio']
                deepest_pass_capped30 = max(deepest_pass_capped30, min(year1, 30))

            best_winrate = max(best_winrate, winrate)
            worst_winrate = min(worst_winrate, winrate)
            if best_sharpe is None or row['sharpe_ratio'] > best_sharpe:
                best_sharpe = row['sharpe_ratio']
                best_row = row
                best_combo = combo_name

        if best_row is None:
            return None, None

        max_possible_depth = max(
            (d['year1'] for d in definitions if not d['is_pe']), default=0)
        depth_denom = min(data_years, MAX_DEPTH_CAP)
        pe_denom = depth_denom / 4.0

        # The nullable ratio mirrors V3 training: absence of a perfect 10/10
        # pass is represented as missing, which all three model families learned.
        recent_vs_deep = np.nan
        if sharpe_at_10 and sharpe_at_deepest:
            recent_vs_deep = sharpe_at_10 / sharpe_at_deepest

        avg_profit = best_row['avg_profit']
        avg_profit2 = best_row['avg_profit2']
        if days_out <= 15:
            days_bucket = 0
        elif days_out <= 45:
            days_bucket = 1
        elif days_out <= 90:
            days_bucket = 2
        else:
            days_bucket = 3

        profile = {
            'pat_sharpe_ratio': best_row['sharpe_ratio'],
            'pat_avg_profit': avg_profit,
            'pat_median_profit': best_row['median_profit'],
            'pat_avg_profit2': avg_profit2,
            'pat_mfe_ratio': avg_profit2 / avg_profit if avg_profit != 0 else 1.0,
            'pat_profit_per_day': avg_profit / days_out if days_out > 0 else 0.0,
            'pat_sharpe_per_day': (
                best_row['sharpe_ratio'] / math.sqrt(days_out) if days_out > 0 else 0.0),
            'pat_daysOut': int(days_out),
            'pat_daysOut_bucket': days_bucket,
            'pat_direction': 1 if dir_char == 'l' else 0,
            'pat_data_years': data_years,
            'pat_deepest_pass': deepest_pass,
            'pat_depth_utilization': deepest_pass / depth_denom if depth_denom > 0 else 0,
            'pat_passes_at_max_depth': 1 if deepest_pass >= max_possible_depth else 0,
            'pat_passes_recent_10': passes_recent_10,
            'pat_recent_vs_deep_sharpe': recent_vs_deep,
            'pat_num_combos_qualifying': num_combos_qualifying,
            'pat_pe_match': 1 if pe_deepest > 0 else 0,
            'pat_pe_deepest': pe_deepest,
            'pat_pe_utilization': pe_deepest / pe_denom if pe_denom > 0 else 0,
            'pat_best_winrate': best_winrate,
            'pat_worst_winrate': worst_winrate if worst_winrate < 1.0 else best_winrate,
            'pat_deepest_pass_capped30': deepest_pass_capped30,
            'pat_consistency_std': (
                float(np.std(non_pe_winrates)) if len(non_pe_winrates) > 1 else 0.0),
        }
        return profile, {
            'best_combo': best_combo,
            'max_possible_depth': max_possible_depth,
        }

    @staticmethod
    def _profile_value_differences(left, right, keys=None, tolerance=1e-10,
                                   tolerances=None):
        keys = tuple(keys) if keys is not None else tuple(left)
        if any(key not in left or key not in right for key in keys):
            return [
                key for key in keys if key not in left or key not in right
            ]
        differences = []
        tolerance_by_key = tolerances or {}
        for key in keys:
            a = left[key]
            b = right[key]
            try:
                if np.isnan(a) and np.isnan(b):
                    continue
            except TypeError:
                pass
            if not (math.isfinite(float(a)) and math.isfinite(float(b))):
                differences.append(key)
                continue
            absolute_tolerance, relative_tolerance = tolerance_by_key.get(
                key, (tolerance, 0.0))
            allowed = max(
                float(absolute_tolerance),
                float(relative_tolerance) * max(abs(float(a)), abs(float(b))),
            )
            if abs(float(a) - float(b)) > allowed:
                differences.append(key)
        return differences

    @classmethod
    def _profile_values_equal(cls, left, right, keys=None, tolerance=1e-10,
                              tolerances=None):
        return not cls._profile_value_differences(
            left,
            right,
            keys=keys,
            tolerance=tolerance,
            tolerances=tolerances,
        )

    @staticmethod
    def _profile_hash(profile):
        normalized = {
            key: (None if isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
                  else float(value) if isinstance(value, (np.floating, np.integer)) else value)
            for key, value in profile.items()
        }
        payload = json.dumps(
            normalized, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def compute_recalculated_pattern_profile(self, resource_id, symbol, date,
                                             days_out, direction):
        """Build the model-faithful all-combo V3 profile at an exact horizon."""
        if isinstance(date, str):
            date = pd.Timestamp(date)
        dir_char = direction[0].lower()
        price_df, price_path, opp_dir, file_version = self._prepare_context_target(
            resource_id, symbol)
        definitions = self._combo_definitions(opp_dir)
        if not definitions:
            raise PatternProfileUnavailable('pattern_definitions_unavailable')

        definitions_version = self._definitions_version(definitions)
        cache_key = (
            str(resource_id), symbol, str(date)[:10], int(days_out), dir_char,
            file_version, definitions_version,
        )
        cached = self._lru_get(self._recalculated_profile_cache, cache_key)
        if cached is not None:
            return cached

        max_regular_depth = max(
            (d['year1'] for d in definitions if not d['is_pe']), default=0)
        max_pe_depth = max(
            (d['year1'] for d in definitions if d['is_pe']), default=0)
        earliest_year = int(price_df.index[0].year)
        # Extra calendar buffer lets sparse early data fall away while retaining
        # enough completed observations for the deepest actual definition.
        lookback_years = max(max_regular_depth + 8, max_pe_depth * 4 + 8)
        first_year = max(earliest_year, int(date.year) - lookback_years)

        regular_observations = []
        pe2_observations = []
        for year in range(first_year, int(date.year)):
            observation = self._compute_historical_observation(
                price_df, date.month, date.day, year, int(days_out), dir_char)
            if observation is None:
                continue
            regular_observations.append(observation)
            if get_pe_year(year) == 2:
                pe2_observations.append(observation)

        dynamic_rows = {}
        for definition in definitions:
            observations = pe2_observations if definition['is_pe'] else regular_observations
            row = self._combo_row_from_observations(
                observations, definition['year1'], definition['year2'], int(days_out))
            if row is not None:
                dynamic_rows[definition['name']] = row

        data_years = len(price_df) / 252.0
        dynamic_profile, aggregate_meta = self._aggregate_pattern_profile(
            dynamic_rows, definitions, data_years, dir_char, int(days_out))
        if dynamic_profile is None:
            raise PatternProfileUnavailable(
                'pattern_profile_unavailable',
                {
                    'completed_observations': len(regular_observations),
                    'completed_pe2_observations': len(pe2_observations),
                    'combo_definitions': len(definitions),
                },
            )

        # Reject genuinely incomplete/non-finite core data.  One field is
        # intentionally nullable because that exact missingness was present in
        # V3 training when a pattern did not pass the perfect 10/10 combo.
        nullable = {'pat_recent_vs_deep_sharpe'}
        for key, value in dynamic_profile.items():
            if key in nullable and not math.isfinite(float(value)):
                continue
            if not math.isfinite(float(value)):
                raise PatternProfileUnavailable(
                    'nonfinite_pattern_profile', {'feature': key})

        date_str = str(date)[:10]
        actual_entry, concurrent_dates = self._context_concurrent_dates(
            price_df.index, date)
        if actual_entry is None:
            raise PatternProfileUnavailable(
                'target_entry_unavailable', {'date': date_str})
        snapshot = self._context_opp_snapshot(
            opp_dir,
            date_str,
            definitions,
            active_date_strs=concurrent_dates,
        )
        prebuilt_rows = {
            name: rows[(int(days_out), dir_char)]
            for name, rows in snapshot['rows'].items()
            if (int(days_out), dir_char) in rows
        }
        prebuilt_profile, prebuilt_aggregate_meta = self._aggregate_pattern_profile(
            prebuilt_rows, definitions, data_years, dir_char, int(days_out))
        qualifying_sets_match = set(prebuilt_rows) == set(dynamic_rows)
        strict_differences = (
            self._profile_value_differences(
                prebuilt_profile,
                dynamic_profile,
                keys=_V3_AGGREGATE_PROFILE_KEYS,
            )
            if prebuilt_profile is not None else list(_V3_AGGREGATE_PROFILE_KEYS)
        )
        model_differences = (
            self._profile_value_differences(
                prebuilt_profile,
                dynamic_profile,
                keys=_V3_AGGREGATE_PROFILE_KEYS,
                tolerances=_V3_PROFILE_VALIDATION_TOLERANCES,
            )
            if prebuilt_profile is not None else list(_V3_AGGREGATE_PROFILE_KEYS)
        )
        values_match = not model_differences
        if not qualifying_sets_match or prebuilt_profile is None or not values_match:
            raise PatternProfileUnavailable(
                'prebuilt_profile_mismatch',
                {
                    'dynamic_combo_count': len(dynamic_rows),
                    'prebuilt_combo_count': len(prebuilt_rows),
                    'qualifying_sets_match': qualifying_sets_match,
                    'model_values_match': values_match,
                    'mismatched_model_fields': model_differences,
                },
            )
        # The dynamic rebuild proves the checkpoint's qualifying combo set. The
        # on-disk row values are then used for exact V3 training parity, including
        # rounded-Sharpe tie behavior and generator rounding.
        served_profile = prebuilt_profile
        prebuilt_validated = True
        source = 'dynamic_recalculation_validated'

        for key, value in served_profile.items():
            if key in nullable and not math.isfinite(float(value)):
                continue
            if not math.isfinite(float(value)):
                raise PatternProfileUnavailable(
                    'nonfinite_pattern_profile', {'feature': key})

        result = (
            served_profile,
            {
                'source': source,
                'prebuilt_validated': prebuilt_validated,
                'profile_validation': (
                    'bounded_authoritative' if strict_differences else 'exact'
                ),
                'reconciled_model_fields': strict_differences,
                'qualifying_combo_count': len(dynamic_rows),
                'prebuilt_combo_count': len(prebuilt_rows),
                'profile_hash': self._profile_hash(served_profile),
                'dynamic_profile_hash': self._profile_hash(dynamic_profile),
                'data_as_of': str(price_df.index[-1])[:10],
                'price_path': price_path,
                'best_combo': prebuilt_aggregate_meta['best_combo'],
                # Internal-only support for pat_concurrent_count; app.py omits
                # these raw pairs from the public response.
                '_active_pairs': snapshot['active_pairs'],
                '_active_pairs_by_date': snapshot.get(
                    'active_pairs_by_date', {date_str: snapshot['active_pairs']}),
                'actual_entry_date': actual_entry.strftime('%Y-%m-%d'),
                'concurrent_nominal_dates': list(concurrent_dates),
                'nullable_features': sorted(
                    key for key in nullable
                    if not math.isfinite(float(served_profile[key]))),
            },
        )
        self._lru_put(
            self._recalculated_profile_cache,
            cache_key,
            result,
            CONTEXT_PROFILE_CACHE_MAX,
        )
        return result

    def compute_pattern_features(self, symbol, date, daysOut, direction):
        """
        Group 1: Pattern-Intrinsic features (23 V1 + 8 V2).
        Scans all year combos to build depth profile for this unique opportunity.
        """
        features = {}
        if isinstance(date, str):
            date = pd.Timestamp(date)
        combos = self._load_opp_files(symbol, date_hint=str(date)[:10])

        dir_char = direction[0].lower()  # 'l' or 's'
        date_str = str(date)[:10]
        lookup_key = (date_str, int(daysOut), dir_char)

        # Find this pattern across all combos using O(1) lookups
        best_sharpe = None
        best_combo = None
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

        # V2: collect non-PE win rates for pat_consistency_std
        non_pe_winrates = []

        for combo_name, lookup in combos.items():
            row = lookup.get(lookup_key)
            if row is None:
                continue

            year1, year2, is_pe = self._parse_combo(combo_name)
            winrate = year2 / year1 if year1 > 0 else 0
            num_combos_qualifying += 1

            if is_pe:
                if year1 > pe_deepest:
                    pe_deepest = year1
            else:
                non_pe_winrates.append(winrate)
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

            # Track best qualifying combo (highest sharpe at deepest)
            if best_sharpe is None or row['sharpe_ratio'] > best_sharpe:
                best_sharpe = row['sharpe_ratio']
                best_combo = combo_name
                best_row = row

        # V1 NaN-fill keys
        v1_nan_keys = [
            'sharpe_ratio', 'avg_profit', 'median_profit', 'avg_profit2',
            'mfe_ratio', 'profit_per_day', 'sharpe_per_day',
            'daysOut', 'daysOut_bucket', 'direction',
            'data_years', 'deepest_pass', 'depth_utilization',
            'passes_at_max_depth', 'passes_recent_10', 'recent_vs_deep_sharpe',
            'num_combos_qualifying',
            'pe_match', 'pe_deepest', 'pe_utilization',
            'best_winrate', 'worst_winrate', 'deepest_pass_capped30',
        ]
        # V2 NaN-fill keys
        v2_nan_keys = [
            'consistency_std', 'concurrent_count',
            'neighbor_avg_wr', 'sharpness', 'pre_slope', 'post_cliff',
            'hit_last_year',
        ]

        if best_row is None:
            # Pattern not found in any combo
            nan_dict = {f'pat_{k}': np.nan for k in v1_nan_keys + v2_nan_keys}
            return nan_dict

        # Get data_years from price CSV
        price_df = self._get_price_df(symbol)
        if price_df is not None and len(price_df) > 0:
            data_years = len(price_df) / 252.0
        else:
            data_years = np.nan

        # Find max possible depth for this stock
        max_depth_files = [c for c in combos.keys() if not c.endswith('_PE2')]
        max_possible_depth = 0
        for c in max_depth_files:
            y1, _, _ = self._parse_combo(c)
            if y1 > max_possible_depth:
                max_possible_depth = y1

        avg_profit = best_row['avg_profit']
        avg_profit2 = best_row.get('avg_profit2', avg_profit)

        features['pat_sharpe_ratio'] = best_row['sharpe_ratio']
        features['pat_avg_profit'] = avg_profit
        features['pat_median_profit'] = best_row['median_profit']
        features['pat_avg_profit2'] = avg_profit2
        features['pat_mfe_ratio'] = avg_profit2 / avg_profit if avg_profit != 0 else 1.0
        features['pat_profit_per_day'] = avg_profit / daysOut if daysOut > 0 else 0
        features['pat_sharpe_per_day'] = best_row['sharpe_ratio'] / math.sqrt(daysOut) if daysOut > 0 else 0
        features['pat_daysOut'] = daysOut

        # daysOut bucket: 0=swing(5-15), 1=short(16-45), 2=medium(46-90), 3=long(91+)
        if daysOut <= 15:
            bucket = 0
        elif daysOut <= 45:
            bucket = 1
        elif daysOut <= 90:
            bucket = 2
        else:
            bucket = 3
        features['pat_daysOut_bucket'] = bucket
        features['pat_direction'] = 1 if dir_char == 'l' else 0

        features['pat_data_years'] = data_years
        features['pat_deepest_pass'] = deepest_pass
        depth_denom = min(data_years, MAX_DEPTH_CAP) if not np.isnan(data_years) else MAX_DEPTH_CAP
        features['pat_depth_utilization'] = deepest_pass / depth_denom if depth_denom > 0 else 0
        features['pat_passes_at_max_depth'] = 1 if deepest_pass >= max_possible_depth else 0
        features['pat_passes_recent_10'] = passes_recent_10

        if sharpe_at_10 is not None and sharpe_at_deepest is not None and sharpe_at_deepest != 0:
            features['pat_recent_vs_deep_sharpe'] = sharpe_at_10 / sharpe_at_deepest
        else:
            features['pat_recent_vs_deep_sharpe'] = np.nan

        features['pat_num_combos_qualifying'] = num_combos_qualifying
        features['pat_pe_match'] = 1 if pe_deepest > 0 else 0
        features['pat_pe_deepest'] = pe_deepest
        pe_denom = min(data_years, MAX_DEPTH_CAP) / 4.0 if not np.isnan(data_years) else MAX_DEPTH_CAP / 4.0
        features['pat_pe_utilization'] = pe_deepest / pe_denom if pe_denom > 0 else 0
        features['pat_best_winrate'] = best_winrate
        features['pat_worst_winrate'] = worst_winrate if worst_winrate < 1.0 else best_winrate
        features['pat_deepest_pass_capped30'] = deepest_pass_capped30

        # ------ V2 Pattern Features ------

        # pat_consistency_std: std deviation of win rates across non-PE depth levels
        if len(non_pe_winrates) >= 2:
            features['pat_consistency_std'] = float(np.std(non_pe_winrates))
        else:
            features['pat_consistency_std'] = 0.0

        # pat_concurrent_count: total unique (daysOut, direction) patterns active for
        # this symbol on this date, including self. Scan ALL combos for correctness.
        # Matches training: date_pattern_counter counts all qualifying patterns on entry_date.
        active_pairs = set()
        for _combo_lookup in combos.values():
            for _key in _combo_lookup:
                if _key[0] == date_str:
                    active_pairs.add((_key[1], _key[2]))  # (daysOut, direction)
        features['pat_concurrent_count'] = float(len(active_pairs))

        # Neighborhood features: use realized price history across prior years,
        # matching build_training_data.py:compute_neighborhood_features exactly.
        nbr = self._compute_neighborhood_features(symbol, date, daysOut, dir_char, price_df)
        features['pat_neighbor_avg_wr'] = nbr['pat_neighbor_avg_wr']
        features['pat_sharpness'] = nbr['pat_sharpness']
        features['pat_pre_slope'] = nbr['pat_pre_slope']
        features['pat_post_cliff'] = nbr['pat_post_cliff']

        # pat_hit_last_year: did this pattern work last year?
        features['pat_hit_last_year'] = self._compute_hit_last_year(
            symbol, date, daysOut, dir_char, price_df)

        return features

    def _compute_hit_last_year(self, symbol, date, daysOut, dir_char, price_df):
        """Check if the pattern worked last year. Return 1/0 or NaN.

        Matches build_training_data.py exactly:
        - Feb 29 in a non-leap prior year -> NaN (training skips via ValueError, not Feb 28)
        - Entry and exit both use FORWARD trading-day lookup (range 0-3 / 0-4 days)
        """
        if price_df is None or price_df.empty:
            return np.nan
        # Build prior-year base date. Feb 29 -> NaN to match training (ValueError -> continue).
        try:
            last_year_base = pd.Timestamp(f"{date.year - 1}-{date.month:02d}-{date.day:02d}")
        except ValueError:
            return np.nan

        trading_days_set = set(price_df.index)
        close_values = price_df['close']

        # Forward entry lookup (matches training range(0, 4))
        entry = None
        for offset in range(0, 4):
            candidate = last_year_base + pd.Timedelta(days=offset)
            if candidate in trading_days_set:
                entry = candidate
                break
        if entry is None:
            return np.nan

        # Forward exit lookup (matches training range(0, 5))
        exit_raw = entry + pd.Timedelta(days=daysOut)
        exit_date = None
        for offset in range(0, 5):
            candidate = exit_raw + pd.Timedelta(days=offset)
            if candidate in trading_days_set:
                exit_date = candidate
                break
        if exit_date is None:
            return np.nan

        try:
            p_in = close_values[entry]
            p_out = close_values[exit_date]
            if p_in == 0 or np.isnan(p_in) or np.isnan(p_out):
                return np.nan
            ret = (p_out - p_in) / p_in
            if dir_char == 's':
                ret = -ret
            return 1.0 if ret > 0 else 0.0
        except (KeyError, IndexError):
            return np.nan

    def _compute_neighborhood_features(self, symbol, date, daysOut, dir_char, price_df):
        """
        Compute neighborhood win-rate features from realized price history.

        Mirrors compute_neighborhood_features() in build_training_data.py exactly:
        for each temporal shift (+-7, +-14 days), replay the shifted pattern across
        the prior 10 years and measure realized win rates from actual price data.

        Returns dict with pat_neighbor_avg_wr, pat_sharpness, pat_pre_slope,
        pat_post_cliff -- all NaN if price data is unavailable.
        """
        nan_result = {
            'pat_neighbor_avg_wr': np.nan,
            'pat_sharpness': np.nan,
            'pat_pre_slope': np.nan,
            'pat_post_cliff': np.nan,
        }
        if price_df is None or price_df.empty:
            return nan_result

        dir_mult = 1 if dir_char == 'l' else -1
        trading_days_set = set(price_df.index)
        close_values = price_df['close']

        if isinstance(date, str):
            date = pd.Timestamp(date)

        sample_year = date.year
        month_day = f"{date.month:02d}-{date.day:02d}"
        prior_years = list(range(max(sample_year - 10, 2000), sample_year))

        def _year_win(yr, md):
            """Return 1.0/0.0 for realized return in year yr at month-day md, or None."""
            try:
                base = pd.Timestamp(f"{yr}-{md}")
            except (ValueError, Exception):
                return None
            entry = None
            for offset in range(0, 4):
                candidate = base + pd.Timedelta(days=offset)
                if candidate in trading_days_set:
                    entry = candidate
                    break
            if entry is None:
                return None
            exit_raw = entry + pd.Timedelta(days=daysOut)
            exit_date = None
            for offset in range(0, 5):
                candidate = exit_raw + pd.Timedelta(days=offset)
                if candidate in trading_days_set:
                    exit_date = candidate
                    break
            if exit_date is None:
                return None
            try:
                p_in = close_values[entry]
                p_out = close_values[exit_date]
                if p_in == 0 or np.isnan(p_in) or np.isnan(p_out):
                    return None
                ret = (p_out - p_in) / p_in * dir_mult
                return 1.0 if ret > 0 else 0.0
            except (KeyError, IndexError):
                return None

        # Compute win rates at each shifted date across prior years.
        # Build hist_base = {yr}-{month_day} first then add shift so year-crossing
        # is handled correctly (matches Round 1 Fix 4 in build_training_data.py).
        shifts = [(-14, 'pre2w'), (-7, 'pre1w'), (7, 'post1w'), (14, 'post2w')]
        shifted_wrs = {}  # label -> win rate

        for shift_days, label in shifts:
            wins = []
            for yr in prior_years:
                try:
                    hist_base = pd.Timestamp(f"{yr}-{month_day}")
                    shifted = hist_base + pd.Timedelta(days=shift_days)
                    shifted_md = f"{shifted.month:02d}-{shifted.day:02d}"
                    shifted_yr = shifted.year
                except (ValueError, Exception):
                    continue
                w = _year_win(shifted_yr, shifted_md)
                if w is not None:
                    wins.append(w)
            if wins:
                shifted_wrs[label] = float(np.mean(wins))

        all_shifted_wrs = list(shifted_wrs.values())
        pre_wrs = [shifted_wrs[l] for l in ('pre2w', 'pre1w') if l in shifted_wrs]
        post_wrs = [shifted_wrs[l] for l in ('post1w', 'post2w') if l in shifted_wrs]

        # Pattern's own prior-year win rate
        pat_wins = []
        for yr in prior_years:
            w = _year_win(yr, month_day)
            if w is not None:
                pat_wins.append(w)

        neighbor_avg = float(np.mean(all_shifted_wrs)) if all_shifted_wrs else np.nan
        pat_wr_prior = float(np.mean(pat_wins)) if pat_wins else np.nan

        sharpness = np.nan
        if not np.isnan(pat_wr_prior) and not np.isnan(neighbor_avg) and neighbor_avg > 0:
            sharpness = pat_wr_prior / neighbor_avg

        # pat_pre_slope: pre1w_wr - pre2w_wr (same sign convention as training)
        pre_slope = (pre_wrs[1] - pre_wrs[0]) if len(pre_wrs) == 2 else np.nan

        # pat_post_cliff: pat_wr_prior - mean(post1w, post2w)
        post_cliff = np.nan
        if post_wrs and not np.isnan(pat_wr_prior):
            post_cliff = pat_wr_prior - float(np.mean(post_wrs))

        return {
            'pat_neighbor_avg_wr': neighbor_avg,
            'pat_sharpness': sharpness,
            'pat_pre_slope': pre_slope,
            'pat_post_cliff': post_cliff,
        }

    # ------------------------------------------------------------------
    # Group 2: Technical Momentum Features (20)
    # ------------------------------------------------------------------

    def _compute_sma(self, closes, period):
        """Simple moving average."""
        if len(closes) < period:
            return np.nan
        return closes[-period:].mean()

    def _compute_ema(self, closes, period):
        """Exponential moving average."""
        if len(closes) < period:
            return np.nan
        return pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1]

    def _compute_rsi(self, closes, period=14):
        """RSI(14)."""
        if len(closes) < period + 1:
            return np.nan
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_macd(self, closes):
        """MACD line, signal, histogram."""
        if len(closes) < 26:
            return np.nan, np.nan, np.nan
        s = pd.Series(closes)
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal
        return macd_line.iloc[-1], signal.iloc[-1], histogram.iloc[-1]

    def _compute_atr(self, highs, lows, closes, period=14):
        """Average True Range."""
        if len(closes) < period + 1:
            return np.nan
        trs = []
        for i in range(-period, 0):
            h = highs[i]
            l = lows[i]
            cp = closes[i - 1]
            tr = max(h - l, abs(h - cp), abs(l - cp))
            trs.append(tr)
        return np.mean(trs)

    def _compute_obv_slope(self, closes, volumes, period=10):
        """OBV 10-day regression slope."""
        if len(closes) < period + 1:
            return np.nan
        obv = [0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        obv_recent = obv[-period:]
        x = np.arange(period)
        if np.std(obv_recent) == 0:
            return 0.0
        slope = np.polyfit(x, obv_recent, 1)[0]
        return slope

    def _compute_bollinger_position(self, closes, period=20):
        """Position within Bollinger Bands (0=lower, 1=upper)."""
        if len(closes) < period:
            return np.nan
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        if std == 0:
            return 0.5
        upper = sma + 2 * std
        lower = sma - 2 * std
        price = closes[-1]
        band_width = upper - lower
        if band_width == 0:
            return 0.5
        return (price - lower) / band_width

    def compute_technical_features(self, symbol, date):
        """
        Group 2: Technical Momentum features (20).
        All computed from OHLCV data.
        """
        features = {}
        df = self._get_price_df(symbol)
        nan_keys = [
            'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
            'sma20_vs_sma50', 'sma50_vs_sma200',
            'roc_5', 'roc_20', 'rsi_14', 'macd_histogram', 'macd_hist_slope',
            'rvol_20', 'obv_slope_10', 'volume_price_confirm',
            'atr_percentile', 'bollinger_position',
            'rs_vs_spy_20d',
            'trend_long', 'trend_short', 'trend_delta', 'trend_direction_match'
        ]
        if df is None:
            return {f'ta_{k}': np.nan for k in nan_keys}

        if isinstance(date, str):
            date = pd.Timestamp(date)

        # Get data up to this date
        mask = df.index <= date
        sub = df[mask].tail(250)  # ~1 year of trading days
        if len(sub) < 30:
            return {f'ta_{k}': np.nan for k in nan_keys}

        closes = sub['close'].values
        highs = sub['high'].values
        lows = sub['low'].values
        volumes = sub['volume'].values
        price = closes[-1]

        # SMAs
        sma20 = self._compute_sma(closes, 20)
        sma50 = self._compute_sma(closes, 50)
        sma200 = self._compute_sma(closes, 200)

        # ATR for normalization
        atr = self._compute_atr(highs, lows, closes, 14)
        if atr is None or np.isnan(atr) or atr == 0:
            atr = 1.0  # fallback

        features['ta_price_vs_sma20'] = (price - sma20) / atr if not np.isnan(sma20) else np.nan
        features['ta_price_vs_sma50'] = (price - sma50) / atr if not np.isnan(sma50) else np.nan
        features['ta_price_vs_sma200'] = (price - sma200) / atr if not np.isnan(sma200) else np.nan
        features['ta_sma20_vs_sma50'] = (sma20 - sma50) / atr if not (np.isnan(sma20) or np.isnan(sma50)) else np.nan
        features['ta_sma50_vs_sma200'] = (sma50 - sma200) / atr if not (np.isnan(sma50) or np.isnan(sma200)) else np.nan

        # Momentum
        features['ta_roc_5'] = (price - closes[-6]) / closes[-6] if len(closes) >= 6 else np.nan
        features['ta_roc_20'] = (price - closes[-21]) / closes[-21] if len(closes) >= 21 else np.nan
        features['ta_rsi_14'] = self._compute_rsi(closes, 14)

        macd_line, macd_signal, macd_hist = self._compute_macd(closes)
        features['ta_macd_histogram'] = macd_hist

        # MACD histogram slope (today vs yesterday)
        if len(closes) >= 27:
            _, _, hist_prev = self._compute_macd(closes[:-1])
            features['ta_macd_hist_slope'] = macd_hist - hist_prev if not np.isnan(hist_prev) else np.nan
        else:
            features['ta_macd_hist_slope'] = np.nan

        # Volume
        avg_vol_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.nan
        features['ta_rvol_20'] = volumes[-1] / avg_vol_20 if avg_vol_20 and avg_vol_20 > 0 else np.nan
        features['ta_obv_slope_10'] = self._compute_obv_slope(closes, volumes, 10) if len(closes) >= 11 else np.nan

        # Volume-price confirmation
        if len(closes) >= 2 and len(volumes) >= 2:
            price_up = closes[-1] > closes[-2]
            vol_up = volumes[-1] > volumes[-2]
            if price_up and vol_up:
                features['ta_volume_price_confirm'] = 1
            elif not price_up and not vol_up:
                features['ta_volume_price_confirm'] = 1
            else:
                features['ta_volume_price_confirm'] = -1
        else:
            features['ta_volume_price_confirm'] = np.nan

        # Volatility - vectorized ATR percentile
        if len(closes) >= 74:
            # Compute true ranges for all bars
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])
                )
            )
            # Rolling 14-day mean of TR
            if len(tr) >= 14:
                # Use cumsum for fast rolling mean
                cs = np.cumsum(tr)
                rolling_atr = np.empty(len(tr) - 13)
                rolling_atr[0] = cs[13] / 14
                rolling_atr[1:] = (cs[14:] - cs[:-14]) / 14
                # Take last 60 values for percentile
                if len(rolling_atr) >= 60:
                    recent_atrs = rolling_atr[-60:]
                    current_atr = rolling_atr[-1]
                    features['ta_atr_percentile'] = np.sum(recent_atrs <= current_atr) / len(recent_atrs)
                else:
                    features['ta_atr_percentile'] = np.nan
            else:
                features['ta_atr_percentile'] = np.nan
        else:
            features['ta_atr_percentile'] = np.nan

        features['ta_bollinger_position'] = self._compute_bollinger_position(closes, 20)

        # Relative strength vs SPY
        spy_df = self._get_price_df('SPY')
        if spy_df is not None and len(closes) >= 21:
            spy_sub = spy_df[spy_df.index <= date].tail(21)
            if len(spy_sub) >= 21:
                spy_ret = (spy_sub['close'].iloc[-1] - spy_sub['close'].iloc[0]) / spy_sub['close'].iloc[0]
                stock_ret = (closes[-1] - closes[-21]) / closes[-21]
                features['ta_rs_vs_spy_20d'] = stock_ret - spy_ret
            else:
                features['ta_rs_vs_spy_20d'] = np.nan
        else:
            features['ta_rs_vs_spy_20d'] = np.nan

        # Trend scores (computed from components, matching StockScore logic)
        trend_long, trend_short = self._compute_trend_scores(closes, highs, lows, sma20, sma50, sma200, atr)
        features['ta_trend_long'] = trend_long
        features['ta_trend_short'] = trend_short

        # Trend delta (vs prior day)
        if len(closes) >= 31:
            prev_closes = closes[:-1]
            prev_sma20 = self._compute_sma(prev_closes, 20)
            prev_sma50 = self._compute_sma(prev_closes, 50)
            prev_sma200 = self._compute_sma(prev_closes, 200)
            prev_atr = self._compute_atr(highs[:-1], lows[:-1], prev_closes, 14)
            prev_tl, _ = self._compute_trend_scores(prev_closes, highs[:-1], lows[:-1],
                                                     prev_sma20, prev_sma50, prev_sma200,
                                                     prev_atr if not np.isnan(prev_atr) else 1.0)
            features['ta_trend_delta'] = trend_long - prev_tl
        else:
            features['ta_trend_delta'] = np.nan

        # Trend direction match (does momentum agree with pattern direction?)
        features['ta_trend_direction_match'] = np.nan  # set by caller who knows direction

        return features

    def _compute_trend_scores(self, closes, highs, lows, sma20, sma50, sma200, atr):
        """
        Simplified trend score (0-100) approximating StockScore.
        10 components each 0-10.
        """
        if len(closes) < 26 or atr == 0:
            return np.nan, np.nan

        price = closes[-1]
        lscore = 0
        sscore = 0

        # 1-4: Price vs SMAs (each 0-10)
        for sma, weight in [(sma20, 1), (sma50, 1), (None, 1), (sma200, 1)]:
            if sma is None:
                # SMA100
                sma = self._compute_sma(closes, 100)
            if np.isnan(sma):
                lscore += 5
                sscore += 5
                continue
            diff = (price - sma) / atr
            ls = min(max(5 + diff, 0), 10)
            ss = min(max(5 - diff, 0), 10)
            lscore += ls
            sscore += ss

        # 5-6: MACD direction and value
        macd_line, macd_signal, macd_hist = self._compute_macd(closes)
        if not np.isnan(macd_hist):
            # MACD direction
            if macd_hist > 0:
                lscore += 7
                sscore += 3
            else:
                lscore += 3
                sscore += 7
            # MACD value
            macd_norm = min(abs(macd_hist) / atr, 5) / 5 * 10
            if macd_hist > 0:
                lscore += macd_norm
                sscore += (10 - macd_norm)
            else:
                lscore += (10 - macd_norm)
                sscore += macd_norm
        else:
            lscore += 10
            sscore += 10

        # 7: EMA5 vs WMA13
        ema5 = self._compute_ema(closes, 5)
        # Approximate WMA13
        if len(closes) >= 13:
            weights = np.arange(1, 14)
            wma13 = np.average(closes[-13:], weights=weights)
        else:
            wma13 = np.nan
        if not (np.isnan(ema5) or np.isnan(wma13)):
            if ema5 > wma13:
                lscore += 7
                sscore += 3
            else:
                lscore += 3
                sscore += 7
        else:
            lscore += 5
            sscore += 5

        # 8: Parabolic SAR (simplified - use price vs EMA5 as proxy)
        if not np.isnan(ema5):
            if price > ema5:
                lscore += 7
                sscore += 3
            else:
                lscore += 3
                sscore += 7
        else:
            lscore += 5
            sscore += 5

        # 9: RSI
        rsi = self._compute_rsi(closes, 14)
        if not np.isnan(rsi):
            rsi_l = min(max(rsi / 10, 0), 10)
            rsi_s = 10 - rsi_l
            lscore += rsi_l
            sscore += rsi_s
        else:
            lscore += 5
            sscore += 5

        # 10: ADX (simplified - use abs(ROC20) as proxy for trend strength)
        if len(closes) >= 21:
            roc20 = abs(closes[-1] - closes[-21]) / closes[-21]
            adx_proxy = min(roc20 * 200, 10)  # scale to 0-10
            lscore += adx_proxy / 2 + 2.5  # bias toward neutral
            sscore += adx_proxy / 2 + 2.5
        else:
            lscore += 5
            sscore += 5

        return lscore, sscore

    # ------------------------------------------------------------------
    # Group 3: Market Regime Features (14 V1 + 4 V2)
    # ------------------------------------------------------------------

    def compute_market_regime_features(self, date):
        """
        Group 3: Market Regime features (14 V1 + 4 V2).
        Broad market conditions from VIX, bonds, credit, SPY, breadth, fed rate.
        """
        features = {}
        if isinstance(date, str):
            date = pd.Timestamp(date)

        # VIX
        vix_df = self._get_price_df('VIX')
        if vix_df is not None:
            vix_sub = vix_df[vix_df.index <= date]
            if len(vix_sub) >= 1:
                vix = vix_sub['close'].iloc[-1]
                features['mkt_vix_level'] = vix

                if len(vix_sub) >= 60:
                    vix_60 = vix_sub['close'].tail(60).values
                    features['mkt_vix_percentile_60d'] = sum(1 for v in vix_60 if v <= vix) / len(vix_60)
                else:
                    features['mkt_vix_percentile_60d'] = np.nan

                if len(vix_sub) >= 6:
                    vix_5d_ago = vix_sub['close'].iloc[-6]
                    features['mkt_vix_5d_change'] = (vix - vix_5d_ago) / vix_5d_ago if vix_5d_ago != 0 else np.nan
                else:
                    features['mkt_vix_5d_change'] = np.nan

                # VIX term structure: VIX vs VIX3M
                vix3m_df = self._get_price_df('VIX3M')
                if vix3m_df is not None:
                    vix3m_sub = vix3m_df[vix3m_df.index <= date]
                    if len(vix3m_sub) >= 1:
                        vix3m = vix3m_sub['close'].iloc[-1]
                        features['mkt_vix_term_structure'] = vix - vix3m  # positive = backwardation (panic)
                    else:
                        features['mkt_vix_term_structure'] = np.nan
                else:
                    features['mkt_vix_term_structure'] = np.nan
            else:
                features['mkt_vix_level'] = np.nan
                features['mkt_vix_percentile_60d'] = np.nan
                features['mkt_vix_5d_change'] = np.nan
                features['mkt_vix_term_structure'] = np.nan
        else:
            features['mkt_vix_level'] = np.nan
            features['mkt_vix_percentile_60d'] = np.nan
            features['mkt_vix_5d_change'] = np.nan
            features['mkt_vix_term_structure'] = np.nan

        # Yield curve
        us10y_df = self._get_price_df('US10Y')
        us2y_df = self._get_price_df('US2Y')
        if us10y_df is not None and us2y_df is not None:
            y10_sub = us10y_df[us10y_df.index <= date]
            y2_sub = us2y_df[us2y_df.index <= date]
            if len(y10_sub) >= 1 and len(y2_sub) >= 1:
                # Align to same date
                y10 = self._get_price_on_date(us10y_df, date)
                y2 = self._get_price_on_date(us2y_df, date)
                if y10 is not None and y2 is not None:
                    features['mkt_yield_curve_10y2y'] = y10 - y2
                    # Slope: 5-trading-day change in spread (matches training diff(5))
                    y10_aligned = us10y_df[us10y_df.index <= date]
                    y2_aligned = us2y_df[us2y_df.index <= date]
                    common_idx = y10_aligned.index.intersection(y2_aligned.index)
                    if len(common_idx) >= 6:
                        spread_series = y10_aligned.loc[common_idx, 'close'] - y2_aligned.loc[common_idx, 'close']
                        features['mkt_yield_curve_slope'] = spread_series.iloc[-1] - spread_series.iloc[-6]
                    else:
                        features['mkt_yield_curve_slope'] = np.nan
                else:
                    features['mkt_yield_curve_10y2y'] = np.nan
                    features['mkt_yield_curve_slope'] = np.nan
            else:
                features['mkt_yield_curve_10y2y'] = np.nan
                features['mkt_yield_curve_slope'] = np.nan
        else:
            features['mkt_yield_curve_10y2y'] = np.nan
            features['mkt_yield_curve_slope'] = np.nan

        # Credit spread (HYG - LQD normalized)
        hyg_df = self._get_price_df('HYG')
        lqd_df = self._get_price_df('LQD')
        if hyg_df is not None and lqd_df is not None:
            hyg_sub = hyg_df[hyg_df.index <= date]
            lqd_sub = lqd_df[lqd_df.index <= date]
            # Align to common trading days then form ratio series (matches training)
            common_idx = hyg_sub.index.intersection(lqd_sub.index)
            if len(common_idx) >= 1:
                ratio_series = hyg_sub.loc[common_idx, 'close'] / lqd_sub.loc[common_idx, 'close']
                features['mkt_credit_spread'] = ratio_series.iloc[-1]
                if len(ratio_series) >= 21:
                    features['mkt_credit_spread_change_20d'] = ratio_series.iloc[-1] - ratio_series.iloc[-21]
                else:
                    features['mkt_credit_spread_change_20d'] = np.nan
            else:
                features['mkt_credit_spread'] = np.nan
                features['mkt_credit_spread_change_20d'] = np.nan
        else:
            features['mkt_credit_spread'] = np.nan
            features['mkt_credit_spread_change_20d'] = np.nan

        # SPY trend
        spy_df = self._get_price_df('SPY')
        if spy_df is not None:
            spy_sub = spy_df[spy_df.index <= date].tail(250)
            if len(spy_sub) >= 21:
                spy_ret_20d = (spy_sub['close'].iloc[-1] - spy_sub['close'].iloc[-21]) / spy_sub['close'].iloc[-21]
                features['mkt_spy_roc_20'] = spy_ret_20d
            else:
                features['mkt_spy_roc_20'] = np.nan

            if len(spy_sub) >= 200:
                spy_sma200 = spy_sub['close'].tail(200).mean()
                features['mkt_spy_above_sma200'] = 1 if spy_sub['close'].iloc[-1] > spy_sma200 else 0
            else:
                features['mkt_spy_above_sma200'] = np.nan
        else:
            features['mkt_spy_roc_20'] = np.nan
            features['mkt_spy_above_sma200'] = np.nan

        # Breadth approximation: ADVN/DECN ratio
        advn_df = self._get_price_df('ADVN')
        decn_df = self._get_price_df('DECN')
        if advn_df is not None and decn_df is not None:
            advn_sub = advn_df[advn_df.index <= date].tail(15)
            decn_sub = decn_df[decn_df.index <= date].tail(15)
            if len(advn_sub) >= 10 and len(decn_sub) >= 10:
                adv = advn_sub['close'].tail(10).mean()
                dec = decn_sub['close'].tail(10).mean()
                features['mkt_adv_decl_ratio_10d'] = adv / dec if dec > 0 else np.nan
            else:
                features['mkt_adv_decl_ratio_10d'] = np.nan
        else:
            features['mkt_adv_decl_ratio_10d'] = np.nan

        # spy_breadth_approx - skip for now, too expensive per-sample. Use adv/decl instead.
        features['mkt_spy_breadth_approx'] = np.nan

        # New highs-lows (not available in data, leave as NaN for now)
        features['mkt_new_highs_lows_10d'] = np.nan

        # Sector rotation: XLK vs XLU relative performance 20d
        xlk_df = self._get_price_df('XLK')
        xlu_df = self._get_price_df('XLU')
        if xlk_df is not None and xlu_df is not None:
            xlk_sub = xlk_df[xlk_df.index <= date].tail(25)
            xlu_sub = xlu_df[xlu_df.index <= date].tail(25)
            if len(xlk_sub) >= 21 and len(xlu_sub) >= 21:
                xlk_ret = (xlk_sub['close'].iloc[-1] - xlk_sub['close'].iloc[-21]) / xlk_sub['close'].iloc[-21]
                xlu_ret = (xlu_sub['close'].iloc[-1] - xlu_sub['close'].iloc[-21]) / xlu_sub['close'].iloc[-21]
                features['mkt_sector_rotation'] = xlk_ret - xlu_ret
            else:
                features['mkt_sector_rotation'] = np.nan
        else:
            features['mkt_sector_rotation'] = np.nan

        # ------ V2 Market Regime Features ------

        # mkt_vix_regime_bucket: 0 if VIX<15, 1 if 15-20, 2 if 20-30, 3 if >30
        vix_level = features.get('mkt_vix_level', np.nan)
        if not np.isnan(vix_level):
            if vix_level < 15:
                features['mkt_vix_regime_bucket'] = 0
            elif vix_level < 20:
                features['mkt_vix_regime_bucket'] = 1
            elif vix_level < 30:
                features['mkt_vix_regime_bucket'] = 2
            else:
                features['mkt_vix_regime_bucket'] = 3
        else:
            features['mkt_vix_regime_bucket'] = np.nan

        # mkt_breadth_momentum: 20-trading-day change in 10-day rolling adv/decl ratio
        # Matches training: adl_ratio = adv_10d_roll / dec_10d_roll, then diff(20)
        if advn_df is not None and decn_df is not None:
            advn_long = advn_df[advn_df.index <= date]
            decn_long = decn_df[decn_df.index <= date]
            common_idx = advn_long.index.intersection(decn_long.index)
            if len(common_idx) >= 30:
                adv_roll = advn_long.loc[common_idx, 'close'].rolling(10).mean()
                dec_roll = decn_long.loc[common_idx, 'close'].rolling(10).mean()
                ratio_series = adv_roll / dec_roll.replace(0, np.nan)
                if not np.isnan(ratio_series.iloc[-1]) and not np.isnan(ratio_series.iloc[-21]):
                    features['mkt_breadth_momentum'] = ratio_series.iloc[-1] - ratio_series.iloc[-21]
                else:
                    features['mkt_breadth_momentum'] = np.nan
            else:
                features['mkt_breadth_momentum'] = np.nan
        else:
            features['mkt_breadth_momentum'] = np.nan

        # mkt_fed_rate_level: IRX close (13-week T-bill yield, proxy for fed funds)
        irx_df = self._get_price_df('IRX')
        if irx_df is not None:
            irx_price = self._get_price_on_date(irx_df, date)
            if irx_price is not None:
                features['mkt_fed_rate_level'] = irx_price
                # mkt_fed_rate_direction: exactly 60 trading rows back (matches training diff(60))
                irx_sub = irx_df[irx_df.index <= date]
                if len(irx_sub) >= 61:
                    features['mkt_fed_rate_direction'] = irx_sub['close'].iloc[-1] - irx_sub['close'].iloc[-61]
                else:
                    features['mkt_fed_rate_direction'] = np.nan
            else:
                features['mkt_fed_rate_level'] = np.nan
                features['mkt_fed_rate_direction'] = np.nan
        else:
            features['mkt_fed_rate_level'] = np.nan
            features['mkt_fed_rate_direction'] = np.nan

        # V3: Commodity / FX regime -- 20-day ROC for DXY, CL, GC
        for _sym, _feat in [('DXY', 'mkt_dxy_roc_20'), ('CL', 'mkt_cl_roc_20'), ('GC', 'mkt_gc_roc_20')]:
            _df = (self._get_commodity_price_df(_sym)
                   if _sym in ('CL', 'GC') else self._get_price_df(_sym))
            if _df is not None:
                _sub = _df[_df.index <= date].tail(30)
                if len(_sub) >= 21:
                    features[_feat] = (_sub['close'].iloc[-1] - _sub['close'].iloc[-21]) / _sub['close'].iloc[-21]
                else:
                    features[_feat] = np.nan
            else:
                features[_feat] = np.nan

        return features

    # ------------------------------------------------------------------
    # Group 4: Stock-Specific Context Features (9)
    # ------------------------------------------------------------------

    def compute_stock_context_features(self, symbol, date):
        """
        Group 4: Stock-Specific Context features (9).
        What makes this stock unique right now.
        """
        features = {}
        if isinstance(date, str):
            date = pd.Timestamp(date)

        df = self._get_price_df(symbol)
        if df is None:
            return {f'ctx_{k}': np.nan for k in [
                'pct_from_52w_high', 'pct_from_52w_low', 'position_in_52w_range',
                'avg_volume_20d', 'market_cap_bucket',
                'days_to_earnings', 'near_earnings',
                'sector_rs_20d', 'stock_vs_sector_20d'
            ]}

        sub = df[df.index <= date].tail(252)  # 252 trading days = 1 year (matches training rolling(252))
        if len(sub) < 20:
            return {f'ctx_{k}': np.nan for k in [
                'pct_from_52w_high', 'pct_from_52w_low', 'position_in_52w_range',
                'avg_volume_20d', 'market_cap_bucket',
                'days_to_earnings', 'near_earnings',
                'sector_rs_20d', 'stock_vs_sector_20d'
            ]}

        price = sub['close'].iloc[-1]
        high_52w = sub['high'].max()
        low_52w = sub['low'].min()

        features['ctx_pct_from_52w_high'] = (price - high_52w) / high_52w if high_52w > 0 else np.nan
        features['ctx_pct_from_52w_low'] = (price - low_52w) / low_52w if low_52w > 0 else np.nan
        range_52w = high_52w - low_52w
        features['ctx_position_in_52w_range'] = (price - low_52w) / range_52w if range_52w > 0 else 0.5

        avg_vol = sub['volume'].tail(20).mean()
        features['ctx_avg_volume_20d'] = avg_vol
        features['ctx_market_cap_bucket'] = math.log(price * avg_vol) if price > 0 and avg_vol > 0 else np.nan

        # Earnings proximity
        earnings_path = os.path.join(EARNINGS_DIR, f'{symbol}.json')
        if os.path.exists(earnings_path):
            try:
                with open(earnings_path) as f:
                    earnings_data = json.load(f)
                next_est = earnings_data.get('next_earnings_est')
                if next_est:
                    next_date = pd.Timestamp(next_est)
                    days_to = (next_date - date).days
                    features['ctx_days_to_earnings'] = max(days_to, 0)
                    features['ctx_near_earnings'] = 1 if days_to <= 5 else 0
                else:
                    features['ctx_days_to_earnings'] = np.nan
                    features['ctx_near_earnings'] = np.nan
            except Exception:
                features['ctx_days_to_earnings'] = np.nan
                features['ctx_near_earnings'] = np.nan
        else:
            features['ctx_days_to_earnings'] = np.nan
            features['ctx_near_earnings'] = np.nan

        # Sector relative strength
        sector = TICKER_SECTOR.get(symbol) or ETF_SECTOR.get(symbol)
        if sector:
            etf_sym = SECTOR_ETF.get(sector) or ETF_CATEGORY_SECTOR_ETF.get(sector)
            if etf_sym:
                etf_df = self._get_price_df(etf_sym)
                spy_df = self._get_price_df('SPY')
                if etf_df is not None and spy_df is not None:
                    etf_sub = etf_df[etf_df.index <= date].tail(25)
                    spy_sub = spy_df[spy_df.index <= date].tail(25)
                    if len(etf_sub) >= 21 and len(spy_sub) >= 21:
                        etf_ret = (etf_sub['close'].iloc[-1] - etf_sub['close'].iloc[-21]) / etf_sub['close'].iloc[-21]
                        spy_ret = (spy_sub['close'].iloc[-1] - spy_sub['close'].iloc[-21]) / spy_sub['close'].iloc[-21]
                        features['ctx_sector_rs_20d'] = etf_ret - spy_ret

                        # Stock vs sector
                        stock_ret = (sub['close'].iloc[-1] - sub['close'].iloc[-21]) / sub['close'].iloc[-21] if len(sub) >= 21 else np.nan
                        features['ctx_stock_vs_sector_20d'] = stock_ret - etf_ret if not np.isnan(stock_ret) else np.nan
                    else:
                        features['ctx_sector_rs_20d'] = np.nan
                        features['ctx_stock_vs_sector_20d'] = np.nan
                else:
                    features['ctx_sector_rs_20d'] = np.nan
                    features['ctx_stock_vs_sector_20d'] = np.nan
            else:
                features['ctx_sector_rs_20d'] = np.nan
                features['ctx_stock_vs_sector_20d'] = np.nan
        else:
            features['ctx_sector_rs_20d'] = np.nan
            features['ctx_stock_vs_sector_20d'] = np.nan

        return features

    # ------------------------------------------------------------------
    # Group 5: Calendar/Cycle Features (7)
    # ------------------------------------------------------------------

    def compute_calendar_features(self, date):
        """
        Group 5: Calendar/Cycle features (7).
        Time-based patterns.
        """
        features = {}
        if isinstance(date, str):
            date = pd.Timestamp(date)

        features['cal_month'] = date.month
        features['cal_day_of_year'] = date.timetuple().tm_yday / (366 if date.is_leap_year else 365)
        features['cal_quarter'] = (date.month - 1) // 3 + 1

        # Week of month
        features['cal_week_of_month'] = (date.day - 1) // 7 + 1

        # OpEx week: 3rd Friday of month
        import calendar
        cal = calendar.Calendar()
        fridays = [d for d in cal.itermonthdays2(date.year, date.month) if d[0] != 0 and d[1] == 4]
        if len(fridays) >= 3:
            opex_day = fridays[2][0]
            opex_date = date.replace(day=opex_day)
            # Is current date within the OpEx week (Mon-Fri containing 3rd Friday)?
            opex_week_start = opex_date - pd.Timedelta(days=opex_date.weekday())
            opex_week_end = opex_week_start + pd.Timedelta(days=4)
            features['cal_is_opex_week'] = 1 if opex_week_start <= date <= opex_week_end else 0
        else:
            features['cal_is_opex_week'] = 0

        # Month position: early(1-10)=0, mid(11-20)=1, late(21+)=2
        if date.day <= 10:
            features['cal_month_position'] = 0
        elif date.day <= 20:
            features['cal_month_position'] = 1
        else:
            features['cal_month_position'] = 2

        # Presidential election cycle year
        features['cal_pe_year'] = get_pe_year(date.year)

        return features

    # ------------------------------------------------------------------
    # Group 6: SPX Seasonal Regime Features (4)
    # ------------------------------------------------------------------

    def _get_spx_seasonal_lookup(self, year=None):
        """
        Lazy-load SPX seasonal lookup.
        For production scoring: uses SPX data from 1960 through (year-1).
        If year is None, uses current year - 1 as the end year.
        Caches per year for efficiency.
        """
        if year is None:
            year = pd.Timestamp.now().year
        end_year = year - 1

        path = os.path.join(CSV_DIR, 'INDX', 'SPX.csv')
        source_version = None
        if os.path.exists(path):
            stat = os.stat(path)
            source_version = (
                path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)

        # Check cache by both training cutoff and file generation. This keeps
        # same-date EOD corrections aligned with the live metadata identity.
        if self._spx_seasonal_lookup is not None and isinstance(self._spx_seasonal_lookup, dict):
            if (self._spx_seasonal_lookup.get('_end_year') == end_year
                    and self._spx_seasonal_lookup.get('_source_version') == source_version):
                return self._spx_seasonal_lookup

        if not os.path.exists(path):
            self._spx_seasonal_lookup = {
                '_end_year': end_year, '_source_version': source_version}
            return self._spx_seasonal_lookup

        df = pd.read_csv(path, index_col=0)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[(df.index.year >= 1960) & (df.index.year <= end_year)]

        if len(df) < 500:
            self._spx_seasonal_lookup = {
                '_end_year': end_year, '_source_version': source_version}
            return self._spx_seasonal_lookup

        closes = df['close'].values
        dates = df.index
        fwd_days = SPX_SEASONAL_FORWARD_DAYS

        fwd_returns = np.full(len(closes), np.nan)
        for i in range(len(closes) - fwd_days):
            if closes[i] != 0:
                fwd_returns[i] = (closes[i + fwd_days] - closes[i]) / closes[i] * 100.0

        weeks = np.array([d.isocalendar()[1] for d in dates])
        pe_phases = np.array([get_pe_year(d.year) for d in dates])

        lookup = {
            '_end_year': end_year,
            '_source_version': source_version,
        }
        for wk in range(1, 54):
            for pe in range(1, 5):
                mask = (weeks == wk) & (pe_phases == pe) & ~np.isnan(fwd_returns)
                if mask.sum() < 5:
                    continue
                rets = fwd_returns[mask]
                wr = (rets > 0).sum() / len(rets)
                avg_ret = rets.mean()
                lookup[(wk, pe)] = {'wr': wr, 'ret': avg_ret}

        self._spx_seasonal_lookup = lookup
        return lookup

    def compute_spx_seasonal_features(self, date, direction):
        """
        Group 6: SPX Seasonal Regime features (4).
        Uses SPX data from 1960 through (date.year - 1) to determine seasonal
        bullish/bearish tendency for the given (week_of_year, PE_cycle_phase).
        """
        features = {}
        if isinstance(date, str):
            date = pd.Timestamp(date)

        lookup = self._get_spx_seasonal_lookup(year=date.year)
        if len(lookup) <= 2:  # only internal cache metadata keys
            features['mkt_spx_seasonal_wr'] = np.nan
            features['mkt_spx_seasonal_ret'] = np.nan
            features['mkt_spx_seasonal_regime'] = np.nan
            features['mkt_spx_dir_alignment'] = np.nan
            return features

        wk = date.isocalendar()[1]
        pe = get_pe_year(date.year)
        entry = lookup.get((wk, pe))

        if entry is None:
            features['mkt_spx_seasonal_wr'] = np.nan
            features['mkt_spx_seasonal_ret'] = np.nan
            features['mkt_spx_seasonal_regime'] = np.nan
            features['mkt_spx_dir_alignment'] = np.nan
            return features

        wr = entry['wr']
        features['mkt_spx_seasonal_wr'] = wr
        features['mkt_spx_seasonal_ret'] = entry['ret']

        # Regime bucket
        if wr < 0.40:
            features['mkt_spx_seasonal_regime'] = -2
        elif wr < 0.50:
            features['mkt_spx_seasonal_regime'] = -1
        elif wr < 0.60:
            features['mkt_spx_seasonal_regime'] = 0
        elif wr < 0.70:
            features['mkt_spx_seasonal_regime'] = 1
        else:
            features['mkt_spx_seasonal_regime'] = 2

        # Direction alignment: +1 (aligned with strong season), 0 (neutral), -1 (against)
        dir_char = direction[0].lower() if isinstance(direction, str) else direction
        pat_dir = 1 if dir_char == 'l' else -1
        if wr >= 0.55:
            features['mkt_spx_dir_alignment'] = pat_dir * 1
        elif wr <= 0.45:
            features['mkt_spx_dir_alignment'] = pat_dir * -1
        else:
            features['mkt_spx_dir_alignment'] = 0

        return features

    # ------------------------------------------------------------------
    # V2 Interaction Features (4)
    # ------------------------------------------------------------------

    def compute_interaction_features(self, features, symbol, date, daysOut, direction):
        """
        Compute V2 interaction features from the combined feature dict.

        Args:
            features: dict with all pattern + market features already computed
            symbol: ticker
            date: entry date (Timestamp)
            daysOut: holding period
            direction: 'l' or 's'

        Returns:
            dict of 4 interaction features
        """
        result = {}
        if isinstance(date, str):
            date = pd.Timestamp(date)
        dir_char = direction[0].lower() if isinstance(direction, str) else direction
        dir_sign = 1.0 if dir_char == 'l' else -1.0

        # pat_dir_x_mkt_trend: direction_sign * spy_above_sma200_sign
        spy_above = features.get('mkt_spy_above_sma200', np.nan)
        if not np.isnan(spy_above):
            spy_sign = 1.0 if spy_above == 1 else -1.0
            result['pat_dir_x_mkt_trend'] = dir_sign * spy_sign
        else:
            result['pat_dir_x_mkt_trend'] = np.nan

        # pat_dir_x_sector_trend: direction_sign * sign(sector_etf_price > SMA200)
        # Matches training: sector_trend = (sector_etf_close > sector_etf_sma200) ? +1 : -1
        sector = TICKER_SECTOR.get(symbol) or ETF_SECTOR.get(symbol)
        if sector:
            etf_sym = SECTOR_ETF.get(sector) or ETF_CATEGORY_SECTOR_ETF.get(sector)
            if etf_sym:
                etf_df = self._get_price_df(etf_sym)
                if etf_df is not None:
                    etf_sub = etf_df[etf_df.index <= date]
                    if len(etf_sub) >= 200:
                        etf_price = etf_sub['close'].iloc[-1]
                        etf_sma200 = etf_sub['close'].tail(200).mean()
                        sector_sign = 1.0 if etf_price > etf_sma200 else -1.0
                        result['pat_dir_x_sector_trend'] = dir_sign * sector_sign
                    else:
                        result['pat_dir_x_sector_trend'] = np.nan
                else:
                    result['pat_dir_x_sector_trend'] = np.nan
            else:
                result['pat_dir_x_sector_trend'] = np.nan
        else:
            result['pat_dir_x_sector_trend'] = np.nan

        # pat_depth_x_vix: pat_deepest_pass * (1 / vix) -- matches training formula
        deepest = features.get('pat_deepest_pass', np.nan)
        vix = features.get('mkt_vix_level', np.nan)
        if not np.isnan(deepest) and not np.isnan(vix) and vix > 0:
            result['pat_depth_x_vix'] = deepest * (1.0 / vix)
        else:
            result['pat_depth_x_vix'] = np.nan

        # pat_quality_x_regime: pat_sharpe_ratio * mkt_spy_above_sma200 (0 or 1)
        sharpe = features.get('pat_sharpe_ratio', np.nan)
        if not np.isnan(sharpe) and not np.isnan(spy_above):
            result['pat_quality_x_regime'] = sharpe * spy_above
        else:
            result['pat_quality_x_regime'] = np.nan

        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _assemble_features(self, symbol, date, daysOut, dir_char, pat):
        """Combine a supplied pattern profile with the remaining V3 groups."""
        ta = self.compute_technical_features(symbol, date)
        mkt = self.compute_market_regime_features(date)
        ctx = self.compute_stock_context_features(symbol, date)
        cal = self.compute_calendar_features(date)
        spx = self.compute_spx_seasonal_features(date, dir_char)

        # Set trend_direction_match now that we know direction.
        if not np.isnan(ta.get('ta_trend_long', np.nan)):
            if dir_char == 'l':
                if ta['ta_trend_long'] >= 60:
                    ta['ta_trend_direction_match'] = 1
                elif ta['ta_trend_long'] <= 40:
                    ta['ta_trend_direction_match'] = -1
                else:
                    ta['ta_trend_direction_match'] = 0
            else:
                if ta['ta_trend_short'] >= 60:
                    ta['ta_trend_direction_match'] = 1
                elif ta['ta_trend_short'] <= 40:
                    ta['ta_trend_direction_match'] = -1
                else:
                    ta['ta_trend_direction_match'] = 0

        features = {}
        features.update(pat)
        features.update(ta)
        features.update(mkt)
        features.update(ctx)
        features.update(cal)
        features.update(spx)
        features.update(self.compute_interaction_features(
            features, symbol, date, daysOut, dir_char))
        return features

    def compute_recalculated_features(self, resource_id, symbol, date,
                                      daysOut, direction):
        """Compute 62 V3 inputs with a freshly rebuilt exact-horizon profile."""
        if isinstance(date, str):
            date = pd.Timestamp(date)
        dir_char = direction[0].lower()
        profile, profile_meta = self.compute_recalculated_pattern_profile(
            resource_id, symbol, date, daysOut, dir_char)
        price_df = self._get_price_df(symbol)

        pat = dict(profile)
        pat['pat_daysOut'] = int(daysOut)
        active_pairs_by_date = profile_meta.get('_active_pairs_by_date') or {
            str(date)[:10]: set(profile_meta.get('_active_pairs', set()))
        }
        pat['pat_concurrent_count'] = self._context_concurrent_count(
            active_pairs_by_date, int(daysOut))

        neighborhood = self._compute_neighborhood_features(
            symbol, date, int(daysOut), dir_char, price_df)
        pat.update(neighborhood)
        pat['pat_hit_last_year'] = self._compute_hit_last_year(
            symbol, date, int(daysOut), dir_char, price_df)

        return (
            self._assemble_features(symbol, date, int(daysOut), dir_char, pat),
            profile_meta,
        )

    def compute_features(self, symbol, date, daysOut, direction):
        """
        Compute all features for a single opportunity (V1 + V2).

        Args:
            symbol: ticker (e.g., 'AAPL')
            date: entry date (str or Timestamp)
            daysOut: holding period in days
            direction: 'l' for long, 's' for short

        Returns:
            dict of feature_name -> value
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)
        dir_char = direction[0].lower()

        # Ensure price data loaded
        self.load_price_data([symbol])

        # Compute the legacy prebuilt pattern profile.  /score/context uses the
        # additive resource-aware recalculation method above instead.
        pat = self.compute_pattern_features(symbol, date, daysOut, dir_char)
        return self._assemble_features(symbol, date, daysOut, dir_char, pat)

    def compute_label(self, symbol, date, daysOut, direction):
        """
        Compute the actual outcome label for backtesting/training.
        Returns (hit_target, actual_return) or (None, None) if data unavailable.
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)
        dir_char = direction[0].lower()

        df = self._get_price_df(symbol)
        if df is None:
            return None, None

        entry_price = self._get_price_on_date(df, date)
        exit_date = date + pd.Timedelta(days=daysOut)
        exit_price = self._get_price_on_date(df, exit_date)

        if entry_price is None or exit_price is None or entry_price == 0:
            return None, None

        actual_return = (exit_price - entry_price) / entry_price * 100  # percent
        if dir_char == 's':
            actual_return = -actual_return

        hit_target = 1 if actual_return > 0 else 0
        return hit_target, actual_return

    def get_feature_names(self):
        """Return ordered list of all feature names (V1 + V2)."""
        return [
            # Group 1: Pattern-Intrinsic (23 V1)
            'pat_sharpe_ratio', 'pat_avg_profit', 'pat_median_profit', 'pat_avg_profit2',
            'pat_mfe_ratio', 'pat_profit_per_day', 'pat_sharpe_per_day',
            'pat_daysOut', 'pat_daysOut_bucket', 'pat_direction',
            'pat_data_years', 'pat_deepest_pass', 'pat_depth_utilization',
            'pat_passes_at_max_depth', 'pat_passes_recent_10', 'pat_recent_vs_deep_sharpe',
            'pat_num_combos_qualifying',
            'pat_pe_match', 'pat_pe_deepest', 'pat_pe_utilization',
            'pat_best_winrate', 'pat_worst_winrate', 'pat_deepest_pass_capped30',
            # V2 Pattern (8)
            'pat_consistency_std', 'pat_concurrent_count',
            'pat_neighbor_avg_wr', 'pat_sharpness', 'pat_pre_slope', 'pat_post_cliff',
            'pat_hit_last_year',
            # Group 2: Technical Momentum (20)
            'ta_price_vs_sma20', 'ta_price_vs_sma50', 'ta_price_vs_sma200',
            'ta_sma20_vs_sma50', 'ta_sma50_vs_sma200',
            'ta_roc_5', 'ta_roc_20', 'ta_rsi_14', 'ta_macd_histogram', 'ta_macd_hist_slope',
            'ta_rvol_20', 'ta_obv_slope_10', 'ta_volume_price_confirm',
            'ta_atr_percentile', 'ta_bollinger_position',
            'ta_rs_vs_spy_20d',
            'ta_trend_long', 'ta_trend_short', 'ta_trend_delta', 'ta_trend_direction_match',
            # Group 3: Market Regime (14 V1 + 4 V2)
            'mkt_vix_level', 'mkt_vix_percentile_60d', 'mkt_vix_5d_change', 'mkt_vix_term_structure',
            'mkt_yield_curve_10y2y', 'mkt_yield_curve_slope',
            'mkt_credit_spread', 'mkt_credit_spread_change_20d',
            'mkt_spy_roc_20', 'mkt_spy_above_sma200',
            'mkt_spy_breadth_approx', 'mkt_adv_decl_ratio_10d',
            'mkt_new_highs_lows_10d', 'mkt_sector_rotation',
            # V2 Market Regime
            'mkt_vix_regime_bucket', 'mkt_breadth_momentum',
            'mkt_fed_rate_level', 'mkt_fed_rate_direction',
            # SPX Seasonal (4)
            'mkt_spx_seasonal_wr', 'mkt_spx_seasonal_ret',
            'mkt_spx_seasonal_regime', 'mkt_spx_dir_alignment',
            # Group 4: Stock-Specific Context (9)
            'ctx_pct_from_52w_high', 'ctx_pct_from_52w_low', 'ctx_position_in_52w_range',
            'ctx_avg_volume_20d', 'ctx_market_cap_bucket',
            'ctx_days_to_earnings', 'ctx_near_earnings',
            'ctx_sector_rs_20d', 'ctx_stock_vs_sector_20d',
            # Group 5: Calendar/Cycle (7)
            'cal_month', 'cal_day_of_year', 'cal_quarter', 'cal_week_of_month',
            'cal_is_opex_week', 'cal_month_position', 'cal_pe_year',
            # V2 Interaction (4)
            'pat_dir_x_mkt_trend', 'pat_dir_x_sector_trend',
            'pat_depth_x_vix', 'pat_quality_x_regime',
        ]


# ------------------------------------------------------------------
# CLI test
# ------------------------------------------------------------------

if __name__ == '__main__':
    import time
    engine = FeatureEngine()

    # Test with a real opportunity
    symbol = 'AAPL'
    date = '2025-01-14'
    daysOut = 20
    direction = 'l'

    print(f"Computing features for {symbol} {date} {daysOut}d {direction}...")
    t0 = time.time()
    features = engine.compute_features(symbol, date, daysOut, direction)
    t1 = time.time()

    print(f"\nComputed {len(features)} features in {t1-t0:.2f}s\n")

    # Print by group
    groups = [
        ('Pattern-Intrinsic', 'pat_'),
        ('Technical Momentum', 'ta_'),
        ('Market Regime', 'mkt_'),
        ('Stock Context', 'ctx_'),
        ('Calendar/Cycle', 'cal_'),
    ]
    for group_name, prefix in groups:
        group_feats = {k: v for k, v in features.items() if k.startswith(prefix)}
        print(f"--- {group_name} ({len(group_feats)} features) ---")
        for k, v in sorted(group_feats.items()):
            if isinstance(v, float):
                print(f"  {k:40s} = {v:.4f}" if not np.isnan(v) else f"  {k:40s} = NaN")
            else:
                print(f"  {k:40s} = {v}")
        print()

    # Verify count
    expected_names = engine.get_feature_names()
    print(f"Expected: {len(expected_names)} features")
    print(f"Computed: {len(features)} features")

    # Check label
    hit, ret = engine.compute_label(symbol, date, daysOut, direction)
    if hit is not None:
        print(f"\nLabel: hit_target={hit}, actual_return={ret:.2f}%")

    # Missing features
    missing = set(expected_names) - set(features.keys())
    extra = set(features.keys()) - set(expected_names)
    if missing:
        print(f"\nMISSING features: {missing}")
    if extra:
        print(f"\nEXTRA features: {extra}")
