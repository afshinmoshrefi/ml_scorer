"""
TRAIN/SERVE PARITY HARNESS
==========================

Quantifies, per feature, how much the PRODUCTION serving feature engine
(ml_scorer/feature_engine.py) diverges from the TRAINING-TIME feature values
(features/training_data_*.parquet) for the SAME opportunity.

Why this matters
----------------
Training features are computed vectorized with reindex + forward-fill onto a
master calendar (build_training_data.py). The serving engine recomputes per
request with tail()/iloc slicing. These two code paths can silently disagree
and degrade live ML scores. Nothing else in the repo checks this.

What it does
------------
For each tier parquet:
  1. Sample N rows (default 300), STRATIFIED across calendar years and symbols
     with a fixed seed so the sample spans 2000-2025 and many tickers.
  2. For each sampled row, call
        FeatureEngine().compute_features(symbol, date, daysOut, direction)
     and compare the serving value against the parquet (training) value for
     each of the 62 FEATURE_COLS.
  3. Aggregate per feature: n_compared, n_mismatch, mismatch_rate,
     n_nan_mismatch, max/mean abs diff, max rel diff, and worst-case examples.
  4. Save results/experiments/parity_report_<tier>.csv and print the worst
     ~15 features per tier plus an overall summary.

This is a READ-ONLY diagnostic. It imports the production engine and calls it;
it never modifies production code.

Usage
-----
  python tools/parity_test.py                 # all tiers, 300 rows each
  python tools/parity_test.py --n 150         # 150 rows each
  python tools/parity_test.py --tier 10_30    # single tier
  python tools/parity_test.py --seed 7        # different sample

Comparison rule (per feature, per row)
--------------------------------------
  train value = parquet cell; serve value = engine dict (missing key => NaN)
  - both NaN                  => EQUAL
  - exactly one NaN           => MISMATCH (nan_mismatch)
  - both finite:
        equal if abs(a-b) <= ABS_TOL  OR  abs(a-b) <= REL_TOL * max(|a|,|b|)
        else MISMATCH (records abs diff + rel diff)
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

# Make the production package importable regardless of cwd.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml_scorer.config import FEATURE_COLS, DATA_DIR  # the 62 production features
import ml_scorer.feature_engine as fe_module
from ml_scorer.feature_engine import FeatureEngine


def ensure_opp_dir_resolves():
    """Point the engine at the real opp_by_symbol directory if the configured one
    is absent on this machine.

    Production uses DATA_DIR/sp500/opp_by_symbol (correct on the prod box). The
    Windows dev machine stores it at DATA_DIR/sp500_/opp_by_symbol (trailing
    underscore). config.py already rewrites ML_PARQUET_MARKETS for the parquet
    path, but the gzip fallback in feature_engine reads the module-level
    OPP_BY_SYMBOL_DIR, which still points at the non-underscore path. If we left
    it unresolved, EVERY pattern feature would come back NaN purely because the
    opp files can't be found -- an environment artifact, not a real train/serve
    divergence. We monkeypatch the engine module's global (NOT the production
    source) so the gzip scan finds the files and pattern parity is genuinely
    exercised.

    Returns the path actually used (or None if no candidate exists).
    """
    configured = fe_module.OPP_BY_SYMBOL_DIR
    if os.path.isdir(configured):
        return configured
    # Try the dev-machine variant: sp500_ instead of sp500.
    for variant in [
        os.path.join(DATA_DIR, 'sp500_', 'opp_by_symbol'),
        os.path.join(DATA_DIR, 'sp500', 'opp_by_symbol'),
    ]:
        if os.path.isdir(variant):
            fe_module.OPP_BY_SYMBOL_DIR = variant
            print(f'[setup] OPP_BY_SYMBOL_DIR repointed to existing dir: {variant}')
            return variant
    print(f'[setup] WARNING: no opp_by_symbol dir found (configured={configured}). '
          f'Pattern features will be NaN at serve time.')
    return None

# ---------------------------------------------------------------------------
# Tolerances (match the spec)
# ---------------------------------------------------------------------------
ABS_TOL = 1e-6
REL_TOL = 1e-4

TIER_PARQUETS = {
    '10_30': os.path.join(REPO_ROOT, 'features', 'training_data_10_30.parquet'),
    '31_60': os.path.join(REPO_ROOT, 'features', 'training_data_31_60.parquet'),
    '61_90': os.path.join(REPO_ROOT, 'features', 'training_data_61_90.parquet'),
}

OUT_DIR = os.path.join(REPO_ROOT, 'results', 'experiments')

# Sampling guardrails. Training data spans ~2000-2025; keep it simple and just
# trust the parquet years (production price/opp data covers this range).
MIN_YEAR = 2000
MAX_YEAR = 2025

META_COLS = ['date', 'symbol', 'direction', 'daysOut',
             'actual_return', 'hit_target', 'mfe_return']


# ---------------------------------------------------------------------------
# Opp-file reference-year remapping
# ---------------------------------------------------------------------------
# WHY THIS IS NEEDED (and why it is faithful, not a cheat):
#
# The opp_by_symbol files contain ONE row per (month-day, daysOut, direction)
# for the upcoming REFERENCE year only -- every `date` value is e.g.
# 2026-MM-DD. Training (build_training_data.py) looks the pattern up with the
# reference-year key `(opp_year-MM-DD, daysOut, dir)` and then BROADCASTS the
# resulting pattern features across every historical entry year (2000-2025).
# The price-history pattern features (neighbour win-rates, hit_last_year,
# concurrent_count) are then computed relative to the ACTUAL historical entry
# date.
#
# The production serving engine builds its lookup key from the date it is
# asked to score. In production it is asked to score CURRENT/upcoming-year
# opportunities (2026-MM-DD), so the lookup succeeds. But if we feed it a
# historical entry date (2000-04-24) -- which is what a training row's date is
# -- the lookup finds nothing (the opp file has no 2000 rows), every pattern
# feature returns NaN, and (worse) compute_pattern_features short-circuits
# before it ever computes the price-history pattern features.
#
# To compare the SERVING pattern-feature CODE PATH against the TRAINING values
# for the same opportunity, we make the engine's opp loader behave as it does
# in production: we load the real reference-year combos for the symbol and
# REMAP the date key from `{ref_year}-MM-DD` to the requested historical
# `YYYY-MM-DD` (same month-day). The engine then (a) finds the pattern via the
# real opp data (exactly the rows training used) and (b) proceeds to compute
# neighbour/hit/concurrent features against the historical entry date -- which
# is precisely what training did. Time-sensitive non-pattern features (TA,
# market, calendar, ...) are untouched and still use the historical date.
#
# We patch ONLY the engine instance (its bound _load_opp_files), never the
# production source.
class OppRemapper:
    """Bound replacement for FeatureEngine._load_opp_files that remaps the opp
    files' reference-year date keys onto the requested historical date."""

    def __init__(self, engine):
        self.engine = engine
        self._raw_cache = {}        # symbol -> (ref_year:str, combos_by_md)
        # combos_by_md: {combo_name: {(MM-DD, daysOut, dir): row}}
        self._target = None         # (daysOut, dir) of the row currently scored

    def set_target(self, days_out, direction):
        self._target = (int(days_out), direction[0].lower())

    def _load_raw(self, symbol):
        """Load the symbol's combos once via the engine's real gzip scan with a
        date_hint inside the reference year (so all 5 scan dates hit real rows),
        then index them by month-day for fast remap."""
        if symbol in self._raw_cache:
            return self._raw_cache[symbol]

        # Discover the reference year from the symbol's opp files.
        ref_year = self._discover_ref_year(symbol)
        if ref_year is None:
            self._raw_cache[symbol] = (None, {})
            return self._raw_cache[symbol]

        # Use the engine's own gzip loader but with search_dates=None so it loads
        # ALL dates for the symbol (one ref year only -> small), giving us the
        # full pattern set. We call the private gzip method directly.
        combos = self.engine._load_opp_from_gzip(symbol, date_str=None)
        combos_by_md = {}
        for combo_name, lookup in combos.items():
            md_lookup = {}
            for (d, do, dr), row in lookup.items():
                md = d[5:10]  # 'MM-DD'
                md_lookup[(md, do, dr)] = row
            combos_by_md[combo_name] = md_lookup
        self._raw_cache[symbol] = (ref_year, combos_by_md)
        return self._raw_cache[symbol]

    def _discover_ref_year(self, symbol):
        import os, gzip
        opp_dir = os.path.join(fe_module.OPP_BY_SYMBOL_DIR, symbol)
        if not os.path.isdir(opp_dir):
            return None
        for fname in os.listdir(opp_dir):
            if fname.endswith('.csv.gz'):
                path = os.path.join(opp_dir, fname)
                try:
                    with gzip.open(path, 'rt') as gz:
                        header = gz.readline().strip().split(',')
                        di = header.index('date')
                        first = gz.readline().strip().split(',')
                        return first[di][:4]
                except Exception:
                    continue
        return None

    def __call__(self, symbol, date_hint=None):
        """Return combos keyed by the requested date (remapped from ref year).

        Training derives the actual entry date by forward-searching up to 3
        calendar days from the pattern's defining month-day (to land on a
        trading day). So a parquet row dated 2020-11-30 may correspond to a
        pattern defined at month-day 11-27 (Fri) when 28/29 were a weekend.
        We therefore accept any pattern whose defining month-day is the entry
        date minus 0..3 days, preferring the closest (exact day first). This
        mirrors training's forward trading-day adjustment and avoids false
        NaN-mismatches from month-day drift.
        """
        ref_year, combos_by_md = self._load_raw(symbol)
        if not combos_by_md:
            return {}
        date_str = str(date_hint)[:10] if date_hint is not None else None
        if date_str is None:
            return {}
        target = pd.Timestamp(date_str)
        # Candidate defining month-days, ordered by closeness to the entry date.
        candidate_mds = [(target - pd.Timedelta(days=off)).strftime('%m-%d')
                         for off in range(0, 4)]

        # Pick the SINGLE defining month-day to remap: the closest candidate that
        # actually contains the (daysOut, direction) being scored. Remapping just
        # one month-day (instead of a 4-day window) keeps pat_concurrent_count
        # realistic -- it counts only the patterns that share that one defining
        # day, matching how training places a pattern on one resolved entry date.
        chosen_md = None
        if self._target is not None:
            tdo, tdir = self._target
            for md in candidate_mds:  # closest first
                hit = any((cmd == md and do == tdo and dr == tdir)
                          for md_lookup in combos_by_md.values()
                          for (cmd, do, dr) in md_lookup)
                if hit:
                    chosen_md = md
                    break
        if chosen_md is None:
            # No exact (daysOut,dir) match on any candidate day. Fall back to the
            # closest candidate day that has ANY pattern, so broadcast features
            # can still resolve where possible.
            for md in candidate_mds:
                hit = any(cmd == md for md_lookup in combos_by_md.values()
                          for (cmd, _do, _dr) in md_lookup)
                if hit:
                    chosen_md = md
                    break
        if chosen_md is None:
            return {}

        out = {}
        for combo_name, md_lookup in combos_by_md.items():
            remapped = {(date_str, do, dr): row
                        for (cmd, do, dr), row in md_lookup.items()
                        if cmd == chosen_md}
            if remapped:
                out[combo_name] = remapped
        return out


def install_opp_remapper(engine):
    """Bind an OppRemapper as the engine's _load_opp_files. Returns the remapper."""
    remap = OppRemapper(engine)
    engine._load_opp_files = remap  # bound override on the instance
    return remap


# ---------------------------------------------------------------------------
# Sampling: stratified across (year, symbol)
# ---------------------------------------------------------------------------
def stratified_sample(df, n, seed):
    """Sample ~n rows spread across calendar years and symbols.

    Strategy: allocate the budget evenly across the available years, then
    within each year pick rows trying to maximise distinct symbols. Deterministic
    given seed.

    The returned DataFrame KEEPS the original positional index of `df` (its rows
    map 1:1 to row positions in the source parquet), so callers can fetch the
    feature columns for exactly these positions without loading the whole file.
    """
    rng = np.random.RandomState(seed)
    df = df[(df['date'].dt.year >= MIN_YEAR) & (df['date'].dt.year <= MAX_YEAR)].copy()
    df['_year'] = df['date'].dt.year
    years = sorted(df['_year'].unique())
    if not years:
        return df.head(0).drop(columns=['_year'], errors='ignore')

    per_year = max(1, n // len(years))
    picks = []
    for yr in years:
        yr_df = df[df['_year'] == yr]
        # Shuffle deterministically (preserves original index labels), then take
        # rows favouring symbol diversity: one-per-symbol first, then fill.
        yr_df = yr_df.sample(frac=1.0, random_state=rng)
        first_per_symbol = yr_df.drop_duplicates(subset='symbol', keep='first')
        if len(first_per_symbol) >= per_year:
            picks.append(first_per_symbol.head(per_year))
        else:
            remainder = per_year - len(first_per_symbol)
            rest = yr_df.drop(first_per_symbol.index, errors='ignore').head(remainder)
            picks.append(pd.concat([first_per_symbol, rest]))

    out = pd.concat(picks)  # keep original index labels (= parquet row positions)
    # If we overshot the target, trim deterministically (keep index labels).
    if len(out) > n:
        out = out.sample(n=n, random_state=rng)
    out = out.drop(columns=['_year'], errors='ignore')
    return out


# ---------------------------------------------------------------------------
# Per-feature comparison
# ---------------------------------------------------------------------------
def values_equal(a, b):
    """Return (is_equal, kind, abs_diff, rel_diff).

    kind in {'equal', 'nan_mismatch', 'value_mismatch'}.
    abs_diff / rel_diff are NaN when not both finite.
    """
    a_nan = (a is None) or (isinstance(a, float) and np.isnan(a)) or (pd.isna(a))
    b_nan = (b is None) or (isinstance(b, float) and np.isnan(b)) or (pd.isna(b))

    if a_nan and b_nan:
        return True, 'equal', np.nan, np.nan
    if a_nan != b_nan:
        return False, 'nan_mismatch', np.nan, np.nan

    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        # Non-numeric: fall back to direct equality.
        eq = (a == b)
        return eq, ('equal' if eq else 'value_mismatch'), np.nan, np.nan

    abs_diff = abs(a - b)
    denom = max(abs(a), abs(b))
    rel_diff = (abs_diff / denom) if denom > 0 else 0.0
    if abs_diff <= ABS_TOL or abs_diff <= REL_TOL * denom:
        return True, 'equal', abs_diff, rel_diff
    return False, 'value_mismatch', abs_diff, rel_diff


class FeatureStats:
    """Accumulator for one feature across all sampled rows."""

    __slots__ = ('name', 'n_compared', 'n_mismatch', 'n_nan_mismatch',
                 'max_abs_diff', 'sum_abs_diff', 'n_abs_diff', 'max_rel_diff',
                 'examples')

    def __init__(self, name):
        self.name = name
        self.n_compared = 0
        self.n_mismatch = 0
        self.n_nan_mismatch = 0
        self.max_abs_diff = 0.0
        self.sum_abs_diff = 0.0
        self.n_abs_diff = 0          # how many mismatches had a finite abs diff
        self.max_rel_diff = 0.0
        self.examples = []           # list of (severity, symbol, date, train, serve, kind)

    def update(self, is_equal, kind, abs_diff, rel_diff, symbol, date, train_v, serve_v):
        self.n_compared += 1
        if is_equal:
            return
        self.n_mismatch += 1
        if kind == 'nan_mismatch':
            self.n_nan_mismatch += 1
            severity = np.inf  # treat as most severe for example ranking
        else:
            severity = abs_diff if not np.isnan(abs_diff) else 0.0
            if not np.isnan(abs_diff):
                self.max_abs_diff = max(self.max_abs_diff, abs_diff)
                self.sum_abs_diff += abs_diff
                self.n_abs_diff += 1
            if not np.isnan(rel_diff):
                self.max_rel_diff = max(self.max_rel_diff, rel_diff)
        self.examples.append((severity, symbol, str(date)[:10], train_v, serve_v, kind))

    def top_examples(self, k=3):
        # Worst-first: nan_mismatch (severity == inf) first, then largest abs diff.
        nan_ex = [e for e in self.examples if e[0] == np.inf]
        val_ex = sorted([e for e in self.examples if e[0] != np.inf], key=lambda e: -e[0])
        ordered = nan_ex + val_ex
        return ordered[:k]

    def as_row(self):
        mismatch_rate = (self.n_mismatch / self.n_compared) if self.n_compared else np.nan
        mean_abs = (self.sum_abs_diff / self.n_abs_diff) if self.n_abs_diff else 0.0
        ex = self.top_examples(3)
        row = {
            'feature': self.name,
            'n_compared': self.n_compared,
            'n_mismatch': self.n_mismatch,
            'mismatch_rate': round(mismatch_rate, 4) if not np.isnan(mismatch_rate) else np.nan,
            'n_nan_mismatch': self.n_nan_mismatch,
            'max_abs_diff': self.max_abs_diff,
            'mean_abs_diff': mean_abs,
            'max_rel_diff': self.max_rel_diff,
        }
        for i in range(3):
            if i < len(ex):
                sev, sym, dt, tv, sv, kind = ex[i]
                row[f'ex{i+1}'] = f'{sym}@{dt} train={_fmt(tv)} serve={_fmt(sv)} ({kind})'
            else:
                row[f'ex{i+1}'] = ''
        return row


def _fmt(v):
    if v is None:
        return 'None'
    try:
        if pd.isna(v):
            return 'NaN'
    except (TypeError, ValueError):
        pass
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v)


def _read_features_at_positions(path, feat_cols, positions):
    """Read `feat_cols` for the given global row positions, scanning only the
    row groups that actually contain a sampled position.

    Returns a DataFrame indexed by the global row position. Keeps memory low: we
    never materialise the full feature matrix, only the row groups we need.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    positions = np.asarray(positions)
    pos_set = set(int(p) for p in positions)

    frames = []
    row_start = 0
    for rg in range(pf.num_row_groups):
        rg_rows = pf.metadata.row_group(rg).num_rows
        row_end = row_start + rg_rows
        # Which sampled positions fall in this row group?
        local = [p - row_start for p in positions if row_start <= p < row_end]
        if local:
            tbl = pf.read_row_group(rg, columns=feat_cols)
            sub = tbl.to_pandas()
            sub = sub.iloc[local]
            sub.index = [row_start + li for li in local]
            frames.append(sub)
        row_start = row_end
    if not frames:
        return pd.DataFrame(columns=feat_cols)
    out = pd.concat(frames)
    # Sanity: we should have exactly the requested positions.
    missing = pos_set - set(out.index.tolist())
    if missing:
        # Should not happen, but guard against silent row-count drift.
        print(f'  [warn] {len(missing)} sampled positions not located in row groups')
    return out


# ---------------------------------------------------------------------------
# Run one tier
# ---------------------------------------------------------------------------
def run_tier(tier, n, seed):
    path = TIER_PARQUETS[tier]
    if not os.path.exists(path):
        print(f'[{tier}] parquet not found: {path} -- SKIPPING')
        return None

    print(f'\n{"="*78}\nTIER {tier}\n{"="*78}')
    import pyarrow.parquet as pq
    avail = set(pq.read_schema(path).names)

    # Two-stage load to avoid pulling the full 2GB feature matrix into memory:
    #   1. Read only the light meta columns (date/symbol/direction/daysOut) and
    #      stratified-sample N rows from them.
    #   2. Read the 62 feature columns ONLY for the sampled row positions.
    meta_present = [c for c in ['date', 'symbol', 'direction', 'daysOut'] if c in avail]
    print(f'Reading meta columns {meta_present} to sample {n} rows ...')
    meta_df = pd.read_parquet(path, columns=meta_present)
    print(f'  {len(meta_df):,} total rows')

    sample = stratified_sample(meta_df, n, seed)
    del meta_df
    # Original parquet row positions for the sampled rows (sorted for chunked read).
    sample_positions = np.sort(sample.index.to_numpy())
    feat_present = [c for c in FEATURE_COLS if c in avail]
    print(f'  fetching {len(feat_present)} feature columns for {len(sample)} sampled rows '
          f'(chunked by row group) ...')
    feat_sample = _read_features_at_positions(path, feat_present, sample_positions)
    # Align: sample currently indexed by original position; reindex features to match.
    feat_sample = feat_sample.loc[sample.index]
    sample = sample.reset_index(drop=True)
    feat_sample = feat_sample.reset_index(drop=True)
    # Attach the feature values back onto the sample (meta + features side by side).
    sample = pd.concat([sample, feat_sample], axis=1)
    yr_counts = sample['date'].dt.year.value_counts().sort_index()
    print(f'  sampled {len(sample)} rows across {sample["date"].dt.year.nunique()} years '
          f'and {sample["symbol"].nunique()} symbols (seed={seed})')
    print(f'  per-year counts: {dict(yr_counts)}')

    engine = FeatureEngine()
    # Remap opp reference-year keys onto the historical entry date so the serving
    # pattern-feature path is exercised the way it is in production (see the long
    # note on OppRemapper). Without this, all 23 pattern features falsely report
    # NaN-mismatch because the opp files contain only reference-year dates.
    remapper = install_opp_remapper(engine)
    stats = {f: FeatureStats(f) for f in FEATURE_COLS}

    n_engine_error = 0
    n_engine_empty = 0
    engine_error_examples = []
    n_ok = 0

    t_start = pd.Timestamp.now()
    for i, row in enumerate(sample.itertuples(index=False)):
        rd = row._asdict()
        symbol = rd['symbol']
        date_str = str(rd['date'])[:10]
        days_out = int(rd['daysOut'])
        direction = rd['direction']

        remapper.set_target(days_out, direction)
        try:
            serve = engine.compute_features(symbol, date_str, days_out, direction)
        except Exception as e:  # engine blew up -- itself a reliability finding
            n_engine_error += 1
            if len(engine_error_examples) < 5:
                engine_error_examples.append(
                    f'{symbol}@{date_str} d={days_out} {direction}: '
                    f'{type(e).__name__}: {e}')
            continue

        if not serve:  # engine returned nothing / empty dict
            n_engine_empty += 1
            if len(engine_error_examples) < 5:
                engine_error_examples.append(
                    f'{symbol}@{date_str} d={days_out} {direction}: empty result')
            continue

        n_ok += 1
        for f in FEATURE_COLS:
            train_v = rd[f] if f in rd else np.nan
            serve_v = serve.get(f, np.nan)
            is_eq, kind, ad, rdf = values_equal(train_v, serve_v)
            stats[f].update(is_eq, kind, ad, rdf, symbol, date_str, train_v, serve_v)

        if (i + 1) % 50 == 0:
            elapsed = (pd.Timestamp.now() - t_start).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f'    ...{i+1}/{len(sample)} rows  ({rate:.1f} rows/s, '
                  f'{n_ok} scored, {n_engine_error} errors, {n_engine_empty} empty)')

    elapsed = (pd.Timestamp.now() - t_start).total_seconds()
    print(f'  done in {elapsed:.0f}s: {n_ok} scored, '
          f'{n_engine_error} engine errors, {n_engine_empty} empty results')

    # Build report dataframe sorted worst-first.
    rows = [stats[f].as_row() for f in FEATURE_COLS]
    rep = pd.DataFrame(rows)
    rep = rep.sort_values(['mismatch_rate', 'max_abs_diff'],
                          ascending=[False, False]).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, f'parity_report_{tier}.csv')
    rep.to_csv(out_path, index=False)

    # ---- Print worst ~15 ----
    print(f'\n  WORST 15 FEATURES (of {len(FEATURE_COLS)}) for tier {tier}:')
    hdr = f'  {"feature":28s} {"mm_rate":>8s} {"n_mm":>5s} {"nan_mm":>6s} {"max_abs":>12s} {"max_rel":>9s}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for _, r in rep.head(15).iterrows():
        print(f'  {r["feature"]:28s} {r["mismatch_rate"]:>8.3f} {int(r["n_mismatch"]):>5d} '
              f'{int(r["n_nan_mismatch"]):>6d} {r["max_abs_diff"]:>12.4g} {r["max_rel_diff"]:>9.3g}')
    # Show one representative example for the worst few.
    print('\n  Representative example diffs (worst features):')
    for _, r in rep.head(8).iterrows():
        if r['n_mismatch'] > 0 and r['ex1']:
            print(f'    {r["feature"]:28s} {r["ex1"]}')

    # ---- Clean vs skewed summary ----
    clean = rep[rep['n_mismatch'] == 0]['feature'].tolist()
    skewed = rep[rep['n_mismatch'] > 0]['feature'].tolist()
    print(f'\n  CLEAN within tolerance ({len(clean)}/{len(FEATURE_COLS)}): '
          f'{", ".join(sorted(clean)) if clean else "NONE"}')
    print(f'  SKEWED ({len(skewed)}/{len(FEATURE_COLS)}): '
          f'{", ".join(skewed) if skewed else "NONE"}')

    if engine_error_examples:
        print(f'\n  ENGINE ERROR/EMPTY examples (count={n_engine_error + n_engine_empty}):')
        for e in engine_error_examples:
            print(f'    - {e}')

    print(f'\n  -> report saved: {out_path}')

    return {
        'tier': tier,
        'n_sampled': len(sample),
        'n_scored': n_ok,
        'n_engine_error': n_engine_error,
        'n_engine_empty': n_engine_empty,
        'n_clean': len(clean),
        'n_skewed': len(skewed),
        'out_path': out_path,
        'report': rep,
        'engine_error_examples': engine_error_examples,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Train/serve feature parity harness')
    ap.add_argument('--tier', choices=list(TIER_PARQUETS.keys()), default=None,
                    help='single tier (default: all three)')
    ap.add_argument('--n', type=int, default=300, help='rows per tier (default 300)')
    ap.add_argument('--seed', type=int, default=42, help='sampling seed (default 42)')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    # Make sure the serving engine can find opp files on this machine before we
    # start (otherwise all 23 pattern features falsely report NaN-mismatch).
    ensure_opp_dir_resolves()
    tiers = [args.tier] if args.tier else list(TIER_PARQUETS.keys())

    summaries = []
    for tier in tiers:
        s = run_tier(tier, args.n, args.seed)
        if s:
            summaries.append(s)

    # ---- Overall summary across tiers ----
    print(f'\n{"#"*78}\nOVERALL SUMMARY\n{"#"*78}')
    for s in summaries:
        print(f'  tier {s["tier"]:6s}: {s["n_scored"]}/{s["n_sampled"]} scored | '
              f'CLEAN {s["n_clean"]}/{len(FEATURE_COLS)}  SKEWED {s["n_skewed"]}/{len(FEATURE_COLS)} | '
              f'engine_err={s["n_engine_error"]} empty={s["n_engine_empty"]}')
        print(f'             report: {s["out_path"]}')

    # Features skewed in EVERY tier (systemic divergences worth fixing first).
    if len(summaries) > 1:
        skewed_sets = []
        for s in summaries:
            sk = set(s['report'][s['report']['n_mismatch'] > 0]['feature'])
            skewed_sets.append(sk)
        common = set.intersection(*skewed_sets) if skewed_sets else set()
        if common:
            print(f'\n  Features SKEWED in ALL {len(summaries)} tiers '
                  f'({len(common)}): {", ".join(sorted(common))}')


if __name__ == '__main__':
    main()
