#!/usr/bin/env python3
"""Preflight checks for an ml_scorer deployment.

Exists because two deployment faults reached a running service unnoticed:

  1. config.py TIERS still named the previous run's model files. That failed
     loudly only because the named files were absent. On a box that still holds
     the older models -- production does -- the wrong models load cleanly and
     the service reports status ok, because the feature-count check passes for
     any two model sets of the same width.

  2. pat_concurrent_count was computed ~6x larger at inference than in
     training. Every live sample sat beyond the training maximum, and the
     tier-31_60 XGBoost member collapsed to a NEGATIVE mean predicted return.
     The retrain gate missed it: it scores ensemble AUC and win-rate, which are
     RANKING measures, so one member's prediction LEVEL can collapse while the
     other two preserve ranking.

Both are silent failures, so "the service started" is not evidence of a good
deployment. These checks make them loud.

    python preflight.py                 # all checks, exit 1 on failure
    python preflight.py --expect-run 20260802,20260803
    python preflight.py --skip-live     # static checks only (no data needed)

Checks:
    models      every TIERS file exists; optionally matches expected run dates
    features    LightGBM/XGBoost stored feature_names match config order
    ranges      live feature values sit inside recorded training bounds
    levels      each ensemble member's mean prediction is sane, not just the
                ensemble average
"""
import argparse
import hashlib
import json
import math
import os
import sys

try:
    import config as C
except ImportError:                                          # pragma: no cover
    import scorer_config as C

BOUNDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'training_bounds.json')

# A member whose mean prediction falls outside this multiple of the member
# median is treated as collapsed. The 31_60 XGBoost failure sat at -0.07x.
LEVEL_LO, LEVEL_HI = 0.35, 2.5
# Fraction of sampled opportunities allowed outside the training range before a
# feature is reported at all.
RANGE_TOLERANCE = 0.25
# How far outside, measured against the width of the training range, separates
# genuine drift from a train/serve mismatch. Credit spreads drifting 1.5% past
# the training max is the market moving; pat_concurrent_count sitting 51% past
# it was a computation mismatch. Anything under this is a warning, not a
# failure, so ordinary macro drift does not block a deploy.
RANGE_EXCURSION_FAIL = 0.15


class Result:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def fail(self, check, msg):
        self.failures.append((check, msg))
        print(f'  FAIL  [{check}] {msg}')

    def warn(self, check, msg):
        self.warnings.append((check, msg))
        print(f'  warn  [{check}] {msg}')

    def ok(self, check, msg):
        print(f'  ok    [{check}] {msg}')


def check_models(r, expect_runs):
    """Every referenced model file exists, and dates are consistent."""
    seen_dates = set()
    missing = []
    for tier, cfg in C.TIERS.items():
        for target in ('sr', 'mfe'):
            for algo, fname in cfg[target].items():
                path = os.path.join(C.MODEL_DIR, fname)
                if not os.path.exists(path):
                    missing.append(f'{tier}/{target}/{algo} -> {fname}')
                for token in fname.replace('.', '_').split('_'):
                    if token.isdigit() and len(token) == 8:
                        seen_dates.add(token)
        for key in ('calibration_sr', 'calibration_mfe'):
            path = os.path.join(C.CALIBRATION_DIR, cfg[key])
            if not os.path.exists(path):
                missing.append(f'{tier}/{key} -> {cfg[key]}')

    if missing:
        for m in missing:
            r.fail('models', f'referenced file missing: {m}')
    else:
        r.ok('models', f'all TIERS model + calibration files present '
                       f'(run dates: {",".join(sorted(seen_dates))})')

    if expect_runs:
        unexpected = seen_dates - set(expect_runs)
        if unexpected:
            r.fail('models', f'model dates {sorted(unexpected)} not in expected '
                             f'{sorted(expect_runs)} -- TIERS may be stale')
        else:
            r.ok('models', f'all model dates within expected {sorted(expect_runs)}')


def _lgb_feature_names(path):
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('feature_names='):
                return line.strip().split('=', 1)[1].split()
    return None


def _xgb_feature_names(path):
    with open(path, encoding='utf-8', errors='replace') as fh:
        return (json.load(fh).get('learner', {}) or {}).get('feature_names') or None


def check_feature_order(r):
    """Stored feature_names must match config order.

    scorer.py validates feature COUNT only, so a same-width reordering is
    otherwise silent and produces confident nonsense.
    """
    want = list(C.FEATURE_COLS)
    checked = bad = 0
    for tier, cfg in C.TIERS.items():
        for target in ('sr', 'mfe'):
            for algo, reader in (('lgb', _lgb_feature_names), ('xgb', _xgb_feature_names)):
                fname = cfg[target].get(algo)
                if not fname:
                    continue
                path = os.path.join(C.MODEL_DIR, fname)
                if not os.path.exists(path):
                    continue
                try:
                    names = reader(path)
                except Exception as exc:                      # noqa: BLE001
                    r.warn('features', f'{fname}: unreadable ({exc})')
                    continue
                if not names:
                    r.warn('features', f'{fname}: no feature_names stored -- '
                                       f'positional matching, unvalidated')
                    continue
                checked += 1
                if list(names) != want:
                    bad += 1
                    if len(names) != len(want):
                        r.fail('features', f'{fname}: {len(names)} features, config has {len(want)}')
                    else:
                        idx = next(i for i, (a, b) in enumerate(zip(names, want)) if a != b)
                        r.fail('features', f'{fname}: order differs at index {idx} '
                                           f'(model={names[idx]} config={want[idx]})')
    if checked and not bad:
        r.ok('features', f'{checked} model files match config feature order ({len(want)} features)')


def _sample_opportunities(tier, lo, hi, n, date):
    import pandas as pd
    path = os.path.join(C.DATA_DIR, 'sp500', f'ml_cache_{date}.parquet')
    if not os.path.exists(path) or n <= 0:
        return []
    df = pd.read_parquet(path)
    df = df[(df['date'].astype(str) == date) &
            (df['daysOut'] >= lo) & (df['daysOut'] <= hi)]
    identities = {
        (str(r.sym), int(r.daysOut), str(r.LorS))
        for r in df.itertuples()
    }
    if not identities:
        return []

    def stable_rank(item):
        payload = '\x1f'.join((item[0], str(item[1]), item[2])).encode('utf-8')
        return hashlib.sha256(payload).digest()

    groups = {}
    for item in identities:
        groups.setdefault(item[2], []).append(item)
    for items in groups.values():
        items.sort(key=stable_rank)

    target = min(n, len(identities))
    total = len(identities)
    raw_quotas = {
        direction: target * len(items) / total
        for direction, items in groups.items()
    }
    quotas = {
        direction: min(len(groups[direction]), int(raw_quotas[direction]))
        for direction in groups
    }

    # Direction is a learned feature. Preserve its live population mix while
    # guaranteeing coverage of every present direction when the sample allows
    # it. Parquet row order is not a sampling contract and is commonly grouped
    # by direction, which made the former head(n) gate evaluate all-long rows.
    minimum = 1 if target >= len(groups) else 0
    if minimum:
        for direction in quotas:
            quotas[direction] = max(minimum, quotas[direction])

    while sum(quotas.values()) > target:
        candidates = [
            direction for direction, quota in quotas.items()
            if quota > minimum
        ]
        direction = max(
            candidates,
            key=lambda name: (
                quotas[name] - raw_quotas[name], quotas[name], name),
        )
        quotas[direction] -= 1

    while sum(quotas.values()) < target:
        candidates = [
            direction for direction, quota in quotas.items()
            if quota < len(groups[direction])
        ]
        direction = max(
            candidates,
            key=lambda name: (
                raw_quotas[name] - quotas[name],
                len(groups[name]) - quotas[name],
                name,
            ),
        )
        quotas[direction] += 1

    selected = []
    for direction in sorted(groups):
        selected.extend(groups[direction][:quotas[direction]])
    return sorted(selected, key=stable_rank)


def check_live(r, date, n):
    """Feature ranges and per-member prediction levels on live data."""
    import numpy as np
    from feature_engine import FeatureEngine

    if not os.path.exists(BOUNDS_PATH):
        r.warn('ranges', f'{os.path.basename(BOUNDS_PATH)} absent -- range check skipped')
        bounds = {}
    else:
        with open(BOUNDS_PATH, encoding='utf-8') as fh:
            bounds = json.load(fh).get('tiers', {})

    tiers = (('10_30', 10, 30), ('31_60', 31, 60), ('61_90', 61, 90))
    cols = list(C.FEATURE_COLS)
    engine = FeatureEngine()

    for tier, lo, hi in tiers:
        opps = _sample_opportunities(tier, lo, hi, n, date)
        if not opps:
            r.warn('ranges', f'{tier}: no opportunities for {date} -- skipped')
            continue
        directions = {name: 0 for name in sorted({item[2] for item in opps})}
        for _, _, direction in opps:
            directions[direction] += 1
        r.ok('sample', f'{tier}: deterministic representative sample '
                        f'({len(opps)} opportunities; directions={directions}; '
                        f'days={min(item[1] for item in opps)}-'
                        f'{max(item[1] for item in opps)})')
        engine.load_price_data(sorted({s for s, _, _ in opps}))
        X = []
        for sym, days, dirn in opps:
            try:
                f = engine.compute_features(sym, date, days, dirn)
                X.append([f.get(c, np.nan) for c in cols])
            except Exception:                                 # noqa: BLE001
                pass
        if not X:
            r.warn('ranges', f'{tier}: no features computed -- skipped')
            continue
        X = np.array(X, dtype=float)

        tb = bounds.get(tier, {})
        offenders = []
        for i, name in enumerate(cols):
            b = tb.get(name)
            col = X[:, i]
            col = col[np.isfinite(col)]
            if not b or col.size == 0:
                continue
            out = ((col < b['min']) | (col > b['max'])).mean()
            if out > RANGE_TOLERANCE:
                width = max(b['max'] - b['min'], 1e-9)
                excursion = max(float(col.max()) - b['max'], b['min'] - float(col.min()), 0.0) / width
                offenders.append((name, out, excursion,
                                  float(col.min()), float(col.max()), b['min'], b['max']))
        if offenders:
            for name, frac, exc, lo_v, hi_v, bmin, bmax in sorted(offenders, key=lambda x: -x[2]):
                msg = (f'{tier}: {name} {frac:.0%} of samples outside training '
                       f'[{bmin:.3f},{bmax:.3f}] (live [{lo_v:.3f},{hi_v:.3f}], '
                       f'{exc:.0%} of range width beyond)')
                if exc >= RANGE_EXCURSION_FAIL:
                    r.fail('ranges', msg + ' -- likely train/serve mismatch')
                else:
                    r.warn('ranges', msg + ' -- small, likely genuine drift')
        if not any(e[2] >= RANGE_EXCURSION_FAIL for e in offenders):
            r.ok('ranges', f'{tier}: all {len(tb)} features within training bounds '
                           f'({X.shape[0]} opportunities)')

        _check_levels(r, tier, X, cols, tb)


def _check_levels(r, tier, X, cols, tb):
    """Each ensemble member's mean prediction, not just the ensemble average."""
    import numpy as np
    try:
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostRegressor
    except ImportError as exc:                                # pragma: no cover
        r.warn('levels', f'{tier}: {exc} -- skipped')
        return

    cfg = C.TIERS[tier]['sr']
    preds = {}
    try:
        m = lgb.Booster(model_file=os.path.join(C.MODEL_DIR, cfg['lgb']))
        preds['lgb'] = float(np.mean(m.predict(X)))
        b = xgb.Booster(); b.load_model(os.path.join(C.MODEL_DIR, cfg['xgb']))
        preds['xgb'] = float(np.mean(b.predict(xgb.DMatrix(X, feature_names=cols))))
        c = CatBoostRegressor(); c.load_model(os.path.join(C.MODEL_DIR, cfg['catboost']))
        preds['cb'] = float(np.mean(c.predict(X)))
    except Exception as exc:                                  # noqa: BLE001
        r.fail('levels', f'{tier}: could not score members ({exc})')
        return

    members = sorted(preds.values())
    median = members[len(members) // 2]
    bad = False
    for name, val in sorted(preds.items()):
        if val <= 0:
            r.fail('levels', f'{tier}/{name}: mean prediction {val:+.4f} is non-positive '
                             f'-- member has collapsed')
            bad = True
        elif median and not (LEVEL_LO <= val / median <= LEVEL_HI):
            r.fail('levels', f'{tier}/{name}: mean prediction {val:.4f} is {val/median:.2f}x '
                             f'the member median {median:.4f} -- outside [{LEVEL_LO},{LEVEL_HI}]')
            bad = True
    if not bad:
        r.ok('levels', f'{tier}: members consistent '
                       f'({", ".join(f"{k}={v:.3f}" for k, v in sorted(preds.items()))})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expect-run', default='',
                    help='comma-separated run dates the TIERS models must come from, '
                         'e.g. 20260802,20260803')
    ap.add_argument('--date', default=None, help='scoring date for live checks')
    ap.add_argument('--n', type=int, default=120, help='opportunities sampled per tier')
    ap.add_argument('--skip-live', action='store_true', help='static checks only')
    args = ap.parse_args()

    date = args.date
    if date is None:
        import glob
        found = sorted(glob.glob(os.path.join(C.DATA_DIR, 'sp500', 'ml_cache_*.parquet')))
        date = os.path.basename(found[-1])[len('ml_cache_'):-len('.parquet')] if found else None

    print('ml_scorer preflight')
    print(f'  MODEL_DIR = {C.MODEL_DIR}')
    print(f'  DATA_DIR  = {C.DATA_DIR}')
    print(f'  date      = {date}')
    print()

    r = Result()
    expect = [d for d in args.expect_run.split(',') if d]
    check_models(r, expect)
    check_feature_order(r)
    if args.skip_live:
        print('  (live checks skipped)')
    elif date is None:
        r.warn('ranges', 'no ml_cache parquet found -- live checks skipped')
    else:
        check_live(r, date, args.n)

    print()
    if r.failures:
        print(f'PREFLIGHT FAILED: {len(r.failures)} failure(s), {len(r.warnings)} warning(s)')
        return 1
    print(f'PREFLIGHT PASSED ({len(r.warnings)} warning(s))')
    return 0


if __name__ == '__main__':
    sys.exit(main())
