"""
Canonical evaluation utilities + guardrail scoreboard for the ML scorer
improvement loop.

Given a walk-forward out-of-fold predictions parquet (results/wf_predictions_*.parquet)
this computes, from one place so every experiment is comparable:

  - discrimination:   AUC (predicted return as ranking score vs hit_target)
  - calibration:      Brier / log-loss / ECE / MCE of a calibrated win probability
  - tradeable guards: win rate + annualised Sharpe at the ML_70 / ML_85 / ML_90
                      percentile cutoffs, plus the unfiltered baseline

The Sharpe convention (mean/std * sqrt(252)) matches
train_model.evaluate_trading_performance so numbers line up with the baseline
table in CLAUDE.md.

It also provides fit_ensemble_weights() for Phase 1c (learned ensemble
combination) which needs the per-model OOF columns pred_lgb / pred_xgb / pred_cb
that train_model.py now saves.

Pure offline: reads parquet / json only. Never loads price data or models.

Usage:
  python tools/scorer_eval.py --preds results/wf_predictions_sr.parquet --tag baseline_sr_10_30
  python tools/scorer_eval.py --preds results/wf_predictions_sr.parquet --by-year
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'experiments')

ANNUALIZE = np.sqrt(252)


# ----------------------------------------------------------------------
# Metric primitives
# ----------------------------------------------------------------------

def auc(y_hit, score):
    y_hit = np.asarray(y_hit)
    if len(np.unique(y_hit)) < 2:
        return 0.5
    return float(roc_auc_score(y_hit, np.asarray(score)))


def calibration_metrics(prob, y_hit, n_bins=15):
    """Brier, log-loss, ECE, MCE for a probability vector vs binary outcome."""
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    y_hit = np.asarray(y_hit, dtype=float)
    brier = float(np.mean((prob - y_hit) ** 2))
    ll = float(log_loss(y_hit, prob, labels=[0, 1]))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
    n = len(prob)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        conf = float(prob[m].mean())
        acc = float(y_hit[m].mean())
        gap = abs(conf - acc)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
    return {'brier': brier, 'logloss': ll, 'ece': float(ece), 'mce': float(mce)}


def _block(actual_return, hit):
    a = np.asarray(actual_return, dtype=float)
    h = np.asarray(hit, dtype=float)
    sharpe = float(a.mean() / a.std() * ANNUALIZE) if a.std() > 0 else 0.0
    return {
        'n': int(len(a)),
        'win_rate': float(h.mean()),
        'avg_return': float(a.mean()),
        'median_return': float(np.median(a)),
        'sharpe': sharpe,
    }


def trading_metrics(df, score_col='predicted'):
    """Baseline + ML_70/85/90 (percentile of score_col). Matches train_model."""
    out = {'baseline': _block(df['actual_return'].values, df['hit_target'].values)}
    pct = df[score_col].rank(pct=True).values * 100.0
    ar = df['actual_return'].values
    hit = df['hit_target'].values
    for t in (70, 85, 90):
        m = pct >= t
        if m.sum() >= 10:
            out[f'ML_{t}'] = _block(ar[m], hit[m])
    return out


def loyo_isotonic_prob(df, score_col='predicted', year_col='val_year'):
    """Leave-one-year-out isotonic-calibrated win probability (no leakage).

    For each val_year Y, fit isotonic (predicted -> hit_target) on the other
    years and score year Y. Returns a probability array aligned to df.
    """
    prob = np.full(len(df), np.nan)
    years = df[year_col].unique()
    score = df[score_col].values
    hit = df['hit_target'].values
    for y in years:
        te = (df[year_col] == y).values
        tr = ~te
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(score[tr], hit[tr])
        prob[te] = iso.predict(score[te])
    return prob


# ----------------------------------------------------------------------
# Phase 1c: learned ensemble weights
# ----------------------------------------------------------------------

def fit_ensemble_weights(P, df, n_grid=21):
    """Find non-negative weights (sum=1) over per-model preds minimizing
    leave-one-year-out isotonic-calibrated log-loss.

    P: (n, k) array of per-model OOF predictions (k models).
    df: frame with val_year, hit_target (aligned to P rows).
    Returns (best_weights, best_logloss, mean_logloss). Grid search over the
    simplex for k=3 (cheap, robust, no optimiser dependency).
    """
    k = P.shape[1]
    assert k == 3, 'grid simplex search assumes 3 models'
    hit = df['hit_target'].values.astype(float)
    years = df['val_year'].values

    def loyo_ll(weights):
        blended = P @ np.asarray(weights)
        prob = np.full(len(blended), np.nan)
        for y in np.unique(years):
            te = years == y
            tr = ~te
            if tr.sum() < 1000 or te.sum() == 0:
                continue
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            iso.fit(blended[tr], hit[tr])
            prob[te] = iso.predict(blended[te])
        ok = ~np.isnan(prob)
        p = np.clip(prob[ok], 1e-6, 1 - 1e-6)
        return float(log_loss(hit[ok], p, labels=[0, 1]))

    best = None
    grid = np.linspace(0.0, 1.0, n_grid)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            w = np.array([w1, w2, max(w3, 0.0)])
            s = w.sum()
            if s <= 0:
                continue
            w = w / s
            ll = loyo_ll(w)
            if best is None or ll < best[1]:
                best = (w, ll)
    mean_ll = loyo_ll(np.array([1 / 3, 1 / 3, 1 / 3]))
    return best[0], best[1], mean_ll


# ----------------------------------------------------------------------
# Scoreboard
# ----------------------------------------------------------------------

def scoreboard(preds_path, tag=None, by_year=False, n_sub=None):
    t0 = time.time()
    df = pd.read_parquet(preds_path)
    if n_sub and len(df) > n_sub:
        df = df.sample(n_sub, random_state=42).reset_index(drop=True)
    tag = tag or os.path.basename(preds_path).replace('.parquet', '')

    result = {'tag': tag, 'preds_path': preds_path, 'n': int(len(df))}
    result['auc'] = auc(df['hit_target'].values, df['predicted'].values)

    # Honest (LOYO isotonic) calibrated probability for calibration metrics
    prob = loyo_isotonic_prob(df)
    ok = ~np.isnan(prob)
    result['calibration_loyo_isotonic'] = calibration_metrics(prob[ok], df['hit_target'].values[ok])

    result['trading'] = trading_metrics(df)

    if by_year:
        result['by_year'] = {}
        for y, g in df.groupby('val_year'):
            yr = {'n': int(len(g)), 'auc': auc(g['hit_target'].values, g['predicted'].values)}
            yr['trading'] = trading_metrics(g)
            result['by_year'][int(y)] = yr

    result['elapsed_s'] = round(time.time() - t0, 1)

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    out_path = os.path.join(EXPERIMENTS_DIR, f'scoreboard_{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    _print_scoreboard(result)
    print(f"\nSaved -> {out_path}")
    return result


def _print_scoreboard(r):
    print(f"\n{'='*64}\nSCOREBOARD: {r['tag']}  (n={r['n']:,})\n{'='*64}")
    print(f"  AUC                : {r['auc']:.4f}")
    c = r['calibration_loyo_isotonic']
    print(f"  Calibration (LOYO isotonic):")
    print(f"    Brier={c['brier']:.4f}  LogLoss={c['logloss']:.4f}  "
          f"ECE={c['ece']:.4f}  MCE={c['mce']:.4f}")
    print(f"  Trading guardrails:")
    for k, v in r['trading'].items():
        print(f"    {k:9s}: n={v['n']:>9,}  WR={v['win_rate']:.4f}  "
              f"avg_ret={v['avg_return']:6.2f}%  Sharpe={v['sharpe']:6.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', required=True, help='walk-forward predictions parquet')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--by-year', action='store_true')
    ap.add_argument('--n-sub', type=int, default=None, help='subsample for speed')
    args = ap.parse_args()
    scoreboard(args.preds, tag=args.tag, by_year=args.by_year, n_sub=args.n_sub)
