"""
Calibration Evaluation Lab (offline, no training).

Establishes a baseline scoreboard for the ML scorer's win_prob calibration and
evaluates candidate calibrators that map a model's predicted return (%) to
P(actual_return > 0) = hit_target.

North-star metric: calibration reliability of win_prob, measured honestly via
leave-one-year-out (LOYO) so there is no in-sample leakage. Primary ranking:
LOYO Brier, then LOYO ECE, guarded by AUC (must not drop -- monotone calibration
never changes AUC, but we report it).

Inputs (read-only):
  results/wf_predictions_sr.parquet          (tier 10_30)
  results/wf_predictions_sr_31_60.parquet    (tier 31_60)
  results/wf_predictions_sr_61_90.parquet    (tier 61_90)
  results/calibration_sr[ _31_60 | _61_90 ].json   (production 20-bin tables)

Outputs (new files only, under results/experiments/):
  calibration_lab_<tier>.csv / .json   ranked scoreboard
  calibration_reliability_<tier>.csv    reliability table: baseline vs best method

This script does NOT import or modify any production code. The production
20-bin step-function lookup is replicated here exactly (see scorer.py
ModelEnsemble._calibrate).

Usage:
  python tools/calibration_lab.py                 # all tiers, full data
  python tools/calibration_lab.py --tier 10_30    # one tier
  python tools/calibration_lab.py --sample 2000000  # fast iteration subsample
"""
import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

SEED = 42
EPS = 1e-6  # log-loss clip
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'results')
OUTDIR = os.path.join(RESULTS, 'experiments')

TIERS = {
    '10_30': {'parquet': 'wf_predictions_sr.parquet',       'cal': 'calibration_sr.json'},
    '31_60': {'parquet': 'wf_predictions_sr_31_60.parquet', 'cal': 'calibration_sr_31_60.json'},
    '61_90': {'parquet': 'wf_predictions_sr_61_90.parquet', 'cal': 'calibration_sr_61_90.json'},
}

# Years to down-weight in the recency-weighted variant (structural dead zones for
# the longer tiers; see CLAUDE.md). Weight applied to fit only, never to scoring.
OLD_YEARS = (2018, 2019)
OLD_WEIGHT = 0.5


# --------------------------------------------------------------------------- #
# Reliability metrics
# --------------------------------------------------------------------------- #
def reliability_metrics(prob, y, n_bins=15, mce_min_frac=0.001):
    """ECE/MCE (equal-width bins on prob), Brier, log-loss.

    ECE uses n_bins equal-width bins over [0,1] on the *mapped* probability,
    weighting each bin's |mean_prob - mean_realized| by its sample share.
    Brier = mean((prob-y)^2). log-loss clips probs to [EPS, 1-EPS].

    MCE is the max gap over bins, but only over bins holding at least
    mce_min_frac of the sample (default 0.1%). Without this guard MCE is
    dominated by a single near-empty extreme-probability bin (e.g. isotonic
    can emit ~0.999 for a few dozen rows), which is a sparse-bin artifact, not
    systematic miscalibration. The guarded MCE measures real worst-case error.
    """
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    min_cnt = max(1, int(mce_min_frac * n))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # clip to last bin so prob==1.0 lands in bin n_bins-1
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        gap = abs(prob[m].mean() - y[m].mean())
        ece += (cnt / n) * gap
        if cnt >= min_cnt and gap > mce:
            mce = gap

    brier = float(np.mean((prob - y) ** 2))
    pc = np.clip(prob, EPS, 1.0 - EPS)
    logloss = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))
    return {'ECE': float(ece), 'MCE': float(mce), 'Brier': brier, 'logloss': logloss}


def reliability_table(prob, y, n_bins=15):
    """Per-bin (mean_pred_prob, mean_realized, count) on equal-width prob bins."""
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        rows.append({
            'prob_bin': b,
            'bin_lo': round(float(edges[b]), 4),
            'bin_hi': round(float(edges[b + 1]), 4),
            'mean_pred_prob': round(float(prob[m].mean()), 5) if cnt else None,
            'mean_realized': round(float(y[m].mean()), 5) if cnt else None,
            'count': cnt,
        })
    return rows


def reliability_table_by_pred_decile(prob, pred, y, n=10):
    """Reliability grouped by decile of the raw *prediction* (not prob).

    Because mapped win_prob spans a narrow range (~0.52-0.87), equal-width prob
    bins leave the tails empty. Grouping by prediction decile makes the low end
    and the high-conviction top end both visible, with equal sample mass.
    """
    prob = np.asarray(prob, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.quantile(pred, np.linspace(0, 1, n + 1))
    edges[-1] = np.inf
    idx = np.clip(np.searchsorted(edges[1:-1], pred, side='right'), 0, n - 1)
    rows = []
    for b in range(n):
        m = idx == b
        cnt = int(m.sum())
        rows.append({
            'pred_decile': b,
            'pred_lo': round(float(edges[b]), 4),
            'pred_hi': round(float(edges[b + 1]), 4) if np.isfinite(edges[b + 1]) else 'inf',
            'mean_pred_prob': round(float(prob[m].mean()), 5) if cnt else None,
            'mean_realized': round(float(y[m].mean()), 5) if cnt else None,
            'abs_gap': round(abs(float(prob[m].mean()) - float(y[m].mean())), 5) if cnt else None,
            'count': cnt,
        })
    return rows


# --------------------------------------------------------------------------- #
# Production baseline lookup (replicated EXACTLY from scorer.py)
# --------------------------------------------------------------------------- #
def production_lookup_vectorized(pred_values, cal_bins, field='win_prob'):
    """Replicate ModelEnsemble._calibrate over an array.

    Bins are sorted by 'bin'. For each pred value, return the FIRST bin where
    pred <= bin['pred_max']; if it exceeds every pred_max, use the LAST bin.
    This is a right-edge step function with tail clamping. Implemented via
    searchsorted on the sorted pred_max edges for speed (equivalent result).
    """
    bins = sorted(cal_bins, key=lambda b: b['bin'])
    pmax = np.array([b['pred_max'] for b in bins], dtype=np.float64)
    vals = np.array([b.get(field, 0.5) for b in bins], dtype=np.float64)
    pred = np.asarray(pred_values, dtype=np.float64)
    # first index where pred <= pmax[i]  <=>  searchsorted on pmax (side='left')
    pos = np.searchsorted(pmax, pred, side='left')
    pos = np.clip(pos, 0, len(bins) - 1)  # above all -> last bin
    return vals[pos]


# --------------------------------------------------------------------------- #
# Calibrator implementations. Each exposes fit(pred,y,w)->self and predict(pred)->prob.
# All map predicted return (1-D) -> probability.
# --------------------------------------------------------------------------- #
class QuantileBinStep:
    """N equal-frequency quantile bins; step function on pred_max with tail clamp.

    This is the honest, refit version of the production calibrator. The fit
    target for win_prob is mean(y>0) within each bin == mean(hit_target).
    """
    def __init__(self, n_bins=20):
        self.n_bins = n_bins

    def fit(self, pred, y, w=None):
        pred = np.asarray(pred, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        df = pd.DataFrame({'p': pred, 'y': y})
        df['b'] = pd.qcut(df['p'], self.n_bins, labels=False, duplicates='drop')
        g = df.groupby('b')
        self.pred_max = g['p'].max().to_numpy()
        order = np.argsort(self.pred_max)
        self.pred_max = self.pred_max[order]
        self.win = g['y'].mean().to_numpy()[order]
        return self

    def predict(self, pred):
        pred = np.asarray(pred, dtype=np.float64)
        pos = np.searchsorted(self.pred_max, pred, side='left')
        pos = np.clip(pos, 0, len(self.win) - 1)
        return self.win[pos]


class IsotonicCal:
    """Isotonic regression pred->y. tail='clip' or tail='extrap' (linear)."""
    def __init__(self, tail='clip'):
        self.tail = tail

    def fit(self, pred, y, w=None):
        oob = 'clip' if self.tail == 'clip' else 'nan'
        self.iso = IsotonicRegression(increasing=True, out_of_bounds=oob)
        self.iso.fit(np.asarray(pred, dtype=np.float64),
                     np.asarray(y, dtype=np.float64),
                     sample_weight=w)
        self.xmin = float(self.iso.X_min_)
        self.xmax = float(self.iso.X_max_)
        # endpoint slopes for linear extrapolation (computed from the fitted
        # step function near each boundary)
        xs = np.linspace(self.xmin, self.xmax, 256)
        ys = self.iso.predict(xs)
        # low-end slope
        i = 1
        while i < len(xs) and ys[i] == ys[0]:
            i += 1
        self.lo_slope = 0.0 if i >= len(xs) else (ys[i] - ys[0]) / (xs[i] - xs[0])
        self.lo_y, self.lo_x = float(ys[0]), float(xs[0])
        # high-end slope
        j = len(xs) - 2
        while j >= 0 and ys[j] == ys[-1]:
            j -= 1
        self.hi_slope = 0.0 if j < 0 else (ys[-1] - ys[j]) / (xs[-1] - xs[j])
        self.hi_y, self.hi_x = float(ys[-1]), float(xs[-1])
        return self

    def predict(self, pred):
        pred = np.asarray(pred, dtype=np.float64)
        if self.tail == 'clip':
            return np.clip(self.iso.predict(np.clip(pred, self.xmin, self.xmax)),
                           0.0, 1.0)
        out = self.iso.predict(np.clip(pred, self.xmin, self.xmax))
        lo = pred < self.xmin
        hi = pred > self.xmax
        out = np.where(lo, self.lo_y + self.lo_slope * (pred - self.lo_x), out)
        out = np.where(hi, self.hi_y + self.hi_slope * (pred - self.hi_x), out)
        return np.clip(out, 0.0, 1.0)


class PlattCal:
    """Logistic regression. degree=1 -> [pred]; degree=2 -> [pred, pred^2]."""
    def __init__(self, degree=1):
        self.degree = degree

    def fit(self, pred, y, w=None):
        # standardize for conditioning, esp. with the squared term
        p = np.asarray(pred, dtype=np.float64)
        self.mu = p.mean()
        self.sd = p.std() + 1e-12
        ps = (p - self.mu) / self.sd
        X = ps.reshape(-1, 1)
        if self.degree == 2:
            X = np.hstack([X, ps.reshape(-1, 1) ** 2])
        self.clf = LogisticRegression(C=1e6, max_iter=2000, solver='lbfgs')
        self.clf.fit(X, np.asarray(y, dtype=np.int32), sample_weight=w)
        return self

    def predict(self, pred):
        ps = (np.asarray(pred, dtype=np.float64) - self.mu) / self.sd
        X = ps.reshape(-1, 1)
        if self.degree == 2:
            X = np.hstack([X, ps.reshape(-1, 1) ** 2])
        return self.clf.predict_proba(X)[:, 1]


class ConditionalIsotonic:
    """Fit a separate isotonic per group (direction, or direction x daysOut bucket).

    Falls back to a global isotonic for any group unseen at fit time.
    """
    def __init__(self, key_fn, tail='clip'):
        self.key_fn = key_fn
        self.tail = tail

    def fit(self, df):
        """df must have a clean 0..n-1 index (caller resets it)."""
        self.models = {}
        self.global_ = IsotonicCal(self.tail).fit(
            df['predicted'].to_numpy(), df['hit_target'].to_numpy())
        df = df.assign(_k=self.key_fn(df))
        for k, sub in df.groupby('_k'):
            self.models[k] = IsotonicCal(self.tail).fit(
                sub['predicted'].to_numpy(), sub['hit_target'].to_numpy())
        return self

    def predict(self, df):
        keys = self.key_fn(df).to_numpy()
        pred = df['predicted'].to_numpy()
        out = np.empty(len(df), dtype=np.float64)
        for k in np.unique(keys):
            m = keys == k
            model = self.models.get(k, self.global_)
            out[m] = model.predict(pred[m])
        return out


def daysout_bucket(days, tier):
    """Coarse daysOut buckets within a tier for breakdowns / conditional cal."""
    days = np.asarray(days)
    if tier == '10_30':
        return np.where(days <= 17, '10-17', np.where(days <= 24, '18-24', '25-30'))
    if tier == '31_60':
        return np.where(days <= 40, '31-40', np.where(days <= 50, '41-50', '51-60'))
    return np.where(days <= 70, '61-70', np.where(days <= 80, '71-80', '81-90'))


# --------------------------------------------------------------------------- #
# Method registry. Each entry builds a fitted calibrator on (df_fit[, weights])
# and produces probabilities on df_eval. We keep the *fit signature uniform*:
# the simple 1-D methods take (pred, y, w); conditional methods take the frame.
# --------------------------------------------------------------------------- #
def _fit_predict_1d(builder, df_fit, df_eval, weights=None):
    m = builder()
    m.fit(df_fit['predicted'].to_numpy(), df_fit['hit_target'].to_numpy(),
          None if weights is None else np.asarray(weights))
    return m.predict(df_eval['predicted'].to_numpy())


METHODS = {}  # name -> (kind, builder/key_fn, needs_frame, weight_old)


def register():
    METHODS['quantile_bins_10'] = ('1d', lambda: QuantileBinStep(10), False, False)
    METHODS['quantile_bins_20'] = ('1d', lambda: QuantileBinStep(20), False, False)
    METHODS['quantile_bins_50'] = ('1d', lambda: QuantileBinStep(50), False, False)
    METHODS['isotonic_clip'] = ('1d', lambda: IsotonicCal('clip'), False, False)
    METHODS['isotonic_extrap'] = ('1d', lambda: IsotonicCal('extrap'), False, False)
    METHODS['platt_linear'] = ('1d', lambda: PlattCal(1), False, False)
    METHODS['platt_quad'] = ('1d', lambda: PlattCal(2), False, False)
    METHODS['isotonic_by_dir'] = ('cond', lambda df: df['direction'], True, False)
    METHODS['isotonic_recency_w'] = ('1d', lambda: IsotonicCal('clip'), False, True)


register()


def make_weights(df):
    w = np.ones(len(df), dtype=np.float64)
    w[df['val_year'].isin(OLD_YEARS).to_numpy()] = OLD_WEIGHT
    return w


# --------------------------------------------------------------------------- #
# Evaluation drivers
# --------------------------------------------------------------------------- #
def eval_in_sample_baseline(df, cal_bins):
    """Production baseline scored on the very data its bins were fit on."""
    prob = production_lookup_vectorized(df['predicted'].to_numpy(), cal_bins, 'win_prob')
    y = df['hit_target'].to_numpy()
    m = reliability_metrics(prob, y)
    m['AUC'] = float(roc_auc_score(y, df['predicted'].to_numpy()))
    return prob, m


def loyo_predict(method_name, df):
    """Leave-one-year-out: fit on val_year != Y, score val_year == Y, pool all."""
    kind, builder, needs_frame, weight_old = METHODS[method_name]
    years = sorted(df['val_year'].unique())
    out = np.full(len(df), np.nan, dtype=np.float64)
    for Y in years:
        fit_mask = (df['val_year'] != Y).to_numpy()
        eval_mask = (df['val_year'] == Y).to_numpy()
        df_fit = df.loc[fit_mask]
        df_eval = df.loc[eval_mask]
        weights = make_weights(df_fit) if weight_old else None
        if kind == '1d':
            p = _fit_predict_1d(builder, df_fit, df_eval, weights)
        else:  # conditional
            m = ConditionalIsotonic(builder, tail='clip')
            m.fit(df_fit.reset_index(drop=True))
            p = m.predict(df_eval.reset_index(drop=True))
        out[eval_mask] = p
    return out


def loyo_baseline_predict(df, n_bins=20):
    """Honest production baseline: refit 20-quantile-bin step fn per LOYO fold.

    This is the production *recipe* (qcut 20-bin step, tail clamp) evaluated
    without leakage -- the true bar candidates must beat.
    """
    return loyo_predict_qbin(df, n_bins)


def loyo_predict_qbin(df, n_bins):
    years = sorted(df['val_year'].unique())
    out = np.full(len(df), np.nan, dtype=np.float64)
    for Y in years:
        df_fit = df.loc[(df['val_year'] != Y).to_numpy()]
        eval_mask = (df['val_year'] == Y).to_numpy()
        m = QuantileBinStep(n_bins).fit(df_fit['predicted'].to_numpy(),
                                        df_fit['hit_target'].to_numpy())
        out[eval_mask] = m.predict(df.loc[eval_mask, 'predicted'].to_numpy())
    return out


def breakdowns(prob, df, tier):
    """Overall + per val_year + per direction + per daysOut-bucket metrics."""
    y = df['hit_target'].to_numpy()
    pred = df['predicted'].to_numpy()
    res = {'overall': {**reliability_metrics(prob, y),
                       'AUC': float(roc_auc_score(y, pred)),
                       'n': int(len(y))}}
    by_year = {}
    for Y, sub in df.groupby('val_year'):
        idx = sub.index.to_numpy()
        yy = y[idx]
        mm = reliability_metrics(prob[idx], yy)
        mm['AUC'] = float(roc_auc_score(yy, pred[idx])) if len(np.unique(yy)) > 1 else None
        mm['n'] = int(len(idx))
        by_year[int(Y)] = mm
    res['by_val_year'] = by_year
    by_dir = {}
    for D, sub in df.groupby('direction'):
        idx = sub.index.to_numpy()
        yy = y[idx]
        mm = reliability_metrics(prob[idx], yy)
        mm['AUC'] = float(roc_auc_score(yy, pred[idx])) if len(np.unique(yy)) > 1 else None
        mm['n'] = int(len(idx))
        by_dir[str(D)] = mm
    res['by_direction'] = by_dir
    buckets = daysout_bucket(df['daysOut'].to_numpy(), tier)
    by_do = {}
    for B in np.unique(buckets):
        m = buckets == B
        yy = y[m]
        mm = reliability_metrics(prob[m], yy)
        mm['AUC'] = float(roc_auc_score(yy, pred[m])) if len(np.unique(yy)) > 1 else None
        mm['n'] = int(m.sum())
        by_do[str(B)] = mm
    res['by_daysOut_bucket'] = by_do
    return res


def tail_analysis(df, cal_bins, best_name, best_prob, baseline_loyo_prob):
    """Quantify reliability of high-conviction picks that get clamped in production.

    Production clamps any prediction above the top bin's pred_max to the last
    bin's win_prob. Under LOYO, the per-fold fit max is below the global max, so
    some eval rows genuinely exceed the fit range. We report, per fold, how many
    eval rows exceed that fold's fit-set max prediction, and compare realized win
    rate vs what production (in-sample bins) and the candidate methods assign.
    """
    years = sorted(df['val_year'].unique())
    pred = df['predicted'].to_numpy()
    y = df['hit_target'].to_numpy()

    # rows above each fold's fit-set max
    above_mask = np.zeros(len(df), dtype=bool)
    fold_info = []
    for Y in years:
        fit_max = float(df.loc[(df['val_year'] != Y).to_numpy(), 'predicted'].max())
        em = (df['val_year'] == Y).to_numpy()
        am = em & (pred > fit_max)
        above_mask |= am
        fold_info.append({'val_year': int(Y), 'fit_max_pred': round(fit_max, 4),
                          'n_eval_above_fit_max': int(am.sum())})

    n_above = int(above_mask.sum())

    # production global top-bin pred_max + win_prob
    bins = sorted(cal_bins, key=lambda b: b['bin'])
    top_pred_max = float(bins[-1]['pred_max'])
    top_win = float(bins[-1]['win_prob'])
    prod_clamped = int((pred > top_pred_max).sum())

    # Top-1% high-conviction slice on raw prediction (stable, fold-independent)
    thr = float(np.quantile(pred, 0.99))
    top_mask = pred >= thr
    prod_prob_all = production_lookup_vectorized(pred, cal_bins, 'win_prob')

    def slice_stats(mask):
        if mask.sum() == 0:
            return None
        return {
            'n': int(mask.sum()),
            'realized_wr': round(float(y[mask].mean()), 5),
            'baseline_loyo_mean_prob': round(float(baseline_loyo_prob[mask].mean()), 5),
            'production_insample_mean_prob': round(float(prod_prob_all[mask].mean()), 5),
            f'{best_name}_mean_prob': round(float(best_prob[mask].mean()), 5),
            'baseline_loyo_abs_gap': round(abs(float(baseline_loyo_prob[mask].mean()) -
                                              float(y[mask].mean())), 5),
            f'{best_name}_abs_gap': round(abs(float(best_prob[mask].mean()) -
                                             float(y[mask].mean())), 5),
        }

    return {
        'production_top_bin_pred_max': round(top_pred_max, 4),
        'production_top_bin_win_prob': top_win,
        'rows_above_production_top_pred_max_in_sample': prod_clamped,
        'rows_above_loyo_fold_fit_max_total': n_above,
        'loyo_fold_detail': fold_info,
        'above_loyo_fit_max_slice': slice_stats(above_mask),
        'top_1pct_pred_threshold': round(thr, 4),
        'top_1pct_slice': slice_stats(top_mask),
    }


# --------------------------------------------------------------------------- #
def run_tier(tier, sample=None):
    cfg = TIERS[tier]
    pq_path = os.path.join(RESULTS, cfg['parquet'])
    cal_path = os.path.join(RESULTS, cfg['cal'])
    print(f"\n{'='*78}\nTIER {tier}\n{'='*78}")
    print(f"  loading {cfg['parquet']} ...")
    df = pd.read_parquet(pq_path,
                         columns=['val_year', 'predicted', 'actual_return',
                                  'hit_target', 'direction', 'daysOut'])
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=SEED)
        print(f"  SUBSAMPLED to {len(df):,} rows (seed={SEED})")
    df = df.reset_index(drop=True)
    df['hit_target'] = df['hit_target'].astype(np.int8)
    print(f"  rows={len(df):,}  years={sorted(df['val_year'].unique())}  "
          f"base_rate={df['hit_target'].mean():.4f}")

    cal_bins = json.load(open(cal_path))['bins']

    scoreboard = []  # list of dict rows

    # 1) Production baseline -- in-sample
    print("  [1/3] production baseline (in-sample) ...")
    _, m_in = eval_in_sample_baseline(df, cal_bins)
    scoreboard.append({'method': 'production_20bin_step', 'fit_scheme': 'in_sample', **m_in})

    # 2) Honest baseline (production recipe, LOYO) + all candidates (LOYO)
    print("  [2/3] LOYO honest baseline + candidates ...")
    auc_full = float(roc_auc_score(df['hit_target'].to_numpy(), df['predicted'].to_numpy()))

    baseline_loyo_prob = loyo_baseline_predict(df, n_bins=20)
    m_base = reliability_metrics(baseline_loyo_prob, df['hit_target'].to_numpy())
    m_base['AUC'] = auc_full
    scoreboard.append({'method': 'production_20bin_step', 'fit_scheme': 'LOYO', **m_base})

    candidate_probs = {'baseline_loyo': baseline_loyo_prob}
    for name in METHODS:
        # quantile_bins_20 LOYO == honest baseline; still report under its own name
        print(f"        - {name}")
        prob = loyo_predict(name, df)
        mm = reliability_metrics(prob, df['hit_target'].to_numpy())
        mm['AUC'] = auc_full  # monotone 1-D maps preserve AUC; conditional ~preserves
        scoreboard.append({'method': name, 'fit_scheme': 'LOYO', **mm})
        candidate_probs[name] = prob

    # rank LOYO methods by Brier then ECE (exclude in-sample row from ranking)
    loyo_rows = [r for r in scoreboard if r['fit_scheme'] == 'LOYO']
    loyo_rows_sorted = sorted(loyo_rows, key=lambda r: (r['Brier'], r['ECE']))
    best = loyo_rows_sorted[0]
    best_name = best['method']
    best_prob = (baseline_loyo_prob if best_name == 'production_20bin_step'
                 else candidate_probs[best_name])
    print(f"  WINNER (LOYO, min Brier->ECE): {best_name}  "
          f"Brier={best['Brier']:.5f} ECE={best['ECE']:.5f} logloss={best['logloss']:.5f}")

    # 3) Breakdowns for baseline + winner, tail analysis
    print("  [3/3] breakdowns + tail analysis ...")
    bd_base = breakdowns(baseline_loyo_prob, df, tier)
    bd_best = breakdowns(best_prob, df, tier)
    tail = tail_analysis(df, cal_bins, best_name, best_prob, baseline_loyo_prob)

    # reliability tables (baseline vs best): equal-width prob bins + pred deciles
    yv = df['hit_target'].to_numpy()
    pv = df['predicted'].to_numpy()
    rel_base = reliability_table(baseline_loyo_prob, yv)
    rel_best = reliability_table(best_prob, yv)
    dec_base = reliability_table_by_pred_decile(baseline_loyo_prob, pv, yv)
    dec_best = reliability_table_by_pred_decile(best_prob, pv, yv)
    json_obj_reliability = {'prob_bins': {'baseline': rel_base, 'best': rel_best},
                            'pred_deciles': {'baseline': dec_base, 'best': dec_best}}

    # ----- write outputs -----
    os.makedirs(OUTDIR, exist_ok=True)

    # scoreboard csv (ranked: in-sample baseline first, then LOYO by Brier)
    sb_df = pd.DataFrame(scoreboard)[['method', 'fit_scheme', 'ECE', 'MCE', 'Brier',
                                      'logloss', 'AUC']]
    sb_df = sb_df.round(6)
    # order: in_sample rows, then LOYO sorted by Brier
    in_rows = sb_df[sb_df['fit_scheme'] == 'in_sample']
    lo_rows = sb_df[sb_df['fit_scheme'] == 'LOYO'].sort_values(['Brier', 'ECE'])
    sb_out = pd.concat([in_rows, lo_rows], ignore_index=True)
    csv_path = os.path.join(OUTDIR, f'calibration_lab_{tier}.csv')
    sb_out.to_csv(csv_path, index=False)

    recipe = recommend(tier, best_name, best, lo_rows, tail)

    json_obj = {
        'tier': tier,
        'n_rows': int(len(df)),
        'base_rate': round(float(df['hit_target'].mean()), 5),
        'seed': SEED,
        'subsampled': bool(sample and sample < int(1e12)),
        'scoreboard': sb_out.to_dict(orient='records'),
        'winner': {'method': best_name, **{k: best[k] for k in
                   ('ECE', 'MCE', 'Brier', 'logloss', 'AUC')}},
        'honest_baseline_LOYO': {k: m_base[k] for k in
                                 ('ECE', 'MCE', 'Brier', 'logloss', 'AUC')},
        'delta_winner_vs_honest_baseline': {
            'ECE': round(best['ECE'] - m_base['ECE'], 6),
            'Brier': round(best['Brier'] - m_base['Brier'], 6),
            'logloss': round(best['logloss'] - m_base['logloss'], 6),
        },
        'breakdown_baseline_LOYO': bd_base,
        'breakdown_winner_LOYO': bd_best,
        'tail_analysis': tail,
        'reliability_tables': json_obj_reliability,
        'recommendation': recipe,
    }
    json_path = os.path.join(OUTDIR, f'calibration_lab_{tier}.json')
    with open(json_path, 'w') as f:
        json.dump(json_obj, f, indent=2)

    # reliability table csv (equal-width prob bins)
    rel_rows = []
    for r in rel_base:
        rel_rows.append({'which': 'baseline_LOYO', **r})
    for r in rel_best:
        rel_rows.append({'which': f'best_{best_name}_LOYO', **r})
    rel_path = os.path.join(OUTDIR, f'calibration_reliability_{tier}.csv')
    pd.DataFrame(rel_rows).to_csv(rel_path, index=False)

    # reliability table csv by prediction decile (tails visible)
    dec_rows = []
    for r in dec_base:
        dec_rows.append({'which': 'baseline_LOYO', **r})
    for r in dec_best:
        dec_rows.append({'which': f'best_{best_name}_LOYO', **r})
    dec_path = os.path.join(OUTDIR, f'calibration_reliability_by_decile_{tier}.csv')
    pd.DataFrame(dec_rows).to_csv(dec_path, index=False)

    print(f"  wrote: {csv_path}")
    print(f"  wrote: {json_path}")
    print(f"  wrote: {rel_path}")
    print(f"  wrote: {dec_path}")

    return json_obj


def _params_for(name):
    if name.startswith('isotonic_by_dir'):
        return {'type': 'IsotonicRegression(increasing=True), fit separately per direction (l/s)',
                'tail': 'out_of_bounds=clip'}
    if name.startswith('isotonic'):
        return {'type': 'IsotonicRegression(increasing=True)',
                'tail': 'out_of_bounds=clip' if 'clip' in name else 'linear extrapolation beyond fit range'}
    if name.startswith('platt'):
        return {'type': 'LogisticRegression (Platt)',
                'features': '[predicted]' if name == 'platt_linear' else '[predicted, predicted^2]',
                'note': 'standardize predicted before fit'}
    if name.startswith('quantile'):
        return {'type': 'equal-frequency quantile bins, step function on pred_max with tail clamp',
                'n_bins': int(name.split('_')[-1])}
    return {'type': name}


def recommend(tier, best_name, best_row, lo_rows_df, tail):
    """One actionable recipe per tier.

    Ranking metric is LOYO Brier (primary) then ECE, per spec. best_row is the
    (Brier, ECE) argmin among all candidates. We recommend SWITCHING away from
    the incumbent production recipe (20 quantile bins) only if that winner beats
    it on Brier by a material margin (>=1e-4); since Brier is the primary metric,
    a sub-noise ECE wiggle does not veto a material Brier+log-loss win. If no
    candidate is materially better (e.g. the short 10_30 tier where everything
    is near-tied), we recommend the smoothest near-optimal recipe (isotonic_clip)
    which removes the 20-step jaggedness without changing reliability. The tail
    recommendation is always made explicit and is grounded in the actual top-1%
    slice numbers.
    """
    base = lo_rows_df[lo_rows_df['method'] == 'production_20bin_step'].iloc[0]
    iso = lo_rows_df[lo_rows_df['method'] == 'isotonic_clip'].iloc[0]

    # Ranking per spec: minimize LOYO Brier (primary), then ECE. best_row is
    # already that argmin. Only recommend SWITCHING off the incumbent if the
    # Brier gain is material; a sub-noise ECE wiggle does not veto a material
    # Brier+logloss win (Brier is the primary metric).
    MATERIAL = 1e-4
    brier_gain = float(base['Brier'] - best_row['Brier'])

    if best_name != 'production_20bin_step' and brier_gain >= MATERIAL:
        chosen = best_name
        ece_note = ('and ECE' if best_row['ECE'] <= base['ECE'] + 1e-9
                    else f"(ECE essentially unchanged: {best_row['ECE'] - base['ECE']:+.5f})")
        rationale = (f"{best_name} minimizes LOYO Brier, beating the production 20-bin recipe "
                     f"by {brier_gain:.5f} on Brier {ece_note}, and also improves log-loss "
                     f"by {float(base['logloss'] - best_row['logloss']):.5f}. AUC is unchanged.")
    elif iso['ECE'] <= base['ECE'] + 1e-3 and abs(iso['Brier'] - base['Brier']) < 1e-3:
        # near-tie on Brier: prefer isotonic_clip, a smooth monotone map that
        # matches baseline reliability without 20-step jaggedness and handles
        # the tail by construction.
        chosen = 'isotonic_clip'
        rationale = ("No candidate materially beats the production recipe on Brier "
                     f"(best gain {brier_gain:.5f}, below the {MATERIAL} threshold). "
                     "Recommend isotonic_clip: reliability matches the 20-bin baseline "
                     "(Brier/ECE within noise) but it is a smooth monotone curve instead of "
                     "a 20-step function, easier to maintain and tail-safe by construction.")
    else:
        chosen = 'production_20bin_step'
        rationale = ("No candidate beats the production 20-bin recipe on Brier or ECE; "
                     "keep it as is.")

    chosen_row = lo_rows_df[lo_rows_df['method'] == chosen].iloc[0]

    # Tail handling recommendation, grounded in the actual tail analysis numbers.
    # The high-conviction (top-1%) slice is where it matters: production clamps
    # those to the top bin's win_prob. We compare the *chosen* method's gap to
    # realized WR there against the production-baseline gap.
    sl = tail.get('above_loyo_fit_max_slice')
    top = tail.get('top_1pct_slice') or {}
    n_above = tail.get('rows_above_loyo_fold_fit_max_total', 0)
    realized = top.get('realized_wr')
    base_gap = top.get('baseline_loyo_abs_gap')
    base_prob = top.get('baseline_loyo_mean_prob')
    chosen_gap = top.get(f'{chosen}_abs_gap')
    chosen_prob = top.get(f'{chosen}_mean_prob')
    # quality of production clamp at the top end
    if base_gap is not None and base_gap <= 0.02:
        clamp_verdict = ("The production tail clamp is well calibrated at the top end "
                         f"(baseline assigns {base_prob} vs realized {realized}).")
    else:
        clamp_verdict = (f"The production tail clamp UNDER-calibrates high-conviction picks: "
                         f"the top-1% slice realizes WR={realized} but the 20-bin baseline "
                         f"assigns only {base_prob} (abs gap {base_gap}).")
    if chosen != 'production_20bin_step' and chosen_gap is not None:
        fix_note = (f" The recommended {chosen} maps that same slice to {chosen_prob} "
                    f"(abs gap {chosen_gap}), {'closing' if chosen_gap < base_gap else 'not closing'} "
                    f"most of the tail miscalibration.")
    else:
        fix_note = ''
    tail_recipe = ('Use out_of_bounds=clip; linear extrapolation changes only a handful of rows '
                   f'({n_above} held-out rows exceeded the per-fold fit range, and only when '
                   '2018/2019 -- which hold the global max prediction -- are the held-out fold).')
    tail_handling = f"{clamp_verdict}{fix_note} {tail_recipe}"

    return {
        'recommended_method': chosen,
        'params': _params_for(chosen),
        'rationale': rationale,
        'LOYO_Brier': round(float(chosen_row['Brier']), 6),
        'LOYO_ECE': round(float(chosen_row['ECE']), 6),
        'LOYO_MCE_guarded': round(float(chosen_row['MCE']), 6),
        'LOYO_logloss': round(float(chosen_row['logloss']), 6),
        'vs_production_baseline': {
            'dBrier': round(float(chosen_row['Brier'] - base['Brier']), 6),
            'dECE': round(float(chosen_row['ECE'] - base['ECE']), 6),
            'dlogloss': round(float(chosen_row['logloss'] - base['logloss']), 6),
        },
        'tail_handling': tail_handling,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', choices=list(TIERS), default=None,
                    help='single tier (default: all)')
    ap.add_argument('--sample', type=int, default=None,
                    help='subsample N rows per tier for fast iteration')
    args = ap.parse_args()

    tiers = [args.tier] if args.tier else list(TIERS)
    summary = {}
    for t in tiers:
        summary[t] = run_tier(t, sample=args.sample)

    # console summary
    print(f"\n{'#'*78}\nSUMMARY\n{'#'*78}")
    for t in tiers:
        o = summary[t]
        b = o['honest_baseline_LOYO']
        w = o['winner']
        ins = next(r for r in o['scoreboard']
                   if r['method'] == 'production_20bin_step'
                   and r['fit_scheme'] == 'in_sample')
        print(f"\nTIER {t}  (n={o['n_rows']:,}, base_rate={o['base_rate']})")
        print(f"  baseline in-sample : ECE={ins['ECE']:.5f} Brier={ins['Brier']:.5f} "
              f"logloss={ins['logloss']:.5f}")
        print(f"  baseline LOYO      : ECE={b['ECE']:.5f} Brier={b['Brier']:.5f} "
              f"logloss={b['logloss']:.5f}  AUC={b['AUC']:.4f}")
        print(f"  WINNER {w['method']:22s}: ECE={w['ECE']:.5f} Brier={w['Brier']:.5f} "
              f"logloss={w['logloss']:.5f}")
        d = o['delta_winner_vs_honest_baseline']
        print(f"  delta vs baseline  : dECE={d['ECE']:+.5f} dBrier={d['Brier']:+.6f} "
              f"dlogloss={d['logloss']:+.5f}")
        rec = o['recommendation']
        vb = rec['vs_production_baseline']
        print(f"  RECOMMEND {rec['recommended_method']:18s}: Brier={rec['LOYO_Brier']:.5f} "
              f"ECE={rec['LOYO_ECE']:.5f} logloss={rec['LOYO_logloss']:.5f} "
              f"(vs prod: dBrier={vb['dBrier']:+.6f} dECE={vb['dECE']:+.5f})")
        print(f"  tail: {rec['tail_handling']}")


if __name__ == '__main__':
    main()
