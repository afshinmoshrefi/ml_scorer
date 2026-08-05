"""
Honest model A/B: leaky (current 62) vs honest (PIT-replaced) feature sets,
trained under purged walk-forward on the SAME rows.

Arm LEAKY : the production 62 features (opp-file pattern features included).
Arm HONEST: the 62 minus the 17 opp-file-derived features (which encode
            full-history selection look-ahead), plus 7 PIT replacements
            computed strictly from prior-year prices (tools/pit_join.py).

Both arms: identical rows (loaded once), identical tuned params, purged
walk-forward (--purge-overlap semantics), same ensemble (LGB+XGB+CatBoost).
The walk-forward AUC/win-rate gap between arms estimates how much of the
model's measured skill was the leak.

Run AFTER tools/pit_join.py has produced features/training_data_<tier>_pit.parquet:
  python tools/honest_ab.py --tier 10_30
Outputs: results/experiments/honest_ab_<tier>.json
         results/wf_predictions_sr_ab_{leaky,honest}.parquet
"""
import os
import sys
import gc
import json
import argparse

import numpy as np
import pandas as pd

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TOOLS)
sys.path.insert(0, _REPO)
import train_model as tm  # noqa: E402

# Opp-file-derived features: their values (and the rows' existence) come from
# full-history mining, i.e. they see the validation years' outcomes.
LEAKY_FEATURES = [
    'pat_sharpe_ratio', 'pat_avg_profit2', 'pat_deepest_pass',
    'pat_depth_utilization', 'pat_passes_recent_10', 'pat_recent_vs_deep_sharpe',
    'pat_num_combos_qualifying', 'pat_pe_match', 'pat_pe_deepest',
    'pat_pe_utilization', 'pat_best_winrate', 'pat_worst_winrate',
    'pat_deepest_pass_capped30', 'pat_consistency_std', 'pat_concurrent_count',
    'pat_depth_x_vix', 'pat_quality_x_regime',
]
PIT_FEATURES = ['pit_deepest', 'pit_q10_8', 'pit_wr5', 'pit_wr10', 'pit_wr20',
                'pit_n_prior', 'pit_depth_x_vix']


def load_once(tier, arm_cols_union):
    path = os.path.join(_REPO, 'features', f'training_data_{tier}_pit.parquet')
    label_cols = ['date', 'actual_return', 'hit_target', 'symbol', 'direction', 'daysOut']
    cols = list(dict.fromkeys(arm_cols_union + label_cols))
    df = pd.read_parquet(path, columns=cols)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year.astype(np.int16)
    # VIX hurricane filter, matching load_training_data
    if 'mkt_vix_level' in df.columns:
        n0 = len(df)
        df = df[~(df['mkt_vix_level'] > 35)].reset_index(drop=True)
        print(f'VIX>35 filter: removed {n0-len(df):,} rows')
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    gc.collect()
    print(f'Loaded {len(df):,} rows, {len(df.columns)} cols')
    return df


def summarize(wf_results):
    out = []
    for r in wf_results:
        e = {'val_year': r['val_year'], 'auc': r['metrics']['auc_roc'],
             'rmse': r['metrics']['rmse']}
        for k in ('baseline', 'ML_70', 'ML_85', 'ML_90'):
            t = r['trading'].get(k)
            if t:
                e[k] = {'wr': round(t['win_rate'], 4), 'sharpe': round(t['sharpe'], 2),
                        'n': t['n_trades']}
        out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default='10_30')
    args = ap.parse_args()

    base62 = list(tm.FEATURE_COLS)
    honest = [f for f in base62 if f not in LEAKY_FEATURES] + PIT_FEATURES
    arms = {'leaky': base62, 'honest': honest}
    print(f'leaky arm: {len(base62)} features | honest arm: {len(honest)} features '
          f'(dropped {len(LEAKY_FEATURES)}, added {len(PIT_FEATURES)})')

    tm.ACTIVE_TIER = args.tier
    tm.ACTIVE_TARGET = 'sr'
    tm.LABEL_COL = 'actual_return'
    tm.PURGE_OVERLAP = True   # honest validation for both arms

    tier_tag = f'_{args.tier}' if args.tier != '10_30' else ''
    with open(os.path.join(_REPO, 'results', f'v2_tuned_params{tier_tag}.json')) as f:
        params = json.load(f)

    union = list(dict.fromkeys(base62 + honest))
    df = load_once(args.tier, union)

    report = {'tier': args.tier, 'purged': True,
              'leaky_features_dropped': LEAKY_FEATURES,
              'pit_features_added': PIT_FEATURES}
    for name, cols in arms.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f'{name}: missing columns {missing}')
        print(f'\n{"#"*20} ARM: {name} ({len(cols)} features) {"#"*20}')
        tm.PRED_SUFFIX = f'_ab_{name}'
        wf = tm.walk_forward_train(df, cols, params, save_predictions=True)
        report[name] = summarize(wf)
        gc.collect()

    # head-to-head deltas
    deltas = []
    for a, b in zip(report['leaky'], report['honest']):
        d = {'val_year': a['val_year'], 'auc_delta': round(b['auc'] - a['auc'], 4)}
        if 'ML_90' in a and 'ML_90' in b:
            d['ml90_wr_delta_pp'] = round((b['ML_90']['wr'] - a['ML_90']['wr']) * 100, 2)
        deltas.append(d)
    report['honest_minus_leaky'] = deltas

    out = os.path.join(_REPO, 'results', 'experiments', f'honest_ab_{args.tier}.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nSaved {out}')
    for d in deltas:
        print(d)


if __name__ == '__main__':
    main()
