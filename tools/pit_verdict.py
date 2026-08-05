"""
Corrected-inference verdict for the PIT premise test.

Implements the adversarial audit's required fixes (workflow pit-leak-audit,
2026-06-09) on top of pit_engine.py's rows parquet:

  1. YEAR-CLUSTERED inference (the audit's CRITICAL): the decision statistic
     is the per-forward-year lift series (26 obs), not pooled rows. Reports
     mean lift, plain t, Newey-West HAC t (2 lags), 95% CI, and the minimum
     detectable effect at 80% power -- the design cannot reliably detect
     edges below the MDE, so non-significance there is INCONCLUSIVE, not no.
  2. MATCHED-STRATUM baseline: q10_8 == 1 vs q10_8 == 0 *within p10 == 10*
     (patterns that could have qualified but did not), removing the
     firm-age / data-continuity confound.
  3. WITHIN-(symbol, year, daysOut) contrast: qualified vs never-qualified
     month-days on the SAME stock, year and horizon -- removes stock-level
     drift and the survivor-momentum channel entirely.
  4. VIX <= 35 sensitivity (matches the training pipeline's hurricane filter).
  5. Tradability framing per direction: WR excess vs 50% break-even and avg
     return, so a short "lift" cannot be misread as a tradable edge.
  6. Context note reframed: the gap vs the training-side 13pp is an UPPER
     BOUND on the look-ahead component (universes/filters/baselines differ).

Usage:
  python tools/pit_verdict.py --rows results/experiments/pit_premise2_rows.parquet \
      --out results/experiments/pit_verdict2
"""
import argparse
import json

import numpy as np
import pandas as pd

T95 = 2.06    # two-sided 5% critical value, ~25 df
T80_POWER = 0.86


def hac_t(series, lags=2):
    """Newey-West t-stat of the mean of a short time series."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan
    e = x - x.mean()
    g0 = np.mean(e * e)
    var = g0
    for l in range(1, lags + 1):
        gl = np.mean(e[l:] * e[:-l])
        var += 2 * (1 - l / (lags + 1)) * gl
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else np.nan


def yearly_lift(df, treat_mask, base_mask, value='fwd_win'):
    """Per-year (mean[treat] - mean[base]) series; NaN where a side is empty."""
    d = df[treat_mask | base_mask]
    grp = d.groupby(['year', treat_mask[treat_mask | base_mask].values])[value].mean()
    piv = grp.unstack()
    if True not in piv.columns or False not in piv.columns:
        return pd.Series(dtype=float)
    return (piv[True] - piv[False]).dropna()


def lift_block(df, treat_mask, base_mask, label):
    """Audit-compliant inference block for one comparison."""
    wl = yearly_lift(df, treat_mask, base_mask, 'fwd_win') * 100      # pp
    rl = yearly_lift(df, treat_mask, base_mask, 'fwd_ret') * 100      # pp
    n = len(wl)
    if n < 5:
        return {'label': label, 'n_years': n, 'insufficient': True}
    se = wl.std(ddof=1) / np.sqrt(n)
    mean = wl.mean()
    return {
        'label': label,
        'n_years': int(n),
        'n_treat_rows': int(treat_mask.sum()),
        'n_base_rows': int(base_mask.sum()),
        'wr_lift_pp_yearmean': round(float(mean), 3),
        'wr_lift_t': round(float(mean / se), 2) if se > 0 else None,
        'wr_lift_t_hac2': round(hac_t(wl), 2),
        'wr_lift_ci95_pp': [round(float(mean - T95 * se), 3),
                            round(float(mean + T95 * se), 3)],
        'mde_80pct_power_pp': round(float((T95 + T80_POWER) * wl.std(ddof=1) / np.sqrt(n)), 3),
        'years_positive': int((wl > 0).sum()),
        'ret_lift_pp_yearmean': round(float(rl.mean()), 3),
        'ret_lift_t_hac2': round(hac_t(rl), 2),
    }


def within_cell_contrast(df, treat_mask, base_mask, label,
                         cell=('symbol', 'year', 'daysOut')):
    """Audit fix #3: same-stock same-year same-horizon contrast. Each cell
    containing BOTH qualified and never-qualified month-days contributes
    (mean treat - mean base); cells average within year, years average with
    HAC inference. Stock-level drift and survivor momentum cancel exactly."""
    d = df[treat_mask | base_mask].copy()
    d['is_treat'] = treat_mask[treat_mask | base_mask].values
    g = d.groupby([*cell, 'is_treat'])['fwd_win'].mean().unstack()
    g = g.dropna(subset=[True, False])
    if g.empty:
        return {'label': label, 'insufficient': True}
    g['diff'] = (g[True] - g[False]) * 100
    per_year = g.reset_index().groupby('year')['diff'].mean()
    n = len(per_year)
    se = per_year.std(ddof=1) / np.sqrt(n)
    mean = per_year.mean()
    return {
        'label': label,
        'n_cells': int(len(g)),
        'n_years': int(n),
        'wr_lift_pp_yearmean': round(float(mean), 3),
        'wr_lift_t': round(float(mean / se), 2) if se > 0 else None,
        'wr_lift_t_hac2': round(hac_t(per_year), 2),
        'years_positive': int((per_year > 0).sum()),
    }


def tradability(df, mask, label):
    d = df[mask]
    per_year_wr = d.groupby('year')['fwd_win'].mean() * 100
    per_year_ret = d.groupby('year')['fwd_ret'].mean() * 100
    n = len(per_year_wr)
    wr = per_year_wr.mean()
    se_w = per_year_wr.std(ddof=1) / np.sqrt(n)
    ret = per_year_ret.mean()
    se_r = per_year_ret.std(ddof=1) / np.sqrt(n)
    return {
        'label': label,
        'wr_pct_yearmean': round(float(wr), 2),
        'wr_excess_vs_breakeven_pp': round(float(wr - 50.0), 2),
        'wr_ci95': [round(float(wr - T95 * se_w), 2), round(float(wr + T95 * se_w), 2)],
        'avg_ret_pct_yearmean': round(float(ret), 3),
        'avg_ret_ci95': [round(float(ret - T95 * se_r), 3), round(float(ret + T95 * se_r), 3)],
        'tradable_wr_above_50': bool(wr - T95 * se_w > 50.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.rows)
    df = df[df['pit_n_prior'] >= 10].copy()
    has_p10 = 'p10' in df.columns
    has_vix = 'vix_entry' in df.columns

    out = {'rows_file': args.rows, 'n_rows': int(len(df)),
           'inference': 'year-clustered (per-year lifts), HAC(2) t, 95% CI, '
                        'MDE at 80% power; pooled row t-stats are NOT used '
                        '(audit: rows are massively cross-correlated)'}

    for direction in ('l', 's'):
        d = df[df['direction'] == direction]
        q = (d['pit_q10_8'] == 1)
        never = (d['pit_deepest'] == 0)
        deep = (d['pit_deepest'] >= 15)
        blocks = {
            'C1_q10_8_vs_never': lift_block(d, q, never, 'q10_8 vs never-qualified'),
            'C1b_deep15_vs_never': lift_block(d, deep, never, 'depth>=15 vs never-qualified'),
            'C3_within_stock_year': within_cell_contrast(
                d, q, never, 'q10_8 vs never, same symbol+year+daysOut'),
            'tradability_q10_8': tradability(d, q, 'q10_8'),
            'tradability_never': tradability(d, never, 'never-qualified'),
        }
        if has_p10:
            stratum = d['p10'] == 10
            blocks['C2_matched_stratum'] = lift_block(
                d[stratum], q[stratum], ~q[stratum] & (d.loc[stratum, 'pit_q10_8'] == 0),
                'q10_8 vs not, within p10==10 (could-have-qualified)')
        if has_vix:
            calm = d['vix_entry'] <= 35
            blocks['C4_vix_le35'] = lift_block(
                d[calm], q[calm], never[calm], 'C1 under VIX<=35 (training-filter match)')
        out[direction] = blocks

    out['context'] = {
        'note': ('The honest lift stands alone. The gap to the training-side '
                 '13pp qualify/not figure is an UPPER BOUND on the look-ahead '
                 'component: universes, VIX filtering, baseline definitions and '
                 'direction mix differ between the two measurements (audit '
                 'finding). Long-side absolute levels remain survivor-biased '
                 'upward when run on current index membership; the C3 '
                 'within-stock contrast and the --universe all rerun are the '
                 'controls for that channel.'),
    }

    with open(f'{args.out}.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f'\nSaved -> {args.out}.json')


if __name__ == '__main__':
    main()
