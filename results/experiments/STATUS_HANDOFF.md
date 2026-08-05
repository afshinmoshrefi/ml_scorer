# ML Scorer Upgrade — Session Handoff (updated 2026-06-09)

## 2026-06-09 — LIVE-FLAT INVESTIGATION RESOLVED (read this first)
The live system's flat 8 weeks is explained: candidate-set selection look-ahead. The hindsight-free
premise test (tools/pit_engine.py + tools/pit_verdict.py, adversarially audited; conventions verified
100% vs training labels on 8k rows) measured the honest seasonal edge:
- LONG: within same stock+year+horizon (survivor momentum removed): **+2.08pp WR, HAC t~2, identical
  across S&P-475 and all-US-2346 universes**. Raw qualified-vs-never lifts (3-5pp) overstate it because
  qualification doubles as a stock-quality screen. Honest q10_8 long base: ~58-61% WR, +2.4%/trade gross.
- SHORT: +1.3pp within-stock (not significant); 44-49% WR, negative avg returns. NOT tradable standalone.
- The 13pp qualify/not gap in the mined training data and the 0.78-0.84 top-decile win_prob were
  predominantly look-ahead. corr(pit_deepest, pat_deepest_pass) = 0.227 -- the mined depth barely agrees
  with the honest depth.
Verdicts: results/experiments/pit_verdict2.json (S&P), pit_verdict_allus.json (all-US, final control).
Memory: project_live_overfit_investigation.md (running state).

HONEST A/B RESULT (complete): purged-WF on identical 34.7M rows, 10_30 SR. LEAKY arm (production 62
features): AUC 0.633, ML_90 WR 83.2%. HONEST arm (17 opp-derived -> 7 PIT features): AUC 0.549, ML_90 WR
75.5%. **Feature leak = -0.084 AUC / -7.7pp ML_90.** Honest arm still row-selection-inflated (base 67.6%
vs ~58% hindsight-free), so live top-decile expectation is high-50s/low-60s WR -> after costs ~flat: the
live result is fully explained at both layers. Honest AUC collapses in recent years (2023 0.52, 2024 0.47,
2025 0.55). Artifacts: honest_ab_10_30.json, wf_predictions_sr_ab_*.parquet, training_data_10_30_pit.parquet.

Pipeline changes landed 2026-06-09: --purge-overlap (purged WF) in train_model.py; get_feature_names()
stale-list fix (3 commodity features) in ml_scorer/feature_engine.py; pit_engine/pit_join/pit_verdict
tools. DECISION GUIDANCE: long-only; size/cost for a ~2-3pp edge; rebuild calibration from honest runs;
upstream re-mine with FDR control remains the structural fix.

---

# (2026-06-08 session notes below)

Status checkpoint for the "peak performance & reliability" effort. Plan:
`C:\Users\afshi\.claude\plans\resilient-watching-scroll.md`. Primary objective:
**calibration reliability of `win_prob`** (guardrails: ML_85/90 WR, ML_70 Sharpe, AUC).

## TL;DR honest finding
The scorer is already well-built. Measured on the existing 11M-row walk-forward OOF
predictions: **calibration is already excellent** — every populated decile's predicted
vs realized win-rate gap is < 0.25pp, including the top decile (0.825 vs 0.824). So the
calibration *method* is a minor lever (LOYO Brier varies only 0.2114–0.2121 across all
methods; AUC is unchanged by any calibrator). The real levers for the Brier objective are
**resolution (model AUC)** and **serving fidelity (train/serve parity)**. Don't expect a
"dramatic" calibration jump — the seasonal signal ceiling is ~0.62 AUC. Dramatic headroom,
if anywhere, is in **Phase 2 (new features / point-in-time candidates)**.

## Baseline (locked) — `results/experiments/scoreboard_baseline_sr_10_30.json`
10_30 SR, pooled 2018-2025 WF OOF: AUC 0.6234; LOYO-isotonic Brier 0.2116, ECE 0.0214;
ML_70 WR 0.791 / Sharpe 8.28; ML_85 0.816 / 9.76; ML_90 0.825 / 10.39.

## RESULTS — SR tuned ensemble, ALL 3 TIERS VALIDATED (walk-forward, vs baseline)
Recipe shipped: `--tune-ensemble` (XGBoost + CatBoost given their own native Optuna search; CatBoost
depth no longer fixed at 6) + Platt win_prob. **Decision: keep simple-mean ensemble** (learned weights add
only LOYO-logloss -0.0007 to -0.0026; not worth the prediction-path complexity). Scoreboards in
`results/experiments/scoreboard_v4_tuned_sr_*.json`.

| Tier  | AUC          | Brier(LOYO)     | ML_70 WR        | ML_90 WR / Sharpe        |
|-------|--------------|-----------------|-----------------|--------------------------|
| 10_30 | .6234->.6291 | .2116->.2107    | .791->.794      | .825->.833 / 10.39->10.83|
| 31_60 | ~.606->.627  | .2098->.2086    | .781->.792      | .826->.842 / 11.22->11.97|
| 61_90 | .5906->.6115 | .2097->.2058    | .764->.794      | .823->.837 / 10.90->11.51|

Consistent gains on every guardrail, largest where the baseline was weakest (61_90: AUC +0.021, ML_70 WR
+3.0pp, Brier -0.0038). Brier (the calibration-reliability objective) improved on all three tiers. Tuning
fixed the previously-untuned XGBoost (on 10_30 it went from worst, AUC 0.587, to best individual, 0.621).

The Platt calibration files (`calibration_sr*_v4.json`) were built from these WF preds and are deployable.

### End-to-end verification (DONE)
New tuned SR models load (feature validation 62/62 SR+MFE), Platt active (`win_prob: platt`), and a real
qualifying pattern scores cleanly: AAPL 2026-04-08 29d l -> nan=1 (only the legitimately-sparse
pat_recent_vs_deep_sharpe), win_prob 0.7433, ml_score 76.1, pred_ret 1.90, pred_mfe 8.29. NaN guard works.
BUG FIXED: `ml_scorer/config.py` `OPP_BY_SYMBOL_DIR` hardcoded `sp500`; now auto-detects `sp500_` (dev) like
ML_PARQUET_MARKETS does. Previously the gzip opp-fallback found no patterns on the dev box (all pattern
features NaN) -- surfaced by the new feature_nan_count guard. Production (server uses `sp500`) was unaffected.

STAGED (ready, NOT activated): new SR models `models/v2_*_20260608.*` and `results/calibration_sr*_v4.json`
were copied into `ml_scorer/ml_scorer/{models,calibration}/` (coexist with current; config.py TIERS still
points to the 20260403/04 models, so the dev service is unchanged). To ACTIVATE in the dev package: rename
`calibration_sr*_v4.json` -> `calibration_sr*.json` and bump the SR model dates in config.py TIERS to 20260608.

### Deploy (SR only, after final models finish + user sign-off)
1. Final deployable models training now: `final_sr_{10_30,31_60,61_90}.log` -> save to `C:/seasonals/ml_scorer/models/`
   as `v2_lgb[_tier]_20260608.txt`, `v2_xgb...json`, `v2_catboost...cbm` (use today's tuned params via self-load).
2. Copy new SR models + `calibration_sr*_v4.json` (rename to `calibration_sr*.json`) into
   `ml_scorer/ml_scorer/{models,calibration}/`; update SR model dates in `ml_scorer/config.py` TIERS.
3. Platt win_prob activates automatically (calibration JSON now carries a `"platt"` block); scorer logs `win_prob: platt`.
4. Restart service; verify `/health` (feature_count 62) and `/score` (finite scores, win_prob sane, `feature_nan_count` present).

### Deferred (clean follow-ups)
- MFE retraining: same untuned-models defect, but tuning needs a magnitude objective (Spearman/RMSE vs
  mfe_return), not the win/loss AUC used for SR. Add a `metric=` arg to `tune_xgb/tune_catboost`, then retrain.
- Parity calendar fix for `mkt_fed_rate_direction` (#8 importance, ~0.225 skew) + commodity ROCs; re-verify
  with `tools/parity_test.py`. Low quantitative impact (GBDTs robust to small input shifts).
- Phase 2 (real AUC headroom): new features + point-in-time candidates + survivorship; needs ~10-12h rebuilds.

## Code changes this session (all compile; production scorer runtime-verified)
- **train_model.py**
  - `load_training_data`: keeps `date` (was dropped) → dated WF preds + cluster weighting.
  - WF save: now writes per-model OOF preds `pred_lgb/pred_xgb/pred_cb` + `date`.
  - **Per-model tuning** `tune_xgb` / `tune_catboost` (multi-window AUC); `train_xgb`/
    `train_catboost` self-load `results/v2_tuned_params_{xgb,catboost}*.json`. CatBoost
    depth no longer hardcoded to 6. Gate: `--tune-ensemble`.
  - **Cluster weighting** `1/cluster_size` (symbol+month-day+direction). Gate: `--cluster-weight`.
  - **Platt calibration**: `build_calibration_tables` now fits `sigmoid(a*pred+b)` for
    win_prob and stores it under `"platt"` in the calibration JSON.
  - `--pred-suffix` isolates experiment outputs (wf_predictions / calibration / results).
- **ml_scorer/scorer.py** (production, backward compatible)
  - Uses Platt `win_prob` when calibration JSON has a `"platt"` block; else binned (current).
  - Adds `feature_nan_count` to output + warns when >8 features are NaN (silent-degradation guard).
- **tools/scorer_eval.py** (new): canonical scoreboard (AUC, Brier/ECE/MCE/logloss, ML_70/85/90
  WR+Sharpe) + `fit_ensemble_weights()` for Phase 1c. Validated.
- **tools/calibration_lab.py**, **tools/parity_test.py** (new, from agents).

## Key findings
- **Calibration recipe = Platt (logistic).** Most consistent across tiers; clear win on the
  worst tier 61_90 (Brier 0.2106→0.2078, MCE 0.229→0.053), neutral on 10_30. `isotonic_by_dir`
  was best-MCE on 10_30 but WORST on 31_60/61_90 — rejected. Files:
  `results/experiments/calibration_lab_{10_30,31_60,61_90}.csv`.
- **XGBoost is the weak, untuned ensemble member** (smoke AUC 0.587 vs CatBoost 0.614) — 1b
  tuning + 1c ensemble weights should lift ensemble resolution.
- **Parity skews** (`results/experiments/parity_report_*.csv`), by feature importance:
  - `mkt_fed_rate_direction` (#8 importance): real ~0.225 skew — WORTH FIXING (serving `iloc[-61]`
    on IRX's own index vs training `diff(60)` on the ffilled master calendar).
  - `mkt_cl/dxy/gc_roc_20` (#16/17/19): moderate 0.02–0.035 calendar-lookback skews.
  - `mkt_credit_spread` (#4): negligible (0.0019).
  - `pat_concurrent_count` (train≈19 vs serve≈350): real definitional skew (serving isn't
    tier-scoped to the daysOut range) BUT it's the #58/62 least-important feature — low priority.
  - Pattern-feature skews on old dates (e.g. MAA@2000) are historical-replay artifacts — they
    align when scoring current dates (production case).

## CURRENTLY RUNNING (detached, NOT harness-tracked)
`python train_model.py --tier 10_30 --target sr --wf-only --tune-ensemble --save-predictions --pred-suffix _v4`
Log: `results/experiments/retrain_v4_sr_10_30.log`. Started ~10:26, ETA ~2.5–4h.
Produces (isolated): `results/wf_predictions_sr_v4.parquet` (with per-model preds+date),
`results/calibration_sr_v4.json` (with Platt), `results/v2_tuned_params_{xgb,catboost}.json`.
Does NOT touch baseline or production. Check progress: `tail results/experiments/retrain_v4_sr_10_30.log`.

## RESUME STEPS (after `_v4` retrain finishes)
1. **Score the tuned WF vs baseline**:
   `python tools/scorer_eval.py --preds results/wf_predictions_sr_v4.parquet --tag v4_tuned_sr_10_30 --by-year`
   Compare AUC + ML_70/85/90 to `scoreboard_baseline_sr_10_30.json`. Guardrail: must not regress.
2. **Fit ensemble weights (1c)** on the v4 per-model preds:
   `python -c "import sys;sys.path.insert(0,'tools');import pandas as pd,scorer_eval as se;d=pd.read_parquet('results/wf_predictions_sr_v4.parquet');w,ll,m=se.fit_ensemble_weights(d[['pred_lgb','pred_xgb','pred_cb']].values,d);print('weights',w,'logloss',ll,'vs mean',m)"`
   If weighted logloss < mean by a meaningful margin, wire weights into `scorer.py._predict_ensemble`
   and `train_model.predict_ensemble` (store in calibration JSON or a weights file; default mean if absent).
3. If v4 beats baseline (or matches with better calibration): repeat for **31_60 and 61_90 SR**
   (`--tier 31_60/61_90`), then the **MFE** target (note: for MFE the AUC-vs-hit tuning objective is a
   cross-metric proxy — consider a Spearman/RMSE objective instead before trusting MFE tuning).
4. **Train final models** (drop `--wf-only`) for the winning recipe; they save to
   `C:/seasonals/ml_scorer/models/` (outer dir, NOT the production package).
5. **Parity fixes (1f)**: align serving `mkt_fed_rate_direction` + `*_roc_20` to training's
   master-calendar ffill semantics in `ml_scorer/feature_engine.py`; re-run
   `python tools/parity_test.py` to confirm skews drop.
6. **Deploy (only with user sign-off)**: copy new `models/v2_*` + `calibration_*` into
   `ml_scorer/ml_scorer/{models,calibration}/`, update `TIERS` dates in `ml_scorer/config.py`,
   restart service. The Platt `win_prob` activates automatically once a calibration JSON with a
   `"platt"` block is deployed.

## Safety / rollback
- Baseline WF preds + calibration + WF-results JSONs snapshotted in `results/baseline_snapshot/`.
- All experiment outputs use `_v4` (or other) suffixes — production paths untouched.
- Production code edits are backward compatible (binned win_prob if no Platt; mean ensemble if no weights).
- `_smoke_10_30.parquet` (6-symbol subsample) retained for fast smoke tests.

## Honest expectation
Phase-1 model-layer gains will be **modest** (resolution-bound by weak signal) plus **reliability
hardening** (Platt tails on 61_90, NaN guard, parity). The big swing is **Phase 2** (new signal:
DXY/CL/GC 60-day ROC, realized vol, deeper pattern history; point-in-time candidate generation;
survivorship), which needs the ~10–12h data rebuilds.
