"""
Train ML Pattern Scorer V2.

V2 Changes:
  - Extended walk-forward: Train 2012-201X, Validate 2018-2025
  - Regression target (actual_return) instead of binary classification
  - Optuna hyperparameter tuning
  - Ensemble: LightGBM + XGBoost + CatBoost
  - Updated feature set (~50 features vs 73 in V1)

Walk-forward validation:
  Window 1: Train 2012-2017, Validate 2018
  Window 2: Train 2012-2018, Validate 2019
  Window 3: Train 2012-2019, Validate 2020
  Window 4: Train 2012-2020, Validate 2021
  Window 5: Train 2012-2021, Validate 2022
  Window 6: Train 2012-2022, Validate 2023
  Window 7: Train 2012-2023, Validate 2024
  Window 8: Train 2012-2024, Validate 2025 (holdout)
"""

import os
import sys
import time
import json
import logging
import argparse
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import optuna
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_squared_error
)

from config_ml import FEATURE_CACHE_DIR, MODEL_DIR, RESULTS_DIR, N_JOBS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================================================================
# Configuration
# ======================================================================

# Tier definitions (must match build_training_data.py)
TIERS = {
    '10_30':  (10, 30),
    '31_60':  (31, 60),
    '61_90':  (61, 90),
    '91_120': (91, 120),
    '121_200': (121, 200),
    '201_300': (201, 300),
}

# Default tier
ACTIVE_TIER = '10_30'

def get_training_data_path(tier=None):
    if tier is None:
        tier = ACTIVE_TIER
    dmin, dmax = TIERS[tier]
    return os.path.join(FEATURE_CACHE_DIR, f'training_data_{dmin}_{dmax}.parquet')

TRAINING_DATA_PATH = get_training_data_path()

# V2 Feature set (~50 features: kept V1 features with importance + new V2 features)
FEATURE_COLS = [
    # Group 1: Pattern-Intrinsic (V1 kept: 15 + V2 new: 6 = 21)
    'pat_sharpe_ratio', 'pat_avg_profit2', 'pat_direction',
    'pat_data_years', 'pat_deepest_pass', 'pat_depth_utilization',
    'pat_passes_recent_10', 'pat_recent_vs_deep_sharpe',
    'pat_num_combos_qualifying',
    'pat_pe_match', 'pat_pe_deepest', 'pat_pe_utilization',
    'pat_best_winrate', 'pat_worst_winrate', 'pat_deepest_pass_capped30',
    # V2 new pattern features
    'pat_consistency_std', 'pat_concurrent_count',
    'pat_neighbor_avg_wr', 'pat_sharpness', 'pat_pre_slope', 'pat_post_cliff',
    'pat_hit_last_year', 'pat_daysOut',
    # Group 2: Technical (V1 kept: 5)
    'ta_trend_long', 'ta_price_vs_sma200', 'ta_sma50_vs_sma200',
    'ta_trend_direction_match', 'ta_rvol_20',
    # Group 3: Market Regime (V1 kept: 11 + V2 new: 4 = 15)
    'mkt_vix_level', 'mkt_vix_percentile_60d', 'mkt_vix_5d_change', 'mkt_vix_term_structure',
    'mkt_yield_curve_10y2y', 'mkt_yield_curve_slope',
    'mkt_credit_spread', 'mkt_credit_spread_change_20d',
    'mkt_spy_roc_20', 'mkt_spy_above_sma200',
    'mkt_adv_decl_ratio_10d', 'mkt_sector_rotation',
    # V2 new regime features
    'mkt_vix_regime_bucket', 'mkt_breadth_momentum',
    'mkt_fed_rate_level', 'mkt_fed_rate_direction',
    # V2 SPX seasonal regime features
    'mkt_spx_seasonal_wr', 'mkt_spx_seasonal_ret',
    'mkt_spx_seasonal_regime', 'mkt_spx_dir_alignment',
    # Group 4: Context (V1 kept: 2)
    'ctx_pct_from_52w_high', 'ctx_pct_from_52w_low',
    # Group 5: Calendar (V1 kept: 5)
    'cal_month', 'cal_day_of_year', 'cal_week_of_month',
    'cal_is_opex_week', 'cal_pe_year',
    # Group 6: V2 Interaction features (4)
    'pat_dir_x_mkt_trend', 'pat_dir_x_sector_trend',
    'pat_depth_x_vix', 'pat_quality_x_regime',
]

# V2: Regression target
LABEL_COL = 'actual_return'
# Keep hit_target for binary evaluation metrics
BINARY_LABEL_COL = 'hit_target'

TRAIN_START = 2000

# V2: Walk-forward windows (8 windows, validate same years as before for comparability)
WALK_FORWARD_WINDOWS = [
    {'train_end': 2017, 'val_year': 2018},
    {'train_end': 2018, 'val_year': 2019},
    {'train_end': 2019, 'val_year': 2020},
    {'train_end': 2020, 'val_year': 2021},
    {'train_end': 2021, 'val_year': 2022},
    {'train_end': 2022, 'val_year': 2023},
    {'train_end': 2023, 'val_year': 2024},
    {'train_end': 2024, 'val_year': 2025},
]

# PE-cycle walk-forward: train on same PE-phase years, validate next instance
# PE phases: 0=election, 1=post-election, 2=midterm, 3=pre-election
# With 2000-2025 data: progressive expanding windows within each phase
PE_CYCLE_WINDOWS = [
    # Phase 0 (election): 2000, 2004, 2008, 2012, 2016, 2020, 2024
    {'train_years': [2000, 2004, 2008, 2012, 2016],       'val_year': 2020, 'phase': 'PE0_elec_5cyc'},
    {'train_years': [2000, 2004, 2008, 2012, 2016, 2020], 'val_year': 2024, 'phase': 'PE0_elec_6cyc'},
    # Phase 1 (post-election): 2001, 2005, 2009, 2013, 2017, 2021, 2025
    {'train_years': [2001, 2005, 2009, 2013, 2017],       'val_year': 2021, 'phase': 'PE1_post_5cyc'},
    {'train_years': [2001, 2005, 2009, 2013, 2017, 2021], 'val_year': 2025, 'phase': 'PE1_post_6cyc'},
    # Phase 2 (midterm): 2002, 2006, 2010, 2014, 2018, 2022
    {'train_years': [2002, 2006, 2010, 2014],       'val_year': 2018, 'phase': 'PE2_mid_4cyc'},
    {'train_years': [2002, 2006, 2010, 2014, 2018], 'val_year': 2022, 'phase': 'PE2_mid_5cyc'},
    # Phase 3 (pre-election): 2003, 2007, 2011, 2015, 2019, 2023
    {'train_years': [2003, 2007, 2011, 2015],       'val_year': 2019, 'phase': 'PE3_pre_4cyc'},
    {'train_years': [2003, 2007, 2011, 2015, 2019], 'val_year': 2023, 'phase': 'PE3_pre_5cyc'},
]

# Default LightGBM parameters (overridden by Optuna if tuning is run)
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 100,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'max_bin': 255,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

NUM_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50

# VIX hurricane filter: exclude samples where VIX > threshold from training & evaluation
# Seasonal patterns don't hold during extreme volatility regimes
VIX_CUTOFF = 35  # None to disable

# SPX seasonal pre-filter: remove trades that go against strong seasonal tendency
# Tested: hurts model (less training data outweighs cleaner signal). Keep as feature only.
FILTER_AGAINST_SEASON = False  # enable with --season-filter CLI flag

# Split-direction: features to drop when training separate long/short models
# pat_direction is constant within each split (always 1 or always 0)
SPLIT_DROP_FEATURES = ['pat_direction']


# ======================================================================
# Evaluation helpers
# ======================================================================

def evaluate_regression_as_binary(y_return, y_pred_return, y_hit_target):
    """
    Evaluate regression predictions using binary classification metrics.
    V2: Model predicts return, we threshold at 0 to get binary predictions.
    Also compute AUC using predicted return as a score.
    """
    y_pred_binary = (y_pred_return > 0).astype(int)
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_return, y_pred_return))),
        'accuracy': accuracy_score(y_hit_target, y_pred_binary),
        'precision': precision_score(y_hit_target, y_pred_binary, zero_division=0),
        'recall': recall_score(y_hit_target, y_pred_binary, zero_division=0),
        'f1': f1_score(y_hit_target, y_pred_binary, zero_division=0),
        'n_samples': len(y_return),
        'n_positive': int(y_hit_target.sum()),
        'base_rate': float(y_hit_target.mean()),
        'pred_return_mean': float(y_pred_return.mean()),
        'pred_return_std': float(y_pred_return.std()),
    }
    # AUC: use predicted return as score for ranking
    try:
        metrics['auc_roc'] = roc_auc_score(y_hit_target, y_pred_return)
    except ValueError:
        metrics['auc_roc'] = 0.5
    return metrics


def evaluate_trading_performance(df, score_col='ml_score'):
    """
    Evaluate trading performance of ML-filtered vs unfiltered.
    V2: score_col is predicted return (regression). Threshold by percentile score 0-100.
    df must have: actual_return, hit_target, ml_score
    """
    results = {}

    # Unfiltered baseline
    results['baseline'] = {
        'n_trades': len(df),
        'win_rate': df['hit_target'].mean(),
        'avg_return': df['actual_return'].mean(),
        'median_return': df['actual_return'].median(),
        'sharpe': df['actual_return'].mean() / df['actual_return'].std() * np.sqrt(252) if df['actual_return'].std() > 0 else 0,
        'max_drawdown_pct': _max_drawdown(df['actual_return']),
    }

    # ML-filtered at various percentile thresholds
    # Convert predicted returns to 0-100 percentile scores
    pct_scores = df[score_col].rank(pct=True) * 100
    for t in [50, 60, 70, 80, 85, 90]:
        filtered = df[pct_scores >= t]
        if len(filtered) < 10:
            continue
        results[f'ML_{t}'] = {
            'n_trades': len(filtered),
            'win_rate': filtered['hit_target'].mean(),
            'avg_return': filtered['actual_return'].mean(),
            'median_return': filtered['actual_return'].median(),
            'sharpe': filtered['actual_return'].mean() / filtered['actual_return'].std() * np.sqrt(252) if filtered['actual_return'].std() > 0 else 0,
            'max_drawdown_pct': _max_drawdown(filtered['actual_return']),
            'pct_of_trades': len(filtered) / len(df) * 100,
        }

    # Also filter by raw predicted return > 0 (positive prediction)
    pos_pred = df[df[score_col] > 0]
    if len(pos_pred) >= 10:
        results['ML_pos'] = {
            'n_trades': len(pos_pred),
            'win_rate': pos_pred['hit_target'].mean(),
            'avg_return': pos_pred['actual_return'].mean(),
            'median_return': pos_pred['actual_return'].median(),
            'sharpe': pos_pred['actual_return'].mean() / pos_pred['actual_return'].std() * np.sqrt(252) if pos_pred['actual_return'].std() > 0 else 0,
            'max_drawdown_pct': _max_drawdown(pos_pred['actual_return']),
            'pct_of_trades': len(pos_pred) / len(df) * 100,
        }

    return results


def _max_drawdown(returns):
    """Simple max drawdown from a series of % returns."""
    cum = (1 + returns / 100).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min() * 100 if len(dd) > 0 else 0


# ======================================================================
# Training Pipeline
# ======================================================================

def load_training_data():
    """Load and prepare training data with minimal memory footprint."""
    import pyarrow.parquet as pq
    data_path = get_training_data_path(ACTIVE_TIER)
    log.info(f"Loading training data from {data_path} (tier: {ACTIVE_TIER})")

    # Peek at column names to find available features
    schema = pq.read_schema(data_path)
    all_cols = [f.name for f in schema]
    available = [c for c in FEATURE_COLS if c in all_cols]
    missing = [c for c in FEATURE_COLS if c not in all_cols]
    if missing:
        log.warning(f"Missing {len(missing)} features: {missing}")

    # Read ONLY needed columns from parquet
    keep_cols = available + ['date', 'actual_return', 'hit_target']
    # Also load daysOut if pat_daysOut missing (can derive it)
    if 'pat_daysOut' not in all_cols and 'daysOut' in all_cols:
        keep_cols.append('daysOut')
    keep_cols = [c for c in keep_cols if c in all_cols]
    keep_cols = list(dict.fromkeys(keep_cols))  # deduplicate preserving order
    df = pd.read_parquet(data_path, columns=keep_cols)
    # Derive pat_daysOut from daysOut if needed
    if 'pat_daysOut' not in df.columns and 'daysOut' in df.columns:
        df['pat_daysOut'] = df['daysOut'].astype(np.float32)
        if 'pat_daysOut' not in available:
            available.append('pat_daysOut')
        missing = [m for m in missing if m != 'pat_daysOut']
    log.info(f"Loaded {len(df):,} samples, {len(df.columns)} columns")

    # Add year column
    df['year'] = pd.to_datetime(df['date']).dt.year.astype(np.int16)
    df.drop(columns=['date'], inplace=True)

    # Downcast float64 -> float32
    for col in available + ['actual_return']:
        if col in df.columns and df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
    gc.collect()

    # Drop features that are always NaN
    always_nan = [c for c in available if df[c].isna().all()]
    if always_nan:
        log.info(f"Dropping {len(always_nan)} always-NaN features: {always_nan}")
        df.drop(columns=always_nan, inplace=True)
        available = [c for c in available if c not in always_nan]

    # VIX hurricane filter: remove extreme volatility samples
    if VIX_CUTOFF is not None and 'mkt_vix_level' in df.columns:
        hurricane_mask = df['mkt_vix_level'] > VIX_CUTOFF
        n_removed = hurricane_mask.sum()
        log.info(f"VIX hurricane filter (>{VIX_CUTOFF}): removing {n_removed:,} samples ({n_removed/len(df)*100:.1f}%)")
        df = df[~hurricane_mask].reset_index(drop=True)
        gc.collect()

    # SPX seasonal pre-filter: remove against-season trades
    if FILTER_AGAINST_SEASON and 'mkt_spx_dir_alignment' in df.columns:
        against_mask = df['mkt_spx_dir_alignment'] == -1
        n_removed = against_mask.sum()
        log.info(f"Against-season filter: removing {n_removed:,} samples ({n_removed/len(df)*100:.1f}%)")
        df = df[~against_mask].reset_index(drop=True)
        gc.collect()

    log.info(f"Using {len(available)} features")
    log.info(f"Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    log.info(f"Regression target stats: mean={df['actual_return'].mean():.3f}%, std={df['actual_return'].std():.3f}%")
    if 'hit_target' in df.columns:
        log.info(f"Base win rate: {df['hit_target'].mean():.4f}")
    log.info(f"Year distribution:\n{df.groupby('year')['actual_return'].agg(['count', 'mean', 'std'])}")

    return df, available


# ======================================================================
# Optuna Hyperparameter Tuning
# ======================================================================

def run_optuna_tuning(df, feature_cols, n_trials=75):
    """Run Optuna hyperparameter search on a subsample."""
    log.info(f"\n{'='*60}")
    log.info(f"OPTUNA HYPERPARAMETER TUNING ({n_trials} trials)")
    log.info(f"{'='*60}")

    # Use train 2012-2022, val 2023 for tuning
    train_mask = (df['year'] <= 2022).values
    val_mask = (df['year'] == 2023).values

    X_train_full = df.loc[train_mask, feature_cols].values
    y_train_full = df.loc[train_mask, LABEL_COL].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, LABEL_COL].values

    # Subsample training data for speed (max 2M rows)
    max_rows = 2_000_000
    if len(X_train_full) > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train_full), max_rows, replace=False)
        X_train = X_train_full[idx]
        y_train = y_train_full[idx]
        log.info(f"Subsampled training: {max_rows:,} of {len(X_train_full):,}")
    else:
        X_train = X_train_full
        y_train = y_train_full

    log.info(f"Tuning train: {len(X_train):,}, val: {len(X_val):,}")

    # Fixed dataset params - must match across all trials
    ds_params = {
        'max_bin': 255,
        'min_data_in_leaf': 20,  # lowest value Optuna can pick for min_child_samples
        'feature_pre_filter': False,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols,
                         free_raw_data=False, params=ds_params)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain,
                       free_raw_data=False, params=ds_params)
    # Force construction so params are locked in
    dtrain.construct()
    dval.construct()

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
            'feature_pre_filter': False,
            'max_bin': 255,
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 500),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': 5,
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        }

        model = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )

        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        return rmse

    study = optuna.create_study(direction='minimize')
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    best = study.best_params
    log.info(f"Optuna completed in {elapsed:.1f}s")
    log.info(f"Best RMSE: {study.best_value:.4f}")
    log.info(f"Best params: {json.dumps(best, indent=2)}")

    # Build full param dict
    tuned_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'bagging_freq': 5,
        'max_bin': 255,
    }
    tuned_params.update(best)

    del dtrain, dval, X_train, X_train_full
    gc.collect()

    return tuned_params


# ======================================================================
# Ensemble Training
# ======================================================================

def train_lgb(X_train, y_train, X_val, y_val, feature_cols, params):
    """Train a LightGBM regression model."""
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

    model = lgb.train(
        params, dtrain, num_boost_round=NUM_BOOST_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(100)],
    )
    return model


def train_xgb(X_train, y_train, X_val, y_val, feature_cols, params):
    """Train an XGBoost regression model."""
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'n_jobs': -1,
        'seed': 42,
        'max_leaves': params.get('num_leaves', 127),
        'learning_rate': params.get('learning_rate', 0.05),
        'min_child_weight': params.get('min_child_samples', 100),
        'colsample_bytree': params.get('feature_fraction', 0.8),
        'subsample': params.get('bagging_fraction', 0.8),
        'reg_alpha': params.get('lambda_l1', 0.1),
        'reg_lambda': params.get('lambda_l2', 0.1),
        'max_bin': params.get('max_bin', 255),
        'grow_policy': 'lossguide',
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    model = xgb.train(
        xgb_params, dtrain, num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(dval, 'val')],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=100,
    )
    return model


def train_catboost(X_train, y_train, X_val, y_val, feature_cols, params):
    """Train a CatBoost regression model."""
    model = CatBoostRegressor(
        iterations=NUM_BOOST_ROUNDS,
        learning_rate=params.get('learning_rate', 0.05),
        depth=6,
        l2_leaf_reg=params.get('lambda_l2', 0.1),
        random_seed=42,
        verbose=100,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        task_type='CPU',
        thread_count=-1,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model


def predict_ensemble(models, X, feature_cols):
    """Average predictions from ensemble of models."""
    preds = []
    for name, model in models:
        if name == 'lgb':
            preds.append(model.predict(X))
        elif name == 'xgb':
            dm = xgb.DMatrix(X, feature_names=feature_cols)
            preds.append(model.predict(dm))
        elif name == 'catboost':
            preds.append(model.predict(X))
    return np.mean(preds, axis=0)


# ======================================================================
# Training Pipeline
# ======================================================================

def walk_forward_train_split(df, feature_cols, params):
    """Walk-forward with separate long/short models, combined evaluation."""
    results = []

    split_features = [f for f in feature_cols if f not in SPLIT_DROP_FEATURES]
    log.info(f"Split-direction mode: {len(split_features)} features (dropped {SPLIT_DROP_FEATURES})")

    is_long = (df['pat_direction'] == 1).values
    is_short = (df['pat_direction'] == 0).values
    n_long = is_long.sum()
    n_short = is_short.sum()
    log.info(f"Direction split: {n_long:,} long ({n_long/len(df)*100:.1f}%), "
             f"{n_short:,} short ({n_short/len(df)*100:.1f}%)")

    for window in WALK_FORWARD_WINDOWS:
        val_year = window['val_year']
        train_end = window['train_end']

        log.info(f"\n{'='*60}")
        log.info(f"SPLIT Window: Train {TRAIN_START}-{train_end}, Validate {val_year}")
        log.info(f"{'='*60}")

        train_mask = (df['year'] <= train_end).values
        val_mask = (df['year'] == val_year).values

        # Combined val targets
        y_val_binary_all = df.loc[val_mask, BINARY_LABEL_COL].values if BINARY_LABEL_COL in df.columns else None
        actual_return_all = df.loc[val_mask, 'actual_return'].values

        # Prediction array for combined evaluation
        all_val_idx = df.index[val_mask]
        pred_series = pd.Series(np.nan, index=all_val_idx)

        dir_results = {}
        t0_window = time.time()

        for dir_name, dir_flag in [('LONG', is_long), ('SHORT', is_short)]:
            train_dir = train_mask & dir_flag
            val_dir = val_mask & dir_flag

            X_train = df.loc[train_dir, split_features].values
            y_train = df.loc[train_dir, LABEL_COL].values
            X_val = df.loc[val_dir, split_features].values
            y_val = df.loc[val_dir, LABEL_COL].values
            y_val_binary = df.loc[val_dir, BINARY_LABEL_COL].values if BINARY_LABEL_COL in df.columns else (y_val > 0).astype(int)
            actual_return = df.loc[val_dir, 'actual_return'].values

            log.info(f"\n  --- {dir_name} ---")
            log.info(f"  Train: {len(X_train):,} (mean ret: {y_train.mean():.3f}%, WR: {(y_train > 0).mean():.3f})")
            log.info(f"  Val:   {len(X_val):,} (mean ret: {y_val.mean():.3f}%, WR: {y_val_binary.mean():.3f})")

            # Train 3 models
            t0 = time.time()
            models = []

            log.info(f"  Training LightGBM ({dir_name})...")
            lgb_model = train_lgb(X_train, y_train, X_val, y_val, split_features, params)
            models.append(('lgb', lgb_model))
            log.info(f"    LGB best iter: {lgb_model.best_iteration}")

            log.info(f"  Training XGBoost ({dir_name})...")
            xgb_model = train_xgb(X_train, y_train, X_val, y_val, split_features, params)
            models.append(('xgb', xgb_model))
            log.info(f"    XGB best iter: {xgb_model.best_iteration}")

            log.info(f"  Training CatBoost ({dir_name})...")
            cb_model = train_catboost(X_train, y_train, X_val, y_val, split_features, params)
            models.append(('catboost', cb_model))
            log.info(f"    CB best iter: {cb_model.best_iteration_}")

            train_time = time.time() - t0

            # Predict
            y_pred = predict_ensemble(models, X_val, split_features)

            # Store predictions in combined series
            val_dir_idx = df.index[val_dir]
            pred_series.loc[val_dir_idx] = y_pred

            # Evaluate this direction
            metrics = evaluate_regression_as_binary(actual_return, y_pred, y_val_binary)
            log.info(f"  {dir_name}: AUC={metrics['auc_roc']:.4f}, RMSE={metrics['rmse']:.4f}")

            val_df = pd.DataFrame({
                'hit_target': y_val_binary,
                'actual_return': actual_return,
                'ml_score': y_pred,
            })
            trading = evaluate_trading_performance(val_df)

            log.info(f"  {dir_name} trading:")
            for tname, perf in trading.items():
                log.info(f"    {tname:10s}: {perf['n_trades']:6d} trades, "
                        f"win={perf['win_rate']:.3f}, "
                        f"avg_ret={perf['avg_return']:.2f}%, "
                        f"sharpe={perf['sharpe']:.2f}")

            dir_results[dir_name.lower()] = {
                'metrics': metrics,
                'trading': trading,
                'best_lgb_iter': lgb_model.best_iteration,
                'best_xgb_iter': xgb_model.best_iteration,
                'best_cb_iter': cb_model.best_iteration_,
                'train_time': train_time,
            }

            del X_train, X_val, y_train, y_val, models
            gc.collect()

        # Combined evaluation
        y_pred_combined = pred_series.values
        if y_val_binary_all is None:
            y_val_binary_all = (actual_return_all > 0).astype(int)

        combined_metrics = evaluate_regression_as_binary(actual_return_all, y_pred_combined, y_val_binary_all)

        val_df_combined = pd.DataFrame({
            'hit_target': y_val_binary_all,
            'actual_return': actual_return_all,
            'ml_score': y_pred_combined,
        })
        combined_trading = evaluate_trading_performance(val_df_combined)

        window_time = time.time() - t0_window

        log.info(f"\n  COMBINED: AUC={combined_metrics['auc_roc']:.4f}, RMSE={combined_metrics['rmse']:.4f}")
        for tname, perf in combined_trading.items():
            log.info(f"  COMBINED {tname:10s}: {perf['n_trades']:6d} trades, "
                    f"win={perf['win_rate']:.3f}, "
                    f"avg_ret={perf['avg_return']:.2f}%, "
                    f"sharpe={perf['sharpe']:.2f}")

        results.append({
            'val_year': val_year,
            'train_end': train_end,
            'metrics': combined_metrics,
            'trading': combined_trading,
            'long': dir_results['long'],
            'short': dir_results['short'],
            'best_iteration': 0,
            'train_time': window_time,
            # Placeholders for summary compatibility
            'lgb_metrics': combined_metrics,
            'xgb_metrics': combined_metrics,
            'cb_metrics': combined_metrics,
        })

    return results


def train_final_model_split(df, feature_cols, params):
    """Train separate long/short final production ensembles."""
    split_features = [f for f in feature_cols if f not in SPLIT_DROP_FEATURES]

    log.info(f"\n{'='*60}")
    log.info(f"Training FINAL split models ({TRAIN_START}-2024)")
    log.info(f"{'='*60}")

    train_mask = (df['year'] <= 2024).values
    val_mask = (df['year'] == 2025).values
    is_long = (df['pat_direction'] == 1).values
    is_short = (df['pat_direction'] == 0).values

    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_models = {}

    # Prediction array for combined 2025 evaluation
    all_val_idx = df.index[val_mask]
    pred_series = pd.Series(np.nan, index=all_val_idx)

    for dir_name, dir_flag, suffix in [('LONG', is_long, 'long'), ('SHORT', is_short, 'short')]:
        train_dir = train_mask & dir_flag
        val_dir = val_mask & dir_flag

        X_train = df.loc[train_dir, split_features].values
        y_train = df.loc[train_dir, LABEL_COL].values
        X_val = df.loc[val_dir, split_features].values
        y_val = df.loc[val_dir, LABEL_COL].values

        log.info(f"\n  --- {dir_name} Final Model ---")
        log.info(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

        t0 = time.time()

        log.info(f"  Training LightGBM ({dir_name})...")
        lgb_model = train_lgb(X_train, y_train, X_val, y_val, split_features, params)
        lgb_path = os.path.join(MODEL_DIR, f'v2_lgb_{suffix}_{date_str}.txt')
        lgb_model.save_model(lgb_path)
        log.info(f"  LGB saved: {lgb_path} (iter {lgb_model.best_iteration})")

        log.info(f"  Training XGBoost ({dir_name})...")
        xgb_model = train_xgb(X_train, y_train, X_val, y_val, split_features, params)
        xgb_path = os.path.join(MODEL_DIR, f'v2_xgb_{suffix}_{date_str}.json')
        xgb_model.save_model(xgb_path)
        log.info(f"  XGB saved: {xgb_path} (iter {xgb_model.best_iteration})")

        log.info(f"  Training CatBoost ({dir_name})...")
        cb_model = train_catboost(X_train, y_train, X_val, y_val, split_features, params)
        cb_path = os.path.join(MODEL_DIR, f'v2_catboost_{suffix}_{date_str}.cbm')
        cb_model.save_model(cb_path)
        log.info(f"  CB saved: {cb_path} (iter {cb_model.best_iteration_})")

        models = [('lgb', lgb_model), ('xgb', xgb_model), ('catboost', cb_model)]
        all_models[suffix] = models

        # Predict on validation
        y_pred = predict_ensemble(models, X_val, split_features)
        val_dir_idx = df.index[val_dir]
        pred_series.loc[val_dir_idx] = y_pred

        # Direction-specific evaluation
        y_val_binary = df.loc[val_dir, BINARY_LABEL_COL].values if BINARY_LABEL_COL in df.columns else (y_val > 0).astype(int)
        actual_return = df.loc[val_dir, 'actual_return'].values
        metrics = evaluate_regression_as_binary(actual_return, y_pred, y_val_binary)

        val_df = pd.DataFrame({
            'hit_target': y_val_binary,
            'actual_return': actual_return,
            'ml_score': y_pred,
        })
        trading = evaluate_trading_performance(val_df)

        log.info(f"  {dir_name} 2025 holdout: AUC={metrics['auc_roc']:.4f}")
        for tname, perf in trading.items():
            log.info(f"  {dir_name} {tname:10s}: {perf['n_trades']:6d} trades, "
                    f"win={perf['win_rate']:.3f}, sharpe={perf['sharpe']:.2f}")

        log.info(f"  {dir_name} trained in {time.time()-t0:.1f}s")

        del X_train, X_val
        gc.collect()

    # Combined 2025 evaluation
    y_pred_combined = pred_series.values
    y_val_binary_all = df.loc[val_mask, BINARY_LABEL_COL].values if BINARY_LABEL_COL in df.columns else None
    actual_return_all = df.loc[val_mask, 'actual_return'].values
    if y_val_binary_all is None:
        y_val_binary_all = (actual_return_all > 0).astype(int)

    combined_metrics = evaluate_regression_as_binary(actual_return_all, y_pred_combined, y_val_binary_all)
    val_df_combined = pd.DataFrame({
        'hit_target': y_val_binary_all,
        'actual_return': actual_return_all,
        'ml_score': y_pred_combined,
    })
    combined_trading = evaluate_trading_performance(val_df_combined)

    log.info(f"\n  COMBINED 2025 holdout: AUC={combined_metrics['auc_roc']:.4f}")
    for tname, perf in combined_trading.items():
        log.info(f"  COMBINED {tname:10s}: {perf['n_trades']:6d} trades, "
                f"win={perf['win_rate']:.3f}, sharpe={perf['sharpe']:.2f}")

    # Feature importance (from long LGB model)
    lgb_long = all_models['long'][0][1]
    lgb_short = all_models['short'][0][1]
    importance = pd.DataFrame({
        'feature': split_features,
        'lgb_long_gain': lgb_long.feature_importance(importance_type='gain'),
        'lgb_short_gain': lgb_short.feature_importance(importance_type='gain'),
    }).sort_values('lgb_long_gain', ascending=False)

    importance_path = os.path.join(RESULTS_DIR, 'v2_feature_importance_split.csv')
    importance.to_csv(importance_path, index=False)
    log.info(f"\nTop 20 features (LONG model):")
    log.info(importance.sort_values('lgb_long_gain', ascending=False).head(20).to_string())
    log.info(f"\nTop 20 features (SHORT model):")
    log.info(importance.sort_values('lgb_short_gain', ascending=False).head(20).to_string())

    return all_models, importance


def walk_forward_train(df, feature_cols, params, pe_cycle=False):
    """Run walk-forward validation with ensemble (LightGBM + XGBoost + CatBoost)."""
    results = []

    windows = PE_CYCLE_WINDOWS if pe_cycle else WALK_FORWARD_WINDOWS

    for window in windows:
        val_year = window['val_year']

        if pe_cycle:
            train_years = window['train_years']
            phase = window['phase']
            log.info(f"\n{'='*60}")
            log.info(f"PE-Cycle: Train {train_years}, Validate {val_year} ({phase})")
            log.info(f"{'='*60}")
            train_mask = df['year'].isin(train_years).values
        else:
            train_end = window['train_end']
            log.info(f"\n{'='*60}")
            log.info(f"Window: Train {TRAIN_START}-{train_end}, Validate {val_year}")
            log.info(f"{'='*60}")
            train_mask = (df['year'] <= train_end).values

        val_mask = (df['year'] == val_year).values

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, LABEL_COL].values
        X_val = df.loc[val_mask, feature_cols].values
        y_val = df.loc[val_mask, LABEL_COL].values
        y_val_binary = df.loc[val_mask, BINARY_LABEL_COL].values if BINARY_LABEL_COL in df.columns else (y_val > 0).astype(int)
        actual_return = df.loc[val_mask, 'actual_return'].values

        if len(X_val) == 0:
            log.warning(f"No validation data for {val_year}, skipping")
            continue

        log.info(f"Train: {len(X_train):,} samples (mean ret: {y_train.mean():.3f}%)")
        log.info(f"Val:   {len(X_val):,} samples (mean ret: {y_val.mean():.3f}%, win rate: {y_val_binary.mean():.4f})")

        # Train all 3 models
        t0 = time.time()
        models = []

        log.info("Training LightGBM...")
        lgb_model = train_lgb(X_train, y_train, X_val, y_val, feature_cols, params)
        models.append(('lgb', lgb_model))
        log.info(f"  LGB best iteration: {lgb_model.best_iteration}")

        log.info("Training XGBoost...")
        xgb_model = train_xgb(X_train, y_train, X_val, y_val, feature_cols, params)
        models.append(('xgb', xgb_model))
        log.info(f"  XGB best iteration: {xgb_model.best_iteration}")

        log.info("Training CatBoost...")
        cb_model = train_catboost(X_train, y_train, X_val, y_val, feature_cols, params)
        models.append(('catboost', cb_model))
        log.info(f"  CatBoost best iteration: {cb_model.best_iteration_}")

        train_time = time.time() - t0
        log.info(f"All 3 models trained in {train_time:.1f}s")

        # Predict: ensemble average
        y_pred = predict_ensemble(models, X_val, feature_cols)

        # Also get individual model predictions for comparison
        lgb_pred = lgb_model.predict(X_val)
        xgb_dm = xgb.DMatrix(X_val, feature_names=feature_cols)
        xgb_pred = xgb_model.predict(xgb_dm)
        cb_pred = cb_model.predict(X_val)

        # Evaluate
        metrics = evaluate_regression_as_binary(actual_return, y_pred, y_val_binary)
        lgb_metrics = evaluate_regression_as_binary(actual_return, lgb_pred, y_val_binary)
        xgb_metrics = evaluate_regression_as_binary(actual_return, xgb_pred, y_val_binary)
        cb_metrics = evaluate_regression_as_binary(actual_return, cb_pred, y_val_binary)

        log.info(f"Val metrics (ensemble): AUC={metrics['auc_roc']:.4f}, RMSE={metrics['rmse']:.4f}, "
                f"Acc={metrics['accuracy']:.4f}")
        log.info(f"  LGB only: AUC={lgb_metrics['auc_roc']:.4f}, XGB only: AUC={xgb_metrics['auc_roc']:.4f}, "
                f"CB only: AUC={cb_metrics['auc_roc']:.4f}")

        # Trading performance
        val_df = pd.DataFrame({
            'hit_target': y_val_binary,
            'actual_return': actual_return,
            'ml_score': y_pred,
        })
        trading = evaluate_trading_performance(val_df)

        log.info(f"\nTrading performance ({val_year}):")
        for name, perf in trading.items():
            log.info(f"  {name:10s}: {perf['n_trades']:6d} trades, "
                    f"win={perf['win_rate']:.3f}, "
                    f"avg_ret={perf['avg_return']:.2f}%, "
                    f"sharpe={perf['sharpe']:.2f}")

        del X_train, X_val, y_train, y_val, models
        gc.collect()

        result_entry = {
            'val_year': val_year,
            'metrics': metrics,
            'lgb_metrics': lgb_metrics,
            'xgb_metrics': xgb_metrics,
            'cb_metrics': cb_metrics,
            'trading': trading,
            'best_iteration': lgb_model.best_iteration,
            'train_time': train_time,
        }
        if pe_cycle:
            result_entry['train_years'] = window['train_years']
            result_entry['phase'] = window['phase']
        else:
            result_entry['train_end'] = train_end
        results.append(result_entry)

    return results


def train_final_model(df, feature_cols, params):
    """Train the final production ensemble on all data through 2024."""
    log.info(f"\n{'='*60}")
    log.info(f"Training FINAL ensemble ({TRAIN_START}-2024)")
    log.info(f"{'='*60}")

    train_mask = (df['year'] <= 2024).values
    val_mask = (df['year'] == 2025).values

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, LABEL_COL].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, LABEL_COL].values
    y_val_binary = df.loc[val_mask, BINARY_LABEL_COL].values if BINARY_LABEL_COL in df.columns else (y_val > 0).astype(int)

    log.info(f"Train: {len(X_train):,} samples")
    log.info(f"Val (2025 holdout): {len(X_val):,} samples")

    # Train all 3 models
    t0 = time.time()
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs(MODEL_DIR, exist_ok=True)

    log.info("Training final LightGBM...")
    lgb_model = train_lgb(X_train, y_train, X_val, y_val, feature_cols, params)
    tier_tag = f'_{ACTIVE_TIER}' if ACTIVE_TIER != '10_30' else ''
    lgb_path = os.path.join(MODEL_DIR, f'v2_lgb{tier_tag}_{date_str}.txt')
    lgb_model.save_model(lgb_path)
    log.info(f"LGB saved: {lgb_path} (best iter: {lgb_model.best_iteration})")

    log.info("Training final XGBoost...")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val, feature_cols, params)
    xgb_path = os.path.join(MODEL_DIR, f'v2_xgb{tier_tag}_{date_str}.json')
    xgb_model.save_model(xgb_path)
    log.info(f"XGB saved: {xgb_path} (best iter: {xgb_model.best_iteration})")

    log.info("Training final CatBoost...")
    cb_model = train_catboost(X_train, y_train, X_val, y_val, feature_cols, params)
    cb_path = os.path.join(MODEL_DIR, f'v2_catboost{tier_tag}_{date_str}.cbm')
    cb_model.save_model(cb_path)
    log.info(f"CatBoost saved: {cb_path} (best iter: {cb_model.best_iteration_})")

    log.info(f"Final ensemble trained in {time.time()-t0:.1f}s")

    # Feature importance (from LightGBM)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'lgb_gain': lgb_model.feature_importance(importance_type='gain'),
        'lgb_split': lgb_model.feature_importance(importance_type='split'),
    }).sort_values('lgb_gain', ascending=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tier_tag = f'_{ACTIVE_TIER}' if ACTIVE_TIER != '10_30' else ''
    importance_path = os.path.join(RESULTS_DIR, f'v2_feature_importance{tier_tag}.csv')
    importance.to_csv(importance_path, index=False)
    log.info(f"\nTop 20 features by LGB gain:")
    log.info(importance.head(20).to_string())

    # 2025 holdout evaluation
    models = [('lgb', lgb_model), ('xgb', xgb_model), ('catboost', cb_model)]
    if len(X_val) > 0:
        y_pred = predict_ensemble(models, X_val, feature_cols)
        actual_return = df.loc[val_mask, 'actual_return'].values

        metrics = evaluate_regression_as_binary(actual_return, y_pred, y_val_binary)
        log.info(f"\n2025 holdout: AUC={metrics['auc_roc']:.4f}, RMSE={metrics['rmse']:.4f}, "
                f"Acc={metrics['accuracy']:.4f}")

        val_df = pd.DataFrame({
            'hit_target': y_val_binary,
            'actual_return': actual_return,
            'ml_score': y_pred,
        })
        trading = evaluate_trading_performance(val_df)
        log.info(f"\n2025 holdout trading performance:")
        for name, perf in trading.items():
            log.info(f"  {name:10s}: {perf['n_trades']:6d} trades, "
                    f"win={perf['win_rate']:.3f}, "
                    f"avg_ret={perf['avg_return']:.2f}%, "
                    f"sharpe={perf['sharpe']:.2f}")

    return models, importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final-only', action='store_true', help='Skip walk-forward, train final model only')
    parser.add_argument('--skip-optuna', action='store_true', help='Skip Optuna tuning, use default params')
    parser.add_argument('--optuna-trials', type=int, default=75, help='Number of Optuna trials')
    parser.add_argument('--pe-cycle', action='store_true', help='Use PE-cycle walk-forward (train on same PE-phase years)')
    parser.add_argument('--wf-only', action='store_true', help='Walk-forward only, skip final model training')
    parser.add_argument('--vix-cutoff', type=float, default=None, help='Override VIX cutoff (e.g. 30, 35, 40). 0 to disable.')
    parser.add_argument('--no-season-filter', action='store_true', help='Disable against-season pre-filter')
    parser.add_argument('--split-direction', action='store_true', help='Train separate long/short models')
    parser.add_argument('--tier', type=str, default=None, choices=list(TIERS.keys()),
                        help='DaysOut tier to train (e.g. 31_60, 91_120)')
    parser.add_argument('--all-tiers', action='store_true',
                        help='Train all 6 daysOut tiers sequentially')
    args = parser.parse_args()

    # Apply CLI overrides to globals
    global VIX_CUTOFF, FILTER_AGAINST_SEASON, ACTIVE_TIER
    if args.vix_cutoff is not None:
        VIX_CUTOFF = args.vix_cutoff if args.vix_cutoff > 0 else None
    if args.no_season_filter:
        FILTER_AGAINST_SEASON = False

    # Handle --all-tiers: run main() for each tier
    if args.all_tiers:
        for tier_name in TIERS:
            log.info(f"\n{'#'*60}")
            log.info(f"# TIER: {tier_name} (daysOut {TIERS[tier_name][0]}-{TIERS[tier_name][1]})")
            log.info(f"{'#'*60}")
            ACTIVE_TIER = tier_name
            args.all_tiers = False  # prevent recursion
            args.tier = tier_name
            main_single(args)
        return

    if args.tier:
        ACTIVE_TIER = args.tier

    main_single(args)


def main_single(args):
    """Run training pipeline for a single tier."""
    log.info("ML Pattern Scorer V2 - Training Pipeline")
    log.info("=" * 60)
    log.info(f"Tier: {ACTIVE_TIER} (daysOut {TIERS[ACTIVE_TIER][0]}-{TIERS[ACTIVE_TIER][1]})")
    if VIX_CUTOFF:
        log.info(f"VIX hurricane cutoff: {VIX_CUTOFF}")
    log.info(f"Against-season filter: {'ON' if FILTER_AGAINST_SEASON else 'OFF'}")
    if args.split_direction:
        log.info("MODE: Split-direction (separate long/short models)")

    # Load data
    df, feature_cols = load_training_data()

    # Optuna tuning
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tier_tag = f'_{ACTIVE_TIER}' if ACTIVE_TIER != '10_30' else ''
    params_path = os.path.join(RESULTS_DIR, f'v2_tuned_params{tier_tag}.json')
    if not args.skip_optuna and not args.split_direction:
        params = run_optuna_tuning(df, feature_cols, n_trials=args.optuna_trials)
        with open(params_path, 'w') as f:
            json.dump({k: v for k, v in params.items() if not callable(v)}, f, indent=2)
        log.info(f"Params saved to {params_path}")
    elif os.path.exists(params_path):
        log.info(f"Loading saved params from {params_path}")
        with open(params_path) as f:
            params = json.load(f)
        log.info(f"Params: {json.dumps(params, indent=2)}")
    else:
        log.info("Using default LGB params (Optuna skipped, no saved params)")
        params = LGB_PARAMS.copy()

    if not args.final_only:
        if args.split_direction:
            log.info(f"\n\nPHASE 1: Split-Direction Walk-Forward (6 models per window)")
            wf_results = walk_forward_train_split(df, feature_cols, params)
        else:
            mode_label = "PE-Cycle" if args.pe_cycle else "Standard"
            log.info(f"\n\nPHASE 1: {mode_label} Walk-Forward Validation (3-model ensemble)")
            wf_results = walk_forward_train(df, feature_cols, params, pe_cycle=args.pe_cycle)

        # Summary
        log.info(f"\n{'='*60}")
        mode_str = "SPLIT-DIRECTION" if args.split_direction else ("PE-CYCLE" if args.pe_cycle else "STANDARD")
        log.info(f"{mode_str} WALK-FORWARD SUMMARY")
        log.info(f"{'='*60}")
        for r in wf_results:
            m = r['metrics']
            t = r['trading']
            base = t['baseline']
            label = r.get('phase', r['val_year'])
            best_ml = None
            best_ml_name = 'none'
            for k, v in t.items():
                if k.startswith('ML_') and (best_ml is None or v['win_rate'] > best_ml['win_rate']):
                    best_ml = v
                    best_ml_name = k
            if best_ml:
                improvement = (best_ml['win_rate'] - base['win_rate']) * 100
                log.info(f"  {label}: AUC={m['auc_roc']:.3f}, RMSE={m['rmse']:.4f}, "
                        f"base_wr={base['win_rate']:.3f}, "
                        f"best_ml_wr={best_ml['win_rate']:.3f} ({best_ml_name}, {best_ml['n_trades']} trades), "
                        f"+{improvement:.1f}pp")
            else:
                log.info(f"  {label}: AUC={m['auc_roc']:.3f}, RMSE={m['rmse']:.4f}, "
                        f"base_wr={base['win_rate']:.3f}")

        # Split-direction: print per-direction summary
        if args.split_direction:
            log.info(f"\n{'='*60}")
            log.info("PER-DIRECTION BREAKDOWN")
            log.info(f"{'='*60}")
            for r in wf_results:
                yr = r['val_year']
                for dname in ['long', 'short']:
                    dm = r[dname]['metrics']
                    dt = r[dname]['trading']
                    dbase = dt['baseline']
                    best_ml = None
                    best_ml_name = 'none'
                    for k, v in dt.items():
                        if k.startswith('ML_') and (best_ml is None or v['win_rate'] > best_ml['win_rate']):
                            best_ml = v
                            best_ml_name = k
                    if best_ml:
                        log.info(f"  {yr} {dname.upper():5s}: AUC={dm['auc_roc']:.3f}, "
                                f"base_wr={dbase['win_rate']:.3f}, "
                                f"best_ml_wr={best_ml['win_rate']:.3f} ({best_ml_name}), "
                                f"sharpe={best_ml['sharpe']:.2f}")
                    else:
                        log.info(f"  {yr} {dname.upper():5s}: AUC={dm['auc_roc']:.3f}, "
                                f"base_wr={dbase['win_rate']:.3f}")

            # Load combined model results for comparison if available
            combined_path = os.path.join(RESULTS_DIR, 'v2_walk_forward_results.json')
            if os.path.exists(combined_path):
                with open(combined_path) as f:
                    combined_results = json.load(f)
                log.info(f"\n{'='*60}")
                log.info("SPLIT vs COMBINED MODEL COMPARISON")
                log.info(f"{'='*60}")
                log.info(f"  {'Year':>6s}  {'Comb AUC':>9s}  {'Split AUC':>10s}  {'Comb ML70 WR':>13s}  {'Split ML70 WR':>14s}  {'Comb ML70 Sh':>13s}  {'Split ML70 Sh':>14s}")
                for cr, sr in zip(combined_results, wf_results):
                    yr = cr['val_year']
                    c_auc = cr['metrics']['auc_roc']
                    s_auc = sr['metrics']['auc_roc']
                    c_ml70 = cr['trading'].get('ML_70', {})
                    s_ml70 = sr['trading'].get('ML_70', {})
                    c_wr = f"{c_ml70['win_rate']:.3f}" if c_ml70 else 'n/a'
                    s_wr = f"{s_ml70['win_rate']:.3f}" if s_ml70 else 'n/a'
                    c_sh = f"{c_ml70['sharpe']:.2f}" if c_ml70 else 'n/a'
                    s_sh = f"{s_ml70['sharpe']:.2f}" if s_ml70 else 'n/a'
                    auc_delta = s_auc - c_auc
                    marker = '+' if auc_delta > 0 else ''
                    log.info(f"  {yr:>6d}  {c_auc:>9.3f}  {s_auc:>10.3f}  {c_wr:>13s}  {s_wr:>14s}  {c_sh:>13s}  {s_sh:>14s}  ({marker}{auc_delta:.3f})")

    else:
        log.info("Skipping walk-forward validation (--final-only)")
        wf_results = None

    gc.collect()

    # Train final model (skip if --wf-only)
    if not args.wf_only:
        log.info("\n\nPHASE 2: Final Ensemble Training")
        if args.split_direction:
            models, importance = train_final_model_split(df, feature_cols, params)
        else:
            models, importance = train_final_model(df, feature_cols, params)
    else:
        log.info("\nSkipping final model training (--wf-only)")

    # Save walk-forward results
    if wf_results:
        tier_tag = f'_{ACTIVE_TIER}' if ACTIVE_TIER != '10_30' else ''
        suffix = '_split' if args.split_direction else ('_pe_cycle' if args.pe_cycle else '')
        results_path = os.path.join(RESULTS_DIR, f'v2_walk_forward_results{tier_tag}{suffix}.json')
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = []
        for r in wf_results:
            sr = {
                'val_year': r['val_year'],
                'best_iteration': r['best_iteration'],
                'train_time': r['train_time'],
            }
            if 'train_end' in r:
                sr['train_end'] = r['train_end']
            if 'train_years' in r:
                sr['train_years'] = r['train_years']
                sr['phase'] = r['phase']
            for key in ['metrics', 'lgb_metrics', 'xgb_metrics', 'cb_metrics']:
                sr[key] = {k: convert(v) for k, v in r[key].items()}
            sr['trading'] = {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in r['trading'].items()}
            # Save per-direction results if split mode
            if 'long' in r:
                for dname in ['long', 'short']:
                    dr = r[dname]
                    sr[dname] = {
                        'metrics': {k: convert(v) for k, v in dr['metrics'].items()},
                        'trading': {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in dr['trading'].items()},
                        'best_lgb_iter': dr['best_lgb_iter'],
                        'best_xgb_iter': dr['best_xgb_iter'],
                        'best_cb_iter': dr['best_cb_iter'],
                        'train_time': dr['train_time'],
                    }
            serializable.append(sr)

        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        log.info(f"\nResults saved to {results_path}")

    log.info("\nV2 Training Complete!")


if __name__ == '__main__':
    main()
