"""Production configuration for ML Pattern Scorer service."""
import os

# Base data directory -- override with ML_SCORER_DATA_DIR env var
DATA_DIR = os.environ.get('ML_SCORER_DATA_DIR', 'C:/seasonals/data')

CSV_DIR = os.path.join(DATA_DIR, 'csv')
US_CSV_DIR = os.path.join(CSV_DIR, 'US')
ETF_CSV_DIR = os.path.join(CSV_DIR, 'ETF')
INDX_CSV_DIR = os.path.join(CSV_DIR, 'INDX')
OPP_BY_SYMBOL_DIR = os.path.join(DATA_DIR, 'sp500', 'opp_by_symbol')
EARNINGS_DIR = os.environ.get('ML_SCORER_EARNINGS_DIR',
                               os.path.join(os.path.dirname(DATA_DIR), 'edgar', 'earnings'))

# Model/calibration dirs (relative to this package)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PACKAGE_DIR, 'models')
CALIBRATION_DIR = os.path.join(PACKAGE_DIR, 'calibration')

# Service
HOST = os.environ.get('ML_SCORER_HOST', '0.0.0.0')
PORT = int(os.environ.get('ML_SCORER_PORT', 5090))

# Pattern depth
MAX_DEPTH_CAP = 35

# SPX seasonal
SPX_SEASONAL_FORWARD_DAYS = 15

# VIX hurricane cutoff -- refuse to score when VIX > this
VIX_CUTOFF = 35

# Supported tiers: map tier name -> model file patterns
# Add new tiers here as models are trained
TIERS = {
    '10_30': {
        'sr': {
            'lgb': 'v2_lgb_20260312.txt',
            'xgb': 'v2_xgb_20260312.json',
            'catboost': 'v2_catboost_20260312.cbm',
        },
        'mfe': {
            'lgb': 'v2_lgb_mfe_20260312.txt',
            'xgb': 'v2_xgb_mfe_20260312.json',
            'catboost': 'v2_catboost_mfe_20260312.cbm',
        },
        'calibration_sr': 'calibration_sr.json',
        'calibration_mfe': 'calibration_mfe.json',
    },
    '31_60': {
        'sr': {
            'lgb': 'v2_lgb_31_60_20260314.txt',
            'xgb': 'v2_xgb_31_60_20260314.json',
            'catboost': 'v2_catboost_31_60_20260314.cbm',
        },
        'mfe': {
            'lgb': 'v2_lgb_31_60_mfe_20260314.txt',
            'xgb': 'v2_xgb_31_60_mfe_20260314.json',
            'catboost': 'v2_catboost_31_60_mfe_20260314.cbm',
        },
        'calibration_sr': 'calibration_sr_31_60.json',
        'calibration_mfe': 'calibration_mfe_31_60.json',
    },
    '61_90': {
        'sr': {
            'lgb': 'v2_lgb_61_90_20260314.txt',
            'xgb': 'v2_xgb_61_90_20260314.json',
            'catboost': 'v2_catboost_61_90_20260314.cbm',
        },
        'mfe': {
            'lgb': 'v2_lgb_61_90_mfe_20260314.txt',
            'xgb': 'v2_xgb_61_90_mfe_20260314.json',
            'catboost': 'v2_catboost_61_90_mfe_20260314.cbm',
        },
        'calibration_sr': 'calibration_sr_61_90.json',
        'calibration_mfe': 'calibration_mfe_61_90.json',
    },
}

# Feature columns the models expect (59 features, order must match training)
# IMPORTANT: pat_daysOut is a pattern-defining feature and MUST be included.
# A pattern = [start_date, ticker, days_out, history_years]. Never remove pat_daysOut.
FEATURE_COLS = [
    # Pattern-Intrinsic (22 -- includes pat_daysOut)
    'pat_sharpe_ratio', 'pat_avg_profit2', 'pat_direction',
    'pat_data_years', 'pat_deepest_pass', 'pat_depth_utilization',
    'pat_passes_recent_10', 'pat_recent_vs_deep_sharpe',
    'pat_num_combos_qualifying',
    'pat_pe_match', 'pat_pe_deepest', 'pat_pe_utilization',
    'pat_best_winrate', 'pat_worst_winrate', 'pat_deepest_pass_capped30',
    # V2 pattern
    'pat_consistency_std', 'pat_concurrent_count',
    'pat_neighbor_avg_wr', 'pat_sharpness', 'pat_pre_slope', 'pat_post_cliff',
    'pat_hit_last_year', 'pat_daysOut',
    # Technical (5)
    'ta_trend_long', 'ta_price_vs_sma200', 'ta_sma50_vs_sma200',
    'ta_trend_direction_match', 'ta_rvol_20',
    # Market Regime (16)
    'mkt_vix_level', 'mkt_vix_percentile_60d', 'mkt_vix_5d_change', 'mkt_vix_term_structure',
    'mkt_yield_curve_10y2y', 'mkt_yield_curve_slope',
    'mkt_credit_spread', 'mkt_credit_spread_change_20d',
    'mkt_spy_roc_20', 'mkt_spy_above_sma200',
    'mkt_adv_decl_ratio_10d', 'mkt_sector_rotation',
    'mkt_vix_regime_bucket', 'mkt_breadth_momentum',
    'mkt_fed_rate_level', 'mkt_fed_rate_direction',
    # SPX Seasonal (4)
    'mkt_spx_seasonal_wr', 'mkt_spx_seasonal_ret',
    'mkt_spx_seasonal_regime', 'mkt_spx_dir_alignment',
    # Context (2)
    'ctx_pct_from_52w_high', 'ctx_pct_from_52w_low',
    # Calendar (5)
    'cal_month', 'cal_day_of_year', 'cal_week_of_month',
    'cal_is_opex_week', 'cal_pe_year',
    # Interactions (4)
    'pat_dir_x_mkt_trend', 'pat_dir_x_sector_trend',
    'pat_depth_x_vix', 'pat_quality_x_regime',
]

# After SR retrain (2026-03-12), both SR and MFE use identical 59-feature FEATURE_COLS.
# This alias exists for backwards compatibility but should be the same as FEATURE_COLS.
FEATURE_COLS_MFE = FEATURE_COLS


def get_pe_year(year):
    """Presidential election cycle phase: 1=post-election, 2=midterm, 3=pre-election, 4=election."""
    return ((year - 2001) % 4) + 1


# Sector ETF mapping
SECTOR_ETF = {
    'Information Technology': 'XLK', 'Health Care': 'XLV', 'Financials': 'XLF',
    'Consumer Discretionary': 'XLY', 'Communication Services': 'XLC',
    'Industrials': 'XLI', 'Consumer Staples': 'XLP', 'Energy': 'XLE',
    'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB',
}

# TICKER_SECTOR: imported from parent config or defined inline
# For production, we import from the parent config_ml if available,
# otherwise fall back to a minimal version
try:
    import sys
    parent_dir = os.path.dirname(PACKAGE_DIR)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from config_ml import TICKER_SECTOR, YEAR_COMBOS, PE_COMBOS, ETF_SECTOR, ETF_CATEGORY_SECTOR_ETF
except ImportError:
    # Minimal fallback -- in production, config_ml.py should be available
    TICKER_SECTOR = {}
    ETF_SECTOR = {}
    ETF_CATEGORY_SECTOR_ETF = {}
    YEAR_COMBOS = [f"{y1}_{y2}" for y1 in range(5, 41)
                   for y2 in range(int(0.8 * y1), y1 + 1)]
    PE_COMBOS = [f"{y}_{y}_PE2" for y in range(4, 12)]
