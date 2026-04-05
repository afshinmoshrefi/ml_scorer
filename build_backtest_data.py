"""
Build the merged backtester input dataset.

Joins:
  - training_data_{tier}.parquet (validation years 2018-2025, VIX <= 35)
  - wf_predictions_sr*.parquet (SR model predicted returns)
  - wf_predictions_mfe*.parquet (MFE model predicted MFE)
  - calibration_sr*.json (predicted_return -> win_prob, ml_score)
  - TICKER_SECTOR from config_ml.py (symbol -> GICS sector)

Also computes 14-day ATR (as % of price) for each symbol/date from price CSVs.

Output: results/backtester_input_{tier}.parquet

Usage:
    python build_backtest_data.py --tier 10_30
    python build_backtest_data.py --tier 31_60
    python build_backtest_data.py --tier 61_90
"""

import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FEATURES = ROOT / "features"
CAL_DIR = ROOT / "ml_scorer" / "calibration"
US_CSV_DIR = Path("C:/seasonals/data/csv/US")


# ============================================================
# Calibration helpers
# ============================================================

def load_calibration(path):
    with open(path) as f:
        cal = json.load(f)
    return cal["bins"]


def calibrate_win_prob(pred_values, cal_bins):
    edges = np.array([b["pred_max"] for b in cal_bins])
    win_probs = np.array([b["win_prob"] for b in cal_bins])
    bin_idx = np.searchsorted(edges, pred_values, side="right")
    bin_idx = np.clip(bin_idx, 0, len(cal_bins) - 1)
    return win_probs[bin_idx]


def calibrate_p_hit(pred_values, cal_bins, field="p_hit_pred"):
    edges = np.array([b["pred_max"] for b in cal_bins])
    p_hits = np.array([b.get(field, 0.5) for b in cal_bins])
    bin_idx = np.searchsorted(edges, pred_values, side="right")
    bin_idx = np.clip(bin_idx, 0, len(cal_bins) - 1)
    return p_hits[bin_idx]


def compute_ml_score(pred_values, cal_bins):
    n_bins = len(cal_bins)
    edges = np.array([b["pred_max"] for b in cal_bins])
    mins = np.array([b["pred_min"] for b in cal_bins])
    maxs = np.array([b["pred_max"] for b in cal_bins])

    bin_idx = np.searchsorted(edges, pred_values, side="right")
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_min = mins[bin_idx]
    bin_max = maxs[bin_idx]
    bin_range = bin_max - bin_min
    frac = np.where(bin_range > 0, (pred_values - bin_min) / bin_range, 0.5)
    frac = np.clip(frac, 0.0, 1.0)

    ml_score = (bin_idx + frac) / n_bins * 100.0
    return np.round(ml_score, 1)


def load_sector_map():
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_ml", ROOT / "config_ml.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TICKER_SECTOR


# ============================================================
# ATR computation
# ============================================================

def _atr_for_symbol(sym, csv_dir):
    """Compute 14-day ATR % for one symbol. Returns (symbol, DataFrame) or None."""
    path = csv_dir / f"{sym}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["date", "high", "low", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        prev_close = df["close"].shift(1)
        hl = df["high"] - df["low"]
        hc = (df["high"] - prev_close).abs()
        lc = (df["low"] - prev_close).abs()
        df["tr"] = np.maximum(hl, np.maximum(hc, lc))
        df["atr14"] = df["tr"].rolling(14, min_periods=10).mean()
        df["atr_14d_pct"] = (df["atr14"] / df["close"]).clip(lower=0)
        df["symbol"] = sym
        return df[["symbol", "date", "atr_14d_pct"]].dropna(subset=["atr_14d_pct"])
    except Exception:
        return None


def compute_atr_data(symbols, csv_dir=US_CSV_DIR, n_jobs=12):
    """Compute 14-day ATR as % of price for all symbols in parallel.

    Returns a DataFrame with [symbol, date, atr_14d_pct].
    """
    print(f"  Computing 14-day ATR for {len(symbols)} symbols ({n_jobs} workers)...")
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_atr_for_symbol)(sym, csv_dir) for sym in symbols
    )
    valid = [r for r in results if r is not None]
    if not valid:
        print("  WARNING: No ATR data computed")
        return pd.DataFrame(columns=["symbol", "date", "atr_14d_pct"])
    atr_df = pd.concat(valid, ignore_index=True)
    print(f"  ATR computed for {len(valid)}/{len(symbols)} symbols, {len(atr_df):,} rows")
    return atr_df


# ============================================================
# Tier configuration
# ============================================================

TIER_CONFIG = {
    "10_30": {
        "training_file": "training_data_10_30.parquet",
        "sr_pred_file": "wf_predictions_sr.parquet",
        "mfe_pred_file": "wf_predictions_mfe.parquet",
        "cal_sr": "calibration_sr.json",
        "cal_mfe": "calibration_mfe.json",
        "output_file": "backtester_input_10_30.parquet",
    },
    "31_60": {
        "training_file": "training_data_31_60.parquet",
        "sr_pred_file": "wf_predictions_sr_31_60.parquet",
        "mfe_pred_file": "wf_predictions_mfe_31_60.parquet",
        "cal_sr": "calibration_sr_31_60.json",
        "cal_mfe": "calibration_mfe_31_60.json",
        "output_file": "backtester_input_31_60.parquet",
    },
    "61_90": {
        "training_file": "training_data_61_90.parquet",
        "sr_pred_file": "wf_predictions_sr_61_90.parquet",
        "mfe_pred_file": "wf_predictions_mfe_61_90.parquet",
        "cal_sr": "calibration_sr_61_90.json",
        "cal_mfe": "calibration_mfe_61_90.json",
        "output_file": "backtester_input_61_90.parquet",
    },
}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build backtester input parquet")
    parser.add_argument("--tier", default="10_30", choices=["10_30", "31_60", "61_90"],
                        help="Which tier to build (default: 10_30)")
    parser.add_argument("--jobs", type=int, default=12,
                        help="Parallel workers for ATR computation (default: 12)")
    args = parser.parse_args()

    cfg = TIER_CONFIG[args.tier]
    t0 = time.time()
    print(f"\nBuilding backtester input for tier {args.tier}")
    print("=" * 60)

    # --- Load training data ---
    print("Loading training data...")
    keep_cols = [
        "date", "symbol", "daysOut", "direction",
        "actual_return", "mfe_return", "mkt_vix_level",
        "ta_rvol_20",
    ]
    td = pd.read_parquet(FEATURES / cfg["training_file"], columns=keep_cols)
    print(f"  Loaded {len(td):,} rows in {time.time()-t0:.1f}s")

    td["year"] = td["date"].dt.year
    td = td[(td["year"] >= 2018) & (td["mkt_vix_level"] <= 35)].copy()
    td = td.drop(columns=["mkt_vix_level"])
    print(f"  After VIX filter: {len(td):,} rows")

    # --- Load WF predictions ---
    print("Loading WF predictions...")
    sr = pd.read_parquet(RESULTS / cfg["sr_pred_file"])
    mfe = pd.read_parquet(RESULTS / cfg["mfe_pred_file"])
    print(f"  SR: {len(sr):,}, MFE: {len(mfe):,}")

    # --- Row-align per year ---
    print("Joining predictions to training data...")
    td_pieces = []
    for y in range(2018, 2026):
        td_y = td[td["year"] == y].reset_index(drop=True)
        sr_y = sr[sr["val_year"] == y].reset_index(drop=True)
        mfe_y = mfe[mfe["val_year"] == y].reset_index(drop=True)

        assert len(td_y) == len(sr_y), f"Year {y}: td={len(td_y)} sr={len(sr_y)}"
        assert len(td_y) == len(mfe_y), f"Year {y}: td={len(td_y)} mfe={len(mfe_y)}"

        td_y["predicted_return"] = sr_y["predicted"].values
        td_y["predicted_mfe"] = mfe_y["predicted"].values
        td_pieces.append(td_y)

    df = pd.concat(td_pieces, ignore_index=True)
    del td, sr, mfe, td_pieces
    print(f"  Merged: {len(df):,} rows")

    # --- Add sector ---
    print("Adding sector mapping...")
    sector_map = load_sector_map()
    df["sector"] = df["symbol"].map(sector_map).fillna("Unknown")
    unmapped = df[df["sector"] == "Unknown"]["symbol"].nunique()
    if unmapped:
        print(f"  WARNING: {unmapped} symbols without sector mapping")

    # --- Calibration ---
    print("Applying calibration...")
    cal_sr = load_calibration(CAL_DIR / cfg["cal_sr"])
    cal_mfe = load_calibration(CAL_DIR / cfg["cal_mfe"])

    df["win_probability"] = calibrate_win_prob(df["predicted_return"].values, cal_sr)
    df["p_hit_return"] = calibrate_p_hit(df["predicted_return"].values, cal_sr)
    df["p_hit_mfe"] = calibrate_p_hit(df["predicted_mfe"].values, cal_mfe)
    df["ml_score"] = compute_ml_score(df["predicted_return"].values, cal_sr)

    # --- ATR computation ---
    print("Computing ATR...")
    symbols = df["symbol"].unique().tolist()
    atr_df = compute_atr_data(symbols, csv_dir=US_CSV_DIR, n_jobs=args.jobs)

    if len(atr_df) > 0:
        # Normalize dates for merge
        df["date_key"] = pd.to_datetime(df["date"])
        atr_df["date"] = pd.to_datetime(atr_df["date"])
        df = df.merge(
            atr_df.rename(columns={"date": "date_key"}),
            on=["symbol", "date_key"],
            how="left"
        )
        df = df.drop(columns=["date_key"])
        coverage = df["atr_14d_pct"].notna().mean()
        print(f"  ATR coverage: {coverage:.1%} of rows")
    else:
        df["atr_14d_pct"] = np.nan
        print("  WARNING: ATR computation failed, using NaN")

    # --- Rename for backtester conventions ---
    df = df.rename(columns={
        "daysOut": "holding_days",
        "ta_rvol_20": "stock_volatility_20d",
        "mfe_return": "actual_mfe",
    })

    # --- Select and order final columns ---
    final_cols = [
        "date", "year", "symbol", "sector", "direction", "holding_days",
        "ml_score", "predicted_return", "predicted_mfe",
        "win_probability", "p_hit_return", "p_hit_mfe",
        "actual_return", "actual_mfe",
        "stock_volatility_20d", "atr_14d_pct",
    ]
    df = df[final_cols]

    # --- Summary stats ---
    print(f"\nFinal dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Tier: {args.tier}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Sectors: {df['sector'].nunique()}")
    print(f"  Direction: {df['direction'].value_counts().to_dict()}")
    print(f"  ML score range: {df['ml_score'].min():.1f} - {df['ml_score'].max():.1f}")
    print(f"  Predicted return range: {df['predicted_return'].min():.2f} - {df['predicted_return'].max():.2f}")
    print(f"  Win prob range: {df['win_probability'].min():.3f} - {df['win_probability'].max():.3f}")
    print(f"  ATR range: {df['atr_14d_pct'].min():.4f} - {df['atr_14d_pct'].max():.4f} (mean {df['atr_14d_pct'].mean():.4f})")

    print("\n  Per-year sample counts:")
    for y, cnt in df.groupby("year").size().items():
        ml85 = (df[df["year"] == y]["ml_score"] >= 85).sum()
        print(f"    {y}: {cnt:>10,} total, {ml85:>8,} at ML_85+")

    # --- Save ---
    out_path = RESULTS / cfg["output_file"]
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
