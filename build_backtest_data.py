"""
Build the merged backtester input dataset for the Stock Strategy Playbook.

Joins:
  - training_data_10_30.parquet (validation years 2018-2025, VIX <= 35)
  - wf_predictions_sr.parquet (SR model predicted returns)
  - wf_predictions_mfe.parquet (MFE model predicted MFE)
  - calibration_sr.json (predicted_return -> win_prob, ml_score)
  - TICKER_SECTOR from config_ml.py (symbol -> GICS sector)

Output: results/backtester_input_10_30.parquet
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FEATURES = ROOT / "features"
CAL_DIR = ROOT / "ml_scorer" / "calibration"


def load_calibration(path):
    """Load calibration JSON and return bins list."""
    with open(path) as f:
        cal = json.load(f)
    return cal["bins"]


def calibrate_win_prob(pred_values, cal_bins):
    """Vectorized calibration lookup: predicted_return -> win_prob."""
    edges = np.array([b["pred_max"] for b in cal_bins])
    win_probs = np.array([b["win_prob"] for b in cal_bins])
    # np.searchsorted finds the bin index for each prediction
    bin_idx = np.searchsorted(edges, pred_values, side="right")
    bin_idx = np.clip(bin_idx, 0, len(cal_bins) - 1)
    return win_probs[bin_idx]


def calibrate_p_hit(pred_values, cal_bins, field="p_hit_pred"):
    """Vectorized calibration lookup: predicted value -> P(hit predicted)."""
    edges = np.array([b["pred_max"] for b in cal_bins])
    p_hits = np.array([b.get(field, 0.5) for b in cal_bins])
    bin_idx = np.searchsorted(edges, pred_values, side="right")
    bin_idx = np.clip(bin_idx, 0, len(cal_bins) - 1)
    return p_hits[bin_idx]


def compute_ml_score(pred_values, cal_bins):
    """Vectorized ml_score: percentile rank (0-100) via calibration bins."""
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
    """Load TICKER_SECTOR from config_ml.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_ml", ROOT / "config_ml.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TICKER_SECTOR


def main():
    t0 = time.time()

    # --- Load training data (only columns needed for backtester) ---
    print("Loading training data...")
    keep_cols = [
        "date", "symbol", "daysOut", "direction",
        "actual_return", "mfe_return", "mkt_vix_level",
        "ta_rvol_20",  # stock volatility proxy
    ]
    td = pd.read_parquet(FEATURES / "training_data_10_30.parquet", columns=keep_cols)
    print(f"  Loaded {len(td):,} rows in {time.time()-t0:.1f}s")

    # Filter to validation years + VIX <= 35
    td["year"] = td["date"].dt.year
    td = td[(td["year"] >= 2018) & (td["mkt_vix_level"] <= 35)].copy()
    td = td.drop(columns=["mkt_vix_level"])
    print(f"  After VIX filter: {len(td):,} rows")

    # --- Load WF predictions ---
    print("Loading WF predictions...")
    sr = pd.read_parquet(RESULTS / "wf_predictions_sr.parquet")
    mfe = pd.read_parquet(RESULTS / "wf_predictions_mfe.parquet")
    print(f"  SR: {len(sr):,}, MFE: {len(mfe):,}")

    # --- Row-align per year ---
    print("Joining predictions to training data...")
    # Build year-aligned index
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

    # --- Calibration: win_prob, p_hit_return, p_hit_mfe, ml_score ---
    print("Applying calibration...")
    cal_sr = load_calibration(CAL_DIR / "calibration_sr.json")
    cal_mfe = load_calibration(CAL_DIR / "calibration_mfe.json")

    df["win_probability"] = calibrate_win_prob(df["predicted_return"].values, cal_sr)
    df["p_hit_return"] = calibrate_p_hit(df["predicted_return"].values, cal_sr)
    df["p_hit_mfe"] = calibrate_p_hit(df["predicted_mfe"].values, cal_mfe)
    df["ml_score"] = compute_ml_score(df["predicted_return"].values, cal_sr)

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
        "stock_volatility_20d",
    ]
    df = df[final_cols]

    # --- Summary stats ---
    print(f"\nFinal dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Sectors: {df['sector'].nunique()}")
    print(f"  Direction: {df['direction'].value_counts().to_dict()}")
    print(f"  ML score range: {df['ml_score'].min():.1f} - {df['ml_score'].max():.1f}")
    print(f"  Predicted return range: {df['predicted_return'].min():.2f} - {df['predicted_return'].max():.2f}")
    print(f"  Win prob range: {df['win_probability'].min():.3f} - {df['win_probability'].max():.3f}")

    # Per-year counts
    print("\n  Per-year sample counts:")
    for y, cnt in df.groupby("year").size().items():
        ml85 = (df[df["year"] == y]["ml_score"] >= 85).sum()
        print(f"    {y}: {cnt:>10,} total, {ml85:>8,} at ML_85+")

    # --- Save ---
    out_path = RESULTS / "backtester_input_10_30.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
