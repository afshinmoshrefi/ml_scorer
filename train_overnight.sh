#!/bin/bash
# Overnight training for 31_60 and 61_90 tiers (both SR + MFE)
# Expected runtime: ~7-8 hours on 24-core machine
# Started: $(date)

set -e
cd C:/seasonals/ml_scorer

echo "=========================================="
echo "OVERNIGHT TRAINING - Started $(date)"
echo "=========================================="

# --- 31_60 TIER ---
# SR already trained with 59 features (pat_daysOut included). Need WF predictions for calibration + MFE.

echo ""
echo ">>> 31_60 SR: Walk-forward + save predictions (for calibration)"
echo ">>> Started: $(date)"
python train_model.py --tier 31_60 --target sr --skip-optuna --wf-only --save-predictions
echo ">>> 31_60 SR WF done: $(date)"

echo ""
echo ">>> 31_60 MFE: Optuna tuning"
echo ">>> Started: $(date)"
python train_model.py --tier 31_60 --target mfe --optuna-trials 75 --wf-only --save-predictions
echo ">>> 31_60 MFE done (Optuna + WF): $(date)"

echo ""
echo ">>> 31_60 MFE: Final model"
echo ">>> Started: $(date)"
python train_model.py --tier 31_60 --target mfe --skip-optuna --final-only
echo ">>> 31_60 MFE final done: $(date)"

# --- 61_90 TIER ---
# Nothing exists yet. Full pipeline for both targets.

echo ""
echo ">>> 61_90 SR: Optuna tuning + Walk-forward + save predictions"
echo ">>> Started: $(date)"
python train_model.py --tier 61_90 --target sr --optuna-trials 75 --save-predictions
echo ">>> 61_90 SR done (Optuna + WF + Final): $(date)"

echo ""
echo ">>> 61_90 MFE: Optuna tuning + Walk-forward + save predictions"
echo ">>> Started: $(date)"
python train_model.py --tier 61_90 --target mfe --optuna-trials 75 --save-predictions
echo ">>> 61_90 MFE done (Optuna + WF + Final): $(date)"

echo ""
echo "=========================================="
echo "ALL TRAINING COMPLETE - $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check results in results/v2_walk_forward_results_*.json"
echo "  2. Copy models to ml_scorer/ml_scorer/models/"
echo "  3. Copy calibration to ml_scorer/ml_scorer/calibration/"
echo "  4. Update ml_scorer/ml_scorer/config.py TIERS dict"
echo "  5. Restart and test service"
