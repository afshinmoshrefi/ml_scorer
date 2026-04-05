"""Model loading, ensemble prediction, and calibration lookup."""
import os
import json
import logging
import math

import numpy as np

log = logging.getLogger('ml_scorer')


class ModelEnsemble:
    """Loads and manages SR + MFE model ensembles for a single tier."""

    def __init__(self, tier_config, model_dir, calibration_dir, feature_cols_sr,
                 feature_cols_mfe=None):
        self.feature_cols_sr = list(feature_cols_sr)
        self.feature_cols_mfe = list(feature_cols_mfe or feature_cols_sr)
        self.models_sr = self._load_ensemble(tier_config['sr'], model_dir)
        self.models_mfe = self._load_ensemble(tier_config['mfe'], model_dir)
        self.cal_sr = self._load_calibration(
            os.path.join(calibration_dir, tier_config['calibration_sr']))
        self.cal_mfe = self._load_calibration(
            os.path.join(calibration_dir, tier_config['calibration_mfe']))
        log.info(f"  SR models: {len(self.models_sr)} ({len(self.feature_cols_sr)} feats), "
                 f"MFE models: {len(self.models_mfe)} ({len(self.feature_cols_mfe)} feats)")
        log.info(f"  SR cal bins: {len(self.cal_sr)}, MFE cal bins: {len(self.cal_mfe)}")

        # Validate loaded models match expected feature counts
        self._validate_features('SR', self.models_sr, self.feature_cols_sr)
        self._validate_features('MFE', self.models_mfe, self.feature_cols_mfe)

    def _validate_features(self, label, models, feature_cols):
        """Verify all models in an ensemble agree on feature count."""
        expected = len(feature_cols)
        for name, model in models:
            if name == 'lgb':
                n = model.num_feature()
            elif name == 'xgb':
                n = len(model.feature_names)
            elif name == 'catboost':
                n = len(model.feature_names_)
            else:
                continue
            if n != expected:
                raise RuntimeError(
                    f"FATAL: {label} {name} model has {n} features but config expects {expected}. "
                    f"Model and config are out of sync -- retrain or fix config."
                )
        log.info(f"  {label} feature validation: OK ({expected} features)")

    def _load_ensemble(self, model_files, model_dir):
        """Load LGB + XGB + CatBoost models.

        Raises RuntimeError if any model file is missing. All three are required
        so that a partial deployment (e.g., copy interrupted) fails loudly at
        startup rather than silently returning zero predictions.
        """
        models = []

        # LightGBM
        lgb_path = os.path.join(model_dir, model_files['lgb'])
        if not os.path.exists(lgb_path):
            raise RuntimeError(f"FATAL: LightGBM model file missing: {lgb_path}")
        import lightgbm as lgb
        m = lgb.Booster(model_file=lgb_path)
        models.append(('lgb', m))
        log.info(f"    Loaded LGB: {model_files['lgb']}")

        # XGBoost
        xgb_path = os.path.join(model_dir, model_files['xgb'])
        if not os.path.exists(xgb_path):
            raise RuntimeError(f"FATAL: XGBoost model file missing: {xgb_path}")
        import xgboost as xgb
        m = xgb.Booster()
        m.load_model(xgb_path)
        models.append(('xgb', m))
        log.info(f"    Loaded XGB: {model_files['xgb']}")

        # CatBoost
        cb_path = os.path.join(model_dir, model_files['catboost'])
        if not os.path.exists(cb_path):
            raise RuntimeError(f"FATAL: CatBoost model file missing: {cb_path}")
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(cb_path)
        models.append(('catboost', m))
        log.info(f"    Loaded CatBoost: {model_files['catboost']}")

        return models

    def _load_calibration(self, path):
        """Load calibration JSON, return sorted list of bin dicts.

        Raises RuntimeError if the file is missing -- calibration is required
        for win_prob and ml_score to be meaningful.
        """
        if not os.path.exists(path):
            raise RuntimeError(f"FATAL: Calibration file missing: {path}")
        with open(path) as f:
            data = json.load(f)
        return sorted(data['bins'], key=lambda b: b['bin'])

    def predict(self, feature_dict):
        """
        Score a single opportunity.

        Args:
            feature_dict: dict of feature_name -> value (from FeatureEngine)

        Returns:
            dict with pred_return, pred_mfe, win_prob, p_hit_return, p_hit_mfe, ml_score
        """
        # Build separate feature arrays -- SR and MFE may have different feature sets
        X_sr = np.array([[feature_dict.get(f, np.nan) for f in self.feature_cols_sr]],
                        dtype=np.float32)
        X_mfe = np.array([[feature_dict.get(f, np.nan) for f in self.feature_cols_mfe]],
                         dtype=np.float32)

        pred_sr = self._predict_ensemble(self.models_sr, X_sr, self.feature_cols_sr)
        pred_mfe = self._predict_ensemble(self.models_mfe, X_mfe, self.feature_cols_mfe)

        if not math.isfinite(pred_sr) or not math.isfinite(pred_mfe):
            raise RuntimeError(
                f'Model ensemble returned non-finite prediction: '
                f'pred_sr={pred_sr}, pred_mfe={pred_mfe}'
            )

        # Calibration lookup
        win_prob = self._calibrate(self.cal_sr, pred_sr, 'win_prob')
        p_hit_return = self._calibrate(self.cal_sr, pred_sr, 'p_hit_pred')
        p_hit_mfe = self._calibrate(self.cal_mfe, pred_mfe, 'p_hit_pred')

        # ml_score: percentile rank of pred_sr within calibration bins (0-100)
        ml_score = self._percentile_score(self.cal_sr, pred_sr)

        return {
            'pred_return': round(float(pred_sr), 4),
            'pred_mfe': round(float(pred_mfe), 4),
            'win_prob': round(float(win_prob), 4),
            'p_hit_return': round(float(p_hit_return), 4),
            'p_hit_mfe': round(float(p_hit_mfe), 4),
            'ml_score': round(float(ml_score), 1),
        }

    def _predict_ensemble(self, models, X, feature_cols):
        """Average predictions from ensemble."""
        if not models:
            return 0.0
        preds = []
        for name, model in models:
            if name == 'lgb':
                preds.append(model.predict(X)[0])
            elif name == 'xgb':
                import xgboost as xgb
                dm = xgb.DMatrix(X, feature_names=feature_cols)
                preds.append(model.predict(dm)[0])
            elif name == 'catboost':
                preds.append(model.predict(X)[0])
        return float(np.mean(preds))

    def _calibrate(self, cal_bins, pred_value, field):
        """Look up calibrated probability from bins."""
        if not cal_bins:
            return 0.5
        # Find the bin this prediction falls into
        for b in cal_bins:
            if pred_value <= b['pred_max']:
                return b.get(field, 0.5)
        # Above all bins: use last bin
        return cal_bins[-1].get(field, 0.5)

    def _percentile_score(self, cal_bins, pred_value):
        """Convert prediction to 0-100 percentile score."""
        if not cal_bins:
            return 50.0
        n_bins = len(cal_bins)
        for i, b in enumerate(cal_bins):
            if pred_value <= b['pred_max']:
                # Linear interpolation within bin
                bin_min = b['pred_min']
                bin_max = b['pred_max']
                bin_range = bin_max - bin_min
                if bin_range > 0:
                    frac = (pred_value - bin_min) / bin_range
                else:
                    frac = 0.5
                return (i + frac) / n_bins * 100.0
        return 100.0


class ScorerManager:
    """Manages model ensembles for all tiers."""

    def __init__(self):
        self.tiers = {}  # tier_name -> ModelEnsemble

    def load_tier(self, tier_name, tier_config, model_dir, calibration_dir,
                  feature_cols_sr, feature_cols_mfe=None):
        log.info(f"Loading tier: {tier_name}")
        self.tiers[tier_name] = ModelEnsemble(
            tier_config, model_dir, calibration_dir, feature_cols_sr, feature_cols_mfe)

    def get_tier(self, tier_name):
        return self.tiers.get(tier_name)

    def available_tiers(self):
        return list(self.tiers.keys())
