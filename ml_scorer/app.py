"""
ML Pattern Scorer -- Flask Service

Scores seasonal stock pattern opportunities using SR + MFE ensemble models.

Endpoints:
  POST /score       -- Score one or more opportunities
  GET  /health      -- Service health check
  GET  /tiers       -- List available scoring tiers

Usage:
  python -m ml_scorer.app                    # debug mode
  ML_SCORER_PORT=5090 python -m ml_scorer.app  # custom port
"""
import logging
import time
import os

from flask import Flask, request, jsonify

try:
    from .config import HOST, PORT, TIERS, MODEL_DIR, CALIBRATION_DIR, FEATURE_COLS, FEATURE_COLS_MFE, VIX_CUTOFF
    from .scorer import ScorerManager
    from .feature_engine import FeatureEngine
except ImportError:
    from config import HOST, PORT, TIERS, MODEL_DIR, CALIBRATION_DIR, FEATURE_COLS, FEATURE_COLS_MFE, VIX_CUTOFF
    from scorer import ScorerManager
    from feature_engine import FeatureEngine

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s'
)
log = logging.getLogger('ml_scorer')

app = Flask(__name__)

# Global state -- initialized on startup
scorer_mgr = ScorerManager()
engine = FeatureEngine()
_startup_time = None


def init_service():
    """Load all models and warm up the feature engine."""
    global _startup_time
    t0 = time.time()
    log.info("=" * 60)
    log.info("ML Pattern Scorer Service - Starting")
    log.info("=" * 60)

    # Load all configured tiers
    for tier_name, tier_config in TIERS.items():
        scorer_mgr.load_tier(tier_name, tier_config, MODEL_DIR, CALIBRATION_DIR,
                             FEATURE_COLS, FEATURE_COLS_MFE)

    _startup_time = time.time()
    log.info(f"Service ready in {_startup_time - t0:.1f}s. Tiers: {scorer_mgr.available_tiers()}")


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'tiers': scorer_mgr.available_tiers(),
        'uptime_seconds': round(time.time() - _startup_time, 0) if _startup_time else 0,
        'feature_count': len(FEATURE_COLS),
        'vix_cutoff': VIX_CUTOFF,
    })


@app.route('/tiers', methods=['GET'])
def tiers():
    return jsonify({'tiers': scorer_mgr.available_tiers()})


@app.route('/score', methods=['POST'])
def score():
    """
    Score one or more opportunities.

    Request JSON:
      {
        "opportunities": [
          {"symbol": "AAPL", "date": "2026-03-15", "daysOut": 20, "direction": "l"},
          ...
        ],
        "tier": "10_30"          // optional, default "10_30"
      }

    Or single opportunity:
      {"symbol": "AAPL", "date": "2026-03-15", "daysOut": 20, "direction": "l"}

    Response JSON:
      {
        "results": [
          {
            "symbol": "AAPL", "date": "2026-03-15", "daysOut": 20, "direction": "l",
            "pred_return": 2.34, "pred_mfe": 5.12, "win_prob": 0.78,
            "p_hit_return": 0.58, "p_hit_mfe": 0.47, "ml_score": 82.3
          }
        ],
        "tier": "10_30",
        "elapsed_ms": 145
      }
    """
    t0 = time.time()
    data = request.get_json(force=True)

    # Parse tier
    tier_name = data.get('tier', '10_30')
    tier = scorer_mgr.get_tier(tier_name)
    if tier is None:
        return jsonify({
            'error': f'Unknown tier: {tier_name}',
            'available_tiers': scorer_mgr.available_tiers()
        }), 400

    # Parse opportunities
    if 'opportunities' in data:
        opps = data['opportunities']
    elif 'symbol' in data:
        opps = [data]
    else:
        return jsonify({'error': 'Missing "opportunities" or "symbol" in request'}), 400

    if not isinstance(opps, list) or len(opps) == 0:
        return jsonify({'error': 'Empty opportunities list'}), 400

    # Pre-load price data for all symbols in batch
    symbols = list(set(o.get('symbol', '') for o in opps))
    engine.load_price_data(symbols)

    results = []
    for opp in opps:
        symbol = opp.get('symbol')
        date = opp.get('date')
        days_out = opp.get('daysOut')
        direction = opp.get('direction')

        if not all([symbol, date, days_out, direction]):
            results.append({
                'symbol': symbol, 'date': date, 'error': 'Missing required fields'
            })
            continue

        days_out = int(days_out)

        try:
            # Compute features
            features = engine.compute_features(symbol, date, days_out, direction)

            # Check VIX cutoff
            vix = features.get('mkt_vix_level')
            if vix is not None and vix > VIX_CUTOFF:
                results.append({
                    'symbol': symbol, 'date': date, 'daysOut': days_out,
                    'direction': direction,
                    'error': f'VIX={vix:.1f} exceeds cutoff ({VIX_CUTOFF})',
                    'vix_blocked': True,
                })
                continue

            # Score
            scores = tier.predict(features)
            scores.update({
                'symbol': symbol, 'date': date, 'daysOut': days_out,
                'direction': direction,
            })
            results.append(scores)

        except Exception as e:
            log.exception(f"Error scoring {symbol} {date}")
            results.append({
                'symbol': symbol, 'date': date, 'error': str(e)
            })

    elapsed = (time.time() - t0) * 1000
    return jsonify({
        'results': results,
        'tier': tier_name,
        'elapsed_ms': round(elapsed, 1),
    })


# ---------------------------------------------------------------
# Startup
# ---------------------------------------------------------------

init_service()

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=True)
