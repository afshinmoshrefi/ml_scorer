"""
ML Pattern Scorer -- Flask Service

Scores seasonal stock pattern opportunities using SR + MFE ensemble models.

Endpoints:
  POST /score       -- Score one or more opportunities
  POST /select      -- Find and score today's best opportunities from parquet cache
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
    from .config import HOST, PORT, TIERS, MODEL_DIR, CALIBRATION_DIR, FEATURE_COLS, FEATURE_COLS_MFE, VIX_CUTOFF, tier_for_days_out
    from .scorer import ScorerManager
    from .feature_engine import FeatureEngine
    from .daily_opp_selection import select_daily_opps
except ImportError:
    from config import HOST, PORT, TIERS, MODEL_DIR, CALIBRATION_DIR, FEATURE_COLS, FEATURE_COLS_MFE, VIX_CUTOFF, tier_for_days_out
    from scorer import ScorerManager
    from feature_engine import FeatureEngine
    from daily_opp_selection import select_daily_opps

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
          {"symbol": "XLE",  "date": "2026-03-15", "daysOut": 45, "direction": "l"},
          ...
        ],
        "tier": "10_30"    // optional: request-level default; auto-detected from daysOut if omitted
      }

    Tier resolution per opportunity (highest precedence first):
      1. "tier" field on the individual opportunity object
      2. Top-level "tier" field on the request
      3. Auto-detected from daysOut: <=30 -> 10_30, <=60 -> 31_60, else 61_90

    Or single opportunity shorthand:
      {"symbol": "AAPL", "date": "2026-03-15", "daysOut": 20, "direction": "l"}

    Response JSON:
      {
        "results": [
          {
            "symbol": "AAPL", "date": "2026-03-15", "daysOut": 20, "direction": "l",
            "tier": "10_30",
            "pred_return": 2.34, "pred_mfe": 5.12, "win_prob": 0.78,
            "p_hit_return": 0.58, "p_hit_mfe": 0.47, "ml_score": 82.3
          }
        ],
        "tiers_used": ["10_30"],
        "elapsed_ms": 145
      }
    """
    t0 = time.time()
    data = request.get_json(force=True)

    # Parse opportunities
    if 'opportunities' in data:
        opps = data['opportunities']
    elif 'symbol' in data:
        opps = [data]
    else:
        return jsonify({'error': 'Missing "opportunities" or "symbol" in request'}), 400

    if not isinstance(opps, list) or len(opps) == 0:
        return jsonify({'error': 'Empty opportunities list'}), 400

    # Top-level tier is an optional default. Per-opportunity tier is auto-detected
    # from daysOut when not specified, so mixed-horizon batches work correctly.
    default_tier = data.get('tier')

    # Pre-load price data for all symbols in batch.
    # Only extract symbols from dict items to avoid AttributeError on malformed entries.
    symbols = list(set(
        o.get('symbol', '') for o in opps if isinstance(o, dict)
    ))
    engine.load_price_data(symbols)

    results = []
    tiers_used = set()
    for opp in opps:
        # Guard against non-dict entries in the batch
        if not isinstance(opp, dict):
            results.append({'error': f'Invalid opportunity format: expected object, got {type(opp).__name__}'})
            continue

        symbol = opp.get('symbol')
        date = opp.get('date')
        days_out = opp.get('daysOut')
        direction = opp.get('direction')

        if not all([symbol, date, days_out, direction]):
            results.append({
                'symbol': symbol, 'date': date, 'error': 'Missing required fields'
            })
            continue

        try:
            days_out = int(days_out)
        except (TypeError, ValueError):
            results.append({
                'symbol': symbol, 'date': date,
                'error': f'daysOut must be an integer, got {days_out!r}',
            })
            continue

        if days_out <= 0:
            results.append({
                'symbol': symbol, 'date': date,
                'error': f'daysOut must be positive, got {days_out}',
            })
            continue

        # Resolve tier: per-opportunity override -> request-level default -> auto-detect
        tier_name = opp.get('tier') or default_tier or tier_for_days_out(days_out)
        tier = scorer_mgr.get_tier(tier_name)
        if tier is None:
            results.append({
                'symbol': symbol, 'date': date,
                'error': f'Unknown tier: {tier_name}',
                'available_tiers': scorer_mgr.available_tiers(),
            })
            continue

        auto_tier = tier_for_days_out(days_out)
        if tier_name != auto_tier:
            log.warning(
                f'Tier mismatch for {symbol}: requested {tier_name} but daysOut={days_out} '
                f'suggests {auto_tier}'
            )

        tiers_used.add(tier_name)

        try:
            # Compute features
            features = engine.compute_features(symbol, date, days_out, direction)

            # Check VIX cutoff
            vix = features.get('mkt_vix_level')
            if vix is None or vix != vix:  # None or NaN (NaN != NaN is True)
                log.warning(f'VIX data unavailable for {symbol} {date} -- VIX block bypassed')
            elif vix > VIX_CUTOFF:
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
                'tier': tier_name,
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
        'tiers_used': sorted(tiers_used),
        'elapsed_ms': round(elapsed, 1),
    })


@app.route('/select', methods=['POST'])
def select():
    """
    Find and score today's best opportunities from parquet cache.

    Request JSON:
      {
        "date": "2026-03-25",
        "resource_ids": ["sp500"],
        "num_picks": 10,
        "direction": "l",
        "days_out_min": 10,
        "days_out_max": 30,
        "min_avg_return": 3.0,
        "min_win_prob": 0.70,
        "exclude_symbols": ["AAPL"]
      }

    Response JSON:
      {
        "picks": [ { symbol, date, daysOut, direction, ml_score, win_prob, ... } ],
        "candidates_after_prefilter": 120,
        "candidates_scored": 120,
        "elapsed_ms": 4500
      }
    """
    data = request.get_json(force=True)

    date = data.get('date')
    if not date:
        return jsonify({'error': 'Missing "date" field'}), 400

    # Parse and validate numeric parameters at the API boundary.
    try:
        num_picks = int(data.get('num_picks', 10))
        days_out_min = int(data.get('days_out_min', 10))
        days_out_max = int(data.get('days_out_max', 30))
        min_avg_return = float(data.get('min_avg_return', 3.0))
        min_win_prob = float(data.get('min_win_prob', 0.70))
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'Invalid parameter type: {e}'}), 400

    if days_out_min <= 0 or days_out_max <= 0 or days_out_min > days_out_max:
        return jsonify({'error': f'Invalid days_out range: {days_out_min}-{days_out_max}'}), 400
    if not (0.0 <= min_win_prob <= 1.0):
        return jsonify({'error': f'min_win_prob must be between 0 and 1, got {min_win_prob}'}), 400

    direction = data.get('direction', 'l')
    if direction not in ('l', 's', 'both'):
        return jsonify({'error': f'direction must be "l", "s", or "both", got {direction!r}'}), 400

    resource_ids = data.get('resource_ids', ['sp500'])
    if not isinstance(resource_ids, list):
        return jsonify({'error': 'resource_ids must be a list'}), 400

    exclude_symbols = data.get('exclude_symbols', [])
    if not isinstance(exclude_symbols, list):
        return jsonify({'error': 'exclude_symbols must be a list'}), 400

    result = select_daily_opps(
        date=date,
        resource_ids=resource_ids,
        num_picks=num_picks,
        direction=direction,
        days_out_min=days_out_min,
        days_out_max=days_out_max,
        min_avg_return=min_avg_return,
        min_win_prob=min_win_prob,
        exclude_symbols=exclude_symbols,
        engine=engine,
        scorer_mgr=scorer_mgr,
    )

    return jsonify(result)


# ---------------------------------------------------------------
# Startup
# ---------------------------------------------------------------

init_service()

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=True)
