"""
ML Pattern Scorer -- Opportunity Scoring Service

Scores seasonal stock/ETF pattern opportunities using SR + MFE ensemble models.

Endpoints:
  POST /score       -- Score one or more opportunities
  GET  /health      -- Service health check
  GET  /tiers       -- List available scoring tiers

Usage:
  python opp_scorer.py                                  # default :5090
  ML_SCORER_PORT=8080 python opp_scorer.py              # custom port
  ML_SCORER_DATA_DIR=/home/flask/data python opp_scorer.py  # production paths
"""

import sys
import os

# Allow imports from this package without -m syntax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from config import HOST, PORT

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=True)
