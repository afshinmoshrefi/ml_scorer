"""Allow running as: python -m ml_scorer"""
from .app import app
from .config import HOST, PORT

app.run(host=HOST, port=PORT, debug=True)
