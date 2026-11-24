"""Live VIX for the scorer's volatility gate (2026-11-24).

The gate used to compare the cutoff with the daily VIX close loaded from the price
files, so a scorer whose files were a day or more old kept refusing to score on an
old close (Nov 24, 2026: 35.89 from Nov 20 while the live VIX was about 27).

For an opportunity dated at or after the latest VIX bar in the price files ("now"),
the gate now uses the live VIX quote: the TradeWave realtime price service
(ML_SCORER_LIVE_VIX_URL, the same service the site's market bar uses), then EODHD
real-time (ML_SCORER_EOD_TOKEN) as a fallback, cached for 60 seconds. Historical
dates keep their own daily close. Model features are unchanged: only the gate reads
the live value. Every gate decision reports the value, its source and its time.
"""
import datetime as dt
import json
import logging
import math
import os
import threading
import time
import urllib.parse
import urllib.request

log = logging.getLogger('ml_scorer.live_vix')

LIVE_URL = os.environ.get('ML_SCORER_LIVE_VIX_URL', '').strip()
EOD_TOKEN = os.environ.get('ML_SCORER_EOD_TOKEN', '').strip()
TTL_SECONDS = 60
MAX_AGE_SECONDS = 4 * 24 * 3600
_lock = threading.Lock()
_cache = {'at': 0.0, 'quote': None}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _fetch():
    if LIVE_URL:
        try:
            data = _get_json(LIVE_URL, 3)
            value = _number(data.get('price')) if isinstance(data, dict) and 'error' not in data else None
            if value is not None:
                return value, _number(data.get('timestamp'))
        except Exception:  # noqa: BLE001
            log.info('realtime VIX quote unavailable', exc_info=True)
    if EOD_TOKEN:
        try:
            query = urllib.parse.urlencode({'api_token': EOD_TOKEN, 'fmt': 'json'})
            data = _get_json('https://eodhd.com/api/real-time/VIX.INDX?' + query, 4)
            value = _number(data.get('close')) if isinstance(data, dict) else None
            if value is not None:
                return value, _number(data.get('timestamp'))
        except Exception:  # noqa: BLE001
            log.info('EODHD VIX quote unavailable', exc_info=True)
    return None


def live_vix(now=None, fetch=None):
    """{'value', 'source': 'live', 'time'} or None when no recent quote is available."""
    now = now or time.time()
    if fetch is None:
        with _lock:
            if now - _cache['at'] < TTL_SECONDS:
                quote = _cache['quote']
            else:
                quote = _fetch()
                _cache.update(at=now, quote=quote)
    else:
        quote = fetch()
    if quote is None:
        return None
    value, stamp = quote
    if stamp is not None and abs(now - stamp) > MAX_AGE_SECONDS:
        return None
    when = dt.datetime.fromtimestamp(stamp or now, dt.timezone.utc)
    return {'value': round(value, 2), 'source': 'live', 'time': when.isoformat(timespec='minutes')}


def gate_vix(features, date, latest_bar_date, close_date=None, now=None, fetch=None):
    """The VIX the gate compares with the cutoff, with its provenance.

    `date` is the opportunity date and `latest_bar_date` the last VIX bar in the
    price files (both comparable as 'YYYY-MM-DD' strings or Timestamps).
    """
    close = features.get('mkt_vix_level')
    close = _number(close)
    is_now = latest_bar_date is None or str(date)[:10] >= str(latest_bar_date)[:10]
    if is_now:
        live = live_vix(now=now, fetch=fetch)
        if live is not None:
            return live
    if close is None:
        return None
    result = {'value': round(close, 2), 'source': 'close'}
    if close_date is not None:
        result['date'] = str(close_date)[:10]
    return result


def blocked_fields(vix, cutoff):
    fields = {'vix_value': vix['value'], 'vix_cutoff': cutoff, 'vix_source': vix['source']}
    if vix.get('time'):
        fields['vix_time'] = vix['time']
    if vix.get('date'):
        fields['vix_date'] = vix['date']
    return fields
