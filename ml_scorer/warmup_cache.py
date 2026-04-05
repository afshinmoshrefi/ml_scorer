#!/usr/bin/env python3
"""Pre-warm the ML Scorer cache by scoring all S&P 500 symbols.

Run after opp_to_parquet.py and after service restart so that by market open,
every symbol is cached and responses are instant (<100ms).

Usage:
  python3 warmup_cache.py              # warm tomorrow's date
  python3 warmup_cache.py 2026-03-25   # warm specific date

Schedule after opp_to_parquet.py:
  0 1 * * * root cd /home/flask/ml_scorer && python3.12 opp_to_parquet.py >> opp_to_parquet.log 2>&1
  5 1 * * * root systemctl restart ml_scorer && sleep 5 && cd /home/flask/ml_scorer && python3.12 warmup_cache.py >> warmup_cache.log 2>&1
"""
import json
import os
import sys
import time
import logging
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('warmup_cache')

# Scorer endpoint
# On production: use curl with Unix socket (bypasses nginx IP whitelist)
SOCKET_PATH = os.environ.get('ML_SCORER_SOCKET',
                              '/home/flask/ml_scorer/ml_scorer.sock')
USE_SOCKET = os.path.exists(SOCKET_PATH)
SCORER_URL = os.environ.get('ML_SCORER_URL', 'http://127.0.0.1:7675')


def get_next_trading_date():
    """Return next weekday as YYYY-MM-DD."""
    from datetime import datetime, timedelta
    today = datetime.now().date()
    d = today + timedelta(days=1)
    while d.weekday() >= 5:  # skip weekends
        d += timedelta(days=1)
    return d.strftime('%Y-%m-%d')


def get_symbols():
    """Load S&P 500 symbols."""
    try:
        import pandas as pd
        data_dir = os.environ.get('ML_SCORER_DATA_DIR', '/home/flask/data')
        path = os.path.join(data_dir, 'sp500_symbols.csv')
        df = pd.read_csv(path)
        return sorted(df['symbols'].tolist())
    except Exception as e:
        log.error(f'Cannot load symbols: {e}')
        return []


def _request(path, payload=None, timeout=120):
    """Make a request to the scorer, using Unix socket if available."""
    import subprocess
    if USE_SOCKET:
        cmd = ['curl', '-s', '--unix-socket', SOCKET_PATH,
               '--max-time', str(timeout)]
        if payload:
            cmd += ['-X', 'POST', '-H', 'Content-Type: application/json',
                    '-d', json.dumps(payload)]
        cmd.append(f'http://localhost{path}')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        if result.returncode != 0:
            raise RuntimeError(f'curl failed: {result.stderr}')
        return json.loads(result.stdout)
    else:
        url = f'{SCORER_URL}{path}'
        if payload:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data,
                                         headers={'Content-Type': 'application/json'})
        else:
            req = url
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())


def wait_for_health(max_wait=60):
    """Wait for scorer to be healthy after restart."""
    for i in range(max_wait):
        try:
            data = _request('/health', timeout=5)
            if data.get('status') == 'ok':
                log.info(f'Scorer healthy (uptime {data.get("uptime_seconds", "?")}s)')
                return True
        except Exception:
            pass
        time.sleep(1)
    log.error(f'Scorer not healthy after {max_wait}s')
    return False


def warmup(date_str, symbols):
    """Score one candidate per symbol to populate caches."""
    log.info(f'Warming {len(symbols)} symbols for {date_str} '
             f'({"socket" if USE_SOCKET else "http"})...')

    t0 = time.time()
    success = 0
    failed = 0
    blocked = 0
    slow = 0

    for i, sym in enumerate(symbols):
        sym_t0 = time.time()
        try:
            result = _request('/score', {
                'symbol': sym, 'date': date_str,
                'daysOut': 20, 'direction': 'l', 'tier': '10_30'
            })
            elapsed = time.time() - sym_t0
            if elapsed > 5:
                slow += 1
                log.warning(f'  SLOW: {sym} took {elapsed:.1f}s')

            # Inspect per-opportunity result, not just HTTP status
            res = (result.get('results') or [{}])[0]
            if 'error' in res:
                if res.get('vix_blocked'):
                    blocked += 1
                else:
                    failed += 1
                    log.error(f'  SCORE_ERR: {sym}: {res["error"]}')
            else:
                success += 1
        except Exception as e:
            failed += 1
            elapsed = time.time() - sym_t0
            log.error(f'  FAIL: {sym}: {e} ({elapsed:.1f}s)')

        if (i + 1) % 50 == 0:
            log.info(f'  {i+1}/{len(symbols)} done ({time.time()-t0:.0f}s elapsed)')

    total = time.time() - t0
    log.info(f'Warmup complete: {success} scored, {blocked} vix_blocked, {failed} failed, '
             f'{slow} slow ({total:.0f}s total, {total/max(len(symbols),1):.1f}s/symbol avg)')


def main():
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = get_next_trading_date()

    log.info(f'=== Cache warmup for {date_str} ===')

    if not wait_for_health():
        sys.exit(1)

    symbols = get_symbols()
    if not symbols:
        sys.exit(1)

    warmup(date_str, symbols)
    log.info('=== Done ===')


if __name__ == '__main__':
    main()
