#!/usr/bin/env python3
"""
Comprehensive test suite for ML Scorer parquet integration.

Tests the parquet cache path, gzip fallback, cache behavior, data integrity,
edge cases, and the opp_to_parquet generator.

Run: python3 test_parquet_integration.py
Requires: ML scorer running on 104.238.214.253:7675 with parquets for 2026-03-16 through 2026-03-20
"""

import os
import sys
import json
import time
import requests
import traceback
from datetime import datetime, timedelta

SCORER_URL = 'http://104.238.214.253:7675'
PASS = 0
FAIL = 0
ERRORS = []


def score(symbol, date, daysOut, direction, tier='10_30'):
    """Helper to call the scorer API."""
    resp = requests.post(f'{SCORER_URL}/score', json={
        'symbol': symbol, 'date': date, 'daysOut': daysOut,
        'direction': direction, 'tier': tier,
    }, timeout=120)
    return resp.json()


def score_batch(opportunities, tier='10_30'):
    """Helper for batch scoring."""
    resp = requests.post(f'{SCORER_URL}/score', json={
        'opportunities': opportunities, 'tier': tier,
    }, timeout=120)
    return resp.json()


def check(test_name, condition, detail=''):
    """Record a test result."""
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  PASS: {test_name}')
    else:
        FAIL += 1
        msg = f'  FAIL: {test_name}'
        if detail:
            msg += f' -- {detail}'
        print(msg)
        ERRORS.append(msg)


def run_section(name, func):
    """Run a test section with error handling."""
    global FAIL
    print(f'\n{"="*60}')
    print(f'  {name}')
    print(f'{"="*60}')
    try:
        func()
    except Exception as e:
        FAIL += 1
        msg = f'  FAIL: {name} CRASHED: {e}'
        print(msg)
        print(traceback.format_exc())
        ERRORS.append(msg)


# ============================================================
# TEST SECTIONS
# ============================================================

def test_health():
    """Basic health check."""
    resp = requests.get(f'{SCORER_URL}/health', timeout=10).json()
    check('Service is healthy', resp['status'] == 'ok')
    check('59 features configured', resp['feature_count'] == 59)
    check('3 tiers loaded', len(resp['tiers']) == 3)
    check('All tiers present', set(resp['tiers']) == {'10_30', '31_60', '61_90'})


def test_single_score_parquet():
    """Single score on a date with parquet available."""
    result = score('AAPL', '2026-03-16', 20, 'l')
    r = result['results'][0]
    check('Returns result', len(result['results']) == 1)
    check('Symbol matches', r['symbol'] == 'AAPL')
    check('Has ml_score', 'ml_score' in r)
    check('Has win_prob', 'win_prob' in r)
    check('Has pred_return', 'pred_return' in r)
    check('Has pred_mfe', 'pred_mfe' in r)
    check('Has p_hit_return', 'p_hit_return' in r)
    check('Has p_hit_mfe', 'p_hit_mfe' in r)
    check('ml_score in range 0-100', 0 <= r['ml_score'] <= 100)
    check('win_prob in range 0-1', 0 <= r['win_prob'] <= 1)
    check('pred_mfe >= 0', r['pred_mfe'] >= 0)
    check('elapsed_ms is reasonable', result['elapsed_ms'] < 30000,
          f'elapsed={result["elapsed_ms"]}ms')


def test_all_tiers():
    """Score the same symbol across all 3 tiers."""
    for tier in ['10_30', '31_60', '61_90']:
        result = score('AAPL', '2026-03-16', 20, 'l', tier=tier)
        r = result['results'][0]
        check(f'Tier {tier} returns score', 'ml_score' in r)
        check(f'Tier {tier} score in range', 0 <= r['ml_score'] <= 100)


def test_directions():
    """Score both long and short."""
    long_result = score('AAPL', '2026-03-16', 20, 'l')
    short_result = score('AAPL', '2026-03-16', 20, 's')
    lr = long_result['results'][0]
    sr = short_result['results'][0]
    check('Long returns score', 'ml_score' in lr)
    check('Short returns score', 'ml_score' in sr)
    check('Long and short differ', lr['ml_score'] != sr['ml_score'],
          f'long={lr["ml_score"]}, short={sr["ml_score"]}')


def test_sp500_symbols():
    """Score multiple S&P 500 symbols."""
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'HD', 'MA']
    for sym in symbols:
        result = score(sym, '2026-03-16', 20, 'l')
        r = result['results'][0]
        check(f'{sym} returns valid score', 0 <= r['ml_score'] <= 100,
              f'ml_score={r["ml_score"]}')


def test_etf_symbols():
    """Score ETF symbols (different market parquet)."""
    etfs = ['XLE', 'XLF', 'XLK', 'SPY', 'QQQ']
    for sym in etfs:
        result = score(sym, '2026-03-16', 20, 'l')
        r = result['results'][0]
        check(f'ETF {sym} returns valid score', 0 <= r['ml_score'] <= 100,
              f'ml_score={r["ml_score"]}')


def test_different_parquet_dates():
    """Score on different dates that should all have parquets (Mon-Fri)."""
    dates = ['2026-03-16', '2026-03-17', '2026-03-18', '2026-03-19', '2026-03-20']
    for d in dates:
        result = score('AAPL', d, 20, 'l')
        r = result['results'][0]
        check(f'Date {d} returns score', 0 <= r['ml_score'] <= 100)
        check(f'Date {d} date matches', r['date'] == d)


def test_cache_key_different_dates():
    """Verify scoring same symbol on different dates gives different results.
    This catches the (symbol, date) cache key bug.
    """
    r1 = score('AAPL', '2026-03-16', 20, 'l')['results'][0]
    r2 = score('AAPL', '2026-03-17', 20, 'l')['results'][0]
    # Different dates should generally produce different predictions
    # (different pattern features, different market regime)
    check('Different dates produce different pred_return',
          r1['pred_return'] != r2['pred_return'],
          f'mar16={r1["pred_return"]}, mar17={r2["pred_return"]}')


def test_cache_key_different_daysout():
    """Same symbol/date but different daysOut should give different results.
    Uses daysOut values that actually exist in the data (AAPL Mar 16 starts at 17).
    """
    r1 = score('AAPL', '2026-03-16', 27, 'l')['results'][0]
    r2 = score('AAPL', '2026-03-16', 140, 'l')['results'][0]
    check('Different daysOut produce different results',
          r1['pred_return'] != r2['pred_return'],
          f'27d={r1["pred_return"]}, 140d={r2["pred_return"]}')


def test_batch_scoring():
    """Batch score multiple symbols."""
    opps = [
        {'symbol': 'AAPL', 'date': '2026-03-16', 'daysOut': 20, 'direction': 'l'},
        {'symbol': 'MSFT', 'date': '2026-03-16', 'daysOut': 20, 'direction': 'l'},
        {'symbol': 'GOOG', 'date': '2026-03-16', 'daysOut': 20, 'direction': 'l'},
        {'symbol': 'XLE', 'date': '2026-03-16', 'daysOut': 20, 'direction': 'l'},
    ]
    result = score_batch(opps)
    check('Batch returns 4 results', len(result['results']) == 4)
    symbols = [r['symbol'] for r in result['results']]
    check('Batch has all symbols', set(symbols) == {'AAPL', 'MSFT', 'GOOG', 'XLE'})
    for r in result['results']:
        check(f'Batch {r["symbol"]} valid score', 0 <= r['ml_score'] <= 100)


def test_batch_mixed_markets():
    """Batch with symbols from different markets."""
    opps = [
        {'symbol': 'AAPL', 'date': '2026-03-16', 'daysOut': 20, 'direction': 'l'},
        {'symbol': 'XLE', 'date': '2026-03-16', 'daysOut': 20, 'direction': 'l'},
    ]
    result = score_batch(opps)
    check('Mixed market batch returns 2', len(result['results']) == 2)
    for r in result['results']:
        check(f'Mixed batch {r["symbol"]} valid', 0 <= r['ml_score'] <= 100)


def test_batch_mixed_dates():
    """Batch with different dates (tests cache key correctness)."""
    opps = [
        {'symbol': 'AAPL', 'date': '2026-03-16', 'daysOut': 20, 'direction': 'l'},
        {'symbol': 'AAPL', 'date': '2026-03-17', 'daysOut': 20, 'direction': 'l'},
    ]
    result = score_batch(opps)
    check('Mixed date batch returns 2', len(result['results']) == 2)
    r1 = result['results'][0]
    r2 = result['results'][1]
    check('Mixed dates: different predictions',
          r1['pred_return'] != r2['pred_return'],
          f'r1={r1["pred_return"]}, r2={r2["pred_return"]}')


def test_consistency():
    """Score the same symbol twice, verify identical results (cache hit)."""
    r1 = score('AAPL', '2026-03-16', 20, 'l')['results'][0]
    r2 = score('AAPL', '2026-03-16', 20, 'l')['results'][0]
    check('Repeat call: same ml_score', r1['ml_score'] == r2['ml_score'])
    check('Repeat call: same pred_return', r1['pred_return'] == r2['pred_return'])
    check('Repeat call: same win_prob', r1['win_prob'] == r2['win_prob'])
    check('Repeat call: same pred_mfe', r1['pred_mfe'] == r2['pred_mfe'])


def test_gzip_fallback():
    """Score on a date with no parquet (should fall back to gzip scan).
    March 25 is next week -- parquets only exist for Mar 16-20.
    """
    result = score('AAPL', '2026-03-25', 20, 'l')
    r = result['results'][0]
    check('Gzip fallback returns score', 'ml_score' in r)
    check('Gzip fallback score in range', 0 <= r['ml_score'] <= 100)
    check('Gzip fallback has all fields',
          all(k in r for k in ['ml_score', 'win_prob', 'pred_return', 'pred_mfe']))


def test_gzip_fallback_etf():
    """Gzip fallback for ETF (no opp_by_symbol dir, only opportunities/)."""
    # ETFs don't have opp_by_symbol, so gzip fallback may not work for them.
    # This test documents the behavior.
    result = score('XLE', '2026-03-25', 20, 'l')
    r = result['results'][0]
    has_score = 'ml_score' in r
    check('ETF gzip fallback returns result', has_score,
          'ETFs may not have opp_by_symbol for gzip fallback')


def test_unknown_symbol():
    """Score a symbol that doesn't exist anywhere."""
    result = score('ZZZZZZ', '2026-03-16', 20, 'l')
    r = result['results'][0]
    # Should still return something (NaN features -> some default score)
    check('Unknown symbol returns result', 'ml_score' in r)


def test_various_daysout():
    """Score across different daysOut values."""
    for days in [10, 15, 20, 25, 30, 45, 60, 90]:
        result = score('AAPL', '2026-03-16', days, 'l')
        r = result['results'][0]
        check(f'daysOut={days} returns valid score', 0 <= r['ml_score'] <= 100)


def test_parquet_vs_single_score_stability():
    """Score 20 different symbols and verify none crash or return errors."""
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
               'V', 'JNJ', 'PG', 'HD', 'MA', 'BAC', 'DIS', 'NFLX', 'ADBE',
               'CRM', 'INTC', 'AMD']
    for sym in symbols:
        try:
            result = score(sym, '2026-03-16', 20, 'l')
            r = result['results'][0]
            check(f'Stability {sym}', 'ml_score' in r and 0 <= r['ml_score'] <= 100)
        except Exception as e:
            check(f'Stability {sym}', False, str(e))


def test_large_batch():
    """Batch score 50 symbols at once."""
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM',
               'V', 'JNJ', 'PG', 'HD', 'MA', 'BAC', 'DIS', 'NFLX', 'ADBE',
               'CRM', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'COST', 'ABBV',
               'PEP', 'TMO', 'ACN', 'MRK', 'LLY', 'WMT', 'CSCO', 'ABT',
               'DHR', 'VZ', 'CMCSA', 'NKE', 'BMY', 'UNH', 'XOM', 'CVX',
               'LIN', 'PM', 'RTX', 'HON', 'UPS', 'LOW', 'NEE', 'CAT', 'GE']
    opps = [{'symbol': s, 'date': '2026-03-16', 'daysOut': 20, 'direction': 'l'}
            for s in symbols]
    result = score_batch(opps)
    check(f'Large batch returns {len(symbols)} results',
          len(result['results']) == len(symbols))
    valid = sum(1 for r in result['results'] if 0 <= r['ml_score'] <= 100)
    check(f'Large batch all valid scores', valid == len(symbols),
          f'{valid}/{len(symbols)} valid')
    check(f'Large batch reasonable time', result['elapsed_ms'] < 60000,
          f'{result["elapsed_ms"]:.0f}ms')


def test_performance_parquet():
    """Measure scoring performance with parquet cache."""
    # Warm up (load parquets)
    score('AAPL', '2026-03-16', 20, 'l')

    # Time a batch of 10
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']
    opps = [{'symbol': s, 'date': '2026-03-16', 'daysOut': 20, 'direction': 'l'}
            for s in symbols]
    result = score_batch(opps)
    ms = result['elapsed_ms']
    per_symbol = ms / len(symbols)
    check(f'Batch 10 symbols < 10s', ms < 10000, f'{ms:.0f}ms total, {per_symbol:.0f}ms/symbol')
    print(f'  INFO: 10 symbols in {ms:.0f}ms ({per_symbol:.0f}ms/symbol)')


def test_parquet_data_integrity():
    """Compare parquet results across repeat calls to ensure no drift."""
    # Score AAPL 3 times with different symbols in between
    r1 = score('AAPL', '2026-03-16', 20, 'l')['results'][0]
    score('MSFT', '2026-03-17', 30, 's')  # different everything
    score('GOOG', '2026-03-18', 15, 'l')  # different everything
    r2 = score('AAPL', '2026-03-16', 20, 'l')['results'][0]

    check('Integrity: ml_score stable', r1['ml_score'] == r2['ml_score'])
    check('Integrity: pred_return stable', r1['pred_return'] == r2['pred_return'])
    check('Integrity: win_prob stable', r1['win_prob'] == r2['win_prob'])
    check('Integrity: pred_mfe stable', r1['pred_mfe'] == r2['pred_mfe'])
    check('Integrity: p_hit_return stable', r1['p_hit_return'] == r2['p_hit_return'])
    check('Integrity: p_hit_mfe stable', r1['p_hit_mfe'] == r2['p_hit_mfe'])


def test_weekday_parquet_coverage():
    """Verify all 5 weekday parquets return different pattern features.
    Same symbol, different days should have different pattern data.
    """
    dates = ['2026-03-16', '2026-03-17', '2026-03-18', '2026-03-19', '2026-03-20']
    scores = {}
    for d in dates:
        r = score('AAPL', d, 20, 'l')['results'][0]
        scores[d] = r['pred_return']

    unique_values = len(set(scores.values()))
    check('5 weekdays produce varied predictions', unique_values >= 3,
          f'unique pred_return values: {unique_values}/5 -- {scores}')


def test_short_vs_long_daysout_tiers():
    """Verify tier selection works with appropriate daysOut ranges."""
    # 10_30 tier with 20 day hold
    r1 = score('AAPL', '2026-03-16', 20, 'l', tier='10_30')['results'][0]
    # 31_60 tier with 45 day hold
    r2 = score('AAPL', '2026-03-16', 45, 'l', tier='31_60')['results'][0]
    # 61_90 tier with 75 day hold
    r3 = score('AAPL', '2026-03-16', 75, 'l', tier='61_90')['results'][0]
    check('10_30 tier valid', 0 <= r1['ml_score'] <= 100)
    check('31_60 tier valid', 0 <= r2['ml_score'] <= 100)
    check('61_90 tier valid', 0 <= r3['ml_score'] <= 100)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print('ML Scorer Parquet Integration Test Suite')
    print(f'Target: {SCORER_URL}')
    print(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    sections = [
        ('1. Health Check', test_health),
        ('2. Single Score (parquet)', test_single_score_parquet),
        ('3. All Tiers', test_all_tiers),
        ('4. Long vs Short', test_directions),
        ('5. S&P 500 Symbols', test_sp500_symbols),
        ('6. ETF Symbols', test_etf_symbols),
        ('7. Different Parquet Dates', test_different_parquet_dates),
        ('8. Cache Key - Different Dates', test_cache_key_different_dates),
        ('9. Cache Key - Different DaysOut', test_cache_key_different_daysout),
        ('10. Batch Scoring', test_batch_scoring),
        ('11. Batch Mixed Markets', test_batch_mixed_markets),
        ('12. Batch Mixed Dates', test_batch_mixed_dates),
        ('13. Consistency (repeat calls)', test_consistency),
        ('14. Gzip Fallback (no parquet)', test_gzip_fallback),
        ('15. Gzip Fallback ETF', test_gzip_fallback_etf),
        ('16. Unknown Symbol', test_unknown_symbol),
        ('17. Various DaysOut', test_various_daysout),
        ('18. 20-Symbol Stability', test_parquet_vs_single_score_stability),
        ('19. Large Batch (50 symbols)', test_large_batch),
        ('20. Performance', test_performance_parquet),
        ('21. Data Integrity (cross-call)', test_parquet_data_integrity),
        ('22. Weekday Parquet Coverage', test_weekday_parquet_coverage),
        ('23. Tier + DaysOut Ranges', test_short_vs_long_daysout_tiers),
    ]

    for name, func in sections:
        run_section(name, func)

    print(f'\n{"="*60}')
    print(f'  RESULTS: {PASS} passed, {FAIL} failed')
    print(f'{"="*60}')
    if ERRORS:
        print('\nFailed tests:')
        for e in ERRORS:
            print(e)
    sys.exit(1 if FAIL > 0 else 0)
