"""The scorer's volatility gate uses the live VIX for current dates (2026-11-24)."""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_scorer'))
import live_vix  # noqa: E402

NOW = dt.datetime(2026, 11, 24, 15, 42, tzinfo=dt.timezone.utc).timestamp()


def test_current_date_uses_the_live_quote_not_the_old_close():
    # Price files end Nov 20 (35.89); the live VIX is 27.26.
    vix = live_vix.gate_vix({'mkt_vix_level': 35.89}, '2026-11-24', '2026-11-20', '2026-11-20',
                            now=NOW, fetch=lambda: (27.26, NOW - 60))
    assert vix == {'value': 27.26, 'source': 'live', 'time': '2026-11-24T15:41+00:00'}


def test_historical_date_keeps_its_own_close():
    vix = live_vix.gate_vix({'mkt_vix_level': 41.2}, '2026-04-07', '2026-11-23', '2026-04-07',
                            now=NOW, fetch=lambda: (27.26, NOW))
    assert vix == {'value': 41.2, 'source': 'close', 'date': '2026-04-07'}


def test_without_a_live_quote_the_close_is_used():
    vix = live_vix.gate_vix({'mkt_vix_level': 35.89}, '2026-11-24', '2026-11-20', '2026-11-20',
                            now=NOW, fetch=lambda: None)
    assert vix['source'] == 'close' and vix['value'] == 35.89


def test_an_old_quote_is_not_live():
    assert live_vix.live_vix(now=NOW, fetch=lambda: (30.0, NOW - 10 * 86400)) is None


def test_no_vix_at_all_returns_none():
    assert live_vix.gate_vix({}, '2026-11-24', None, None, now=NOW, fetch=lambda: None) is None


def test_blocked_fields_report_value_source_and_time():
    fields = live_vix.blocked_fields({'value': 41.27, 'source': 'live', 'time': '2026-11-24T15:41+00:00'}, 35)
    assert fields == {'vix_value': 41.27, 'vix_cutoff': 35, 'vix_source': 'live', 'vix_time': '2026-11-24T15:41+00:00'}


import unittest  # noqa: E402


class LiveVixGateTest(unittest.TestCase):
    """Runs the tests above under unittest (python -m unittest tests.test_live_vix)."""


for _name, _func in list(globals().items()):
    if _name.startswith("test_") and callable(_func):
        setattr(LiveVixGateTest, _name, staticmethod(_func))
