import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault('ML_SCORER_DATA_DIR', '/home/flask/data')
os.environ.setdefault('ML_SCORER_SKIP_INIT', '1')

import numpy as np
import pandas as pd

from ml_scorer.config import FEATURE_COLS
from ml_scorer.context_contract import (
    ContextValidationError,
    validate_context_opportunity,
)
from ml_scorer.feature_engine import FeatureEngine
from ml_scorer import metadata as scorer_metadata


REAL_DATA_ROOT = Path('/home/flask/data')


class ContextContractTests(unittest.TestCase):
    def test_inclusive_calendar_days_are_derived_once(self):
        partial = {
            'enabled': True,
            'min_winning_years': 8,
            'source_pattern_calendar_days': 150,
        }
        item = validate_context_opportunity({
            'resource_id': '2',
            'symbol': 'AAPL',
            'date': '2026-08-05',
            'calendar_days': 30,
            'direction': 'l',
            'years': '20',
            'partial': partial,
        })
        self.assertEqual(item['daysOut'], 29)
        self.assertEqual(item['tier'], '10_30')
        self.assertEqual(item['years'], '20')
        self.assertEqual(item['partial'], partial)

    def test_caller_cannot_supply_raw_offset_or_tier(self):
        base = {
            'resource_id': '2', 'symbol': 'AAPL', 'date': '2026-08-05',
            'calendar_days': 60, 'direction': 's', 'years': 'pe2-10',
            'partial': False,
        }
        for field, value in [('daysOut', 59), ('tier', '31_60'), ('entry_date', '2026-08-05')]:
            invalid = dict(base)
            invalid[field] = value
            with self.assertRaises(ContextValidationError) as caught:
                validate_context_opportunity(invalid)
            self.assertEqual(caught.exception.code, 'derived_field_not_allowed')

    def test_only_us_stock_and_etf_resources_are_accepted(self):
        with self.assertRaises(ContextValidationError) as caught:
            validate_context_opportunity({
                'resource_id': '7', 'symbol': 'CL', 'date': '2026-08-05',
                'calendar_days': 90, 'direction': 'l', 'years': '20',
                'partial': None,
            })
        self.assertEqual(caught.exception.code, 'unsupported_resource')


class RecalculationUnitTests(unittest.TestCase):
    def setUp(self):
        self.engine = FeatureEngine()

    def test_raw_end_is_start_plus_days_out_even_across_feb_29(self):
        dates = pd.bdate_range('2024-01-02', '2024-04-15')
        frame = pd.DataFrame({
            'close': np.linspace(100.0, 130.0, len(dates)),
            'high': np.linspace(101.0, 131.0, len(dates)),
            'low': np.linspace(99.0, 129.0, len(dates)),
        }, index=dates)
        observation = self.engine._compute_historical_observation(
            frame, 2, 1, 2024, 29, 'l')
        self.assertIsNotNone(observation)
        self.assertEqual(observation['nominal_start'], pd.Timestamp('2024-02-01'))
        self.assertEqual(observation['nominal_end'], pd.Timestamp('2024-03-01'))

    def test_combo_order_and_equal_sharpe_tie_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('20_18.csv.gz', '5_5.csv.gz', '10_10_PE2.csv.gz', '10_8.csv.gz'):
                Path(tmp, name).touch()
            definitions = self.engine._combo_definitions(tmp)
        self.assertEqual(
            [item['name'] for item in definitions],
            ['5_5', '10_8', '10_10_PE2', '20_18'],
        )

        rows = {
            '5_5': {
                'sharpe_ratio': 1.0, 'avg_profit': 2.0,
                'median_profit': 2.0, 'avg_profit2': 3.0,
            },
            '10_8': {
                'sharpe_ratio': 1.0, 'avg_profit': 4.0,
                'median_profit': 4.0, 'avg_profit2': 5.0,
            },
        }
        profile, meta = self.engine._aggregate_pattern_profile(
            rows, definitions, 30.0, 'l', 29)
        self.assertEqual(meta['best_combo'], '5_5')
        self.assertEqual(profile['pat_avg_profit2'], 3.0)

    def test_zero_recent_sharpe_is_a_real_zero_ratio(self):
        definitions = [
            {'name': '10_10', 'year1': 10, 'year2': 10, 'is_pe': False},
            {'name': '20_18', 'year1': 20, 'year2': 18, 'is_pe': False},
        ]
        rows = {
            '10_10': {
                'sharpe_ratio': 0.0, 'avg_profit': 1.0,
                'median_profit': 1.0, 'avg_profit2': 2.0,
            },
            '20_18': {
                'sharpe_ratio': 2.0, 'avg_profit': 3.0,
                'median_profit': 3.0, 'avg_profit2': 4.0,
            },
        }
        profile, _ = self.engine._aggregate_pattern_profile(
            rows, definitions, 30.0, 'l', 29)
        self.assertEqual(profile['pat_recent_vs_deep_sharpe'], 0.0)


class MetadataTests(unittest.TestCase):
    def test_global_data_as_of_refreshes_when_eod_files_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            etf = root / 'ETF'
            indx = root / 'INDX'
            comm = root / 'COMM'
            for directory in (etf, indx, comm):
                directory.mkdir()

            paths = [
                etf / 'SPY.csv', indx / 'VIX.csv', indx / 'DXY.csv',
                comm / 'CL.csv', comm / 'GC.csv',
            ]
            for path in paths:
                path.write_text(',date,close\n0,2026-08-04,100\n', encoding='utf-8')

            with (
                mock.patch.object(scorer_metadata, 'CSV_DIR', str(root)),
                mock.patch.object(scorer_metadata, 'INDX_CSV_DIR', str(indx)),
                mock.patch.object(scorer_metadata, 'COMM_CSV_DIR', str(comm)),
            ):
                self.assertEqual(scorer_metadata.global_data_as_of(), '2026-08-04')
                paths[0].write_text(
                    ',date,close\n0,2026-08-04,100\n1,2026-08-05,101\n',
                    encoding='utf-8',
                )
                # The common as-of stays at the slower files.
                self.assertEqual(scorer_metadata.global_data_as_of(), '2026-08-04')
                for path in paths[1:]:
                    path.write_text(
                        ',date,close\n0,2026-08-04,100\n1,2026-08-05,101\n',
                        encoding='utf-8',
                    )
                self.assertEqual(scorer_metadata.global_data_as_of(), '2026-08-05')


@unittest.skipUnless(
    (REAL_DATA_ROOT / 'csv/US/AAPL.csv').is_file()
    and (REAL_DATA_ROOT / 'sp500/opp_by_symbol/AAPL/10_8.csv.gz').is_file(),
    'TradeWave parity fixture data is not installed',
)
class RealDataParityTests(unittest.TestCase):
    def test_aapl_exact_horizon_matches_prebuilt_and_has_complete_v3_vector(self):
        engine = FeatureEngine()
        context_features, meta = engine.compute_recalculated_features(
            '2', 'AAPL', '2026-08-05', 29, 'l')
        self.assertEqual(meta['source'], 'prebuilt_exact_horizon_validated')
        self.assertTrue(meta['prebuilt_validated'])
        self.assertEqual(meta['qualifying_combo_count'], 24)

        # Legacy V3 behavior must remain identical when the exact prebuilt row
        # is authoritative.  The new path differs only when it must rebuild.
        legacy_features = engine.compute_features('AAPL', '2026-08-05', 29, 'l')
        self.assertEqual(len(FEATURE_COLS), 62)
        for name in FEATURE_COLS:
            self.assertIn(name, context_features)
            left = context_features[name]
            right = legacy_features[name]
            if not (math.isfinite(float(left)) and math.isfinite(float(right))):
                self.assertTrue(math.isnan(float(left)) and math.isnan(float(right)), name)
            else:
                self.assertAlmostEqual(float(left), float(right), places=10, msg=name)

        intentional_nulls = {
            name for name in FEATURE_COLS
            if not math.isfinite(float(context_features[name]))
        }
        self.assertEqual(intentional_nulls, {'pat_recent_vs_deep_sharpe'})
        ordered_vector = [context_features[name] for name in FEATURE_COLS]
        self.assertEqual(len(ordered_vector), 62)

    def test_equity_cl_and_crude_context_use_distinct_caches(self):
        if not (REAL_DATA_ROOT / 'csv/US/CL.csv').is_file():
            self.skipTest('CL equity CSV is not installed')
        engine = FeatureEngine()
        engine.load_price_data(['CL'])
        equity = engine._get_price_df('CL')
        crude = engine._get_commodity_price_df('CL')
        self.assertIsNotNone(equity)
        self.assertIsNotNone(crude)
        self.assertIsNot(equity, crude)
        self.assertNotEqual(float(equity['close'].iloc[-1]), float(crude['close'].iloc[-1]))


class ContextEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ml_scorer import app as app_module
        cls.app_module = app_module
        cls.client = app_module.app.test_client()

    def setUp(self):
        self.original_engine = self.app_module.engine
        self.original_tiers = dict(self.app_module.scorer_mgr.tiers)

        class FakeTier:
            def predict(self, features):
                return {
                    'pred_return': 2.0, 'pred_mfe': 4.0,
                    'win_prob': 0.75, 'p_hit_return': 0.55,
                    'p_hit_mfe': 0.45, 'ml_score': 80.0,
                }

        class FakeEngine:
            def compute_recalculated_features(self, *args):
                features = {name: 0.0 for name in FEATURE_COLS}
                features['mkt_vix_level'] = 20.0
                return features, {
                    'source': 'dynamic_raw_prices',
                    'prebuilt_validated': False,
                    'qualifying_combo_count': 3,
                    'prebuilt_combo_count': 0,
                    'profile_hash': 'a' * 64,
                    'data_as_of': '2026-08-04',
                    'best_combo': '10_8',
                    'nullable_features': [],
                    '_active_pairs': {(29, 'l')},
                    'price_path': '/private/path',
                }

        self.app_module.engine = FakeEngine()
        self.app_module.scorer_mgr.tiers = {'10_30': FakeTier()}

    def tearDown(self):
        self.app_module.engine = self.original_engine
        self.app_module.scorer_mgr.tiers = self.original_tiers

    def test_success_shape_keeps_years_as_provenance(self):
        response = self.client.post('/score/context', json={
            'resource_id': '2', 'symbol': 'AAPL', 'date': '2026-08-05',
            'calendar_days': 30, 'direction': 'l', 'years': 'pe2-10',
            'partial': {'enabled': True},
        })
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        item = body['results'][0]
        self.assertEqual(item['status'], 'ok')
        self.assertEqual(item['daysOut'], 29)
        self.assertEqual(item['tier'], '10_30')
        self.assertEqual(item['years'], 'pe2-10')
        self.assertTrue(item['pattern_recalculated'])
        self.assertEqual(len(item['context_hash']), 64)
        self.assertNotIn('price_path', item['pattern_profile'])
        self.assertEqual(body['metadata']['feature_schema_version'], 'v3-62')

    def test_validation_failure_is_per_item_and_structured(self):
        response = self.client.post('/score/context', json={'opportunities': [{
            'resource_id': '7', 'symbol': 'CL', 'date': '2026-08-05',
            'calendar_days': 30, 'direction': 'l', 'years': '20',
            'partial': False,
        }]})
        self.assertEqual(response.status_code, 200)
        item = response.get_json()['results'][0]
        self.assertEqual(item['status'], 'unavailable')
        self.assertEqual(item['error']['code'], 'unsupported_resource')
        self.assertFalse(item['error']['retryable'])

    def test_vix_block_is_stable_and_distinguishable(self):
        class VixBlockedEngine:
            def compute_recalculated_features(self, *args):
                features = {name: 0.0 for name in FEATURE_COLS}
                features['mkt_vix_level'] = 40.0
                return features, {
                    'source': 'dynamic_raw_prices',
                    'prebuilt_validated': False,
                    'qualifying_combo_count': 3,
                    'prebuilt_combo_count': 0,
                    'profile_hash': 'b' * 64,
                    'data_as_of': '2026-08-04',
                    'best_combo': '10_8',
                    'nullable_features': [],
                    '_active_pairs': {(29, 'l')},
                }

        self.app_module.engine = VixBlockedEngine()
        response = self.client.post('/score/context', json={
            'resource_id': '2', 'symbol': 'AAPL', 'date': '2026-08-05',
            'calendar_days': 30, 'direction': 'l', 'years': '20',
            'partial': False,
        })
        self.assertEqual(response.status_code, 200)
        item = response.get_json()['results'][0]
        self.assertTrue(item['vix_blocked'])
        self.assertEqual(item['error']['code'], 'vix_blocked')
        self.assertFalse(item['error']['retryable'])


if __name__ == '__main__':
    unittest.main()
