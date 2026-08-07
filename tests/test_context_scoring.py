import math
import os
import tempfile
import unittest
import gzip
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
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
from ml_scorer.feature_engine import FeatureEngine, PatternProfileUnavailable
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

    def test_combo_order_and_equal_sharpe_tie_match_training_insertion_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ('20_18.csv.gz', '10_8.csv.gz', '5_5.csv.gz', '10_10_PE2.csv.gz')
            for name in names:
                Path(tmp, name).touch()
            with mock.patch('ml_scorer.feature_engine.os.listdir', return_value=list(names)):
                definitions = self.engine._combo_definitions(tmp)
        self.assertEqual(
            [item['name'] for item in definitions],
            ['20_18', '10_8', '5_5', '10_10_PE2'],
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
        self.assertEqual(meta['best_combo'], '10_8')
        self.assertEqual(profile['pat_avg_profit2'], 5.0)

    def test_zero_recent_sharpe_preserves_v3_training_missingness(self):
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
        self.assertTrue(math.isnan(profile['pat_recent_vs_deep_sharpe']))

    def test_empty_recalculated_profile_uses_the_legacy_missing_value_contract(self):
        profile = self.engine._empty_pattern_profile()
        expected = {
            'pat_sharpe_ratio', 'pat_avg_profit2', 'pat_daysOut',
            'pat_concurrent_count', 'pat_neighbor_avg_wr',
            'pat_hit_last_year',
        }
        self.assertTrue(expected.issubset(profile))
        self.assertTrue(all(math.isnan(float(value)) for value in profile.values()))

    def test_context_concurrent_count_uses_training_tier_boundaries(self):
        active_pairs_by_date = {
            '2026-08-03': {
                (9, 'l'), (10, 'l'), (29, 'l'), (30, 's'), (31, 'l'),
                (59, 's'), (60, 'l'), (61, 'l'), (89, 'l'), (90, 's'),
            },
        }
        self.assertEqual(
            self.engine._context_concurrent_count(active_pairs_by_date, 29),
            3.0,
        )
        self.assertEqual(
            self.engine._context_concurrent_count(active_pairs_by_date, 59),
            3.0,
        )
        self.assertEqual(
            self.engine._context_concurrent_count(active_pairs_by_date, 89),
            3.0,
        )

    def test_selected_recurrence_reports_below_threshold_with_sample_size(self):
        dates = pd.to_datetime(['2000-01-03', '2026-08-04'])
        price_df = pd.DataFrame({
            'close': [100.0, 100.0],
            'high': [101.0, 101.0],
            'low': [99.0, 99.0],
        }, index=dates)
        returns = {
            year: (1.0 if year >= 2020 else -1.0)
            for year in range(2016, 2026)
        }

        def observation(_frame, _month, _day, year, _days_out, _direction):
            if year not in returns:
                return None
            return {
                'year': year,
                'return': returns[year],
                'favorable_return': 2.0,
            }

        self.engine._prepare_context_target = mock.Mock(return_value=(
            price_df, '/prices/AAPL.csv', '/opps/AAPL', ('version',),
        ))
        self.engine._compute_historical_observation = mock.Mock(
            side_effect=observation)
        summary = self.engine.compute_selected_recurrence_summary(
            '2', 'AAPL', '2026-08-05', 29, 'l', '10',
            {'selection': {'min_winning_years': '9'}},
        )

        self.assertEqual(summary['status'], 'below_threshold')
        self.assertEqual(summary['positive_years'], 6)
        self.assertEqual(summary['sample_size'], 10)
        self.assertEqual(summary['required_positive_years'], 9)
        self.assertEqual(summary['win_rate'], 0.6)
        self.assertTrue(summary['complete'])

    def test_context_concurrent_dates_match_training_forward_snap(self):
        price_index = pd.to_datetime([
            '2026-07-31',  # Friday
            '2026-08-03',  # Monday
            '2026-08-04',
        ])
        actual_entry, nominal_dates = self.engine._context_concurrent_dates(
            price_index, pd.Timestamp('2026-08-01'))
        self.assertEqual(actual_entry, pd.Timestamp('2026-08-03'))
        self.assertEqual(
            nominal_dates,
            ('2026-08-01', '2026-08-02', '2026-08-03'),
        )

        active_pairs_by_date = {
            date: {(29, 'l')} for date in nominal_dates
        }
        # Training counts each nominal MM-DD pattern separately after all three
        # have snapped to the same actual Monday entry.
        self.assertEqual(
            self.engine._context_concurrent_count(active_pairs_by_date, 29),
            3.0,
        )

    def test_context_lrus_evict_oldest_and_keep_resource_target_frames_correct(self):
        engine = FeatureEngine()
        loaded = {}
        engine.load_price_data = lambda _symbols: None
        engine._context_paths = lambda resource, symbol: (
            f'/prices/{resource}/{symbol}.csv',
            f'/opps/{resource}/{symbol}',
        )

        def load_frame(path):
            frame = object()
            loaded.setdefault(path, []).append(frame)
            return frame

        engine._load_csv = load_frame
        fake_stat = lambda path: SimpleNamespace(
            st_mtime_ns=sum(path.encode('utf-8')),
            st_ctime_ns=sum(path.encode('utf-8')) + 1,
            st_size=len(path),
        )
        with (
            mock.patch('ml_scorer.feature_engine.CONTEXT_TARGET_CACHE_MAX', 2),
            mock.patch('ml_scorer.feature_engine.os.stat', side_effect=fake_stat),
        ):
            us_cl = engine._prepare_context_target('2', 'CL')[0]
            etf_cl = engine._prepare_context_target('11', 'CL')[0]
            self.assertIsNot(us_cl, etf_cl)
            self.assertIs(engine._price_cache['CL'], etf_cl)

            engine._prepare_context_target('2', 'AAPL')
            self.assertNotIn(('2', 'CL'), engine._context_target_cache)
            self.assertIn(('11', 'CL'), engine._context_target_cache)
            self.assertIs(engine._price_cache['CL'], etf_cl)

            reloaded_us_cl = engine._prepare_context_target('2', 'CL')[0]
            self.assertIsNot(reloaded_us_cl, etf_cl)
            self.assertIs(engine._price_cache['CL'], reloaded_us_cl)

        cache = OrderedDict()
        self.assertIsNone(engine._lru_put(cache, 'one', 1, 2))
        self.assertIsNone(engine._lru_put(cache, 'two', 2, 2))
        self.assertEqual(engine._lru_get(cache, 'one'), 1)
        self.assertEqual(engine._lru_put(cache, 'three', 3, 2), ('two', 2))
        self.assertEqual(list(cache), ['one', 'three'])

    def test_evicted_pinned_market_frame_remains_available(self):
        engine = FeatureEngine()
        engine.load_price_data = lambda _symbols: None
        engine._context_paths = lambda resource, symbol: (
            f'/prices/{resource}/{symbol}.csv',
            f'/opps/{resource}/{symbol}',
        )
        engine._load_csv = lambda _path: object()
        fake_stat = SimpleNamespace(st_mtime_ns=1, st_ctime_ns=2, st_size=10)
        with (
            mock.patch('ml_scorer.feature_engine.CONTEXT_TARGET_CACHE_MAX', 1),
            mock.patch('ml_scorer.feature_engine.os.stat', return_value=fake_stat),
        ):
            spy = engine._prepare_context_target('11', 'SPY')[0]
            engine._prepare_context_target('2', 'AAPL')
        self.assertNotIn(('11', 'SPY'), engine._context_target_cache)
        self.assertIs(engine._price_cache['SPY'], spy)

    def test_target_and_definition_versions_detect_ctime_only_corrections(self):
        engine = FeatureEngine()
        engine.load_price_data = lambda _symbols: None
        engine._context_paths = lambda resource, symbol: (
            f'/prices/{resource}/{symbol}.csv',
            f'/opps/{resource}/{symbol}',
        )
        engine._load_csv = mock.Mock(side_effect=lambda _path: object())
        generation = {'ctime_ns': 10}

        def fake_stat(_path):
            return SimpleNamespace(
                st_mtime_ns=1,
                st_ctime_ns=generation['ctime_ns'],
                st_size=100,
            )

        with mock.patch(
                'ml_scorer.feature_engine.os.stat', side_effect=fake_stat):
            first = engine._prepare_context_target('2', 'AAPL')[0]
            self.assertIs(engine._prepare_context_target('2', 'AAPL')[0], first)
            generation['ctime_ns'] = 11
            corrected = engine._prepare_context_target('2', 'AAPL')[0]

        self.assertIsNot(corrected, first)
        self.assertEqual(engine._load_csv.call_count, 2)
        base = [{
            'name': '10_8', 'mtime_ns': 1, 'ctime_ns': 10, 'bytes': 100,
        }]
        changed = [dict(base[0], ctime_ns=11)]
        self.assertNotEqual(
            engine._definitions_version(base),
            engine._definitions_version(changed),
        )

    def test_snapshot_and_profile_call_paths_enforce_their_own_lru_bounds(self):
        engine = FeatureEngine()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, '10_8.csv.gz')
            with gzip.open(path, 'wt') as handle:
                handle.write(
                    'LorS,date,daysOut,sym,sharpe_ratio,avg_profit,'
                    'median_profit,x,avg_profit2\n'
                )
                for day in (1, 2, 3):
                    handle.write(
                        f'l,2026-08-0{day},29,AAPL,1.0,2.0,2.0,,3.0\n')
            stat = path.stat()
            definitions = [{
                'name': '10_8',
                'year1': 10,
                'year2': 8,
                'is_pe': False,
                'path': str(path),
                'mtime_ns': stat.st_mtime_ns,
                'ctime_ns': stat.st_ctime_ns,
                'bytes': stat.st_size,
            }]
            with mock.patch(
                    'ml_scorer.feature_engine.CONTEXT_SNAPSHOT_CACHE_MAX', 2):
                for day in (1, 2, 3):
                    engine._context_opp_snapshot(
                        tmp, f'2026-08-0{day}', definitions)
            self.assertEqual(len(engine._context_opp_snapshot_cache), 2)
            self.assertEqual(
                [key[1] for key in engine._context_opp_snapshot_cache],
                ['2026-08-02', '2026-08-03'],
            )

        price_frame = pd.DataFrame(
            {'close': [100.0, 101.0]},
            index=pd.to_datetime(['2000-01-03', '2026-08-04']),
        )
        definitions = [{
            'name': '10_8', 'year1': 10, 'year2': 8, 'is_pe': False,
            'path': '/opps/10_8.csv.gz', 'mtime_ns': 1, 'ctime_ns': 1,
            'bytes': 100,
        }]
        row = {
            'sharpe_ratio': 1.0,
            'avg_profit': 2.0,
            'median_profit': 2.0,
            'avg_profit2': 3.0,
        }
        snapshot = {
            'rows': {'10_8': {
                (29, 'l'): row, (59, 'l'): row, (89, 'l'): row,
            }},
            'active_pairs': {(29, 'l'), (59, 'l'), (89, 'l')},
        }
        with (
            mock.patch.object(
                engine, '_prepare_context_target',
                return_value=(
                    price_frame, '/prices/AAPL.csv', '/opps/AAPL',
                    ('/prices/AAPL.csv', 1, 1, 100),
                ),
            ),
            mock.patch.object(engine, '_combo_definitions', return_value=definitions),
            mock.patch.object(
                engine, '_compute_historical_observation', return_value={}),
            mock.patch.object(
                engine, '_combo_row_from_observations', return_value=row),
            mock.patch.object(
                engine, '_context_opp_snapshot', return_value=snapshot),
            mock.patch('ml_scorer.feature_engine.CONTEXT_PROFILE_CACHE_MAX', 2),
        ):
            for days_out in (29, 59, 89):
                engine.compute_recalculated_pattern_profile(
                    '2', 'AAPL', '2026-08-05', days_out, 'l')

        self.assertEqual(len(engine._recalculated_profile_cache), 2)
        self.assertEqual(
            [key[3] for key in engine._recalculated_profile_cache], [59, 89])

    def test_recalculation_accepts_bounded_authoritative_mfe_difference(self):
        engine = FeatureEngine()
        price_frame = pd.DataFrame(
            {'close': [100.0, 101.0]},
            index=pd.to_datetime(['2000-01-03', '2026-08-04']),
        )
        definitions = [{
            'name': '10_8', 'year1': 10, 'year2': 8, 'is_pe': False,
            'path': '/opps/10_8.csv.gz', 'mtime_ns': 1, 'ctime_ns': 1,
            'bytes': 100,
        }]
        dynamic_row = {
            'sharpe_ratio': 1.0,
            'avg_profit': 2.0,
            'median_profit': 2.0,
            'avg_profit2': 3.0,
        }
        prebuilt_row = dict(
            dynamic_row,
            sharpe_ratio=1.01,
            avg_profit2=3.1,
        )
        snapshot = {
            'rows': {'10_8': {(29, 'l'): prebuilt_row}},
            'active_pairs': {(29, 'l')},
        }
        with (
            mock.patch.object(
                engine, '_prepare_context_target',
                return_value=(
                    price_frame, '/prices/AAPL.csv', '/opps/AAPL',
                    ('/prices/AAPL.csv', 1, 1, 100),
                ),
            ),
            mock.patch.object(engine, '_combo_definitions', return_value=definitions),
            mock.patch.object(
                engine, '_compute_historical_observation', return_value={}),
            mock.patch.object(
                engine, '_combo_row_from_observations', return_value=dynamic_row),
            mock.patch.object(
                engine, '_context_opp_snapshot', return_value=snapshot),
        ):
            profile, metadata = engine.compute_recalculated_pattern_profile(
                '2', 'AAPL', '2026-08-05', 29, 'l')

        self.assertEqual(profile['pat_avg_profit2'], 3.1)
        self.assertEqual(profile['pat_sharpe_ratio'], 1.01)
        self.assertEqual(metadata['profile_validation'], 'bounded_authoritative')
        self.assertEqual(metadata['reconciled_model_fields'], [
            'pat_sharpe_ratio',
            'pat_avg_profit2',
        ])

    def test_recalculation_rejects_material_or_structural_profile_mismatch(self):
        engine = FeatureEngine()
        price_frame = pd.DataFrame(
            {'close': [100.0, 101.0]},
            index=pd.to_datetime(['2000-01-03', '2026-08-04']),
        )
        definitions = [{
            'name': '10_8', 'year1': 10, 'year2': 8, 'is_pe': False,
            'path': '/opps/10_8.csv.gz', 'mtime_ns': 1, 'ctime_ns': 1,
            'bytes': 100,
        }]
        dynamic_row = {
            'sharpe_ratio': 1.0,
            'avg_profit': 2.0,
            'median_profit': 2.0,
            'avg_profit2': 3.0,
        }
        snapshot = {
            'rows': {'10_8': {(29, 'l'): dict(dynamic_row, avg_profit2=4.0)}},
            'active_pairs': {(29, 'l')},
        }
        with (
            mock.patch.object(
                engine, '_prepare_context_target',
                return_value=(
                    price_frame, '/prices/AAPL.csv', '/opps/AAPL',
                    ('/prices/AAPL.csv', 1, 1, 100),
                ),
            ),
            mock.patch.object(engine, '_combo_definitions', return_value=definitions),
            mock.patch.object(
                engine, '_compute_historical_observation', return_value={}),
            mock.patch.object(
                engine, '_combo_row_from_observations', return_value=dynamic_row),
            mock.patch.object(
                engine, '_context_opp_snapshot', return_value=snapshot),
        ):
            with self.assertRaises(PatternProfileUnavailable) as caught:
                engine.compute_recalculated_pattern_profile(
                    '2', 'AAPL', '2026-08-05', 29, 'l')

        self.assertEqual(caught.exception.reason, 'prebuilt_profile_mismatch')
        self.assertTrue(caught.exception.details['qualifying_sets_match'])
        self.assertFalse(caught.exception.details['model_values_match'])
        self.assertEqual(
            caught.exception.details['mismatched_model_fields'],
            ['pat_avg_profit2'],
        )

        mismatch_engine = FeatureEngine()
        with (
            mock.patch.object(
                mismatch_engine, '_prepare_context_target',
                return_value=(
                    price_frame, '/prices/AAPL.csv', '/opps/AAPL',
                    ('/prices/AAPL.csv', 1, 1, 100),
                ),
            ),
            mock.patch.object(
                mismatch_engine, '_combo_definitions', return_value=definitions),
            mock.patch.object(
                mismatch_engine, '_compute_historical_observation', return_value={}),
            mock.patch.object(
                mismatch_engine, '_combo_row_from_observations',
                return_value=dynamic_row,
            ),
            mock.patch.object(
                mismatch_engine, '_context_opp_snapshot',
                return_value={'rows': {}, 'active_pairs': set()},
            ),
        ):
            with self.assertRaises(PatternProfileUnavailable) as caught:
                mismatch_engine.compute_recalculated_pattern_profile(
                    '2', 'AAPL', '2026-08-05', 29, 'l')
        self.assertEqual(caught.exception.reason, 'prebuilt_profile_mismatch')

    def test_recalculation_accepts_model_faithful_rounded_best_combo_tie(self):
        engine = FeatureEngine()
        price_frame = pd.DataFrame(
            {'close': [100.0, 101.0]},
            index=pd.to_datetime(['2000-01-03', '2026-08-04']),
        )
        definitions = [
            {
                'name': '7_6', 'year1': 7, 'year2': 6, 'is_pe': False,
                'path': '/opps/7_6.csv.gz', 'mtime_ns': 1, 'ctime_ns': 1,
                'bytes': 100,
            },
            {
                'name': '6_6', 'year1': 6, 'year2': 6, 'is_pe': False,
                'path': '/opps/6_6.csv.gz', 'mtime_ns': 1, 'ctime_ns': 1,
                'bytes': 100,
            },
        ]
        dynamic_rows = [
            {
                'sharpe_ratio': 1.10,
                'avg_profit': 4.0,
                'median_profit': 4.0,
                'avg_profit2': 12.0,
            },
            {
                'sharpe_ratio': 1.10,
                'avg_profit': 3.0,
                'median_profit': 3.0,
                'avg_profit2': 10.0,
            },
        ]
        prebuilt_rows = {
            '7_6': dict(dynamic_rows[0], sharpe_ratio=1.09),
            '6_6': dict(dynamic_rows[1]),
        }
        snapshot = {
            'rows': {
                name: {(89, 'l'): row}
                for name, row in prebuilt_rows.items()
            },
            'active_pairs': {(89, 'l')},
        }
        with (
            mock.patch.object(
                engine, '_prepare_context_target',
                return_value=(
                    price_frame, '/prices/PCAR.csv', '/opps/PCAR',
                    ('/prices/PCAR.csv', 1, 1, 100),
                ),
            ),
            mock.patch.object(engine, '_combo_definitions', return_value=definitions),
            mock.patch.object(
                engine, '_compute_historical_observation', return_value={}),
            mock.patch.object(
                engine, '_combo_row_from_observations', side_effect=dynamic_rows),
            mock.patch.object(
                engine, '_context_opp_snapshot', return_value=snapshot),
        ):
            profile, metadata = engine.compute_recalculated_pattern_profile(
                '2', 'PCAR', '2026-08-05', 89, 'l')

        self.assertEqual(profile['pat_avg_profit2'], 10.0)
        self.assertEqual(metadata['prebuilt_best_combo'], '6_6')
        self.assertEqual(metadata['dynamic_best_combo'], '7_6')
        self.assertEqual(
            metadata['profile_validation'], 'rounded_tie_authoritative')
        self.assertEqual(metadata['reconciled_model_fields'], [
            'pat_avg_profit2',
        ])

    def test_shared_tlt_and_spx_sources_refresh_on_ctime_change(self):
        engine = FeatureEngine()
        generation = {'ctime_ns': 10}
        loaded = []

        def exists(path):
            return path.endswith('/TLT.csv') or path.endswith('/SPX.csv')

        def fake_stat(path):
            return SimpleNamespace(
                st_mtime_ns=1,
                st_ctime_ns=generation['ctime_ns'],
                st_size=100 if path.endswith('/TLT.csv') else 200,
            )

        engine._load_csv = lambda path: loaded.append(path) or object()
        with (
            mock.patch('ml_scorer.feature_engine.ETF_CSV_DIR', '/etf'),
            mock.patch('ml_scorer.feature_engine.INDX_CSV_DIR', '/indx'),
            mock.patch('ml_scorer.feature_engine.COMM_CSV_DIR', '/comm'),
            mock.patch('ml_scorer.feature_engine.os.path.exists', side_effect=exists),
            mock.patch('ml_scorer.feature_engine.os.stat', side_effect=fake_stat),
        ):
            engine.load_price_data([])
            tlt_first = engine._price_cache['TLT']
            engine.load_price_data([])
            self.assertIs(engine._price_cache['TLT'], tlt_first)
            generation['ctime_ns'] = 11
            engine.load_price_data([])
            self.assertIsNot(engine._price_cache['TLT'], tlt_first)
        self.assertEqual(loaded, ['/etf/TLT.csv', '/etf/TLT.csv'])

        dates = pd.bdate_range('2020-01-02', periods=600)
        raw_spx = pd.DataFrame({
            'date': dates,
            'close': np.linspace(100.0, 150.0, len(dates)),
        })
        generation['ctime_ns'] = 20
        with (
            mock.patch('ml_scorer.feature_engine.CSV_DIR', '/csv'),
            mock.patch('ml_scorer.feature_engine.os.path.exists', return_value=True),
            mock.patch('ml_scorer.feature_engine.os.stat', side_effect=fake_stat),
            mock.patch(
                'ml_scorer.feature_engine.pd.read_csv',
                side_effect=lambda *_args, **_kwargs: raw_spx.copy(),
            ) as read_csv,
        ):
            engine._get_spx_seasonal_lookup(2026)
            engine._get_spx_seasonal_lookup(2026)
            self.assertEqual(read_csv.call_count, 1)
            generation['ctime_ns'] = 21
            engine._get_spx_seasonal_lookup(2026)
            self.assertEqual(read_csv.call_count, 2)


class MetadataTests(unittest.TestCase):
    def test_required_manifest_includes_fixed_income_sector_proxy(self):
        names = {
            name for name, _path in scorer_metadata._required_context_sources()
        }
        self.assertIn('ETF/TLT', names)

    def test_global_data_as_of_refreshes_when_eod_files_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            etf = root / 'ETF'
            indx = root / 'INDX'
            comm = root / 'COMM'
            for directory in (etf, indx, comm):
                directory.mkdir()

            with (
                mock.patch.object(scorer_metadata, 'CSV_DIR', str(root)),
                mock.patch.object(scorer_metadata, 'ETF_CSV_DIR', str(etf)),
                mock.patch.object(scorer_metadata, 'INDX_CSV_DIR', str(indx)),
                mock.patch.object(scorer_metadata, 'COMM_CSV_DIR', str(comm)),
            ):
                paths = [
                    Path(path)
                    for _name, path in scorer_metadata._required_context_sources()
                ]
                for path in paths:
                    path.write_text(',date,close\n0,2026-08-04,100\n', encoding='utf-8')
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

    def test_manifest_marks_any_declared_source_missing_and_same_date_correction_changes_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            etf = root / 'ETF'
            indx = root / 'INDX'
            comm = root / 'COMM'
            for directory in (etf, indx, comm):
                directory.mkdir()
            with (
                mock.patch.object(scorer_metadata, 'ETF_CSV_DIR', str(etf)),
                mock.patch.object(scorer_metadata, 'INDX_CSV_DIR', str(indx)),
                mock.patch.object(scorer_metadata, 'COMM_CSV_DIR', str(comm)),
            ):
                declared = scorer_metadata._required_context_sources()
                for _name, path in declared[:-1]:
                    Path(path).write_text(
                        ',date,close\n0,2026-08-05,100\n', encoding='utf-8')
                incomplete = scorer_metadata.context_data_manifest()
                self.assertFalse(incomplete['complete'])
                self.assertIsNone(incomplete['data_as_of'])
                self.assertEqual(incomplete['missing_sources'], [declared[-1][0]])

                corrected = Path(declared[0][1])
                Path(declared[-1][1]).write_text(
                    ',date,close\n0,2026-08-05,100\n', encoding='utf-8')
                before = scorer_metadata.context_data_manifest()
                corrected.write_text(
                    ',date,close\n0,2026-08-05,101\n', encoding='utf-8')
                # Some filesystems can assign the same ctime to two very fast
                # same-size writes. Model the ctime advance explicitly so this
                # remains a deterministic unit test of the manifest contract.
                real_stat = os.stat

                class StatWithAdvancedCtime:
                    def __init__(self, value):
                        self._value = value
                        self.st_ctime_ns = value.st_ctime_ns + 1

                    def __getattr__(self, name):
                        return getattr(self._value, name)

                def stat_with_advanced_ctime(path):
                    value = real_stat(path)
                    if os.path.abspath(path) == os.path.abspath(corrected):
                        return StatWithAdvancedCtime(value)
                    return value

                with mock.patch.object(
                    scorer_metadata.os,
                    'stat',
                    side_effect=stat_with_advanced_ctime,
                ):
                    after = scorer_metadata.context_data_manifest()
                self.assertTrue(after['complete'])
                self.assertEqual(after['data_as_of'], '2026-08-05')
                self.assertEqual(after['source_count'], len(declared))
                self.assertNotEqual(
                    before['data_generation_hash'], after['data_generation_hash'])


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
        self.assertEqual(meta['source'], 'dynamic_recalculation_validated')
        self.assertTrue(meta['prebuilt_validated'])
        self.assertEqual(meta['qualifying_combo_count'], 24)

        # All shared feature computations remain identical. Concurrent count is
        # asserted separately because the additive path fixes the legacy serving
        # bug that counted patterns outside the model's training tier.
        # Resource-free legacy scoring can find a current parquet for another
        # index universe first when a stock belongs to several universes. Force
        # the shared S&P gzip definitions here so this is a feature-math parity
        # test, not a market-resolution test.
        with mock.patch.object(
            engine, '_load_opp_from_parquet', return_value=None
        ):
            legacy_features = engine.compute_features(
                'AAPL', '2026-08-05', 29, 'l')
        self.assertEqual(len(FEATURE_COLS), 62)
        for name in FEATURE_COLS:
            self.assertIn(name, context_features)
            if name == 'pat_concurrent_count':
                continue
            left = context_features[name]
            right = legacy_features[name]
            if not (math.isfinite(float(left)) and math.isfinite(float(right))):
                self.assertTrue(math.isnan(float(left)) and math.isnan(float(right)), name)
            else:
                # Prebuilt opportunity rows are stored with float32 precision,
                # while the recalculated path keeps Python float precision.
                # A 1e-7 absolute tolerance catches material feature drift but
                # accepts the expected storage-rounding difference.
                self.assertAlmostEqual(
                    float(left), float(right), delta=1e-7, msg=name)

        patterns = set()
        opp_dir = REAL_DATA_ROOT / 'sp500/opp_by_symbol/AAPL'
        for path in opp_dir.glob('*.csv.gz'):
            with gzip.open(path, 'rt') as handle:
                header = handle.readline().rstrip().split(',')
                date_idx = header.index('date')
                days_idx = header.index('daysOut')
                direction_idx = header.index('LorS')
                for line in handle:
                    fields = line.rstrip().split(',')
                    days_out = int(fields[days_idx])
                    if 10 <= days_out <= 30:
                        patterns.add((
                            fields[date_idx][5:10],
                            days_out,
                            fields[direction_idx],
                        ))
        expected_concurrent = sum(
            1
            for month_day, _days_out, _direction in patterns
            if month_day == '08-05'
        )
        self.assertEqual(
            context_features['pat_concurrent_count'],
            float(expected_concurrent),
        )
        # The legacy route intentionally follows training's first
        # representative combo, while the context route uses its independently
        # validated tier-bounded active-pair snapshot. Their numeric values can
        # coincide for a particular data generation, so validate each source
        # instead of requiring an incidental difference.
        legacy_combos = engine._load_opp_files(
            'AAPL', date_hint='2026-08-05')
        first_lookup = next(iter(legacy_combos.values()), {})
        expected_legacy_concurrent = sum(
            1 for key in first_lookup if key[0] == '2026-08-05')
        self.assertEqual(
            legacy_features['pat_concurrent_count'],
            float(expected_legacy_concurrent),
        )

        intentional_nulls = {
            name for name in FEATURE_COLS
            if not math.isfinite(float(context_features[name]))
        }
        self.assertEqual(intentional_nulls, {'pat_recent_vs_deep_sharpe'})
        ordered_vector = [context_features[name] for name in FEATURE_COLS]
        self.assertEqual(len(ordered_vector), 62)

    def test_trgp_empty_60_and_90_day_profiles_match_manual_scoring_features(self):
        if not (
            (REAL_DATA_ROOT / 'csv/US/TRGP.csv').is_file()
            and (REAL_DATA_ROOT / 'sp500/opp_by_symbol/TRGP/10_8.csv.gz').is_file()
        ):
            self.skipTest('TRGP parity fixture data is not installed')

        for days_out in (59, 89):
            with self.subTest(calendar_days=days_out + 1):
                engine = FeatureEngine()
                context_features, meta = engine.compute_recalculated_features(
                    '2', 'TRGP', '2026-08-06', days_out, 'l')
                manual_features = engine.compute_features(
                    'TRGP', '2026-08-06', days_out, 'l')

                self.assertEqual(meta['profile_state'], 'no_qualifying_profile')
                self.assertEqual(meta['profile_validation'], 'exact_absence')
                self.assertEqual(meta['qualifying_combo_count'], 0)
                self.assertTrue(meta['prebuilt_validated'])
                for name in FEATURE_COLS:
                    left = float(context_features[name])
                    right = float(manual_features[name])
                    if math.isnan(left) or math.isnan(right):
                        self.assertTrue(
                            math.isnan(left) and math.isnan(right), name)
                    else:
                        self.assertAlmostEqual(left, right, places=12, msg=name)

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
            def compute_selected_recurrence_summary(self, *args):
                return {
                    'status': 'qualified',
                    'mode': 'consecutive',
                    'years': str(args[-2]),
                    'requested_observations': 10,
                    'sample_size': 10,
                    'positive_years': 9,
                    'required_positive_years': 8,
                    'win_rate': 0.9,
                    'complete': True,
                    'data_as_of': '2026-08-04',
                }

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
        self.assertEqual(len(item['feature_vector_hash']), 64)
        self.assertEqual(len(item['context_hash']), 64)
        self.assertNotIn('price_path', item['pattern_profile'])
        self.assertEqual(item['selected_recurrence']['sample_size'], 10)
        self.assertEqual(body['metadata']['feature_schema_version'], 'v3-62')

    def test_below_threshold_is_explanatory_and_does_not_gate_prediction(self):
        class BelowThresholdEngine:
            def compute_selected_recurrence_summary(self, *args):
                return {
                    'status': 'below_threshold',
                    'mode': 'consecutive',
                    'years': '10',
                    'requested_observations': 10,
                    'sample_size': 10,
                    'positive_years': 6,
                    'required_positive_years': 9,
                    'win_rate': 0.6,
                    'average_return_pct': 0.4,
                    'complete': True,
                    'data_as_of': '2026-08-04',
                }

            def compute_recalculated_features(self, *args):
                features = {name: 0.0 for name in FEATURE_COLS}
                features['mkt_vix_level'] = 20.0
                return features, {
                    'source': 'dynamic_raw_prices',
                    'prebuilt_validated': True,
                    'profile_validation': 'exact',
                    'reconciled_model_fields': [],
                    'qualifying_combo_count': 3,
                    'prebuilt_combo_count': 3,
                    'profile_hash': 'd' * 64,
                    'data_as_of': '2026-08-04',
                    'best_combo': '10_8',
                    'nullable_features': [],
                    '_active_pairs': {(29, 'l')},
                }

        self.app_module.engine = BelowThresholdEngine()
        response = self.client.post('/score/context', json={
            'resource_id': '2', 'symbol': 'AAPL', 'date': '2026-08-05',
            'calendar_days': 30, 'direction': 'l', 'years': '10',
            'partial': {'selection': {'min_winning_years': '9'}},
        })

        self.assertEqual(response.status_code, 200)
        item = response.get_json()['results'][0]
        self.assertEqual(item['status'], 'ok')
        self.assertTrue(item['pattern_recalculated'])
        self.assertEqual(item['selected_recurrence']['positive_years'], 6)
        self.assertEqual(item['selected_recurrence']['required_positive_years'], 9)
        self.assertEqual(item['ml_score'], 80.0)
        self.assertEqual(len(item['context_hash']), 64)

    def test_incomplete_selected_recurrence_does_not_gate_valid_v3_profile(self):
        class IncompleteRecurrenceEngine:
            def compute_selected_recurrence_summary(self, *args):
                return {
                    'status': 'insufficient_history',
                    'mode': 'consecutive',
                    'years': '10',
                    'requested_observations': 10,
                    'sample_size': 7,
                    'positive_years': 6,
                    'required_positive_years': 9,
                    'win_rate': round(6 / 7, 6),
                    'average_return_pct': 0.4,
                    'complete': False,
                    'data_as_of': '2026-08-04',
                }

            def compute_recalculated_features(self, *args):
                features = {name: 0.0 for name in FEATURE_COLS}
                features['mkt_vix_level'] = 20.0
                return features, {
                    'source': 'dynamic_raw_prices',
                    'prebuilt_validated': True,
                    'profile_validation': 'exact',
                    'reconciled_model_fields': [],
                    'qualifying_combo_count': 3,
                    'prebuilt_combo_count': 3,
                    'profile_hash': 'e' * 64,
                    'data_as_of': '2026-08-04',
                    'best_combo': '10_8',
                    'nullable_features': [],
                    '_active_pairs': {(29, 'l')},
                }

        self.app_module.engine = IncompleteRecurrenceEngine()
        response = self.client.post('/score/context', json={
            'resource_id': '2', 'symbol': 'AAPL', 'date': '2026-08-05',
            'calendar_days': 30, 'direction': 'l', 'years': '10',
            'partial': {'selection': {'min_winning_years': '9'}},
        })

        self.assertEqual(response.status_code, 200)
        item = response.get_json()['results'][0]
        self.assertEqual(item['status'], 'ok')
        self.assertEqual(item['ml_score'], 80.0)
        self.assertEqual(
            item['selected_recurrence']['status'],
            'insufficient_history',
        )

    def test_legacy_score_response_adds_metadata_without_changing_result_shape(self):
        class LegacyEngine:
            def load_price_data(self, symbols):
                self.symbols = symbols

            def compute_features(self, *args):
                features = {name: 0.0 for name in FEATURE_COLS}
                features['mkt_vix_level'] = 20.0
                return features

        self.app_module.engine = LegacyEngine()
        response = self.client.post('/score', json={
            'symbol': 'AAPL', 'date': '2026-08-05',
            'daysOut': 29, 'direction': 'l',
        })

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['results'][0]['ml_score'], 80.0)
        self.assertEqual(body['results'][0]['daysOut'], 29)
        self.assertEqual(body['metadata']['feature_schema_version'], 'v3-62')
        self.assertIn('data_generation_hash', body['metadata'])

    def test_feature_vector_hash_changes_for_same_date_feature_corrections(self):
        features = {name: 0.0 for name in FEATURE_COLS}
        original = self.app_module._feature_vector_hash(features)
        concurrent = dict(features)
        concurrent['pat_concurrent_count'] = 2.0
        market = dict(features)
        market['mkt_vix_level'] = 19.5

        self.assertNotEqual(
            original, self.app_module._feature_vector_hash(concurrent))
        self.assertNotEqual(
            original, self.app_module._feature_vector_hash(market))

    def test_context_batch_allows_15_distinct_identities_and_rejects_16(self):
        def item(index):
            return {
                'resource_id': '2',
                'symbol': f'S{index}',
                'date': '2026-08-05',
                'calendar_days': 30,
                'direction': 'l',
                'years': '20',
                'partial': None,
            }

        accepted = self.client.post(
            '/score/context',
            json={'opportunities': [item(index) for index in range(15)]},
        )
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(len(accepted.get_json()['results']), 15)

        rejected = self.client.post(
            '/score/context',
            json={'opportunities': [item(index) for index in range(16)]},
        )
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(
            rejected.get_json()['error']['code'], 'too_many_context_identities')

    def test_context_batch_allows_60_items_but_rejects_61(self):
        item = {
            'resource_id': '2',
            'symbol': 'AAPL',
            'date': '2026-08-05',
            'calendar_days': 30,
            'direction': 'l',
            'years': '20',
            'partial': None,
        }
        accepted = self.client.post(
            '/score/context', json={'opportunities': [item] * 60})
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(len(accepted.get_json()['results']), 60)

        rejected = self.client.post(
            '/score/context', json={'opportunities': [item] * 61})
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(rejected.get_json()['error']['code'], 'batch_too_large')

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
            def compute_selected_recurrence_summary(self, *args):
                return {'status': 'not_enforced'}

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

    def test_unvalidated_prebuilt_profile_is_structured_unavailable(self):
        class MismatchEngine:
            def compute_selected_recurrence_summary(self, *args):
                return {'status': 'not_enforced'}

            def compute_recalculated_features(self, *args):
                raise PatternProfileUnavailable(
                    'prebuilt_profile_mismatch',
                    {
                        'qualifying_sets_match': True,
                        'model_values_match': False,
                    },
                )

        self.app_module.engine = MismatchEngine()
        response = self.client.post('/score/context', json={
            'resource_id': '2', 'symbol': 'AAPL', 'date': '2026-08-05',
            'calendar_days': 30, 'direction': 'l', 'years': '20',
            'partial': False,
        })
        self.assertEqual(response.status_code, 200)
        item = response.get_json()['results'][0]
        self.assertEqual(item['status'], 'unavailable')
        self.assertEqual(item['error']['code'], 'prebuilt_profile_mismatch')
        self.assertFalse(item['error']['retryable'])
        self.assertEqual(item['error']['details'], {
            'qualifying_sets_match': True,
            'model_values_match': False,
        })


if __name__ == '__main__':
    unittest.main()
