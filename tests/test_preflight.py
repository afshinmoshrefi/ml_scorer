import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_ROOT = REPO_ROOT / 'ml_scorer'
sys.path.insert(0, str(SCORER_ROOT))

os.environ.setdefault('ML_SCORER_DATA_DIR', '/home/flask/data')
os.environ.setdefault('ML_SCORER_SKIP_INIT', '1')

import preflight


class PreflightSamplingTests(unittest.TestCase):
    def _sample(self, frame, n=40):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp, 'sp500')
            cache_dir.mkdir()
            Path(cache_dir, 'ml_cache_2026-08-21.parquet').touch()
            with mock.patch.object(preflight.C, 'DATA_DIR', tmp), \
                    mock.patch('pandas.read_parquet', return_value=frame):
                return preflight._sample_opportunities(
                    '61_90', 61, 90, n, '2026-08-21')

    @staticmethod
    def _ordered_frame():
        rows = []
        for index in range(120):
            rows.append({
                'date': '2026-08-21',
                'sym': f'L{index:03d}',
                'daysOut': 61 + index % 30,
                'LorS': 'l',
            })
        for index in range(40):
            rows.append({
                'date': '2026-08-21',
                'sym': f'S{index:03d}',
                'daysOut': 61 + index % 30,
                'LorS': 's',
            })
        return pd.DataFrame(rows)

    def test_sample_is_direction_representative_and_order_independent(self):
        frame = self._ordered_frame()
        forward = self._sample(frame)
        reverse = self._sample(frame.iloc[::-1].reset_index(drop=True))

        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward), 40)
        self.assertEqual(Counter(item[2] for item in forward), {'l': 30, 's': 10})
        self.assertTrue(all(61 <= item[1] <= 90 for item in forward))

    def test_sample_returns_all_unique_rows_when_limit_exceeds_population(self):
        frame = self._ordered_frame().head(8)
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

        sample = self._sample(duplicate, n=80)

        self.assertEqual(len(sample), 8)
        self.assertEqual(len(set(sample)), 8)


if __name__ == '__main__':
    unittest.main()
