"""Version and artifact metadata for reproducible scorer contexts."""

import hashlib
import os
import time
from functools import lru_cache

try:
    from .config import (
        CALIBRATION_DIR,
        COMM_CSV_DIR,
        CONTEXT_SCHEMA_VERSION,
        CSV_DIR,
        FEATURE_COLS,
        FEATURE_SCHEMA_VERSION,
        ETF_CSV_DIR,
        INDX_CSV_DIR,
        MODEL_DIR,
        MODEL_RELEASE,
        PATTERN_PROFILE_SCHEMA_VERSION,
        TIERS,
    )
    from .context_contract import sha256_json
except ImportError:
    from config import (
        CALIBRATION_DIR,
        COMM_CSV_DIR,
        CONTEXT_SCHEMA_VERSION,
        CSV_DIR,
        FEATURE_COLS,
        FEATURE_SCHEMA_VERSION,
        ETF_CSV_DIR,
        INDX_CSV_DIR,
        MODEL_DIR,
        MODEL_RELEASE,
        PATTERN_PROFILE_SCHEMA_VERSION,
        TIERS,
    )
    from context_contract import sha256_json


_PROCESS_DATA_GENERATION = hashlib.sha256(
    f'{os.getpid()}:{time.time_ns()}'.encode('utf-8')
).hexdigest()


def _required_context_sources():
    """Every shared price series read by the 62-feature serving path."""
    etfs = (
        'SPY', 'HYG', 'LQD', 'XLK', 'XLU', 'XLF', 'XLE', 'XLV', 'XLY',
        'XLC', 'XLI', 'XLP', 'XLRE', 'XLB', 'TLT',
    )
    indices = (
        'VIX', 'VIX3M', 'US10Y', 'US2Y', 'ADVN', 'DECN', 'IRX', 'DXY', 'SPX',
    )
    commodities = ('CL', 'GC')
    records = [
        (f'ETF/{symbol}', os.path.join(ETF_CSV_DIR, f'{symbol}.csv'))
        for symbol in etfs
    ]
    records.extend(
        (f'INDX/{symbol}', os.path.join(INDX_CSV_DIR, f'{symbol}.csv'))
        for symbol in indices
    )
    records.extend(
        (f'COMM/{symbol}', os.path.join(COMM_CSV_DIR, f'{symbol}.csv'))
        for symbol in commodities
    )
    return records


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(kind, tier, target, name, path):
    exists = os.path.isfile(path)
    return {
        'kind': kind,
        'tier': tier,
        'target': target,
        'name': name,
        'path': os.path.basename(path),
        'bytes': os.path.getsize(path) if exists else None,
        'sha256': _sha256_file(path) if exists else None,
        'present': exists,
    }


@lru_cache(maxsize=1)
def model_manifest():
    """Return the 18-model + 6-calibration deployment manifest."""
    artifacts = []
    for tier_name, config in sorted(TIERS.items()):
        for target in ('sr', 'mfe'):
            for algorithm, filename in sorted(config[target].items()):
                artifacts.append(_artifact_record(
                    'model', tier_name, target, algorithm,
                    os.path.join(MODEL_DIR, filename),
                ))
            calibration_name = config[f'calibration_{target}']
            artifacts.append(_artifact_record(
                'calibration', tier_name, target, 'empirical_bins',
                os.path.join(CALIBRATION_DIR, calibration_name),
            ))

    hash_material = [
        {
            'kind': item['kind'],
            'tier': item['tier'],
            'target': item['target'],
            'name': item['name'],
            'path': item['path'],
            'bytes': item['bytes'],
            'sha256': item['sha256'],
            'present': item['present'],
        }
        for item in artifacts
    ]
    return {
        'release': MODEL_RELEASE,
        'artifact_count': len(artifacts),
        'all_present': all(item['present'] for item in artifacts),
        'artifacts': artifacts,
        'manifest_hash': sha256_json(hash_material),
    }


def _tail_date(path):
    """Read the final CSV record's YYYY-MM-DD without loading the full file."""
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell() - 1
        line = b''
        while position >= 0:
            handle.seek(position)
            char = handle.read(1)
            if char == b'\n' and line:
                break
            if char not in (b'\n', b'\r'):
                line = char + line
            position -= 1
    try:
        fields = line.decode('utf-8').split(',')
    except UnicodeDecodeError:
        return None
    for field in fields[:3]:
        candidate = field.strip()
        if len(candidate) == 10 and candidate[4] == '-' and candidate[7] == '-':
            return candidate
    return None


def context_data_manifest():
    """Return a live completeness and generation fingerprint for V3 inputs.

    ``ctime_ns`` intentionally participates in the identity so a same-date
    correction still invalidates downstream caches even when a copy preserves
    file mtime and byte size. The process generation adds a final safe boundary
    around the scorer's authoritative nightly restart.
    """
    sources = []
    missing = []
    dates = []
    for name, path in _required_context_sources():
        present = os.path.isfile(path)
        record = {'name': name, 'present': present}
        if present:
            stat = os.stat(path)
            tail_date = _tail_date(path)
            record.update({
                'bytes': stat.st_size,
                'mtime_ns': stat.st_mtime_ns,
                'ctime_ns': stat.st_ctime_ns,
                'tail_date': tail_date,
            })
            if tail_date:
                dates.append(tail_date)
            else:
                missing.append(name)
        else:
            record.update({
                'bytes': None,
                'mtime_ns': None,
                'ctime_ns': None,
                'tail_date': None,
            })
            missing.append(name)
        sources.append(record)
    complete = not missing
    source_manifest_hash = sha256_json(sources)
    return {
        'complete': complete,
        'missing_sources': missing,
        'data_as_of': min(dates) if complete and dates else None,
        'source_count': len(sources),
        'source_manifest_hash': source_manifest_hash,
        'data_generation_hash': sha256_json({
            'source_manifest_hash': source_manifest_hash,
            'process_generation': _PROCESS_DATA_GENERATION,
        }),
    }


def global_data_as_of():
    """Return the latest common date only when every declared input is present."""
    return context_data_manifest()['data_as_of']


def service_metadata():
    manifest = model_manifest()
    data_manifest = context_data_manifest()
    return {
        'model_release': MODEL_RELEASE,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'feature_schema_hash': sha256_json(FEATURE_COLS),
        'context_schema_version': CONTEXT_SCHEMA_VERSION,
        'pattern_profile_schema_version': PATTERN_PROFILE_SCHEMA_VERSION,
        'model_manifest_hash': manifest['manifest_hash'],
        'data_as_of': data_manifest['data_as_of'],
        'data_generation_hash': data_manifest['data_generation_hash'],
        'data_source_manifest_hash': data_manifest['source_manifest_hash'],
        'context_data_complete': data_manifest['complete'],
        'context_data_source_count': data_manifest['source_count'],
        'missing_context_data_sources': data_manifest['missing_sources'],
    }
