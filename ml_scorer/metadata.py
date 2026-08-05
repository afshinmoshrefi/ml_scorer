"""Version and artifact metadata for reproducible scorer contexts."""

import hashlib
import os
from functools import lru_cache

try:
    from .config import (
        CALIBRATION_DIR,
        COMM_CSV_DIR,
        CONTEXT_SCHEMA_VERSION,
        CSV_DIR,
        FEATURE_COLS,
        FEATURE_SCHEMA_VERSION,
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
        INDX_CSV_DIR,
        MODEL_DIR,
        MODEL_RELEASE,
        PATTERN_PROFILE_SCHEMA_VERSION,
        TIERS,
    )
    from context_contract import sha256_json


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


def global_data_as_of():
    """Return the latest common date across the daily V3 market context.

    This deliberately is not process-lifetime cached.  EOD files are replaced
    while the service stays up, and stale metadata would poison downstream
    cache identity.  Five tail-line reads are cheap enough for health/context.
    """
    paths = [
        os.path.join(CSV_DIR, 'ETF', 'SPY.csv'),
        os.path.join(INDX_CSV_DIR, 'VIX.csv'),
        os.path.join(INDX_CSV_DIR, 'DXY.csv'),
        os.path.join(COMM_CSV_DIR, 'CL.csv'),
        os.path.join(COMM_CSV_DIR, 'GC.csv'),
    ]
    dates = [value for value in (_tail_date(path) for path in paths) if value]
    return min(dates) if dates else None


def service_metadata():
    manifest = model_manifest()
    return {
        'model_release': MODEL_RELEASE,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'feature_schema_hash': sha256_json(FEATURE_COLS),
        'context_schema_version': CONTEXT_SCHEMA_VERSION,
        'pattern_profile_schema_version': PATTERN_PROFILE_SCHEMA_VERSION,
        'model_manifest_hash': manifest['manifest_hash'],
        'data_as_of': global_data_as_of(),
    }
