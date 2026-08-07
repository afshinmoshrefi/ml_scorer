"""Validation and hashing helpers for the additive /score/context contract."""

import hashlib
import json
import math
import re
from datetime import datetime

try:
    from .config import CONTEXT_RESOURCE_IDS, tier_for_days_out
except ImportError:
    from config import CONTEXT_RESOURCE_IDS, tier_for_days_out


CONTEXT_CALENDAR_DAYS = frozenset((30, 60, 90))
_SYMBOL_RE = re.compile(r'^[A-Z0-9][A-Z0-9._-]{0,31}$')
_ALLOWED_FIELDS = frozenset({
    'resource_id', 'symbol', 'date', 'calendar_days',
    'direction', 'years', 'partial',
})
_FORBIDDEN_DERIVED_FIELDS = frozenset({'daysOut', 'days_out', 'tier', 'entry_date'})


class ContextValidationError(ValueError):
    """A stable, per-item API validation error."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def canonical_json(value):
    """Return deterministic strict JSON (NaN/Infinity are not permitted)."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def _validate_json_provenance(value):
    """Require a finite JSON value while preserving it byte-for-byte logically."""
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ContextValidationError(
            'invalid_partial_provenance',
            f'partial must be a finite JSON value: {exc}',
        ) from exc

    def _walk(node):
        if isinstance(node, float) and not math.isfinite(node):
            raise ContextValidationError(
                'invalid_partial_provenance',
                'partial cannot contain NaN or Infinity',
            )
        if isinstance(node, dict):
            for key, child in node.items():
                if not isinstance(key, str):
                    raise ContextValidationError(
                        'invalid_partial_provenance',
                        'partial object keys must be strings',
                    )
                _walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)

    _walk(value)


def validate_context_opportunity(raw):
    """Validate one request item and derive raw daysOut/tier.

    TradeWave windows are inclusive CALENDAR-day windows.  The entry day is
    day 1, so the scorer's legacy raw offset is always ``calendar_days - 1``.
    The API intentionally does not accept a caller-supplied daysOut or tier.
    """
    if not isinstance(raw, dict):
        raise ContextValidationError(
            'invalid_opportunity',
            f'opportunity must be an object, got {type(raw).__name__}',
        )

    forbidden = sorted(_FORBIDDEN_DERIVED_FIELDS.intersection(raw))
    if forbidden:
        raise ContextValidationError(
            'derived_field_not_allowed',
            f'{", ".join(forbidden)} is derived by /score/context and must not be supplied',
        )

    unknown = sorted(set(raw) - _ALLOWED_FIELDS)
    if unknown:
        raise ContextValidationError(
            'unknown_field',
            f'unknown field(s): {", ".join(unknown)}',
        )

    missing = sorted(_ALLOWED_FIELDS - set(raw))
    if missing:
        raise ContextValidationError(
            'missing_required_field',
            f'missing required field(s): {", ".join(missing)}',
        )

    resource_id = raw['resource_id']
    if not isinstance(resource_id, str) or resource_id not in CONTEXT_RESOURCE_IDS:
        raise ContextValidationError(
            'unsupported_resource',
            'resource_id must be one of 0, 1, 2, 3, 4, or 11',
        )

    symbol = raw['symbol']
    if not isinstance(symbol, str) or not symbol:
        raise ContextValidationError('invalid_symbol', 'symbol must be a nonempty string')
    normalized_symbol = symbol.strip().upper()
    if symbol != normalized_symbol or not _SYMBOL_RE.fullmatch(symbol):
        raise ContextValidationError(
            'invalid_symbol',
            'symbol must already be uppercase and contain only letters, numbers, dot, underscore, or hyphen',
        )

    date_text = raw['date']
    if not isinstance(date_text, str):
        raise ContextValidationError('invalid_date', 'date must be a YYYY-MM-DD string')
    try:
        parsed_date = datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError as exc:
        raise ContextValidationError('invalid_date', 'date must be a real YYYY-MM-DD date') from exc
    if parsed_date.strftime('%Y-%m-%d') != date_text:
        raise ContextValidationError('invalid_date', 'date must use exact YYYY-MM-DD format')

    calendar_days = raw['calendar_days']
    if isinstance(calendar_days, bool) or not isinstance(calendar_days, int):
        raise ContextValidationError(
            'invalid_calendar_days',
            'calendar_days must be the integer 30, 60, or 90',
        )
    if calendar_days not in CONTEXT_CALENDAR_DAYS:
        raise ContextValidationError(
            'invalid_calendar_days',
            'calendar_days must be exactly 30, 60, or 90',
        )

    direction = raw['direction']
    if direction not in ('l', 's'):
        raise ContextValidationError('invalid_direction', 'direction must be exactly "l" or "s"')

    years = raw['years']
    if not isinstance(years, str) or not years.strip():
        raise ContextValidationError('invalid_years', 'years must be a nonempty string')

    partial = raw['partial']
    _validate_json_provenance(partial)

    days_out = calendar_days - 1
    return {
        'resource_id': resource_id,
        'symbol': symbol,
        'date': date_text,
        'calendar_days': calendar_days,
        'daysOut': days_out,
        'direction': direction,
        # This is provenance/cache identity.  It is intentionally not converted
        # to a V3 model feature because V3 was trained on the all-combo profile.
        'years': years,
        'partial': partial,
        'tier': tier_for_days_out(days_out),
    }


def error_payload(code, message, retryable=False):
    return {
        'status': 'unavailable',
        'error': {
            'code': code,
            'message': message,
            'retryable': bool(retryable),
        },
    }
