"""Address reconciliation does less work and returns the same.

plans/slow-steps-same-output-faster.md, R2. Three changes, each held
against what it replaces, on fabricated addresses only (invented
street names and numbers):

- `parse_address_components` equals ``parse_address(...).components``
  on every branch: empty and non-text input, the 'none' backend,
  usaddress, and the regex fallback for strings usaddress cannot tag;
- `assemble_formatted_components(format_address_components(**c))`
  equals `harmonize_address_case(**c)`;
- `clean_text` remembers results for exact str input only, in a cache
  of bounded size, and returns what the unremembered function returns.

`reconcile_addresses_df` is then run with the previous parse and an
unremembered `clean_text` patched back in and compared with the new
code exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openplaces.geo import address
from openplaces.geo.address import (
    ADDRESS_COMPONENTS,
    CLEAN_TEXT_CACHE_SIZE,
    _clean_str,
    _clean_uncached,
    assemble_formatted_components,
    clean_text,
    format_address_components,
    harmonize_address_case,
    parse_address,
    parse_address_components,
)
from openplaces.io.harmonizer import addresses as haddr

ADDRESSES = [
    '12 Fernvale Rd',
    '40 Quillon Ave Unit 2B',
    '40 Quillon Ave #3',
    '7-9 Brantmoor St',
    '704-706 Brantmoor St Apt 4',
    'PO Box 5521',
    'P.O. BOX 12, Mirelton, NC 28599',
    '3 Oakhurst Ln, Coastal Vale, NC 28500',
    '3 Oakhurst Ln Coastal Vale NC 285001234',
    '18 Rue Élodie Château',
    '55 Ñandú Ct',
    'Oakhurst Ln',
    '1200 N Seagrass Blvd S',
    '22 Twentysecond St NE Ste 100',
    '1 1 1 1',
    'Lot 7 Harrowgate Subdivision',
    '12 Fernvale Rd + 14 Fernvale Rd',
    '12 Fernvale Rd; 9 Pellow Way',
    '#5 Quillon Ave',
    '   ',
    '',
    '000 Zero Rd',
    'Fernvale Rd Fernvale Rd Fernvale Rd 12 12',
    # Two cities and ZIP codes: usaddress repeats a label, regex parses.
    '12 Fernvale Rd, Mirelton, NC 28599, Coastal Vale, NC 28500',
    '704-706 Brantmoor St Mirelton NC 28599 Mirelton NC 28599',
]


@pytest.mark.parametrize('text', ADDRESSES + [None, 42, float('nan')])
@pytest.mark.parametrize('backend', ['auto', 'usaddress', 'none'])
@pytest.mark.parametrize('admin1_id', [None, 'US', 'DE'])
def test_components_equal_parse_address(text, backend, admin1_id):
    assert parse_address_components(text, backend, admin1_id) == (
        parse_address(text, backend, admin1_id).components
    )


def test_regex_fallback_branch_is_covered():
    # A string usaddress tags with a repeated label goes to the regex
    # fallback; at least one fixture must reach it.
    address._parse_cached.cache_clear()
    parsers = {
        parse_address(text, admin1_id='US').metadata['parser']
        for text in ADDRESSES
        if text.strip()
    }
    assert {'usaddress', 'regex'} <= parsers


def _component_sets():
    for text in ADDRESSES:
        parsed = parse_address(text, admin1_id='US').components
        yield {k: v or None for k, v in parsed.items()}
    yield dict.fromkeys(ADDRESS_COMPONENTS)
    yield {
        'address_number': '12',
        'address_street': 'II FERNVALE 3RD RD',
        'unit_number': '4B',
        'city': 'MIRELTON',
        'state': 'NC',
        'postal_code': '285991234',
    }


@pytest.mark.parametrize('admin1_id', [None, 'US', 'DE'])
def test_assembled_line_equals_harmonize_address_case(admin1_id):
    for components in _component_sets():
        formatted = format_address_components(**components, admin1_id=admin1_id)
        assert assemble_formatted_components(formatted, admin1_id) == (
            harmonize_address_case(**components, admin1_id=admin1_id)
        )


class _Text(str):
    """A str subclass that writes itself differently in upper case."""

    def upper(self):
        return 'SUBCLASS'


def test_clean_text_cache_is_pure_and_bounded():
    _clean_str.cache_clear()
    samples = ADDRESSES + ['ÉLODIE', 'élodie', 'a  b\t c', '#4-B', '']
    for text in samples * 3:
        assert clean_text(text) == _clean_uncached(text)
    assert clean_text(None) == '' and clean_text(12) == ''
    # A subclass is not served the entry of an equal plain string.
    assert clean_text('ab') == 'AB'
    assert clean_text(_Text('ab')) == _clean_uncached(_Text('ab'))
    assert _clean_str.cache_info().maxsize == CLEAN_TEXT_CACHE_SIZE == 65_536
    for i in range(CLEAN_TEXT_CACHE_SIZE + 10):
        clean_text(f'{i} Fernvale Rd')
    assert _clean_str.cache_info().currsize == CLEAN_TEXT_CACHE_SIZE


def _old_parse_address_list(value, admin1_id):
    """`_parse_address_list` with the previous full parse."""
    from openplaces.geo.address import (
        compress_number_list,
        normalize_address_components,
        split_address_list,
    )

    def _one(text):
        parsed = parse_address(text, admin1_id=admin1_id).components
        return normalize_address_components(
            address_street=parsed['address_street'] or '',
            address_number=parsed['address_number'],
            unit_number=parsed['unit_number'],
            postal_code=parsed['postal_code'],
            city=parsed['city'],
            state=parsed['state'],
            admin1_id=admin1_id,
        )

    parts = split_address_list(value)
    if len(parts) < 2:
        return _one(value)
    normalized = [_one(p) for p in parts]
    first = next((n for n in normalized if n['address_street']), normalized[0])
    street = first['address_street']
    numbers = [
        n['address_number']
        for n in normalized
        if n['address_street'] == street and n['address_number']
    ]
    tokens = [t for number in numbers for t in number.split('-') if t]
    combined = compress_number_list(' '.join(tokens)) if len(tokens) > 1 else None
    result = dict(first)
    if combined is not None:
        result['address_number'] = combined
    return result


def _frame(n=600, seed=3):
    rng = np.random.default_rng(seed)
    full = rng.choice(np.array(ADDRESSES + [None], dtype=object), n)
    streets = np.array(
        ['Fernvale Rd', 'FERNVALE ROAD', 'Quillon Ave', 'Brantmoor St', None],
        dtype=object,
    )
    return pd.DataFrame(
        {
            'address_parcel': pd.Series(full, dtype='str'),
            'address_street_dwelling_overture': rng.choice(streets, n),
            'address_number_dwelling_overture': pd.Series(
                rng.choice([12.0, 40.0, 7.0, np.nan], n)
            ),
            'city_dwelling_overture': rng.choice(
                np.array(['Mirelton', 'Coastal Vale', None], dtype=object), n
            ),
            'postal_code_dwelling_overture': rng.choice(
                np.array(['28599', '28500', None], dtype=object), n
            ),
        },
        index=pd.Index([f'b{i:04d}' for i in range(n)], name='building_id'),
    )


SOURCES = {
    'parcel': {'address_full': 'address_parcel'},
    'dwelling_overture': {
        'address_street': 'address_street_dwelling_overture',
        'address_number': 'address_number_dwelling_overture',
        'city': 'city_dwelling_overture',
        'postal_code': 'postal_code_dwelling_overture',
    },
}


def test_reconcile_addresses_df_matches_previous_calls(monkeypatch):
    kwargs = dict(
        sources=SOURCES,
        complete_from_admin={'state': 2},
        number_output_col='address_number',
        unit_output_col='address_unit',
        city_output_col='city',
    )
    _clean_str.cache_clear()
    address._parse_cached.cache_clear()
    new = haddr.reconcile_addresses_df(_frame(), 'US-NC-XXX', False, **kwargs)

    # The previous parse, and clean_text without its memory; the line
    # assembly is held against harmonize_address_case above.
    monkeypatch.setattr(haddr, '_parse_address_list', _old_parse_address_list)
    monkeypatch.setattr(
        address,
        'clean_text',
        lambda t: _clean_uncached(t) if isinstance(t, str) else '',
    )
    address._parse_cached.cache_clear()
    old = haddr.reconcile_addresses_df(_frame(), 'US-NC-XXX', False, **kwargs)
    pd.testing.assert_frame_equal(new, old, check_exact=True, check_dtype=True)
    for column in old.columns:
        assert [type(v) for v in new[column]] == [type(v) for v in old[column]]
