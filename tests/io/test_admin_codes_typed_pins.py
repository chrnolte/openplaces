"""Siblings that share a name pin on their name and type together.

Minsk the region and Minsk the city are both called Minsk under Belarus
and carry no license-clean code, so neither name nor code can say which
spine unit a re-ingested row is. Without a pin both rows were minted
afresh, their polygons matched no spine unit, and their population
weights fell back to an estimate, which is enough to flip a contested
code on the next re-mint.
"""

import pandas as pd
import pytest

from openplaces.io.admin_codes import assign_admin_ids, registry
from openplaces.io.admin_codes import frame as frame_module
from openplaces.io.admin_codes.anchors import normalize_name


@pytest.fixture
def twin_names(monkeypatch):
    # A fabricated parent whose region and city share one name.
    pins = {
        ('XX-AA', normalize_name('Twin'), 'Region'): 'TR',
        ('XX-AA', normalize_name('Twin'), 'City'): 'TC',
    }
    monkeypatch.setattr(
        frame_module,
        'load_registry',
        lambda level, sep: (pins, {}, {'XX-AA': {'TR', 'TC'}}),
    )
    monkeypatch.setattr(frame_module, 'load_group_code_lengths', lambda: {})


def _assign(types):
    df = pd.DataFrame(
        {'admin2_id': ['XX-AA', 'XX-AA'], 'name': ['Twin', 'Twin'], 'type': types}
    )
    return assign_admin_ids(
        df, new_admin_id_col='admin3_id', parent_admin_id_col='admin2_id'
    )


def test_name_and_type_pin_each_twin(twin_names):
    out = _assign(['City', 'Region'])
    assert dict(zip(out['type'], out.index)) == {
        'City': 'XX-AA-TC',
        'Region': 'XX-AA-TR',
    }
    assert (out['admin3_id_source'] == 'pinned').all()


def test_a_repeated_pair_pins_neither(twin_names):
    # Two rows reading Twin, City cannot both be the one spine unit.
    out = _assign(['City', 'City'])
    assert out.index.is_unique
    assert not (out['admin3_id_source'] == 'pinned').any()


def test_the_registry_records_name_and_type(monkeypatch, tmp_path):
    spine = tmp_path / 'spine.csv'
    spine.write_text(
        'admin3_id,name,type\nXX-AA-TR,Twin,Region\nXX-AA-TC,Twin,City\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(registry, 'spine_path', lambda level: spine)
    registry.load_registry.cache_clear()
    try:
        pins, _, _ = registry.load_registry(3)
    finally:
        registry.load_registry.cache_clear()
    assert ('XX-AA', normalize_name('Twin')) not in pins
    assert pins[('XX-AA', normalize_name('Twin'), 'Region')] == 'TR'
    assert pins[('XX-AA', normalize_name('Twin'), 'City')] == 'TC'


def test_a_punctuation_variant_still_pins(monkeypatch):
    # The spine says "Trans Nzoia"; the source says "Trans-Nzoia".
    from openplaces.io.admin_codes.anchors import normalize_name

    pins = {('XX', normalize_name('Trans Nzoia')): 'TN'}
    monkeypatch.setattr(
        frame_module, 'load_registry', lambda level, sep: (pins, {}, {'XX': {'TN'}})
    )
    monkeypatch.setattr(frame_module, 'load_group_code_lengths', lambda: {})
    out = assign_admin_ids(
        pd.DataFrame({'admin1_id': ['XX'], 'name': ['Trans-Nzoia']}),
        new_admin_id_col='admin2_id',
        parent_admin_id_col='admin1_id',
    )
    assert list(out.index) == ['XX-TN']
    assert out['admin2_id_source'].iloc[0] == 'pinned'


def test_two_names_that_normalize_alike_pin_neither(monkeypatch):
    # "Nairobi" and "Nairobi City" both normalize to the pinned key, so
    # the pin identifies neither; both are minted, and stay distinct.
    from openplaces.io.admin_codes.anchors import normalize_name

    pins = {('XX', normalize_name('Nairobi')): 'NA'}
    monkeypatch.setattr(
        frame_module, 'load_registry', lambda level, sep: (pins, {}, {'XX': {'NA'}})
    )
    monkeypatch.setattr(frame_module, 'load_group_code_lengths', lambda: {})
    out = assign_admin_ids(
        pd.DataFrame({'admin1_id': ['XX', 'XX'], 'name': ['Nairobi', 'Nairobi City']}),
        new_admin_id_col='admin2_id',
        parent_admin_id_col='admin1_id',
    )
    assert out.index.is_unique
    assert not (out['admin2_id_source'] == 'pinned').any()
