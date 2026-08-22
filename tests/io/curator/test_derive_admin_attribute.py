"""Tests for `derive_admin_attribute`.

Its motivating use is a stable external identifier: `admin3_id` is
openplaces' own and can be re-minted (a re-mint may hand `US-NC-CD` to a
different county than it named before), while a federal FIPS code is not
ours to change.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_admin_attribute


def _state(frame, admin_id=None):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=admin_id or AdminId('US', 'NC', 'CD'),
        verbose=False,
        timer=None,
        curated=frame,
    )


def _spine(monkeypatch, frame):
    import openplaces.io.readers as readers

    monkeypatch.setattr(readers, 'get_admin', lambda *a, **k: frame)


def test_the_attribute_is_copied_onto_every_entity(monkeypatch):
    _spine(
        monkeypatch,
        pd.DataFrame(
            {
                'admin3_id': ['US-NC-CD', 'US-NC-PD'],
                'admin3_id_admin1': ['37029', '37141'],
            }
        ),
    )
    frame = pd.DataFrame({'admin3_id': ['US-NC-CD', 'US-NC-CD', 'US-NC-PD']})
    out = derive_admin_attribute(
        _state(frame), attribute='admin3_id_admin1', output='county_fips'
    ).curated
    assert list(out['county_fips']) == ['37029', '37029', '37141']


def test_an_unknown_admin_id_gets_no_value(monkeypatch):
    _spine(
        monkeypatch,
        pd.DataFrame({'admin3_id': ['US-NC-CD'], 'admin3_id_admin1': ['37029']}),
    )
    out = derive_admin_attribute(
        _state(pd.DataFrame({'admin3_id': ['US-NC-ZZ']})),
        attribute='admin3_id_admin1',
        output='county_fips',
    ).curated
    assert pd.isna(out['county_fips'].iloc[0])


def test_a_missing_admin_id_column_is_a_no_op(monkeypatch):
    _spine(monkeypatch, pd.DataFrame({'admin3_id': [], 'admin3_id_admin1': []}))
    out = derive_admin_attribute(
        _state(pd.DataFrame({'other': [1]})),
        attribute='admin3_id_admin1',
        output='county_fips',
    ).curated
    assert 'county_fips' not in out.columns


def test_a_spine_without_the_attribute_is_a_no_op(monkeypatch):
    _spine(monkeypatch, pd.DataFrame({'admin3_id': ['US-NC-CD'], 'name': ['Camden']}))
    out = derive_admin_attribute(
        _state(pd.DataFrame({'admin3_id': ['US-NC-CD']})),
        attribute='admin3_id_admin1',
        output='county_fips',
    ).curated
    assert 'county_fips' not in out.columns


def test_the_level_is_inferred_from_the_column_name(monkeypatch):
    seen = {}

    import openplaces.io.readers as readers

    def fake(admin_id=None, level=None, **kwargs):
        seen['level'] = level
        return pd.DataFrame(
            {'admin4_id': ['US-NC-CD-X'], 'admin4_id_admin1': ['3702912345']}
        )

    monkeypatch.setattr(readers, 'get_admin', fake)
    derive_admin_attribute(
        _state(pd.DataFrame({'admin4_id': ['US-NC-CD-X']})),
        attribute='admin4_id_admin1',
        output='cousub',
        admin_id_column='admin4_id',
    )
    assert seen['level'] == 4


def test_an_uninferable_column_name_raises(monkeypatch):
    _spine(monkeypatch, pd.DataFrame({'zone': ['a'], 'x': ['1']}))
    with pytest.raises(ValueError, match='cannot infer admin_level'):
        derive_admin_attribute(
            _state(pd.DataFrame({'zone': ['a']})),
            attribute='x',
            output='y',
            admin_id_column='zone',
        )
