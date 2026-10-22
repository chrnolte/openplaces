"""Tests for `link_by_id`'s `roll_values` rules (links/roll_values.py).

A county roll joined onto a spine whose values a statewide layer
supplied: the tax year follows the values the roll wrote, a parcel the
roll has not yet valued takes the roll's own prior-year figures, and,
where the roll covers the unit, a statewide value does not remain on a
parcel the roll does not carry. All ids and values are fabricated.
"""

import pandas as pd
import pytest

import openplaces.io.harmonizer.links as links
from openplaces.core.provenance import NOT_ON_ROLL
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.links import roll_values as rv
from tests.links_patching import patch_links

ROLL = 'US-XX-YY_property-roll-2026'
VALUES = ['land_value', 'improvement_value', 'total_value']
SPEC = {
    'values': VALUES,
    'year': 'tax_year',
    'prior_year': {
        'land_value': 'land_value_prior_year',
        'improvement_value': 'improvement_value_prior_year',
        'total_value': 'total_value_prior_year',
    },
}


def _state(spine):
    return HarmonizeState(
        recipe={}, admin_id='US-XX-YY', verbose=False, timer=None, spine=spine
    )


def _statewide_spine(keys, land=None):
    """A spine whose values and labels a statewide layer supplied."""
    n = len(keys)
    land = land if land is not None else [100.0 * (i + 1) for i in range(n)]
    frame = pd.DataFrame(
        {
            'parcel_id_local': keys,
            'land_value': land,
            'improvement_value': [10.0] * n,
            'total_value': [x + 10.0 for x in land],
            'tax_year': [2025.0] * n,
        }
    )
    for v in VALUES:
        frame[f'{v}_source'] = pd.Series(['statewide'] * n, dtype=object)
    return frame


def _roll(rows):
    columns = ['parcel_id_local', *VALUES, 'tax_year', *SPEC['prior_year'].values()]
    return pd.DataFrame(rows, columns=columns)


def _link(monkeypatch, spine, roll, spec, **kwargs):
    patch_links(monkeypatch, 'get_entities', lambda *a, **k: roll)
    monkeypatch.setattr(
        rv, '_source_scopes', lambda state: {'statewide': 2, 'county': 3, 'roll': 3}
    )
    return links.link_by_id(
        _state(spine),
        ROLL,
        mode='aggregate',
        columns=[*VALUES, 'tax_year'],
        track_provenance=VALUES,
        roll_values=spec,
        **kwargs,
    ).spine


def test_tax_year_follows_only_the_values_the_roll_wrote(monkeypatch):
    spine = _statewide_spine(['A', 'B', 'C'])
    nan = float('nan')
    roll = _roll(
        [
            ['A', 150.0, 20.0, 170.0, 2026.0, nan, nan, nan],
            # On the roll, but no value and no prior year.
            ['B', nan, nan, nan, 2026.0, nan, nan, nan],
            ['C', 350.0, 20.0, 370.0, 2026.0, nan, nan, nan],
        ]
    )
    out = _link(monkeypatch, spine, roll, {**SPEC, 'prior_year': {}})
    # B keeps the statewide value, and with it the statewide year.
    assert out['tax_year'].tolist() == [2026.0, 2025.0, 2026.0]
    assert out['land_value'].tolist() == [150.0, 200.0, 350.0]
    assert out['land_value_source'].tolist() == ['roll', 'statewide', 'roll']


def test_pending_record_takes_its_own_prior_year(monkeypatch):
    spine = _statewide_spine(['A', 'B', 'C'])
    nan = float('nan')
    roll = _roll(
        [
            ['A', 150.0, 20.0, 170.0, 2026.0, 140.0, 15.0, 155.0],
            ['B', nan, nan, nan, 2026.0, 190.0, 40.0, 230.0],
        ]
    )
    out = _link(monkeypatch, spine, roll, SPEC)
    # A keeps its 2026 figures; B takes its own 2025 ones, before the
    # statewide layer's, labeled with the roll and the year before.
    assert out['land_value'].tolist() == [150.0, 190.0, 300.0]
    assert out['improvement_value'].tolist() == [20.0, 40.0, 10.0]
    assert out['total_value'].tolist() == [170.0, 230.0, 310.0]
    assert out['tax_year'].tolist() == [2026.0, 2025.0, 2025.0]
    assert out['total_value_source'].tolist() == ['roll', 'roll', 'statewide']


def test_prior_year_never_mixes_with_a_partly_valued_record(monkeypatch):
    spine = _statewide_spine(['A', 'B'])
    nan = float('nan')
    # B states a 2026 total but no land: it is valued, not pending.
    roll = _roll([['B', nan, nan, 260.0, 2026.0, 190.0, 40.0, 230.0]])
    out = _link(monkeypatch, spine, roll, SPEC)
    assert out.loc[1, 'land_value'] == 200.0
    assert out.loc[1, 'total_value'] == 260.0
    assert out.loc[1, 'tax_year'] == 2026.0


def test_prior_year_columns_are_not_written_to_the_spine(monkeypatch):
    spine = _statewide_spine(['A'])
    nan = float('nan')
    roll = _roll([['A', nan, nan, nan, 2026.0, 90.0, 5.0, 95.0]])
    out = _link(monkeypatch, spine, roll, SPEC)
    assert not any(c.endswith('_prior_year') for c in out.columns)


def _covering_roll(keys):
    """A roll valuing every key it carries, at 2026 figures."""
    return _roll([[k, 1.0, 1.0, 2.0, 2026.0, None, None, None] for k in keys])


def test_statewide_value_is_emptied_where_the_covering_roll_omits_the_key(
    monkeypatch,
):
    keys = [f'K{i:03d}' for i in range(200)]
    spine = _statewide_spine(keys)
    # K198 and K199 are not on the roll: 99.0% coverage.
    roll = _covering_roll(keys[:198])
    out = _link(monkeypatch, spine, roll, {**SPEC, 'not_on_roll': True})
    gone = out['parcel_id_local'].isin(['K198', 'K199'])
    for v in VALUES:
        assert out.loc[gone, v].isna().all()
        assert (out.loc[gone, f'{v}_source'] == NOT_ON_ROLL).all()
        assert out.loc[~gone, v].notna().all()
    # The year stays: it is the statewide record's, and nothing says
    # the parcel has another one.
    assert (out.loc[gone, 'tax_year'] == 2025.0).all()


def test_values_from_a_county_scoped_source_are_kept(monkeypatch):
    keys = [f'K{i:03d}' for i in range(100)]
    spine = _statewide_spine(keys)
    spine.loc[99, 'land_value_source'] = 'county'
    roll = _covering_roll(keys[:99])
    out = _link(monkeypatch, spine, roll, {**SPEC, 'not_on_roll': True})
    assert out.loc[99, 'land_value'] == 10000.0
    assert out.loc[99, 'land_value_source'] == 'county'
    assert pd.isna(out.loc[99, 'improvement_value'])


def test_a_roll_under_the_coverage_bar_empties_nothing(monkeypatch):
    keys = [f'K{i:03d}' for i in range(100)]
    spine = _statewide_spine(keys)
    roll = _covering_roll(keys[:98])  # 98%, under the 99% default
    out = _link(monkeypatch, spine, roll, {**SPEC, 'not_on_roll': True})
    assert out[VALUES].notna().all().all()
    assert (out.loc[98:, 'land_value_source'] == 'statewide').all()


def test_a_stacked_lot_whose_units_the_roll_carries_is_kept(monkeypatch):
    keys = [f'K{i:03d}' for i in range(99)] + ['LOT-1']
    spine = _statewide_spine(keys)
    # The lot's key is minted at ingest and on no roll; its units are.
    roll = _covering_roll(keys[:99] + ['U-1', 'U-2'])
    pairs = pd.DataFrame({'unit_key': ['U-1', 'U-2'], 'lot_key': ['LOT-1'] * 2})
    out = _link(
        monkeypatch,
        spine,
        roll,
        {**SPEC, 'not_on_roll': True},
        _unit_lot_pairs=pairs,
    )
    lot = out['parcel_id_local'] == 'LOT-1'
    assert out.loc[lot, 'land_value'].tolist() == [2.0]
    assert out.loc[lot, 'land_value_source'].tolist() == ['roll']


def test_a_spine_row_without_a_key_is_kept(monkeypatch):
    keys = [f'K{i:03d}' for i in range(199)] + [None]
    spine = _statewide_spine(keys)
    roll = _covering_roll(keys[:199])
    out = _link(monkeypatch, spine, roll, {**SPEC, 'not_on_roll': True})
    assert out.loc[199, 'land_value'] == 20000.0


def test_a_table_without_values_triggers_no_rule(monkeypatch):
    keys = [f'K{i:03d}' for i in range(100)]
    spine = _statewide_spine(keys)
    detail = pd.DataFrame({'parcel_id_local': keys[:1], 'n_bedrooms': [3.0]})
    patch_links(monkeypatch, 'get_entities', lambda *a, **k: detail)
    out = links.link_by_id(
        _state(spine),
        ROLL,
        mode='aggregate',
        columns=['n_bedrooms'],
        roll_values={**SPEC, 'not_on_roll': True, 'min_coverage': 0.01},
    ).spine
    assert out[VALUES].notna().all().all()


def test_without_roll_values_nothing_changes(monkeypatch):
    spine = _statewide_spine(['A', 'B', 'C'])
    nan = float('nan')
    roll = _roll(
        [
            ['A', 150.0, 20.0, 170.0, 2026.0, nan, nan, nan],
            ['B', nan, nan, nan, 2026.0, 190.0, 40.0, 230.0],
            ['C', 350.0, 20.0, 370.0, 2026.0, nan, nan, nan],
        ]
    )
    patch_links(monkeypatch, 'get_entities', lambda *a, **k: roll)
    out = links.link_by_id(
        _state(spine), ROLL, mode='aggregate', columns=[*VALUES, 'tax_year']
    ).spine
    # The behavior the rules exist to change: B's statewide 2025 value
    # under the roll's 2026 year, and no prior-year fill.
    assert out['land_value'].tolist() == [150.0, 200.0, 350.0]
    assert out['tax_year'].tolist() == [2026.0, 2026.0, 2026.0]


@pytest.mark.parametrize(
    'spec',
    [
        {'values': []},
        {'values': ['land_value'], 'extra': 1},
        {'values': ['land_value'], 'prior_year': {'n_rooms': 'x'}},
        {'values': ['land_value'], 'min_coverage': 0},
    ],
)
def test_parse_rejects_a_malformed_block(spec):
    with pytest.raises(ValueError):
        rv.parse_roll_values(spec)


def test_recipe_scope_counts_admin_components():
    assert rv._recipe_scope('US-TX-HAR_property-hcad-2026') == 3
    assert rv._recipe_scope('US-TX_parcel-txgio-2025') == 2
    assert rv._recipe_scope('dwelling-overture-2025') == 0


def test_source_scopes_rank_harris_sources():
    state = HarmonizeState(
        recipe={},
        admin_id=AdminId('US-TX-HAR'),
        verbose=False,
        timer=None,
        spine=None,
    )
    scopes = rv._source_scopes(state)
    assert scopes['txgio'] == 2
    assert scopes['hcad'] == 3
    labels = pd.Series(['txgio', 'hcad', 'txgio+hcad', 'unknown', None])
    assert rv._coarser_than(labels, scopes, 3).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_parcel_geospine_declares_the_roll_rules():
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id('US_parcel-geospine-2026')
    step = next(
        s
        for s in recipe['pipeline']
        if s.get('step') == 'link_by_id' and s.get('entity_type') == 'property'
    )
    spec = rv.parse_roll_values(step['roll_values'])
    assert spec['year'] == 'tax_year'
    assert spec['not_on_roll'] is True
    assert set(spec['prior_year']) == set(VALUES)
    passes = [
        s
        for s in recipe['pipeline']
        if s.get('step') == 'link_by_id' and s.get('entity_type') == 'parcel'
    ]
    assert all('total_value' in s['track_provenance'] for s in passes)


def test_hcad_maps_its_prior_year_columns():
    from openplaces.core.attribute_registry import load_registry
    from openplaces.recipe import get_recipe_by_id

    columns = get_recipe_by_id('US-TX-HAR_property-hcad-2026')['columns']
    assert columns['land_value_prior_year'] == 'prior_land_val'
    assert columns['improvement_value_prior_year'] == 'prior_bld_val'
    assert columns['total_value_prior_year'] == 'prior_tot_mkt_val'
    registry = load_registry()
    for v in VALUES:
        assert registry.loc[f'{v}_prior_year', 'aggregation'] == 'sum'
