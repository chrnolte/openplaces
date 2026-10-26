"""A curated parcel gets its property attributes reduced from the property
spine, with the rule each column needs, filling only what it lacks.

The parcel geospine copies no property-level attribute (year built,
living area, room counts) from a roll any more; the curate step reads the
harmonized property spine for the unit and reduces it per parcel. The
year built is the earliest property's, the living area the sum, and a
parcel whose own layer already states a value keeps it.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, aggregation


def _state(curated):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US', 'XX', 'YY'),
        verbose=False,
        timer=None,
        curated=curated,
    )


@pytest.fixture
def properties(monkeypatch):
    rows = pd.DataFrame(
        {
            'parcel_id_local': ['p1', 'p1', 'p2', None, 'p9', 'p4'],
            'year_built': [1990, 1965, 2001, 1950, 1880, None],
            'living_area_sqft': [1200, 800, 950, 300, 500, None],
            'n_bedrooms': [3, 2, 4, 1, 1, None],
        }
    )
    calls = []

    def fake_get_entities(recipe_id, admin_id, columns=None, missing='raise'):
        calls.append((recipe_id, str(admin_id), tuple(columns)))
        return rows[[c for c in columns if c in rows.columns]]

    monkeypatch.setattr(aggregation, 'get_entities', fake_get_entities)
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )
    return calls


def test_all_columns_carry_a_building_curate_onto_its_outlines(monkeypatch, tmp_path):
    """b3 step 2: the footprint curate reads the building curate through
    the footprint-to-building link, which link_entities_by_id names after
    the building geospine. One building per outline: its values and
    sources, cell for cell. Two in one outline (an envelope): sums add,
    a class they disagree on keeps both values, the earliest year wins by
    override, and an imputed marker on either survives in the sidecar.
    Fabricated ids and values throughout."""
    import openplaces.geo.link as link_module
    import openplaces.io.harmonizer.entity_links as entity_links

    buildings = pd.DataFrame(
        {
            'footprint_id': ['f1', 'f2', 'f2'],
            'occupancy_type': ['Single-Family', 'Single-Family', 'Multi-Family'],
            'occupancy_type_source': ['parcel', 'parcel', 'nsi'],
            'structure_value': [100.0, 30.0, 50.0],
            'structure_value_source': ['parcel', 'parcel', 'parcel+imputed'],
            'year_built': [1950.0, 1990.0, 1970.0],
            'lat': [1.0, 2.0, 2.0],
        },
        index=pd.Index(['b1', 'b2', 'b3'], name='building_id'),
    )
    links = pd.DataFrame(
        {
            'building_id': ['b1', 'b2', 'b3'],
            'footprint_id': ['f1', 'f2', 'f2'],
            'link_method': 'footprint_id',
            'link_source': 'footprint',
            'share': pd.Series([pd.NA] * 3, dtype='Float64'),
        }
    )
    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': buildings[
            [c for c in columns if c in buildings.columns]
        ],
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=buildings.columns),
    )
    monkeypatch.setattr(
        aggregation, '_linked_recipe_id', lambda recipe_id: 'US_building-spine-2026'
    )
    monkeypatch.setattr(
        link_module,
        'get_link_owner_recipe_id',
        lambda recipe: (
            'US_building-geospine-2026'
            if recipe == 'US_building-spine-2026'
            else 'US_footprint-geospine-2026'
        ),
    )
    monkeypatch.setattr(
        link_module,
        'get_entity_link_path',
        lambda a, b, admin_id=None: tmp_path / f'{a}__{b}.parquet',
    )
    # Only the geospine-named table exists, as link_entities_by_id writes it.
    (tmp_path / 'US_building-geospine-2026__US_footprint-geospine-2026.parquet').touch()
    monkeypatch.setattr(
        entity_links,
        'read_entity_link',
        lambda path: links if path.name.startswith('US_building-geospine') else None,
    )

    curated = pd.DataFrame(
        {'lat': [1.5, 2.5]}, index=pd.Index(['f1', 'f2'], name='footprint_id')
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'US_building-openplaces-2026',
        columns='all',
        key=None,
        aggregations={'year_built': 'min'},
        count_as='n_buildings_per_footprint',
    )
    out = state.curated
    assert out['occupancy_type'].tolist()[0] == 'Single-Family'
    # A class the buildings disagree on keeps both, in row order.
    assert out['occupancy_type']['f2'] == 'Single-Family | Multi-Family'
    assert out['occupancy_type_source']['f2'] == 'parcel+nsi'
    assert out['occupancy_type_source'].tolist()[0] == 'parcel'
    assert out['structure_value'].tolist() == [100.0, 80.0]
    assert out['structure_value_source'].tolist() == ['parcel', 'parcel+imputed']
    assert out['year_built'].tolist() == [1950.0, 1970.0]
    assert out['n_buildings_per_footprint'].tolist() == [1.0, 2.0]
    # The outline's own value stays (fill_only), and the link key column
    # is not carried onto the table it names.
    assert out['lat'].tolist() == [1.5, 2.5]
    assert 'footprint_id' not in out.columns


def test_an_override_needs_columns_all(properties):
    curated = pd.DataFrame({'parcel_id_local': ['p1']})
    with pytest.raises(ValueError, match='columns: all'):
        aggregation.aggregate_from_entities(
            _state(curated), 'r', columns=['year_built'], aggregations={'x': 'min'}
        )


def test_each_column_is_reduced_with_its_own_rule(properties):
    curated = pd.DataFrame(
        {'parcel_id_local': ['p1', 'p2', 'p3'], 'year_built': [None, None, 1977]}
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'US_property-spine-2026',
        columns={'year_built': 'min', 'living_area_sqft': 'sum', 'n_bedrooms': 'sum'},
    )
    out = state.curated
    assert out['year_built'].tolist() == [1965, 2001, 1977]
    assert out['living_area_sqft'].tolist()[:2] == [2000, 950]
    assert pd.isna(out['living_area_sqft'][2])
    assert out['n_bedrooms'].tolist()[:2] == [5, 4]
    assert properties[0][0] == 'US_property-spine-2026'
    assert properties[0][1] == 'US-XX-YY'


def test_a_parcel_whose_properties_state_nothing_gets_no_zero(properties):
    curated = pd.DataFrame({'parcel_id_local': ['p4']})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'living_area_sqft': 'sum', 'year_built': 'min'}
    )
    assert state.curated[['living_area_sqft', 'year_built']].isna().all().all()


def test_a_value_the_parcel_layer_states_is_kept(properties):
    curated = pd.DataFrame({'parcel_id_local': ['p1'], 'year_built': [1999]})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'year_built': 'min'}
    )
    assert state.curated['year_built'].tolist() == [1999]


def test_fill_only_false_replaces_the_stated_value(properties):
    curated = pd.DataFrame({'parcel_id_local': ['p1'], 'year_built': [1999]})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'year_built': 'min'}, fill_only=False
    )
    assert state.curated['year_built'].tolist() == [1965]


def test_a_list_uses_the_registry_rule(properties):
    curated = pd.DataFrame({'parcel_id_local': ['p1', 'p2']})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns=['n_bedrooms']
    )
    # The registry averages a bedroom count across the rows sharing a key.
    assert state.curated['n_bedrooms'].tolist() == [2.5, 4]


def test_a_missing_reference_is_skipped_with_a_warning(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError('no spine')

    monkeypatch.setattr(aggregation, 'describe_recipe', missing)
    curated = pd.DataFrame({'parcel_id_local': ['p1']})
    with pytest.warns(UserWarning, match='could not read'):
        state = aggregation.aggregate_from_entities(
            _state(curated), 'r', columns={'year_built': 'min'}
        )
    assert 'year_built' not in state.curated.columns


def test_the_first_key_present_on_both_sides_is_used(properties):
    curated = pd.DataFrame({'parcel_id_admin2': ['p1'], 'other_id': ['x']})
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'r',
        columns={'year_built': 'min'},
        key=['other_id', 'parcel_id_admin2'],
        reference_key=['missing_here', 'parcel_id_local'],
    )
    assert state.curated['year_built'].tolist() == [1965]


def test_a_link_table_replaces_the_key_join(monkeypatch):
    # Lot A: two roll rows keyed on its units, plus the split's own
    # unit rows describing the same units (ids that did not converge).
    # Lots C and D: one account on both, three quarters on C by area.
    rows = pd.DataFrame(
        {
            'parcel_id_local': ['A1', 'A2', 'A1', 'A2', 'CD'],
            'living_area_sqft': [800.0, 900.0, 810.0, 910.0, 4000.0],
            'year_built': [1990, 1985, 1991, 1986, 1970],
        },
        index=pd.Index(['r1', 'r2', 'u1', 'u2', 'r4'], name='property_id'),
    )
    links = pd.DataFrame(
        {
            'property_id': ['r1', 'r2', 'u1', 'u2', 'r4', 'r4'],
            'parcel_id': ['pa', 'pa', 'pa', 'pa', 'pc', 'pd'],
            'link_method': ['stacked_units_crosswalk'] * 2
            + ['stacked_units'] * 2
            + ['stacked_units_crosswalk'] * 2,
            'link_source': [
                'roll',
                'roll',
                'layer:units',
                'layer:units',
                'roll',
                'roll',
            ],
            'share': pd.array([None, None, None, None, 0.75, 0.25], dtype='Float64'),
            'share_basis': [None, None, None, None, 'area', 'area'],
        }
    )
    from pathlib import Path

    from openplaces.geo import link as geo_link
    from openplaces.io.harmonizer import entity_links

    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': rows[
            [c for c in columns if c in rows.columns]
        ],
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )
    monkeypatch.setattr(geo_link, 'get_link_owner_recipe_id', lambda recipe: 'owner')
    monkeypatch.setattr(
        geo_link, 'get_entity_link_path', lambda a, b, admin_id=None: Path('link')
    )
    monkeypatch.setattr(entity_links, 'read_entity_link', lambda path: links)

    curated = pd.DataFrame(
        {'parcel_id_local': ['A', 'C', 'D']},
        index=pd.Index(['pa', 'pc', 'pd'], name='parcel_id'),
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'US_property-spine-2026',
        columns={'living_area_sqft': 'sum', 'year_built': 'min'},
    )
    out = state.curated
    # The roll's two rows, not also the split's two: 1,700, not 3,420.
    assert out.loc['pa', 'living_area_sqft'] == 1700.0
    assert out.loc['pa', 'year_built'] == 1985
    # One account on two lots is divided, and its total is conserved.
    assert out.loc['pc', 'living_area_sqft'] == 3000.0
    assert out.loc['pd', 'living_area_sqft'] == 1000.0
    assert out.loc['pd', 'year_built'] == 1970


def test_a_curated_entity_on_the_finer_side_gathers_its_parcels(monkeypatch):
    """Curated sales take their parcels' values through the
    transaction-to-parcel table: a one-parcel sale its parcel's, a deed
    of two parcels their sums scaled by the sale's area share, a use
    class only where the parcels agree. No key column is needed."""
    parcels = pd.DataFrame(
        {
            'area_ha': [1.0, 3.0, 0.5],
            'land_value': [100.0, 300.0, 50.0],
            'use_group': ['Residential', 'Commercial', 'Residential'],
            'land_value_source': ['roll', 'roll', 'layer'],
        },
        index=pd.Index(['pa', 'pb', 'pc'], name='parcel_id'),
    )
    links = pd.DataFrame(
        {
            'transaction_id': ['t1', 't1', 't2', 't3'],
            'parcel_id': ['pa', 'pb', 'pc', 'pa'],
            'link_method': ['parcel_id_local'] * 3 + ['parcel_id_local_shared_key'],
            'link_source': ['s'] * 4,
            'share': pd.array([0.25, 0.75, None, None], dtype='Float64'),
            'share_basis': ['area', 'area', None, None],
        }
    )
    from pathlib import Path

    from openplaces.geo import link as geo_link
    from openplaces.io.harmonizer import entity_links

    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': parcels[
            [c for c in columns if c in parcels.columns]
        ],
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=parcels.columns),
    )
    monkeypatch.setattr(geo_link, 'get_link_owner_recipe_id', lambda recipe: 'tx')
    monkeypatch.setattr(
        geo_link, 'get_entity_link_path', lambda a, b, admin_id=None: Path('link')
    )
    monkeypatch.setattr(entity_links, 'read_entity_link', lambda path: links)

    curated = pd.DataFrame(
        {'price': [1000.0, 200.0, 5.0]},
        index=pd.Index(['t1', 't2', 't3'], name='transaction_id'),
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'US_parcel-spine-2026',
        key=None,
        columns={'area_ha': 'sum', 'land_value': 'sum', 'use_group': 'agree'},
    )
    out = state.curated
    # Sums over the deed's parcels, scaled by share: 0.25 + 2.25, 25 + 225.
    assert out.loc['t1', 'area_ha'] == 2.5
    assert out.loc['t1', 'land_value'] == 250.0
    assert pd.isna(out.loc['t1', 'use_group'])
    assert out.loc['t2', 'area_ha'] == 0.5
    assert out.loc['t2', 'use_group'] == 'Residential'
    # A shared-key link is left out of the gather.
    assert pd.isna(out.loc['t3', 'land_value'])
    # Provenance follows the parcels' sidecars.
    assert out.loc['t1', 'land_value_source'] == 'roll'
    assert out.loc['t2', 'land_value_source'] == 'layer'


def test_link_only_with_no_link_table_skips_with_a_warning(monkeypatch, properties):
    from openplaces.io.harmonizer import entity_links

    monkeypatch.setattr(entity_links, 'read_entity_link', lambda path: None)
    curated = pd.DataFrame(
        {'price': [1.0]}, index=pd.Index(['t1'], name='transaction_id')
    )
    with pytest.warns(UserWarning, match='no link table'):
        state = aggregation.aggregate_from_entities(
            _state(curated), 'r', key=None, columns={'year_built': 'min'}
        )
    assert 'year_built' not in state.curated.columns


def test_no_shared_key_skips_with_a_warning(properties):
    curated = pd.DataFrame({'parcel_id_admin2': ['p1']})
    with pytest.warns(UserWarning, match='no key'):
        state = aggregation.aggregate_from_entities(
            _state(curated), 'r', columns={'year_built': 'min'}
        )
    assert 'year_built' not in state.curated.columns


@pytest.fixture
def valued_units(monkeypatch):
    # Lot L1: two units, one of them valued. Lot L2: two units, both
    # recorded as zero. Lot L3: two units stating nothing. Lot L4: one
    # valued unit on a lot that already states its own figure.
    rows = pd.DataFrame(
        {
            'parcel_id_local': ['L1', 'L1', 'L2', 'L2', 'L3', 'L3', 'L4'],
            'improvement_value': [150000.0, 0.0, 0.0, 0.0, None, None, 90000.0],
        }
    )
    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': rows[
            [c for c in columns if c in rows.columns]
        ],
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )


def test_recover_writes_a_recorded_positive_sum_onto_an_empty_or_zero_lot(
    valued_units,
):
    curated = pd.DataFrame(
        {
            'parcel_id_local': ['L1', 'L2', 'L3', 'L4', 'L5'],
            'improvement_value': [0.0, 0.0, None, 250000.0, 0.0],
        }
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'r',
        columns={'improvement_value': 'sum'},
        recover=['improvement_value'],
    )
    out = state.curated['improvement_value']
    # L1: a unit records the figure, so the lot's zero is replaced.
    assert out[0] == 150000.0
    # L2 and L3: no unit records a positive value; the lot stays exactly
    # as it was (a zero stays a zero, a blank stays blank). Writing here
    # would be the imputation the maintainer declined.
    assert out[1] == 0.0
    assert pd.isna(out[2])
    # L4: a lot that states its own positive figure keeps it.
    assert out[3] == 250000.0
    # L5: no property row at all.
    assert out[4] == 0.0


def test_recover_on_an_absent_column_writes_only_positive_sums(valued_units):
    curated = pd.DataFrame({'parcel_id_local': ['L1', 'L2', 'L3']})
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'r',
        columns={'improvement_value': 'sum'},
        recover=['improvement_value'],
    )
    out = state.curated['improvement_value']
    assert out[0] == 150000.0
    assert pd.isna(out[1]) and pd.isna(out[2])


def test_fill_only_alone_would_have_left_the_zero(valued_units):
    # The contrast the recover rule exists for: fill_only fills blanks
    # only, so a present zero on L1 survives it.
    curated = pd.DataFrame({'parcel_id_local': ['L1'], 'improvement_value': [0.0]})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'improvement_value': 'sum'}
    )
    assert state.curated['improvement_value'].tolist() == [0.0]


@pytest.fixture
def classed_properties(monkeypatch):
    rows = pd.DataFrame(
        {
            'parcel_id_local': ['p1', 'p1', 'p2', 'p2', 'p3', 'p3', 'p4'],
            'foundation_type': ['S', 'S', 'S', 'C', 'S', None, None],
            'foundation_type_source': [
                'srca+keywords',
                'srcb+override',
                'srca+keywords',
                'srca+keywords',
                'srca+override',
                None,
                None,
            ],
        }
    )
    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': rows[
            [c for c in columns if c in rows.columns]
        ],
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )


def test_agree_writes_a_class_only_where_the_properties_agree(classed_properties):
    curated = pd.DataFrame({'parcel_id_local': ['p1', 'p2', 'p3', 'p4', 'p5']})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'foundation_type': 'agree'}
    )
    out = state.curated['foundation_type']
    assert out.iloc[0] == 'S'
    # A slab house and a crawl-space house: no single foundation.
    assert pd.isna(out.iloc[1])
    # A property that states nothing does not disagree.
    assert out.iloc[2] == 'S'
    assert out.iloc[3:].isna().all()


@pytest.fixture
def condo_lots(monkeypatch):
    # Fabricated accounts. Lot C1: a master record repeating the
    # complex's total (3) and three units stating 1 each. Lot C2: a
    # master stating 1 and four units stating 0 or nothing, plus a
    # parking unit (108) and a commercial unit (357). Lot C3: a
    # two-family house, no condominium. Lot C4: a 4-digit unit code.
    rows = pd.DataFrame(
        {
            'parcel_id_local': ['C1'] * 4 + ['C2'] * 7 + ['C3', 'C4', 'C4'],
            'use_group_code': ['995', '102', '102', '102']
            + ['995', '102', '102', '102', '102', '108', '357']
            + ['104', '1020', '1020'],
            'n_dwellings': [3.0, 1.0, 1.0, 1.0]
            + [1.0, 0.0, None, 0.0, None, 0.0, 0.0]
            + [2.0, 2.0, None],
        }
    )
    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': rows[
            [c for c in columns if c in rows.columns]
        ],
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )


def _count_condo_units(curated):
    return aggregation.aggregate_from_entities(
        _state(curated),
        'r',
        columns={'n_dwellings': 'sum'},
        where={'column': 'use_group_code', 'pattern': '102.*'},
        row_minimum={'n_dwellings': 1},
        prefer=['n_dwellings'],
    ).curated


def test_a_condominium_lot_counts_its_units_ahead_of_its_master(condo_lots):
    curated = pd.DataFrame(
        {
            'parcel_id_local': ['C1', 'C2', 'C3', 'C4'],
            # What the spine brought: master and units added (6), the
            # master's 1, the house's 2, nothing.
            'n_dwellings': [6.0, 1.0, 2.0, None],
        }
    )
    out = _count_condo_units(curated)['n_dwellings']
    # Three units, not the master's 3 added to them.
    assert out[0] == 3.0
    # Four residential units, each one dwelling though it states none;
    # the parking and commercial units are not dwellings.
    assert out[1] == 4.0
    # No condominium unit on the lot: the lot keeps what it had.
    assert out[2] == 2.0
    # A unit stating 2 keeps its 2; a 4-digit code is a 102 unit.
    assert out[3] == 3.0


def test_where_and_row_minimum_leave_the_plain_sum_unchanged(condo_lots):
    curated = pd.DataFrame({'parcel_id_local': ['C1', 'C2']})
    out = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'n_dwellings': 'sum'}
    ).curated['n_dwellings']
    # Without the options every account is summed as stated.
    assert out.tolist() == [6.0, 1.0]


def test_where_rejects_unknown_keys(condo_lots):
    curated = pd.DataFrame({'parcel_id_local': ['C1']})
    with pytest.raises(ValueError, match='where takes'):
        aggregation.aggregate_from_entities(
            _state(curated),
            'r',
            columns={'n_dwellings': 'sum'},
            where={'column': 'use_group_code', 'values': ['102']},
        )


def test_the_massachusetts_patch_counts_condominium_units():
    from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

    recipe, _ = apply_recipe_patches(
        get_recipe_by_id('US_parcel-openplaces-2026'), AdminId('US', 'MA', 'SOM')
    )
    steps = [s for s in recipe['pipeline'] if s['step'] == 'aggregate_from_entities']
    # Right after the national property reduction, which fills first.
    assert steps[0].get('where') is None
    condo = steps[1]
    assert condo['where'] == {'column': 'use_group_code', 'pattern': '102.*'}
    assert condo['row_minimum'] == {'n_dwellings': 1}
    assert condo['prefer'] == ['n_dwellings']
    assert condo['columns'] == {'n_dwellings': 'sum'}


def test_the_sidecar_names_the_rows_the_value_came_from(classed_properties):
    curated = pd.DataFrame({'parcel_id_local': ['p1', 'p2', 'p3']})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'foundation_type': 'agree'}
    )
    sources = state.curated['foundation_type_source']
    assert sources.iloc[0] == 'srca+keywords+srcb+override'
    # No value written, no provenance.
    assert pd.isna(sources.iloc[1])
    assert sources.iloc[2] == 'srca+override'
