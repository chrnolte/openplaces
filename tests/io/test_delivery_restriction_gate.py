"""A delivery withholds what restricted sources supplied and ships the rest.

A source recording `redistribution_restricted: true` or a
`usage_requirement` may feed a curated entity. Its values are withheld
from a delivery (rows drawn from its geometry left out, cells it filled
emptied), and the gate behind that fires only if such values are still
present. A source whose data never reaches the delivered columns changes
nothing, and no single source holds the rest of a bundle hostage. Sources
here are fabricated, so the contract does not move as recipes change.
"""

from __future__ import annotations

import pandas as pd
import pytest

import openplaces.io.bundle_terms as bundle_terms_module
import openplaces.io.delivery as delivery_module
from openplaces.core.schema import UsageRequirement
from openplaces.io.bundle_terms import restricted_inputs
from openplaces.io.delivery import RestrictedInputError, export_delivery
from openplaces.io.redaction import WITHHELD, find_restricted, withhold
from tests.io.test_delivery import _county, _recipe


def _entry(source_id, **overrides):
    return {
        'source_id': source_id,
        'recipe_id': f'US-XX-YY_property-{source_id}-2026',
        'license': 'fabricated terms',
        'terms_url': None,
        'portal_url': None,
        'redistribution_restricted': None,
        'resale_restricted': None,
        'usage_requirement': None,
        **overrides,
    }


def _source(source_id, admin_id, attributes=(), entity_type='parcel'):
    """A restricted input as `restricted_inputs` describes one."""
    return {
        'source_id': source_id,
        'recipe_id': f'{admin_id}_{entity_type}-{source_id}-2026',
        'license': 'fabricated: no redistribution',
        'terms_url': 'https://example.org/terms',
        'reasons': ['redistribution restricted'],
        'entity_type': entity_type,
        'admin_id': admin_id,
        'attributes': list(attributes),
    }


@pytest.fixture
def upstream(monkeypatch):
    """Replace the dependency walk with a chosen set of source entries."""

    def use(*entries):
        terms = {bundle_terms_module._terms_key(e): e for e in entries}
        monkeypatch.setattr(
            bundle_terms_module, '_source_terms', lambda *_args, **_kw: terms
        )

    return use


@pytest.fixture
def restricted(monkeypatch):
    """Make `export_delivery` see a chosen list of restricted inputs."""

    def use(*sources):
        monkeypatch.setattr(
            delivery_module, 'restricted_inputs', lambda *_args, **_kw: list(sources)
        )

    return use


# Which sources are candidates (the terms walk).


def test_an_open_recipe_has_no_restricted_inputs(upstream):
    upstream(_entry('opencounty', redistribution_restricted=False))
    assert restricted_inputs(_recipe()) == []


def test_a_redistribution_restriction_is_found(upstream):
    upstream(
        _entry('opencounty'),
        _entry('closedcounty', redistribution_restricted=True),
    )
    blocked = restricted_inputs(_recipe())
    assert [e['source_id'] for e in blocked] == ['closedcounty']
    assert blocked[0]['reasons'] == ['redistribution restricted']


def test_a_usage_requirement_is_found(upstream):
    upstream(
        _entry('nccounty', usage_requirement=UsageRequirement(non_commercial=True))
    )
    blocked = restricted_inputs(_recipe())
    assert blocked[0]['reasons'] == ['non-commercial use only']


def test_a_resale_clause_alone_is_not_a_restriction(upstream):
    upstream(_entry('resalecounty', resale_restricted=True))
    assert restricted_inputs(_recipe()) == []


def test_auto_discovered_county_sources_are_checked():
    """County parcel recipes reach a spine only through auto-discovery.

    They resolve per admin unit, so a walk without the delivered units
    never saw them: Edgecombe County's parcels feed the NC footprint
    bundle and read as absent. Checked on the real recipes because the
    defect was in the walk, not in any one source.
    """
    recipe = 'US_footprint-openplaces-2026'
    unscoped = {e['source_id'] for e in restricted_inputs(recipe)}
    scoped = restricted_inputs(recipe, ['US-NC-EDG'])
    edgecombe = [e for e in scoped if e['source_id'] == 'edgecombecounty']

    assert 'edgecombecounty' not in unscoped
    assert edgecombe
    # Where its data can land, for redaction.
    assert edgecombe[0]['entity_type'] == 'parcel'
    assert edgecombe[0]['admin_id'] == 'US-NC-EDG'
    assert 'improvement_value' in edgecombe[0]['attributes']


# Which values are the source's (the redaction rules).


def _frame():
    """Two rows in the restricted county, one outside it."""
    return pd.DataFrame(
        {
            'admin3_id': ['US-NC-AL', 'US-NC-AL', 'US-NC-BB'],
            'geometry_source': ['obm', 'parcel.closedcounty', 'obm'],
            'address': ['1 Fabricated Rd', '2 Fabricated Rd', '3 Fabricated Rd'],
            'address_source': ['parcel', 'dwelling_overture', 'parcel'],
            'address_number': ['1', '2', '3'],
            'address_street_dwelling_overture': ['Rd', 'Rd', 'Rd'],
            'land_value_parcel': [10.0, 20.0, 30.0],
            'structure_value': [100.0, 200.0, 300.0],
            'structure_value_source': ['parcel', 'nsi', 'parcel.closedcounty'],
            'year_built': [1990, 1991, 1992],
        },
        index=pd.Index(['a', 'b', 'c'], name='footprint_id'),
    )


CLOSED = _source('closedcounty', 'US-NC-AL', ['address', 'land_value'])


def test_a_row_drawn_from_the_source_is_left_out():
    found = find_restricted(_frame(), [CLOSED], 'admin3_id')
    assert found['rows'].tolist() == [False, True, False]


def test_a_sidecar_naming_the_source_withholds_the_value_anywhere():
    """Row c is outside the county; its token still names the source."""
    found = find_restricted(_frame(), [CLOSED], 'admin3_id')
    assert found['cells']['structure_value'].tolist() == [True, False, True]


def test_in_the_county_its_attributes_are_withheld_unless_another_source_says():
    cells = find_restricted(_frame(), [CLOSED], 'admin3_id')['cells']

    # 'parcel' names only the layer, which is the source's in its county.
    assert cells['address'].tolist() == [True, False, False]
    # A parsed component follows its base column's sidecar.
    assert cells['address_number'].tolist() == [True, False, False]
    # The source's entity suffix marks its layer's copy.
    assert cells['land_value_parcel'].tolist() == [True, True, False]
    # A column named for another entity's source is not the source's.
    assert 'address_street_dwelling_overture' not in cells
    # Nothing ties year_built to the source.
    assert 'year_built' not in cells


def test_withholding_empties_values_and_says_why():
    redacted, counts = withhold(_frame(), [CLOSED], 'admin3_id')

    assert list(redacted.index) == ['a', 'c']
    assert pd.isna(redacted.loc['a', 'address'])
    assert redacted.loc['a', 'address_source'] == WITHHELD
    assert redacted.loc['c', 'address'] == '3 Fabricated Rd'
    assert pd.isna(redacted.loc['c', 'structure_value'])
    assert counts['closedcounty']['rows'] == 1
    # And nothing is left for the gate to find.
    again = find_restricted(redacted, [CLOSED], 'admin3_id')
    assert again['counts'] == {}


def test_a_source_whose_columns_are_not_delivered_changes_nothing():
    """Room counts from a restricted roll, in a bundle without them."""
    rooms = _source('roomcounty', 'US-NC-AL', ['n_bedrooms'], 'property')
    redacted, counts = withhold(_frame(), [rooms], 'admin3_id')

    assert counts == {}
    pd.testing.assert_frame_equal(redacted, _frame())


# The delivery.


def _read(paths, role):
    return pd.read_parquet(paths[role])


def test_export_withholds_restricted_values_and_ships_the_rest(
    mock_data_root, restricted
):
    _county(['a', 'b'], admin3_id='US-NC-AL')
    _county(['c', 'd'], admin3_id='US-NC-BB')
    restricted(_source('closedcounty', 'US-NC-AL', ['year_built']))

    paths = export_delivery(_recipe(), 'US-NC', admin_ids=['US-NC-AL', 'US-NC-BB'])
    canonical = _read(paths, 'canonical')

    assert canonical.loc[['a', 'b'], 'year_built'].isna().all()
    assert canonical.loc[['c', 'd'], 'year_built'].notna().all()
    # occupancy_type_source 'parcel' names the source's layer there.
    assert (canonical.loc[['a', 'b'], 'occupancy_type_source'] == WITHHELD).all()
    assert (canonical.loc[['c', 'd'], 'occupancy_type_source'] == 'parcel').all()
    # The evidence file repeats the sidecar, redacted the same way.
    evidence = _read(paths, 'evidence')
    assert (evidence.loc[['a', 'b'], 'occupancy_type_source'] == WITHHELD).all()

    notice = paths['terms'].read_text(encoding='utf-8')
    assert 'Withheld' in notice
    assert 'closedcounty' in notice.split('Withheld')[1]


def test_export_leaves_out_rows_drawn_from_the_source(mock_data_root, restricted):
    _county(
        ['a', 'b'],
        admin3_id='US-NC-AL',
        extra={'geometry_source': ['parcel.closedcounty', 'obm']},
    )
    restricted(_source('closedcounty', 'US-NC-XX'))

    paths = export_delivery(_recipe(), 'US-NC', admin_ids=['US-NC-AL'])

    for role in ('canonical', 'point', 'geo', 'evidence'):
        assert list(_read(paths, role).index) == ['b'], role


def test_export_is_unchanged_by_a_source_it_does_not_carry(mock_data_root, restricted):
    _county(['a', 'b'], admin3_id='US-NC-AL')
    restricted(_source('roomcounty', 'US-NC-AL', ['n_bedrooms'], 'property'))

    paths = export_delivery(_recipe(), 'US-NC', admin_ids=['US-NC-AL'])

    assert _read(paths, 'canonical')['year_built'].notna().all()
    assert 'Withheld' not in paths['terms'].read_text(encoding='utf-8')


def test_the_gate_refuses_values_redaction_missed(
    mock_data_root, restricted, monkeypatch
):
    """With redaction switched off, restricted values reach the gate."""
    _county(['a', 'b'], admin3_id='US-NC-AL')
    restricted(_source('closedcounty', 'US-NC-AL', ['year_built']))
    monkeypatch.setattr(
        delivery_module, 'withhold', lambda frame, sources, column: (frame, {})
    )

    with pytest.raises(RestrictedInputError, match='closedcounty'):
        export_delivery(_recipe(), 'US-NC', admin_ids=['US-NC-AL'])

    for path in delivery_module.delivery_paths(_recipe(), 'US-NC').values():
        assert not path.exists()


# Team bundles: values cleared for team sharing, never published.


def test_team_regions_are_declared_beside_their_region():
    specs = delivery_module.delivery_regions('US_footprint-openplaces-2026')
    team = [spec for spec in specs if spec['audience'] == 'team']
    base = next(spec for spec in specs if spec['region_id'] == 'cheer-eastern-nc')

    assert [spec['region_id'] for spec in team] == ['cheer-eastern-nc.team']
    assert team[0]['admin_ids'] == base['admin_ids']
    assert base['audience'] == 'public'


def test_a_team_bundle_lands_beside_the_public_one_without_moving_it():
    recipe = 'US_footprint-openplaces-2026'
    public = delivery_module.delivery_paths(recipe, region='cheer-eastern-nc')
    team = delivery_module.delivery_paths(recipe, region='cheer-eastern-nc.team')

    for role, path in public.items():
        assert team[role] == path.parent / 'team' / path.name
    # The public bundle keeps the location it had before team bundles.
    import copy

    from openplaces.recipe import get_recipe_by_id

    without = copy.deepcopy(get_recipe_by_id(recipe))
    without['share']['delivery'].pop('team_regions')
    assert delivery_module.delivery_paths(without, region='cheer-eastern-nc') == public


def test_an_undeclared_team_region_is_refused():
    recipe = _recipe()
    recipe['share']['delivery'] = {
        'admin_level': 2,
        'regions': ['northern-wisconsin'],
        'team_regions': ['nowhere'],
    }
    with pytest.raises(KeyError, match='nowhere'):
        delivery_module.delivery_regions(recipe)


@pytest.fixture
def two_audiences(monkeypatch):
    """One region shipped as a public and as a team bundle."""
    public = {
        'region_id': 'test-region',
        'admin_level': 2,
        'admin_ids': ['US-NC-AL', 'US-NC-BB'],
        'admin_id': None,
        'audience': 'public',
    }
    team = {
        **public,
        'region_id': 'test-region.team',
        'audience': 'team',
        'base_region_id': 'test-region',
    }
    monkeypatch.setattr(delivery_module, 'delivery_regions', lambda _: [public, team])


def test_only_the_team_bundle_keeps_values_cleared_for_the_team(
    mock_data_root, restricted, two_audiences
):
    _county(['a', 'b'], admin3_id='US-NC-AL')
    _county(['c', 'd'], admin3_id='US-NC-BB')
    cleared = {
        **_source('teamcounty', 'US-NC-AL', ['year_built']),
        'team_sharing_permitted': True,
    }
    restricted(cleared)

    public = export_delivery(_recipe(), region='test-region')
    team = export_delivery(_recipe(), region='test-region.team')

    assert _read(public, 'canonical').loc[['a', 'b'], 'year_built'].isna().all()
    assert _read(team, 'canonical').loc[['a', 'b'], 'year_built'].notna().all()
    assert team['canonical'].parent.name == 'team'
    team_notice = team['terms'].read_text(encoding='utf-8')
    assert 'TEAM-INTERNAL' in team_notice
    assert 'teamcounty' in team_notice.split('TEAM-INTERNAL')[1]
    assert 'TEAM-INTERNAL' not in public['terms'].read_text(encoding='utf-8')


def test_a_team_bundle_still_withholds_sources_not_cleared_for_it(
    mock_data_root, restricted, two_audiences
):
    _county(['a', 'b'], admin3_id='US-NC-AL')
    _county(['c', 'd'], admin3_id='US-NC-BB')
    restricted(_source('closedcounty', 'US-NC-AL', ['year_built']))

    team = export_delivery(_recipe(), region='test-region.team')

    assert _read(team, 'canonical').loc[['a', 'b'], 'year_built'].isna().all()


# A source that is not a layer.


def test_a_column_source_is_withheld_by_its_columns_only():
    """A permit table linked for one evidence column touches nothing else.

    Its recipe maps year_built and county_fips like any roll, and its scope
    is national, but `restricted_inputs` lists only the columns the link
    let through. Read as a layer, the layer rules emptied year_built and
    county_fips across every county of a bundle; read as what it is,
    only its own column goes, and 'parcel' tokens say nothing about it.
    """
    permits = {
        **_source(
            'permitvendor',
            'US',
            ['occupancy_type_property_permitvendor', 'n_permits_per_footprint'],
            'property',
        ),
        'layer_source': False,
    }
    frame = _frame()
    frame['occupancy_type_property_permitvendor'] = ['Single', None, 'Multi']
    frame['county_fips'] = ['37001', '37003', '37005']

    cells = find_restricted(frame, [permits], 'admin3_id')['cells']

    assert cells['occupancy_type_property_permitvendor'].tolist() == [
        True,
        False,
        True,
    ]
    assert 'year_built' not in cells
    assert 'county_fips' not in cells
    assert 'address' not in cells


def test_a_layer_source_keeps_the_layer_rules():
    """The Edgecombe case is unchanged by the column-source distinction."""
    layer = {**CLOSED, 'layer_source': True}
    cells = find_restricted(_frame(), [layer], 'admin3_id')['cells']
    assert cells['address'].tolist() == [True, False, False]
