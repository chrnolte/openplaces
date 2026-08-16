"""Regressions for the CHEER footprint occupancy vote, on the shipped recipe.

Loads `US_footprint-cheer-2026`'s own `derive_indicators` and both
`resolve_by_vote` blocks rather than hand-copied fixtures, so a recipe edit
that reintroduces one of these behaviors fails here.

Occupancy is decided in two steps -- dwelling multiplicity first, then
construction type among single-dwelling footprints -- so these tests run both
votes in order, exactly as the pipeline does.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_indicators
from openplaces.io.curator.reconcilers import resolve_by_vote
from openplaces.recipe import get_recipe_by_id

RECIPE_ID = 'US_footprint-cheer-2026'

# Every column the votes may read. A column absent from the frame makes
# evaluate_indicator return all-False, so an indicator referencing it would
# silently never fire -- see test_every_vote_input_is_available.
DEFAULTS = {
    'occupancy_type': None,
    'use_group_combined_parcel': None,
    'group_parcel': None,
    'group_footprint_fema': None,
    'occupancy_type_building_nsi': None,
    'land_use_class_parcel': None,
    'land_use_class_source_parcel': None,
    'manufactured_home_community': False,
    'priority_on_parcel': 'primary',
    'p_manufactured_home': 0.0,
    # 0 is the zero-filled "Overture saw nothing here" value, so this
    # default really is no evidence; cases that want a count set one.
    'n_dwellings_overture': 0,
    'improvement_value_parcel': 150_000.0,
    'land_value_parcel': 50_000.0,
    'n_stories': None,
    # A plainly non-manufactured shape unless a case overrides it.
    'length_m': 20.0,
    'width_m': 15.0,
}


@pytest.fixture(scope='module')
def recipe():
    return get_recipe_by_id(RECIPE_ID)


def _votes(recipe, target):
    return [
        s
        for s in recipe['pipeline']
        if s['step'] == 'resolve_by_vote' and s['target'] == target
    ]


def _box(lon, lat, length_m, width_m):
    """A rectangle of the given metric dimensions, in lon/lat."""
    dx = length_m / (111_320.0 * 0.82)
    dy = width_m / 110_574.0
    return Polygon([(lon, lat), (lon + dx, lat), (lon + dx, lat + dy), (lon, lat + dy)])


def _run(recipe, rows: list[dict]) -> pd.DataFrame:
    """Run derive_indicators plus both occupancy votes, in pipeline order."""
    merged = [{**DEFAULTS, **row} for row in rows]
    geometry = [
        _box(-78.0 + 0.01 * i, 35.0, r.pop('length_m'), r.pop('width_m'))
        for i, r in enumerate(merged)
    ]
    frame = gpd.GeoDataFrame(merged, geometry=geometry, crs='EPSG:4326')
    frame['area_m2'] = frame.to_crs(frame.estimate_utm_crs()).area

    state = CurateState(
        recipe=recipe,
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=frame,
    )
    derive = next(s for s in recipe['pipeline'] if s['step'] == 'derive_indicators')
    state = derive_indicators(state, derive['indicators'])

    for target in ('dwelling_multiplicity', 'occupancy_type'):
        for vote in _votes(recipe, target):
            state = resolve_by_vote(
                state,
                target=vote['target'],
                decisions=vote['decisions'],
                preserve_base=vote.get('preserve_base', True),
                base_output=vote.get('base_output'),
                default_source=vote.get('default_source', 'vote'),
            )
    return state.curated


HEIGHT_BANDS = [
    (2, 'Low-Rise Multi-Family'),
    (3, 'Low-Rise Multi-Family'),
    (5, 'Mid-Rise Multi-Family'),
    (7, 'Mid-Rise Multi-Family'),
    (20, 'High-Rise Multi-Family'),
    (None, 'Multi-Family'),
]


def _classify(recipe, rows: list[dict]) -> pd.Series:
    return _run(recipe, rows)['occupancy_type'].astype(object)


def _multiplicity(recipe, rows: list[dict]) -> pd.Series:
    return _run(recipe, rows)['dwelling_multiplicity'].astype(object)


def _referenced_columns(indicators, out):
    """Collect every column an indicator tree reads."""
    for ind in indicators:
        if ind.get('type') in ('any_of', 'all_of'):
            _referenced_columns(ind.get('indicators', []), out)
            continue
        if ind.get('column'):
            out.add(ind['column'])
        if ind.get('type') in ('value_share_below', 'value_share_at_least'):
            out.add(ind['value'])
            out.update(ind.get('total', []))


def test_every_vote_input_is_available(recipe):
    """No indicator may reference a column nothing provides.

    `evaluate_indicator` returns all-False for an absent column, so a typo or
    a forgotten fixture entry makes a gate silently never fire while every
    behavioral test still passes. This asserts the fixture plus the recipe's
    own derived outputs cover every column both votes read.
    """
    referenced: set[str] = set()
    for target in ('dwelling_multiplicity', 'occupancy_type'):
        for vote in _votes(recipe, target):
            for decision in vote['decisions']:
                _referenced_columns(decision.get('indicators', []), referenced)
                _referenced_columns(decision.get('require', []), referenced)

    derive = next(s for s in recipe['pipeline'] if s['step'] == 'derive_indicators')
    # A vote also provides its own target and any base_output snapshot, which
    # later decisions in the same or a following vote may require.
    written = {'area_m2', 'geometry'}
    for target in ('dwelling_multiplicity', 'occupancy_type'):
        for vote in _votes(recipe, target):
            written.add(vote['target'])
            if vote.get('base_output'):
                written.add(vote['base_output'])
    provided = (
        set(DEFAULTS) | {spec['output'] for spec in derive['indicators']} | written
    )
    assert not referenced - provided, (
        f'vote reads columns nothing provides: {sorted(referenced - provided)}'
    )


def test_two_dwellings_resolve_to_multi_then_multi_family(recipe):
    # Question 1 is answered by the dwelling count alone -- it is 92% precise
    # for multi-dwelling -- and question 2 simply carries that through.
    out = _run(recipe, [{'n_dwellings_overture': 2}])
    assert out['dwelling_multiplicity'].astype(object).iloc[0] == 'multi'
    assert out['occupancy_type'].astype(object).iloc[0] == 'Multi-Family'


def test_one_dwelling_alone_resolves_to_single_family(recipe):
    # Single-family is the residual of the single-dwelling group: one
    # dwelling and no manufactured-home evidence is a site-built house.
    out = _run(recipe, [{'n_dwellings_overture': 1}])
    assert out['dwelling_multiplicity'].astype(object).iloc[0] == 'single'
    assert out['occupancy_type'].astype(object).iloc[0] == 'Single-Family'


def test_manufactured_evidence_claims_a_single_dwelling_building(recipe):
    # The separation the split buys: the same dwelling count supports both
    # classes at question 1, and question 2 is decided by shape, which is 90%
    # precise once restricted to single-dwelling footprints.
    result = _classify(
        recipe, [{'n_dwellings_overture': 1, 'length_m': 18.0, 'width_m': 5.0}]
    )
    assert result.iloc[0] == 'Manufactured Home'


def test_manufactured_home_signals_also_prove_single_dwelling(recipe):
    # A manufactured home is one dwelling, so NSI/FEMA saying Manufactured
    # Home is 100% precise evidence for question 1. The flat vote threw this
    # away by treating the two classes as competitors at this stage.
    out = _run(
        recipe,
        [
            {
                'occupancy_type_building_nsi': 'Manufactured Home',
                'group_footprint_fema': 'Manufactured Home',
                'length_m': 18.0,
                'width_m': 5.0,
            }
        ],
    )
    assert out['dwelling_multiplicity'].astype(object).iloc[0] == 'single'
    assert out['occupancy_type'].astype(object).iloc[0] == 'Manufactured Home'


def test_elongation_alone_does_not_score_without_a_small_area(recipe):
    # A 60m x 12m warehouse is elongated but far too large; both halves of
    # the shape signal must hold together to count at all.
    result = _classify(
        recipe, [{'n_dwellings_overture': 1, 'length_m': 60.0, 'width_m': 12.0}]
    )
    assert result.iloc[0] != 'Manufactured Home'


def test_minimum_area_floor_applies_to_every_route_to_the_class(recipe):
    # Overwhelming manufactured-home evidence on a 10 m2 footprint. The
    # `require` floor vetoes the decision outright.
    result = _classify(
        recipe,
        [
            {
                'n_dwellings_overture': 1,
                'use_group_combined_parcel': 'MANUFACTURED HOME',
                'group_footprint_fema': 'Manufactured Home',
                'occupancy_type_building_nsi': 'Manufactured Home',
                'p_manufactured_home': 0.99,
                'length_m': 5.0,
                'width_m': 2.0,
            }
        ],
    )
    assert result.iloc[0] != 'Manufactured Home'


def test_accessory_structures_resolve_to_secondary(recipe):
    # Without the priority gate the single-family residual would relabel
    # every shed on a single-dwelling parcel as a house.
    result = _classify(
        recipe,
        [{'n_dwellings_overture': 1, 'priority_on_parcel': 'secondary'}],
    )
    assert result.iloc[0] == 'Secondary'


def test_habitable_home_in_a_community_outscores_the_secondary_residual(recipe):
    # A manufactured-home community's own dwellings are frequently marked
    # non-primary on their parcel. They are homes, not accessory structures,
    # so the paired community/size indicator has to beat Secondary -- this is
    # the carve-out that used to be a special case inside impute_occupancy_type.
    result = _classify(
        recipe,
        [
            {
                'n_dwellings_overture': 1,
                'priority_on_parcel': 'secondary',
                'manufactured_home_community': True,
                'occupancy_type_building_nsi': 'Manufactured Home',
                'length_m': 18.0,
                'width_m': 5.0,
            }
        ],
    )
    assert result.iloc[0] == 'Manufactured Home'


def test_a_shed_in_a_community_stays_secondary(recipe):
    # The mirror, and the reason the size half of the pair exists: a small
    # structure on the same community parcel is still a shed.
    result = _classify(
        recipe,
        [
            {
                'n_dwellings_overture': 1,
                'priority_on_parcel': 'secondary',
                'manufactured_home_community': True,
                'length_m': 4.0,
                'width_m': 3.0,
            }
        ],
    )
    assert result.iloc[0] == 'Secondary'


def test_park_signal_counts_only_on_the_rule_route(recipe):
    # land_use_class_parcel == 'Manufactured Home Park' is trustworthy only
    # where the parcel lane's own vote assigned it (that route carries a hard
    # footprint-count require). A reconcile_land_use fill is an NSI/FEMA
    # re-vote and must not score.
    rule = _classify(
        recipe,
        [
            {
                'n_dwellings_overture': 1,
                'land_use_class_parcel': 'Manufactured Home Park',
                'land_use_class_source_parcel': 'rule',
                'p_manufactured_home': 0.8,
            }
        ],
    )
    filled = _classify(
        recipe,
        [
            {
                'n_dwellings_overture': 1,
                'land_use_class_parcel': 'Manufactured Home Park',
                'land_use_class_source_parcel': 'nsi/fema',
                'p_manufactured_home': 0.8,
            }
        ],
    )
    assert rule.iloc[0] == 'Manufactured Home'
    assert filled.iloc[0] != 'Manufactured Home'


def test_zero_evidence_footprint_falls_through_unclassified(recipe):
    # No dwelling count, no source agreeing on anything: question 1 has no
    # answer, so both gated decisions are vetoed and the incoming value
    # stands.
    result = _classify(recipe, [{'occupancy_type': None}])
    assert pd.isna(result.iloc[0])


@pytest.mark.parametrize(('n_stories', 'expected'), HEIGHT_BANDS)
def test_height_bands_reproduce_the_retired_cascade(recipe, n_stories, expected):
    """The band vote must match refine_occupancy_height's cascade exactly.

    Three overlapping upper bounds plus earliest-listed-wins is a cascade:
    two stories satisfies both the 3 and the 7 bound and takes Low-Rise. A
    missing story count satisfies none, so no decision is eligible and the
    plain Multi-Family class survives.
    """
    frame = _run(recipe, [{'n_dwellings_overture': 4, 'n_stories': n_stories}])
    assert frame['occupancy_type'].astype(object).iloc[0] == expected
    # The pre-refinement class is preserved for provenance.
    assert frame['occupancy_type_base'].astype(object).iloc[0] == 'Multi-Family'


def test_condominium_parcel_class_is_never_read(recipe):
    """Condominium describes tenure, not structure.

    Measured on ground truth it splits 35/36/29 across the three classes,
    because near-zero land value covers condo buildings, detached condos and
    manufactured homes on leased pads alike. It must not appear in any
    indicator -- and since the predicate vocabulary has no negation,
    exclusion is only expressible by omission, which is easy to undo by
    accident.
    """
    values: list = []
    for target in ('dwelling_multiplicity', 'occupancy_type'):
        for vote in _votes(recipe, target):
            for decision in vote['decisions']:
                for group in (
                    decision.get('indicators', []),
                    decision.get('require', []),
                ):
                    for ind in group:
                        nested = ind.get('indicators', []) or [ind]
                        for sub in nested:
                            if sub.get('value') is not None:
                                values.append(sub['value'])
                            values.extend(sub.get('values', []) or [])
    assert 'Condominium' not in values


def test_residuals_do_not_overwrite_a_nonresidential_base(recipe):
    """A positively non-residential base class survives both residuals.

    Permits confirmed ~2,650 New Hanover footprints (churches, retail,
    warehouses) that the Single-Family residual had claimed; the
    occupancy_type_prevote gate stops the overwrite while a missing base
    still resolves normally (one-dwelling residual case above).
    """
    out = _run(
        recipe,
        [
            {'occupancy_type': 'Retail', 'n_dwellings_overture': 1},
            {
                'occupancy_type': 'Commercial',
                'n_dwellings_overture': 2,
                'use_group_combined_parcel': 'APARTMENTS',
            },
        ],
    )
    got = out['occupancy_type'].astype(object)
    assert got.iloc[0] == 'Retail'
    assert got.iloc[1] == 'Commercial'


def test_nsi_and_fema_agreement_alone_cannot_prove_multi(recipe):
    """The NSI/FEMA pair pools to one corroborating vote, not two.

    The two sources are near-copies (97.5% identical), so their agreement
    must not reach the multi decision's min_score by itself -- that is
    what flipped true single-family homes wherever FEMA coverage is
    complete. With a dwelling count of one, the row resolves single.
    """
    out = _run(
        recipe,
        [
            {
                'occupancy_type_building_nsi': 'Multi-Family, 2 units',
                'group_footprint_fema': 'Multi-Family',
                'n_dwellings_overture': 1,
            }
        ],
    )
    assert out['dwelling_multiplicity'].astype(object).iloc[0] == 'single'
    assert out['occupancy_type'].astype(object).iloc[0] == 'Single-Family'


def test_a_specific_multi_unit_claim_corroborates_the_pooled_pair(recipe):
    """NSI's 3-plus-unit claims are evidence; its duplex guess is noise.

    Measured against CHEER, 'Multi-Family, 2 units' is 35% right while
    the 3-plus-unit claims are 80-86% right -- so a specific claim earns
    the second vote that generic NSI/FEMA agreement deliberately does
    not, and the row resolves multi even with a dwelling count of one.
    """
    out = _run(
        recipe,
        [
            {
                'occupancy_type_building_nsi': 'Multi-Family, 10-19 units',
                'group_footprint_fema': 'Multi-Family',
                'n_dwellings_overture': 1,
            }
        ],
    )
    assert out['dwelling_multiplicity'].astype(object).iloc[0] == 'multi'
    assert out['occupancy_type'].astype(object).iloc[0] == 'Multi-Family'


def test_restricted_shovels_columns_are_never_published(recipe):
    """Shovels permit evidence is restricted-use and must never be published.

    Permit-derived columns may inform curation, but the shipped recipe's
    order_columns drop list has to remove every one of them from the output
    -- including columns another branch's steps might join in. Runs the
    shipped drop list over a frame carrying the known shovels columns and
    asserts none survive.
    """
    from openplaces.io.curator.formatters import order_columns

    restricted = [
        'occupancy_type_property_shovels',
        'occupancy_type_property_shovels_source',
        'n_permits_per_footprint',
        'n_permits_per_footprint_address',
    ]
    frame = pd.DataFrame(
        {'occupancy_type': ['Single-Family'], **{c: [1] for c in restricted}}
    )
    state = CurateState(
        recipe=recipe,
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=frame,
    )
    step = next(s for s in recipe['pipeline'] if s['step'] == 'order_columns')
    out = order_columns(
        state, overrides=step.get('overrides'), drop=step.get('drop')
    ).curated
    leaked = [c for c in out.columns if 'shovels' in c or 'permit' in c]
    assert not leaked, f'restricted permit columns survive order_columns: {leaked}'
