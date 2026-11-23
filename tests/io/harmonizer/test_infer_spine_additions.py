"""Tests for `infer_spine_additions`' improvement-value gate.

Regression coverage for a real bug found in the aggregated CHEER inventory:
a parcel reference carrying no ``use_group``/``purpose_group`` column (a
dozen NC county assessor sources publish no land-use code at all) sent both
of the step's gates down their ``else`` branch, where each evaluated to an
unconditional ``True``. Every parcel without a linked footprint therefore
became an inferred footprint -- vacant land included -- so those counties
showed 20-37% of their footprint spine sourced from parcel polygons against
~2-6% everywhere else, and >80% of the inferred rows sat on parcels whose
assessor improvement value is exactly $0.

The gate now always applies: per-group where a group column exists, against
the reference-wide floor otherwise, with a $0/null/infinite improvement
value rejected outright either way.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.links import infer_spine_additions
from openplaces.recipe import get_recipe_by_id

SPINE = 'US_footprint-spine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
COUNTY = 'US-NC-BR'

# Every parcel below is 1 ha, so its improvement value doubles as its per-ha
# density and the thresholds can be read straight off the dollar figures.
HA_SIDE = 0.01
# 20 linked parcels all worth the same makes the reference-wide floor -- the
# 5th percentile of the linked, positive densities -- exactly $100,000/ha.
LINKED_VALUE = 100_000.0
N_LINKED = 20


def _parcels(candidate_values, candidate_groups=None, linked_extra=None):
    """Build a row of 1 ha parcels: the linked block, then the candidates.

    *linked_extra* appends ``(value, group)`` pairs to the linked block, for
    tests that need a second land-use group with its own value profile.
    """
    values = [LINKED_VALUE] * N_LINKED
    groups = ['RES'] * N_LINKED
    for value, group in linked_extra or []:
        values.append(value)
        groups.append(group)
    n_linked = len(values)

    values += list(candidate_values)
    groups += list(candidate_groups or ['RES'] * len(candidate_values))

    geoms = [
        box(i * HA_SIDE, 0, (i + 1) * HA_SIDE, HA_SIDE) for i in range(len(values))
    ]
    ids = [f'p{i}' for i in range(len(values))]
    ref = gpd.GeoDataFrame(
        {'improvement_value': values, 'use_group': groups, 'geometry': geoms},
        index=pd.Index(ids, name='parcel_id'),
        crs='epsg:4326',
    )
    area_ha = ref.geometry.to_crs(ref.estimate_utm_crs()).area / 10_000
    ref['improvement_value_per_ha'] = ref['improvement_value'] / area_ha
    return ref, ids[:n_linked]


def _state(ref, linked_ids, with_use_group):
    if not with_use_group:
        ref = ref.drop(columns='use_group')
    crosswalk = pd.DataFrame(
        {'link': ['spatial_overlay'] * len(linked_ids)},
        index=pd.MultiIndex.from_arrays(
            [[f'f{i}' for i in range(len(linked_ids))], list(linked_ids)],
            names=['footprint_id', 'parcel_id'],
        ),
    )
    spine = gpd.GeoDataFrame(
        {'geometry': []},
        index=pd.Index([], name='footprint_id'),
        crs='epsg:4326',
    )
    return HarmonizeState(
        recipe=get_recipe_by_id(SPINE),
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=spine,
        references={PARCEL: ref},
        crosswalks={PARCEL: crosswalk},
        reference_types={PARCEL: 'parcel'},
    )


def _infer(ref, linked_ids, with_use_group):
    """Run the step and return the parcel ids it turned into footprints."""
    state = infer_spine_additions(
        _state(ref, linked_ids, with_use_group), entity_type='parcel'
    )
    inferred = state.metadata.get(f'inferred_from_{PARCEL}')
    return set() if inferred is None else set(inferred['parcel_id'])


@pytest.mark.parametrize('with_use_group', [False, True])
def test_only_candidates_above_the_floor_are_inferred(with_use_group):
    # Candidates in order: two real buildings, then vacant land spelled the
    # three ways an assessor spells it -- $0, null, and a token value far
    # below anything that carries a building here.
    ref, linked = _parcels([150_000, 120_000, 0, None, 5_000])
    assert _infer(ref, linked, with_use_group) == {'p20', 'p21'}


def test_zero_valued_candidate_is_dropped_without_a_use_group_column():
    # The bug in isolation: with no group column both gates used to become
    # unconditionally True, so every unlinked parcel was inferred.
    ref, linked = _parcels([0, 0, 0])
    assert _infer(ref, linked, with_use_group=False) == set()


def test_zeros_do_not_drag_a_mixed_group_threshold_down_to_the_floor():
    # 'MIX' is mostly vacant, with one expensive linked building. Counting
    # its $0 rows puts its 5th percentile at 0, which collapses the group
    # threshold onto the reference-wide floor and lets a $150k/ha candidate
    # through; over positives only the threshold is $400k/2 = $200k/ha, and
    # the candidate is correctly held back.
    ref, linked = _parcels(
        [150_000],
        candidate_groups=['MIX'],
        linked_extra=[(0.0, 'MIX')] * 9 + [(400_000.0, 'MIX')],
    )
    assert _infer(ref, linked, with_use_group=True) == set()
    # Same parcel above MIX's own threshold is still inferred, so the group
    # is gated, not disabled.
    ref.loc['p30', 'improvement_value_per_ha'] = 250_000.0
    assert _infer(ref, linked, with_use_group=True) == {'p30'}


def test_reference_without_any_positive_linked_value_infers_nothing():
    # A source that maps improvement_value but ships it empty (or folds it
    # into a combined total) gives nothing to calibrate against. Inferring
    # on an uncalibrated threshold is what produced the vacant-land
    # footprints, so the step bails out loudly instead.
    ref, linked = _parcels([250_000, 300_000])
    ref['improvement_value_per_ha'] = 0.0
    ref.loc[['p20', 'p21'], 'improvement_value_per_ha'] = [250_000.0, 300_000.0]

    with pytest.warns(UserWarning, match='no value threshold can be calibrated'):
        assert _infer(ref, linked, with_use_group=False) == set()


def test_zero_area_parcel_is_not_inferred_from_an_infinite_density():
    ref, linked = _parcels([150_000, 150_000])
    ref.loc['p20', 'improvement_value_per_ha'] = np.inf
    assert _infer(ref, linked, with_use_group=False) == {'p21'}


def _roll_values(monkeypatch, values, uses=None):
    """Make the replayed parcel geospine passes return *values* (and
    *uses*, a land use per lot) per lot."""
    import openplaces.io.harmonizer.links.parcel_pieces as pieces
    import openplaces.io.harmonizer.links.spatial as spatial

    def rows(recipe_id, admin_id, kept_by=None, verbose=False):
        return pd.DataFrame({'geo_id': list(values)}), None

    def evidence(state, lots, reference_id, values_recipe, columns=()):
        frame = pd.DataFrame(
            {
                'improvement_value': pd.Series(values),
                'has_structure': np.nan,
                'use_group': pd.Series(uses or {}, dtype='object'),
            }
        ).rename_axis('parcel_id')
        return frame.reindex(columns=list(columns))

    monkeypatch.setattr(spatial, 'reference_rows', rows)
    monkeypatch.setattr(pieces, 'lot_structure_evidence', evidence)


def _cadastre_without_values(candidate_values):
    """The fixture's parcels as a cadastral layer: area, no value."""
    ref, linked = _parcels(candidate_values)
    roll = ref['improvement_value'].copy()
    ref['area_ha'] = ref['improvement_value'] / ref['improvement_value_per_ha']
    ref['area_ha'] = ref['area_ha'].fillna(ref['area_ha'].median())
    ref = ref.drop(columns=['improvement_value', 'improvement_value_per_ha'])
    return ref, linked, roll


@pytest.mark.parametrize('replay', [False, True])
def test_a_reference_without_values_reads_the_roll_through_the_parcel_passes(
    monkeypatch, replay
):
    # A cadastral layer states no value; the roll the parcel geospine
    # joins by id does. Without the replay the step has nothing to gate
    # on and makes no fallback in the unit.
    ref, linked, roll = _cadastre_without_values([150_000, 0, 5_000])
    _roll_values(monkeypatch, roll.to_dict())
    state = _state(ref, linked, with_use_group=True)
    state = infer_spine_additions(
        state,
        entity_type='parcel',
        lot_values_from='US_parcel-geospine-2026' if replay else None,
    )
    inferred = state.metadata.get(f'inferred_from_{PARCEL}')
    got = set() if inferred is None else set(inferred['parcel_id'])
    assert got == ({'p20'} if replay else set())


@pytest.mark.parametrize('replayed_value', [np.nan, 0.0])
def test_a_lot_the_replay_values_lower_keeps_the_references(
    monkeypatch, replayed_value
):
    # The parcel spine can pick a stale 0 from a lower-ranked layer
    # where the reference values the house; missing is kept likewise.
    ref, linked, roll = _cadastre_without_values([150_000, 150_000])
    ref['improvement_value'] = roll
    replayed = roll.copy()
    replayed['p21'] = replayed_value
    _roll_values(monkeypatch, replayed.to_dict())
    state = infer_spine_additions(
        _state(ref, linked, with_use_group=False),
        entity_type='parcel',
        lot_values_from='US_parcel-geospine-2026',
    )
    assert set(state.metadata[f'inferred_from_{PARCEL}']['parcel_id']) == {
        'p20',
        'p21',
    }


def _inferred(state):
    inferred = state.metadata.get(f'inferred_from_{PARCEL}')
    return set() if inferred is None else set(inferred['parcel_id'])


@pytest.mark.parametrize('bypass', [False, True])
def test_a_valued_lot_in_a_group_without_footprints_passes_on_value_per_m2(bypass):
    # 'VAC' has no linked lot, so its footprint rate is 0 and the group
    # gate refuses both candidates; with the bypass set between their
    # values per m2, the denser one passes and the other stays out.
    ref, linked = _parcels([150_000, 120_000], candidate_groups=['VAC', 'VAC'])
    per_m2 = ref['improvement_value_per_ha'] / 10_000
    thresholds = {'n_per_group_min': 0.2, 'value_per_ha_quantile': 0.05}
    if bypass:
        thresholds['group_gate_value_per_m2'] = (per_m2['p20'] + per_m2['p21']) / 2
    state = infer_spine_additions(
        _state(ref, linked, with_use_group=True),
        entity_type='parcel',
        thresholds=thresholds,
    )
    assert _inferred(state) == ({'p20'} if bypass else set())


def test_a_land_use_matching_the_pattern_is_never_an_outline(monkeypatch):
    # The cadastral layer states no land use; the parcel spine's replay
    # names one candidate parking and another boat slips.
    ref, linked, roll = _cadastre_without_values([150_000, 150_000, 150_000])
    _roll_values(
        monkeypatch,
        roll.to_dict(),
        uses={'p20': 'COM PARKING LOT', 'p21': 'Slip Only', 'p22': 'RES'},
    )
    state = infer_spine_additions(
        _state(ref, linked, with_use_group=False),
        entity_type='parcel',
        lot_values_from='US_parcel-geospine-2026',
        exclude_use_pattern=r'open.?space|\bslips?\b|parking',
    )
    assert _inferred(state) == {'p22'}


def test_a_lot_larger_than_the_footprint_lots_is_not_one_outline():
    ref, linked = _parcels([150_000, 150_000])
    ref['area_ha'] = 1.0
    ref.loc['p21', 'area_ha'] = 5.0
    state = infer_spine_additions(
        _state(ref, linked, with_use_group=False),
        entity_type='parcel',
        thresholds={'lot_area_quantile_max': 0.95},
    )
    assert _inferred(state) == {'p20'}


@pytest.mark.parametrize('floor', [None, 10.0])
def test_a_sliver_of_a_lot_is_not_an_outline(floor):
    # p21 is 4 m2 and dense enough to pass the value gate; only the
    # size floor keeps it out.
    ref, linked = _parcels([150_000, 150_000])
    ref['area_ha'] = 1.0
    ref.loc['p21', 'area_ha'] = 0.0004
    thresholds = {} if floor is None else {'lot_area_m2_min': floor}
    state = infer_spine_additions(
        _state(ref, linked, with_use_group=False),
        entity_type='parcel',
        thresholds=thresholds,
    )
    assert _inferred(state) == ({'p20', 'p21'} if floor is None else {'p20'})
