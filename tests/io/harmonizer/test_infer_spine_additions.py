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
