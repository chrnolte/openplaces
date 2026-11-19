"""Stand-in outlines read an account drawn as several lots once.

`infer_spine_additions` (plans/account-values-one-deciding-step.md, V4):
an account's repeated value counts once over its lots (the value gate),
and a lot of an account whose building is drawn on another lot gets no
stand-in (`skip_beside_drawn_account`, default on). A single-lot
account, and an account with no drawn building, keep one stand-in per
lot that passes. All values and ids are fabricated.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.links import infer_spine_additions
from openplaces.io.harmonizer.links.additions import beside_drawn_account
from openplaces.recipe import get_recipe_by_id

SPINE = 'US_footprint-spine-2026'
FOOTPRINT_GEOSPINE = 'US_footprint-geospine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
COUNTY = 'US-NC-BR'
SIDE = 0.01  # about 1 ha at this latitude
N_LINKED = 20


def _reference(lots):
    """20 drawn 1 ha lots worth 100,000 each, then *lots*:
    (key, value, drawn) per candidate lot, also 1 ha each."""
    rows = [(f'D{i}', 100_000.0, True) for i in range(N_LINKED)] + list(lots)
    ids = [f'p{i}' for i in range(len(rows))]
    ref = gpd.GeoDataFrame(
        {
            'parcel_id_local': [k for k, _, _ in rows],
            'improvement_value': [v for _, v, _ in rows],
            'use_group': ['RES'] * len(rows),
            'geometry': [
                box(i * SIDE, 0, (i + 1) * SIDE, SIDE) for i in range(len(rows))
            ],
        },
        index=pd.Index(ids, name='parcel_id'),
        crs='epsg:4326',
    )
    ref['area_ha'] = ref.geometry.to_crs(ref.estimate_utm_crs()).area / 10_000
    ref['improvement_value_per_ha'] = ref['improvement_value'] / ref['area_ha']
    drawn = [i for i, (_, _, d) in zip(ids, rows, strict=True) if d]
    return ref, drawn


def _infer(lots, cluster_lots=(), **params):
    """*cluster_lots*: drawn lots whose outline is a condominium cluster."""
    ref, drawn = _reference(lots)
    footprints = [f'f{i}' for i in range(len(drawn))]
    crosswalk = pd.DataFrame(
        {'link': ['spatial_overlay'] * len(drawn)},
        index=pd.MultiIndex.from_arrays(
            [footprints, drawn], names=['footprint_id', 'parcel_id']
        ),
    )
    spine = gpd.GeoDataFrame(
        {
            'geometry_source': [
                'condo_cluster.x' if lot in cluster_lots else 'obm' for lot in drawn
            ],
            'geometry': list(ref.geometry.reindex(drawn)),
        },
        index=pd.Index(footprints, name='footprint_id'),
        crs='epsg:4326',
    )
    state = HarmonizeState(
        recipe=get_recipe_by_id(SPINE),
        admin_id=AdminId(COUNTY),
        verbose=False,
        timer=None,
        spine=spine,
        references={PARCEL: ref},
        crosswalks={PARCEL: crosswalk},
        reference_types={PARCEL: 'parcel'},
    )
    state = infer_spine_additions(state, entity_type='parcel', **params)
    inferred = state.metadata.get(f'inferred_from_{PARCEL}')
    return set() if inferred is None else set(inferred['parcel_id'])


def test_a_single_lot_account_keeps_its_stand_in():
    # Rule 1: p20 is the only lot of its account.
    assert _infer([('A', 150_000.0, False)]) == {'p20'}


def test_an_account_with_no_drawn_lot_keeps_one_stand_in_per_lot():
    # Two lots repeat one account's 300,000: over 2 ha it still passes
    # the 100,000/ha floor, so each lot keeps a stand-in of its own.
    lots = [('B', 300_000.0, False), ('B', 300_000.0, False)]
    assert _infer(lots) == {'p20', 'p21'}


def test_an_account_counted_once_over_its_lots_can_fail_the_gate():
    # 150,000 repeated on 3 lots is 50,000/ha over the account's land.
    lots = [('C', 150_000.0, False)] * 3
    assert _infer(lots) == set()


def test_no_stand_in_beside_the_accounts_drawn_building():
    # Rule 3: the account's building is drawn on p21; p20 gets none.
    lots = [('E', 300_000.0, False), ('E', 300_000.0, True)]
    assert _infer(lots) == set()


def test_a_cluster_outline_is_not_the_accounts_drawn_building():
    # The parcel curate divides such an account by lot area, so the
    # undrawn lot keeps its stand-in to receive its share.
    lots = [('K', 300_000.0, False), ('K', 300_000.0, True)]
    assert _infer(lots, cluster_lots={'p21'}) == {'p20'}


def test_the_rule_beside_a_drawn_building_can_be_switched_off():
    lots = [('E', 300_000.0, False), ('E', 300_000.0, True)]
    assert _infer(lots, skip_beside_drawn_account=False) == {'p20'}


def test_a_shared_key_with_different_values_is_no_account():
    # Each lot states its own value: no account reading, rule 1 each.
    lots = [('F', 150_000.0, False), ('F', 400_000.0, True)]
    assert _infer(lots) == {'p20'}


def test_beside_drawn_account_marks_only_the_undrawn_lots():
    ref, drawn = _reference(
        [('G', 300_000.0, False), ('G', 300_000.0, True), ('H', 9.0, False)]
    )
    beside = beside_drawn_account(ref, pd.Index(drawn))
    assert beside[beside].index.tolist() == ['p20']


def test_the_footprint_geospine_switches_the_rule_on():
    pipeline = get_recipe_by_id(FOOTPRINT_GEOSPINE)['pipeline']
    step = next(s for s in pipeline if s['step'] == 'infer_spine_additions')
    assert step['skip_beside_drawn_account'] is True
