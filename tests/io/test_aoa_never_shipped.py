"""No PLACES-FMV area of applicability reaches a bundle, team ones included.

PLACES-FMV computed its area of applicability over building footprint
covariates (design limit L4), so no bundle carries it
(maintainer, 2026-10-24). The Nantucket patch of the parcel curate
used to merge the seven `area_of_applicability_region_nb_*` columns
beside the land values of `US_parcel-placeslab-fmv2026open`, and the
parcel recipe listed them under `share: columns:`. A column left out of
`share: columns:` alone would still ship, in the `_evidence` file,
which holds every remaining curated column; so the columns are kept out
of the curated file itself. Checked on the real recipes, with
fabricated rows.
"""

import re

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

import openplaces.io.curator.evidence as evidence_module
from openplaces.core.schema import AdminId
from openplaces.diagnostics import _recipe_index
from openplaces.io import save_parquet, to_parquet
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import merge_enrichments
from openplaces.io.delivery import export_delivery, share_columns
from openplaces.recipe import get_output_path, get_recipe_by_id
from openplaces.recipe.patches import apply_recipe_patches

AOA = re.compile(r'area_of_applicability_region_nb')
PARCEL = 'US_parcel-openplaces-2026'
OPEN = 'US_parcel_parcel-placeslab-fmv2026open'
UNIT = 'US-MA-NTC'
VARIANTS = (
    'bare',
    'people',
    'buildings',
    'wealth',
    'peoplewealth',
    'peoplebuildings',
    'popbldwealth',
)
LAND_VALUE = 'land_value_per_ha_log_region_nb_bare'
DATA_ROLES = ('canonical', 'point', 'geo', 'evidence')


def _aoa(names):
    return [name for name in names if AOA.search(str(name))]


def test_no_recipe_declares_an_aoa_column_for_a_bundle():
    index = _recipe_index()
    for recipe_id in index.loc[index['stage'] == 'curate', 'recipe_id']:
        recipe = get_recipe_by_id(recipe_id)
        columns, point_columns = share_columns(recipe)
        assert not _aoa([*columns, *point_columns]), recipe_id


def _merge_specs(recipe):
    return [
        spec
        for step in recipe['pipeline']
        if step.get('step') == 'merge_enrichments'
        for spec in step.get('recipes') or []
    ]


def test_no_curate_merge_maps_an_aoa_column():
    """Every curate recipe, with every patch that amends it applied."""
    index = _recipe_index()
    curate = index[index['stage'] == 'curate']
    patches = curate[curate['patches'].fillna('') != '']
    runs = [(rid, 'US') for rid in curate['recipe_id']]
    runs += list(zip(patches['patches'], patches['admin_id'], strict=True))
    for recipe_id, admin_id in runs:
        recipe = get_recipe_by_id(recipe_id)
        if not recipe.get('pipeline'):
            continue
        recipe, _ = apply_recipe_patches(recipe, admin_id)
        for spec in _merge_specs(recipe):
            targets = list((spec.get('columns') or {}).values())
            assert not _aoa(targets), (recipe_id, admin_id, spec['recipe_id'])


@pytest.fixture
def nantucket_curated(mock_data_root, tmp_path, monkeypatch):
    """Nantucket parcels curated with fabricated placeslab evidence.

    The evidence carries the source's own AOA columns beside its land
    values, as the enrichment writes them; the merge runs the real
    patched pipeline's specs.
    """
    ids = pd.Index(['a', 'b'], name='parcel_id')
    evidence = pd.DataFrame(index=ids)
    for variant in VARIANTS:
        evidence[f'land_value_per_ha_log_region_nb_{variant}_2010_fmv2026open'] = [
            11.0,
            12.0,
        ]
        evidence[f'aoa_region_nb_{variant}_2010_blockgroup_fmv2026open'] = [
            0.5,
            1.5,
        ]
    evidence_path = tmp_path / 'fabricated_fmv2026open.parquet'
    to_parquet(evidence, evidence_path)

    def _evidence_path(recipe, *_args, **_kwargs):
        # Only the placeslab evidence exists; every other merge finds
        # no file and is skipped, as on a unit it was not run for.
        if recipe.get('recipe_id') == OPEN:
            return evidence_path
        return tmp_path / 'absent.parquet'

    monkeypatch.setattr(evidence_module, 'get_output_path', _evidence_path)

    recipe, applied = apply_recipe_patches(get_recipe_by_id(PARCEL), UNIT)
    assert 'US-MA-NTC_parcel-openplaces-2026' in applied
    curated = gpd.GeoDataFrame(
        {
            'admin3_id': [UNIT, UNIT],
            'lat': [41.28, 41.29],
            'long': [-70.10, -70.11],
            'geometry_source': ['parcel', 'parcel'],
        },
        geometry=[box(-70.10, 41.28, -70.09, 41.29), box(-70.11, 41.29, -70.10, 41.3)],
        crs='EPSG:4326',
        index=ids,
    )
    state = CurateState(
        recipe=recipe,
        entity_recipe=None,
        admin_id=AdminId(UNIT),
        verbose=False,
        timer=None,
        curated=curated,
    )
    state = merge_enrichments(state, _merge_specs(recipe))
    save_parquet(
        state.curated, get_output_path(get_recipe_by_id(PARCEL), UNIT), combined=True
    )
    return state.curated


def test_the_nantucket_merge_keeps_the_land_values_without_their_aoa(
    nantucket_curated,
):
    assert nantucket_curated[LAND_VALUE].notna().all()
    assert not _aoa(nantucket_curated.columns)


@pytest.mark.parametrize('region', ['nantucket-ma', 'nantucket-ma.team'])
def test_no_nantucket_bundle_carries_an_aoa_column(nantucket_curated, region):
    paths = export_delivery(PARCEL, region=region)
    for role in DATA_ROLES:
        names = pq.read_schema(paths[role]).names
        assert not _aoa(names), (region, role)
    canonical = pd.read_parquet(paths['canonical'])
    # Not vacuous: the team bundle ships the land values, the public
    # one withholds them.
    if region.endswith('.team'):
        assert canonical[LAND_VALUE].notna().all()
    else:
        assert canonical[LAND_VALUE].isna().all()
