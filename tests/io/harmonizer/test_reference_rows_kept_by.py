"""`link_to_reference(kept_by=...)`: an overlay link whose reference is the
rows another spine's `resolve_spine` keeps, from every layer it ranks.

Fabricated geometry and ids. A tax lot is a combine of two record lots
beneath it; a third record lot lies apart. The parcel spine ranks tax
lots first, so it keeps the tax lot and the lone record lot, and the
footprint link has to see exactly those two (DC, F2 of
plans/findings/dc-public-land-zero-values.md).
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.harmonizer.links.spatial as spatial_mod
import openplaces.io.harmonizer.spine as spine_mod
import openplaces.recipe as recipe_mod
from openplaces.core.schema import AdminId
from openplaces.recipe.dependencies import get_recipe_dependencies


def _frame(ids, geoms, area):
    return gpd.GeoDataFrame(
        {'parcel_id_local': ids, 'land_area_sqft': area, 'geo_id': ids},
        geometry=geoms,
        crs='EPSG:4326',
        index=pd.Index(ids, name='parcel_id'),
    )


FRAMES = {
    'XX_parcel-taxlots-2026': _frame(['t1'], [box(0, 0, 1, 1)], [100.0]),
    'XX_parcel-recordlots-2026': _frame(
        ['r1', 'r2', 'r3'],
        [box(0, 0, 0.5, 1), box(0.5, 0, 1, 1), box(3, 0, 4, 1)],
        [50.0, 50.0, 100.0],
    ),
}

KEPT_RECIPE = {
    'recipe_id': 'XX_parcel-geospine-2026',
    'admin_id': AdminId('XX'),
    'stage': 'harmonize',
    'pipeline': [
        {
            'step': 'resolve_spine',
            'sources': [
                {'recipe_id': 'XX_parcel-taxlots-2026', 'label': 'taxlots'},
                {'recipe_id': 'XX_parcel-recordlots-2026', 'label': 'recordlots'},
                {'auto_discover': False},
            ],
            'keep_columns': ['parcel_id_local'],
        }
    ],
}


def _patch(monkeypatch, frames=FRAMES, kept=KEPT_RECIPE):
    def entities(recipe_id, *args, **kwargs):
        return frames[recipe_id].copy()

    monkeypatch.setattr(spine_mod, '_drop_out_of_unit_rows', lambda gdf, *a, **k: gdf)
    monkeypatch.setattr(spine_mod, 'get_entities', entities)
    monkeypatch.setattr(spatial_mod, 'get_entities', entities)
    monkeypatch.setattr(recipe_mod, 'get_recipe_by_id', lambda rid, **k: kept)
    monkeypatch.setattr(recipe_mod, 'apply_recipe_patches', lambda r, a: (r, []))


def test_the_reference_is_the_rows_the_spine_keeps(monkeypatch):
    _patch(monkeypatch)
    rows, config = spatial_mod.reference_rows(
        'XX_parcel-recordlots-2026', AdminId('XX-YY'), 'XX_parcel-geospine-2026'
    )
    # The tax lot replaces the two record lots it combines; the lone
    # record lot stays. Full ingest rows, not the spine's kept columns.
    assert sorted(rows.index) == ['r3', 't1']
    assert rows.loc['t1', 'land_area_sqft'] == 100.0
    assert config['sources'] == [
        'XX_parcel-recordlots-2026',
        'XX_parcel-taxlots-2026',
    ]
    assert config['recipe_id'] == 'XX_parcel-geospine-2026'


def test_a_spine_of_one_layer_reads_the_plain_reference(monkeypatch):
    kept = {
        **KEPT_RECIPE,
        'pipeline': [
            {
                'step': 'resolve_spine',
                'sources': [
                    {'recipe_id': 'XX_parcel-recordlots-2026', 'label': 'recordlots'},
                    {'auto_discover': False},
                ],
            }
        ],
    }
    _patch(monkeypatch, kept=kept)
    rows, config = spatial_mod.reference_rows(
        'XX_parcel-recordlots-2026', AdminId('XX-YY'), 'XX_parcel-geospine-2026'
    )
    # Nothing changes where the spine keeps the one reference: same rows,
    # and no kept_by entry in the fingerprint.
    assert config is None
    assert sorted(rows.index) == ['r1', 'r2', 'r3']
    step_config = spatial_mod.overlay_step_config(1 / 6, 10, None, None, None, config)
    assert 'kept_by' not in step_config


def test_without_kept_by_the_reference_is_unchanged(monkeypatch):
    _patch(monkeypatch)
    rows, config = spatial_mod.reference_rows(
        'XX_parcel-recordlots-2026', AdminId('XX-YY')
    )
    assert config is None
    assert sorted(rows.index) == ['r1', 'r2', 'r3']


def test_kept_by_schedules_the_layers_not_the_spine(monkeypatch):
    # The footprint geospine runs before the parcel geospine; a kept_by
    # link reads the parcel spine's sources, never its output.
    linking = {
        'recipe_id': 'XX_footprint-geospine-2026',
        'admin_id': AdminId('XX'),
        'stage': 'harmonize',
        'pipeline': [
            {
                'step': 'link_to_reference',
                'entity_type': 'parcel',
                'kept_by': 'XX_parcel-geospine-2026',
            }
        ],
    }
    monkeypatch.setattr(recipe_mod, 'get_recipe_by_id', lambda rid, **k: KEPT_RECIPE)
    upstream = {e.upstream_recipe_id for e in get_recipe_dependencies(linking)}
    assert {'XX_parcel-taxlots-2026', 'XX_parcel-recordlots-2026'} <= upstream
    assert 'XX_parcel-geospine-2026' not in upstream
