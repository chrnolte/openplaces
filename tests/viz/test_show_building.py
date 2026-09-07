"""Tests for `show_building` and the ingested-geometry overview map."""

from types import SimpleNamespace
from unittest.mock import patch

import geopandas as gpd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from shapely.geometry import LineString, box  # noqa: E402

from openplaces.core.schema import Entity  # noqa: E402
from openplaces.viz.maps import show_building, show_ingested_geometries  # noqa: E402

# A point inside the parcel below, and its surroundings, in EPSG:4326.
LON, LAT = -78.0, 35.0
STEP = 0.0002


def _square(i=0, j=0, index_name='parcel_id', ids=('p1',), **columns):
    geometry = [
        box(
            LON + (i + k) * STEP,
            LAT + j * STEP,
            LON + (i + k + 1) * STEP,
            LAT + (j + 1) * STEP,
        )
        for k in range(len(ids))
    ]
    return gpd.GeoDataFrame(
        columns,
        geometry=geometry,
        index=pd.Index(list(ids), name=index_name),
        crs='EPSG:4326',
    )


@pytest.fixture
def parcels():
    """One parcel around the plot center, with no attributes filled in."""
    return _square(
        i=-1,
        ids=('p1',),
        geo_id=['8QRX+2C'],
        parcel_id_admin3=['0001'],
        address=[None],
        use_group=[np.nan],
        use_subgroup=[None],
        value=[np.nan],
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close('all')


def test_parcel_label_box_tolerates_missing_values(parcels):
    """A NaN float is truthy, so a null read as present and .title() raised.

    year_built, value, improvement_value, land_value and
    legal_description are absent from many parcel layers, four of them
    with no presence guard at all, so the box raised after the map had
    already been drawn.
    """
    fig, ax = show_building(
        (LAT + STEP / 2, LON - STEP / 2),
        {'parcels': parcels},
        show_basemap=False,
        return_fig_ax=True,
    )
    box_text = [text.get_text() for text in ax.texts if 'Parcel ID' in text.get_text()]
    assert box_text
    assert 'Value:' not in box_text[0]
    assert 'nan' not in box_text[0].lower()


def test_crosshair_match_requires_a_named_index(parcels):
    """None == None matched any frame, labeling an unrelated row.

    The membership test then reduced to "does label 0 exist", so an
    arbitrary row of an unnamed-index frame was labeled as the
    crosshair building.
    """
    unrelated = gpd.GeoDataFrame(
        {'address': ['somewhere else entirely']},
        # Far away, so nothing puts it on the parcel or in the bbox.
        geometry=[box(LON + 5, LAT + 5, LON + 5.001, LAT + 5.001)],
        crs='EPSG:4326',
    )
    location = gpd.GeoDataFrame(
        geometry=[box(LON - STEP, LAT, LON, LAT + STEP)], crs='EPSG:4326'
    )
    assert unrelated.index.name is None and location.index.name is None

    fig, ax = show_building(
        location,
        {'parcels': parcels, 'footprints': unrelated},
        show_basemap=False,
        return_fig_ax=True,
    )
    assert not any('somewhere else' in text.get_text() for text in ax.texts)


def test_overlay_survives_a_shared_index_name(parcels):
    """The two layers commonly share an index name, and overlay suffixes.

    Both frames are reset before the overlay, so the colliding id
    columns came back as name_1/name_2 and the lookup by index name
    raised a KeyError.
    """
    footprints = _square(
        i=-1,
        index_name='parcel_id',
        ids=('f1',),
        address=['1 Test Street'],
    )
    location = gpd.GeoDataFrame(
        geometry=[box(LON - STEP, LAT, LON, LAT + STEP)],
        index=pd.Index(['f1'], name='parcel_id'),
        crs='EPSG:4326',
    )

    fig, ax = show_building(
        location,
        {'parcels': parcels, 'footprints': footprints},
        show_basemap=False,
        return_fig_ax=True,
    )
    assert any('1 Test Street' in text.get_text() for text in ax.texts)


def _ingester(entities):
    recipe = {'entity': Entity('footprint', 'test', '2026')}
    return SimpleNamespace(
        admin_ids_to_save=['US'],
        recipe=recipe,
        _first_partition_id=None,
    ), entities


def test_line_geometry_is_plotted_and_not_left_to_the_default_extent():
    """Lines fell through both branches, leaving the (0, 1) limits.

    The basemap call then fetched tiles for a degree square in the
    Atlantic off West Africa: a plausible-looking map of nothing.
    """
    lines = gpd.GeoDataFrame(
        {'name': ['a', 'b']},
        geometry=[
            LineString([(LON, LAT), (LON + STEP, LAT + STEP)]),
            LineString([(LON + STEP, LAT), (LON + 2 * STEP, LAT + STEP)]),
        ],
        crs='EPSG:4326',
    )
    ingester, entities = _ingester(lines)

    with (
        patch('openplaces.viz.maps._has_geometry_output', return_value=True),
        patch('openplaces.viz.maps.get_entities', return_value=entities),
        patch('openplaces.viz.maps.cx.add_basemap') as add_basemap,
    ):
        fig, ax = show_ingested_geometries(ingester)

    assert ax.collections, 'the lines were not drawn'
    assert add_basemap.called
    xmin, xmax = ax.get_xlim()
    assert xmin <= LON <= xmax
