"""The attribute/geometry join says how many geometries it discarded.

A split recipe reads its attributes from one file and its geometry from
the entity_recipe predecessor. Per-admin geometry files are concatenated
on read, so a boundary entity curated by two counties can arrive twice.
The join keeps the first copy, which is a read-time convenience rather
than a decision about the data, so it now reports what it dropped
instead of discarding rows in silence.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.readers as readers
from openplaces.core.schema import AdminId
from openplaces.io import save_parquet

ADMIN_ID = AdminId('US-NC-WAK')


def _setup(monkeypatch, tmp_path, geometry_labels):
    attributes = pd.DataFrame(
        {'n_dwellings': [1.0, 2.0]},
        index=pd.Index(['a', 'b'], name='footprint_id'),
    )
    geometry = gpd.GeoDataFrame(
        {'geometry': [box(i, 0, i + 1, 1) for i in range(len(geometry_labels))]},
        index=pd.Index(geometry_labels, name='footprint_id'),
        crs='epsg:6933',
    )
    attribute_path = tmp_path / 'attributes.parquet'
    geometry_path = tmp_path / 'geometry.parquet'
    save_parquet(attributes, attribute_path)
    save_parquet(geometry, geometry_path)

    attribute_recipe = {'recipe_id': 'US_footprint-spine-2026'}
    geometry_recipe = {'recipe_id': 'US_footprint-geospine-2026'}
    paths = {
        id(attribute_recipe): attribute_path,
        id(geometry_recipe): geometry_path,
    }

    monkeypatch.setattr(
        readers, 'saves_geometry', lambda recipe: recipe is geometry_recipe
    )
    monkeypatch.setattr(readers, 'get_recipe_by_id', lambda rid: geometry_recipe)
    monkeypatch.setattr(readers, 'get_save_admin_level', lambda recipe: 0)
    monkeypatch.setattr(
        readers,
        '_get_output_admin_ids',
        # (files to read, requests finer than the save level, units asked
        # for whole)
        lambda recipe, admin_id: ([admin_id], [], [admin_id]),
    )
    monkeypatch.setattr(
        readers, 'get_output_path', lambda recipe, *a, **k: paths[id(recipe)]
    )
    attribute_recipe['entity_recipe'] = 'US_footprint-geospine-2026'
    return attribute_recipe


def test_repeated_geometry_label_is_reported(monkeypatch, tmp_path):
    recipe = _setup(monkeypatch, tmp_path, ['a', 'a', 'b'])
    with pytest.warns(UserWarning, match='repeated footprint_id label'):
        out = readers.get_entities(recipe, ADMIN_ID, geom=True)
    assert len(out) == 2
    assert out.index.tolist() == ['a', 'b']


def test_report_names_the_geometry_recipe_and_the_admin_unit(monkeypatch, tmp_path):
    recipe = _setup(monkeypatch, tmp_path, ['a', 'a', 'b'])
    with pytest.warns(UserWarning) as record:
        readers.get_entities(recipe, ADMIN_ID, geom=True)
    message = str(record[0].message)
    assert '1 repeated' in message
    assert 'US_footprint-geospine-2026' in message
    assert 'US-NC-WAK' in message


def test_unique_geometry_reads_without_a_warning(monkeypatch, tmp_path):
    recipe = _setup(monkeypatch, tmp_path, ['a', 'b'])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = readers.get_entities(recipe, ADMIN_ID, geom=True)
    assert out['n_dwellings'].tolist() == [1.0, 2.0]
    assert out.geometry.notna().all()
