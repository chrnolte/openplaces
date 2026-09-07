"""An attribute-only recipe resolving geometry through its predecessor.

`US_footprint-spine-2026` declares `save_to: geometry: false`, so
`get_entities` reads its attributes and then reads the geospine named by
`entity_recipe` for the geometry. These pin what happens when that second
read comes back with nothing, and that the join key is not renumbered
underneath it.
"""

import geopandas as gpd
import pandas as pd
import pytest

from openplaces.core.schema import AdminId, Entity
from openplaces.io import readers
from openplaces.io.readers import get_entities

GEOSPINE = {
    'recipe_id': 'US_footprint-geotest-2026',
    'admin_id': AdminId('US'),
    'entity': Entity('footprint', 'geotest', '2026'),
    'save_to': {'admin_level': 3, 'data_dir': 'core'},
}
ATTRIBUTES = {
    'recipe_id': 'US_footprint-test-2026',
    'admin_id': AdminId('US'),
    'entity': Entity('footprint', 'test', '2026'),
    'entity_recipe': 'US_footprint-geotest-2026',
    'save_to': {'admin_level': 3, 'data_dir': 'core', 'geometry': False},
}


class _Path:
    def __init__(self, recipe_id, key, partition_id, present):
        self.recipe_id = recipe_id
        self.key = key
        self.partition_id = partition_id
        self._present = present

    def exists(self):
        return self._present

    def __str__(self):
        return f'<{self.recipe_id}/{self.key}/{self.partition_id}>'


@pytest.fixture
def recipes(monkeypatch):
    """Patch recipe lookup and file reads for the pair above."""
    state = {'frames': {}, 'requested': []}

    def fake_get_recipe_by_id(recipe_id, **kwargs):
        return {'US_footprint-geotest-2026': GEOSPINE}[recipe_id]

    def fake_get_output_path(recipe, admin_id=None, partition_id=None, **kwargs):
        key = (recipe['recipe_id'], str(admin_id), partition_id)
        state['requested'].append(key)
        return _Path(*key, present=key[:2] in state['frames'])

    def fake_read_parquet(path, geom=False, columns=None, bbox=None, **kwargs):
        return state['frames'][(path.recipe_id, path.key)].copy()

    monkeypatch.setattr(readers, 'get_recipe_by_id', fake_get_recipe_by_id)
    monkeypatch.setattr(readers, 'get_output_path', fake_get_output_path)
    monkeypatch.setattr(readers, 'read_parquet', fake_read_parquet)
    return state


def _attributes(index, values):
    return pd.DataFrame({'value': values}, index=pd.Index(index, name='geo_id'))


class TestMissingPredecessorGeometry:
    def test_ignore_returns_an_empty_geometry_column(self, recipes):
        recipes['frames'][('US_footprint-test-2026', 'US-NC-AAA')] = _attributes(
            ['g1', 'g2'], [1, 2]
        )
        data = get_entities(
            ATTRIBUTES, admin_id='US-NC-AAA', geom=True, missing='ignore'
        )
        assert isinstance(data, gpd.GeoDataFrame)
        assert 'geometry' in data
        assert data['geometry'].isna().all()

    def test_raise_names_the_geometry_recipe(self, recipes):
        recipes['frames'][('US_footprint-test-2026', 'US-NC-AAA')] = _attributes(
            ['g1'], [1]
        )
        with pytest.raises(FileNotFoundError):
            get_entities(ATTRIBUTES, admin_id='US-NC-AAA', geom=True, missing='raise')


class TestPredecessorReadScope:
    def test_the_partition_the_caller_named_reaches_the_predecessor(self, recipes):
        recipes['frames'][('US_footprint-test-2026', 'US-NC-AAA')] = _attributes(
            ['g1'], [1]
        )
        get_entities(
            ATTRIBUTES,
            admin_id='US-NC-AAA',
            geom=True,
            partition_id='2024',
            missing='ignore',
        )
        assert (
            'US_footprint-geotest-2026',
            'US-NC-AAA',
            '2024',
        ) in recipes['requested']


class TestIndexIsNotRenumbered:
    def test_range_indexed_files_keep_their_labels(self, recipes):
        for unit in ('US-NC-AAA', 'US-NC-BBB'):
            recipes['frames'][('US_footprint-test-2026', unit)] = pd.DataFrame(
                {'value': [1, 2]}
            )
        recipes['frames'][('US_footprint-geotest-2026', 'US-NC-AAA')] = (
            gpd.GeoDataFrame(
                {'value': [1, 2]},
                geometry=gpd.points_from_xy([0, 1], [0, 1]),
                crs='EPSG:4326',
            )
        )
        with pytest.warns(UserWarning, match='repeat index labels'):
            data = get_entities(
                ATTRIBUTES,
                admin_id=['US-NC-AAA', 'US-NC-BBB'],
                geom=True,
                missing='ignore',
            )
        # Renumbering would make this 0, 1, 2, 3 and hand rows of the
        # second county the first county's geometry.
        assert list(data.index) == [0, 1, 0, 1]
