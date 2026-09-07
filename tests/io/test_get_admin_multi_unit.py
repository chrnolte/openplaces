"""get_admin() reads every output file a multi-unit request covers.

An admin recipe that ships one file per level-2 unit
(`admin-openplaces-2026_admin3`) used to be keyed off a single deepest
requested id, so a two-state request opened one state's file and
returned the other state as spine-only rows with null geometry.
"""

import geopandas as gpd
import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io import readers
from openplaces.io.readers import get_admin

RECIPE_ID = 'admin-openplaces-2026_admin3'


@pytest.fixture
def two_state_files(monkeypatch):
    """Stand in for two per-state output files of the admin3 recipe."""
    frames = {
        'US-NC': gpd.GeoDataFrame(
            {'name': ['Alpha County']},
            geometry=gpd.points_from_xy([-79.0], [35.6]),
            index=pd.Index(['US-NC-ALP'], name='admin3_id'),
            crs='EPSG:4326',
        ),
        'US-TX': gpd.GeoDataFrame(
            {'name': ['Beta County']},
            geometry=gpd.points_from_xy([-97.7], [30.3]),
            index=pd.Index(['US-TX-BET'], name='admin3_id'),
            crs='EPSG:4326',
        ),
    }
    paths = {}

    def fake_get_output_path(recipe, admin_id=None, **kwargs):
        class _Path:
            def __init__(self, key):
                self.key = key

            def exists(self):
                return self.key in frames

            def __str__(self):
                return f'<{self.key}>'

        path = _Path(str(admin_id))
        paths[str(admin_id)] = path
        return path

    def fake_read_parquet(path, geom=False, columns=None, filters=None, **kwargs):
        frame = frames[path.key]
        if columns == []:
            return frame[[]]
        return frame if geom else frame.drop(columns='geometry')

    monkeypatch.setattr(readers, 'get_output_path', fake_get_output_path)
    monkeypatch.setattr(readers, 'read_parquet', fake_read_parquet)
    return frames


class TestGetAdminReadsEveryRequestedUnit:
    def test_both_states_come_back_with_geometry(self, two_state_files):
        admin = get_admin(
            ['US-NC', 'US-TX'], 3, recipe=RECIPE_ID, geom=True, all_columns=True
        )
        assert 'US-NC-ALP' in admin.index
        assert 'US-TX-BET' in admin.index
        assert admin.loc[['US-NC-ALP', 'US-TX-BET'], 'geometry'].notna().all()

    def test_a_unit_without_a_file_still_comes_back_from_the_spine(
        self, two_state_files
    ):
        # Wisconsin has no stand-in file here: its rows must survive as
        # spine rows rather than aborting the whole read.
        admin = get_admin(
            ['US-NC', 'US-WI'], 3, recipe=RECIPE_ID, geom=True, all_columns=True
        )
        assert 'US-NC-ALP' in admin.index
        assert any(str(i).startswith('US-WI-') for i in admin.index)

    def test_no_file_at_all_raises(self, two_state_files):
        with pytest.raises(FileNotFoundError):
            get_admin('US-WI', 3, recipe=RECIPE_ID, geom=True)


class TestGetAdminWithoutAdminId:
    def test_a_global_recipe_saving_per_unit_resolves_its_paths(self, two_state_files):
        # The recipe's own admin_id is NULL, so there is no single path
        # to key on. Every save-level unit is resolved instead.
        admin = get_admin(level=3, recipe=RECIPE_ID, geom=True, all_columns=True)
        assert 'US-NC-ALP' in admin.index
        assert 'US-TX-BET' in admin.index


class TestConcatRecipeFiles:
    def test_a_single_frame_is_returned_unchanged(self):
        frame = pd.DataFrame({'a': [1]})
        assert (
            readers._concat_recipe_files([object()], reader=lambda path: frame) is frame
        )

    def test_geometry_and_crs_survive_concatenation(self):
        frames = [
            gpd.GeoDataFrame(
                {'a': [i]},
                geometry=gpd.points_from_xy([i], [i]),
                index=pd.Index([f'US-XX-{i}'], name='admin3_id'),
                crs='EPSG:4326',
            )
            for i in (1, 2)
        ]
        combined = readers._concat_recipe_files(
            [0, 1], reader=lambda path: frames[path]
        )
        assert isinstance(combined, gpd.GeoDataFrame)
        assert combined.crs == frames[0].crs
        assert len(combined) == 2


def test_admin_id_helper_resolves_a_null_scope_recipe():
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id(RECIPE_ID)
    output_ids, finer, whole = readers._get_output_admin_ids(
        recipe, [AdminId('US-NC-WAK')]
    )
    assert output_ids == [AdminId('US-NC')]
    assert finer == [AdminId('US-NC-WAK')]
    assert whole == []
