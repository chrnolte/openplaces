"""get_admin(geom='land') draws units clipped to the shoreline.

The checks that read polygons need the cartographic boundary recipes
ingested on this machine and skip where they are not.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from openplaces.io import readers
from openplaces.io.readers import get_admin
from openplaces.recipe import get_recipe_id

# Land area Census reports for Michigan (ALAND), km2.
MICHIGAN_LAND_KM2 = 146_626


def _land(*args, **kwargs):
    try:
        return get_admin(*args, **kwargs)
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f'land-only boundaries are not ingested here: {error}')


class TestArguments:
    def test_a_recipe_cannot_be_combined_with_a_land_geometry(self):
        with pytest.raises(ValueError, match='omit `recipe`'):
            get_admin('US-MI', geom='land', recipe='admin-openplaces-2026_admin2')

    @pytest.mark.parametrize('level', [1, 5])
    def test_levels_outside_two_to_four_are_refused(self, level):
        with pytest.raises(ValueError, match='covers levels'):
            get_admin(level=level, geom='land')

    def test_a_country_without_a_level_is_refused(self):
        # A country id alone infers level 1, which has no land layer.
        with pytest.raises(ValueError, match='covers levels'):
            get_admin('US', geom='land')


@pytest.fixture
def stand_in_files(monkeypatch):
    """Stand in for the ingested state file of the 500k land recipe."""
    frame = gpd.GeoDataFrame(
        {'name': ['Alpha']},
        geometry=[box(-85, 43, -84, 44)],
        index=pd.Index(['US-MI'], name='admin2_id'),
        crs='EPSG:4326',
    )
    requested = []

    class _Path:
        def __init__(self, recipe_id):
            self.recipe_id = recipe_id

        def exists(self):
            return self.recipe_id == 'US_admin-censuscb-2025_admin2'

    def fake_get_output_path(recipe, admin_id=None, **kwargs):
        return _Path(get_recipe_id(recipe))

    def fake_read_parquet(path, geom=False, columns=None, filters=None, **kwargs):
        requested.append((path.recipe_id, geom, filters))
        return frame if geom else frame.drop(columns='geometry')

    monkeypatch.setattr(readers, 'get_output_path', fake_get_output_path)
    monkeypatch.setattr(readers, 'read_parquet', fake_read_parquet)
    return requested


class TestJoin:
    def test_the_polygon_of_the_land_recipe_is_attached_to_the_spine_row(
        self, stand_in_files
    ):
        state = get_admin('US-MI', geom='land')
        assert state.loc['US-MI', 'geometry'].equals(box(-85, 43, -84, 44))
        assert state.crs == 'EPSG:4326'

    def test_only_the_selected_units_are_read(self, stand_in_files):
        get_admin('US-MI', geom='land')
        recipe_id, geom, filters = stand_in_files[-1]
        assert recipe_id == 'US_admin-censuscb-2025_admin2'
        assert geom is True
        assert filters == [('admin2_id', 'in', ['US-MI'])]

    def test_a_resolution_not_ingested_names_what_to_ingest(self, stand_in_files):
        # No 20m file is stood in, and the state recipe has the 20m
        # source, so the request cannot be answered.
        with pytest.raises(FileNotFoundError, match='censuscb20m'):
            get_admin('US-MI', geom='land_coarsest')


class TestMichigan:
    def test_state_area_is_land_not_water(self):
        """Measured 2026-10-03 on US-MI: 500k area is 150,910 km2, 3% above ALAND.

        The TIGER state layer gives 250,486 km2 because it includes the
        Great Lakes.
        """
        state = _land('US-MI', geom='land')
        area = state.to_crs(5070).area.sum() / 1e6
        assert abs(area / MICHIGAN_LAND_KM2 - 1) < 0.05
        assert area < 160_000

    def test_no_county_covers_the_middle_of_lake_michigan(self):
        counties = _land('US-MI', level=3, geom='land').set_crs(
            4326, allow_override=True
        )
        lake = Point(-87.0, 43.5)
        assert not counties.geometry.contains(lake).any()

    @pytest.mark.parametrize('geom', ['land', 'land_coarse', 'land_coarsest'])
    def test_every_resolution_returns_every_unit_with_a_polygon(self, geom):
        counties = _land('US-MI', level=3, geom=geom)
        assert len(counties) == 83
        assert counties.geometry.notna().all()

    def test_coarser_resolutions_are_lighter(self):
        fine = _land('US-MI', geom='land').geometry.iloc[0]
        coarse = _land('US-MI', geom='land_coarsest').geometry.iloc[0]
        assert len(coarse.wkb) < len(fine.wkb)


class TestNewEngland:
    def test_level_three_is_towns_at_every_resolution(self):
        for geom in ['land', 'land_coarse']:
            towns = _land('US-MA', level=3, geom=geom)
            assert len(towns) > 300
            assert towns.geometry.notna().all()
