"""What get_admin() returns when part of the answer is empty.

Three narrow contracts: a geom=True read stays a GeoDataFrame even when
none of the selected units has geometry, reporting progress does not
crash, and a region member list ignores blank cells.
"""

import geopandas as gpd
import pandas as pd
import pytest

from openplaces.io import readers
from openplaces.io.readers import get_admin, get_region_admin_ids

RECIPE_ID = 'admin-openplaces-2026_admin3'


@pytest.fixture
def county_file_without_geometry(monkeypatch):
    """One county file whose every geometry is missing."""
    frame = gpd.GeoDataFrame(
        {'name': ['Alpha County']},
        geometry=gpd.GeoSeries([None], crs='EPSG:4326'),
        index=pd.Index(['US-NC-ALP'], name='admin3_id'),
        crs='EPSG:4326',
    )

    class _Path:
        def __init__(self, key):
            self.key = key

        def exists(self):
            return self.key == 'US-NC'

        def __str__(self):
            return f'<{self.key}>'

    monkeypatch.setattr(
        readers,
        'get_output_path',
        lambda recipe, admin_id=None, **kw: _Path(str(admin_id)),
    )

    def fake_read_parquet(path, geom=False, columns=None, filters=None, **kwargs):
        if columns == []:
            return frame[[]]
        return frame if geom else frame.drop(columns='geometry')

    monkeypatch.setattr(readers, 'read_parquet', fake_read_parquet)
    return frame


class TestGeometrySurvivesAnEmptyColumn:
    def test_geom_true_still_returns_a_geodataframe(self, county_file_without_geometry):
        # Dropping an all-null geometry column hands a geom=True caller a
        # plain DataFrame, and every caller that reaches for .geometry
        # then raises instead of reporting the absent geometry.
        admin = get_admin('US-NC-ALP', 3, recipe=RECIPE_ID, geom=True)
        assert 'geometry' in admin
        assert isinstance(admin, gpd.GeoDataFrame)

    def test_geom_false_still_drops_it(self, county_file_without_geometry):
        admin = get_admin('US-NC-ALP', 3, recipe=RECIPE_ID, geom=False)
        assert 'geometry' not in admin


class TestProgressReporting:
    def test_reporting_inferred_ids_does_not_raise(self, capsys):
        # A str + list concatenation used to raise TypeError here.
        get_admin(['US-NC-WAK', 'US-NC-WAR'], 2, silent=False)
        assert 'Inferred Admin IDs' in capsys.readouterr().out


class TestRegionMembers:
    def test_a_blank_member_cell_is_not_a_member(self, monkeypatch):
        # The registry is read with keep_default_na=False, so a blank
        # cell is an empty string rather than NaN.
        rows = pd.DataFrame(
            {
                'region_id': ['r', 'r', 'r'],
                'admin_id': ['US-NC-WAK', '', 'US-NC-WAR'],
            }
        )
        monkeypatch.setattr(readers, 'get_regions', lambda region_id=None: rows)
        assert get_region_admin_ids('r') == ['US-NC-WAK', 'US-NC-WAR']
