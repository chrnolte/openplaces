"""Tests for the pinned CRS-transform registry (geo.crs_transforms) and reproject()."""

from types import SimpleNamespace

import geopandas as gpd
import pyproj
import pytest
from shapely.geometry import Point

from openplaces.geo import crs_transforms as ct
from openplaces.geo.polygon import reproject


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Redirect the registry/grid directories so tests never touch the real ones."""
    monkeypatch.setattr(ct, 'REGISTRY_DIR', tmp_path / 'crs_transforms')
    monkeypatch.setattr(ct, 'GRIDS_DIR', tmp_path / 'crs_grids')
    return tmp_path


def _fake_transformer(accuracy, description, definition):
    return SimpleNamespace(
        accuracy=accuracy, description=description, definition=definition
    )


def test_crs_slug_prefers_authority_code():
    assert ct._crs_slug('EPSG:26986') == 'EPSG-26986'
    assert ct._crs_slug('EPSG:4326') == 'EPSG-4326'


# A real-world NC State Plane (ftUS) definition (as parsed from a source .prj) whose
# false-easting constant is rounded a hair differently from the canonical EPSG:2264
# definition -- enough to drop `to_authority()`'s default-confidence match to nothing.
_NC_FTUS_ESRI_VARIANT_WKT = (
    'PROJCRS["NAD83 / North Carolina (ftUS)",'
    'BASEGEOGCRS["NAD83",'
    'DATUM["North American Datum 1983",'
    'ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]],'
    'ID["EPSG",6269]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["Degree",0.0174532925199433]]],'
    'CONVERSION["unnamed",'
    'METHOD["Lambert Conic Conformal (2SP)",ID["EPSG",9802]],'
    'PARAMETER["Latitude of false origin",33.75,'
    'ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8821]],'
    'PARAMETER["Longitude of false origin",-79,'
    'ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8822]],'
    'PARAMETER["Latitude of 1st standard parallel",34.3333333333333,'
    'ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],'
    'PARAMETER["Latitude of 2nd standard parallel",36.1666666666667,'
    'ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8824]],'
    'PARAMETER["Easting at false origin",2000000.00261667,'
    'LENGTHUNIT["US survey foot",0.304800609601219],ID["EPSG",8826]],'
    'PARAMETER["Northing at false origin",0,'
    'LENGTHUNIT["US survey foot",0.304800609601219],ID["EPSG",8827]]],'
    'CS[Cartesian,2],'
    'AXIS["easting",east,ORDER[1],'
    'LENGTHUNIT["US survey foot",0.304800609601219,ID["EPSG",9003]]],'
    'AXIS["northing",north,ORDER[2],'
    'LENGTHUNIT["US survey foot",0.304800609601219,ID["EPSG",9003]]]]'
)


def test_crs_slug_resolves_near_match_to_epsg_code():
    assert ct._crs_slug(_NC_FTUS_ESRI_VARIANT_WKT) == 'EPSG-2264'


def test_crs_slug_does_not_guess_for_ambiguous_ad_hoc_crs():
    # Same proj4 template as local_metric_crs() in geo/polygon.py. It reports
    # name 'unknown' and matches several unrelated ESRI world-projection codes at
    # low confidence, so it must fall back to the sanitized-name slug rather than
    # picking one arbitrarily -- otherwise unrelated local projections centered on
    # different datasets would collide onto the same registry filename.
    proj4 = (
        '+proj=aeqd +lat_0=42.4 +lon_0=-71.1 +x_0=0 +y_0=0 +datum=WGS84 '
        '+units=m +no_defs'
    )
    slug = ct._crs_slug(proj4)
    assert not slug.startswith('ESRI-')
    assert not slug.startswith('EPSG-')
    assert slug == 'unknown'


def test_registry_round_trip(isolated_registry):
    entry = {
        'source_crs': 'EPSG:26986',
        'target_crs': 'EPSG:4326',
        'operation_name': 'fake op',
        'operation_definition': 'proj=pipeline ...',
        'accuracy_m': 2.0,
        'grid_files': [],
        'registered': '2026-01-01',
    }
    assert ct.load_crs_transform('EPSG:26986', 'EPSG:4326') is None
    ct._write_registry_entry(ct._registry_path('EPSG:26986', 'EPSG:4326'), entry)
    assert ct.load_crs_transform('EPSG:26986', 'EPSG:4326') == entry


def test_resolve_picks_lowest_accuracy_and_downloads_grids(
    isolated_registry, monkeypatch
):
    grid_free = _fake_transformer(4.0, 'z-ballpark', 'proj=pipeline no grid step')
    grid_based = _fake_transformer(
        2.0, 'a-accurate', 'proj=pipeline step proj=hgridshift grids=fake_grid.tif step'
    )
    fake_group = SimpleNamespace(
        transformers=[grid_free, grid_based], unavailable_operations=[]
    )
    monkeypatch.setattr(
        ct.pyproj.transformer,
        'TransformerGroup',
        lambda *a, **kw: fake_group,
    )
    monkeypatch.setattr(ct, '_download_grid', lambda filename: None)
    monkeypatch.setattr(ct.pyproj.network, 'set_network_enabled', lambda enabled: None)

    entry = ct.resolve_crs_transform('EPSG:26986', 'EPSG:4326')

    assert entry['operation_name'].startswith('a-accurate')
    assert entry['accuracy_m'] == 2.0
    assert entry['grid_files'] == ['fake_grid.tif']
    assert 'grids=fake_grid.tif,@null' in entry['operation_definition']
    assert ct.load_crs_transform('EPSG:26986', 'EPSG:4326') == entry


def test_resolve_exact_grid_free_operation_is_not_persisted(
    isolated_registry, monkeypatch
):
    # e.g. WGS84 to Web Mercator: a pure formula, no datum grid involved, so
    # there is nothing environment-dependent to pin against.
    exact = _fake_transformer(0.0, 'webmerc-inverse', 'proj=pipeline no grid step')
    fake_group = SimpleNamespace(transformers=[exact], unavailable_operations=[])
    monkeypatch.setattr(
        ct.pyproj.transformer, 'TransformerGroup', lambda *a, **kw: fake_group
    )
    monkeypatch.setattr(ct, '_download_grid', lambda filename: None)
    monkeypatch.setattr(ct.pyproj.network, 'set_network_enabled', lambda enabled: None)

    entry = ct.resolve_crs_transform('EPSG:3857', 'EPSG:4326')

    assert entry['accuracy_m'] == 0.0
    assert entry['grid_files'] == []
    assert ct.load_crs_transform('EPSG:3857', 'EPSG:4326') is None


def test_pipeline_template_blanks_grid_filename():
    a = 'proj=pipeline step proj=hgridshift grids=us_noaa_nehpgn.tif step'
    b = 'proj=pipeline step proj=hgridshift grids=us_noaa_gahpgn.tif step'
    assert ct._pipeline_template(a) == ct._pipeline_template(b)


def test_resolve_merges_same_template_region_grids(isolated_registry, monkeypatch):
    # Two candidates sharing a pipeline template (differ only by grids=) must be
    # merged into one multi-region-safe operation instead of only the single
    # best-rated one being kept -- picking just one silently returns inf for any
    # point outside that one grid's own area-of-use (the EPSG:5070 incident: its
    # single best-rated candidate is a Canadian grid that covers no US state).
    new_england = _fake_transformer(
        2.0,
        'a-new-england',
        'proj=pipeline step proj=hgridshift grids=us_noaa_nehpgn.tif step',
    )
    georgia = _fake_transformer(
        2.0,
        'b-georgia',
        'proj=pipeline step proj=hgridshift grids=us_noaa_gahpgn.tif step',
    )
    unrelated_ballpark = _fake_transformer(4.0, 'c-ballpark', 'proj=pipeline step')
    fake_group = SimpleNamespace(
        transformers=[georgia, new_england, unrelated_ballpark],
        unavailable_operations=[],
    )
    monkeypatch.setattr(
        ct.pyproj.transformer, 'TransformerGroup', lambda *a, **kw: fake_group
    )
    monkeypatch.setattr(ct, '_download_grid', lambda filename: None)
    monkeypatch.setattr(ct.pyproj.network, 'set_network_enabled', lambda enabled: None)

    entry = ct.resolve_crs_transform('EPSG:5070', 'EPSG:4326')

    assert entry['grid_files'] == ['us_noaa_gahpgn.tif', 'us_noaa_nehpgn.tif']
    assert (
        'grids=us_noaa_gahpgn.tif,us_noaa_nehpgn.tif' in entry['operation_definition']
    )
    assert 'merged with 1' in entry['operation_name']
    # A point outside both merged grids must not resolve to `inf` (the
    # EPSG:4269/Alaska incident: `unrelated_ballpark` here plays the role of
    # Alaska's structurally different, grid-free `proj=noop` candidate -- it
    # doesn't share the hgridshift template, so it's excluded from the merge
    # by design, but the merged operation must still fall back to identity
    # for points neither real grid covers, instead of returning inf.
    assert entry['operation_definition'].count('@null') == 1
    assert (
        'grids=us_noaa_gahpgn.tif,us_noaa_nehpgn.tif,@null'
        in (entry['operation_definition'])
    )


def test_resolve_raises_when_nothing_available(isolated_registry, monkeypatch):
    fake_group = SimpleNamespace(transformers=[], unavailable_operations=[object()])
    monkeypatch.setattr(
        ct.pyproj.transformer, 'TransformerGroup', lambda *a, **kw: fake_group
    )
    monkeypatch.setattr(ct.pyproj.network, 'set_network_enabled', lambda enabled: None)

    with pytest.raises(ValueError, match='No usable reprojection'):
        ct.resolve_crs_transform('EPSG:26986', 'EPSG:4326')


def test_get_or_resolve_reuses_existing_entry(isolated_registry, monkeypatch):
    entry = {
        'source_crs': 'EPSG:26986',
        'target_crs': 'EPSG:4326',
        'operation_name': 'cached',
        'operation_definition': 'proj=pipeline ...',
        'accuracy_m': 2.0,
        'grid_files': [],
        'registered': '2026-01-01',
    }
    ct._write_registry_entry(ct._registry_path('EPSG:26986', 'EPSG:4326'), entry)

    def _boom(*a, **kw):
        raise AssertionError('resolve_crs_transform should not be called')

    monkeypatch.setattr(ct, 'resolve_crs_transform', _boom)
    assert ct.get_or_resolve_crs_transform('EPSG:26986', 'EPSG:4326') == entry


def test_reproject_pinned_false_matches_plain_to_crs():
    gdf = gpd.GeoDataFrame({'x': [1]}, geometry=[Point(0, 0)], crs='EPSG:4326')
    out = reproject(gdf, 'EPSG:3857', pinned=False)
    expected = gdf.to_crs('EPSG:3857')
    assert out.geometry.iloc[0].equals_exact(expected.geometry.iloc[0], tolerance=1e-9)


def test_reproject_same_crs_is_a_no_op(isolated_registry, monkeypatch):
    def _boom(*a, **kw):
        raise AssertionError('get_or_resolve_crs_transform should not be called')

    monkeypatch.setattr('openplaces.geo.polygon.get_or_resolve_crs_transform', _boom)
    gdf = gpd.GeoDataFrame({'x': [1]}, geometry=[Point(0, 0)], crs='EPSG:4326')
    out = reproject(gdf, 'EPSG:4326')
    assert out.geometry.iloc[0] == gdf.geometry.iloc[0]


def test_reproject_pinned_uses_registered_pipeline(isolated_registry):
    # Register EPSG:4326 -> EPSG:3857 using pyproj's own bare Transformer.from_crs
    # pipeline definition (a projection change only, no datum grid needed) so the
    # test stays hermetic while still exercising the real Transformer.from_pipeline
    # code path used at read time.
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    entry = {
        'source_crs': 'EPSG:4326',
        'target_crs': 'EPSG:3857',
        'operation_name': transformer.description,
        'operation_definition': transformer.definition,
        'accuracy_m': transformer.accuracy,
        'grid_files': [],
        'registered': '2026-01-01',
    }
    ct._write_registry_entry(ct._registry_path('EPSG:4326', 'EPSG:3857'), entry)

    gdf = gpd.GeoDataFrame({'x': [1]}, geometry=[Point(-71.1, 42.4)], crs='EPSG:4326')
    out = reproject(gdf, 'EPSG:3857')
    expected = gdf.to_crs('EPSG:3857')
    assert out.geometry.iloc[0].equals_exact(expected.geometry.iloc[0], tolerance=1e-6)
    assert str(out.crs) == str(pyproj.CRS('EPSG:3857'))


def test_reproject_auto_registers_unknown_pair(isolated_registry, monkeypatch):
    calls = []

    def _tracking_resolve(source_crs, target_crs='EPSG:4326'):
        calls.append((str(source_crs), str(target_crs)))
        transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )
        entry = {
            'source_crs': pyproj.CRS(source_crs).to_string(),
            'target_crs': pyproj.CRS(target_crs).to_string(),
            'operation_name': transformer.description,
            'operation_definition': transformer.definition,
            'accuracy_m': transformer.accuracy,
            'grid_files': [],
            'registered': '2026-01-01',
        }
        ct._write_registry_entry(ct._registry_path(source_crs, target_crs), entry)
        return entry

    monkeypatch.setattr(ct, 'resolve_crs_transform', _tracking_resolve)

    gdf = gpd.GeoDataFrame({'x': [1]}, geometry=[Point(-71.1, 42.4)], crs='EPSG:4326')
    out = reproject(gdf, 'EPSG:3857')

    assert calls == [('EPSG:4326', 'EPSG:3857')]
    assert ct.load_crs_transform('EPSG:4326', 'EPSG:3857') is not None
    expected = gdf.to_crs('EPSG:3857')
    assert out.geometry.iloc[0].equals_exact(expected.geometry.iloc[0], tolerance=1e-6)
