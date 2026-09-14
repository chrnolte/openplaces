"""A unit the source does not publish gets an empty table, not no file.

The fmv parcel source has not published Tyrrell County, so all three of
its table partitions report 'not published' and the ingest soft-skips
them. The job then succeeds, exits 0 and writes nothing, which reads to
an orchestrator exactly like a failed job: it demands the declared
output, fails the job, and fails everything downstream of it.

The harmonize stage records the same kind of empty answer, in
`tests/io/harmonizer/test_empty_spine_is_recorded.py`.
"""

import geopandas as gpd
import pytest
from shapely.geometry import box

import openplaces.io.ingester as ingester_module
from openplaces.core.schema import AdminId
from openplaces.io import read_parquet, save_parquet

ADMIN_IDS = [AdminId('US-NC-TYR'), AdminId('US-NC-CAM')]


def _ingester(monkeypatch, tmp_path, admin_ids):
    """An Ingester stub whose outputs land in tmp_path."""
    monkeypatch.setattr(
        ingester_module,
        'get_output_path',
        lambda recipe, admin_id, *a, **k: tmp_path / f'{admin_id}.parquet',
    )
    ingester = ingester_module.Ingester.__new__(ingester_module.Ingester)
    ingester.recipe = {'recipe_id': 'US_parcel-placeslab-fmv2026'}
    ingester.admin_ids_to_save = admin_ids
    ingester.verbose = False
    return ingester


def test_a_unit_with_no_output_gets_an_empty_table(monkeypatch, tmp_path):
    ingester = _ingester(monkeypatch, tmp_path, ADMIN_IDS)

    ingester._record_units_the_source_does_not_cover()

    for admin_id in ADMIN_IDS:
        out_path = tmp_path / f'{admin_id}.parquet'
        assert out_path.exists(), admin_id
        assert len(read_parquet(out_path)) == 0


def test_a_unit_the_source_did_cover_is_left_alone(monkeypatch, tmp_path):
    """Only a missing output is filled in; real data is never touched."""
    ingester = _ingester(monkeypatch, tmp_path, ADMIN_IDS)
    covered = tmp_path / 'US-NC-CAM.parquet'
    save_parquet(
        gpd.GeoDataFrame({'geometry': [box(0, 0, 1, 1)]}, crs='epsg:4326'), covered
    )
    before = covered.read_bytes()

    ingester._record_units_the_source_does_not_cover()

    assert covered.read_bytes() == before
    assert len(read_parquet(covered)) == 1
    # The unit that really is uncovered still gets its empty table.
    assert len(read_parquet(tmp_path / 'US-NC-TYR.parquet')) == 0


def test_no_units_to_save_writes_nothing(monkeypatch, tmp_path):
    """A recipe saving nothing per unit must not invent files."""
    ingester = _ingester(monkeypatch, tmp_path, [])

    ingester._record_units_the_source_does_not_cover()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('admin_ids', [None, []])
def test_an_unset_save_list_is_tolerated(monkeypatch, tmp_path, admin_ids):
    """`admin_ids_to_save` is None before resolution and empty after."""
    ingester = _ingester(monkeypatch, tmp_path, admin_ids)

    ingester._record_units_the_source_does_not_cover()

    assert list(tmp_path.iterdir()) == []


def test_a_unit_that_became_unavailable_is_emptied_on_reprocess(monkeypatch, tmp_path):
    """A stale output from an earlier run must not read as current."""
    ingester = _ingester(monkeypatch, tmp_path, ADMIN_IDS)
    ingester._unavailable_units = {'US-NC-CAM'}
    stale = tmp_path / 'US-NC-CAM.parquet'
    save_parquet(
        gpd.GeoDataFrame({'geometry': [box(0, 0, 1, 1)]}, crs='epsg:4326'), stale
    )

    ingester._record_units_the_source_does_not_cover(reprocess=False)
    assert len(read_parquet(stale)) == 1

    ingester._record_units_the_source_does_not_cover(reprocess=True)
    assert len(read_parquet(stale)) == 0
