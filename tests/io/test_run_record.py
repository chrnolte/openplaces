"""Every stage output says how many rows each step received and passed
on: the run record in its footer, and one log line per step that
changed the count.

All tables are fabricated.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.curator as cur
from openplaces.core.schema import AdminId, Entity
from openplaces.flow import run_record as run_record_cli
from openplaces.io.curator import _STEP_REGISTRY, Curator
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.io.readers import run_record
from openplaces.io.steps import RowTrail, recipe_digest
from openplaces.io.tables import (
    METADATA,
    RUN_RECORD_KEY,
    file_digests,
    same_table,
    save_parquet,
    to_parquet,
)
from openplaces.recipe import _add_stacked_units_layer
from openplaces.timing import Timer


def _frame():
    return pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})


def _record(rows_after):
    trail = RowTrail('curate', rows_read=3)
    trail.after('step 00 keep', 3)
    trail.after('step 01 drop', rows_after)
    return trail.record('XX_thing-src-2026', 'XX-AA', {'pipeline': []})


def test_the_record_is_written_and_read_back(tmp_path):
    path = tmp_path / 'out.parquet'
    save_parquet(_frame(), path, run_record=_record(2))
    record = run_record(path)
    assert record['stage'] == 'curate'
    assert record['recipe_id'] == 'XX_thing-src-2026'
    assert record['admin_id'] == 'XX-AA'
    assert record['steps'] == [['read', 3], ['step 00 keep', 3], ['step 01 drop', 2]]
    assert len(record['recipe_digest']) == 12
    assert 'commit' in record and 'version' in record


def test_the_record_is_written_beside_a_geometry_sidecar(tmp_path):
    path = tmp_path / 'geo.parquet'
    gdf = gpd.GeoDataFrame(_frame(), geometry=[box(0, 0, 1, 1)] * 3, crs='EPSG:4326')
    save_parquet(gdf, path, run_record=_record(3))
    assert run_record(path)['steps'][-1] == ['step 01 drop', 3]


def test_a_file_without_the_key_reads_as_none(tmp_path):
    path = tmp_path / 'old.parquet'
    to_parquet(_frame(), path)
    assert run_record(path) is None
    assert run_record(tmp_path / 'missing.parquet') is None


def test_two_runs_writing_the_same_table_compare_as_the_same_file(tmp_path):
    first, second = tmp_path / 'first.parquet', tmp_path / 'second.parquet'
    save_parquet(_frame(), first, run_record=_record(2))
    save_parquet(_frame(), second, run_record=_record(1))
    assert run_record(first) != run_record(second)
    # What a job record compares (flow.job_record.file_state): every
    # column's digest, the schema, and the footer apart from keys that
    # describe the run rather than the table.
    assert file_digests(first) == file_digests(second)
    assert METADATA in file_digests(first)
    # What `--keep-if-equal` compares.
    assert same_table(first, second)


def test_the_footer_metadata_digest_still_sees_other_keys(tmp_path):
    first, second = tmp_path / 'first.parquet', tmp_path / 'second.parquet'
    save_parquet(_frame(), first, file_metadata={'openplaces:x': '1'})
    save_parquet(_frame(), second, file_metadata={'openplaces:x': '2'})
    assert file_digests(first)[METADATA] != file_digests(second)[METADATA]
    assert RUN_RECORD_KEY == 'openplaces:run_record'


def test_a_step_that_changes_the_count_prints_one_line(capsys):
    trail = RowTrail('curate', rows_read=5971)
    trail.after('step 00 keep', 5971)
    trail.after('step 01 merge', 5725)
    trail.after('step 02 keep', 5725)
    lines = capsys.readouterr().out.splitlines()
    assert lines == ['  step 01 merge: rows 5,971 to 5,725']


def test_a_spine_not_built_yet_records_no_count(capsys):
    trail = RowTrail('harmonize')
    trail.after('step 00 resolve_spine', None)
    trail.after('step 01 load', 10)
    trail.after('step 02 drop', 9)
    assert trail.steps == [
        ['step 00 resolve_spine', None],
        ['step 01 load', 10],
        ['step 02 drop', 9],
    ]
    assert capsys.readouterr().out.splitlines() == ['  step 02 drop: rows 10 to 9']


def test_the_recipe_digest_ignores_key_and_set_order():
    a = {'b': 1, 'a': [1, {'y': 2, 'x': {'q', 'p'}}], 'e': Entity('parcel', 's', '1')}
    b = {'e': Entity('parcel', 's', '1'), 'a': [1, {'x': {'p', 'q'}, 'y': 2}], 'b': 1}
    assert recipe_digest(a) == recipe_digest(b)
    assert recipe_digest(a) != recipe_digest({**a, 'b': 2})


def _drop_one(state):
    state.curated = state.curated.iloc[1:]
    return state


def _keep(state):
    return state


def test_a_curate_step_that_drops_rows_is_logged_and_recorded(monkeypatch, capsys):
    monkeypatch.setitem(_STEP_REGISTRY, 't_keep', _keep)
    monkeypatch.setitem(_STEP_REGISTRY, 't_drop_one', _drop_one)
    saved = {}

    def fake_save(frame, path, **kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(cur, 'get_entities', lambda *a, **k: _frame())
    monkeypatch.setattr(cur, 'save_parquet', fake_save)
    monkeypatch.setattr(cur, 'get_output_path', lambda *a, **k: 'unused')
    curator = Curator.__new__(Curator)
    curator.recipe = {
        'recipe_id': 'XX_thing-src-2026',
        'pipeline': [{'step': 't_keep'}, {'step': 't_drop_one'}],
    }
    curator.entity_recipe = {}
    curator.verbose = False
    curator.skip_steps = set()
    curator.save_statistics = False
    timer = Timer('Curator')
    curator._timer = timer
    curator._curate_one(AdminId('US'))

    assert '  step 01 t_drop_one: rows 3 to 2' in capsys.readouterr().out
    record = saved['run_record']
    assert record['steps'] == [
        ['read', 3],
        ['step 00 t_keep', 3],
        ['step 01 t_drop_one', 2],
    ]
    marks = {r['label']: r.get('metadata') for r in timer.records}
    assert marks['step 01 t_drop_one'] == {'rows': 2}


def _parcel_ingester():
    recipe = {
        'stage': 'ingest',
        'recipe_id': 'XX_parcel-src-2026',
        'entity': Entity('parcel', 'src', '2026'),
        'save_to': {'data_dir': 'core'},
    }
    _add_stacked_units_layer(recipe)
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    ingester.processing_chunk = {'admin_id_to_process': 'XX-AA'}
    ingester.download_partition = {}
    ingester.verbose = False
    ingester.timer = Timer('test')
    return ingester


def test_the_stacked_units_line_is_printed_when_rows_merge_without_stacks(capsys):
    # The same record shipped twice on one lot: merged, and no stack.
    gdf = gpd.GeoDataFrame(
        {'parcel_id_local': ['a', 'a', 'b'], 'land_value': [1.0, 1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs='EPSG:4326',
    )
    ingester = _parcel_ingester()
    ingester._note_read(len(gdf))
    gdf = ingester._preprocess_recipe_data(gdf)
    parcels, properties = ingester._split_stacked_units(gdf)
    out = capsys.readouterr().out
    assert properties is None and len(parcels) == 2
    assert 'stacked units: rows 3 to 2; ' in out
    assert '0 stacks' in out
    # Counted once: the split's own line, no second generic one.
    assert out.count('rows 3 to 2') == 1
    record = ingester._run_record(ingester.recipe, 'XX-AA', parcels)
    assert record['stage'] == 'ingest'
    assert record['steps'][0] == ['read', 3]
    assert record['steps'][-2:] == [['stacked units', 2], ['saved', 2]]


def test_an_ingest_query_that_drops_rows_prints_one_line(capsys):
    ingester = _parcel_ingester()
    ingester.recipe['query'] = 'land_value > 1'
    gdf = gpd.GeoDataFrame(
        {'parcel_id_local': ['a', 'b'], 'land_value': [1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)],
        crs='EPSG:4326',
    )
    ingester._note_read(len(gdf))
    ingester._preprocess_recipe_data(gdf)
    out = capsys.readouterr().out
    assert 'query: rows 2 to 1' in out


def test_the_command_prints_the_steps(tmp_path, capsys):
    path = tmp_path / 'out.parquet'
    save_parquet(_frame(), path, run_record=_record(2))
    assert run_record_cli.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert 'step 01 drop' in out and '-1' in out
    old = tmp_path / 'old.parquet'
    to_parquet(_frame(), old)
    assert run_record_cli.main([str(old)]) == 1
