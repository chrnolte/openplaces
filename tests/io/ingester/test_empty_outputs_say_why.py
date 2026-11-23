"""Every empty table an ingest writes says why, and one that does not
is not a build.

A stale empty parcel table, written before an empty download raised,
read as "built": a later run skipped the unit ("already have outputs")
and every downstream table was built from no rows without an error.
Each writer of an empty ingest table now stamps a reason in the parquet
footer (`openplaces:empty_reason`), and a table with no rows and no
reason is ingested again, and raises where a consumer would read it as
"none". Every file here is fabricated by the test.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.ingester.outputs as outputs_module
import openplaces.io.ingester.scope as scope_module
import openplaces.io.ingester.table_save as table_save_module
from openplaces.core.schema import AdminId, Entity
from openplaces.flow.dag import needs_rerun
from openplaces.io import save_parquet
from openplaces.io.aggregate import _aggregate_to_file, join_partitions_by_index
from openplaces.io.ingester import Ingester
from openplaces.io.ingester.empty_outputs import (
    UnexplainedEmptyIngestError,
    save_empty_table,
)
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.io.readers import (
    EMPTY_DETAIL_KEY,
    empty_reason,
    is_unexplained_empty,
)


def _detail(path):
    import pyarrow.parquet as pq

    return pq.read_metadata(path).metadata[EMPTY_DETAIL_KEY.encode()].decode()


def _reasonless_empty(path):
    """An empty table as ingests wrote it before reasons existed."""
    save_parquet(
        gpd.GeoDataFrame({'geometry': gpd.GeoSeries([], dtype='geometry')}, crs=4326),
        path,
    )


def _with_rows(path, n=1):
    save_parquet(
        gpd.GeoDataFrame(
            {'value': list(range(n))}, geometry=[box(0, 0, 1, 1)] * n, crs=4326
        ),
        path,
    )


class _Timer:
    def mark(self, *args, **kwargs):
        pass

    def finish(self):
        pass


# Reader


def test_reader_tells_reasoned_reasonless_and_full_tables_apart(tmp_path):
    reasoned = tmp_path / 'a.parquet'
    save_empty_table(reasoned, 'not_published', 'fabricated')
    reasonless = tmp_path / 'b.parquet'
    _reasonless_empty(reasonless)
    full = tmp_path / 'c.parquet'
    _with_rows(full)
    text = tmp_path / 'd.parquet'
    text.write_text('not parquet')

    assert empty_reason(reasoned) == 'not_published'
    assert not is_unexplained_empty(reasoned)
    assert empty_reason(reasonless) is None and is_unexplained_empty(reasonless)
    assert empty_reason(full) is None and not is_unexplained_empty(full)
    assert not is_unexplained_empty(tmp_path / 'missing.parquet')
    assert not is_unexplained_empty(text)


def test_save_empty_table_refuses_rows_and_unknown_reasons(tmp_path):
    with pytest.raises(ValueError, match='Unknown'):
        save_empty_table(tmp_path / 'x.parquet', 'because', 'no')
    with pytest.raises(ValueError, match='rows'):
        save_empty_table(
            tmp_path / 'x.parquet', 'not_covered', 'no', frame=pd.DataFrame({'a': [1]})
        )


# Writer 1: the after-run fill


def _filler(monkeypatch, tmp_path, admin_ids, recipe=None):
    monkeypatch.setattr(
        outputs_module,
        'get_output_path',
        lambda recipe, admin_id, *a, **k: tmp_path / f'{admin_id}.parquet',
    )
    ing = Ingester.__new__(Ingester)
    ing.recipe = recipe or {
        'admin_id': AdminId('US-ZZ'),
        'entity': Entity('parcel', 'example', '2026'),
    }
    ing.admin_ids_to_save = admin_ids
    ing.verbose = False
    return ing


def test_fill_stamps_not_published(monkeypatch, tmp_path):
    ing = _filler(monkeypatch, tmp_path, ['US-ZZ-AAA'])
    ing._unavailable_units = {'US-ZZ'}

    ing._record_units_the_source_does_not_cover()

    out = tmp_path / 'US-ZZ-AAA.parquet'
    assert empty_reason(out) == 'not_published'
    assert 'US-ZZ' in _detail(out)


def test_fill_stamps_marker_for_a_partitioned_base(monkeypatch, tmp_path):
    recipe = {
        'admin_id': AdminId('US-ZZ'),
        'entity': Entity('transaction', 'example', '2026'),
        'download_by': {'partition': 'year'},
        'aggregate_by': {'single_file': True},
    }
    ing = _filler(monkeypatch, tmp_path, ['US-ZZ-AAA'], recipe)

    ing._record_units_the_source_does_not_cover()

    out = tmp_path / 'US-ZZ-AAA.parquet'
    assert empty_reason(out) == 'marker'
    assert '_all' in _detail(out)


def test_fill_uses_the_reason_a_writer_recorded(monkeypatch, tmp_path):
    ing = _filler(monkeypatch, tmp_path, ['US-ZZ-AAA'])
    ing._empty_reasons = {'US-ZZ-AAA': ('not_covered', '5 rows, none for AAA')}

    ing._record_units_the_source_does_not_cover()

    assert empty_reason(tmp_path / 'US-ZZ-AAA.parquet') == 'not_covered'


def test_fill_refuses_an_absence_nothing_explains(monkeypatch, tmp_path):
    ing = _filler(monkeypatch, tmp_path, ['US-ZZ-AAA'])

    with pytest.raises(UnexplainedEmptyIngestError, match='US-ZZ-AAA'):
        ing._record_units_the_source_does_not_cover()
    assert not (tmp_path / 'US-ZZ-AAA.parquet').exists()


def test_fill_stamps_a_declined_unit(monkeypatch, tmp_path):
    ing = _filler(monkeypatch, tmp_path, ['US-ZZ-AAA'])
    ing.download_partition = {'admin_id_to_download': 'US-ZZ'}
    ing._record_declined('the usage profile (usage_requirement)')

    ing._record_units_the_source_does_not_cover()

    out = tmp_path / 'US-ZZ-AAA.parquet'
    assert empty_reason(out) == 'declined'
    # The key name only: no profile or config value.
    assert _detail(out) == (
        'the download was declined by the usage profile (usage_requirement)'
    )


def test_a_withheld_terms_consent_is_declined_not_unpublished(monkeypatch, tmp_path):
    from openplaces.io.consent import require_terms_consent

    def fetch(**kwargs):
        if not require_terms_consent('Fabricated portal', accept_terms=False):
            return None

    ing = _filler(monkeypatch, tmp_path, ['US-ZZ-AAA'])
    ing.timer = _Timer()
    ing.admin_ids_to_download = ['US-ZZ']
    ing.download_partition = {
        'admin_id_to_download': 'US-ZZ',
        'partition_id_to_download': None,
        'data_path': tmp_path / 'never.csv',
        'downloaded_path': None,
    }
    monkeypatch.setattr(Ingester, '_load_scraper_fetch', staticmethod(lambda n: fetch))

    ing._run_download_scraper('fabricated_scraper')
    ing._record_units_the_source_does_not_cover()

    out = tmp_path / 'US-ZZ-AAA.parquet'
    assert empty_reason(out) == 'declined'
    assert 'accept_terms' in _detail(out)
    assert 'Fabricated portal' not in _detail(out)
    assert not getattr(ing, '_unavailable_units', set())


def test_a_declined_table_reads_as_none_for_a_consumer(monkeypatch, tmp_path):
    out = tmp_path / 'US-ZZ-AAA.parquet'
    save_empty_table(out, 'declined', 'the download was declined by x')
    recipe = {
        'admin_id': AdminId('US-ZZ'),
        'entity': Entity('parcel', 'example', '2026'),
    }
    monkeypatch.setattr(scope_module, 'get_output_path', lambda *a, **k: out)
    monkeypatch.setattr(
        scope_module,
        '_get_output_admin_ids',
        lambda recipe, admin_id: ([AdminId('US-ZZ-AAA')], [], []),
    )
    monkeypatch.setattr(scope_module, 'crosswalk_coverage', lambda recipe: None)

    assert scope_module.unbuilt_outputs(recipe, 'US-ZZ-AAA') == []


def test_fill_replaces_a_reasonless_empty_table(monkeypatch, tmp_path):
    ing = _filler(monkeypatch, tmp_path, ['US-ZZ-AAA'])
    ing._unavailable_units = {'US-ZZ-AAA'}
    _reasonless_empty(tmp_path / 'US-ZZ-AAA.parquet')

    ing._record_units_the_source_does_not_cover()

    assert empty_reason(tmp_path / 'US-ZZ-AAA.parquet') == 'not_published'


# Writers 2, 3 and 5: a processed chunk, a split and the stacked layer


def _table(tmp_path, monkeypatch, *, recipe=None, partition=None, unit='US-ZZ-AAA'):
    monkeypatch.setattr(
        table_save_module,
        'get_output_path',
        lambda recipe, admin_id, *a, **k: tmp_path / f'{admin_id}_t.parquet',
    )
    table = TableIngester(
        recipe or {'admin_id': 'US-ZZ', 'entity': 'parcel-example-2026'},
        {'partition_id_to_download': None, 'empty_reasons': {}, **(partition or {})},
        {'admin_id_to_process': unit},
        tmp_path,
        _Timer(),
        admin_ids_to_save=[unit],
    )
    table.is_primary = True
    return table


EMPTY = gpd.GeoDataFrame({'value': []}, geometry=[], crs=4326)


def test_chunk_whose_unit_filter_matched_nothing_is_not_covered(tmp_path, monkeypatch):
    recipe = {
        'admin_id': 'US-ZZ',
        'entity': 'parcel-example-2026',
        'process_by': {'admin_level': 3, 'admin_id_column': 'COUNTY'},
    }
    fids = pd.Series(['US-ZZ-BBB'] * 7)
    reverse = pd.Series(['007'], index=['US-ZZ-AAA'], name='code')
    table = _table(
        tmp_path,
        monkeypatch,
        recipe=recipe,
        partition={
            'table_fids': {'parcel-example-2026': fids},
            'admin_id_crosswalk_reverse': reverse,
        },
    )
    table._note_read(0, {'fids': []})

    table._save_recipe_data(EMPTY)

    out = tmp_path / 'US-ZZ-AAA_t.parquet'
    assert empty_reason(out) == 'not_covered'
    assert '7 rows' in _detail(out) and 'COUNTY' in _detail(out)
    assert "'007'" in _detail(out)


def test_chunk_emptied_by_preprocessing_is_not_covered(tmp_path, monkeypatch):
    table = _table(tmp_path, monkeypatch)
    table._note_read(12)

    table._save_recipe_data(EMPTY)

    out = tmp_path / 'US-ZZ-AAA_t.parquet'
    assert empty_reason(out) == 'not_covered'
    assert '12 rows read' in _detail(out)


def test_chunk_of_an_empty_source_is_source_has_none(tmp_path, monkeypatch):
    table = _table(tmp_path, monkeypatch)
    table.refuse_empty_read = True
    table._note_read(0)

    table._save_recipe_data(EMPTY)

    out = tmp_path / 'US-ZZ-AAA_t.parquet'
    assert empty_reason(out) == 'source_has_none'
    assert 'empty_download_ok' in _detail(out)


def test_empty_save_without_a_noted_read_raises(tmp_path, monkeypatch):
    table = _table(tmp_path, monkeypatch)

    with pytest.raises(UnexplainedEmptyIngestError):
        table._save_recipe_data(EMPTY)


def test_split_records_a_reason_for_every_unit_it_cannot_save(tmp_path, monkeypatch):
    recipe = {
        'admin_id': 'US-ZZ',
        'entity': 'parcel-example-2026',
        'save_to': {'admin_level': 3},
    }
    table = _table(tmp_path, monkeypatch, recipe=recipe, unit='US-ZZ')
    table.admin_ids_to_save = ['US-ZZ-AAA', 'US-ZZ-BBB']
    rows = gpd.GeoDataFrame(
        {'admin3_id': ['US-ZZ-AAA']}, geometry=[box(0, 0, 1, 1)], crs=4326
    )
    table._note_read(1)

    with pytest.warns(UserWarning, match='not found in data'):
        table._save_recipe_data(rows)

    reasons = table.download_partition['empty_reasons']
    assert reasons['US-ZZ-BBB'][0] == 'not_covered'
    assert 'US-ZZ-AAA' not in reasons

    # An empty chunk records every unit it would have saved.
    table.download_partition['empty_reasons'] = {}
    table._note_read(0)
    table._save_recipe_data(EMPTY)
    assert set(table.download_partition['empty_reasons']) == {
        'US-ZZ-AAA',
        'US-ZZ-BBB',
    }


def test_stacked_layer_without_units_is_nothing_stacked(tmp_path, monkeypatch):
    table = _table(tmp_path, monkeypatch)
    stacked = {'admin_id': 'US-ZZ', 'entity': 'property-example-2026'}

    table._save_recipe_data(
        pd.DataFrame({'parcel_id_local': []}),
        recipe=stacked,
        empty_reason=('nothing_stacked', '3 parcel rows, none stacked'),
    )

    assert empty_reason(tmp_path / 'US-ZZ-AAA_t.parquet') == 'nothing_stacked'


# Writer 4: a unit no tile holds rows for


def test_tile_merge_records_a_unit_no_tile_covers(monkeypatch, tmp_path):
    monkeypatch.setattr(
        outputs_module,
        'get_output_path',
        lambda recipe, admin_id, partition_id=None, **k: (
            tmp_path / f'{admin_id}_{partition_id}.parquet'
        ),
    )
    ing = Ingester.__new__(Ingester)
    ing.recipe = {'admin_id': AdminId('US-ZZ'), 'entity': 'footprint-example-2026'}
    ing.verbose = False
    ing.admin_ids_to_save = ['US-ZZ-AAA']
    ing.partition_ids_to_download = ['t1', 't2']
    ing.tile_admin_link = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([('t1', 'US-ZZ-AAA'), ('t2', 'US-ZZ-AAA')])
    )

    ing._merge_tile_partials()

    reason, detail = ing._empty_reasons['US-ZZ-AAA']
    assert reason == 'not_covered' and 't1' in detail


# Writer 6: aggregates and joins inherit their inputs' reason


def test_aggregate_of_empty_inputs_inherits_their_reason(tmp_path):
    a, b = tmp_path / 'a.parquet', tmp_path / 'b.parquet'
    save_empty_table(a, 'not_covered', 'none in A')
    save_empty_table(b, 'not_published', 'B not published')
    final = tmp_path / 'final.parquet'

    _aggregate_to_file(final, [('a', a), ('b', b)], keep_original=True)

    assert empty_reason(final) == 'not_covered'
    assert 'B not published' in _detail(final)


def test_aggregate_with_a_reasonless_input_records_none(tmp_path):
    a, b = tmp_path / 'a.parquet', tmp_path / 'b.parquet'
    save_empty_table(a, 'not_covered', 'none in A')
    _reasonless_empty(b)
    final = tmp_path / 'final.parquet'

    _aggregate_to_file(final, [('a', a), ('b', b)], keep_original=True)

    assert is_unexplained_empty(final)


def test_aggregate_with_rows_records_no_reason(tmp_path):
    a, b = tmp_path / 'a.parquet', tmp_path / 'b.parquet'
    save_empty_table(a, 'not_covered', 'none in A')
    _with_rows(b)
    final = tmp_path / 'final.parquet'

    _aggregate_to_file(final, [('a', a), ('b', b)], keep_original=True)

    assert empty_reason(final) is None and not is_unexplained_empty(final)


def _county_rows(path, unit, n):
    save_parquet(
        gpd.GeoDataFrame(
            {'admin3_id': [unit] * n}, geometry=[box(0, 0, 1, 1)] * n, crs=4326
        ),
        path,
    )


def test_replace_that_drops_every_row_inherits_the_inputs_reason(tmp_path):
    final = tmp_path / 'state.parquet'
    _county_rows(final, 'US-ZZ-AAA', 2)
    chunk = tmp_path / 'aaa.parquet'
    save_empty_table(chunk, 'not_covered', 'no rows for AAA this run')

    _aggregate_to_file(
        final, [('US-ZZ-AAA', chunk)], replace_by='admin3_id', keep_original=True
    )

    assert empty_reason(final) == 'not_covered'


def test_replace_that_drops_every_row_without_reasons_raises(tmp_path):
    final = tmp_path / 'state.parquet'
    _county_rows(final, 'US-ZZ-AAA', 2)
    before = final.read_bytes()
    chunk = tmp_path / 'aaa.parquet'
    _reasonless_empty(chunk)

    with pytest.raises(UnexplainedEmptyIngestError, match='left no rows'):
        _aggregate_to_file(
            final, [('US-ZZ-AAA', chunk)], replace_by='admin3_id', keep_original=True
        )
    assert final.read_bytes() == before


def test_join_of_an_empty_base_table_inherits_its_reason(monkeypatch, tmp_path):
    import openplaces.io.aggregate as aggregate_module

    monkeypatch.setattr(
        aggregate_module,
        'get_output_path',
        lambda recipe, admin_id, partition_id=None, **k: (
            tmp_path / f'{admin_id}{"_" + partition_id if partition_id else ""}.parquet'
        ),
    )
    base = tmp_path / 'US-ZZ-AAA_base.parquet'
    save_empty_table(base, 'not_covered', 'no parcels for AAA')

    join_partitions_by_index({}, ['base'], admin_ids=['US-ZZ-AAA'])

    assert empty_reason(tmp_path / 'US-ZZ-AAA.parquet') == 'not_covered'


# Enforcement: the skip check, the DAG and the consumer


def _arkansas(monkeypatch, tmp_path):
    out = tmp_path / 'US-AR-NE_parcel-example-2026.parquet'
    monkeypatch.setattr(
        scope_module,
        'get_output_path',
        lambda recipe, admin_id, partition_id=None, **k: out,
    )
    monkeypatch.setattr(outputs_module, 'get_output_path', lambda *a, **k: out)
    monkeypatch.setattr(
        scope_module,
        'get_admin',
        lambda *a, **k: pd.DataFrame(index=pd.Index(['US-AR-NE'])),
    )
    monkeypatch.setattr(scope_module, 'receipt_justifies_skip', lambda *a: False)
    ing = Ingester(
        {
            'admin_id': AdminId('US-AR'),
            'entity': Entity('parcel', 'example', '2026'),
            'save_to': {'admin_level': 3},
        },
        admin_ids='US-AR-NE',
        timer=_Timer(),
    )
    ran = []

    def fake_process(self, **kwargs):
        ran.append(kwargs['admin_id_to_download'])
        _with_rows(out, 3)

    monkeypatch.setattr(
        Ingester,
        '_resolve_admin_ids_to_process',
        lambda self: setattr(self, 'admin_ids_to_process', self.admin_ids_to_save),
    )
    monkeypatch.setattr(
        Ingester,
        '_resolve_admin_ids_to_download',
        lambda self: setattr(self, 'admin_ids_to_download', self.admin_ids_to_save),
    )
    monkeypatch.setattr(
        Ingester,
        '_resolve_partition_ids',
        lambda self, reprocess: setattr(self, 'partition_ids_to_download', [None]),
    )
    monkeypatch.setattr(Ingester, '_ingest_download_partition', fake_process)
    return ing, out, ran


def test_stale_reasonless_output_is_ingested_again(monkeypatch, tmp_path):
    """The Arkansas case: the stale empty file no longer reads as built."""
    ing, out, ran = _arkansas(monkeypatch, tmp_path)
    _reasonless_empty(out)

    ing.ingest()

    assert ran == ['US-AR-NE']
    assert not is_unexplained_empty(out)


def test_reasoned_empty_output_is_not_ingested_again(monkeypatch, tmp_path, capsys):
    ing, out, ran = _arkansas(monkeypatch, tmp_path)
    save_empty_table(out, 'not_published', 'fabricated')

    ing.ingest()

    assert ran == []
    assert 'already have outputs' in capsys.readouterr().out


def test_dag_reruns_a_unit_whose_output_is_unexplained(tmp_path):
    out = tmp_path / 'u.parquet'
    _reasonless_empty(out)
    geo = out.with_stem('u_geo')

    assert needs_rerun([out, geo], []) is True

    save_empty_table(out, 'not_covered', 'fabricated')
    assert needs_rerun([out], []) is False
