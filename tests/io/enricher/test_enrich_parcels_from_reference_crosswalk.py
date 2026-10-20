"""Tests for the reference-parcel crosswalk enrichment step."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.enricher.parcels as parcels_mod
from openplaces.core.schema import AdminId
from openplaces.io import save_parquet
from openplaces.io.enricher import EnrichState
from openplaces.io.enricher.parcels import (
    _source_suffix,
    enrich_parcels_from_reference_crosswalk,
)


def _gdf(boxes, extra=None, crs='EPSG:4326'):
    data = {'parcel_id': list(boxes)}
    if extra:
        data.update(extra)
    return gpd.GeoDataFrame(
        data, geometry=[boxes[k] for k in boxes], crs=crs
    ).set_index('parcel_id')


def _make_state(spine, reprocess=False, verbose=False):
    return EnrichState(
        recipe={'reference_parcel_recipe_id': 'FAKE_placeslab-fmv'},
        entity_recipe={},
        admin_id=AdminId('US-MA-MI'),
        verbose=verbose,
        timer=None,
        spine=spine,
        evidence=pd.DataFrame(index=spine.index),
        reprocess=reprocess,
    )


def test_missing_reference_coverage_returns_state_unchanged(monkeypatch):
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: {})

    def raise_not_found(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(parcels_mod, 'get_entities', raise_not_found)

    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    state = _make_state(spine)

    result = enrich_parcels_from_reference_crosswalk(state)

    assert result is state
    assert result.evidence.empty
    assert result.evidence.index.equals(spine.index)


def test_writes_weighted_evidence_columns(monkeypatch):
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: {})

    old = _gdf(
        {'old_1': box(0, 0, 0.001, 0.001)},
        extra={'slope': [5.0], 'm2_bld_fp': [200.0]},
    )
    monkeypatch.setattr(parcels_mod, 'get_entities', lambda *a, **k: old)

    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    state = _make_state(spine)

    result = enrich_parcels_from_reference_crosswalk(
        state, mean_columns=['slope'], sum_columns=['m2_bld_fp'], silent_qa=True
    )

    assert result.evidence.loc['new_1', 'slope'] == pytest.approx(5.0)
    assert result.evidence.loc['new_1', 'm2_bld_fp'] == pytest.approx(200.0)
    assert result.metadata['attempted_keys'] == {'new_1'}


def test_empty_reference_returns_state_unchanged(monkeypatch):
    """A reference file with no rows for the unit is treated like no file.

    Verbose mode reports the matched share as a fraction of the reference
    rows, which divided by zero before the guard.
    """
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: {})
    old = _gdf({'old_1': box(0, 0, 0.001, 0.001)}, extra={'slope': [5.0]})
    monkeypatch.setattr(parcels_mod, 'get_entities', lambda *a, **k: old.iloc[:0])

    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    state = _make_state(spine, verbose=True)

    result = enrich_parcels_from_reference_crosswalk(
        state, mean_columns=['slope'], silent_qa=True
    )

    assert result is state
    assert result.evidence.empty


def test_explicit_id_columns_are_picked_from_the_largest_overlap(monkeypatch):
    """`id_columns` passed with explicit `mean_columns` is honored, not dropped."""
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: {})
    old = _gdf(
        {'old_1': box(0, 0, 0.001, 0.0002), 'old_2': box(0, 0.0002, 0.001, 0.001)},
        extra={'slope': [5.0, 7.0], 'zip_raw': ['00001', '00002']},
    )
    monkeypatch.setattr(parcels_mod, 'get_entities', lambda *a, **k: old)

    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    state = _make_state(spine)

    result = enrich_parcels_from_reference_crosswalk(
        state, mean_columns=['slope'], id_columns=['zip_raw'], silent_qa=True
    )

    assert result.evidence.loc['new_1', 'zip_raw'] == '00002'
    assert result.evidence.loc['new_1', 'slope'] == pytest.approx(6.6, abs=0.1)


def test_calibration_towns_may_overlap(monkeypatch):
    """A reference centroid inside two town polygons gets one town, not an error."""
    towns = gpd.GeoDataFrame(
        {'admin4_id_admin1': ['2500101', '2500102']},
        geometry=[box(-1, -1, 1, 1), box(-1, -1, 1, 1)],
        index=pd.Index(['US-MA-MI-A', 'US-MA-MI-B'], name='admin4_id'),
        crs='EPSG:4326',
    )
    monkeypatch.setattr(parcels_mod, 'get_admin', lambda *a, **k: towns)

    old = _gdf({'old_1': box(0, 0, 0.001, 0.001)})
    spine = _gdf(
        {'new_1': box(0, 0, 0.001, 0.001)}, extra={'admin4_id': ['US-MA-MI-A']}
    )
    state = _make_state(spine)

    resolved, group_col = parcels_mod._resolve_calibrate_group_col(state, old)

    assert group_col == 'admin4_id'
    assert resolved.index.equals(old.index)
    assert resolved.loc['old_1', 'admin4_id'] in {'US-MA-MI-A', 'US-MA-MI-B'}


def test_missing_reference_parcel_recipe_raises():
    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    state = EnrichState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US-MA-MI'),
        verbose=False,
        timer=None,
        spine=spine,
        evidence=pd.DataFrame(index=spine.index),
    )

    with pytest.raises(ValueError, match='reference_parcel_recipe_id'):
        enrich_parcels_from_reference_crosswalk(state)


def test_source_suffix_uses_version_not_source_id():
    reference_recipe = {
        'entity': SimpleNamespace(
            version='fmv2026', source=SimpleNamespace(source_id='placeslab')
        )
    }

    assert _source_suffix(reference_recipe) == 'fmv2026'


def _old_with_join_key(slope=5.0, parcel_id_int=42):
    return _gdf(
        {'old_1': box(0, 0, 0.001, 0.001)},
        extra={'slope': [slope], 'parcel_id_int': [parcel_id_int]},
    )


def test_save_crosswalk_writes_sidecar_with_reference_join_key(monkeypatch, tmp_path):
    reference_recipe = {'join_partitions_by': {'join_key_name': 'parcel_id_int'}}
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: reference_recipe)

    old = _old_with_join_key()
    monkeypatch.setattr(parcels_mod, 'get_entities', lambda *a, **k: old)

    out_path = tmp_path / 'evidence.parquet'
    monkeypatch.setattr(parcels_mod, 'get_output_path', lambda *a, **k: out_path)

    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    state = _make_state(spine)

    enrich_parcels_from_reference_crosswalk(
        state, mean_columns=['slope'], save_crosswalk=True, silent_qa=True
    )

    crosswalk_path = out_path.with_stem(out_path.stem + '_crosswalk')
    assert crosswalk_path.exists()
    saved = pd.read_parquet(crosswalk_path)
    assert {
        'parcel_id',
        'parcel_id_int',
        'area_ha',
        'fraction_of_old',
        'match_type',
    } <= set(saved.columns)
    assert 'parcel_id_old' not in saved.columns
    assert saved['parcel_id'].iloc[0] == 'new_1'
    assert saved['parcel_id_int'].iloc[0] == 42
    assert saved['fraction_of_old'].iloc[0] == pytest.approx(1.0)


def test_save_crosswalk_falls_back_to_reference_parcel_id_without_join_key(
    monkeypatch, tmp_path
):
    reference_recipe = {}  # no join_partitions_by
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: reference_recipe)

    old = _old_with_join_key()
    monkeypatch.setattr(parcels_mod, 'get_entities', lambda *a, **k: old)

    out_path = tmp_path / 'evidence.parquet'
    monkeypatch.setattr(parcels_mod, 'get_output_path', lambda *a, **k: out_path)

    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    state = _make_state(spine)

    enrich_parcels_from_reference_crosswalk(
        state, mean_columns=['slope'], save_crosswalk=True, silent_qa=True
    )

    saved = pd.read_parquet(out_path.with_stem(out_path.stem + '_crosswalk'))
    assert 'reference_parcel_id' in saved.columns
    assert 'parcel_id_int' not in saved.columns
    assert saved['reference_parcel_id'].iloc[0] == 'old_1'


def test_save_crosswalk_false_writes_nothing(monkeypatch):
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: {})
    monkeypatch.setattr(
        parcels_mod, 'get_entities', lambda *a, **k: _old_with_join_key()
    )

    calls = []
    monkeypatch.setattr(parcels_mod, 'save_parquet', lambda *a, **k: calls.append(1))

    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    state = _make_state(spine)

    enrich_parcels_from_reference_crosswalk(
        state, mean_columns=['slope'], silent_qa=True
    )  # save_crosswalk left at its default (False)

    assert calls == []


def test_reloads_existing_crosswalk_when_not_reprocessing(monkeypatch, tmp_path):
    reference_recipe = {'join_partitions_by': {'join_key_name': 'parcel_id_int'}}
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: reference_recipe)

    old = _old_with_join_key()
    monkeypatch.setattr(parcels_mod, 'get_entities', lambda *a, **k: old)

    out_path = tmp_path / 'evidence.parquet'
    monkeypatch.setattr(parcels_mod, 'get_output_path', lambda *a, **k: out_path)
    crosswalk_path = out_path.with_stem(out_path.stem + '_crosswalk')
    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    save_parquet(
        pd.DataFrame(
            {
                'parcel_id': ['new_1'],
                'parcel_id_int': [42],
                'area_ha': [0.0001],
                'fraction_of_old': [1.0],
                'match_type': ['geo_id'],
            }
        ),
        crosswalk_path,
        file_metadata={
            parcels_mod._CROSSWALK_INPUTS_KEY: parcels_mod._crosswalk_inputs(spine, old)
        },
    )

    def _raise(*a, **k):
        raise AssertionError('build_id_or_overlay_crosswalk should not run')

    monkeypatch.setattr(parcels_mod, 'build_id_or_overlay_crosswalk', _raise)

    state = _make_state(spine, reprocess=False)

    result = enrich_parcels_from_reference_crosswalk(
        state, mean_columns=['slope'], save_crosswalk=True, silent_qa=True
    )

    assert result.evidence.loc['new_1', 'slope'] == pytest.approx(5.0)


def test_reprocess_true_rebuilds_crosswalk_even_if_sidecar_exists(
    monkeypatch, tmp_path
):
    reference_recipe = {'join_partitions_by': {'join_key_name': 'parcel_id_int'}}
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: reference_recipe)

    old = _old_with_join_key()
    monkeypatch.setattr(parcels_mod, 'get_entities', lambda *a, **k: old)

    out_path = tmp_path / 'evidence.parquet'
    monkeypatch.setattr(parcels_mod, 'get_output_path', lambda *a, **k: out_path)
    crosswalk_path = out_path.with_stem(out_path.stem + '_crosswalk')
    pd.DataFrame(
        {
            'parcel_id': ['new_1'],
            'parcel_id_int': [42],
            'area_ha': [999.0],  # deliberately wrong, to prove it gets overwritten
            'fraction_of_old': [1.0],
            'match_type': ['geo_id'],
        }
    ).to_parquet(crosswalk_path)

    calls = []
    real_build = parcels_mod.build_id_or_overlay_crosswalk

    def _spy(*a, **k):
        calls.append(1)
        return real_build(*a, **k)

    monkeypatch.setattr(parcels_mod, 'build_id_or_overlay_crosswalk', _spy)

    spine = _gdf({'new_1': box(0, 0, 0.001, 0.001)})
    state = _make_state(spine, reprocess=True)

    enrich_parcels_from_reference_crosswalk(
        state, mean_columns=['slope'], save_crosswalk=True, silent_qa=True
    )

    assert len(calls) == 1
    saved = pd.read_parquet(crosswalk_path)
    assert saved['area_ha'].iloc[0] != pytest.approx(999.0)


def _stack_renamed_spines():
    """Two spine vintages of one place, before and after a re-ingest.

    Before, two rows stacked on one lot polygon carried the suffixed ids
    `hash_a-1` and `hash_a-2`; after the stack collapsed, the lot is
    indexed by its bare hash `hash_a`. Ids and geometry are fabricated.
    """
    lot = box(0, 0, 0.001, 0.001)
    other = box(0.002, 0, 0.003, 0.001)
    before = _gdf({'hash_a-1': lot, 'hash_a-2': lot, 'hash_b': other})
    after = _gdf({'hash_a': lot, 'hash_b': other})
    old = _gdf(
        {'old_1': lot, 'old_2': other},
        extra={'slope': [5.0, 9.0], 'parcel_id_int': [41, 42]},
    )
    return before, after, old


@pytest.mark.parametrize('with_footer', [True, False])
def test_sidecar_built_from_renamed_spine_is_rebuilt(
    monkeypatch, tmp_path, with_footer
):
    """A sidecar naming `hash_a-1` is not reused once the spine says `hash_a`.

    Reproduces US-NC-BRU on 2026-10-18: a 2026-09-09 sidecar reused after
    the stacked-units fix renamed its parcels raised a KeyError in the
    area QA check. A sidecar written before the footer existed is
    treated the same way.
    """
    reference_recipe = {'join_partitions_by': {'join_key_name': 'parcel_id_int'}}
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: reference_recipe)
    before, after, old = _stack_renamed_spines()
    monkeypatch.setattr(parcels_mod, 'get_entities', lambda *a, **k: old)
    out_path = tmp_path / 'evidence.parquet'
    monkeypatch.setattr(parcels_mod, 'get_output_path', lambda *a, **k: out_path)
    crosswalk_path = out_path.with_stem(out_path.stem + '_crosswalk')

    stale = parcels_mod.build_id_or_overlay_crosswalk(
        before, old, on_duplicate_geo_id='groupby', min_overlap_m2=0
    )
    assert stale['parcel_id_new'].str.endswith('-1').any()
    if with_footer:
        parcels_mod._save_published_crosswalk(
            stale,
            old,
            'parcel_id_int',
            crosswalk_path,
            parcels_mod._crosswalk_inputs(before, old),
        )
    else:
        published = stale.rename(columns={'parcel_id_new': 'parcel_id'})
        published['parcel_id_int'] = published.pop('parcel_id_old').map(
            old['parcel_id_int']
        )
        published.to_parquet(crosswalk_path)

    state = _make_state(after, reprocess=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = enrich_parcels_from_reference_crosswalk(
            state,
            mean_columns=['slope'],
            save_crosswalk=True,
            silent_qa=True,
            min_overlap_m2=0,
        )

    assert result.evidence.loc['hash_a', 'slope'] == pytest.approx(5.0)
    assert result.evidence.loc['hash_b', 'slope'] == pytest.approx(9.0)
    saved = pd.read_parquet(crosswalk_path)
    assert set(saved['parcel_id']) == {'hash_a', 'hash_b'}


# A delivered crosswalk: the reference ships no geometry, only pairs
# of (current parcel, reference parcel). Every id, value and footer
# entry below is fabricated.
_FOOTER = {
    'openplaces_commit': 'c0ffee0',
    'bundle_file': 'fake-region_parcel.parquet',
    'bundle_sha256': '0' * 64,
    'bundle_rows': '2',
}


def _delivered_pairs(**overrides):
    pairs = {
        'parcel_id_new': ['new_1', 'new_1', 'new_2'],
        'reference_parcel_id': ['sha_a', 'sha_b', 'sha_b'],
        'area_ha': [0.2, 0.1, 0.3],
        'fraction_of_old': [1.0, 0.25, 0.75],
        'match_type': ['geo_id', 'overlay', 'overlay'],
    }
    pairs.update(overrides)
    return pd.DataFrame(pairs)


def _write_delivered(path, pairs, footer=_FOOTER):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(pairs, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({k.encode(): v.encode() for k, v in footer.items()})
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _reference_without_geometry():
    """Joined reference tables as ingest writes them without boundaries."""
    return pd.DataFrame(
        {
            'parcel_id_int': ['sha_a', 'sha_b'],
            'slope': [4.0, 8.0],
            'm2_bld_fp': [100.0, 200.0],
        },
        index=pd.Index(['sha_a', 'sha_b'], name='parcel_id'),
    )


@pytest.fixture
def delivered_setup(monkeypatch, tmp_path):
    """Reference recipe paths in tmp_path, a delivered crosswalk on disk."""
    reference_recipe = {'join_partitions_by': {'join_key_name': 'parcel_id_int'}}
    monkeypatch.setattr(parcels_mod, 'get_recipe_by_id', lambda rid: reference_recipe)

    def fake_output_path(recipe, admin_id, partition_id=None, **kwargs):
        if recipe is reference_recipe:
            name = f'ref_{partition_id}' if partition_id else 'ref'
        else:
            name = 'evidence'
        return tmp_path / f'{name}.parquet'

    monkeypatch.setattr(parcels_mod, 'get_output_path', fake_output_path)
    geom_requests = []

    def fake_get_entities(recipe, admin_id, geom=False, **kwargs):
        geom_requests.append(geom)
        return _reference_without_geometry()

    monkeypatch.setattr(parcels_mod, 'get_entities', fake_get_entities)
    _write_delivered(tmp_path / 'ref_crosswalk.parquet', _delivered_pairs())
    return SimpleNamespace(tmp_path=tmp_path, geom_requests=geom_requests)


def _spine(*ids):
    return _gdf({pid: box(i, 0, i + 0.001, 0.001) for i, pid in enumerate(ids)})


def test_delivered_crosswalk_is_read_and_aggregated(monkeypatch, delivered_setup):
    def _raise(*a, **k):
        raise AssertionError('a delivered crosswalk is read, not built or QAed')

    monkeypatch.setattr(parcels_mod, 'build_id_or_overlay_crosswalk', _raise)
    monkeypatch.setattr(parcels_mod, 'warn_on_geo_id_area_mismatch', _raise)
    state = _make_state(_spine('new_1', 'new_2'))

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = enrich_parcels_from_reference_crosswalk(
            state,
            mean_columns=['slope'],
            sum_columns=['m2_bld_fp'],
            save_crosswalk=True,
            delivered_crosswalk_table='crosswalk',
        )

    assert delivered_setup.geom_requests == [False]
    # Mean weighted by overlap area, sum apportioned by fraction_of_old.
    assert result.evidence.loc['new_1', 'slope'] == pytest.approx(1.6 / 0.3)
    assert result.evidence.loc['new_2', 'slope'] == pytest.approx(8.0)
    assert result.evidence.loc['new_1', 'm2_bld_fp'] == pytest.approx(150.0)
    assert result.evidence.loc['new_2', 'm2_bld_fp'] == pytest.approx(150.0)
    # The delivered file is the publication; no second sidecar is written.
    assert not (delivered_setup.tmp_path / 'evidence_crosswalk.parquet').exists()


def test_delivered_crosswalk_for_other_parcels_is_refused(delivered_setup):
    state = _make_state(_spine('new_1', 'new_2', 'new_3', 'new_4'))

    with pytest.raises(ValueError) as excinfo:
        enrich_parcels_from_reference_crosswalk(
            state, mean_columns=['slope'], delivered_crosswalk_table='crosswalk'
        )

    message = str(excinfo.value)
    assert 'bundle_file=fake-region_parcel.parquet' in message
    assert 'openplaces_commit=c0ffee0' in message
    assert '50.00%' in message  # spine ids found in the crosswalk
    assert '100.00%' in message  # crosswalk ids found in the spine


def test_delivered_crosswalk_missing_newer_spine_parcels_warns(delivered_setup):
    """Spine parcels newer than the reference (90 to 99 percent) warn."""
    named = [f'new_{i}' for i in range(19)]
    _write_delivered(
        delivered_setup.tmp_path / 'ref_crosswalk.parquet',
        _delivered_pairs(
            parcel_id_new=named,
            reference_parcel_id=['sha_a'] * 19,
            area_ha=[0.1] * 19,
            fraction_of_old=[1 / 19] * 19,
            match_type=['overlay'] * 19,
        ),
    )
    state = _make_state(_spine(*named, 'new_split'))

    with pytest.warns(UserWarning) as record:
        result = enrich_parcels_from_reference_crosswalk(
            state, mean_columns=['slope'], delivered_crosswalk_table='crosswalk'
        )

    messages = [str(w.message) for w in record]
    coverage = [m for m in messages if 'of spine parcel ids appear' in m]
    assert len(coverage) == 1
    assert '95.00%' in coverage[0]
    assert 'bundle_file=fake-region_parcel.parquet' in coverage[0]
    assert 'openplaces_commit=c0ffee0' in coverage[0]
    assert result.evidence.loc['new_0', 'slope'] == pytest.approx(4.0)


def test_delivered_crosswalk_naming_unknown_parcels_is_refused(delivered_setup):
    _write_delivered(
        delivered_setup.tmp_path / 'ref_crosswalk.parquet',
        _delivered_pairs(parcel_id_new=['new_1', 'gone_1', 'new_2']),
    )
    state = _make_state(_spine('new_1', 'new_2'))

    with pytest.raises(ValueError, match='66.67%'):
        enrich_parcels_from_reference_crosswalk(
            state, mean_columns=['slope'], delivered_crosswalk_table='crosswalk'
        )


def test_delivered_crosswalk_with_unknown_match_type_is_refused(delivered_setup):
    _write_delivered(
        delivered_setup.tmp_path / 'ref_crosswalk.parquet',
        _delivered_pairs(match_type=['geo_id', 'fuzzy', 'overlay']),
    )
    state = _make_state(_spine('new_1', 'new_2'))

    with pytest.raises(ValueError, match='fuzzy'):
        enrich_parcels_from_reference_crosswalk(
            state, mean_columns=['slope'], delivered_crosswalk_table='crosswalk'
        )


def test_delivered_fractions_off_one_warn(delivered_setup):
    _write_delivered(
        delivered_setup.tmp_path / 'ref_crosswalk.parquet',
        _delivered_pairs(fraction_of_old=[1.0, 0.25, 0.70]),
    )
    state = _make_state(_spine('new_1', 'new_2'))

    with pytest.warns(UserWarning, match='1 of 2 reference parcels'):
        result = enrich_parcels_from_reference_crosswalk(
            state, mean_columns=['slope'], delivered_crosswalk_table='crosswalk'
        )

    assert result.evidence.loc['new_2', 'slope'] == pytest.approx(8.0)


def test_reference_with_geometry_builds_its_crosswalk(monkeypatch, delivered_setup):
    """A county that keeps its boundaries ignores a delivered crosswalk."""
    tmp_path = delivered_setup.tmp_path
    (tmp_path / 'ref.parquet').touch()
    (tmp_path / 'ref_geo.parquet').touch()
    old = _gdf(
        {'h1': box(0, 0, 0.001, 0.001)},
        extra={'slope': [5.0], 'parcel_id_int': ['sha_a']},
    )
    monkeypatch.setattr(parcels_mod, 'get_entities', lambda *a, **k: old)
    calls = []
    real_build = parcels_mod.build_id_or_overlay_crosswalk

    def _spy(*a, **k):
        calls.append(1)
        return real_build(*a, **k)

    monkeypatch.setattr(parcels_mod, 'build_id_or_overlay_crosswalk', _spy)
    state = _make_state(_gdf({'new_1': box(0, 0, 0.001, 0.001)}))

    result = enrich_parcels_from_reference_crosswalk(
        state,
        mean_columns=['slope'],
        silent_qa=True,
        delivered_crosswalk_table='crosswalk',
    )

    assert calls == [1]
    assert result.evidence.loc['new_1', 'slope'] == pytest.approx(5.0)
