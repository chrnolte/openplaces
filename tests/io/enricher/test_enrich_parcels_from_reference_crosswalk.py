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
