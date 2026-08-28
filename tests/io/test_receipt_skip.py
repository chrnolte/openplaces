"""Tests for receipt-aware skip-if-exists in the ingester and harmonizer."""

import pandas as pd
import pytest

import openplaces.io.harmonizer as harmonizer_mod
import openplaces.io.ingester as ingester_mod
from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.io import cleanup as cl
from openplaces.recipe import get_output_path, get_recipe_by_id

NSI = 'US_building-nsi-2022'
FOOTPRINT_SPINE = 'US_footprint-spine-2026'
PARCEL_SPINE = 'US_parcel-spine-2026'
COUNTY = 'US-NC-BRU'


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    monkeypatch.delenv('SNAKEMAKE', raising=False)
    monkeypatch.delenv('OPENPLACES_ORCHESTRATED', raising=False)
    return tmp_path


def _write_parquet(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'geo_id': ['a'], 'value': [1.0]}).to_parquet(path)
    return path


def _valid_nsi_receipt():
    """NSI deleted; both tree consumers exist on disk."""
    verified = []
    for recipe_id in (FOOTPRINT_SPINE, PARCEL_SPINE):
        path = get_output_path(get_recipe_by_id(recipe_id), admin_id=COUNTY)
        _write_parquet(path)
        verified.append(
            {
                'recipe_id': recipe_id,
                'admin_id': COUNTY,
                'path': cl._relative_posix(path),
            }
        )
    nsi_out = get_output_path(get_recipe_by_id(NSI), admin_id=COUNTY)
    cl.write_receipt(
        nsi_out,
        {'recipe_id': NSI, 'admin_id': COUNTY, 'consumers_verified': verified},
    )
    return nsi_out


def _ingester(monkeypatch):
    """Bare Ingester with just what _resolve_output_admin_ids needs."""
    monkeypatch.setattr(
        ingester_mod,
        'get_admin',
        lambda admin_id, level: pd.DataFrame(index=[COUNTY]),
    )
    instance = ingester_mod.Ingester.__new__(ingester_mod.Ingester)
    instance.recipe = get_recipe_by_id(NSI)
    instance.admin_ids = [AdminId(COUNTY)]
    instance.verbose = False
    return instance


def test_ingester_honors_receipt(data_root, monkeypatch):
    ingester = _ingester(monkeypatch)
    assert ingester._resolve_output_admin_ids(reprocess=False) == [COUNTY]
    _valid_nsi_receipt()
    assert ingester._resolve_output_admin_ids(reprocess=False) == []


def test_ingester_ignores_receipt_when_disabled(data_root, monkeypatch):
    ingester = _ingester(monkeypatch)
    _valid_nsi_receipt()
    retention = {'cleanup': {'honor_receipts': False}, 'recipes': {}}
    monkeypatch.setitem(cfg.config, 'retention', retention)
    assert ingester._resolve_output_admin_ids(reprocess=False) == [COUNTY]


def test_ingester_ignores_receipt_when_orchestrated(data_root, monkeypatch):
    ingester = _ingester(monkeypatch)
    _valid_nsi_receipt()
    monkeypatch.setenv('OPENPLACES_ORCHESTRATED', '1')
    assert ingester._resolve_output_admin_ids(reprocess=False) == [COUNTY]


def test_ingester_reprocess_lists_and_receipt_survives_enumeration(
    data_root, monkeypatch
):
    ingester = _ingester(monkeypatch)
    nsi_out = _valid_nsi_receipt()
    # Enumeration with reprocess=True must not delete the receipt (the
    # aggregate-mode path calls it just to list candidates)
    assert ingester._resolve_output_admin_ids(reprocess=True) == [COUNTY]
    assert cl.read_receipt(nsi_out) is not None


def test_harmonizer_honors_receipt_and_reprocess_discards(data_root, monkeypatch):
    recipe = get_recipe_by_id(FOOTPRINT_SPINE)
    out_path = get_output_path(recipe, admin_id=COUNTY)
    cheer_out = get_output_path(
        get_recipe_by_id('US_footprint-openplaces-2026'), admin_id=COUNTY
    )
    _write_parquet(cheer_out)
    cl.write_receipt(
        out_path,
        {
            'recipe_id': FOOTPRINT_SPINE,
            'admin_id': COUNTY,
            'consumers_verified': [
                {
                    'recipe_id': 'US_footprint-openplaces-2026',
                    'admin_id': COUNTY,
                    'path': cl._relative_posix(cheer_out),
                }
            ],
        },
    )

    calls = {'n': 0}
    monkeypatch.setattr(
        harmonizer_mod.Harmonizer,
        '_harmonize_one',
        lambda self, admin_id, reprocess=False: calls.__setitem__('n', calls['n'] + 1),
    )
    harmonizer = harmonizer_mod.Harmonizer(recipe, admin_ids=[COUNTY])

    # The receipt records the CHEER curation as its only consumer, but the
    # current tree has more consumers of the spine (parcel spine, enrich
    # recipes) — the receipt must NOT justify skipping (fail safe)
    harmonizer.harmonize(reprocess=False)
    assert calls['n'] == 1

    # reprocess=True discards the receipt
    harmonizer.harmonize(reprocess=True)
    assert calls['n'] == 2
    assert cl.read_receipt(out_path) is None


def test_harmonizer_skips_when_output_exists(data_root, monkeypatch):
    recipe = get_recipe_by_id(FOOTPRINT_SPINE)
    _write_parquet(get_output_path(recipe, admin_id=COUNTY))
    calls = {'n': 0}
    monkeypatch.setattr(
        harmonizer_mod.Harmonizer,
        '_harmonize_one',
        lambda self, admin_id, reprocess=False: calls.__setitem__('n', calls['n'] + 1),
    )
    harmonizer_mod.Harmonizer(recipe, admin_ids=[COUNTY]).harmonize(reprocess=False)
    assert calls['n'] == 0
