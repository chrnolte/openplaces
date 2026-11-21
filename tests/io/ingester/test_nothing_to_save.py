"""An ingest with no unit left to save stops after resolving its scope.

A direct `run_stage ingest` of a tile-partitioned recipe on a unit whose
output already existed raised `ValueError: No objects to concatenate`:
`_resolve_tile_ids` concatenated one tile crosswalk per unit to save, and
there were none. Snakemake never saw it, since it skips an existing output
before calling the step. All units, ids and paths here are fabricated.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.geo import link as geo_link
from openplaces.io.ingester import Ingester
from openplaces.io.ingester import partitions as ingester_partitions
from openplaces.io.ingester import scope as ingester_scope
from openplaces.timing import Timer

RECIPE_ENTITY = 'footprint-fabricated-2026'
BUILT_UNIT = 'XA-AA-AAA'


def _recipe():
    return {
        'admin_id': AdminId(None),
        'entity': RECIPE_ENTITY,
        'save_to': {'admin_level': 3},
        'download_by': {
            'partition': 'tile_id',
            'tile_recipe_id': 'tile-fabricated-2026',
        },
        'overlay_admin_ids': {
            'admin_recipe_id': 'admin-fabricated-2026',
            'admin_level': 3,
        },
    }


def _bare(admin_ids):
    ingester = Ingester.__new__(Ingester)
    ingester.recipe = _recipe()
    ingester.admin_ids = [AdminId(a) for a in admin_ids]
    ingester.partition_ids = None
    ingester.timer = Timer('test')
    ingester.verbose = False
    ingester._owns_timer = False
    return ingester


@pytest.fixture
def built(monkeypatch, tmp_path):
    """A spine of two countries, one county built, nothing reachable."""
    by_level = {1: ['XA', 'XB'], 3: [BUILT_UNIT]}

    def fake_get_admin(admin_id, level, **kwargs):
        return pd.DataFrame(index=pd.Index(by_level.get(level, []), name='admin_id'))

    def fake_output_path(recipe, admin_id=None, partition_id=None, **kwargs):
        stem = f'{admin_id}_{RECIPE_ENTITY}'
        if partition_id is not None:
            stem += f'_{partition_id}'
        return tmp_path / f'{stem}.parquet'

    def no_tile_link(*args, **kwargs):
        raise AssertionError('the tile crosswalk must not be built')

    for module in (ingester_scope, ingester_partitions):
        monkeypatch.setattr(module, 'get_admin', fake_get_admin)
    monkeypatch.setattr(ingester_scope, 'get_output_path', fake_output_path)
    monkeypatch.setattr(ingester_scope, 'receipt_justifies_skip', lambda *a: False)
    # Stand-ins for the admin recipe lookup, so that before the fix the
    # run reached the concat over an empty unit list rather than a
    # recipe lookup error.
    monkeypatch.setattr(ingester_partitions, 'get_recipe_by_id', lambda rid: {})
    monkeypatch.setattr(ingester_partitions, 'get_save_admin_level', lambda r: 3)
    monkeypatch.setattr(geo_link, 'ensure_scoped_tile_link', no_tile_link)

    output = fake_output_path(None, BUILT_UNIT)
    output.write_bytes(b'fabricated output')
    return tmp_path, output


def _snapshot(folder):
    return {
        path.name: (path.stat().st_mtime_ns, path.read_bytes())
        for path in folder.iterdir()
    }


def test_a_built_unit_skips_and_says_so(built, capsys):
    folder, output = built
    before = _snapshot(folder)
    ingester = _bare([BUILT_UNIT])

    ingester.ingest(reprocess=False)

    assert ingester.admin_ids_to_save == []
    assert not hasattr(ingester, 'partition_ids_to_download')
    assert _snapshot(folder) == before
    out = capsys.readouterr().out
    assert RECIPE_ENTITY in out
    assert BUILT_UNIT in out
    assert 'already have outputs' in out


def test_a_childless_unit_skips_with_its_own_reason(built, capsys):
    folder, _ = built
    before = _snapshot(folder)
    ingester = _bare(['XB'])

    ingester.ingest(reprocess=False)

    assert _snapshot(folder) == before
    out = capsys.readouterr().out
    assert 'covers no unit at the save level' in out
    assert 'already have outputs' not in out


def test_an_unknown_unit_still_raises(built):
    folder, _ = built
    before = _snapshot(folder)

    with pytest.raises(ValueError, match='re-mint may have renamed'):
        _bare(['XZ']).ingest(reprocess=False)

    assert _snapshot(folder) == before
