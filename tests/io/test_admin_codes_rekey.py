"""Re-keying an admin id column by national code after a re-mint.

Only the present spine is kept, so the only safe way to move a file
written before a re-mint onto the new ids is the code the unit's own
country assigns it. These cases fabricate a spine and one file each.
"""

import stat

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openplaces.io.admin_codes import rekey


@pytest.fixture
def fabricated(tmp_path, monkeypatch):
    """A two-unit spine whose ids swapped, and a share tree with one file."""
    spine = tmp_path / 'admin-spine-2026_admin4.csv'
    spine.write_text(
        'admin4_id,name,admin4_id_admin1\n'
        'XX-AA-BB-AL,Alpha,101\n'
        'XX-AA-BB-BR,Bravo,202\n'
        'XX-AA-BB-CH,Charlie,\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(rekey, 'spine_path', lambda level: spine)
    share = tmp_path / 'share'
    monkeypatch.setattr(rekey.cfg, 'get_dir', lambda root: tmp_path / root)
    folder = share / 'XX' / 'AA' / 'BB'
    folder.mkdir(parents=True)
    path = folder / 'units.parquet'
    # Before the re-mint Alpha held BR and Bravo held AL; the third row's
    # code is one the spine does not carry.
    table = pa.table(
        {
            'admin4_id': ['XX-AA-BB-BR', 'XX-AA-BB-AL', 'XX-AA-BB-ZZ'],
            'census_subdivision_id': ['101', '202', '999'],
            'value': [1, 2, 3],
        }
    ).replace_schema_metadata({b'fingerprint': b'abc'})
    pq.write_table(table, path)
    return path


def test_ids_are_rekeyed_by_code(fabricated):
    report = rekey.rekey_files(
        4, 'XX', 'census_subdivision_id', apply=True, verbose=False
    )
    assert report['cells'].tolist() == [2]
    assert report['unknown'].tolist() == [1]
    written = pd.read_parquet(fabricated)
    assert written['admin4_id'].tolist() == [
        'XX-AA-BB-AL',
        'XX-AA-BB-BR',
        'XX-AA-BB-ZZ',
    ]


def test_a_dry_run_writes_nothing(fabricated):
    before = fabricated.read_bytes()
    report = rekey.rekey_files(
        4, 'XX', 'census_subdivision_id', apply=False, verbose=False
    )
    assert report['cells'].tolist() == [2]
    assert fabricated.read_bytes() == before


def test_schema_metadata_and_read_only_bit_survive(fabricated):
    fabricated.chmod(fabricated.stat().st_mode & ~stat.S_IWUSR)
    rekey.rekey_files(4, 'XX', 'census_subdivision_id', apply=True, verbose=False)
    assert pq.read_schema(fabricated).metadata[b'fingerprint'] == b'abc'
    assert not fabricated.stat().st_mode & stat.S_IWUSR


def test_a_code_carried_twice_is_refused(tmp_path, monkeypatch):
    spine = tmp_path / 'spine.csv'
    spine.write_text(
        'admin4_id,name,admin4_id_admin1\n'
        'XX-AA-BB-AL,Alpha,101\n'
        'XX-AA-BB-BR,Bravo,101\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(rekey, 'spine_path', lambda level: spine)
    with pytest.raises(ValueError, match='more than one unit'):
        rekey.code_to_admin_id(4, 'XX')
