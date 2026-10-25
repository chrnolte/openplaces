"""The 2020 Census P.L. 94-171 block counts: the `join_members` reader,
the table P2 field mapping and the sum check, on fabricated files.

Every code and count here is made up; only the file layout (pipe
delimited, no header, a geo file and a data segment sharing LOGRECNO)
follows the Census Bureau's legacy format.
"""

import numpy as np
import pandas as pd
import pytest

from openplaces.io.ingester._helpers import _transform_partition_key
from openplaces.io.ingester.table_read import _TableReadMixin
from openplaces.io.transform import apply_transformations
from openplaces.recipe import get_output_path, get_recipe_by_id

RID = 'US_population-census-2020pl'
GROUPS = [
    'n_people_white',
    'n_people_black',
    'n_people_asian',
    'n_people_hispanic',
    'n_people_other',
]
N_GEO_FIELDS = 97
N_SEGMENT_FIELDS = 149
P2_START = 76  # 5 header fields, then P1's 71


class _Reader(_TableReadMixin):
    def __init__(self, recipe):
        self.recipe = recipe


def _p2_row(rng):
    """Fabricated P2 counts that satisfy the table's own identities."""
    p = dict.fromkeys(range(1, 12), 0)
    for code in (2, 5, 6, 7, 8, 9, 10, 11):
        p[code] = int(rng.integers(0, 40))
    p[4] = sum(p[c] for c in range(5, 11))
    p[3] = p[4] + p[11]
    p[1] = p[2] + p[3]
    return p


def _write_state(tmp_path, blocks, rng, other_levels=1):
    """Write a geo file and segment 1 for fabricated blocks.

    Returns the P2 counts written, keyed by block code.
    """
    geo_rows, seg_rows, written = [], [], {}
    logrecno = 0
    levels = [('040', None)] * other_levels + [('750', b) for b in blocks]
    for sumlev, block in levels:
        logrecno += 1
        rec = f'{logrecno:07d}'
        geo = [''] * N_GEO_FIELDS
        geo[:10] = ['PLST', 'ZZ', sumlev, '00', '00', '000', '00', rec, '', '']
        geo[9] = block or '99'
        seg = ['0'] * N_SEGMENT_FIELDS
        seg[:5] = ['PLST', 'ZZ', '000', '01', rec]
        p = _p2_row(rng)
        for code, value in p.items():
            seg[P2_START + code - 1] = str(value)
        if block:
            written[block] = p
        geo_rows.append('|'.join(geo))
        seg_rows.append('|'.join(seg))
    # The geo file in a different row order: the join is by key.
    (tmp_path / 'zzgeo2020.pl').write_text('\n'.join(reversed(geo_rows)) + '\n')
    (tmp_path / 'zz000012020.pl').write_text('\n'.join(seg_rows) + '\n')
    return written


def _members(recipe, tmp_path):
    return [
        (tmp_path / m['file'].replace('{admin2_id_leaf}', 'zz'), m['fields'])
        for m in recipe['join_members']['members']
    ]


def _ingest_like(recipe, tmp_path):
    """Read and transform as the ingester's preprocessing does."""
    df = _Reader(recipe)._read_joined_members(
        _members(recipe, tmp_path), recipe['join_members']['key'], 'latin-1'
    )
    df = df.rename(columns={v: k for k, v in recipe['columns'].items()})
    df = df.query(recipe['query'])
    df = apply_transformations(df, recipe, silent=True)
    return df.set_index(recipe['set_index'])


@pytest.fixture(scope='module')
def recipe():
    return get_recipe_by_id(RID)


def test_recipe_maps_the_documented_p2_fields(recipe):
    assert recipe['columns'] == {
        'census_block_geoid': 'GEOCODE',
        'n_people': 'P0020001',
        'n_people_white': 'P0020005',
        'n_people_black': 'P0020006',
        'n_people_asian': 'P0020008',
        'n_people_hispanic': 'P0020002',
    }
    (other,) = [t for t in recipe['transformations'] if t['output'] == GROUPS[-1]]
    assert other['operation'] == 'sum'
    assert other['inputs'] == ['P0020007', 'P0020009', 'P0020010', 'P0020011']
    geo, segment = (m['fields'] for m in recipe['join_members']['members'])
    assert geo == {2: 'SUMLEV', 7: 'LOGRECNO', 9: 'GEOCODE'}
    # P0020001 sits right after P1's 71 cells; each code at its offset.
    for position, name in segment.items():
        if name.startswith('P002'):
            assert position == P2_START + int(name[-4:]) - 1, name
    assert segment[4] == 'LOGRECNO'


def test_groups_sum_to_the_total_on_every_block(recipe, tmp_path):
    rng = np.random.default_rng(7)
    # Fabricated block codes; a leading zero must survive.
    blocks = [f'0{i:014d}' for i in range(1, 41)]
    written = _write_state(tmp_path, blocks, rng)
    df = _ingest_like(recipe, tmp_path)
    assert len(df) == len(blocks)
    assert (df[GROUPS].sum(axis=1) == df['n_people']).all()
    for block, p in written.items():
        row = df.loc[block]
        assert row['n_people'] == p[1]
        assert row['n_people_hispanic'] == p[2]
        assert row['n_people_white'] == p[5]
        assert row['n_people_black'] == p[6]
        assert row['n_people_asian'] == p[8]
        assert row['n_people_other'] == p[7] + p[9] + p[10] + p[11]


def test_key_is_the_15_digit_block_code_as_text(recipe, tmp_path):
    rng = np.random.default_rng(11)
    blocks = ['019990000001001', '019990000001002']
    _write_state(tmp_path, blocks, rng, other_levels=3)
    df = _ingest_like(recipe, tmp_path)
    assert df.index.name == 'census_block_geoid'
    assert sorted(df.index) == blocks
    assert df.index.is_unique
    # Only tabulation blocks (summary level 750) are kept.
    assert df['SUMLEV'].eq('750').all()
    assert (df['admin2_id_admin1'] == '01').all()


def test_join_is_strict(tmp_path):
    fields_a = {0: 'K', 1: 'A'}
    fields_b = {0: 'K', 1: 'B'}
    (tmp_path / 'a.txt').write_text('001|x\n002|y\n')
    (tmp_path / 'b.txt').write_text('002|5\n001|4\n')
    reader = _Reader({'delimiter': '|'})
    members = [(tmp_path / 'a.txt', fields_a), (tmp_path / 'b.txt', fields_b)]
    df = reader._read_joined_members(members, 'K')
    assert df.set_index('K')['B'].to_dict() == {'001': 4, '002': 5}

    (tmp_path / 'b.txt').write_text('002|5\n003|4\n')
    with pytest.raises(ValueError, match='not in every member'):
        reader._read_joined_members(members, 'K')

    (tmp_path / 'b.txt').write_text('001|5\n001|4\n002|1\n')
    with pytest.raises(ValueError, match='repeats K'):
        reader._read_joined_members(members, 'K')

    with pytest.raises(ValueError, match='does not name'):
        reader._read_joined_members([(tmp_path / 'a.txt', {1: 'A'})], 'K')


def test_partition_key_lower():
    assert _transform_partition_key('MA', {'operation': 'lower'}) == 'ma'
    assert (
        _transform_partition_key(
            'District of Columbia', {'operation': 'replace', 'args': [' ', '_']}
        )
        == 'District_of_Columbia'
    )


@pytest.mark.parametrize(
    'admin_id',
    ['US-MA', 'US-NC', 'US-TX', 'US-WI', 'US-DC', 'US-MD', 'US-VA', 'US-FL'],
)
def test_ingested_blocks_sum_to_their_total(admin_id):
    path = get_output_path(RID, admin_id)
    if not path.exists():
        pytest.skip(f'{RID} is not ingested for {admin_id} in this data root')
    df = pd.read_parquet(path)
    assert df.index.name == 'census_block_geoid'
    assert df.index.is_unique
    assert df.index.str.fullmatch(r'\d{15}').all()
    assert (df[GROUPS].sum(axis=1) == df['n_people']).all()
