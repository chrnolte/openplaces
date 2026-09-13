"""Per-state admin crosswalk sidecars agree with the present spine.

A crosswalk maps a source's own county or town codes to spine ids, and
the ingester splits a statewide download by it. A re-mint recycles ids,
so a crosswalk that is not regenerated afterwards can send two source
codes to one id (two counties merged into one file) or name an id the
spine no longer has. Both rotted every state's file after the 2026-08
re-mint; this pins them.
"""

import re

import pandas as pd
import pytest

from openplaces.path import recipe_roots, spine_path

CROSSWALKS = sorted(
    path for root in recipe_roots() for path in root.glob('**/*_admin3-crosswalk*.csv')
)


def _spine():
    frame = pd.read_csv(spine_path(3), dtype=str, keep_default_na=False)
    return dict(zip(frame['admin3_id'], frame['admin3_id_admin1']))


@pytest.mark.parametrize('path', CROSSWALKS, ids=lambda p: p.stem)
def test_crosswalk_ids_are_in_the_spine_and_codes_agree(path):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    code_of = _spine()
    unknown = sorted(set(table['admin3_id']) - set(code_of))
    assert not unknown, f'{path.name}: ids the spine lacks: {unknown[:8]}'
    source = [c for c in table.columns if c != 'admin3_id'][0]
    codes = table[source][table[source] != '']
    assert not codes.duplicated().any(), f'{path.name}: source code listed twice'
    if codes.str.fullmatch(r'\d{5}').all():
        expected = table['admin3_id'].map(code_of)
        wrong = table[expected != table[source]]
        assert wrong.empty, f'{path.name}: FIPS disagrees with the spine:\n{wrong}'


def test_every_crosswalk_is_named_for_a_recipe_beside_it():
    for path in CROSSWALKS:
        recipe = re.sub(r'_admin3-crosswalk.*$', '', path.stem)
        assert (path.parent / f'{recipe}.yaml').exists(), path.name
