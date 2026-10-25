"""Maryland's common area flag is read from the legal text, which is not kept.

Every legal description below is fabricated.
"""

import pandas as pd

from openplaces.io.transform import apply_transformations
from openplaces.recipe import get_recipe_by_id
from openplaces.recipe.tables import get_recipe_output_columns

RECIPE_ID = 'US-MD_parcel-mdimap-2026'


def _flags(legal):
    frame = pd.DataFrame(legal, columns=['LEGAL1', 'LEGAL2', 'LEGAL3'])
    out = apply_transformations(frame, get_recipe_by_id(RECIPE_ID), silent=True)
    return out['is_common_area'].tolist()


def test_common_area_and_open_space_are_flagged():
    flags = _flags(
        [
            ['PARCEL Q COMMON AREA', 'EXAMPLE PLACE', 'PLAT 99999'],
            ['PAR B', 'OPEN SPACE', None],
            ['LT 7 BLK C', 'EXAMPLE PLACE', None],
            ['PARCEL R COMMON', 'AREA EXAMPLE', None],
            [None, None, None],
            ['COMMONWEALTH ROAD', 'AREAWAY', None],
        ]
    )
    assert flags == [True, True, False, True, False, False]


def test_the_legal_text_is_never_part_of_the_output():
    recipe = get_recipe_by_id(RECIPE_ID)
    produced = get_recipe_output_columns(recipe)
    assert 'is_common_area' in produced
    assert {'subdivision_code', 'plat_id_local'} <= produced
    assert not {c for c in produced if 'legal' in c.lower()}
    assert 'legal_text' in recipe['drop_columns']
