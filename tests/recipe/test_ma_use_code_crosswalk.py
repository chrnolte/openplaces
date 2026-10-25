"""Massachusetts use-code crosswalk: DOR classes and Boston's own list.

The statewide rows follow the DOR "Property Type Classification Codes"
(April 2019): the first digit is the major class, a multiple-use code
is 0 and its classes in order. Boston's rows follow the Boston
Assessing Department's occupancy code list, where 995 is a condominium
main record, not exempt land. Codes fabricated in the shapes the towns
write them.
"""

import pandas as pd

import openplaces.io.harmonizer.links as links
from openplaces.io.harmonizer import HarmonizeState
from openplaces.recipe import get_recipe_by_id

MASSGIS = 'US-MA_parcel-massgis-2025'


def _apply(admin_id, codes):
    spine = pd.DataFrame(
        {'use_group_code': codes},
        index=pd.Index([f'p{i}' for i in range(len(codes))], name='parcel_id'),
    )
    state = HarmonizeState(
        recipe={}, admin_id=admin_id, verbose=False, timer=None, spine=spine
    )
    return links._apply_remap_csvs(state, MASSGIS).spine.set_index('use_group_code')


def test_boston_codes_follow_bostons_list():
    out = _apply(
        'US-MA-BOS', ['995', '985', '983', '925', '986', '356', '908', '025', '902']
    )
    # A condominium main record: no use group, not exempt.
    assert pd.isna(out.loc['995', 'use_group'])
    assert out.loc['995', 'use_subgroup'] == 'condominium main'
    for code in ['985', '983', '925', '986', '902']:
        assert out.loc[code, 'use_group'] == 'exempt', code
    assert out.loc['983', 'use_subgroup'] == 'cemetery'
    assert out.loc['356', 'use_group'] == 'commercial'
    assert out.loc['908', 'use_group'] == 'residential'
    assert out.loc['025', 'use_group'] == 'mixed'


def test_elsewhere_995_stays_dors_exempt_and_classes_fill_gaps():
    out = _apply(
        'US-MA-SOM',
        [
            '995',
            '3270',
            '343',
            '4021',
            '0104',
            '0322',
            '017',
            '9590',
            '959R',
            '945',
            '996',
            '925V',
            '000',
            '101',
        ],
    )
    assert out.loc['995', 'use_group'] == 'exempt'
    # Commercial and industrial condominium codes take their class.
    assert out.loc['3270', 'use_group'] == 'commercial'
    assert out.loc['343', 'use_group'] == 'commercial'
    assert out.loc['4021', 'use_group'] == 'industrial'
    # Multiple use, three- and four-digit.
    assert out.loc['0104', 'use_group'] == 'mixed'
    assert out.loc['0322', 'use_subgroup'] == 'mixed use, primarily commercial'
    assert out.loc['017', 'use_subgroup'] == 'mixed use, primarily residential'
    # Housing under exempt owners is residential, other 9xx exempt.
    assert out.loc['9590', 'use_group'] == 'residential'
    assert out.loc['959R', 'use_group'] == 'residential'
    assert out.loc['945', 'use_group'] == 'residential'
    assert out.loc['996', 'use_group'] == 'exempt'
    assert out.loc['925V', 'use_group'] == 'exempt'
    # A code no class defines stays empty; listed codes are unchanged.
    assert pd.isna(out.loc['000', 'use_group'])
    assert out.loc['101', 'use_subgroup'] == 'single family'


def test_each_key_appears_once_per_scope():
    table = get_recipe_by_id(f'{MASSGIS}_use-group-code-remap', dtype=str)
    scope = table['admin_id'].fillna('')
    assert not table.assign(scope=scope).duplicated(['use_group_code', 'scope']).any()


def test_no_commercial_or_industrial_subgroup_names_a_condominium():
    # The land-use keyword table reads "condo" as the residential
    # Condominium class.
    table = get_recipe_by_id(f'{MASSGIS}_use-group-code-remap', dtype=str)
    new = table[table['use_group_code'].str.match(r'3|4') & table['admin_id'].notna()]
    assert not new['use_subgroup'].str.contains('condo', case=False).any()
