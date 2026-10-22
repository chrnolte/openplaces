"""Harris County's building type table and the parcel curate patch.

The table translates HCAD's building type codes into the occupancy
groups the land-use class map reads; the US-TX-HAR patch reduces the
code onto parcels and lets the table overwrite the cohort fill.
"""

import re

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.path import recipe_path
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

PARCEL = 'US_parcel-openplaces-2026'


def _folder():
    recipe = get_recipe_by_id(PARCEL)
    return recipe_path(recipe['admin_id'], recipe['entity'], as_dir=True)


def _table():
    return pd.read_csv(
        _folder() / f'{PARCEL}_group-parcel-tx-har.csv', comment='#', dtype=str
    )


def _translate(code, table):
    for pattern, group in zip(table['pattern'], table['group_parcel'], strict=True):
        if re.search(pattern, code):
            return group
    return None


def test_every_group_is_one_the_land_use_class_map_knows():
    table = _table()
    known = pd.read_csv(_folder() / f'{PARCEL}_land-use-class-map.csv')['group']
    assert set(table['group_parcel']) <= set(known)
    assert table['match_type'].eq('regex').all()
    assert table['reviewed'].eq('true').all()


def test_codes_translate_to_the_group_their_definition_names():
    table = _table()
    expected = {
        '4211': 'Multi Family',
        '4353': 'Professional Technical Services',
        '4354': 'Professional Technical Services',
        '4640': 'Hospital',
        '4380': 'Wholesale',
        '4373': 'Retail',
        '4314': 'Hotel',
        '4372': 'Garage',
        '4416': 'Food/Drug/Chemical',
        '4612': 'Average School',
    }
    for code, group in expected.items():
        assert _translate(code, table) == group, code
    # Mixed use, utilities and storage tanks keep the cohort fill.
    for code in ('1005', '4319', '4301', '4752', '9781'):
        assert _translate(code, table) is None, code


def test_patch_reduces_the_type_onto_parcels_before_translating_it():
    recipe, _ = apply_recipe_patches(get_recipe_by_id(PARCEL), AdminId('US-TX-HAR'))
    steps = recipe['pipeline']
    names = [s['step'] for s in steps]
    reduce_at = next(
        i
        for i, s in enumerate(steps)
        if s['step'] == 'aggregate_from_entities'
        and s.get('columns') == {'building_type_code': 'agree'}
    )
    cohort_at = names.index('impute_from_group_statistic')
    translate_at = next(
        i
        for i, s in enumerate(steps)
        if s['step'] == 'translate_descriptions'
        and s.get('keywords') == 'group-parcel-tx-har'
    )
    assert reduce_at < cohort_at < translate_at == cohort_at + 1
    assert steps[translate_at]['fill_only'] is False

    other, _ = apply_recipe_patches(get_recipe_by_id(PARCEL), AdminId('US-TX-GAL'))
    assert not any(
        s.get('keywords') == 'group-parcel-tx-har' for s in other['pipeline']
    )
