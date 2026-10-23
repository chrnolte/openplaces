"""Maryland's account keys stay whole in every recipe keyed by ACCTID.

The bundled per-county table (geo/parcel_id_links.csv) carries rows fit
to ZTRAX APNs for every Maryland county. Applied to SDAT's account id
they cut a digit, dropped the county code and merged distinct accounts,
so each MD recipe states `instruction: {US-MD: {conv: simple}}`, which
outranks the table for every county. These tests read the real recipes
and convert fabricated account ids.
"""

from pathlib import Path

import pandas as pd
import pytest

from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.recipe import get_recipe_by_id

MD_RECIPES = [
    'US-MD_parcel-mdimap-2026',
    'US-MD_property-mdplanning-2026',
    'US-MD_property-mdplanning-2026_buildings',
    'US-MD_property-mdplanning-2026_subareas',
    'US-MD_transaction-sdat-2026',
    'US-MD_transaction-sdat-2026_segment-2',
    'US-MD_transaction-sdat-2026_segment-3',
]

# Fabricated account ids in the shapes the counties use (aggregate
# shapes measured 2026-10-23): 11 digits in Prince George's, 12 in
# Montgomery, 10 in most counties; the last two are placeholders.
FABRICATED = {
    'US-MD-PG': ['17020000001', '17020000002', '17099999999', '17099999998'],
    'US-MD-MO': ['160200000001', '160200000002'],
    'US-MD-SO': ['2002000001'],
}


def _key(recipe_id, admin_id, raw):
    recipe = get_recipe_by_id(recipe_id)
    ingester = TableIngester(
        recipe, {}, {'admin_id_to_process': admin_id}, Path('.'), None
    )
    frame = pd.DataFrame({'parcel_id_assessor': raw})
    return ingester._add_parcel_id_local(frame)['parcel_id_local'].tolist()


@pytest.mark.parametrize('recipe_id', MD_RECIPES)
def test_every_md_recipe_keeps_the_whole_account_id(recipe_id):
    instruction = get_recipe_by_id(recipe_id)['parcel_id_local']['instruction']
    assert instruction == {'US-MD': {'conv': 'simple'}}
    for admin_id, raw in FABRICATED.items():
        assert _key(recipe_id, admin_id, raw) == raw


def test_parcel_and_sales_keys_match_exactly():
    raw = FABRICATED['US-MD-PG']
    parcel = _key('US-MD_parcel-mdimap-2026', 'US-MD-PG', raw)
    sales = _key('US-MD_transaction-sdat-2026', 'US-MD-PG', raw)
    assert parcel == sales == raw
    # The two placeholder accounts stay two keys.
    assert len(set(parcel)) == len(raw)


def test_the_bundled_table_alone_would_cut_the_account_id():
    """Why the instruction is needed: the default row for the county
    truncates and pipe-joins, so no key equals the account id."""
    from openplaces.geo.ids import compute_parcel_id_local

    # Distinct accounts only: ids that collapse would trip the duplicate
    # guard, which then falls back to the raw id.
    raw = pd.Series(['17021234561', '17035678902', '17041357913'])
    default = compute_parcel_id_local(raw, admin_unit_id='US-MD-PG', kind='parcel')
    assert not set(default.dropna()) & set(raw)
    assert all('|' in key for key in default.dropna())
