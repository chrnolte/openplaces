"""Florida DEP's SOLARIS acquisitions stay internal.

The layer states no licence, only an "as is" disclaimer, and counsel
question CAP-1 is open: until it is answered the deals feed internal
validation only. Two guards keep them out of every delivery, and these
tests pin both:

1. Both recipes are held, so no spine, curate recipe or dependency scan
   discovers them; only the deals recipe names its ingest by id, and it
   reads nothing else.
2. The source is recorded as restricted, so a delivery that were ever
   pointed at the deals withholds every value the source supplied.

All values below are fabricated.
"""

import pandas as pd
import pytest

from openplaces.core.held_recipes import is_held
from openplaces.diagnostics import find_recipes
from openplaces.io.delivery.redaction import withhold
from openplaces.io.delivery.terms import restricted_inputs
from openplaces.recipe import get_recipe_by_id, get_recipe_dependencies

INGEST = 'US-FL_transaction-fldep-2026'
DEALS = 'US-FL_transaction-fldep-2026_deals'
COUNTY = 'US-FL-LF'


@pytest.mark.parametrize('recipe_id', [INGEST, DEALS])
def test_both_recipes_are_held_and_record_the_restriction(recipe_id):
    recipe = get_recipe_by_id(recipe_id)
    source = recipe['entity'].source

    assert is_held(recipe_id, recipe)
    assert source.source_id == 'fldep'
    assert source.redistribution_restricted is True
    assert 'CAP-1' in source.license
    assert source.terms_url


def test_no_auto_discovery_finds_them():
    discoverable = set(find_recipes('transaction')['recipe_id'])
    held = set(find_recipes('transaction', include_held=True)['recipe_id'])

    assert {INGEST, DEALS} <= held
    assert not {INGEST, DEALS} & discoverable


@pytest.mark.parametrize(
    'recipe_id',
    [
        'US_transaction-spine-2026',
        'US_transaction-openplaces-2026',
        'US-FL_transaction-openplaces-2026',
    ],
)
def test_the_national_transaction_lane_never_reads_them(recipe_id):
    upstream = {
        edge.upstream_recipe_id
        for edge in get_recipe_dependencies(recipe_id, admin_id=COUNTY)
    }
    assert INGEST not in upstream and DEALS not in upstream
    # Only this source's ids: a held recipe present on one machine
    # (Shovels, git-ignored) is reported here too, by design.
    found = {row['recipe_id'] for row in restricted_inputs(recipe_id, [COUNTY])}
    assert not found & {INGEST, DEALS}


def test_the_deals_read_only_their_own_ingest():
    sources = next(
        step['sources']
        for step in get_recipe_by_id(DEALS)['pipeline']
        if step['step'] == 'union_spine_sources'
    )
    assert {'auto_discover': False} in sources
    assert [s['recipe_id'] for s in sources if 'recipe_id' in s] == [INGEST]
    upstream = {
        edge.upstream_recipe_id
        for edge in get_recipe_dependencies(DEALS, admin_id=COUNTY)
        if edge.kind != 'admin_recipe_id'
    }
    assert upstream <= {INGEST, 'US_parcel-spine-2026', None}


def test_a_delivery_withholds_every_solaris_value():
    restricted = restricted_inputs(DEALS, [COUNTY])
    entry = next(e for e in restricted if e['source_id'] == 'fldep')
    assert 'redistribution restricted' in entry['reasons']

    solaris = [
        'transaction_id_source',
        'price',
        'price_raw',
        'sale_date',
        'sale_year',
        'sale_month',
        'area_ha_nominal',
        'conveyance_type',
        'price_period',
        'source_date',
        'holding_agency',
        'conservation_status',
    ]
    assert {'holding_agency', 'conservation_status'} <= set(entry['attributes'])
    frame = pd.DataFrame(
        {
            'admin3_id': [COUNTY, COUNTY],
            'geometry_source': ['fldep', 'parcel'],
            'transaction_id_source': ['999999', '888888'],
            'price': [123456.0, 654321.0],
            'price_raw': ['123456', '654321'],
            'sale_date': pd.to_datetime(['1999-01-02', '2001-03-04']),
            'sale_year': [1999.0, 2001.0],
            'sale_month': [1.0, 3.0],
            'area_ha_nominal': [12.5, 3.25],
            'conveyance_type': ['acquisition', 'acquisition'],
            'price_period': ['total', 'total'],
            'source_date': pd.to_datetime(['2026-01-01', '2026-01-01']),
            'holding_agency': pd.Categorical(['DEP', 'SFWMD']),
            'conservation_status': pd.Categorical(['Y', 'N']),
        },
        index=pd.Index(['US-FL-LF_999999', 'US-FL-LF_888888'], name='id'),
    )

    out, counts = withhold(frame, restricted, 'admin3_id')

    # The row whose outline is the source's goes whole; on the other
    # row every value the source maps is emptied.
    assert 'US-FL-LF_999999' not in out.index
    assert counts['fldep']['rows'] == 1
    leaked = [c for c in solaris if out[c].notna().any()]
    assert not leaked, f'SOLARIS values survive a delivery: {leaked}'
