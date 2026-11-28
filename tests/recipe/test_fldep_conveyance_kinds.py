"""Florida DEP's acquisitions say what kind of conveyance each was.

FUNDING_SOURCE_NAME names the fund that paid, except for four values
that name a kind of transfer instead (a donation, an exchange, a
transfer between agencies). The ingest reads the kind into
`conveyance_type`, and Florida's curate patch grades each kind into the
existing arm's-length columns, so no Florida-only column is needed.

All values below are fabricated.
"""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_indicators
from openplaces.io.transform import apply_transformations
from openplaces.recipe import get_recipe_by_id

INGEST = 'US-FL_transaction-fldep-2026'
PATCH = 'US-FL_transaction-openplaces-2026'


def _kind_transformation():
    recipe = get_recipe_by_id(INGEST)
    steps = [t for t in recipe['transformations'] if t['output'] == 'conveyance_type']
    return {'transformations': steps}


def test_funding_source_names_the_kind_of_conveyance():
    frame = pd.DataFrame(
        {
            'FUNDING_SOURCE_NAME': [
                'Some Trust Fund (STF)',
                'Agency Funded-Other',
                'Donation Value',
                'Exchange Value',
                'Settlement Agreement Exchange',
                'Agency Transfer',
                'Some Trust Fund (STF);Another Fund',
                'Donation Value;Donation Value',
                'Some Trust Fund (STF);Donation Value',
                'Unknown',
                None,
            ]
        }
    )
    out = apply_transformations(frame, _kind_transformation(), silent=True)
    assert out['conveyance_type'].tolist()[:8] == [
        'acquisition',
        'acquisition',
        'donation',
        'exchange',
        'exchange',
        'agency_transfer',
        'acquisition',
        'donation',
    ]
    # A list mixing kinds, 'Unknown' and an empty cell state no kind.
    assert out['conveyance_type'].iloc[8:].isna().all()


def test_florida_grades_each_kind_and_keeps_a_coded_grade():
    patch = get_recipe_by_id(PATCH)
    indicators = patch['pipeline_patch'][0]['step']['indicators']
    curated = pd.DataFrame(
        {
            'sale_qualification_code': [
                None,
                None,
                None,
                None,
                None,
                "Qualified arm's length - documented evidence",
            ],
            'conveyance_type': [
                'acquisition',
                'donation',
                'exchange',
                'agency_transfer',
                None,
                None,
            ],
        }
    )
    state = CurateState(
        recipe=patch,
        entity_recipe={},
        admin_id=AdminId('US', 'FL'),
        verbose=False,
        timer=None,
        curated=curated,
    )
    out = derive_indicators(state, indicators).curated

    assert out['sale_arms_length_confidence'].tolist()[:4] == [0.0] * 4
    assert pd.isna(out['sale_arms_length_confidence'].iloc[4])
    assert out['sale_arms_length_confidence'].iloc[5] == 1.0
    assert out['sale_qualification_group'].tolist()[:4] == [
        'government_transfer',
        'not_a_sale',
        'not_a_sale',
        'not_a_sale',
    ]
    assert pd.isna(out['sale_qualification_group'].iloc[4])
    assert out['sale_qualification_group'].iloc[5] == 'qualified'
