"""`impute_land_value(open_only=True)` learns only from open rows.

Every row here is fabricated: one county, two value sources, one of
which (`restrictedroll`) records restricted redistribution. The
restricted inputs the step asks the delivery for are patched in, so no
recipe catalog is read.
"""

import numpy as np
import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import _apportioned_sources
from openplaces.io.curator.land_value import impute_land_value
from openplaces.io.delivery import redaction, terms

UNIT = 'US-XX-AAA'
RESTRICTED = 'restrictedroll'
OPEN_SOURCE = 'openroll'

# Read as a layer, so rule 5 withholds every estimate in the unit not
# marked open; mapping no value column, so rules 3 and 4 stay out of
# the way and only the tokens decide.
SOURCE = {
    'source_id': RESTRICTED,
    'recipe_id': f'{UNIT}_property-{RESTRICTED}-2026',
    'license': 'fabricated',
    'terms_url': None,
    'reasons': ['redistribution restricted'],
    'entity_type': 'property',
    'admin_id': UNIT,
    'attributes': ['n_bedrooms'],
    'layer_source': True,
    'via_entities': [],
    'team_sharing_permitted': None,
}
KNOWN = frozenset({RESTRICTED, OPEN_SOURCE})


@pytest.fixture(autouse=True)
def _restricted_unit(monkeypatch):
    monkeypatch.setattr(terms, 'restricted_inputs', lambda recipe, ids: [SOURCE])
    monkeypatch.setattr(terms, 'known_source_ids', lambda: KNOWN)


def _rows(spec):
    """Rows from (id, source, land, improvement, area_ha) tuples."""
    ids, sources, land, improvement, area = zip(*spec, strict=True)
    frame = pd.DataFrame(
        {
            'land_value': land,
            'improvement_value': improvement,
            'area_ha': area,
            'footprint_area_m2_in_parcel': 150.0,
            'land_use_class': 'Single-Family',
            'use_group_combined': 'residential',
            'city': 'Testville',
        },
        index=pd.Index(ids, name='parcel_id'),
    )
    for column in ('land_value', 'improvement_value'):
        frame[f'{column}_source'] = pd.Series(sources, index=frame.index, dtype=object)
    return frame


def _impute(frame, open_only):
    state = CurateState(
        recipe={'recipe_id': 'US_parcel-test-2026'},
        entity_recipe={},
        admin_id=AdminId(UNIT),
        verbose=False,
        timer=None,
        curated=frame.copy(),
    )
    return impute_land_value(
        state,
        estimator='rate',
        rate_exponent=1.0,
        min_group_size=3,
        open_only=open_only,
    ).curated


# Open donors price land at 100,000 per ha, restricted donors at
# 10,000,000: any restricted donor in the pool moves the median.
DONORS = [(f'o{i}', OPEN_SOURCE, 100_000.0, 300_000.0, 1.0) for i in range(4)] + [
    (f'r{i}', RESTRICTED, 10_000_000.0, 300_000.0, 1.0) for i in range(6)
]
CANDIDATES = [
    ('open_candidate', OPEN_SOURCE, np.nan, 500_000.0, 0.5),
    ('restricted_candidate', RESTRICTED, np.nan, 500_000.0, 0.5),
]


def test_a_restricted_donor_is_never_used():
    result = _impute(_rows(DONORS + CANDIDATES), open_only=True)

    # The open donors' rate alone: 100,000 per ha over 0.5 ha.
    assert result.loc['open_candidate', 'land_value_imputed'] == pytest.approx(50_000)
    assert result.loc['open_candidate', 'improvement_value_imputed'] == pytest.approx(
        450_000
    )
    # With every donor the restricted rate dominates the median.
    pooled = _impute(_rows(DONORS + CANDIDATES), open_only=False)
    assert pooled.loc['open_candidate', 'land_value_imputed'] == pytest.approx(
        5_000_000
    )


def test_an_open_estimate_carries_the_open_token_and_survives_redaction():
    result = _impute(_rows(DONORS + CANDIDATES), open_only=True)
    token = result.loc['open_candidate', 'land_value_imputed_source']
    assert token.split('+')[-2:] == [redaction.OPEN, 'imputed']

    frame = pd.DataFrame(result)
    frame['admin3_id'] = UNIT
    shipped, _ = redaction.withhold(frame, [SOURCE], 'admin3_id', KNOWN)
    assert shipped.loc['open_candidate', 'land_value_imputed'] == pytest.approx(50_000)
    assert shipped.loc['open_candidate', 'improvement_value_imputed'] == pytest.approx(
        450_000
    )

    # The same row estimated from every donor is withheld (rule 5).
    pooled = pd.DataFrame(_impute(_rows(DONORS + CANDIDATES), open_only=False))
    pooled['admin3_id'] = UNIT
    redacted, _ = redaction.withhold(pooled, [SOURCE], 'admin3_id', KNOWN)
    assert pd.isna(redacted.loc['open_candidate', 'land_value_imputed'])
    assert redacted.loc['open_candidate', 'land_value_imputed_source'] == (
        redaction.WITHHELD
    )


def test_a_cell_with_no_open_donor_stays_missing():
    restricted_only = [d for d in DONORS if d[1] == RESTRICTED]
    result = _impute(_rows(restricted_only + CANDIDATES), open_only=True)
    for row in ('open_candidate', 'restricted_candidate'):
        assert pd.isna(result.loc[row, 'land_value_imputed'])
        assert pd.isna(result.loc[row, 'land_value_imputed_source'])
    # Recorded values still pass through, under their own token.
    assert result.loc['r0', 'land_value_imputed'] == pytest.approx(10_000_000)
    assert result.loc['r0', 'land_value_imputed_source'] == RESTRICTED


def test_a_restricted_candidate_is_not_estimated_from_open_donors():
    result = _impute(_rows(DONORS + CANDIDATES), open_only=True)
    assert pd.isna(result.loc['restricted_candidate', 'land_value_imputed'])
    # Its own recorded improvement figure still passes through.
    assert result.loc[
        'restricted_candidate', 'improvement_value_imputed'
    ] == pytest.approx(500_000)


def test_the_open_token_survives_apportionment_only_when_every_estimate_is_open():
    pairs = pd.DataFrame(
        {
            'footprint_id': ['f1', 'f2', 'f2'],
            'parcel_id': ['p1', 'p1', 'p2'],
            'area_intersection_m2': [5.0, 9.0, 1.0],
        }
    )
    ref = pd.DataFrame(
        {
            'parcel_id': ['p1', 'p2'],
            'improvement_value_imputed': [10.0, 20.0],
            'improvement_value_imputed_source': [
                'city__is_residential+open+imputed',
                'city__is_residential+imputed',
            ],
        }
    )
    index = pd.Index(['f1', 'f2'], name='footprint_id')
    tokens = _apportioned_sources(
        pairs,
        ref,
        'parcel_id',
        ['improvement_value_imputed'],
        'footprint_id',
        index,
    )['improvement_value_imputed']

    # f1 draws on the open estimate alone; f2's sum also holds one
    # learned from restricted values, so it is not open.
    assert tokens.tolist() == [
        'parcel.city__is_residential+open+imputed',
        'parcel.city__is_residential+imputed',
    ]
