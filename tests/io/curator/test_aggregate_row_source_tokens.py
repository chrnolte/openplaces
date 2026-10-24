"""A value gathered from a property spine names the source of its row.

A property spine records each row's source in `source`, not a sidecar
per value. The gathered column's `{col}_source` takes it, reduced to the
source id, so that a delivery's withholding still finds a restricted
roll's values (Edgecombe County NC's parcels and the units split off
them are the case). Values are fabricated.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState, aggregation
from openplaces.io.delivery.redaction import find_restricted

EDGECOMBE = {
    'source_id': 'edgecombecounty',
    'entity_type': 'parcel',
    'admin_id': 'US-NC-EDG',
    'attributes': ['land_value', 'improvement_value'],
    'layer_source': True,
}


def _state(curated):
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US-NC-EDG'),
        verbose=False,
        timer=None,
        curated=curated,
    )


def _patch(monkeypatch, rows):
    monkeypatch.setattr(
        aggregation,
        'get_entities',
        lambda recipe_id, admin_id, columns=None, missing='raise': rows[
            [c for c in columns if c in rows.columns]
        ].copy(),
    )
    monkeypatch.setattr(
        aggregation,
        'describe_recipe',
        lambda recipe_id, admin_id: pd.DataFrame(index=rows.columns),
    )


@pytest.fixture
def units(monkeypatch):
    rows = pd.DataFrame(
        {
            'parcel_id_local': ['L1', 'L1', 'L2'],
            'improvement_value': [100000.0, 50000.0, 70000.0],
            'source': [
                'edgecombecounty:units',
                'edgecombecounty:units',
                'US-NC-EDG_property-exampleroll-2026+edgecombecounty:units',
            ],
        }
    )
    _patch(monkeypatch, rows)


def test_the_token_is_the_source_id_of_the_rows(units):
    curated = pd.DataFrame({'parcel_id_local': ['L1', 'L2', 'L3']})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'US_property-spine-2026', columns={'improvement_value': 'sum'}
    )
    sources = state.curated['improvement_value_source']
    assert sources.iloc[0] == 'edgecombecounty'
    # A recipe id label becomes its source id; parts are kept once.
    assert sources.iloc[1] == 'exampleroll+edgecombecounty'
    # No value written, no provenance.
    assert pd.isna(sources.iloc[2])


def test_withholding_finds_the_restricted_roll(units):
    curated = pd.DataFrame(
        {
            'parcel_id_local': ['L1'],
            'admin3_id': ['US-NC-EDG'],
            # The lot row's own figure is missing after the stacked-units
            # split; its units carry it.
            'improvement_value': [None],
            'improvement_value_source': ['nconemap'],
        }
    )
    state = aggregation.aggregate_from_entities(
        _state(curated),
        'US_property-spine-2026',
        columns={'improvement_value': 'sum'},
        recover=['improvement_value'],
    )
    assert state.curated['improvement_value'].iloc[0] == 150000.0
    found = find_restricted(state.curated, [EDGECOMBE], 'admin3_id')
    assert found['cells']['improvement_value'].iloc[0]


def test_a_verbatim_units_label_no_longer_clears_it():
    # The contrast the normalization exists for: copied as written, the
    # label read as a different source and exonerated the cell. Since
    # 2026-10-23 only a token naming a declared source id exonerates, so
    # an unnormalized label is withheld too.
    frame = pd.DataFrame(
        {
            'admin3_id': ['US-NC-EDG'],
            'improvement_value': [150000.0],
            'improvement_value_source': ['edgecombecounty:units'],
        }
    )
    found = find_restricted(frame, [EDGECOMBE], 'admin3_id')
    assert found['cells']['improvement_value'].iloc[0]


def test_an_upstream_sidecar_still_wins_over_the_row_source(monkeypatch):
    rows = pd.DataFrame(
        {
            'parcel_id_local': ['L1'],
            'improvement_value': [100000.0],
            'improvement_value_source': ['wakecounty'],
            'source': ['edgecombecounty:units'],
        }
    )
    _patch(monkeypatch, rows)
    curated = pd.DataFrame({'parcel_id_local': ['L1']})
    state = aggregation.aggregate_from_entities(
        _state(curated), 'r', columns={'improvement_value': 'sum'}
    )
    assert state.curated['improvement_value_source'].iloc[0] == 'wakecounty'
