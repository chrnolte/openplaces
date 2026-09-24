"""A joined value keeps the name of the source that originally supplied
it, not the name of the table it was read out of.

`io.redaction.withhold` withholds a restricted source's values cell by
cell by matching that name, so a value taken from a parcel spine that
reads `spine` hides the county source behind it and cannot be withheld.

Every value here is fabricated.
"""

import pandas as pd
import pytest

from openplaces.core.schema import Entity
from openplaces.io.harmonizer import HarmonizeState, links


def _state(spine):
    return HarmonizeState(
        recipe={'entity': Entity('transaction')},
        admin_id='XX-YY-ZZ',
        verbose=False,
        timer=None,
        spine=spine,
    )


def _spine():
    return pd.DataFrame({'parcel_id_local': ['p1', 'p2', 'p3']})


def _link(monkeypatch, ref, **kwargs):
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref.copy())
    opts = dict(
        mode='attributes',
        columns=['land_value'],
        track_provenance=['land_value'],
    )
    opts.update(kwargs)
    return links.link_by_id(_state(_spine()), 'US_parcel-spine-2026', **opts).spine


def test_the_original_source_survives_a_second_hop(monkeypatch):
    # The reference records where its own values came from; that name
    # is what a delivery must be able to withhold on.
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['p1', 'p2'],
            'land_value': [10, 20],
            'land_value_source': ['edgecombecounty', 'nconemap'],
        }
    )
    out = _link(monkeypatch, ref)
    assert out['land_value_source'].tolist()[:2] == ['edgecombecounty', 'nconemap']


def test_a_reference_with_no_sidecar_falls_back_to_the_recipe(monkeypatch):
    # Nothing more original to reach, so naming the recipe is honest.
    ref = pd.DataFrame({'parcel_id_local': ['p1'], 'land_value': [10]})
    out = _link(monkeypatch, ref)
    assert out['land_value_source'][0] == 'spine'


def test_a_row_the_reference_cannot_place_still_gets_the_recipe_name(
    monkeypatch,
):
    # Partial upstream provenance must not blank the rest: p2's value
    # came from this reference even though it does not say whose it was.
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['p1', 'p2'],
            'land_value': [10, 20],
            'land_value_source': ['edgecombecounty', None],
        }
    )
    out = _link(monkeypatch, ref)
    assert out['land_value_source'].tolist()[:2] == ['edgecombecounty', 'spine']


def test_a_column_not_tracked_gets_no_sidecar(monkeypatch):
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['p1'],
            'land_value': [10],
            'land_value_source': ['edgecombecounty'],
        }
    )
    out = _link(monkeypatch, ref, track_provenance=None)
    assert 'land_value_source' not in out.columns


@pytest.mark.parametrize('fill_only', [False, True])
def test_provenance_marks_only_the_cells_this_pass_wrote(monkeypatch, fill_only):
    # A gap-filling pass must not claim a value an earlier source set.
    spine = _spine()
    spine['land_value'] = [99, None, None]
    spine['land_value_source'] = ['earlier', None, None]
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['p1', 'p2'],
            'land_value': [10, 20],
            'land_value_source': ['edgecombecounty', 'edgecombecounty'],
        }
    )
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref.copy())
    out = links.link_by_id(
        _state(spine),
        'US_parcel-spine-2026',
        mode='attributes',
        columns=['land_value'],
        track_provenance=['land_value'],
        fill_only=fill_only,
    ).spine
    if fill_only:
        assert out['land_value'][0] == 99
        assert out['land_value_source'][0] == 'earlier'
    assert out['land_value_source'][1] == 'edgecombecounty'
