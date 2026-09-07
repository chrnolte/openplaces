"""The link-sidecar readers must be generic over the reference entity.

collect_link_ids resolves any reference recipe through entity_type but
read the sidecar's key as the literal 'parcel_id', and
_apportioned_sources hardcoded the 'parcel.' provenance lane.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.io.curator import CurateState


def _state(curated, admin_id=None):
    return CurateState(
        recipe={'entity': {'entity_type': 'footprint'}},
        entity_recipe={},
        admin_id=admin_id,
        verbose=False,
        timer=None,
        curated=curated,
    )


def test_collect_link_ids_reads_a_non_parcel_reference_key(tmp_path, monkeypatch):
    """A non-parcel reference must collect, not raise KeyError.

    Recipe resolution here is generic over entity_type; only the sidecar
    column read was pinned to the literal 'parcel_id'.
    """
    from openplaces.io.curator import evidence
    from openplaces.io.harmonizer import links as harmonizer_links

    sidecar = tmp_path / 'links.parquet'
    pd.DataFrame(
        {
            'footprint_id': ['f1', 'f1', 'f2'],
            'building_id': ['b1', 'b2', 'b3'],
            'link': [1, 1, 1],
            'area_intersection_m2': [9.0, 4.0, 1.0],
        }
    ).to_parquet(sidecar)

    monkeypatch.setattr(
        harmonizer_links,
        '_resolve_reference_recipe',
        lambda recipe_id, entity_type, admin_id: ('US_building-ref-2026', None),
    )
    monkeypatch.setattr(
        evidence, 'get_link_owner_recipe_id', lambda recipe: 'US_footprint-geospine'
    )
    monkeypatch.setattr(
        evidence, 'get_entity_link_path', lambda *args, **kwargs: sidecar
    )

    curated = pd.DataFrame(
        {'value': [1, 2]}, index=pd.Index(['f1', 'f2'], name='footprint_id')
    )
    result = evidence.collect_link_ids(
        _state(curated, admin_id='US-NC-ALA'),
        entity_type='building',
        column='building_id_all',
    ).curated

    assert result.loc['f1', 'building_id_all'] == 'b1|b2'
    assert result.loc['f2', 'building_id_all'] == 'b3'


def test_collect_link_ids_names_the_missing_reference_key(tmp_path, monkeypatch):
    from openplaces.io.curator import evidence
    from openplaces.io.harmonizer import links as harmonizer_links

    sidecar = tmp_path / 'links.parquet'
    pd.DataFrame({'footprint_id': ['f1'], 'other_id': ['x']}).to_parquet(sidecar)

    monkeypatch.setattr(
        harmonizer_links,
        '_resolve_reference_recipe',
        lambda recipe_id, entity_type, admin_id: ('US_building-ref-2026', None),
    )
    monkeypatch.setattr(
        evidence, 'get_link_owner_recipe_id', lambda recipe: 'US_footprint-geospine'
    )
    monkeypatch.setattr(
        evidence, 'get_entity_link_path', lambda *args, **kwargs: sidecar
    )

    curated = pd.DataFrame({'value': [1]}, index=pd.Index(['f1'], name='footprint_id'))
    with pytest.raises(KeyError, match='building_id'):
        evidence.collect_link_ids(
            _state(curated, admin_id='US-NC-ALA'),
            entity_type='building',
            column='building_id_all',
        )


def test_apportioned_provenance_names_the_reference_lane():
    """The lane prefix must be the reference's entity type, not 'parcel.'."""
    from openplaces.io.curator.evidence import _apportioned_sources

    pairs = pd.DataFrame(
        {
            'footprint_id': ['f1', 'f2'],
            'parcel_id': ['b1', 'b2'],
            'area_intersection_m2': [5.0, 5.0],
        }
    )
    ref = pd.DataFrame(
        {
            'building_id': ['b1', 'b2'],
            'structure_value': [10.0, 20.0],
            'structure_value_source': ['nsi', 'nsi+imputed'],
        }
    )
    index = pd.Index(['f1', 'f2'], name='footprint_id')

    tokens = _apportioned_sources(
        pairs,
        ref,
        'building_id',
        ['structure_value'],
        'footprint_id',
        index,
        lane='building',
    )['structure_value']

    assert tokens.tolist() == ['building.nsi', 'building.nsi+imputed']
