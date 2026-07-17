"""Tests for re-routing FEMA occupancy as parcel-level evidence.

Covers the reusable harmonizer kernels (dominant-by-area attribution, the
source-aware suffix, the per-source value remap), the auto-generated provenance
vocabulary, and that FEMA occupancy orders into the source band.
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.harmonizer.attributes import _dominant_by_area, _resolve_suffix


class _Entity:
    def __init__(self, entity_type):
        self.entity_type = entity_type


class _FakeState:
    def __init__(self, spine_entity_type):
        self.recipe = {'entity': _Entity(spine_entity_type)}


def test_dominant_by_area_picks_largest_total_area():
    # Parcel p: two Manufactured (60+50=110 m2) vs one Single-Family (100 m2) ->
    # Manufactured wins by total area. Parcel q: a single Retail footprint.
    attrs = pd.DataFrame(
        {
            'occupancy_type': [
                'Manufactured',
                'Manufactured',
                'Single-Family',
                'Retail',
            ],
            'area_intersection_m2': [60.0, 50.0, 100.0, 30.0],
        },
        index=pd.Index(['p', 'p', 'p', 'q'], name='parcel_id_local'),
    )
    dominant, joined = _dominant_by_area(
        attrs.reset_index(), 'parcel_id_local', 'occupancy_type', 'area_intersection_m2'
    )
    assert dominant.loc['p'] == 'Manufactured'
    assert dominant.loc['q'] == 'Retail'
    # `_all` lists every class by descending area.
    assert joined.loc['p'].startswith('Manufactured')
    assert 'Single-Family' in joined.loc['p']
    # Parcel q has only one distinct class -> `_all` adds nothing beyond
    # `dominant`, so it's left missing rather than repeating 'Retail'.
    assert pd.isna(joined.loc['q'])


def test_resolve_suffix_is_source_aware():
    parcel_spine = _FakeState('parcel')
    footprint_spine = _FakeState('footprint')
    building_spine = _FakeState('building')

    # FEMA footprints attributed to a parcel spine -> entity + source.
    assert (
        _resolve_suffix('US_footprint-fema-2023', 'footprint', parcel_spine)
        == '_footprint_fema'
    )
    # Parcels are interchangeable -> entity only, regardless of source.
    assert (
        _resolve_suffix('US-NC_parcel-nconemap-2025', 'parcel', footprint_spine)
        == '_parcel'
    )
    # Same entity type as the spine -> source disambiguates.
    assert _resolve_suffix('US_building-nsi-2022', 'building', building_spine) == '_nsi'


def test_provenance_vocabulary_auto_includes_fema():
    from openplaces.io.curator.formatters import _provenance_suffixes, _split_source

    vocab = dict(_provenance_suffixes())
    # Auto-generated from recipes: FEMA footprints + the existing references.
    assert vocab.get('_footprint_fema') == 'fema'
    assert vocab.get('_building_nsi') == 'nsi'
    assert vocab.get('_parcel') == 'parcel'
    # Specific suffix wins over the bare fallback.
    assert _split_source('occupancy_type_footprint_fema') == ('occupancy_type', 'fema')


def test_order_columns_places_fema_in_source_band():
    from openplaces.core.schema import AdminId
    from openplaces.io.curator import CurateState
    from openplaces.io.curator.formatters import order_columns

    df = pd.DataFrame(
        {
            'occupancy_type': ['Single-Family'],
            'occupancy_type_building_nsi': ['Single Family'],
            'occupancy_type_footprint_fema': ['Single Family'],
            'geometry': [None],
        }
    )
    state = CurateState(
        recipe={'entity': 'footprint-cheer-2026', 'admin_id': 'US'},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )
    cols = list(order_columns(state).curated.columns)
    # Canonical occupancy first; the FEMA + NSI evidence sit in the source band
    # after it and before geometry.
    assert cols.index('occupancy_type') < cols.index('occupancy_type_footprint_fema')
    assert cols.index('occupancy_type_footprint_fema') < cols.index('geometry')
    assert cols.index('occupancy_type_building_nsi') < cols.index('geometry')


def test_evidence_comparison_writes_agreement_and_conflicts(monkeypatch, tmp_path):
    import openplaces.io.curator.occupancy as occ_mod
    import openplaces.path as op_path
    from openplaces.core.schema import AdminId
    from openplaces.io.curator import CurateState
    from openplaces.io.curator.diagnostics import save_occupancy_evidence_comparison

    class_map = [
        {'pattern': 'Single', 'match_type': 'contains', 'occupancy_type': 'SF'},
        {'pattern': 'Multi', 'match_type': 'contains', 'occupancy_type': 'MF'},
        {'pattern': 'Manufactured', 'match_type': 'contains', 'occupancy_type': 'MH'},
    ]
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda state, ruleset: class_map)
    monkeypatch.setattr(
        op_path, 'cache_path', lambda admin_id, entity, filename: tmp_path / filename
    )

    # Rows 0-1 agree (SF); row 2 NSI=SF vs FEMA=MF conflict. The fema
    # evidence column is group_footprint_fema (NSI-style group vocabulary,
    # derived at FEMA ingest).
    df = pd.DataFrame(
        {
            'occupancy_type_building_nsi': ['Single', 'Single', 'Single'],
            'group_footprint_fema': ['Single', 'Single', 'Multi'],
            'group_parcel': ['Single', 'Single', 'Single'],
        }
    )
    state = CurateState(
        recipe={
            'entity': 'footprint-cheer-2026',
            'admin_id': 'US',
            'occupancy': {
                'class_map': 'occupancy-class-map.csv',
                # Declared so residential classes survive the non-residential
                # bucketing the comparison shares with occupancy_type_conflict.
                'residential_classes': ['SF', 'MF', 'MH'],
            },
        },
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
        save_statistics=True,
    )
    save_occupancy_evidence_comparison(state)

    agreement = pd.read_csv(tmp_path / 'occupancy-evidence-agreement.csv')
    nsi_fema = agreement[
        (agreement['source_a'] == 'nsi') & (agreement['source_b'] == 'fema')
    ].iloc[0]
    assert nsi_fema['n_conflict'] == 1 and nsi_fema['n_agree'] == 2
    conflicts = pd.read_csv(tmp_path / 'occupancy-evidence-conflicts.csv')
    assert {'SF', 'MF'} <= set(conflicts['class_a']) | set(conflicts['class_b'])
