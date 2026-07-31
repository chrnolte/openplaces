"""Tests for the curated column order (canonical -> source -> flags/viz) and the
per-variable provenance sidecars ({col}_source) written by the curation steps."""

from __future__ import annotations

import pandas as pd

import openplaces.io.curator.occupancy as occ_mod
import openplaces.io.curator.reconcilers as rec
import openplaces.path as op_path
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.formatters import cast_integers, order_columns
from openplaces.io.curator.imputers import impute_n_dwellings
from openplaces.io.curator.inferers import impute_occupancy_type
from openplaces.io.curator.reconcilers import reconcile_values

CLASS_MAP = [
    {
        'pattern': 'Manufactured',
        'match_type': 'contains',
        'occupancy_type': 'Manufactured Home',
    },
    {'pattern': 'Single', 'match_type': 'contains', 'occupancy_type': 'Single-Family'},
    {'pattern': 'Multi', 'match_type': 'contains', 'occupancy_type': 'Multi-Family'},
]

OCC = {
    'class_map': 'occupancy-class-map.csv',
    'residential_classes': ['Single-Family', 'Multi-Family', 'Manufactured Home'],
    'secondary_class': 'Secondary',
    'evidence': [
        {'column': 'occupancy_type_building_nsi', 'label': 'nsi'},
        {'column': 'group_parcel', 'label': 'parcel'},
    ],
    'columns': {
        'improvement_value': 'improvement_value_parcel',
        'land_value': 'land_value_parcel',
        'n_dwellings': 'n_dwellings',
    },
    'rules': {
        'manufactured_home_value': {
            'zero_classifies': True,
            'review_max_ratio': 0.025,
            'class': 'Manufactured Home',
        },
        'multi_family_dwellings': {'min_dwellings': 2, 'class': 'Multi-Family'},
        'single_family_dwellings': {'max_dwellings': 1, 'class': 'Single-Family'},
    },
}


def _state(df, recipe=None):
    return CurateState(
        recipe=recipe or {'entity': 'footprint-cheer-2026', 'admin_id': 'US'},
        entity_recipe={},
        admin_id=AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_order_bands_with_grouped_sidecars():
    df = pd.DataFrame(
        {
            'occupancy_type_building_nsi': ['x'],
            'improvement_value_parcel': [1.0],
            'n_parcels_per_footprint': [1],
            'value': [1.0],
            'value_source': ['parcel'],
            'occupancy_type': ['Single-Family'],
            'occupancy_type_source': ['nsi'],
            'occupancy_type_conflict': [pd.NA],
            'occupancy_type_review': [False],
            'improvement_value_parcel_per_area': [0.1],
            'area_m2': [10.0],
            'priority_on_parcel': ['primary'],
            'geometry_source': ['obm'],
            'geometry': [None],
        }
    )
    cols = list(order_columns(_state(df)).curated.columns)

    def before(a, b):
        return cols.index(a) < cols.index(b)

    # canonical block: values only, no sidecars interspersed
    assert before('occupancy_type', 'value')
    assert before('value', 'area_m2') and before('area_m2', 'priority_on_parcel')
    # priority_on_parcel ends the canonical block, before the grouped source band
    assert before('priority_on_parcel', 'occupancy_type_source')
    # all {col}_source sidecars are grouped and contiguous, ordered by base rank
    assert cols.index('value_source') == cols.index('occupancy_type_source') + 1
    # the grouped sidecar band precedes the linked-source evidence block
    assert before('value_source', 'n_parcels_per_footprint')
    # counts lead the source block, ahead of inherited evidence
    assert before('n_parcels_per_footprint', 'improvement_value_parcel')
    assert before('occupancy_type_building_nsi', 'occupancy_type_conflict')
    # flags then viz
    assert before('occupancy_type_conflict', 'occupancy_type_review')
    assert before('occupancy_type_review', 'improvement_value_parcel_per_area')
    # geometry_source groups with the other {col}_source sidecars (it has no
    # registry rank, so it sorts last among them), ahead of the source block;
    # geometry itself is always the final column.
    assert before('value_source', 'geometry_source')
    assert before('geometry_source', 'n_parcels_per_footprint')
    assert cols[-1] == 'geometry'


def test_order_columns_original_variant_follows_its_base():
    # address_original/city_original (the raw, pre-reconciliation values
    # US_parcel-spine-2026.yaml preserves ahead of reconcile_addresses) are
    # not evidence attributed from another entity, so they carry no
    # provenance suffix to key on -- they need the same "follows its base"
    # treatment as {col}_all.
    df = pd.DataFrame(
        {
            'address': ['1 Sample Ave, North Billerica, MA 01862'],
            'address_original': ['1 SAMPLE AVE'],
            'city': ['North Billerica'],
            'city_original': ['BILLERICA'],
            'n_parcels_per_footprint': [1],
            'geometry': [None],
        }
    )
    cols = list(order_columns(_state(df)).curated.columns)

    assert cols.index('address_original') == cols.index('address') + 1
    assert cols.index('city_original') == cols.index('city') + 1


def test_order_columns_recognizes_parcel_flags():
    # land_use_review/manufactured_home_community are the parcel lane's
    # analogs of occupancy_type_review/manufactured_home_community on
    # footprints -- _FLAG_COLUMNS needs both recognized so they land in the
    # flags band (after canonical/sources/evidence) rather than falling
    # through to the unranked-canonical bucket.
    df = pd.DataFrame(
        {
            'land_use_class': ['Manufactured Home Park'],
            'n_parcels_per_footprint': [1],
            'land_use_class_conflict': [pd.NA],
            'land_use_review': [False],
            'manufactured_home_community': [True],
            'geometry': [None],
        }
    )
    cols = list(order_columns(_state(df)).curated.columns)

    def before(a, b):
        return cols.index(a) < cols.index(b)

    assert before('n_parcels_per_footprint', 'land_use_class_conflict')
    assert before('land_use_class_conflict', 'land_use_review')
    assert before('land_use_review', 'manufactured_home_community')


def test_order_columns_drops_multiple_transient_columns():
    # order_columns's drop param is what US_footprint-cheer-2026.yaml relies
    # on to keep occupancy_type_dwelling_overture (+ its _source sidecar) and
    # group_parcel out of the canonical output while still using them as
    # transient inputs earlier in the pipeline.
    df = pd.DataFrame(
        {
            'occupancy_type': ['Single-Family'],
            'group_parcel': ['Single Family'],
            'occupancy_type_dwelling_overture': ['Single-Family'],
            'occupancy_type_dwelling_overture_source': ['overture'],
            'geometry': [None],
        }
    )
    out = order_columns(
        _state(df),
        drop=[
            'group_parcel',
            'occupancy_type_dwelling_overture',
            'occupancy_type_dwelling_overture_source',
        ],
    ).curated
    assert 'group_parcel' not in out.columns
    assert 'occupancy_type_dwelling_overture' not in out.columns
    assert 'occupancy_type_dwelling_overture_source' not in out.columns
    assert 'occupancy_type' in out.columns


def test_cast_integers_rounds_and_preserves_missing():
    # A reconciled year_built can be non-integer (e.g. an NSI block-median
    # fallback averaged across several buildings); round rather than
    # truncate so early years aren't systematically biased down. A missing
    # value stays missing (pd.NA), not a misleading year 0.
    df = pd.DataFrame({'year_built': [1964.0, 1998.6, None]})
    out = cast_integers(_state(df), columns=['year_built']).curated

    assert out['year_built'].dtype == 'Int64'
    assert out['year_built'].tolist() == [1964, 1999, pd.NA]


def test_cast_integers_skips_missing_columns():
    df = pd.DataFrame({'other': [1.0]})
    out = cast_integers(_state(df), columns=['year_built']).curated
    assert 'year_built' not in out.columns


def test_order_columns_drops_transient_manufactured_home_inputs():
    # occupancy_type_base and p_manufactured_home are each fully consumed by
    # an earlier step (refine_occupancy_height, the manufactured-home vote
    # respectively) in US_footprint-cheer-2026.yaml, so the recipe drops them
    # here too. (manufactured_home_community, formerly manufactured_home_park,
    # is a different case: link_curated_entity seeds it from the parcel lane,
    # impute_occupancy_type consumes that seed, and
    # flag_manufactured_home_communities later overwrites it under the same
    # name with its own refinement -- no drop needed, see that step's
    # docstring.)
    df = pd.DataFrame(
        {
            'occupancy_type': ['Manufactured Home'],
            'occupancy_type_base': ['Multi-Family'],
            'p_manufactured_home': [0.8],
            'geometry': [None],
        }
    )
    out = order_columns(
        _state(df),
        drop=['occupancy_type_base', 'p_manufactured_home'],
    ).curated
    assert 'occupancy_type_base' not in out.columns
    assert 'p_manufactured_home' not in out.columns
    assert 'occupancy_type' in out.columns


def test_reconcile_records_winning_source():
    df = pd.DataFrame(
        {
            'improvement_value_parcel': [100.0, None],
            'structure_value_building_nsi': [200.0, 50.0],
        }
    )
    out = reconcile_values(
        _state(df),
        priority={
            'value': ['improvement_value_parcel', 'structure_value_building_nsi']
        },
    ).curated
    assert out['value'].tolist() == [100.0, 50.0]
    assert out['value_source'].tolist() == ['parcel', 'nsi']


def test_impute_records_imputed_token():
    df = pd.DataFrame({'occupancy_type': ['Manufactured Home'], 'n_dwellings': [None]})
    out = impute_n_dwellings(_state(df)).curated
    assert out['n_dwellings'].iloc[0] == 1.0
    assert out['n_dwellings_source'].iloc[0] == 'imputed'


def test_impute_records_evidence_source(monkeypatch):
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    df = pd.DataFrame(
        {
            'occupancy_type_building_nsi': ['Manufactured Home', None],
            'group_parcel': ['Single Family', 'Single Family'],
        }
    )
    out = impute_occupancy_type(
        _state(df, recipe={'entity': 'e', 'admin_id': 'US', 'occupancy': OCC})
    ).curated
    assert out['occupancy_type'].astype(object).tolist() == [
        'Manufactured Home',
        'Single-Family',
    ]
    # row 0 from NSI, row 1 filled from parcel
    assert out['occupancy_type_source'].astype(object).tolist() == ['nsi', 'parcel']


def test_impute_records_single_family_dwellings_source(monkeypatch):
    # No NSI/parcel evidence at all: only the n_dwellings single-family
    # gap-fill (rules.single_family_dwellings) applies, and its token must be
    # the rule's own name, not the bare 'dwellings' this used to collide with
    # occupancy.rules.multi_family_dwellings's resolve_by_vote source.
    monkeypatch.setattr(occ_mod, 'load_ruleset', lambda s, r: CLASS_MAP)
    df = pd.DataFrame({'n_dwellings': [1.0]})
    out = impute_occupancy_type(
        _state(df, recipe={'entity': 'e', 'admin_id': 'US', 'occupancy': OCC})
    ).curated
    assert out['occupancy_type'].astype(object).tolist() == ['Single-Family']
    assert out['occupancy_type_source'].astype(object).tolist() == [
        'single_family_dwellings'
    ]


def _patch_resolve(monkeypatch, tmp_path, keyword_rules):
    def fake_load(state, ruleset):
        return CLASS_MAP if 'class-map' in ruleset else keyword_rules

    monkeypatch.setattr(occ_mod, 'load_ruleset', fake_load)
    monkeypatch.setattr(
        op_path, 'reports_path', lambda *a, **k: tmp_path / 'occupancy-conflicts.csv'
    )


def _resolve_frame(**overrides):
    df = pd.DataFrame(
        {
            'use_group_combined_parcel': ['RES'],
            'group_parcel': ['Single Family'],
            'occupancy_type': ['Single-Family'],
            'occupancy_type_building_nsi': ['Single Family, 1 story'],
            'improvement_value_parcel': [100000.0],
            'land_value_parcel': [50000.0],
            'n_dwellings': [1.0],
        }
    )
    for k, v in overrides.items():
        df[k] = v
    return df


def test_resolve_correction_tokens(monkeypatch, tmp_path):
    # resolve_occupancy records only the reviewed-keyword correction. The
    # value-share and dwelling-count class decisions moved to resolve_by_vote
    # (covered by test_resolve_by_vote), so they leave no occupancy_type_source
    # token here.
    _patch_resolve(monkeypatch, tmp_path, keyword_rules=[])
    out = rec.resolve_occupancy(
        _state(
            _resolve_frame(improvement_value_parcel=[0.0]),
            recipe={'entity': 'e', 'admin_id': 'US', 'occupancy': OCC},
        ),
        ruleset='kw.csv',
    ).curated
    assert 'occupancy_type_source' not in out.columns

    # reviewed keyword -> 'keyword'
    _patch_resolve(
        monkeypatch,
        tmp_path,
        keyword_rules=[
            {
                'pattern': 'MANUFACTURED',
                'match_type': 'contains',
                'occupancy_type': 'Manufactured Home',
                'reviewed': True,
            }
        ],
    )
    out = rec.resolve_occupancy(
        _state(
            _resolve_frame(use_group_combined_parcel=['MANUFACTURED HOUSING']),
            recipe={'entity': 'e', 'admin_id': 'US', 'occupancy': OCC},
        ),
        ruleset='kw.csv',
    ).curated
    assert out['occupancy_type_source'].iloc[0] == 'keyword'
