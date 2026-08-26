"""Tests for the imputation-provenance invariant.

The rule these enforce: a cell **openplaces itself filled** must say so in
its ``{column}_source`` sidecar, by carrying the ``imputed`` marker. A value
read from a dataset keeps that dataset's name even when the dataset is
itself modeled -- ``nsi`` stays ``nsi`` -- because that token already names
what produced the number.

The interesting part is not any single step. It is that the marker survives
the whole chain from the parcel step that knows a value was estimated,
across the apportionment onto footprints, to the per-town selector that
writes the delivered ``structure_value_source``. Each of those links dropped
or flattened it before.
"""

from __future__ import annotations

import pandas as pd
import pytest

import openplaces.io.curator.evidence as evidence_mod
import openplaces.io.harmonizer.links as links_mod
from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.evidence import apportion_curated_values
from openplaces.io.curator.imputers import impute_from_group_statistic
from openplaces.io.curator.provenance import is_imputed, mark_imputed, record_sources
from openplaces.io.curator.reconcilers import select_value_source_by_admin_unit


def _state(df: pd.DataFrame, admin_id: AdminId | None = None) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=admin_id or AdminId('US'),
        verbose=False,
        timer=None,
        curated=df,
    )


class TestVocabulary:
    """The two functions the whole guarantee rests on agreeing."""

    def test_marker_is_appended_to_a_route_token(self):
        assert mark_imputed('parcel') == 'parcel+imputed'

    def test_marking_is_idempotent(self):
        # A chain of steps each marking what it passes along must not
        # accumulate 'imputed+imputed+imputed'.
        once = mark_imputed('parcel')
        assert mark_imputed(mark_imputed(once)) == once

    @pytest.mark.parametrize('empty', [None, pd.NA, ''])
    def test_derived_value_with_no_known_route_still_reports_derived(self, empty):
        # Better a bare 'imputed' than a null that reads as "not decided".
        assert mark_imputed(empty) == 'imputed'

    def test_marker_is_matched_as_a_whole_part_not_a_substring(self):
        # A source legitimately named 'imputed_rates' is an original source.
        flags = is_imputed(['parcel', 'parcel+imputed', 'imputed', 'imputed_rates'])
        assert list(flags) == [False, True, True, False]

    def test_unknown_provenance_is_not_a_claim_of_derivation(self):
        assert not is_imputed([None]).iloc[0]

    def test_record_sources_leaves_rows_without_a_token_untouched(self):
        # A partially-known provenance must never blank what an earlier step
        # recorded for the rows it says nothing about.
        df = pd.DataFrame({'v': [1.0, 2.0]})
        df['v_source'] = ['keeper', 'keeper']
        record_sources(df, 'v', pd.Series([None, 'nsi'], index=df.index))
        assert list(df['v_source']) == ['keeper', 'nsi']


class TestApportionmentPropagation:
    """The parcel to footprint boundary, where the marker used to die."""

    @staticmethod
    def _patch(monkeypatch, tmp_path, sidecar, ref):
        sidecar_path = tmp_path / 'link.parquet'
        sidecar.to_parquet(sidecar_path)
        monkeypatch.setattr(
            evidence_mod, 'get_recipe_by_id', lambda recipe_id: {'stage': 'curate'}
        )
        monkeypatch.setattr(evidence_mod, 'get_output_path', lambda *a, **k: 'p')
        monkeypatch.setattr(evidence_mod, 'read_parquet', lambda path: ref)
        monkeypatch.setattr(
            evidence_mod, 'get_link_owner_recipe_id', lambda recipe: 'entity-recipe'
        )
        monkeypatch.setattr(
            evidence_mod, 'get_entity_link_path', lambda *a, **k: sidecar_path
        )
        monkeypatch.setattr(
            links_mod, '_resolve_reference_recipe', lambda *a, **k: ('parcel-ref', None)
        )

    @staticmethod
    def _footprints(parcel_ids, index):
        return pd.DataFrame(
            {
                'parcel_id': parcel_ids,
                'priority_on_parcel': ['primary'] * len(index),
                'n_dwellings_overture': [0.0] * len(index),
            },
            index=pd.Index(index, name='footprint_id'),
        )

    def _apportion(self, monkeypatch, tmp_path, footprints, sidecar, ref):
        self._patch(monkeypatch, tmp_path, sidecar, ref)
        return apportion_curated_values(
            _state(footprints),
            recipe_id='US_parcel-openplaces-2026',
            columns={'improvement_value_imputed': 'structure_value'},
        ).curated

    def test_dominant_reference_token_is_carried_onto_the_entity(
        self, monkeypatch, tmp_path
    ):
        footprints = self._footprints(['P1', 'P2'], ['F1', 'F2'])
        sidecar = pd.DataFrame(
            {
                'footprint_id': ['F1', 'F2'],
                'parcel_id': ['P1', 'P2'],
                'area_intersection_m2': [100.0, 100.0],
                'link': ['ok', 'ok'],
            }
        )
        ref = pd.DataFrame(
            {
                'parcel_id': ['P1', 'P2'],
                'improvement_value_imputed': [100000.0, 200000.0],
                'improvement_value_imputed_source': [
                    'improvement_value',
                    '_is_residential+imputed',
                ],
            }
        )
        out = self._apportion(monkeypatch, tmp_path, footprints, sidecar, ref)

        # The apportionment hop is recorded lane-first, like geometry_source
        assert out.loc['F1', 'structure_value_source'] == 'parcel.improvement_value'
        assert (
            out.loc['F2', 'structure_value_source'] == 'parcel._is_residential+imputed'
        )
        assert not is_imputed(out['structure_value_source']).loc['F1']
        assert is_imputed(out['structure_value_source']).loc['F2']

    def test_a_split_containing_one_estimated_contributor_is_marked(
        self, monkeypatch, tmp_path
    ):
        # F1 straddles two parcels. The dominant one (P1, larger overlap)
        # has an observed value, so the base token is P1's -- but P2's
        # share of the sum was estimated, which makes the total an estimate
        # too. Marking is the safe direction: calling a derived value
        # original is a false claim.
        footprints = self._footprints(['P1'], ['F1'])
        sidecar = pd.DataFrame(
            {
                'footprint_id': ['F1', 'F1'],
                'parcel_id': ['P1', 'P2'],
                'area_intersection_m2': [300.0, 100.0],
                'link': ['ok', 'ok'],
            }
        )
        ref = pd.DataFrame(
            {
                'parcel_id': ['P1', 'P2'],
                'improvement_value_imputed': [100000.0, 200000.0],
                'improvement_value_imputed_source': [
                    'improvement_value',
                    '_is_residential+imputed',
                ],
            }
        )
        out = self._apportion(monkeypatch, tmp_path, footprints, sidecar, ref)

        assert (
            out.loc['F1', 'structure_value_source']
            == 'parcel.improvement_value+imputed'
        )
        assert is_imputed(out['structure_value_source']).loc['F1']

    def test_a_sub_threshold_sliver_does_not_contaminate_the_token(
        self, monkeypatch, tmp_path
    ):
        # The unlinked sliver contributes no dollars, so it must contribute
        # no derived marker either -- otherwise the guarantee degrades into
        # marking nearly everything.
        footprints = self._footprints(['P1'], ['F1'])
        sidecar = pd.DataFrame(
            {
                'footprint_id': ['F1', 'F1'],
                'parcel_id': ['P1', 'P9'],
                'area_intersection_m2': [300.0, 0.5],
                'link': ['ok', None],
            }
        )
        ref = pd.DataFrame(
            {
                'parcel_id': ['P1', 'P9'],
                'improvement_value_imputed': [100000.0, 999999.0],
                'improvement_value_imputed_source': [
                    'improvement_value',
                    '_is_residential+imputed',
                ],
            }
        )
        out = self._apportion(monkeypatch, tmp_path, footprints, sidecar, ref)

        # The apportionment hop is recorded lane-first, like geometry_source
        assert out.loc['F1', 'structure_value_source'] == 'parcel.improvement_value'

    def test_reference_without_a_sidecar_writes_no_sidecar(self, monkeypatch, tmp_path):
        footprints = self._footprints(['P1'], ['F1'])
        sidecar = pd.DataFrame(
            {
                'footprint_id': ['F1'],
                'parcel_id': ['P1'],
                'area_intersection_m2': [100.0],
                'link': ['ok'],
            }
        )
        ref = pd.DataFrame(
            {'parcel_id': ['P1'], 'improvement_value_imputed': [100000.0]}
        )
        out = self._apportion(monkeypatch, tmp_path, footprints, sidecar, ref)

        assert out.loc['F1', 'structure_value'] == 100000.0
        assert 'structure_value_source' not in out.columns


class TestSelectorPreservesIncomingProvenance:
    """The last step before delivery, which used to flatten to `parcel`."""

    @staticmethod
    def _call(df, **kwargs):
        return select_value_source_by_admin_unit(
            _state(df, AdminId('US', 'MA', 'MI')),
            output='structure_value',
            parcel_column='structure_value',
            other_column='structure_value_building_nsi',
            **kwargs,
        ).curated

    def test_per_row_token_survives_the_output_sharing_its_input_name(self):
        # `output` and `parcel_column` are the same column in the shipping
        # recipe, so they share one sidecar: writing the flat token first
        # would destroy exactly what has to be carried forward.
        df = pd.DataFrame(
            {
                'admin4_id': ['A'] * 4,
                'structure_value': [100.0, 100.0, 100.0, 100.0],
                'structure_value_source': [
                    'improvement_value',
                    'improvement_value',
                    '_is_residential+imputed',
                    '_is_residential+imputed',
                ],
                'structure_value_building_nsi': [999.0] * 4,
                'priority_on_parcel': ['primary'] * 4,
            }
        )
        out = self._call(df, min_group_size=1)

        assert list(out['structure_value_source']) == [
            'improvement_value',
            'improvement_value',
            '_is_residential+imputed',
            '_is_residential+imputed',
        ]
        assert list(is_imputed(out['structure_value_source'])) == [
            False,
            False,
            True,
            True,
        ]

    def test_flat_token_remains_the_fallback_for_an_input_without_provenance(self):
        df = pd.DataFrame(
            {
                'admin4_id': ['A'] * 4,
                'structure_value': [100.0] * 4,
                'structure_value_building_nsi': [999.0] * 4,
                'priority_on_parcel': ['primary'] * 4,
            }
        )
        out = self._call(df, min_group_size=1)
        assert (out['structure_value_source'] == 'parcel').all()

    def test_a_modeled_dataset_value_is_named_not_marked(self):
        # Policy, not mechanics: the marker means openplaces filled this
        # cell. An NSI structure value is FEMA-modeled, but it is still a
        # value read from a dataset, and `nsi` already names exactly what
        # produced it -- so it stays plain `nsi`. Marking it would flip
        # every row of every low-coverage town and cost the marker its
        # ability to point at what openplaces itself computed.
        #
        # This also covers the mechanical half: a town that switched to NSI
        # discards the parcel value entirely, so the parcel lane's
        # provenance must not linger on the rows.
        df = pd.DataFrame(
            {
                'admin4_id': ['B'] * 8,
                'structure_value': [100.0] * 2 + [None] * 6,
                'structure_value_source': ['_is_residential+imputed'] * 2 + [None] * 6,
                'structure_value_building_nsi': [999.0] * 8,
                'priority_on_parcel': ['primary'] * 8,
            }
        )
        out = self._call(df, min_group_size=1)
        assert (out['structure_value'] == 999.0).all()
        assert (out['structure_value_source'] == 'nsi').all()
        assert not is_imputed(out['structure_value_source']).any()


class TestGroupStatisticImputer:
    def test_a_cohort_statistic_is_marked_derived(self):
        df = pd.DataFrame(
            {
                'use_group_combined': ['A', 'A', 'B'],
                'group_building_nsi': ['RES1', 'RES1', None],
            }
        )
        out = impute_from_group_statistic(
            _state(df),
            group_column='use_group_combined',
            value_column='group_building_nsi',
            output='group_parcel',
        ).curated

        assert out.loc[0, 'group_parcel'] == 'RES1'
        assert out.loc[0, 'group_parcel_source'] == 'group_statistic+imputed'
        assert is_imputed(out['group_parcel_source']).iloc[0]
        # Row 2's cohort had nothing to learn from: no value, no token.
        assert pd.isna(out.loc[2, 'group_parcel_source'])
