"""Vote provenance names the evidence that fired, not the class.

`occupancy_type_source` read `single_family` on a Single-Family row -- a
synonym of the value beside it, on 87.7% of Harris County's 1.5M rows,
because `score_decisions` recorded the winning decision's declared name.
It now joins the `label` of each indicator that actually fired, matching
what `vote_dynamic_values` next to it already reports (`nsi/fema/parcel`).

Labeling is opt-in per recipe: a decision that labels no indicators keeps
its declared `source`, so no unannotated recipe loses provenance.
"""

import pandas as pd

from openplaces.io.curator.indicators import score_decisions


class TestVoteProvenance:
    def test_the_source_names_the_evidence_that_fired(self):
        curated = pd.DataFrame(
            {'nsi': ['Manufactured Home'], 'fema': ['Manufactured Home']}
        )
        decisions = [
            {
                'class': 'Manufactured Home',
                'source': 'manufactured_home',
                'min_score': 1,
                'indicators': [
                    {
                        'type': 'equals',
                        'column': 'nsi',
                        'value': 'Manufactured Home',
                        'label': 'nsi',
                    },
                    {
                        'type': 'equals',
                        'column': 'fema',
                        'value': 'Manufactured Home',
                        'label': 'fema',
                    },
                ],
            }
        ]
        winner, token, _, _, _ = score_decisions(curated, decisions)

        assert winner.iloc[0] == 'Manufactured Home'
        assert token.iloc[0] == 'nsi+fema'

    def test_only_the_indicators_that_actually_matched_are_named(self):
        curated = pd.DataFrame(
            {'nsi': ['Manufactured Home'], 'fema': ['Single Family']}
        )
        decisions = [
            {
                'class': 'Manufactured Home',
                'source': 'manufactured_home',
                'min_score': 1,
                'indicators': [
                    {
                        'type': 'equals',
                        'column': 'nsi',
                        'value': 'Manufactured Home',
                        'label': 'nsi',
                    },
                    {
                        'type': 'equals',
                        'column': 'fema',
                        'value': 'Manufactured Home',
                        'label': 'fema',
                    },
                ],
            }
        ]
        _, token, _, _, _ = score_decisions(curated, decisions)
        assert token.iloc[0] == 'nsi'

    def test_rows_report_their_own_voters(self):
        curated = pd.DataFrame(
            {
                'nsi': ['Manufactured Home', 'Single Family'],
                'fema': ['Single Family', 'Manufactured Home'],
            }
        )
        decisions = [
            {
                'class': 'Manufactured Home',
                'source': 'manufactured_home',
                'min_score': 1,
                'indicators': [
                    {
                        'type': 'equals',
                        'column': 'nsi',
                        'value': 'Manufactured Home',
                        'label': 'nsi',
                    },
                    {
                        'type': 'equals',
                        'column': 'fema',
                        'value': 'Manufactured Home',
                        'label': 'fema',
                    },
                ],
            }
        ]
        _, token, _, _, _ = score_decisions(curated, decisions)
        assert list(token) == ['nsi', 'fema']

    def test_an_unlabeled_recipe_keeps_the_old_behavior(self):
        """Labeling is opt-in: a decision that labels nothing still reports
        its declared source, so no existing recipe loses provenance."""
        curated = pd.DataFrame({'nsi': ['Manufactured Home']})
        decisions = [
            {
                'class': 'Manufactured Home',
                'source': 'manufactured_home',
                'min_score': 1,
                'indicators': [
                    {'type': 'equals', 'column': 'nsi', 'value': 'Manufactured Home'}
                ],
            }
        ]
        _, token, _, _, _ = score_decisions(curated, decisions)
        assert token.iloc[0] == 'manufactured_home'

    def test_a_decision_won_by_an_unlabeled_indicator_falls_back(self):
        curated = pd.DataFrame({'nsi': ['Single Family'], 'other': [True]})
        decisions = [
            {
                'class': 'Single-Family',
                'source': 'single_family',
                'min_score': 1,
                'indicators': [
                    {
                        'type': 'equals',
                        'column': 'nsi',
                        'value': 'Manufactured Home',
                        'label': 'nsi',
                    },
                    {'type': 'equals', 'column': 'other', 'value': True},
                ],
            }
        ]
        _, token, _, _, _ = score_decisions(curated, decisions)
        assert token.iloc[0] == 'single_family'

    def test_the_reported_failure_now_names_its_evidence(self):
        """The shipped column read `single_family` on a Single-Family row --
        a synonym of the value beside it. It should say who decided."""
        curated = pd.DataFrame(
            {
                'occupancy_parcel': ['Single-Family'],
                'group_nsi': ['Single Family'],
                'group_fema': ['Single Family'],
            }
        )
        decisions = [
            {
                'class': 'Single-Family',
                'source': 'single_family',
                'min_score': 1,
                'indicators': [
                    {
                        'type': 'equals',
                        'column': 'occupancy_parcel',
                        'value': 'Single-Family',
                        'label': 'parcel',
                    },
                    {
                        'type': 'equals',
                        'column': 'group_nsi',
                        'value': 'Single Family',
                        'label': 'nsi',
                    },
                    {
                        'type': 'equals',
                        'column': 'group_fema',
                        'value': 'Single Family',
                        'label': 'fema',
                    },
                ],
            }
        ]
        _, token, _, _, _ = score_decisions(curated, decisions)

        assert token.iloc[0] == 'parcel+nsi+fema'
        assert token.iloc[0] != 'single_family'
