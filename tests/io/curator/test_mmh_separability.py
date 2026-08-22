"""The manufactured-home separability harness, pinned so it can be trusted later.

`notebooks/05_curate/mmh_separability.py` exists to be re-run when the
inventory's source mix changes -- specifically if NSI and FEMA stop feeding
it. A measurement that has rotted between now and then is worse than none,
because it will be believed. These tests fix its semantics on synthetic
frames so the numbers it reports later mean what they meant when it was
written.

The stratification is the part worth protecting. An earlier neighborhood
signal scored -0.0071 F1 overall while being wrong on 7 of the 7 points it
moved; a pooled figure could not see that, and this harness is built to.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_NB = Path(__file__).resolve().parents[3] / 'notebooks' / '05_curate'
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))

from mmh_separability import (  # noqa: E402
    MANUFACTURED,
    block_composition,
    label_block_type,
    score_evidence,
    separability_report,
)


def _footprints(block_ids, classes):
    return pd.DataFrame({'census_block_id': block_ids, 'group_a': classes})


class TestBlockComposition:
    def test_the_building_is_excluded_from_its_own_block(self):
        f = _footprints(['b'] * 5, ['Manufactured Home'] * 4 + ['Single Family'])
        share = block_composition(f, ['group_a'])
        assert share.iloc[0] == pytest.approx(3 / 4)  # sees the other three
        assert share.iloc[4] == pytest.approx(1.0)  # sees all four

    def test_a_block_with_too_few_neighbors_reports_nothing(self):
        f = _footprints(['b', 'b', 'solo'], ['Manufactured Home'] * 3)
        share = block_composition(f, ['group_a'], min_neighbors=3)
        assert share.isna().all()

    def test_evidence_columns_are_configurable(self):
        """The whole point of the forward-looking run: ask the same question
        without the sources that may go away."""
        f = pd.DataFrame(
            {
                'census_block_id': ['b'] * 6,
                'nsi': ['Manufactured Home'] * 5 + ['Single Family'],
                'assessor': ['Single Family'] * 6,
            }
        )
        with_nsi = block_composition(f, ['nsi'])
        without_nsi = block_composition(f, ['assessor'])
        assert with_nsi.iloc[5] == pytest.approx(1.0)
        assert without_nsi.iloc[5] == pytest.approx(0.0)

    def test_an_absent_column_yields_no_share_rather_than_zero(self):
        """Missing evidence must not read as 'no manufactured homes here'."""
        f = _footprints(['b'] * 5, ['Manufactured Home'] * 5)
        assert block_composition(f, ['not_a_column']).isna().all()


class TestBlockStrata:
    @pytest.mark.parametrize(
        'share, expected',
        [
            (0.95, 'pure manufactured'),
            (0.80, 'pure manufactured'),
            (0.55, 'mixed'),
            (0.30, 'mixed'),
            (0.20, 'pure site-built'),
            (0.00, 'pure site-built'),
            (np.nan, 'no block context'),
        ],
    )
    def test_bands(self, share, expected):
        assert label_block_type(pd.Series([share])).iloc[0] == expected

    def test_the_mixed_band_is_where_neighbors_stop_helping(self):
        """A site-built house on a manufactured-home lot lands here, and it
        is the stratum any neighborhood rule will get wrong."""
        shares = pd.Series([0.9, 0.5, 0.1])
        assert list(label_block_type(shares)) == [
            'pure manufactured',
            'mixed',
            'pure site-built',
        ]


class TestScoring:
    def test_precision_and_recall(self):
        truth = pd.Series([MANUFACTURED, MANUFACTURED, 'SFH', 'SFH'])
        fires = pd.Series([True, False, True, False])
        s = score_evidence(truth, fires)
        assert s['precision'] == pytest.approx(0.5)
        assert s['recall'] == pytest.approx(0.5)
        assert s['support'] == 2

    def test_a_signal_that_never_fires_scores_no_precision(self):
        truth = pd.Series([MANUFACTURED, 'SFH'])
        s = score_evidence(truth, pd.Series([False, False]))
        assert np.isnan(s['precision'])
        assert s['recall'] == 0.0


class TestReport:
    def test_a_source_good_overall_but_bad_on_mixed_blocks_is_visible(self):
        """The failure the harness exists to expose. Pooled, this source
        looks strong; the mixed stratum shows it is carried entirely by the
        pure blocks."""
        points = pd.DataFrame(
            {
                'occupancy_type': [MANUFACTURED] * 10 + ['SFH'] * 10,
                'block_share': [0.9] * 10 + [0.5] * 10,
            }
        )
        # fires on every high-share point (all correct) and every mixed
        # point (all wrong)
        fires = pd.Series([True] * 20)
        report = separability_report(points, {'neighborhood': fires})

        pooled = report[report['stratum'] == 'ALL'].iloc[0]
        mixed = report[report['stratum'] == 'mixed'].iloc[0]
        pure = report[report['stratum'] == 'pure manufactured'].iloc[0]

        assert pooled['precision'] == pytest.approx(0.5)
        assert pure['precision'] == pytest.approx(1.0)
        assert mixed['precision'] == pytest.approx(0.0)

    def test_every_stratum_and_evidence_pair_is_reported(self):
        points = pd.DataFrame(
            {
                'occupancy_type': [MANUFACTURED, 'SFH', MANUFACTURED, 'SFH'],
                'block_share': [0.9, 0.5, 0.1, np.nan],
            }
        )
        evidence = {
            'keyword': pd.Series([True, False, True, False]),
            'morphology': pd.Series([False, True, True, False]),
        }
        report = separability_report(points, evidence)
        assert set(report['evidence']) == {'keyword', 'morphology'}
        # four strata are populated, plus ALL
        assert report['stratum'].nunique() == 5


class TestAssessorTextMatching:
    """Assessor land-use is free text, so equality finds nothing.

    Measured on the ten surveyed counties: matching the canonical label by
    equality identified zero predominantly-manufactured blocks, where the
    NSI/FEMA vocabularies found 9,216. Any post-NSI block signal has to read
    the assessor's own wording.
    """

    def test_exact_matching_misses_assessor_wording(self):
        f = pd.DataFrame(
            {
                'census_block_id': ['b'] * 5,
                'use_group': ['DOUBLE WIDE MOHO'] * 4 + ['RESIDENTIAL | SFR'],
            }
        )
        share = block_composition(f, ['use_group'], match_value='Manufactured Home')
        assert (share.fillna(0) == 0).all()

    def test_a_pattern_finds_them(self):
        f = pd.DataFrame(
            {
                'census_block_id': ['b'] * 5,
                'use_group': [
                    'DOUBLE WIDE MOHO',
                    'MOBILE HOME',
                    'SINGLE WIDE',
                    'MANUFACTURED HOME',
                    'RESIDENTIAL | SFR',
                ],
            }
        )
        share = block_composition(
            f, ['use_group'], match_pattern='MOBILE|MANUFACTURED|MOHO|WIDE'
        )
        # the site-built row sees all four manufactured neighbors
        assert share.iloc[4] == pytest.approx(1.0)
        # a manufactured row excludes itself, so it sees three of the other four
        assert share.iloc[0] == pytest.approx(0.75)
