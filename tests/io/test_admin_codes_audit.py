"""Tests for the identifier audit."""

import pandas as pd
import pytest

import openplaces.io.admin_codes.frame as frame
from openplaces.io.admin_codes import audit, build
from openplaces.io.admin_codes.audit import audit_spine

needs_population = pytest.mark.skipif(
    not all(build.population_path(level).exists() for level in (2, 3)),
    reason='population tables not built in this data root',
)


def _weights(levels):
    return {
        level: pd.read_csv(
            build.population_path(level), dtype={'admin_id': str}
        ).set_index('admin_id')['population']
        for level in levels
    }


class TestAuditSpine:
    def test_the_committed_spine_satisfies_every_invariant(self):
        report = audit_spine(reproduce=False)
        assert (report['bad_format'] == 0).all()
        assert (report['orphan_parents'] == 0).all()
        assert (report['duplicate_ids'] == 0).all()
        assert (report['mixed_width_parents'] == 0).all()

    def test_reports_one_row_per_level(self):
        report = audit_spine(levels=(2, 3), reproduce=False)
        assert list(report.index) == [2, 3]
        assert (report['units'] > 0).all()


class TestReproducedIsDerived:
    def test_the_check_derives_codes_instead_of_reading_them_back(self, monkeypatch):
        # Alpha's committed code is ZZ, which no rule derives from the
        # name. A pinned run copies ZZ out of the registry and reports
        # the spine reproduced; only a derived run notices that it is
        # not, which is the whole point of the check.
        committed = pd.DataFrame(
            {'admin2_id': ['XX-ZZ', 'XX-BR'], 'name': ['Alpha', 'Bravo']}
        )
        monkeypatch.setattr(audit, '_read', lambda level: committed)
        monkeypatch.setattr(
            frame,
            'load_registry',
            lambda level, sep='-': ({('XX', 'Alpha'): 'ZZ'}, {}, {'XX': {'ZZ'}}),
        )
        report = audit_spine(levels=(2,), reproduce=True)
        assert report.loc[2, 'units'] == 2
        assert report.loc[2, 'reproduced'] == 1

    @needs_population
    def test_weighted_derivation_reproduces_the_spine_exactly(self):
        # Fed the weights the mint used, the derivation is the re-mint
        # itself, so anything short of every code is a regression in the
        # generator or the weights. Level 4 takes minutes and is
        # covered by test_admin_codes_build.py's fixed-point test.
        report = audit_spine(levels=(2, 3), reproduce=True, weights=_weights((2, 3)))
        for level in (2, 3):
            short = report.loc[level, 'units'] - report.loc[level, 'reproduced']
            assert short == 0, (
                f'level {level}: {short} codes do not derive from names and weights'
            )

    def test_unweighted_derivation_stays_above_its_floor(self):
        # Without weights the contested codes fall the other way, so the
        # count is a floor, not an identity. Measured 2026-09-06: 0.913
        # at level 2 and 0.834 at level 3. A derivation regression, or
        # pinning creeping back in (which reports 0.998), moves it.
        report = audit_spine(levels=(2, 3), reproduce=True)
        share = report['reproduced'] / report['units']
        assert 0.85 < share[2] < 0.97, share[2]
        assert 0.78 < share[3] < 0.97, share[3]


class TestAuditReportShape:
    def test_report_is_a_frame_of_counts(self):
        report = audit_spine(levels=(2,), reproduce=False)
        assert isinstance(report, pd.DataFrame)
        for column in ('units', 'bad_format', 'duplicate_ids'):
            assert pd.api.types.is_integer_dtype(report[column])
