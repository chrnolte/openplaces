"""Tests for the spine rebuild pipeline.

The expensive steps need a population raster and a full geometry read, so
these cover what can be checked cheaply: the module imports, the paths it
writes to are the committed ones, and the two dry-run steps report without
touching anything. The guarantee that matters -- that the pipeline
reproduces the committed spine -- is asserted by
`test_admin_codes_audit.py::test_level_four_reproduces_exactly`.
"""

import pandas as pd

from openplaces.io.admin_codes import build
from openplaces.io.admin_codes.registry import spine_path


class TestPaths:
    def test_population_path_points_at_the_committed_sidecar(self):
        for level in build.LEVELS:
            path = build.population_path(level)
            assert path.parent == spine_path(level).parent
            assert path.exists(), f'level {level} population table is missing'

    def test_every_level_is_fully_weighted(self):
        # The whole pipeline exists to make this true; a gap here means a
        # sibling group somewhere is tie-breaking on sort order.
        for level in build.LEVELS:
            spine = pd.read_csv(spine_path(level), dtype=str, keep_default_na=False)
            column = f'admin{level}_id'
            live = set(spine.loc[spine[column].str.strip() != '', column])
            weighted = set(pd.read_csv(build.population_path(level))['admin_id'])
            assert live <= weighted, (
                f'level {level}: {len(live - weighted):,} units unweighted'
            )

    def test_no_unit_is_weighted_zero(self):
        # Zero loses every tie it enters, which is worse than unweighted.
        for level in build.LEVELS:
            table = pd.read_csv(build.population_path(level))
            assert (table['population'] > 0).all()


class TestDryRuns:
    def test_resolving_references_reports_without_writing(self):
        # Run against the committed tree: after a settled re-mint nothing
        # should still name a retired identifier.
        assert build.resolve_stale_references(apply=False, verbose=False) == 0

    def test_the_mint_is_a_fixed_point(self):
        # Re-deriving every identifier from the same weights must return
        # the same spine, or the identifiers are not reproducible.
        report = build.remint_spine(apply=False, verbose=False)
        for level, counts in report.items():
            assert counts['changed'] == 0, (
                f'level {level}: {counts["changed"]:,} identifiers would move'
            )
