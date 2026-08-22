"""Tests for limiting process units to a scoped crosswalk's coverage.

A `recipe_id`-form `process_by.admin_id_crosswalk` is a sidecar listing the
admin units a source actually carries. Maine's parcel layer covers organized
towns only, so a couple of hundred spine units have no rows by design.
Expanding a whole-state request to every unit and then raising on the first
uncovered one made such a source impossible to ingest in bulk.

The distinction these tests pin down is between the *expanded* list, which is
filtered, and an *explicit* request for a named unit, which is not -- there
the caller has asserted the unit should be there, and silence would hide a
real mistake rather than accommodate a known gap.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.ingester import Ingester

RECIPE_ID = 'US-ME_parcel-megis-2026'


def _ingester(admin_ids=None):
    ingester = Ingester(RECIPE_ID, admin_ids=admin_ids)
    ingester._resolve_admin_ids(reprocess=True)
    return ingester


@pytest.fixture(scope='module')
def covered():
    """The admin units Maine's crosswalk sidecar actually lists."""
    from openplaces.io.transform import get_crosswalk
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id(RECIPE_ID)
    spec = dict(recipe['process_by']['admin_id_crosswalk'])
    spec['admin_id'] = 'US-ME'
    return set(get_crosswalk(spec, flip=True).iloc[:, 0])


class TestExpandedRequestIsFiltered:
    def test_every_requested_unit_is_one_the_source_covers(self, covered):
        requested = {str(a) for a in _ingester().admin_ids_to_process}
        assert requested <= covered

    def test_the_uncovered_units_are_actually_dropped(self, covered):
        # If this ever equals the full spine count, the filter has stopped
        # firing and the state will fail on its first unorganized township.
        from openplaces.io.admin_codes import spine_path

        spine = pd.read_csv(spine_path(3), dtype=str, keep_default_na=False)
        towns = spine[spine['admin3_id'].str.startswith('US-ME-')]
        assert len(_ingester().admin_ids_to_process) < len(towns)

    def test_nothing_covered_is_lost(self, covered):
        # The filter must subtract only, never fail to reach a unit the
        # source does carry.
        requested = {str(a) for a in _ingester().admin_ids_to_process}
        assert requested == covered


class TestComparisonIsOnStrings:
    def test_admin_id_does_not_hash_equal_its_own_string(self):
        # The reason the filter casts before comparing. Were this to
        # become true, the cast would be redundant rather than wrong --
        # but while it is false, dropping the cast matches nothing and
        # silently disables the filter.
        assert AdminId('US-ME-PO') not in {'US-ME-PO'}


class TestExplicitRequestStillRaises:
    def test_naming_a_unit_the_source_lacks_is_not_quietly_dropped(self, covered):
        # An explicit request for an uncovered unit must survive the
        # filter so process() can report it. Silently returning an empty
        # list here would turn a typo into a no-op.
        from openplaces.io.admin_codes import spine_path

        spine = pd.read_csv(spine_path(3), dtype=str, keep_default_na=False)
        towns = set(spine[spine['admin3_id'].str.startswith('US-ME-')]['admin3_id'])
        uncovered = sorted(towns - covered)
        assert uncovered, 'Maine now has full coverage; this test needs a new case'

        ingester = _ingester(admin_ids=[uncovered[0]])
        assert [str(a) for a in ingester.admin_ids_to_process] == [uncovered[0]]
