"""discover_sources assigns nothing, rather than raising, for a unit with
no units below it at the target level.

The world admin spine is harmonized one parent unit at a time. Antarctica
at level 2 and an Andorran parish at level 3 have no children, and the
scoped id lookup raised for both, which failed the global rebuild before
any other country was reached. The step now asks the reader for an
empty answer and leaves the spine unset, so the harmonizer writes its
empty-spine output for the unit.
"""

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import discover as discover_module


def _state(admin_id, admin_level):
    return HarmonizeState(
        recipe={'admin_id': AdminId(), 'admin_level': admin_level},
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
        spine=None,
    )


def test_a_childless_unit_gets_an_empty_assignment(monkeypatch):
    sources = [{'recipe_id': 'admin-gadm-2026_admin2', 'admin_id': ''}]
    monkeypatch.setattr(
        discover_module, '_scan_admin_ingest_recipes', lambda level: sources
    )
    seen = {}

    def fake_get_admin_ids(level, admin_id=None, allow_empty=False, **kwargs):
        seen['allow_empty'] = allow_empty
        return []

    monkeypatch.setattr(discover_module, 'get_admin_ids', fake_get_admin_ids)

    state = discover_module.discover_sources(_state('AQ', 2))

    assert seen['allow_empty'] is True
    assert state.metadata['source_assignment'] == {}
    assert state.metadata['unassigned_admin_ids'] == []
    assert state.spine is None
