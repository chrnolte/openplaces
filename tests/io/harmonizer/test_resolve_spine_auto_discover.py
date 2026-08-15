"""_expand_auto_discover excludes recipes flagged exclude_from_auto_discover
(e.g. a legacy/reference parcel dataset meant only for an explicit crosswalk,
not the canonical spine's geometry)."""

import pandas as pd

import openplaces.io.harmonizer.spine as spine_module
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState


def test_expand_auto_discover_skips_excluded_recipe(monkeypatch):
    rows = pd.DataFrame(
        [
            {
                'admin_id': 'US-MA',
                'source_id': 'massgis',
                'version': '2025',
                'entity_type': 'parcel',
                'exclude_from_auto_discover': False,
            },
            {
                'admin_id': 'US-MA',
                'source_id': 'placeslab',
                'version': 'fmv2026',
                'entity_type': 'parcel',
                'exclude_from_auto_discover': True,
            },
        ]
    )
    monkeypatch.setattr(spine_module, 'find_recipes', lambda *a, **k: rows)

    state = HarmonizeState(
        recipe={'admin_id': AdminId('US')},
        admin_id=AdminId('US-MA-MI'),
        verbose=False,
        timer=None,
        spine=None,
    )
    discovered = spine_module._expand_auto_discover(
        [{'auto_discover': True, 'entity_type': 'parcel'}], state
    )

    recipe_ids = {s['recipe_id'] for s in discovered}
    assert recipe_ids == {'US-MA_parcel-massgis-2025'}


def test_expand_auto_discover_orders_most_specific_admin_id_first(monkeypatch):
    # A more-specific (county) recipe must sort before a less-specific
    # (state) one even when it has an OLDER version -- resolve_spine treats
    # discovered[0] as the primary/geometry-tie winner, so specificity, not
    # version, must decide who that is.
    rows = pd.DataFrame(
        [
            {
                'admin_id': 'US-NC',
                'source_id': 'nconemap',
                'version': '2026',
                'entity_type': 'parcel',
                'exclude_from_auto_discover': False,
            },
            {
                'admin_id': 'US-NC-BL',
                'source_id': 'bladenco',
                'version': '2020',
                'entity_type': 'parcel',
                'exclude_from_auto_discover': False,
            },
        ]
    )
    monkeypatch.setattr(spine_module, 'find_recipes', lambda *a, **k: rows)

    state = HarmonizeState(
        recipe={'admin_id': AdminId('US')},
        admin_id=AdminId('US-NC-BL'),
        verbose=False,
        timer=None,
        spine=None,
    )
    discovered = spine_module._expand_auto_discover(
        [{'auto_discover': True, 'entity_type': 'parcel'}], state
    )

    assert [s['recipe_id'] for s in discovered] == [
        'US-NC-BL_parcel-bladenco-2020',
        'US-NC_parcel-nconemap-2026',
    ]
