"""_expand_auto_discover excludes recipes flagged exclude_from_auto_discover
(e.g. a legacy/reference parcel dataset meant only for an explicit crosswalk,
not the canonical spine's geometry)."""

import pandas as pd

import openplaces.io.harmonizer.spine as spine_module
import openplaces.recipe as recipe_module
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState

_ROW_COLS = [
    'admin_id',
    'source_id',
    'version',
    'entity_type',
    'exclude_from_auto_discover',
]


def _rows(records):
    # `find_recipes` carries `recipe_id`, the recipe file's stem, which
    # `_expand_auto_discover` reads rather than rebuilding.
    return pd.DataFrame(
        [
            {
                'recipe_id': (
                    f'{r["admin_id"]}_{r["entity_type"]}-'
                    f'{r["source_id"]}-{r["version"]}'
                ),
                **r,
            }
            for r in records
        ]
    )


def test_expand_auto_discover_skips_excluded_recipe(monkeypatch):
    rows = _rows(
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
    rows = _rows(
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


def test_expand_auto_discover_skips_an_explicitly_listed_layer(monkeypatch):
    # A bundled additional_layers table listed explicitly must not be
    # rediscovered: the guard compared a bare recipe id against the
    # (recipe_id, layer) key, so it could never match and the source was
    # merged twice.
    monkeypatch.setattr(
        spine_module, 'find_recipes', lambda *a, **k: pd.DataFrame(columns=_ROW_COLS)
    )
    monkeypatch.setattr(
        recipe_module,
        'find_additional_layer_recipes',
        lambda *a, **k: [
            {
                'recipe_id': 'US-MA_parcel-massgis-2025',
                'label': 'massgis',
                'layer': 'property',
            }
        ],
    )

    state = HarmonizeState(
        recipe={'admin_id': AdminId('US')},
        admin_id=AdminId('US-MA-MI'),
        verbose=False,
        timer=None,
        spine=None,
    )
    explicit = {
        'recipe_id': 'US-MA_parcel-massgis-2025',
        'label': 'massgis',
        'layer': 'property',
    }
    discovered = spine_module._expand_auto_discover(
        [explicit, {'auto_discover': True, 'entity_type': 'property'}], state
    )

    assert discovered == [explicit]


def test_expand_auto_discover_keeps_a_layer_of_an_explicit_host(monkeypatch):
    # The host recipe listed without a layer is a different table from
    # its bundled layer, so the layer is still discovered.
    monkeypatch.setattr(
        spine_module, 'find_recipes', lambda *a, **k: pd.DataFrame(columns=_ROW_COLS)
    )
    monkeypatch.setattr(
        recipe_module,
        'find_additional_layer_recipes',
        lambda *a, **k: [
            {
                'recipe_id': 'US-MA_parcel-massgis-2025',
                'label': 'massgis',
                'layer': 'property',
            }
        ],
    )

    state = HarmonizeState(
        recipe={'admin_id': AdminId('US')},
        admin_id=AdminId('US-MA-MI'),
        verbose=False,
        timer=None,
        spine=None,
    )
    host = {'recipe_id': 'US-MA_parcel-massgis-2025', 'label': 'massgis'}
    discovered = spine_module._expand_auto_discover(
        [host, {'auto_discover': True, 'entity_type': 'property'}], state
    )

    assert [s.get('layer') for s in discovered] == [None, 'property']
