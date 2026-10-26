"""Pitt County's footprint patches keep share-alike ODbL data out.

Pitt's parcels and sales are CC-BY-SA-4.0 and its bundle ships under CC BY-SA,
which cannot carry ODbL data. The geospine patch closes the outline sources to
two permissive ones, and the curate patch drops the Overture buildings theme
(ODbL-1.0) from the evidence merge.
"""

from __future__ import annotations

from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

PITT = 'US-NC-PIT'
OBM = 'US_footprint-obm-2025'
MICROSOFT_LEGACY = 'US_footprint-microsoft-v2'
OVERTURE = 'US_footprint_footprint-overture-2026'


def _step(recipe, name):
    return next(s for s in recipe['pipeline'] if s.get('step') == name)


def test_pitt_outlines_come_from_two_named_sources_only():
    national = get_recipe_by_id('US_footprint-geospine-2026')
    patched, applied = apply_recipe_patches(national, PITT)
    assert applied == ['US-NC-PIT_footprint-geospine-2026']

    sources = _step(patched, 'resolve_spine')['sources']
    named = [s['recipe_id'] for s in sources if 'recipe_id' in s]
    assert named == ['US_footprint-microsoftglobal-2026', 'US-NC_footprint-ncdps-2023']
    assert {'auto_discover': False} in sources
    # The national list ranks the two ODbL layers first; Pitt reads neither.
    national_ids = [
        s.get('recipe_id') for s in _step(national, 'resolve_spine')['sources']
    ]
    assert OBM in national_ids or MICROSOFT_LEGACY in national_ids
    assert OBM not in named and MICROSOFT_LEGACY not in named


def test_pitt_labels_its_microsoft_outlines_apart_from_the_odbl_layer():
    patched, _ = apply_recipe_patches(
        get_recipe_by_id('US_footprint-geospine-2026'), PITT
    )
    labels = {
        s['recipe_id']: s.get('label')
        for s in _step(patched, 'resolve_spine')['sources']
        if 'recipe_id' in s
    }
    # geometry_source takes the label, and the delivery terms resolve it by
    # source id: `microsoft` is the legacy ODbL layer's id.
    assert labels['US_footprint-microsoftglobal-2026'] == 'microsoftglobal'


def test_pitt_curated_footprints_read_no_overture_evidence():
    national = get_recipe_by_id('US_building-openplaces-2026')
    patched, applied = apply_recipe_patches(national, PITT)
    assert applied == ['US-NC-PIT_building-openplaces-2026']

    def merged(recipe):
        return [
            e.get('recipe_id') if isinstance(e, dict) else e
            for e in _step(recipe, 'merge_enrichments')['recipes']
        ]

    assert OVERTURE in merged(national)
    assert OVERTURE not in merged(patched)
    assert _step(patched, 'apportion_curated_values')['floor_weight'] is None
