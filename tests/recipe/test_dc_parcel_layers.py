"""DC's lot layers: which one the spines rank first, and which one names
the footprint link (plans/findings/dc-public-land-zero-values.md, F1-F3).
"""

from openplaces.recipe import (
    apply_recipe_patches,
    find_reference_recipe_id,
    get_recipe_by_id,
)

DC = 'US-DC-DC'
LAYERS = [
    'US-DC_parcel-dcgovtaxlots-2026',
    'US-DC_parcel-dcgovrecordlots-2026',
    'US-DC_parcel-dcgovunplattedparcels-2026',
    'US-DC_parcel-dcgovreservations-2026',
    'US-DC_parcel-dcgovstratalots-2026',
]


def test_the_parcel_geospine_ranks_tax_lots_first():
    recipe, applied = apply_recipe_patches(
        get_recipe_by_id('US_parcel-geospine-2026'), DC
    )
    assert applied == ['US-DC_parcel-geospine-2026']
    step = next(s for s in recipe['pipeline'] if s['step'] == 'resolve_spine')
    listed = [s['recipe_id'] for s in step['sources'] if s.get('recipe_id')]
    assert listed == LAYERS
    # The national step's columns survive the patch.
    assert 'parcel_id_local' in step['keep_columns']


def test_the_footprint_link_reads_the_lots_the_parcel_spine_keeps():
    recipe, _ = apply_recipe_patches(get_recipe_by_id('US_footprint-geospine-2026'), DC)
    links = [s for s in recipe['pipeline'] if s['step'] == 'link_to_reference']
    assert links[0]['entity_type'] == 'parcel'
    assert links[0]['kept_by'] == 'US_parcel-geospine-2026'
    assert all('kept_by' not in s for s in links[1:])


def test_the_footprint_link_keeps_the_record_lots_name():
    # The link sidecar and every curate step reading it are named after
    # this choice; a new DC layer must not sort ahead of the record lots.
    assert find_reference_recipe_id('parcel', DC) == 'US-DC_parcel-dcgovrecordlots-2026'


def test_new_dc_layers_record_their_terms():
    for recipe_id in LAYERS[2:]:
        source = get_recipe_by_id(recipe_id)['entity'].source
        assert source.license
        assert source.terms_url
        assert source.redistribution_restricted is False
