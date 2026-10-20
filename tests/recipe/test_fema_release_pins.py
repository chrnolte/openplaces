"""FEMA USA Structures states held to an older release.

The 2026-09-13 release replaced most LiDAR outlines (SOURCE NGA),
the only ones with a HEIGHT, by imagery outlines without one. The
recipe pins the states where that cost most heights to the release
before it; see the recipe's PINNED RELEASES header.
"""

from openplaces.recipe import get_recipe_by_id

BUCKET = (
    'https://fema-femadata.s3.us-east-1.amazonaws.com/Partners/ORNL/USA_Structures/'
)


def _pins(recipe_id):
    return get_recipe_by_id(recipe_id)['download_by'].get('pinned_download_urls')


def test_dc_is_pinned_to_the_2025_release():
    pins = _pins('US_footprint-fema-2023')

    assert pins['US-DC'] == (
        BUCKET + 'District%20of%20Columbia/Deliverable20250606DC.zip'
    )


def test_wisconsin_is_pinned_to_the_2025_release():
    pins = _pins('US_footprint-fema-2023')

    assert pins['US-WI'] == BUCKET + 'Wisconsin/Deliverable20250606WI.zip'


def test_every_pin_names_its_own_state_file_on_the_bucket():
    pins = _pins('US_footprint-fema-2023')

    for admin_id, url in pins.items():
        state = admin_id.split('-')[1]
        assert url.startswith(BUCKET)
        assert url.rsplit('/', 1)[1].endswith(f'{state}.zip')
