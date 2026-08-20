"""Imagery is fetched on the fly and never persisted.

Google's Static API policies prohibit "pre-fetching, indexing, storing, or
caching" of content, with an explicit exception for place and panorama ids.
openplaces therefore has no image ingest stage and keeps no image cache:
enrichment fetches pixels, runs inference, and drops them.

These tests replace the former `test_image_redownload.py`, which asserted the
opposite guarantee (that an image already on disk was reused without a network
call). That behaviour is the cache the policy forbids, so it was removed
rather than repaired.
"""

from pathlib import Path

import pytest

from openplaces.io.scrapers.google_satellite import GoogleSatellite
from openplaces.io.scrapers.types import AssetInventory, Image, ImageSet

_FOOTPRINT = [
    (-76.65, 34.64),
    (-76.65, 34.641),
    (-76.649, 34.641),
    (-76.649, 34.64),
    (-76.65, 34.64),
]

_IMAGE_SUFFIXES = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def test_no_image_storage_functions_remain():
    """The ingest-side storage surface is gone, not merely unused."""
    from openplaces.io.ingester import image_ingester

    assert not hasattr(image_ingester, 'fetch_images_by_admin')
    assert not hasattr(image_ingester, 'load_image_metadata')
    assert hasattr(image_ingester, 'fetch_images_in_memory')


def test_satellite_fetch_writes_nothing_to_disk(monkeypatch, tmp_path):
    """A satellite fetch leaves no file behind anywhere under a clean root."""
    scraper = GoogleSatellite()

    # Render deterministic pixels without touching the network.
    def _fake_render(_footprint):
        import numpy as np
        from rasterio.transform import from_bounds

        arr = np.zeros((256, 256, 3), dtype='uint8')
        return arr, from_bounds(-76.65, 34.64, -76.649, 34.641, 256, 256)

    monkeypatch.setattr(scraper, '_render_footprint', _fake_render)
    monkeypatch.chdir(tmp_path)

    payload = scraper.fetch_image_bytes(_FOOTPRINT)

    assert isinstance(payload, bytes) and payload
    strays = [
        path
        for path in Path(tmp_path).rglob('*')
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    assert strays == [], f'imagery was written to disk: {strays}'


def test_image_carries_payload_not_a_stored_path():
    """An in-memory image opens without any filesystem location."""
    from io import BytesIO

    from PIL import Image as PILImage

    buffer = BytesIO()
    PILImage.new('RGB', (4, 4)).save(buffer, format='PNG')
    image = Image('footprint_TEST.png', payload=buffer.getvalue())

    with image.open() as opened:
        assert opened.size == (4, 4)


def test_image_set_has_no_directory_attribute():
    """There is no shared image directory to reason about."""
    assert not hasattr(ImageSet(), 'dir_path')


def test_image_without_payload_or_path_is_an_error():
    """A failed fetch surfaces as a missing payload, not a silent empty read."""
    with pytest.raises(ValueError, match='no payload'):
        Image('footprint_TEST.png').open()


def test_inventory_round_trip_is_unchanged():
    """The asset inventory the scrapers consume is untouched by this change."""
    inventory = AssetInventory.from_dict({'F1': _FOOTPRINT})

    assert list(inventory.inventory) == ['F1']
    assert inventory.inventory['F1'].coordinates == _FOOTPRINT


def test_no_pipeline_step_rebuilds_an_image_set_from_stored_files():
    """No step may reconstruct an ImageSet out of a stored metadata table.

    The harmonize stage carried duplicates of `classify_occupancy` and
    `classify_roof_shape` that read an `image_path` column from a parquet
    and loaded the pixels it pointed at. They were dead -- nothing writes
    that parquet now -- but they were the clearest surviving description
    of the pattern the policy forbids, and a copy-paste from being live.

    The scrapers themselves are exempt: Street View writes its BRAILS++
    intermediates (depth maps, masks, crops) through a per-call scratch
    directory that is destroyed before the call returns, which is a
    temporary file rather than a cache.
    """
    src = Path(__file__).resolve().parents[3] / 'src' / 'openplaces'
    scrapers = src / 'io' / 'scrapers'
    offenders = [
        path.relative_to(src).as_posix()
        for path in src.rglob('*.py')
        if scrapers not in path.parents
        and 'image_path' in path.read_text(encoding='utf-8')
    ]

    assert offenders == [], f'image pixels still read from disk in: {offenders}'


def test_the_harmonize_stage_registers_no_image_steps():
    """Image inference is an enrich-stage concern, and only that."""
    from openplaces.io.harmonizer import _STEP_REGISTRY, _load_steps

    _load_steps()
    image_steps = {
        name
        for name in _STEP_REGISTRY
        if name in {'classify_occupancy', 'classify_roof_shape', 'detect_n_stories'}
    }

    assert image_steps == set(), f'harmonize still registers: {sorted(image_steps)}'


def test_image_recipes_declare_no_output_location():
    """An image recipe carries camera configuration, and writes nothing."""
    import yaml

    recipes = (
        Path(__file__).resolve().parents[3] / 'src' / 'openplaces' / 'recipes' / '_all'
    ).glob('image/*/*/*.yaml')

    checked = 0
    for path in recipes:
        recipe = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert 'save_to' not in recipe, f'{path.name} declares an output location'
        assert recipe.get('image_scraper'), f'{path.name} is not an image recipe'
        checked += 1

    assert checked >= 2, 'expected the satellite and street-view recipes'


def test_ingesting_an_image_recipe_is_a_no_op():
    """There is no image ingest stage at all, not merely a disabled one."""
    from openplaces.io.ingester import Ingester

    ingester = Ingester('image-googlesatellite-z20')

    # Returns without resolving admin ids, downloading, or writing.
    assert ingester.ingest() is None
