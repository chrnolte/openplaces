"""Image fetching: missing images download automatically; `redownload` only
re-fetches images that already exist on disk."""

import pytest

from openplaces.io.scrapers.google_satellite import GoogleSatellite

_FOOTPRINT = [
    (-76.65, 34.64),
    (-76.65, 34.641),
    (-76.649, 34.641),
    (-76.649, 34.64),
    (-76.65, 34.64),
]


def _no_network_satellite(monkeypatch):
    """Return a GoogleSatellite whose tile fetch raises if it is ever reached."""
    scraper = GoogleSatellite()

    def _boom(*_args, **_kwargs):
        raise AssertionError('network fetch reached')

    monkeypatch.setattr(scraper, '_fetch_tiles', _boom)
    return scraper


def test_missing_image_is_fetched_on_first_ingest(monkeypatch, tmp_path):
    scraper = _no_network_satellite(monkeypatch)
    impath = tmp_path / 'footprint_TEST.tif'

    # Missing image -> always fetched, regardless of redownload.
    with pytest.raises(AssertionError, match='network fetch reached'):
        scraper._download_satellite_image(_FOOTPRINT, impath, redownload=False)


def test_existing_image_reused_without_redownload(monkeypatch, tmp_path):
    scraper = _no_network_satellite(monkeypatch)
    impath = tmp_path / 'footprint_TEST.tif'
    impath.write_bytes(b'cached')

    # Existing image, no redownload -> reused with no network.
    scraper._download_satellite_image(_FOOTPRINT, impath, redownload=False)

    assert impath.read_bytes() == b'cached'


def test_existing_image_refetched_with_redownload(monkeypatch, tmp_path):
    scraper = _no_network_satellite(monkeypatch)
    impath = tmp_path / 'footprint_TEST.tif'
    impath.write_bytes(b'cached')

    # Existing image, redownload -> re-fetched (network reached).
    with pytest.raises(AssertionError, match='network fetch reached'):
        scraper._download_satellite_image(_FOOTPRINT, impath, redownload=True)
