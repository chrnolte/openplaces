"""
Tests that the Google scrapers identify themselves and close their session.

These were the only outbound calls in the package that did not go through
:func:`openplaces.io.request_headers`, so they went out under the bare
python-requests agent. No network: the session is a local fake.
"""

from io import BytesIO

import pytest
from PIL import Image

from openplaces.config import cfg
from openplaces.io.scrapers import google_satellite, google_streetview


def _png_bytes() -> bytes:
    """Return a one-pixel PNG the scrapers can open."""
    buffer = BytesIO()
    Image.new('RGB', (1, 1)).save(buffer, format='PNG')
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content):
        self.content = content
        self.status_code = 200
        self.ok = True

    def raise_for_status(self):
        return None


class _FakeSession:
    """Records the headers it was given and whether it was closed."""

    instances: list = []

    def __init__(self):
        self.headers = {}
        self.urls = []
        self.closed = False
        _FakeSession.instances.append(self)

    def mount(self, prefix, adapter):
        return None

    def get(self, url, **kwargs):
        self.urls.append(url)
        return _FakeResponse(_png_bytes())

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture(name='fake_session')
def _fake_session(monkeypatch):
    _FakeSession.instances = []
    monkeypatch.setattr(google_satellite.requests, 'Session', _FakeSession)
    return _FakeSession.instances


def test_satellite_tiles_identify_openplaces(fake_session):
    """Tile requests carry the project's own agent, never a browser one."""
    scraper = google_satellite.GoogleSatellite.__new__(google_satellite.GoogleSatellite)

    scraper._fetch_tiles([10], [20])

    (session,) = fake_session
    assert session.headers['User-Agent'] == cfg.user_agent
    assert 'Mozilla' not in session.headers['User-Agent']
    assert session.urls
    assert session.closed


def test_streetview_tiles_identify_openplaces(fake_session):
    """The Street View tile session is identified and closed as well."""
    google_streetview.GoogleStreetview._download_tiles(['https://tile/1'])

    (session,) = fake_session
    assert session.headers['User-Agent'] == cfg.user_agent
    assert 'Mozilla' not in session.headers['User-Agent']
    assert session.closed
