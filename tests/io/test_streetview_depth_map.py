"""
Tests for the Google Street View depth-map decoder.

The decoder had never been exercised: a missing division operator called a
float, and the base64 payload was wrapped with ``np.array`` (a 0-d array)
rather than ``np.frombuffer``. These tests parse a fabricated depth map end
to end and check the depths against a closed-form value. No network access.
"""

import base64
import struct

import numpy as np
import pytest

from openplaces.io.scrapers.google_streetview import GoogleStreetview

SKY_DEPTH = 9999999999999999999.0


def _build_depth_map_string(
    width: int, height: int, indices: list[int], planes: list[tuple]
) -> str:
    """
    Build a fabricated base64 depth-map payload in Google's byte layout.

    Parameters
    ----------
    width
        Depth-map width in pixels.
    height
        Depth-map height in pixels.
    indices
        Plane index per pixel, one byte each.
    planes
        Plane definitions as (nx, ny, nz, d) tuples.

    Returns
    -------
    str
        URL-safe base64 payload with the padding stripped.
    """
    header_size = 8
    offset = header_size
    # Google's layout overlaps the offset field's high byte with the first
    # index byte: the header is 8 bytes but offset is read as a uint16 at
    # byte 7. A fixture is only faithful when its first index is 0.
    assert indices[0] == 0
    payload = bytearray()
    payload.append(header_size)
    payload += struct.pack('<H', len(planes))
    payload += struct.pack('<H', width)
    payload += struct.pack('<H', height)
    payload.append(offset & 0xFF)
    payload += bytes(indices)
    for normal_x, normal_y, normal_z, dist in planes:
        payload += struct.pack('<ffff', normal_x, normal_y, normal_z, dist)
    encoded = base64.urlsafe_b64encode(bytes(payload)).decode('ascii')
    return encoded.rstrip('=')


@pytest.fixture(name='scraper')
def _scraper() -> GoogleStreetview:
    """Build a GoogleStreetview without running its network validation."""
    return GoogleStreetview.__new__(GoogleStreetview)


def test_parse_dmap_str_returns_a_byte_vector(scraper):
    """The decoded payload is a 1-d uint8 array, not a 0-d bytes scalar."""
    b64_string = _build_depth_map_string(
        2, 1, [0, 1], [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 10.0)]
    )

    data = scraper._parse_dmap_str(b64_string)

    assert data.ndim == 1
    assert data.dtype == np.uint8
    assert data[0] == 8


def test_depth_map_decodes_to_closed_form_depths(scraper):
    """
    A single horizontal plane yields ``|d / cos(theta)|`` at every pixel.

    With a normal of (0, 0, 1) the ray/normal dot product reduces to
    cos(theta), which for a two-row map is +/- sqrt(2)/2, so every lit
    pixel must come back at ``d * sqrt(2)``. Sky pixels (plane index 0)
    keep the sentinel depth.
    """
    width, height = 4, 2
    indices = [0, 1, 1, 1, 1, 1, 1, 0]
    planes = [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 10.0)]
    b64_string = _build_depth_map_string(width, height, indices, planes)

    data = scraper._parse_dmap_str(b64_string)
    header = scraper._parse_dmap_header(data)
    assert header == {
        'headerSize': 8,
        'numberOfPlanes': 2,
        'width': width,
        'height': height,
        'offset': 8,
    }

    parsed = scraper._parse_dmap_planes(header, data)
    assert parsed['indices'] == indices
    assert parsed['planes'][1]['d'] == pytest.approx(10.0)
    assert parsed['planes'][1]['n'] == pytest.approx([0.0, 0.0, 1.0])

    result = scraper._compute_dmap(header, parsed['indices'], parsed['planes'])
    depth_map = result['depthMap'].reshape((height, width))

    # _compute_dmap writes each row mirrored, so pixel x lands at
    # width - x - 1; the two sky pixels are therefore at (0, 3) and (1, 0).
    assert depth_map[0, 3] == SKY_DEPTH
    assert depth_map[1, 0] == SKY_DEPTH

    lit = np.delete(depth_map.ravel(), [3, 4])
    assert np.all(np.isfinite(lit))
    assert lit == pytest.approx(10.0 * np.sqrt(2.0))
