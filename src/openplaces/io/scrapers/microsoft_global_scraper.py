"""Fetch Microsoft GlobalMLBuildingFootprints tiles for one admin unit.

Microsoft publishes its global building footprints as quadkey-partitioned,
gzip-compressed line-delimited GeoJSON, indexed by a `dataset-links.csv`
that carries a `QuadKey` and a `Url` per tile but no tile geometry. That
shape does not fit either ingest partition mode openplaces already has:
`download_by: {partition: tile_id}` needs a companion tile recipe with
tile polygons (which cannot be built without geometry the index does not
publish), and `latlon_tile` reads cloud GeoParquet with a bbox column.

Rather than add a quadkey-to-polygon tile entity and the recipe hook to
derive its geometry, this scraper resolves the tiles for one admin unit
in memory: quadkeys are a deterministic function of longitude, latitude
and zoom, so the covering set can be computed from the unit's bounding
box without any persisted tile grid. It writes one `.geojsonl` per admin
unit, which GDAL's GeoJSONSeq driver reads directly.

Nothing here is specific to a country or region: the region name and the
index URL are recipe inputs (`scraper_options`).
"""

import gzip
import io
import math

import pandas as pd
import requests

from openplaces.api import get_admin
from openplaces.io import request_headers

DEFAULT_INDEX_URL = (
    'https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv'
)

# Microsoft's index is published at zoom 9.
DEFAULT_ZOOM = 9

# Sampling density across the admin unit's bounding box. A zoom-9 tile is
# roughly 0.7 degrees of longitude, so a 60x60 lattice cannot step over a
# tile for any admin unit smaller than a large country.
_LATTICE_STEPS = 60


def quadkey(lat: float, lon: float, zoom: int) -> str:
    """Return the Bing Maps quadkey containing a point.

    Parameters
    ----------
    lat, lon : float
        Point coordinates in EPSG:4326.
    zoom : int
        Tile zoom level.

    Returns
    -------
    str
        Quadkey of length ``zoom``.
    """
    sin_lat = math.sin(lat * math.pi / 180)
    n = 1 << zoom
    tile_x = int(((lon + 180) / 360) * n)
    tile_y = int((0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n)
    tile_x = min(max(tile_x, 0), n - 1)
    tile_y = min(max(tile_y, 0), n - 1)

    digits = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if tile_x & mask:
            digit += 1
        if tile_y & mask:
            digit += 2
        digits.append(str(digit))
    return ''.join(digits)


def quadkeys_for_bounds(bounds, zoom: int = DEFAULT_ZOOM) -> set[str]:
    """Return every quadkey intersecting a bounding box.

    Parameters
    ----------
    bounds : tuple of float
        ``(minx, miny, maxx, maxy)`` in EPSG:4326.
    zoom : int
        Tile zoom level.

    Returns
    -------
    set of str
        Covering quadkeys.
    """
    minx, miny, maxx, maxy = bounds
    keys = set()
    for i in range(_LATTICE_STEPS + 1):
        lon = minx + (maxx - minx) * i / _LATTICE_STEPS
        for j in range(_LATTICE_STEPS + 1):
            lat = miny + (maxy - miny) * j / _LATTICE_STEPS
            keys.add(quadkey(lat, lon, zoom))
    return keys


def fetch(
    partition_id=None,
    target_path=None,
    portal_url=None,
    admin_id_to_download=None,
    label=None,
    redownload=False,
    verbose=False,
    index_url: str = DEFAULT_INDEX_URL,
    region_name: str | None = None,
    zoom: int = DEFAULT_ZOOM,
    admin_recipe_id: str | None = None,
    **_ignored,
):
    """Write one admin unit's Microsoft global footprints as `.geojsonl`.

    Parameters
    ----------
    partition_id : str, optional
        Unused; accepted for the shared scraper entrypoint signature.
    target_path : pathlib.Path
        Where to write the combined line-delimited GeoJSON.
    portal_url : str, optional
        Unused; accepted for the shared scraper entrypoint signature.
    admin_id_to_download : str
        Admin unit whose bounding box selects the tiles.
    label : str, optional
        Unused; accepted for the shared scraper entrypoint signature.
    redownload : bool
        Unused: this scraper always refetches when called.
    verbose : bool
        Print tile selection and byte counts.
    index_url : str
        `dataset-links.csv` index. Recipe input via `scraper_options`.
    region_name : str, optional
        Value of the index's `Location` column to restrict to (e.g.
        `'UnitedStates'`). Tiles straddling a national border appear under
        each region, so leaving this unset fetches both sides.
    zoom : int
        Zoom level the index is published at.
    admin_recipe_id : str, optional
        Admin recipe to read the unit's geometry from. Defaults to
        whatever `get_admin` resolves for the unit's own level.

    Returns
    -------
    pathlib.Path or None
        `target_path`, or None when no tile covers the unit, which the
        Ingester treats as a soft skip.
    """
    # Pass the unit's own level explicitly: `get_admin` otherwise has to
    # guess which admin recipe to read and can land on a global one (GADM)
    # that has no row for a US county, yielding an empty frame.
    level = admin_id_to_download.count('-') + 1
    admin = get_admin(
        admin_id_to_download, level=level, geom=True, recipe=admin_recipe_id
    )
    bounds = admin.total_bounds

    wanted = quadkeys_for_bounds(bounds, zoom)

    index = pd.read_csv(
        io.StringIO(
            requests.get(index_url, headers=request_headers(), timeout=600).text
        )
    )
    index['QuadKey'] = index['QuadKey'].astype(str).str.zfill(zoom)
    selected = index[index['QuadKey'].isin(wanted)]
    if region_name is not None:
        selected = selected[selected['Location'] == region_name]

    if selected.empty:
        if verbose:
            print(f'No Microsoft global tiles cover {admin_id_to_download}.')
        return None

    if verbose:
        print(f'{admin_id_to_download}: {len(selected)} quadkey tile(s) at zoom {zoom}')

    n_bytes = 0
    with open(target_path, 'wb') as out:
        for _, row in selected.iterrows():
            payload = requests.get(
                row['Url'], headers=request_headers(), timeout=1800
            ).content
            body = gzip.decompress(payload)
            if not body.endswith(b'\n'):
                body += b'\n'
            out.write(body)
            n_bytes += len(body)

    if verbose:
        print(f'  wrote {n_bytes / 1e6:.1f} MB to {target_path.name}')
    return target_path
