"""
In-memory imagery fetching from Google Street View and Google Satellite for
building footprints.

There is no image ingest stage, and this module writes nothing: Google's
Static API policy prohibits pre-fetching, indexing, storing, or caching its
content. `Ingester.ingest` returns early for any recipe setting
``image_scraper``, and the enrichment steps that consume imagery call
`fetch_images_in_memory` per run instead, keeping the pixels only for the
inference that follows. Panorama and place ids, the one thing the policy
permits storing, travel on each image's metadata.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon

from openplaces.io.scrapers.types import AssetInventory, ImageSet
from openplaces.recipe import get_recipe_by_id

# Substrings marking a credential field as secret; such fields are never printed.
_SECRET_MARKERS = ('key', 'secret', 'token', 'password', 'signing')


class ImageScraperError(RuntimeError):
    """An image scraper could not be initialized (e.g. credentials/billing).

    Raised by `_build_scraper` so batch callers can skip a single image recipe
    (e.g. Street View when its Google Cloud project lacks billing) and continue
    with the rest of the run, rather than aborting.
    """


def _safe_credential_summary(credentials: dict) -> str:
    """Return identifying credential fields safe to display, secrets redacted.

    Shows non-secret fields (e.g. a ``project`` or ``account`` entry the user
    adds to credentials.yaml) so a failure can name the offending project or
    account without ever printing the api key or signing secret.

    Parameters
    ----------
    credentials
        The credential mapping for one service from credentials.yaml.

    Returns
    -------
    str
        Comma-separated ``field=value`` pairs, or '' when nothing is safe to
        show.
    """
    shown = {
        key: value
        for key, value in credentials.items()
        if not any(marker in key.lower() for marker in _SECRET_MARKERS)
    }
    return ', '.join(f'{key}={value}' for key, value in shown.items())


def fetch_images_in_memory(
    image_recipe_id: str,
    entities,
    verbose: bool = False,
) -> ImageSet:
    """Fetch imagery for *entities* and return it in memory.

    Nothing is written to disk. Google's Static API policy prohibits
    "pre-fetching, indexing, storing, or caching" of content, so openplaces
    has no image ingest stage and keeps no image cache: enrichment calls this
    for each run, consumes the pixels, and lets them go.

    Panorama and place ids are the one thing the policy permits storing
    indefinitely; they travel on each image's ``metadata`` so a
    classification stays explainable after the pixels are gone.

    Parameters
    ----------
    image_recipe_id
        Recipe describing the imagery source and its camera parameters.
    entities
        GeoDataFrame of spine entities to photograph. Point geometries are
        converted to a small proxy polygon so a heading can be derived.
    verbose
        Report the fetch tally.

    Returns
    -------
    ImageSet
        Images carrying in-memory payloads, keyed by entity id.

    Raises
    ------
    ImageScraperError
        If the scraper cannot be initialized (e.g. missing credentials).
    """
    recipe = get_recipe_by_id(image_recipe_id)
    scraper_name = recipe['image_scraper']
    scraper = _build_scraper(scraper_name, recipe, verbose=verbose)

    if entities is None or len(entities) == 0:
        return ImageSet()

    entity_type = recipe.get('entity_type', 'entity')
    inventory = _gdf_to_asset_inventory(entities)
    image_set = scraper.get_images(
        inventory,
        entity_type=entity_type,
        download_year=date.today().year,
    )

    if verbose and image_set.counts:
        tally = ', '.join(
            f'{count:,d} {label}' for label, count in image_set.counts.items() if count
        )
        print(f'  Images fetched in memory: {tally or "none"}')

    return image_set


def _build_scraper(scraper_name: str, recipe: dict, verbose: bool = False):
    """Instantiate the appropriate BRAILS++ scraper from the recipe.

    Parameters
    ----------
    scraper_name
        One of ``'google_streetview'`` or ``'google_satellite'``.
    recipe
        The recipe dict, which may contain ``source.api_key_path`` for
        street view.

    Returns
    -------
    GoogleStreetview or GoogleSatellite
        Instantiated scraper object.

    Raises
    ------
    ValueError
        If *scraper_name* is not recognised.
    """
    if scraper_name == 'google_satellite':
        from openplaces.io.scrapers.google_satellite import GoogleSatellite

        return GoogleSatellite()

    if scraper_name == 'google_streetview':
        from openplaces.config import cfg
        from openplaces.io.scrapers.google_streetview import GoogleStreetview

        credentials = _load_credentials(scraper_name)
        try:
            return GoogleStreetview(
                {
                    'apiKey': credentials['api_key'],
                    'urlSigningSecret': credentials.get('url_signing_secret'),
                    'pitch': recipe.get('street_pitch', 0),
                    'verbose': verbose,
                    'max_workers': recipe.get('max_workers', 8),
                    'batch_size': recipe.get('batch_size', 32),
                }
            )
        except (ValueError, ConnectionError) as exception:
            summary = _safe_credential_summary(credentials)
            using = f'\nUsing credentials: {summary}' if summary else ''
            raise ImageScraperError(
                f"Could not initialize the '{scraper_name}' image scraper from "
                f'{cfg.credentials_path}.{using}\n'
                f'Google reported: {exception}\n'
                'If this is a billing error, enable billing on the Google Cloud '
                'project that owns this key and confirm the Street View Static '
                'API is enabled. Add a non-secret `project:` or `account:` field '
                "to this service's credentials entry to name it here."
            ) from exception

    raise ValueError(
        f"Unknown image_scraper '{scraper_name}'. "
        "Use 'google_satellite' or 'google_streetview'."
    )


def _load_api_key(scraper_name: str) -> str:
    """Return the API key for *scraper_name* from credentials.yaml.

    Parameters
    ----------
    scraper_name
        Service identifier matching a key in credentials.yaml
        (e.g. ``'google_streetview'``).

    Returns
    -------
    str
        API key string.

    Raises
    ------
    ValueError
        If the service or ``api_key`` field is absent from credentials.yaml.
    """
    return _load_credentials(scraper_name)['api_key']


def _load_credentials(scraper_name: str) -> dict:
    """Return and validate the credential mapping for an image scraper."""
    from openplaces.config import cfg

    credentials = cfg.get_credentials(scraper_name)
    if 'api_key' not in credentials:
        raise ValueError(
            f"Credentials for '{scraper_name}' found but missing 'api_key'.\n"
            f'Edit {cfg.credentials_path} and ensure the entry contains:\n\n'
            f'  {scraper_name}:\n'
            f'    api_key: YOUR_KEY_HERE\n'
        )
    return credentials


_POINT_PROXY_DEG = 0.0001


def _gdf_to_asset_inventory(gdf: gpd.GeoDataFrame) -> AssetInventory:
    """Convert a building GeoDataFrame to a BRAILS++ AssetInventory.

    Accepts both polygon footprints and point building locations (e.g. NSI).
    Geometries are reprojected to WGS 84 (EPSG:4326).  For MultiPolygon
    entries the largest polygon is used.  Point geometries are converted to
    a tiny proxy polygon (±``_POINT_PROXY_DEG`` ≈ ±11 m) so that BRAILS can
    derive a heading toward the building; FOV defaults to the buffer minimum.

    Parameters
    ----------
    gdf
        GeoDataFrame with a geometry column containing building locations
        (Polygon, MultiPolygon, or Point).

    Returns
    -------
    AssetInventory
        Inventory keyed by the GeoDataFrame index values.
    """
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    inventory: dict = {}

    for idx, row in gdf_wgs84.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        if isinstance(geom, Point):
            lon, lat = geom.x, geom.y
            d = _POINT_PROXY_DEG
            geom = Polygon(
                [
                    (lon - d, lat - d),
                    (lon + d, lat - d),
                    (lon + d, lat + d),
                    (lon - d, lat + d),
                ]
            )
        if not isinstance(geom, Polygon):
            continue
        coords = list(geom.exterior.coords)
        n_stories = (
            float(row['n_stories'])
            if 'n_stories' in gdf.columns and pd.notna(row.get('n_stories'))
            else None
        )
        inventory[idx] = SimpleNamespace(coordinates=coords, n_stories=n_stories)

    return AssetInventory(inventory=inventory)
