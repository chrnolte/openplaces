"""
Image ingestion from Google Street View and Google Satellite for building
footprints.

Called by `Ingester._ingest_download_partition` when ``recipe['image_scraper']``
is set.  Images are saved to the external data directory; a metadata parquet
(entity_id → image_path + camera fields) is written alongside them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon

from openplaces.core.schema import AdminId
from openplaces.io.readers import get_admin, get_admin_ids, get_entities
from openplaces.io.scrapers.types import AssetInventory, ImageSet
from openplaces.recipe import get_output_path, get_recipe_by_id, get_save_admin_level

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


def fetch_images_by_admin(
    ingester,
    n_sample: int | None = None,
    target_recipe_id: str | None = None,
    redownload: bool = False,
) -> None:
    """Fetch Google images per building and write metadata parquets per admin unit.

    Called by `Ingester._ingest_download_partition` when
    ``recipe['image_scraper']`` is set.  For each admin unit in
    ``ingester.admin_ids_to_process``:

    1. Load buildings from *target_recipe_id* (falls back to
       ``recipe['entity_recipe']``).
    2. Convert to `AssetInventory`.
    3. Run the appropriate scraper (``google_streetview`` or
       ``google_satellite``).
    4. Save images to the external directory under
       ``{admin_path}/{entity_path}/images/``.
    5. Write a metadata parquet (entity_id, image_path, camera fields) to
       the standard external path.

    Parameters
    ----------
    ingester
        `openplaces.io.ingester.Ingester` instance with resolved
        ``admin_ids_to_process`` and ``recipe``.
    n_sample
        If set, cap the number of buildings processed per admin unit.
        Useful for test runs.
    target_recipe_id
        Recipe ID of the harmonized entity to photograph (e.g.
        ``'US_building-nsi-2022'`` for NSI point buildings,
        ``'US_footprint-spine-2026'`` for polygon footprints).  When
        ``None``, falls back to ``recipe['entity_recipe']``.
    redownload
        Missing images are always fetched (first ingest downloads imagery).
        When ``True``, images that already exist on disk are re-downloaded
        and overwritten rather than reused.
    """
    recipe = ingester.recipe
    scraper_name = recipe['image_scraper']
    scraper = _build_scraper(scraper_name, recipe, verbose=ingester.verbose)

    if not target_recipe_id:
        raise ValueError(
            f"Image recipe {recipe.get('entity')} is missing 'entity_recipe' "
            f'and no target_recipe_id was passed.'
        )
    entity_recipe = get_recipe_by_id(target_recipe_id)
    entity_obj = entity_recipe.get('entity')
    entity_type = entity_obj.entity_type if entity_obj else 'entity'
    fp_save_level = get_save_admin_level(entity_recipe)

    for admin_id_raw in ingester.admin_ids_to_process:
        admin_id = (
            AdminId(admin_id_raw) if isinstance(admin_id_raw, str) else admin_id_raw
        )
        if ingester.verbose:
            print(f'Fetching {scraper_name} images for {admin_id}...')

        # Load footprints at the footprint recipe's save level, then clip
        fp_admin_id = AdminId(*admin_id.levels[:fp_save_level])
        footprints_all = get_entities(entity_recipe, fp_admin_id, geom=True)
        if footprints_all is None or footprints_all.empty:
            if ingester.verbose:
                print(f'  No footprints found for {fp_admin_id}, skipping.')
            continue

        if fp_admin_id != admin_id:
            boundary = get_admin(admin_id, geom=True).to_crs(footprints_all.crs)
            footprints = footprints_all[
                footprints_all.geometry.within(boundary.union_all())
            ].copy()
        else:
            footprints = footprints_all

        if footprints is None or footprints.empty:
            if ingester.verbose:
                print(f'  No footprints found for {admin_id}, skipping.')
            continue

        n_before = len(footprints)
        footprints = footprints[~footprints.geometry.to_wkt().duplicated(keep='first')]
        if ingester.verbose and len(footprints) < n_before:
            print(f'  Removed {n_before - len(footprints):,d} duplicate geometries.')

        if n_sample is not None:
            n_total = len(footprints)
            footprints = footprints.head(n_sample)
            if ingester.verbose:
                print(f'  Sampling {n_sample} of {n_total:,} footprints.')

        if (
            ingester.verbose
            and scraper_name == 'google_streetview'
            and 'n_stories' in footprints.columns
        ):
            n_with = footprints['n_stories'].notna().sum()
            print(f'  n_stories: {n_with:,d} of {len(footprints):,d} buildings')

        inventory = _gdf_to_asset_inventory(footprints)

        output_path = get_output_path(ingester.recipe, admin_id)
        image_dir = output_path.parent
        image_dir.mkdir(parents=True, exist_ok=True)

        today = date.today()
        image_set = scraper.get_images(
            inventory,
            str(image_dir),
            entity_type=entity_type,
            download_year=today.year,
            redownload=redownload,
        )

        zoom_level = recipe.get('zoom_level')
        download_date = today.isoformat()
        meta_df = _image_set_to_df(
            image_set, footprints, scraper_name, zoom_level, download_date
        )
        meta_df.to_parquet(output_path)

        if ingester.verbose:
            n_missing = int(meta_df['image_path'].isna().sum())
            if n_missing:
                print(
                    f'  No imagery found for {n_missing:,d} of '
                    f'{len(meta_df):,d} buildings.'
                )
            print(f'  Saved metadata → {output_path}')


def load_image_metadata(image_recipe_id: str, admin_id) -> pd.DataFrame | None:
    """Load persisted image metadata for an admin unit, indexed by entity id.

    Reads the metadata parquet(s) written by `fetch_images_by_admin` for the
    image recipe `image_recipe_id`. When the image recipe saves at a finer admin
    level than `admin_id` (e.g. images saved per town but `admin_id` is a
    county), the per-child frames are concatenated. Tolerant of missing files:
    children without imagery are skipped and ``None`` is returned when no
    metadata exists at all (e.g. a scraper whose ingest was skipped).

    Parameters
    ----------
    image_recipe_id : str
        Recipe ID of the image entity (e.g. ``'image-googlesatellite-z20'``).
    admin_id : str or AdminId
        Admin unit to load metadata for.

    Returns
    -------
    pandas.DataFrame or None
        Metadata indexed by entity (footprint) id with an ``image_path`` column
        (plus ``image_found`` and camera fields), or ``None`` if no metadata
        files are found.
    """
    admin_id = AdminId(admin_id) if isinstance(admin_id, str) else admin_id
    image_recipe = get_recipe_by_id(image_recipe_id)
    save_level = get_save_admin_level(image_recipe)

    if save_level > admin_id.get_level():
        child_admin_ids = get_admin_ids(save_level, admin_id)
    else:
        child_admin_ids = [admin_id]

    frames = []
    for child_admin_id in child_admin_ids:
        try:
            frames.append(get_entities(image_recipe, child_admin_id))
        except FileNotFoundError:
            continue

    if not frames:
        return None
    return pd.concat(frames)


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


def _image_set_to_df(
    image_set: ImageSet,
    footprints: gpd.GeoDataFrame,
    scraper_name: str,
    zoom_level: int | None = None,
    download_date: str | None = None,
) -> pd.DataFrame:
    """Flatten an ImageSet into a metadata DataFrame.

    Parameters
    ----------
    image_set
        Result from ``scraper.get_images()``.
    footprints
        Original footprints GeoDataFrame (used to align index).
    scraper_name
        Scraper identifier; determines which metadata columns to expand.
    zoom_level
        Tile zoom level used during download (satellite only).
    download_date
        ISO-format date string (YYYY-MM-DD) when the images were fetched.

    Returns
    -------
    DataFrame
        One row per queried entity (index matches the entity IDs).  Columns
        include ``image_path`` (null where no imagery was found),
        ``image_found``, ``download_date``, optionally ``zoom_level``, and
        for street view, camera metadata (``cam_lat``, ``cam_lon``,
        ``cam_elev``, ``cam_heading``, ``pano_tilt``, ``pano_fov``,
        ``pano_roll``).
    """
    rows = {}
    image_dir = Path(image_set.dir_path)

    for key, image in image_set.images.items():
        image_path = str(image_dir / image.filename)
        row: dict = {'image_path': image_path}
        if zoom_level is not None:
            row['zoom_level'] = zoom_level
        if download_date is not None:
            row['download_date'] = download_date

        if scraper_name == 'google_streetview' and image.metadata:
            meta = image.metadata
            cam_latlon = meta.get('camLatLon') or (None, None)
            row['cam_lat'] = cam_latlon[0] if cam_latlon else None
            row['cam_lon'] = cam_latlon[1] if cam_latlon else None
            row['cam_elev'] = meta.get('camElev')
            row['cam_heading'] = meta.get('camHeading')
            row['cam_pitch'] = meta.get('camPitch')
            row['pano_tilt'] = meta.get('panoTilt')
            row['pano_fov'] = meta.get('panoFOV')
            row['pano_roll'] = meta.get('panoRoll')

        rows[key] = row

    # Keep a row for every queried entity so misses are tracked in the
    # metadata parquet (null image_path); the enricher skips those rows.
    df = pd.DataFrame.from_dict(rows, orient='index').reindex(footprints.index)
    if 'image_path' not in df.columns:
        df['image_path'] = None
    df['image_found'] = df['image_path'].notna()
    df.index.name = footprints.index.name
    return df
