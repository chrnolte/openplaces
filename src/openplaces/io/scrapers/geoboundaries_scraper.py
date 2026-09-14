"""Download one country's geoBoundaries file at one level, if its terms allow.

geoBoundaries (gbOpen) licenses every country and level separately: CC
BY and public-domain files sit beside ODbL and ShareAlike ones, and a
few were released by direct permission. The spine ships with the
package, so a share-alike or unreviewed file must not enter it. The
recipe therefore does not carry a download URL at all; this scraper
reads the licence sidecar committed beside the recipes
(`admin-geoboundaries-6~0~0_licenses.csv`, derived from the project's
own metadata table) and downloads only a country-level whose `tier` is
`permissive` (CC0, public domain, CC BY and the national open licences).
The other tiers, `share-alike` (ODbL, CC BY-SA: fine as a local
intermediate, never in a shipped artifact) and `unreviewed`, are held:
skipped as an unavailable partition, never fetched. `status` restates
the tier as `open` or `held`, and `held_reason` says why.

The version is pinned to a release tag rather than `main`, so a rebuild
next year reads the same polygons as today's.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request

import pandas as pd

from openplaces.io import request_headers
from openplaces.recipe import recipe_path

RELEASE = 'v6.0.0'
URL = (
    'https://github.com/wmgeolab/geoBoundaries/raw/{release}/releaseData/gbOpen/'
    '{iso3}/{level}/geoBoundaries-{iso3}-{level}.geojson'
)
LICENSES_RECIPE = 'admin-geoboundaries-6~0~0'
# The one tier whose polygons may weigh on a shipped spine or be
# redistributed with attribution.
SHIPPABLE_TIER = 'permissive'
REQUEST_INTERVAL_S = 1.0


def load_licenses() -> pd.DataFrame:
    """Return the per-file licence table committed beside the recipes."""
    path = recipe_path(None, LICENSES_RECIPE, filename='licenses.csv')
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def license_row(iso3: str, admin_level: int) -> pd.Series | None:
    """Return the sidecar row for one country and level, or None."""
    table = load_licenses()
    rows = table[
        (table['admin1_id_a3'] == iso3) & (table['admin_level'] == str(admin_level))
    ]
    return rows.iloc[0] if len(rows) else None


def _parent_polygons(admin1_id, admin_level):
    """Return the level-above polygons of this source for one country.

    None when that level has no output for the country: its file was
    held, not published, or not ingested yet. A level cannot be placed
    under a parent layer that does not exist, so its partition is
    skipped rather than raised on.
    """
    from openplaces.io.readers import get_admin

    column = f'admin{admin_level - 1}_id'
    try:
        parents = get_admin(
            admin1_id,
            level=admin_level - 1,
            recipe=f'{LICENSES_RECIPE}_admin{admin_level - 1}',
            geom=True,
            silent=True,
        ).reset_index()
    except (FileNotFoundError, OSError, ValueError):
        # Absent, or unreadable (pyarrow raises OSError or ValueError
        # for a missing or half-written file): same answer, place by
        # name instead.
        return None
    if 'geometry' not in parents:
        return None
    parents = parents[[column, 'geometry']]
    parents = parents[parents.geometry.notna() & ~parents.geometry.is_empty]
    return None if parents.empty else parents


def _attach_parents(path, target_path, admin1_id, admin_level):
    """Write the polygons with their parent admin id to `target_path`.

    geoBoundaries files carry no parent key, so the hierarchy is
    recovered here: at level 2 the parent is the country; below that,
    the level-above polygon (this source's own, already ingested and
    pinned to the spine) that a polygon overlaps most. Largest overlap
    rather than containment, because two vintages of borders never
    agree exactly and a sliver would otherwise leave a unit orphaned.
    """
    import geopandas as gpd
    import pyogrio

    # Canada's ADM1 file holds a single feature past GDAL's default
    # GeoJSON object limit; the limit is off for this read, and the
    # file is rewritten as GeoPackage so the ingester never meets it.
    pyogrio.set_gdal_config_options({'OGR_GEOJSON_MAX_OBJ_SIZE': 0})
    frame = gpd.read_file(path)
    if admin_level == 2:
        frame['admin1_id'] = admin1_id
    else:
        column = f'admin{admin_level - 1}_id'
        parents = _parent_polygons(admin1_id, admin_level)
        if parents is not None:
            pieces = gpd.overlay(
                frame[['shapeID', 'geometry']].to_crs(parents.crs),
                parents.to_crs(parents.crs),
                how='intersection',
                keep_geom_type=False,
            )
            pieces['area'] = pieces.geometry.area
            best = pieces.sort_values('area').drop_duplicates('shapeID', keep='last')
            frame['parent_admin_id'] = frame['shapeID'].map(
                dict(zip(best['shapeID'], best[column]))
            )
        else:
            # No polygons one level up (that file is held or absent):
            # place each polygon by its name against the spine, where
            # the name is unique within the country, and take that
            # unit's parent. A polygon this cannot place is dropped; it
            # would only ever have weighted a unit it cannot be tied to.
            frame['parent_admin_id'] = _parents_by_name(
                frame['shapeName'], admin1_id, admin_level
            )
        # Only a parent the spine names can hold a child: a level-above
        # polygon that pinned to nothing carries a fresh code the spine
        # does not know, and a child under it would be blanked later.
        live = _live_ids(admin1_id, admin_level - 1)
        placed = frame['parent_admin_id'].isin(live)
        frame = frame[placed]
    frame.to_file(target_path, driver='GPKG')


def _live_ids(admin1_id, level):
    """Return the spine's ids at one level for one country."""
    import pandas as pd

    from openplaces.path import spine_path

    column = f'admin{level}_id'
    spine = pd.read_csv(spine_path(level), dtype=str, keep_default_na=False)
    return set(spine.loc[spine[column].str.startswith(admin1_id + '-'), column])


def _parents_by_name(names, admin1_id, admin_level):
    """Return the parent id of the spine unit each name uniquely matches."""
    import pandas as pd

    from openplaces.io.admin_codes.anchors import normalize_name
    from openplaces.path import spine_path

    column = f'admin{admin_level}_id'
    spine = pd.read_csv(spine_path(admin_level), dtype=str, keep_default_na=False)
    own = spine[spine[column].str.startswith(admin1_id + '-')]
    by_name = {}
    for admin_id, name in zip(own[column], own['name']):
        if name.strip():
            by_name.setdefault(normalize_name(name.strip()), []).append(admin_id)
    unique = {k: v[0].rsplit('-', 1)[0] for k, v in by_name.items() if len(v) == 1}
    return (
        names.fillna('')
        .astype(str)
        .map(lambda n: unique.get(normalize_name(n.strip()), ''))
    )


def fetch(
    partition_id=None,
    target_path=None,
    portal_url=None,
    admin_id_to_download=None,
    label=None,
    redownload=False,
    verbose=False,
    admin_level=2,
    iso3=None,
    release=RELEASE,
    **options,
):
    """Download one country's boundaries at one level, as GeoPackage.

    Parameters
    ----------
    admin_id_to_download : AdminId or str
        The country, as its level-1 id.
    target_path : pathlib.Path
        Where the GeoJSON goes.
    admin_level : int
        openplaces admin level, 2 to 4 (geoBoundaries ADM1 to ADM3).
    iso3 : str
        The country's ISO 3166-1 alpha-3 code, resolved by the ingester
        from the level-1 spine (`{admin1_id_a3}` in `scraper_options`).
    release : str, optional
        geoBoundaries release tag.

    Returns
    -------
    pathlib.Path or None
        The written file, or None when the country-level has no open
        file: not published, or published under terms the spine may not
        carry.
    """
    admin_level = int(admin_level)
    if not iso3:
        from openplaces.io.readers import get_admin

        country = get_admin(str(admin_id_to_download), level=1, columns='admin1_id_a3')
        iso3 = str(country.iloc[0, 0])
    row = license_row(iso3, admin_level)
    if row is None or row['tier'] != SHIPPABLE_TIER:
        if verbose:
            reason = 'not published' if row is None else f'held ({row["tier"]})'
            print(f'  {iso3} level {admin_level}: {reason}')
        return None
    url = URL.format(release=release, iso3=iso3, level=row['geoboundaries_level'])
    request = urllib.request.Request(url, headers=request_headers())
    target_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = target_path.with_suffix('.geojson')
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            downloaded.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        # The metadata table lists a few files the release tag does not
        # carry; that is a file not published, not a failure to stop on.
        if error.code == 404:
            if verbose:
                print(f'  {iso3} level {admin_level}: listed but not in {release}')
            return None
        raise
    time.sleep(REQUEST_INTERVAL_S)
    _attach_parents(downloaded, target_path, str(admin_id_to_download), admin_level)
    downloaded.unlink()
    if verbose:
        spdx, n_units = row['license_spdx'], row['n_units']
        print(f'  {iso3} level {admin_level}: {spdx}, {n_units} units')
    return target_path
