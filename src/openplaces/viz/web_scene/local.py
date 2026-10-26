"""Build a value page for one place and serve it on this machine.

One command for a local look at a place (maintainer, 2026-10-18)::

    python -m openplaces.viz.web_scene.local boston
    python -m openplaces.viz.web_scene.local raleigh --phone

It builds the place's page (and with ``--phone`` its phone page) from
the delivered bundles into ``<share>/web/values-local/<slug>``, marks
every build `local_only` so `deploy_web_scene` refuses it, serves the
folder on 127.0.0.1 and prints the address to open. Nothing here
uploads: the module imports no deploy code, and asking it to deploy
stops it.

The places, their fixed centers and radii (or the whole units a place
shows instead) are the editorial list in `places.yaml` beside this
module.
"""

import argparse
import sys
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import shapely
import yaml

from openplaces.config import cfg
from openplaces.io.delivery import TEAM_SUFFIX, delivery_paths
from openplaces.io.readers import get_admin
from openplaces.viz.web_scene import (
    RESTRICTED_SUFFIX,
    _region_members,
    build_web_scene,
    read_model_files,
)

PLACES_FILE = files('openplaces.viz.web_scene') / 'places.yaml'
PARCEL_RECIPE = 'US_parcel-openplaces-2026'
FOOTPRINT_RECIPE = 'US_footprint-openplaces-2026'
ELEVATION_RECIPE = 'US_land-elevation-usgs-3dep'


def load_places() -> dict:
    """The places file: `radius_km`, `phone_radius_km` and `places`."""
    return yaml.safe_load(Path(PLACES_FILE).read_text(encoding='utf-8'))


def local_root() -> Path:
    """Where local builds go, apart from the published site's folder."""
    return Path(cfg.data_root) / 'share' / 'web' / 'values-local'


def circle_around(center, radius_km):
    """A circle of `radius_km` around (lon, lat), in EPSG:4326 and 5070."""
    point = gpd.GeoSeries([shapely.Point(*center)], crs=4326).to_crs(5070)
    metric = point.buffer(radius_km * 1000, resolution=64).iloc[0]
    return gpd.GeoSeries([metric], crs=5070).to_crs(4326).iloc[0], metric


def delivery_regions(recipe) -> list[str]:
    """The regions a recipe ships bundles for."""
    from openplaces.recipe import get_recipe_by_id

    share = get_recipe_by_id(recipe).get('share') or {}
    return list((share.get('delivery') or {}).get('regions') or [])


def units_and_regions(state, circle_metric=None, admin_ids=None):
    """Level-3 units the circle touches (or `admin_ids`) that both
    recipes deliver, and the regions that pool them. The two recipes
    may deliver a unit under differently named regions (Houston:
    parcels in `houston-tx`, buildings in `cheer-coastal-tx`); each
    layer then reads its own (`build_web_scene`). Units without
    bundles are reported."""
    if admin_ids is not None:
        touched = list(admin_ids)
    else:
        # A circle may reach into several states (Washington DC).
        states = [state] if isinstance(state, str) else list(state)
        touched = units_touched(states, circle_metric)
    members = {}
    for recipe in (PARCEL_RECIPE, FOOTPRINT_RECIPE):
        for region in delivery_regions(recipe):
            try:
                members.setdefault(region, {})[recipe] = set(
                    _region_members(recipe, region)
                )
            except KeyError:
                continue
    delivered = {
        recipe: set().union(
            *(by_recipe.get(recipe, set()) for by_recipe in members.values())
        )
        for recipe in (PARCEL_RECIPE, FOOTPRINT_RECIPE)
    }
    built = [
        a
        for a in touched
        if a in delivered[PARCEL_RECIPE] & delivered[FOOTPRINT_RECIPE]
    ]
    regions = sorted(
        region
        for region, by_recipe in members.items()
        if any(m & set(built) for m in by_recipe.values())
    )
    missing = sorted(set(touched) - set(built))
    return built, regions, missing


# Most data a page of whole units or a city's limits may load, in MB
# (maintainer, 2026-10-19). Boston's 7 km phone page loads 24 MB and
# works on the maintainer's Pixel 9; its 15 km page loads 80 MB and
# crashed it (measured 2026-10-19).
PAGE_DATA_CAP_MB = 25.0


def has_circle(place) -> bool:
    """Whether a place's page is a circle (and has a phone page).

    A place listing `admin_ids` shows those units whole instead, such
    as an island; a place naming an `extent` shows that boundary whole
    (a city's limits). Either kind stays under `PAGE_DATA_CAP_MB` and
    needs no phone page.
    """
    return 'admin_ids' not in place and 'extent' not in place


def extent_shape(place):
    """A place's named boundary, in EPSG:4326 and 5070.

    `extent` names a boundary recipe, its id column and the id: a city's
    limits from Census places (maintainer, 2026-10-19: city pages show
    their limits, so neighboring cities never overlap).
    """
    from openplaces.io.readers import get_dataset

    spec = place['extent']
    units = get_dataset(spec['recipe'], admin_id=place['state'], geom=True)
    column = spec.get('column', 'census_place_geoid')
    match = units[units[column].astype(str) == str(spec['id'])]
    if match.empty:
        raise SystemExit(
            f'{place["slug"]}: no {column} {spec["id"]} in {spec["recipe"]}.'
        )
    shape = match.to_crs(4326).union_all()
    return shape, gpd.GeoSeries([shape], crs=4326).to_crs(5070).iloc[0]


def units_touched(states, circle_metric) -> list:
    """Level-3 units of `states` that the circle (EPSG:5070) touches."""
    touched = []
    for state in states:
        units = get_admin([state], level=3, geom=True)
        if 'admin3_id' not in units.columns:
            units = units.reset_index()
        units = units.to_crs(5070)
        touched += units.loc[units.intersects(circle_metric), 'admin3_id'].tolist()
    return touched


def place_extent(place, radius_km, slug):
    """Crop, units and regions of one page of a place.

    The crop is None for a place shown as whole units. Stops when no
    bundle covers the place; reports units without bundles.
    """
    if place.get('lots_in_model_file') and 'extent' in place:
        raise SystemExit(
            f'{slug}: `extent` with `lots_in_model_file` is not supported; '
            'list `admin_ids` or use a circle.'
        )
    if place.get('lots_in_model_file'):
        # Units with no bundle: their lots come from the model files.
        # A circle reads every unit it touches, in each of `states`
        # (Washington DC reaches into Maryland and Virginia).
        if not has_circle(place):
            return None, list(place['admin_ids']), None
        circle, metric = circle_around(place['center'], radius_km)
        return (
            circle,
            units_touched(place.get('states') or [place['state']], metric),
            None,
        )
    if has_circle(place):
        circle, metric = circle_around(place['center'], radius_km)
        built, regions, missing = units_and_regions(
            place.get('states') or place['state'], metric
        )
    elif 'extent' in place:
        # Features by their center inside the boundary, as for a circle.
        circle, metric = extent_shape(place)
        built, regions, missing = units_and_regions(
            place.get('states') or place['state'], metric
        )
    else:
        circle = None
        built, regions, missing = units_and_regions(
            place['state'], admin_ids=place['admin_ids']
        )
    if place.get('only_units'):
        # Units of the circle a page may show (`only_units`), e.g. those
        # whose sources allow a public page; the rest stay out, and the
        # place's `notes` say so in Sources.
        left_out = [a for a in built if a not in place['only_units']]
        built = [a for a in built if a in place['only_units']]
        if left_out:
            print(f'{slug}: {", ".join(left_out)} left out (only_units).')
    if not built:
        raise SystemExit(f'{slug}: no delivered bundles cover this place.')
    if missing:
        print(f'{slug}: no bundles for {", ".join(missing)}; left out.')
    if place.get('team') and (
        not place.get('model_files') or place.get('team_bundles')
    ):
        # The team twin of each region; the scene is then marked a team
        # page, published only behind a password. A page of model files
        # reads the public bundles and is a team page by its models,
        # unless it asks for the team bundles (`team_bundles`).
        regions = [f'{r}{TEAM_SUFFIX}' for r in regions]
    return circle, built, regions


def place_models(place, built, slug):
    """A place's land value models handed over as files, or None.

    `model_files` in `places.yaml` is one path per unit, under the data
    root, naming it as `{admin_id}`. Stops when the place lists model
    files and none exists, so a team model page never goes out
    without its models.
    """
    template = place.get('model_files')
    if not template:
        return None
    root = Path(cfg.data_root)
    models = read_model_files([root / template.format(admin_id=a) for a in built])
    if models is None:
        raise SystemExit(f'{slug}: no model file found ({template}).')
    return models


def place_elevation(place):
    """The DEM a place's page stands on; None with `terrain: false`
    (flat ground, e.g. a county whose elevation was never fetched)."""
    return None if place.get('terrain') is False else ELEVATION_RECIPE


def place_lots_file(place, built):
    """The model files that carry a place's lots (`lots_in_model_file`),
    one per unit, or None. Stops when one is missing."""
    if not place.get('lots_in_model_file'):
        return None
    paths = [
        Path(cfg.data_root) / place['model_files'].format(admin_id=a) for a in built
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f'{place["slug"]}: no lots file {", ".join(missing)}.')
    return paths


def downtown_of(place):
    """The tour's first stop: the place's `downtown`, else its center."""
    return place.get('downtown') or place.get('center')


def extent_label(place, radius_km) -> str:
    """The extent in words, for build messages."""
    if has_circle(place):
        return f'{radius_km:g} km'
    if 'extent' in place:
        return f'limits {place["extent"]["id"]}'
    return ', '.join(place['admin_ids'])


def page_data_mb(folder) -> float:
    """MB of data a built page loads (its manifest's files)."""
    import json

    from openplaces.viz.web_scene import manifest_files

    folder = Path(folder)
    manifest = json.loads((folder / 'scene.json').read_text(encoding='utf-8'))
    return sum((folder / f).stat().st_size for f in manifest_files(manifest)) / 2**20


def check_page_size(place, folder, slug) -> float:
    """A page's data size; above `PAGE_DATA_CAP_MB` for a page of whole
    units or a city's limits, the build stops (split it into ZIP
    pages). Circle pages, which have phone pages, are reported only."""
    size = page_data_mb(folder)
    print(f'{slug}: {size:.1f} MB of data')
    if not has_circle(place) and size > PAGE_DATA_CAP_MB:
        raise SystemExit(
            f'{slug}: {size:.1f} MB is above the {PAGE_DATA_CAP_MB:g} MB a phone '
            'handles; split the place into smaller pages.'
        )
    return size


def page_variants(place, slug) -> list[tuple[str, dict]]:
    """A page's folder name and its extra `build_web_scene` arguments,
    its restricted twin first where the place lists `restricted_units`.

    The twin (`<slug>-restricted`) fills those units' withheld values
    (`fill_withheld_values`) and adds `restricted_notes` to the page's
    notes; the page itself is unchanged but for the switch to the twin.
    A phone page has no twin.
    """
    notes = list(place.get('notes') or [])
    units = place.get('restricted_units')
    if not units or slug.endswith('-phone'):
        return [(slug, {'notes': notes})]
    twin = f'{slug}{RESTRICTED_SUFFIX}'
    return [
        (
            twin,
            {
                'notes': notes + list(place.get('restricted_notes') or []),
                'restricted_units': list(units),
                'unrestricted_page': f'../{slug}/',
            },
        ),
        (slug, {'notes': notes, 'restricted_page': f'../{twin}/'}),
    ]


def bundle_date(regions) -> str | None:
    """Date the newest public parcel bundle of `regions` was written
    (YYYY-MM-DD), or None where none is on this machine."""
    stamps = []
    for region in regions or []:
        try:
            path = delivery_paths(PARCEL_RECIPE, region=region)['canonical']
        except KeyError:
            continue
        if path.exists():
            stamps.append(path.stat().st_mtime)
    if not stamps:
        return None
    return datetime.fromtimestamp(max(stamps)).strftime('%Y-%m-%d')


def build_place(place, radius_km, slug, phone_scene=None, preview_team=False) -> Path:
    """Build one local page of a place, and its restricted twin where it
    has one (`page_variants`); returns the page's folder.

    `preview_team` marks the page a team page though it reads the public
    bundles (`build_web_scene`), and names the bundle date in its notes.
    """
    circle, built, regions = place_extent(place, radius_km, slug)
    models = place_models(place, built, slug)
    for name, extra in page_variants(place, slug):
        if preview_team:
            extra = dict(extra, preview_team=True)
            extra['notes'] = [
                *extra['notes'],
                {
                    'name': 'Local preview',
                    'credit': 'Built from the public bundles of '
                    f'{bundle_date(regions) or "an unknown date"}; not published.',
                },
            ]
        output = local_root() / name
        print(
            f'Building {name} ({extent_label(place, radius_km)}, {len(built)} units) '
            f'into {output}'
        )
        build_web_scene(
            regions,
            built,
            output,
            crop=circle,
            elevation_recipe=place_elevation(place),
            local_only=True,
            phone_scene=None if name != slug else phone_scene,
            downtown=downtown_of(place),
            place_title=place['title'],
            model_files=models,
            lots_file=place_lots_file(place, built),
            models_public=bool(place.get('models_public')),
            people_basis=place.get('people_basis', 'homes'),
            verbose=False,
            **extra,
        )
        check_page_size(place, output, name)
    return local_root() / slug


def serve(root: Path, slug: str, port: int) -> None:
    """Serve `root` on 127.0.0.1 until interrupted; print the address."""
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(('127.0.0.1', port), handler)
    print(f'\nOpen http://127.0.0.1:{server.server_address[1]}/{slug}/  (Ctrl+C stops)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None) -> None:
    config = load_places()
    places = {p['slug']: p for p in config['places']}
    parser = argparse.ArgumentParser(
        prog='python -m openplaces.viz.web_scene.local',
        description='Build a value page for one place and serve it locally. '
        'Never uploads.',
    )
    parser.add_argument('place', choices=sorted(places))
    parser.add_argument(
        '--phone', action='store_true', help='also build the smaller phone page'
    )
    parser.add_argument('--radius-km', type=float, default=config['radius_km'])
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument(
        '--no-build',
        action='store_true',
        help='serve the last local build without rebuilding',
    )
    parser.add_argument(
        '--people-preview',
        action='store_true',
        help='build <place>-people, a team page of the public bundles with '
        'the people cloud, for a local look only',
    )
    parser.add_argument('--deploy', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.deploy:
        raise SystemExit(
            'This command never uploads. Publishing goes through '
            "deploy_web_scene, on the maintainer's word."
        )
    place = places[args.place]
    slug = f'{args.place}-people' if args.people_preview else args.place
    if args.phone and not has_circle(place):
        print(f'{slug} is shown as whole units and has no phone page.')
        args.phone = False
    if not args.no_build:
        phone_slug = f'{slug}-phone'
        if args.phone:
            build_place(place, config['phone_radius_km'], phone_slug)
        build_place(
            place,
            args.radius_km,
            slug,
            phone_scene=f'../{phone_slug}/' if args.phone else None,
            preview_team=args.people_preview,
        )
    root = local_root()
    if not (root / slug / 'scene.json').exists():
        raise SystemExit(f'No local build of {slug} in {root}; run without --no-build.')
    serve(root, slug, args.port)


if __name__ == '__main__':
    main(sys.argv[1:])
