"""Build a value page for one place and serve it on this machine.

One command for a local look at a place (maintainer, 2026-10-18)::

    python -m openplaces.viz.web_scene.local boston
    python -m openplaces.viz.web_scene.local durham --phone

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
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import shapely
import yaml

from openplaces.config import cfg
from openplaces.io.delivery import TEAM_SUFFIX
from openplaces.io.readers import get_admin
from openplaces.viz.web_scene import _region_members, build_web_scene

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
        units = get_admin([state], level=3, geom=True)
        if 'admin3_id' not in units.columns:
            units = units.reset_index()
        units = units.to_crs(5070)
        touched = units.loc[units.intersects(circle_metric), 'admin3_id'].tolist()
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


def has_circle(place) -> bool:
    """Whether a place's page is a circle (and has a phone page).

    A place listing `admin_ids` shows those units whole instead, such
    as an island, and is small enough to need no phone page.
    """
    return 'admin_ids' not in place


def place_extent(place, radius_km, slug):
    """Crop, units and regions of one page of a place.

    The crop is None for a place shown as whole units. Stops when no
    bundle covers the place; reports units without bundles.
    """
    if has_circle(place):
        circle, metric = circle_around(place['center'], radius_km)
        built, regions, missing = units_and_regions(place['state'], metric)
    else:
        circle = None
        built, regions, missing = units_and_regions(
            place['state'], admin_ids=place['admin_ids']
        )
    if not built:
        raise SystemExit(f'{slug}: no delivered bundles cover this place.')
    if missing:
        print(f'{slug}: no bundles for {", ".join(missing)}; left out.')
    if place.get('team'):
        # The team twin of each region; the scene is then marked a team
        # page, published only behind a password.
        regions = [f'{r}{TEAM_SUFFIX}' for r in regions]
    return circle, built, regions


def downtown_of(place):
    """The tour's first stop: the place's `downtown`, else its center."""
    return place.get('downtown') or place.get('center')


def extent_label(place, radius_km) -> str:
    """The extent in words, for build messages."""
    return f'{radius_km:g} km' if has_circle(place) else ', '.join(place['admin_ids'])


def build_place(place, radius_km, slug, phone_scene=None) -> Path:
    """Build one local page of a place; returns its folder."""
    circle, built, regions = place_extent(place, radius_km, slug)
    output = local_root() / slug
    print(
        f'Building {slug} ({extent_label(place, radius_km)}, {len(built)} units) '
        f'into {output}'
    )
    build_web_scene(
        regions,
        built,
        output,
        crop=circle,
        elevation_recipe=ELEVATION_RECIPE,
        local_only=True,
        phone_scene=phone_scene,
        downtown=downtown_of(place),
        place_title=place['title'],
        verbose=False,
    )
    return output


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
    parser.add_argument('--deploy', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.deploy:
        raise SystemExit(
            'This command never uploads. Publishing goes through '
            "deploy_web_scene, on the maintainer's word."
        )
    place = places[args.place]
    slug = args.place
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
        )
    root = local_root()
    if not (root / slug / 'scene.json').exists():
        raise SystemExit(f'No local build of {slug} in {root}; run without --no-build.')
    serve(root, slug, args.port)


if __name__ == '__main__':
    main(sys.argv[1:])
