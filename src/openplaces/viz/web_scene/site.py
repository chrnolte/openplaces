"""Build and publish the value site (placeslab.org/values).

The published counterpart of `local` (maintainer, 2026-10-18)::

    python -m openplaces.viz.web_scene.site build boston durham
    python -m openplaces.viz.web_scene.site build --all
    python -m openplaces.viz.web_scene.site landing [--no-thumbs]
    python -m openplaces.viz.web_scene.site refresh
    python -m openplaces.viz.web_scene.site publish --all --go

``build`` writes each place's page (15 km) and phone page (7 km) from
the delivered bundles into ``<share>/web/values``; ``landing`` writes
the landing page, its card photos, the list of places and the methods
page; ``refresh`` copies the page files and the known issues into every
built page without rebuilding data (for a change to the viewer alone).
``publish`` uploads, and only with ``--go`` and on the maintainer's
word: each phone page before the desktop page that sends phones to it,
one place every `--gap` seconds (Bluehost blocks bursts of SSH
connections), then the landing page, so no card links to a page that
is not up yet. Without ``--go`` it is a dry run.

The places, their order and their centers are `places.yaml`.
"""

import argparse
import json
import sys
import time
from importlib.resources import files
from pathlib import Path

from openplaces.config import cfg
from openplaces.viz.web_scene import (
    build_web_scene,
    deploy_landing,
    deploy_web_scene,
    known_issues,
)
from openplaces.viz.web_scene.local import (
    ELEVATION_RECIPE,
    circle_around,
    load_places,
    units_and_regions,
)

TITLE = 'Land & structure value'
STATIC_DIR = files('openplaces.viz.web_scene') / 'static'


def site_root() -> Path:
    """The published site's local folder."""
    return Path(cfg.data_root) / 'share' / 'web' / 'values'


def _page(place, radius_km, slug, phone_scene=None) -> None:
    circle, metric = circle_around(place['center'], radius_km)
    built, regions, missing = units_and_regions(place['state'], metric)
    if not built:
        raise SystemExit(f'{slug}: no delivered bundles cover this circle.')
    if missing:
        print(f'{slug}: no bundles for {", ".join(missing)}; left out.')
    started = time.time()
    build_web_scene(
        regions,
        built,
        site_root() / slug,
        title=TITLE,
        crop=circle,
        elevation_recipe=ELEVATION_RECIPE,
        phone_scene=phone_scene,
        verbose=False,
    )
    print(
        f'built {slug} ({radius_km:g} km, {len(built)} units) '
        f'in {time.time() - started:.0f}s',
        flush=True,
    )


def build(places, config) -> None:
    """Each place's phone page, then its page, which sends phones there."""
    for place in places:
        slug = place['slug']
        _page(place, config['phone_radius_km'], f'{slug}-phone')
        _page(place, config['radius_km'], slug, phone_scene=f'../{slug}-phone/')


def landing(config, thumbnails=True) -> None:
    """The landing page, card photos, places.json and the methods page."""
    from openplaces.viz.web_scene.landing import Place, build_landing

    root = site_root()
    cards = [
        Place(
            p['slug'],
            p['title'],
            f'Within {config["radius_km"]:g} km of {p["around"]}',
            root / p['slug'],
        )
        for p in config['places']
        if (root / p['slug'] / 'scene.json').exists()
    ]
    build_landing(cards, root, thumbnails=thumbnails)
    print('landing:', [c.slug for c in cards])


def refresh() -> None:
    """Page files and known issues into every built page, data unchanged."""
    for folder in sorted(
        p for p in site_root().iterdir() if (p / 'scene.json').exists()
    ):
        for item in Path(STATIC_DIR).iterdir():
            if item.name in ('landing.html', 'methods.html'):
                continue
            target = '.htaccess' if item.name == 'htaccess' else item.name
            (folder / target).write_bytes(item.read_bytes())
        manifest_path = folder / 'scene.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        units = sorted({u for layer in manifest['layers'] for u in layer['files']})
        manifest['known_issues'] = known_issues(units)
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        print('refreshed', folder.name)


def publish(places, go=False, gap_s=180) -> None:
    """Upload the pages, phone page first, then the landing page."""
    root = site_root()
    slugs = [s for p in places for s in (f'{p["slug"]}-phone', p['slug'])]
    for i, slug in enumerate(slugs):
        if i and go:
            time.sleep(gap_s)
        deploy_web_scene(root / slug, go=go, subpath=slug)
    if go:
        time.sleep(gap_s)
    deploy_landing(root, go=go)


def main(argv=None) -> None:
    config = load_places()
    by_slug = {p['slug']: p for p in config['places']}
    parser = argparse.ArgumentParser(prog='python -m openplaces.viz.web_scene.site')
    parser.add_argument('action', choices=['build', 'landing', 'refresh', 'publish'])
    parser.add_argument('places', nargs='*', help=', '.join(by_slug))
    parser.add_argument('--all', action='store_true', help='every place in places.yaml')
    parser.add_argument('--no-thumbs', action='store_true', help='landing: keep photos')
    parser.add_argument('--go', action='store_true', help='publish: really upload')
    parser.add_argument('--gap', type=int, default=180, help='publish: seconds apart')
    args = parser.parse_args(argv)
    unknown = sorted(set(args.places) - set(by_slug))
    if unknown:
        raise SystemExit(f'Not in places.yaml: {", ".join(unknown)}.')
    chosen = config['places'] if args.all else [by_slug[s] for s in args.places]
    if args.action in ('build', 'publish') and not chosen:
        raise SystemExit('Name the places, or pass --all.')
    if args.action == 'build':
        build(chosen, config)
    elif args.action == 'landing':
        landing(config, thumbnails=not args.no_thumbs)
    elif args.action == 'refresh':
        refresh()
    else:
        publish(chosen, go=args.go, gap_s=args.gap)


if __name__ == '__main__':
    main(sys.argv[1:])
