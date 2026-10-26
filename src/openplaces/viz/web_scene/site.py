"""Build and publish the value site (placeslab.org/values).

The published counterpart of `local` (maintainer, 2026-10-18)::

    python -m openplaces.viz.web_scene.site build boston raleigh
    python -m openplaces.viz.web_scene.site build --all
    python -m openplaces.viz.web_scene.site landing [--no-thumbs]
    python -m openplaces.viz.web_scene.site refresh
    python -m openplaces.viz.web_scene.site publish --all --go
    python -m openplaces.viz.web_scene.site access --go

``build`` writes each place's page (15 km) and phone page (7 km) from
the delivered bundles into ``<share>/web/values``, or a single page of
whole units for a place listing `admin_ids` (an island). A place with
``team: true`` reads the team bundles (``<region>.team``): its page is
published only behind a password, never listed on the landing page,
and left out of ``--all``; ``landing`` writes
the landing page, its card photos, the list of places and the methods
page; ``refresh`` copies the page files and the known issues into every
built page without rebuilding data (for a change to the viewer alone).
``publish`` uploads, and only with ``--go`` and on the maintainer's
word: each phone page before the desktop page that sends phones to it,
one place every `--gap` seconds (Bluehost blocks bursts of SSH
connections), then the landing page, so no card links to a page that
is not up yet. Without ``--go`` it is a dry run. Every page goes up
behind the site's token gate (`web_scene.access_htaccess`); ``access``
uploads that gate alone to every folder online, after `tokens add` or
`tokens revoke`.

The places, their order and their centers (or units) are `places.yaml`.
"""

import argparse
import gzip
import json
import sys
import time
from importlib.resources import files
from pathlib import Path

from openplaces.config import cfg
from openplaces.viz.web_scene import (
    build_web_scene,
    deploy_access,
    deploy_landing,
    deploy_web_scene,
    known_issues,
)
from openplaces.viz.web_scene.local import (
    check_page_size,
    downtown_of,
    extent_label,
    has_circle,
    load_places,
    page_variants,
    place_elevation,
    place_extent,
    place_lots_file,
    place_models,
)

TITLE = 'Land & structure value'
STATIC_DIR = files('openplaces.viz.web_scene') / 'static'


def site_root() -> Path:
    """The published site's local folder."""
    return Path(cfg.data_root) / 'share' / 'web' / 'values'


def _page(place, radius_km, slug, phone_scene=None) -> None:
    circle, built, regions = place_extent(place, radius_km, slug)
    models = place_models(place, built, slug)
    for name, extra in page_variants(place, slug):
        started = time.time()
        build_web_scene(
            regions,
            built,
            site_root() / name,
            title=TITLE,
            crop=circle,
            elevation_recipe=place_elevation(place),
            phone_scene=None if name != slug else phone_scene,
            downtown=downtown_of(place),
            place_title=place['title'],
            model_files=models,
            lots_file=place_lots_file(place, built),
            models_public=bool(place.get('models_public')),
            verbose=False,
            **extra,
        )
        print(
            f'built {name} ({extent_label(place, radius_km)}, {len(built)} units) '
            f'in {time.time() - started:.0f}s',
            flush=True,
        )
        check_page_size(place, site_root() / name, name)


def build(places, config) -> None:
    """Each place's phone page, then its page, which sends phones there.

    A place shown as whole units has no phone page. A page that stops
    (above the size cap, say) does not stop the places after it; the
    stops are raised together at the end, and the folder of a page over
    the cap stays where it was written, so do not deploy after a stop.
    """
    stops = []
    for place in places:
        slug = place['slug']
        try:
            if not has_circle(place):
                _page(place, config['radius_km'], slug)
                continue
            _page(place, config['phone_radius_km'], f'{slug}-phone')
            _page(place, config['radius_km'], slug, phone_scene=f'../{slug}-phone/')
        except SystemExit as stop:
            stops.append(str(stop))
    if stops:
        raise SystemExit('\n'.join(stops))


def parcel_count(scene_dir) -> int:
    """Lots a built page draws: its bottom layer's `n_features`, or for a
    page built before the count was recorded, the rows of that layer's
    tooltip files (one per lot)."""
    scene_dir = Path(scene_dir)
    manifest = json.loads((scene_dir / 'scene.json').read_text(encoding='utf-8'))
    layers = manifest.get('layers') or [{}]
    if layers[0].get('n_features') is not None:
        return int(layers[0]['n_features'])
    count = 0
    for paths in (layers[0].get('files') or {}).values():
        columns = json.loads(
            gzip.decompress((scene_dir / paths['tooltip']).read_bytes())
        )
        count += len(next(iter(columns.values()), []))
    return count


def landing(config, thumbnails=True) -> None:
    """The landing page, card photos, places.json and the methods page."""
    from openplaces.viz.web_scene.landing import Place, build_landing

    root = site_root()
    cards = [
        Place(
            p['slug'],
            p['title'],
            f'Within {config["radius_km"]:g} km of {p["around"]}'
            if has_circle(p)
            else p['extent_note'],
            root / p['slug'],
        )
        for p in config['places']
        # A team page is never listed: it is reached by its address,
        # with the password; nor is an `unlisted` page (a greeting).
        if not (p.get('team') or p.get('unlisted'))
        and (root / p['slug'] / 'scene.json').exists()
    ]
    # Most parcels first (maintainer, 2026-10-20), as each page's build
    # counted them; the landing cards and every page's places menu
    # follow this order, so it holds as places are added or rebuilt.
    cards.sort(key=lambda c: -parcel_count(c.scene_dir))
    build_landing(cards, root, thumbnails=thumbnails)
    print('landing:', [c.slug for c in cards])


def refresh() -> None:
    """Page files, known issues and place titles into every built page,
    data unchanged."""
    titles = {p['slug']: p['title'] for p in load_places()['places']}
    for folder in sorted(
        p for p in site_root().iterdir() if (p / 'scene.json').exists()
    ):
        for item in Path(STATIC_DIR).iterdir():
            if item.name in ('landing.html', 'methods.html', 'access.html'):
                continue
            target = '.htaccess' if item.name == 'htaccess' else item.name
            (folder / target).write_bytes(item.read_bytes())
        manifest_path = folder / 'scene.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        units = sorted({u for layer in manifest['layers'] for u in layer['files']})
        manifest['known_issues'] = known_issues(units)
        # A phone page (`<slug>-phone`) shows its place's title.
        slug = folder.name.removesuffix('-phone')
        if slug in titles:
            manifest['place_title'] = titles[slug]
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        print('refreshed', folder.name)


def publish(places, go=False, gap_s=180) -> None:
    """Upload the pages, phone page first, then the landing page."""
    root = site_root()
    # The restricted twin before its page, so the switch never leads
    # to a missing page.
    slugs = [
        s
        for p in places
        for s in ((f'{p["slug"]}-phone',) if has_circle(p) else ())
        + tuple(name for name, _ in page_variants(p, p['slug']))
    ]
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
    parser.add_argument(
        'action', choices=['build', 'landing', 'refresh', 'publish', 'access']
    )
    parser.add_argument('places', nargs='*', help=', '.join(by_slug))
    parser.add_argument('--all', action='store_true', help='every place in places.yaml')
    parser.add_argument('--no-thumbs', action='store_true', help='landing: keep photos')
    parser.add_argument('--go', action='store_true', help='publish: really upload')
    parser.add_argument('--gap', type=int, default=180, help='publish: seconds apart')
    args = parser.parse_args(argv)
    unknown = sorted(set(args.places) - set(by_slug))
    if unknown:
        raise SystemExit(f'Not in places.yaml: {", ".join(unknown)}.')
    # `--all` is the listed public pages; a team or `unlisted` page is
    # built and published only when named.
    public = [p for p in config['places'] if not (p.get('team') or p.get('unlisted'))]
    chosen = public if args.all else [by_slug[s] for s in args.places]
    if args.action in ('build', 'publish') and not chosen:
        raise SystemExit('Name the places, or pass --all.')
    if args.action == 'build':
        build(chosen, config)
    elif args.action == 'landing':
        landing(config, thumbnails=not args.no_thumbs)
    elif args.action == 'refresh':
        refresh()
    elif args.action == 'access':
        deploy_access(site_root(), go=args.go)
    else:
        publish(chosen, go=args.go, gap_s=args.gap)


if __name__ == '__main__':
    main(sys.argv[1:])
