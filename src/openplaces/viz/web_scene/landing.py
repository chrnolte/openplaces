"""The landing page of a set of web scenes: one card per place.

The page lists a fixed, editorial set of places, each a separate static
scene built by `build_web_scene` and published in its own folder beside
the page (`/values/<slug>/`). It is navigation between pre-built pages:
it has no search, no filter, no field to type in and no map to pick a
location on, and nothing is computed or looked up when a card is
followed (maintainer's decision, 2026-10-16; the reasoning is in
plans/static-3d-value-viewer-boston-pilot.md).
"""

import html
import json
import threading
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openplaces.viz.web_scene import DEFAULT_TITLE


@dataclass(frozen=True)
class Place:
    """One card of the landing page.

    Parameters
    ----------
    slug : str
        Folder of the place's scene beside the page, e.g. `boston`.
    title : str
        Name on the card.
    subtitle : str
        One short line under the name.
    scene_dir : str or Path
        Local folder of the built scene, for its snapshot.
    """

    slug: str
    title: str
    subtitle: str
    scene_dir: Path


def capture_thumbnail(scene_dir, target, width=960, height=600) -> Path:
    """Photograph a built scene from its first tour stop.

    Serves the scene from a local port, opens it in Edge or Chromium
    through Playwright (an optional dependency, needed only here),
    hides the page's panels and saves a JPEG.
    """
    from playwright.sync_api import sync_playwright

    scene_dir = Path(scene_dir)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(scene_dir))
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        scene = json.loads((scene_dir / 'scene.json').read_text(encoding='utf-8'))
        stop = scene['tour'][0] if scene.get('tour') else scene['view']
        view = {
            'longitude': stop['longitude'],
            'latitude': stop['latitude'],
            'zoom': 13.6,
            'pitch': 55,
            'bearing': 20,
        }
        url = f'http://127.0.0.1:{server.server_address[1]}/index.html?tour=0&all=1'
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel='msedge')
            except Exception:  # noqa: BLE001 - fall back to bundled Chromium
                browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': width, 'height': height})
            page.goto(url)
            page.wait_for_function(
                'window.openplacesScene && window.openplacesScene.loaded',
                timeout=600_000,
            )
            page.add_style_tag(content='.panel{display:none!important}')
            page.evaluate(
                '(v) => window.openplacesScene.deck.setProps('
                '{initialViewState: {...v, maxPitch: 85}})',
                view,
            )
            page.wait_for_timeout(3000)
            page.screenshot(path=str(target), type='jpeg', quality=82)
            browser.close()
    finally:
        server.shutdown()
    return target


def build_landing(places, output_dir, title=DEFAULT_TITLE, thumbnails=True) -> Path:
    """Write the landing page and its card images.

    Parameters
    ----------
    places : sequence of Place
        The cards, in order.
    output_dir : str or Path
        Folder of the page; each place's scene is published beside it
        under its slug.
    title : str
        Page heading.
    thumbnails : bool
        Photograph each scene (`capture_thumbnail`); False keeps the
        images already in `output_dir/thumbs`.

    Returns
    -------
    Path
        The output directory.
    """
    output_dir = Path(output_dir)
    (output_dir / 'thumbs').mkdir(parents=True, exist_ok=True)
    cards = []
    for place in places:
        image = output_dir / 'thumbs' / f'{place.slug}.jpg'
        if thumbnails:
            capture_thumbnail(place.scene_dir, image)
        cards.append(
            f'<a class="card" href="{html.escape(place.slug)}/">'
            f'<img src="thumbs/{html.escape(place.slug)}.jpg" alt="" loading="lazy">'
            f'<span class="name">{html.escape(place.title)}</span>'
            f'<span class="sub">{html.escape(place.subtitle)}</span></a>'
        )
    page = (Path(__file__).parent / 'static' / 'landing.html').read_text(
        encoding='utf-8'
    )
    page = page.replace('{{TITLE}}', html.escape(title)).replace(
        '{{CARDS}}', '\n'.join(cards)
    )
    (output_dir / 'index.html').write_text(page, encoding='utf-8')
    return output_dir
