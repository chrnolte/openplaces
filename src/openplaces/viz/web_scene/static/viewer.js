// Viewer of an openplaces value scene, written by
// openplaces.viz.web_scene. It reads scene.json and the binary files it
// names, and draws them with deck.gl on a plain background. Every unit
// in the manifest loads at start, and hovering a feature shows the
// fields shipped with it.

'use strict';

const AREA_UNIT_LABELS = {ac: 'acre', ha: 'ha', sqft: 'sq ft', m2: 'm²'};
// Words on the layer buttons; a layer not listed shows its label.
const LAYER_SWITCH_LABELS = {parcels: 'Land', buildings: 'Buildings'};
// Soft white: present, but never competing with the values' colors.
const BOUNDARY_COLOR = [235, 240, 255, 90];

function setStatus(text) {
  document.getElementById('status').textContent = text;
}

function loadScript(url) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Could not load ${url}`));
    document.head.appendChild(script);
  });
}

async function fetchGzip(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status}`);
  }
  const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

const TYPED = {f4: Float32Array, u4: Uint32Array, u1: Uint8Array};
// Opacity of an outline drawn too tall for what is mapped there.
const GHOST_ALPHA = 56;

// The layout is described in the openplaces.viz.web_scene docstring.
function decodeScene(buffer) {
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
  if (magic !== 'OPVS') {
    throw new Error('Not an openplaces value-scene file.');
  }
  const headerLength = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength))
  );
  const start = 8 + headerLength;
  const arrays = {};
  for (const entry of header.arrays) {
    const Type = TYPED[entry.dtype];
    arrays[entry.name] = new Type(buffer, start + entry.offset, entry.count);
  }
  return {header, arrays};
}

// deck.gl's binary input: one typed array per attribute, one entry per
// vertex, and the triangles computed at build time. Nothing is created
// per polygon and nothing is triangulated here, which is what keeps a
// phone within its memory.
//
// Each *ring* is one deck.gl object. deck.gl ends an object's walls at
// its last vertex, so a ring per object closes every outer ring and
// every hole on itself; a part per object would draw a wall from the
// end of the outer ring to the start of its first hole. The top faces
// come from `triangles`, which index all vertices, so they do not
// depend on how objects are cut.
// The ground under a feature, in meters: its vertices' own heights
// on a draped layer (their mean is `base`), else what is left of
// `base` without the lift from the layer below.
function groundOf(arrays, feature) {
  if (arrays.z) {
    return arrays.base[feature];
  }
  return arrays.base[feature] - (arrays.stack ? arrays.stack[feature] : 0);
}

// Where a feature's extrusion starts: `base`, lowered by its lift when
// the layer below is switched off (or, in the assessment view, when the
// feature below it was estimated and so lies flat), and by its ground
// when the terrain is switched off.
function featureBase(arrays, feature, dropStack, flat, assessed = false) {
  const lowered = dropStack || (assessed && arrays.on_estimated && arrays.on_estimated[feature]);
  const lift = lowered && arrays.stack ? arrays.stack[feature] : 0;
  return arrays.base[feature] - lift - (flat ? groundOf(arrays, feature) : 0);
}

// How tall a feature stands: no height in the assessment view where
// its value was openplaces' own estimate (`estimated`).
function featureHeight(arrays, feature, assessed = false) {
  return assessed && arrays.estimated && arrays.estimated[feature] ? 0 : arrays.height[feature];
}

// Each ring's top edge as a closed path, for drawing lot lines: the
// ring's vertices lifted by their extrusion, and a little more so the
// line is not lost in its own face. Cached per geometry build.
function topOutlines(entry) {
  if (entry.topOutlines && entry.topOutlines.source === entry.data) {
    return entry.topOutlines.data;
  }
  const {getPolygon, getElevation} = entry.data.attributes;
  const base = getPolygon.value;
  const top = new Float32Array(base.length);
  for (let v = 0; v < base.length / 3; v++) {
    top[3 * v] = base[3 * v];
    top[3 * v + 1] = base[3 * v + 1];
    top[3 * v + 2] = base[3 * v + 2] + getElevation.value[v] + 0.3;
  }
  const data = {
    length: entry.data.length,
    startIndices: entry.data.startIndices,
    attributes: {getPath: {value: top, size: 3}},
  };
  entry.topOutlines = {source: entry.data, data};
  return data;
}

function buildBinary({arrays}, dropStack = false, flat = false, assessed = false) {
  const {xy, ring_offsets: rings, part_ring_offsets: partRings} = arrays;
  // A draped layer carries each vertex's own ground height in `z`;
  // a flat one stands every vertex at its feature's `base`.
  const {part_feature: partFeature, base, z: vertexZ, triangles} = arrays;
  // `stack`: the lift a feature gets from the layer below it; dropped
  // when that layer is switched off, so buildings rest on the ground.
  const nParts = partFeature.length;
  const nRings = rings.length - 1;
  const nVertices = xy.length / 2;
  const positions = new Float32Array(nVertices * 3);
  const elevations = new Float32Array(nVertices);
  const ringFeature = new Uint32Array(nRings);
  for (let p = 0; p < nParts; p++) {
    const feature = partFeature[p];
    for (let r = partRings[p]; r < partRings[p + 1]; r++) {
      ringFeature[r] = feature;
    }
    const start = vertexZ ? 0 : featureBase(arrays, feature, dropStack, flat, assessed);
    const tall = featureHeight(arrays, feature, assessed);
    for (let v = rings[partRings[p]]; v < rings[partRings[p + 1]]; v++) {
      positions[3 * v] = xy[2 * v];
      positions[3 * v + 1] = xy[2 * v + 1];
      positions[3 * v + 2] = vertexZ ? (flat ? 0 : vertexZ[v]) : start;
      elevations[v] = tall;
    }
  }
  return {
    // A picked object is a ring; this gives its feature.
    objectFeature: ringFeature,
    data: {
      length: nRings,
      startIndices: rings,
      attributes: {
        getPolygon: {value: positions, size: 3},
        indices: triangles,
        getElevation: {value: elevations, size: 1},
      },
    },
  };
}

// Palettes (scene.json `palettes`): per layer a 256-step lookup table
// over log1p(value) between the palette's own bounds. Colors are made
// here from each feature's value, so switching palettes recolors what
// is loaded without fetching anything.
function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function preparePalette(palette) {
  for (const spec of Object.values(palette.layers)) {
    spec.rgb = spec.lut.map(hexToRgb);
    // Edges colored by value (`line_lut`), on the fills' scale.
    if (spec.line_lut) {
      spec.lineSpec = {...spec, rgb: spec.line_lut.map(hexToRgb), alpha: spec.line_alpha};
    }
    // Non-residential features in the per-dwelling view.
    if (spec.other_lut) {
      spec.otherRgb = spec.other_lut.map(hexToRgb);
    }
    if (spec.other_line_lut) {
      spec.otherLineRgb = spec.other_line_lut.map(hexToRgb);
    }
  }
  return palette;
}

function featureColors(values, spec, noValue) {
  const out = new Uint8Array(values.length * 4);
  const lo = Math.log1p(spec.vmin);
  const span = Math.log1p(spec.vmax) - lo;
  const last = spec.rgb.length - 1;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!(v > 0)) {
      out.set(noValue, 4 * i);
      continue;
    }
    const t = Math.min(1, Math.max(0, (Math.log1p(v) - lo) / span));
    const [r, g, b] = spec.rgb[Math.round(t * last)];
    out[4 * i] = r;
    out[4 * i + 1] = g;
    out[4 * i + 2] = b;
    out[4 * i + 3] = spec.alpha;
  }
  return out;
}

function choosePalette(palettes, fallback) {
  const wanted = new URLSearchParams(location.search).get('palette');
  let stored = null;
  try {
    stored = localStorage.getItem('openplaces-palette');
  } catch (error) {
    stored = null;
  }
  return (
    palettes.find((p) => p.id === wanted) ||
    palettes.find((p) => p.id === stored) ||
    palettes.find((p) => p.id === fallback) ||
    palettes[0]
  );
}

// The center of a feature's outline, in longitude and latitude: the
// area-weighted centroid of its outer rings (shoelace formula), so the
// pinned tooltip sits on the same spot wherever the polygon is clicked.
// Falls back to the mean of the vertices for a degenerate outline.
function featureCenter(entry, feature) {
  const {xy, ring_offsets: rings, part_ring_offsets: partRings} = entry.decoded.arrays;
  const partFeature = entry.decoded.arrays.part_feature;
  const [ox, oy] = entry.origin;
  let area = 0;
  let cx = 0;
  let cy = 0;
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (let p = 0; p < partFeature.length; p++) {
    if (partFeature[p] !== feature) {
      continue;
    }
    // The part's outer ring only; holes barely move the center.
    const v0 = rings[partRings[p]];
    const v1 = rings[partRings[p] + 1];
    for (let v = v0; v < v1 - 1; v++) {
      const x0 = xy[2 * v];
      const y0 = xy[2 * v + 1];
      const x1 = xy[2 * v + 2];
      const y1 = xy[2 * v + 3];
      const cross = x0 * y1 - x1 * y0;
      area += cross;
      cx += (x0 + x1) * cross;
      cy += (y0 + y1) * cross;
      sx += x0;
      sy += y0;
      n += 1;
    }
  }
  if (Math.abs(area) > 1e-14) {
    return [ox + cx / (3 * area), oy + cy / (3 * area)];
  }
  return n ? [ox + sx / n, oy + sy / n] : [ox, oy];
}

// Load at most this many files at once: each one is decompressed and
// converted in memory before the GPU takes it, and a phone has little.
const CONCURRENT_FILES = 2;

async function inBatches(jobs, run) {
  const queue = [...jobs];
  const workers = Array.from({length: CONCURRENT_FILES}, async () => {
    while (queue.length) {
      await run(queue.shift());
    }
  });
  await Promise.all(workers);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);
}

function formatValue(value, kind, unit) {
  if (value === null || value === undefined) {
    return 'not recorded';
  }
  switch (kind) {
    case 'money':
      return `$${Math.round(value).toLocaleString('en-US')}`;
    case 'money_per_area': {
      const digits = value < 100 ? 2 : 0;
      const text = value.toLocaleString('en-US', {
        minimumFractionDigits: digits, maximumFractionDigits: digits,
      });
      return `$${text} per ${AREA_UNIT_LABELS[unit] || unit}`;
    }
    case 'year':
      return String(value);
    case 'count':
      return value.toLocaleString('en-US');
    default:
      return escapeHtml(value);
  }
}

function tooltipHtml(layerInfo, index) {
  const rows = layerInfo.spec.tooltip.map((tip) => {
    const value = layerInfo.values[tip.column][index];
    // A count without a value means "not applicable" (e.g. records on
    // a lot shared by none): the row is left out.
    if (tip.kind === 'count' && (value === null || value === undefined)) {
      return '';
    }
    let text = formatValue(value, tip.kind, layerInfo.spec.unit);
    const imputed = layerInfo.values[`${tip.column}__imputed`];
    if (tip.imputed && imputed && imputed[index] && value !== null) {
      text += ' <span class="estimated">(estimated)</span>';
    }
    return `<div><b>${escapeHtml(tip.label)}:</b> ${text}</div>`;
  });
  return `<div><i>${escapeHtml(layerInfo.spec.label)}</i></div>` +
    rows.join('');
}

// Tick values at 1 and 3 times each power of ten inside the bounds
// (the notebook's colorbar ticks), placed on the same log1p scale the
// colors use. Decades only when the range spans more than four.
function legendTicks(vmin, vmax) {
  const ticks = [];
  const lo = Math.floor(Math.log10(Math.max(vmin, 1)));
  const hi = Math.ceil(Math.log10(vmax));
  const multiples = hi - lo > 4 ? [1] : [1, 3];
  for (let e = lo; e <= hi; e++) {
    for (const m of multiples) {
      const v = m * 10 ** e;
      if (v >= vmin && v <= vmax) {
        ticks.push(v);
      }
    }
  }
  return ticks;
}

function compactMoney(v) {
  if (v >= 1e6) return `$${+(v / 1e6).toPrecision(2)}M`;
  if (v >= 1e3) return `$${+(v / 1e3).toPrecision(2)}K`;
  return `$${+v.toPrecision(2)}`;
}

function drawLegend(container, legend) {
  const block = document.createElement('div');
  block.className = legend.shown === false ? 'legend hidden' : 'legend';
  const label = document.createElement('div');
  label.className = 'legend-label';
  label.textContent =
    `${legend.label} per ${AREA_UNIT_LABELS[legend.unit] || legend.unit}`;
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 1;
  const context = canvas.getContext('2d');
  const gradient = context.createLinearGradient(0, 0, 256, 0);
  legend.colors.forEach((color, i) => {
    gradient.addColorStop(i / (legend.colors.length - 1), color);
  });
  context.fillStyle = gradient;
  context.fillRect(0, 0, 256, 1);
  const ticks = document.createElement('div');
  ticks.className = 'legend-ticks';
  const lo = Math.log1p(legend.vmin);
  const span = Math.log1p(legend.vmax) - lo;
  for (const value of legendTicks(legend.vmin, legend.vmax)) {
    const tick = document.createElement('span');
    const at = (100 * (Math.log1p(value) - lo)) / span;
    tick.style.left = `${at}%`;
    // Labels at the ends align inward, so they stay inside the panel.
    if (at > 92) {
      tick.style.transform = 'translateX(-100%)';
    } else if (at < 8) {
      tick.style.transform = 'none';
    }
    tick.textContent = compactMoney(value);
    ticks.appendChild(tick);
  }
  block.append(label, canvas, ticks);
  container.appendChild(block);
}

// The opening tour, a slow flyover: from a high, flat view of the
// whole area it descends onto the first focus (the densest value, the
// main downtown), circles it, glides to the next, circles that one,
// and so on, back to the first, until the visitor touches the map.
// The foci come from the manifest (`tour_foci`).
//
// Meant to be calm (maintainer, 2026-10-16: "almost meditative"):
// - the camera turns at one slow, constant rate for the whole tour,
//   flights included; flights no longer swing round to face their
//   destination, which was the sharpest change of the earlier tour;
// - moves ease along a sine, which starts and stops more softly than
//   a cubic;
// - every view value then passes through two smoothing stages, so no
//   speed changes abruptly, even where one phase hands to the next.
// Times in seconds, angles in degrees.
const TOUR = {
  farZoomOffset: -1.5,
  farPitch: 0,
  focusZoom: 14.8,
  focusPitch: 58,
  degreesPerSecond: 2.5,
  descent: 16,
  orbit: 18,
  // A flight takes longer and rises a little higher the farther it
  // goes.
  flightBase: 10,
  flightPerKm: 3,
  flightMax: 28,
  climbPerKm: 0.25,
  climbMax: 1.4,
  pitchDip: 8,
  // Time constant of each smoothing stage.
  smoothing: 1.2,
};

function ease(t) {
  return 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, t)));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function distanceKm(a, b) {
  const kx = 111.32 * Math.cos((a.latitude * Math.PI) / 180);
  return Math.hypot((b.longitude - a.longitude) * kx, (b.latitude - a.latitude) * 110.57);
}

// Each segment maps its own elapsed time to a position (center,
// zoom, pitch); the bearing is the tour's own, shared by all.
function buildTour(view, foci) {
  const far = {
    longitude: view.longitude,
    latitude: view.latitude,
    zoom: view.zoom + TOUR.farZoomOffset,
    pitch: TOUR.farPitch,
  };
  const descend = (to) => ({
    duration: TOUR.descent,
    view: (t) => {
      const s = ease(t / TOUR.descent);
      return {
        longitude: lerp(far.longitude, to.longitude, s),
        latitude: lerp(far.latitude, to.latitude, s),
        zoom: lerp(far.zoom, TOUR.focusZoom, s),
        pitch: lerp(far.pitch, TOUR.focusPitch, s),
      };
    },
  });
  const orbit = (at) => ({
    duration: TOUR.orbit,
    view: () => ({
      longitude: at.longitude,
      latitude: at.latitude,
      zoom: TOUR.focusZoom,
      pitch: TOUR.focusPitch,
    }),
  });
  const fly = (from, to) => {
    const km = distanceKm(from, to);
    const duration = Math.min(TOUR.flightMax, TOUR.flightBase + TOUR.flightPerKm * km);
    const climb = Math.min(TOUR.climbMax, TOUR.climbPerKm * km);
    return {
      duration,
      view: (t) => {
        const s = ease(t / duration);
        const arc = Math.sin(Math.PI * s);
        return {
          longitude: lerp(from.longitude, to.longitude, s),
          latitude: lerp(from.latitude, to.latitude, s),
          zoom: TOUR.focusZoom - climb * arc,
          pitch: TOUR.focusPitch - TOUR.pitchDip * arc,
        };
      },
    };
  };

  // Circle slowly at the high view until the first stop's buildings
  // are on screen (`ready`), so the descent never arrives at bare
  // ground; its duration is set when they are.
  const hold = {duration: Infinity, view: () => ({...far})};

  if (!foci || !foci.length) {
    // No foci: circle the center of the area.
    const center = {longitude: view.longitude, latitude: view.latitude};
    return {hold, intro: [hold, descend(center)], loop: [orbit(center)]};
  }
  const intro = [hold, descend(foci[0])];
  const loop = [];
  for (let i = 0; i < foci.length; i++) {
    loop.push(orbit(foci[i]));
    if (foci.length > 1) {
      loop.push(fly(foci[i], foci[(i + 1) % foci.length]));
    }
  }
  return {hold, intro, loop};
}

function startTour(deckgl, view, foci, element) {
  const {hold, intro, loop} = buildTour(view, foci);
  const segments = [...intro, ...loop];
  const keys = ['longitude', 'latitude', 'zoom', 'pitch'];
  let running = true;
  let index = 0;
  let start = null;
  let segmentStart = null;
  let last = null;
  // Two smoothing stages; both start on the first target, so the
  // tour opens without a jump.
  let stage1 = null;
  let stage2 = null;
  function frame(now) {
    if (!running) {
      return;
    }
    start = start ?? now;
    segmentStart = segmentStart ?? now;
    let t = (now - segmentStart) / 1000;
    while (t >= segments[index].duration) {
      t -= segments[index].duration;
      segmentStart += segments[index].duration * 1000;
      index += 1;
      if (index === segments.length) {
        // Replay the circuit, never the descent.
        index = intro.length;
      }
    }
    const target = segments[index].view(t);
    // The turn is continuous for the whole tour, so it needs no
    // smoothing: it never changes speed.
    const bearing = (TOUR.degreesPerSecond * (now - start)) / 1000;
    if (stage1 === null) {
      stage1 = {...target};
      stage2 = {...target};
    }
    // Frame time, capped so a stalled tab does not jump on return.
    const dt = Math.min(0.1, last === null ? 0 : (now - last) / 1000);
    last = now;
    const k = 1 - Math.exp(-dt / TOUR.smoothing);
    for (const key of keys) {
      stage1[key] += (target[key] - stage1[key]) * k;
      stage2[key] += (stage1[key] - stage2[key]) * k;
    }
    deckgl.setProps({initialViewState: {...stage2, bearing, maxPitch: 85}});
    requestAnimationFrame(frame);
  }
  const stop = () => {
    running = false;
  };
  for (const type of ['pointerdown', 'wheel', 'touchstart', 'keydown']) {
    (type === 'keydown' ? window : element).addEventListener(type, stop, {
      once: true,
      passive: true,
    });
  }
  requestAnimationFrame(frame);
  // Ends the hold where it stands, so the descent starts from there.
  const ready = () => {
    if (hold.duration === Infinity) {
      hold.duration =
        index === 0 && segmentStart !== null
          ? (performance.now() - segmentStart) / 1000
          : 0;
    }
  };
  return {stop, ready};
}

// A back link to the site's landing page and a small menu of the other
// places. Both are plain links to pages built ahead of time, exactly
// like the landing page's cards: nothing is searched, filtered or
// looked up (see the landing page's legal note in plans/). A link
// carries the current view along: with the tour stopped, the zoom,
// pitch and bearing (set at the next place's main center); and which
// layers are off. The palette carries over by itself (stored per site).
function buildPlaceNav(places, deckgl, isTouring, shown) {
  if (!places.length) {
    return;
  }
  const here = location.pathname.split('/').filter(Boolean).pop();
  const back = document.createElement('a');
  back.id = 'home-link';
  back.href = '../';
  back.textContent = '←';
  back.title = 'All places';
  back.setAttribute('aria-label', 'All places');
  document.getElementById('title-panel').prepend(back);

  const menu = document.createElement('details');
  menu.id = 'place-menu';
  const summary = document.createElement('summary');
  const current = places.find((p) => p.slug === here);
  summary.textContent = current ? current.title : 'Places';
  const list = document.createElement('div');
  list.className = 'place-list';
  for (const place of places) {
    if (place.slug === here) {
      continue;
    }
    const link = document.createElement('a');
    link.dataset.slug = place.slug;
    link.textContent = place.title;
    link.href = `../${place.slug}/`;
    list.appendChild(link);
  }
  menu.append(summary, list);

  const hrefFor = (slug) => {
    const query = new URLSearchParams();
    if (!isTouring()) {
      let viewport = null;
      try {
        viewport = deckgl.getViewports()[0];
      } catch (error) {
        viewport = null;
      }
      if (viewport) {
        query.set('tour', '0');
        query.set('zoom', viewport.zoom.toFixed(2));
        query.set('pitch', viewport.pitch.toFixed(1));
        query.set('bearing', viewport.bearing.toFixed(1));
      }
    }
    // Everything is on by default; what is off travels as `hide`.
    const hidden = Object.keys(shown).filter((name) => !shown[name]);
    if (hidden.length) {
      query.set('hide', hidden.join(','));
    }
    const text = query.toString();
    return `../${slug}/${text ? `?${text}` : ''}`;
  };
  const refresh = () => {
    for (const link of list.children) {
      link.href = hrefFor(link.dataset.slug);
    }
  };
  menu.addEventListener('toggle', refresh);
  list.addEventListener('pointerdown', refresh);
  // Top right, on its own, apart from the display controls (and still
  // there when they are folded away).
  const panel = document.getElementById('place-panel');
  panel.appendChild(menu);
  panel.hidden = false;
}

async function main() {
  const params = new URLSearchParams(location.search);
  const scene = await (await fetch('scene.json', {cache: 'no-cache'})).json();
  document.title = scene.title;
  document.getElementById('title').textContent = scene.title;
  // No subtitle on the scene: what the values are follows from the
  // controls (with estimates, or the assessment alone).
  // The other places of the site (`places.json`, written with the
  // landing page beside the scenes); absent for a lone scene.
  const places = await fetch('../places.json', {cache: 'no-cache'})
    .then((r) => (r.ok ? r.json() : []))
    .catch(() => []);
  const notices = document.getElementById('notices');
  for (const [recipe, text] of Object.entries(scene.notices)) {
    const pre = document.createElement('pre');
    pre.textContent = `${recipe}\n\n${text}`;
    notices.appendChild(pre);
  }
  const palettes = scene.palettes.map(preparePalette);
  let palette = choosePalette(palettes, scene.default_palette);
  const legendPanel = document.getElementById('legend-panel');
  // One button per layer, switching the whole layer on or off (display
  // only: it never selects features by their attributes).
  const layerSwitch = document.createElement('div');
  layerSwitch.id = 'layer-switch';
  const switcher = document.createElement('div');
  switcher.id = 'palette-switch';
  // The legend has a box of its own in the bottom bar, always shown;
  // the controls fold away.
  const legends = document.getElementById('legends');
  // Folding the legend changes the bar's height.
  document.getElementById('legend-box').addEventListener('toggle', () => placeControls());
  // The palette chips choose the legend's colors, so they sit with the
  // legend, above its bars (maintainer, 2026-10-16).
  legendPanel.append(layerSwitch);
  // In the legend's title line, beside "Legend" and its arrow.
  document.querySelector('#legend-box > summary').appendChild(switcher);
  // The controls fold away (a third of a phone's screen); the
  // "Controls" chip in the bottom bar opens and closes them.
  // Remembered per browser, like the palette.
  const minimize = document.createElement('button');
  minimize.type = 'button';
  minimize.id = 'legend-minimize';
  minimize.dataset.fold = 'minimize';
  minimize.textContent = '–';
  minimize.title = 'Minimize controls';
  minimize.setAttribute('aria-label', 'Minimize controls');
  legendPanel.prepend(minimize);
  const restore = document.createElement('button');
  restore.type = 'button';
  restore.id = 'controls-chip';
  restore.className = 'panel';
  restore.dataset.fold = 'restore';
  restore.textContent = 'Controls';
  restore.setAttribute('aria-label', 'Show controls');
  const bar = document.getElementById('bottom-bar');
  bar.insertBefore(restore, document.getElementById('about-panel'));
  // The controls open just above their chip. Where the legend shares
  // the chip's row (a wide screen) that is the chip's own top; where it
  // has a row of its own above (a phone), the top of the whole bar, so
  // the controls never cover the legend.
  const placeControls = () => {
    const chip = restore.getBoundingClientRect();
    const legend = document.getElementById('legend-box').getBoundingClientRect();
    // The legend is in the chip's row when it ends beside the chip.
    const sameRow = legend.bottom > chip.top;
    const top = sameRow ? chip.top : bar.getBoundingClientRect().top;
    legendPanel.style.bottom = `${Math.round(window.innerHeight - top + 8)}px`;
  };
  window.addEventListener('resize', placeControls);
  const setFolded = (folded) => {
    document.body.classList.toggle('controls-min', folded);
    restore.setAttribute('aria-expanded', String(!folded));
    restore.setAttribute('aria-label', folded ? 'Show controls' : 'Hide controls');
    placeControls();
    try {
      localStorage.setItem('openplaces-controls-min', folded ? '1' : '0');
    } catch (error) {
      // Private windows may refuse storage; the choice lasts this visit.
    }
  };
  minimize.addEventListener('click', () => setFolded(true));
  restore.addEventListener('click', () => {
    setFolded(!document.body.classList.contains('controls-min'));
  });
  // Folded unless this browser unfolded them before (maintainer,
  // 2026-10-16: minimized by default).
  let startFolded = true;
  try {
    startFolded = localStorage.getItem('openplaces-controls-min') !== '0';
  } catch (error) {
    startFolded = true;
  }
  setFolded(startFolded);
  // Which layers are switched on, from the address when another place
  // passed its settings on (`?hide=buildings`); `toggleLayer` is set
  // once deck.gl exists.
  const hiddenAtStart = (params.get('hide') || '').split(',');
  const shown = Object.fromEntries(
    scene.layers.map((l) => [l.name, !hiddenAtStart.includes(l.name)])
  );
  // Terrain is always on: no switch (maintainer, 2026-10-16); the
  // address can still lay a scene flat (`?hide=terrain`).
  const hasTerrain = Boolean(scene.terrain && scene.terrain.terrain_exaggeration);
  const switches = scene.layers.map((l) => [l.name, LAYER_SWITCH_LABELS[l.name] || l.label]);
  if (hasTerrain) {
    shown.terrain = !hiddenAtStart.includes('terrain');
  }
  // Outlines curate flags as drawn too tall for what is mapped there
  // (`ghost`): see-through at full height, behind a switch of their
  // own, "Ghosts*", on unless the address hides them, with its
  // footnote under the switches in the same panel (maintainer,
  // 2026-10-16).
  const hasGhosts = scene.layers.some((l) => l.n_ghosts > 0);
  if (hasGhosts) {
    shown.ghosts = !hiddenAtStart.includes('ghosts');
    switches.push(['ghosts', 'Ghosts*']);
  }
  // Named landmarks (`scene.landmarks`), a switch of their own.
  const landmarks = scene.landmarks || [];
  if (landmarks.length) {
    shown.landmarks = !hiddenAtStart.includes('landmarks');
    switches.push(['landmarks', 'Landmarks']);
  }
  let flatGround = hasTerrain && !shown.terrain;
  let toggleLayer = () => {};
  for (const [name, label] of switches) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip';
    button.dataset.layer = name;
    button.textContent = label;
    button.addEventListener('click', () => toggleLayer(name));
    layerSwitch.appendChild(button);
  }
  if (hasGhosts) {
    const note = document.createElement('p');
    note.className = 'footnote';
    note.textContent = '* Unexplained high building value per area';
    // In the controls panel, right under the switch it explains.
    layerSwitch.after(note);
  }
  const markLayerSwitch = () => {
    for (const button of layerSwitch.children) {
      const on = shown[button.dataset.layer];
      button.classList.toggle('active', on);
      button.setAttribute('aria-pressed', String(on));
    }
  };
  markLayerSwitch();

  // Two display choices, both recoloring every feature and hiding none:
  // the value per area or per dwelling (where a layer offers it), and
  // the values as the assessor recorded them or with openplaces' own
  // fills (`estimated`), which "Assessment" draws as no value.
  // Per footprint area, imputed, by default (maintainer, 2026-10-17);
  // per dwelling where the address asks (`?view=dwelling`).
  let valueView = params.get('view') === 'dwelling' ? 'dwelling' : 'area';
  let withEstimates = params.get('values') !== 'assessed';
  const hasDwellingView = scene.layers.some((l) => l.views && l.views.dwelling);
  // Two rows: per area or per dwelling, then with estimates or the
  // assessment alone.
  const viewSwitch = document.createElement('div');
  viewSwitch.id = 'view-switch';
  const valuesSwitch = document.createElement('div');
  valuesSwitch.id = 'values-switch';
  const choices = [];
  if (hasDwellingView) {
    // "Value:" per footprint area first, then per dwelling.
    const label = document.createElement('span');
    label.className = 'switch-label';
    label.textContent = 'Value:';
    viewSwitch.appendChild(label);
    choices.push(
      ['view', 'area', 'Per footprint area'],
      ['view', 'dwelling', 'Per dwelling'],
    );
  }
  if (scene.layers.some((l) => l.n_estimated > 0)) {
    // "Imputed": the assessment with openplaces' fills (maintainer,
    // 2026-10-16); a third, "Our estimates", is planned.
    choices.push(['values', 'estimates', 'Imputed'], ['values', 'assessed', 'Assessment']);
  }
  let setChoice = () => {};
  for (const [group, id, label] of choices) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.group = group;
    button.dataset.choice = id;
    button.textContent = label;
    button.title = id === 'assessed'
      ? 'Values the assessor recorded; values openplaces filled in are drawn as no value'
      : '';
    button.addEventListener('click', () => setChoice(group, id));
    (group === 'view' ? viewSwitch : valuesSwitch).appendChild(button);
  }
  layerSwitch.after(viewSwitch, valuesSwitch);
  const markViewSwitch = () => {
    const buttons = [...viewSwitch.children, ...valuesSwitch.children];
    for (const button of buttons.filter((b) => b.dataset.group)) {
      const on = button.dataset.group === 'view'
        ? button.dataset.choice === valueView
        : (button.dataset.choice === 'estimates') === withEstimates;
      button.classList.toggle('active', on);
      button.setAttribute('aria-pressed', String(on));
    }
  };
  markViewSwitch();
  // A layer's coloring under the current choices: its scale (bounds,
  // unit) and the value of each feature.
  const layerScale = (layer, spec) => {
    const view = valueView === 'dwelling' && layer.views && layer.views.dwelling;
    return view
      ? {...spec, vmin: view.vmin, vmax: view.vmax, unit: 'dwelling'}
      : {...spec, unit: layer.unit};
  };
  // The non-residential part of a layer's per-dwelling view, if shown.
  const otherView = (entry) => {
    const view = valueView === 'dwelling' && entry.spec.views && entry.spec.views.dwelling;
    return view && view.other && entry.decoded.arrays.nonresidential ? view.other : null;
  };
  const featureValues = (entry, perArea = false) => {
    const {arrays} = entry.decoded;
    const view = valueView === 'dwelling' && entry.spec.views && entry.spec.views.dwelling;
    let values = view && arrays.value_dwelling && !perArea
      ? arrays.value_dwelling
      : entry.featureValue;
    if (!withEstimates && arrays.estimated) {
      values = values.map((v, i) => (arrays.estimated[i] ? 0 : v));
    }
    // Per area, the assessment view shows the value as recorded where
    // the shown one carries openplaces' land split (`value_assessed`).
    if (!withEstimates && arrays.value_assessed && (perArea || !view)) {
      values = values.map((v, i) => {
        const a = arrays.value_assessed[i];
        return Number.isFinite(a) && a > 0 ? a : v;
      });
    }
    return values;
  };

  const drawLegends = () => {
    legends.replaceChildren();
    for (const layer of scene.layers) {
      const spec = layerScale(layer, palette.layers[layer.name]);
      drawLegend(legends, {
        shown: shown[layer.name],
        label: withEstimates ? layer.label : `${layer.label} (assessed)`,
        unit: spec.unit,
        vmin: spec.vmin,
        vmax: spec.vmax,
        // 32 stops are plenty for a gradient bar.
        // Where the edges carry the value, the legend shows theirs.
        colors: (spec.line_lut || spec.lut).filter(
          (_, i) => i % 8 === 0 || i === spec.lut.length - 1,
        ),
      });
      // Per dwelling: a second bar for non-residential buildings.
      const view = valueView === 'dwelling' && layer.views && layer.views.dwelling;
      const ramp = spec.other_line_lut || spec.other_lut;
      if (view && view.other && ramp) {
        drawLegend(legends, {
          shown: shown[layer.name],
          label: withEstimates ? view.other.label : `${view.other.label} (assessed)`,
          unit: view.other.unit,
          vmin: view.other.vmin,
          vmax: view.other.vmax,
          colors: ramp.filter((_, i) => i % 8 === 0 || i === ramp.length - 1),
        });
      }
    }
    // The color of features without a value, named (on Pastel it is
    // a near-white that read as a value; maintainer, 2026-10-16).
    const none = document.createElement('div');
    none.className = 'legend-none';
    const swatch = document.createElement('span');
    const [r, g, b] = palette.no_value;
    swatch.style.background = `rgb(${r}, ${g}, ${b})`;
    none.append(swatch, document.createTextNode('No value recorded'));
    legends.appendChild(none);
    // A third bar changes the legend's height.
    placeControls();
  };
  document.body.style.background = palette.background;
  document.body.classList.toggle('light-page', palette.theme === 'light');
  drawLegends();

  await loadScript(scene.deck_gl_url);
  const {Deck, SolidPolygonLayer, PathLayer, COORDINATE_SYSTEM} = deck;
  const {LightingEffect, AmbientLight, DirectionalLight} = deck;
  const {ScatterplotLayer, TextLayer, DataFilterExtension} = deck;

  // The tour runs while the files load, so the layers rise into a
  // moving view; ?tour=0 in the address, or a system setting for
  // reduced motion, opens a still view instead.
  const tour =
    params.get('tour') !== '0' &&
    !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // A view passed on from another place (zoom, pitch, bearing), set at
  // this place's first tour stop, its main center; a shared link also
  // carries the center (`lon`, `lat`, the share button).
  const passed = ['zoom', 'pitch', 'bearing'].every((k) => params.has(k));
  const shared = params.has('lon') && params.has('lat');
  const center = shared
    ? {longitude: Number(params.get('lon')), latitude: Number(params.get('lat'))}
    : scene.tour && scene.tour.length ? scene.tour[0] : scene.view;
  const startView = passed
    ? {
      longitude: center.longitude,
      latitude: center.latitude,
      zoom: Number(params.get('zoom')),
      pitch: Number(params.get('pitch')),
      bearing: Number(params.get('bearing')),
    }
    : scene.view;

  const layerInfo = new Map();
  // The pinned tooltip's feature (see `togglePin` below).
  let pin = null;
  let togglePin = () => {};
  let placePin = () => {};
  let turnCompass = () => {};
  // One sun fixed in the world, from the southwest, plus even ambient
  // light: every building has one bright side (facing the sun), the
  // opposite side darkest, and the two between at mid shade. Two lights
  // from opposite directions (deck.gl's default, and a sun with a
  // northeast fill tried here) leave the two walls at right angles to
  // them both dark, so a building showed dark faces on opposite sides
  // (maintainer, 2026-10-16, Sudbury St, Boston). Directions are those
  // the light travels in (x east, y north, z up).
  const lighting = new LightingEffect({
    ambient: new AmbientLight({color: [255, 255, 255], intensity: 1.05}),
    sun: new DirectionalLight({color: [255, 250, 240], intensity: 1.25, direction: [1, 1.2, -1.2]}),
  });
  const deckgl = new Deck({
    parent: document.getElementById('scene'),
    effects: [lighting],
    initialViewState: tour
      ? {
        ...scene.view,
        zoom: scene.view.zoom + TOUR.farZoomOffset,
        pitch: TOUR.farPitch,
        maxPitch: 85,
      }
      : {...startView, maxPitch: 85},
    // On touch screens two fingers rotate (twist) and tilt (drag up
    // or down together) as well as zoom (pinch); deck.gl leaves the
    // twist off by default.
    controller: {touchRotate: true, dragRotate: true},
    layers: [],
    // Binary layers report the picked part's index, not an object.
    getTooltip: ({index, layer, object}) => {
      // A landmark's own name and kind, written into the page.
      if (layer && layer.id.startsWith('landmark') && object) {
        return {html: `<b>${escapeHtml(object.name)}</b><br>${escapeHtml(object.kind)}`};
      }
      if (index < 0 || !layer || !layerInfo.has(layer.id)) {
        return null;
      }
      const info = layerInfo.get(layer.id);
      const feature = info.objectFeature[index];
      // The pinned feature already shows its values.
      if (pin && pin.layerId === layer.id && pin.feature === feature) {
        return null;
      }
      return {html: tooltipHtml(info, feature)};
    },
    // Deferred: both are set once the layers they read exist.
    onClick: (picked) => togglePin(picked),
    onAfterRender: () => {
      placePin();
      turnCompass();
    },
  });

  // A clicked feature's tooltip stays, attached to the point clicked,
  // until the same feature is clicked again or the tooltip is closed.
  // It shows only the fields shipped with that feature, as the hover
  // tooltip does; nothing is looked up.
  const pinBox = document.createElement('div');
  pinBox.className = 'pinned-tip deck-tooltip';
  pinBox.hidden = true;
  document.body.appendChild(pinBox);
  // The line from the tooltip down to the point it belongs to.
  const pinLeader = document.createElement('div');
  pinLeader.className = 'pin-leader';
  pinLeader.hidden = true;
  document.body.appendChild(pinLeader);
  const unpin = () => {
    pin = null;
    pinBox.hidden = true;
    pinLeader.hidden = true;
  };
  togglePin = ({index, layer}) => {
    if (index < 0 || !layer || !layerInfo.has(layer.id)) {
      return;
    }
    const info = layerInfo.get(layer.id);
    const feature = info.objectFeature[index];
    if (pin && pin.layerId === layer.id && pin.feature === feature) {
      unpin();
      return;
    }
    // Anchor at the top of the feature, so the tooltip sits on it in 3D.
    // A ghost is picked in its own layer (`<id>-ghosts`).
    const entry = layers.find((e) => e.id === layer.id || `${e.id}-ghosts` === layer.id);
    const arrays = entry ? entry.decoded.arrays : null;
    const top = arrays
      ? featureBase(arrays, feature, !shown[bottom], flatGround, !withEstimates) +
        featureHeight(arrays, feature, !withEstimates)
      : 0;
    const [longitude, latitude] = featureCenter(entry, feature);
    pin = {layerId: layer.id, feature, position: [longitude, latitude, top]};
    pinBox.innerHTML = tooltipHtml(info, feature);
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'pin-close';
    close.dataset.close = 'tooltip';
    close.setAttribute('aria-label', 'Close');
    close.textContent = '×';
    close.addEventListener('click', unpin);
    pinBox.prepend(close);
    pinBox.hidden = false;
    pinLeader.hidden = false;
    placePin();
  };
  placePin = () => {
    if (!pin) {
      return;
    }
    let viewport = null;
    try {
      viewport = deckgl.getViewports()[0];
    } catch (error) {
      viewport = null;
    }
    if (!viewport) {
      return;
    }
    const [x, y] = viewport.project(pin.position);
    const visible = Number.isFinite(x) && Number.isFinite(y);
    const at = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    for (const element of [pinBox, pinLeader]) {
      element.style.visibility = visible ? 'visible' : 'hidden';
      element.style.transform = at;
    }
  };

  // A device that runs out of graphics memory drops the drawing
  // context: say so, rather than leave a black screen.
  const canvas = deckgl.getCanvas && deckgl.getCanvas();
  if (canvas) {
    canvas.addEventListener('webglcontextlost', () => {
      setStatus('This device ran out of graphics memory. Reload to try again.');
    });
  }
  // Right-drag rotates the view; without this the browser's menu opens
  // when the button is let go. Only on the scene: panels keep theirs.
  document.getElementById('scene').addEventListener('contextmenu', (event) => {
    event.preventDefault();
  });

  const tourControl = tour
    ? startTour(deckgl, scene.view, scene.tour, document.getElementById('scene'))
    : {stop: () => {}, ready: () => {}};

  // For measuring the page (frame rate, load time) from a test browser.
  window.openplacesScene = {deck: deckgl, loaded: false, stopTour: tourControl.stop};

  // Whether the tour still runs; it stops on the visitor's first touch.
  let touring = tour;
  for (const type of ['pointerdown', 'wheel', 'touchstart', 'keydown']) {
    (type === 'keydown' ? window : document.getElementById('scene')).addEventListener(
      type,
      () => {
        touring = false;
      },
      {once: true, passive: true},
    );
  }
  buildPlaceNav(places, deckgl, () => touring, shown);

  // Tilt with two fingers moving up or down together (maintainer,
  // 2026-10-17). deck.gl has the gesture, but its pinch claims nearly
  // every two-finger drag, so the tilt rarely starts on a phone. Here
  // a two-finger drag is taken as a tilt once both fingers have moved
  // the same way vertically while their spread stayed about the same;
  // from then on its moves go to the tilt and not to deck.gl. Any
  // other two-finger gesture (pinch, twist) is left to deck.gl.
  const touches = new Map();
  let tilt = null;
  const sceneElement = document.getElementById('scene');
  const spread = () => {
    const [a, b] = [...touches.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  sceneElement.addEventListener('pointerdown', (event) => {
    if (event.pointerType !== 'touch') {
      return;
    }
    touches.set(event.pointerId, {x: event.clientX, y: event.clientY, x0: event.clientX, y0: event.clientY});
    tilt = null;
    if (touches.size === 2) {
      tilt = {decided: false, spread0: spread(), pitch0: deckgl.getViewports()[0].pitch};
      for (const t of touches.values()) {
        t.x0 = t.x;
        t.y0 = t.y;
      }
    }
  }, true);
  sceneElement.addEventListener('pointermove', (event) => {
    const t = touches.get(event.pointerId);
    if (!t) {
      return;
    }
    t.x = event.clientX;
    t.y = event.clientY;
    if (!tilt || touches.size !== 2) {
      return;
    }
    const [a, b] = [...touches.values()];
    const dyA = a.y - a.y0;
    const dyB = b.y - b.y0;
    if (!tilt.decided) {
      const change = Math.abs(spread() - tilt.spread0);
      const vertical = Math.abs(dyA) > 10 && Math.abs(dyB) > 10 && Math.sign(dyA) === Math.sign(dyB);
      if (vertical && change < 0.5 * Math.min(Math.abs(dyA), Math.abs(dyB))) {
        tilt.decided = 'tilt';
        if (touring) {
          tourControl.stop();
          touring = false;
        }
      } else if (change > 20) {
        tilt.decided = 'other';
      }
    }
    if (tilt.decided === 'tilt') {
      event.stopPropagation();
      event.preventDefault();
      const v = deckgl.getViewports()[0];
      const pitch = Math.max(0, Math.min(85, tilt.pitch0 - ((dyA + dyB) / 2) * 0.35));
      deckgl.setProps({
        initialViewState: {
          longitude: v.longitude,
          latitude: v.latitude,
          zoom: v.zoom,
          bearing: v.bearing,
          pitch,
          maxPitch: 85,
        },
      });
    }
  }, {capture: true, passive: false});
  const release = (event) => {
    touches.delete(event.pointerId);
    if (touches.size < 2) {
      tilt = null;
    }
  };
  sceneElement.addEventListener('pointerup', release, true);
  sceneElement.addEventListener('pointercancel', release, true);

  // Share: a link to this page with the camera and the display choices
  // in its address. It names only a viewpoint and settings; no feature,
  // key or record (design limit L3).
  const shareButton = document.createElement('button');
  shareButton.type = 'button';
  shareButton.id = 'share-chip';
  shareButton.className = 'panel';
  shareButton.dataset.share = 'link';
  shareButton.textContent = 'Share';
  shareButton.setAttribute('aria-label', 'Share a link to this view');
  bar.insertBefore(shareButton, document.getElementById('about-panel'));

  // Compass: its needle points north as the view turns; a click turns
  // the view back to north up (the tilt stays).
  const compass = document.createElement('button');
  compass.type = 'button';
  compass.id = 'compass';
  compass.className = 'panel';
  compass.dataset.compass = 'north';
  compass.title = 'Turn north up';
  compass.setAttribute('aria-label', 'Turn north up');
  compass.innerHTML = '<span class="needle">▲</span><span class="n">N</span>';
  document.getElementById('place-panel').after(compass);
  const needle = compass.querySelector('.needle');
  let lastBearing = null;
  turnCompass = () => {
    let viewport = null;
    try {
      viewport = deckgl.getViewports()[0];
    } catch (error) {
      return;
    }
    if (!viewport || viewport.bearing === lastBearing) {
      return;
    }
    lastBearing = viewport.bearing;
    needle.style.transform = `rotate(${-viewport.bearing}deg)`;
  };
  compass.addEventListener('click', () => {
    if (touring) {
      tourControl.stop();
      touring = false;
    }
    const v = deckgl.getViewports()[0];
    deckgl.setProps({
      initialViewState: {
        longitude: v.longitude,
        latitude: v.latitude,
        zoom: v.zoom,
        pitch: v.pitch,
        bearing: 0,
        maxPitch: 85,
        transitionDuration: 600,
        transitionInterpolator: new deck.LinearInterpolator(['bearing']),
      },
    });
  });
  const shareLink = () => {
    const v = deckgl.getViewports()[0];
    const query = new URLSearchParams({
      tour: '0',
      lon: v.longitude.toFixed(5),
      lat: v.latitude.toFixed(5),
      zoom: v.zoom.toFixed(2),
      pitch: v.pitch.toFixed(1),
      bearing: v.bearing.toFixed(1),
      palette: palette.id,
      view: valueView,
      values: withEstimates ? 'imputed' : 'assessed',
    });
    const hidden = Object.keys(shown).filter((name) => !shown[name]);
    if (hidden.length) {
      query.set('hide', hidden.join(','));
    }
    return `${location.origin}${location.pathname}?${query}`;
  };
  shareButton.addEventListener('click', async () => {
    const url = shareLink();
    try {
      if (navigator.share && matchMedia('(pointer: coarse)').matches) {
        await navigator.share({title: document.title, url});
        return;
      }
      await navigator.clipboard.writeText(url);
      shareButton.textContent = 'Link copied';
    } catch (error) {
      // Clipboard refused (an insecure page, a denied permission):
      // show the link so it can be copied by hand.
      window.prompt('Copy this link:', url);
      return;
    }
    setTimeout(() => {
      shareButton.textContent = 'Share';
    }, 2000);
  });

  // Borders lie on the ground (their `z`, the height of the nearest
  // parcel vertex, `border_paths`), a meter up so they do not sink into
  // it, and are hidden by whatever stands in front of them. With the
  // terrain off they lie at sea level with everything else.
  const borderPath = (d) => d.path.map(([x, y, z]) => [x, y, (flatGround ? 0 : z || 0) + 1]);
  const makeBoundaries = () => new PathLayer({
    id: 'boundaries',
    data: scene.boundaries,
    getPath: borderPath,
    getColor: palette.boundary || BOUNDARY_COLOR,
    getWidth: 1.5,
    widthUnits: 'pixels',
    updateTriggers: {getColor: palette.id, getPath: flatGround},
  });
  let boundaries = makeBoundaries();

  // One loaded file: its deck.gl layer is rebuilt (same id, so deck.gl
  // only recomputes the colors) whenever the palette changes.
  const makeLayer = (entry) => {
    const spec = layerScale(entry.spec, palette.layers[entry.spec.name]);
    const values = featureValues(entry);
    const colors = featureColors(values, spec, palette.no_value);
    let lineColors = spec.lineSpec
      ? featureColors(
        values,
        {...spec.lineSpec, vmin: spec.vmin, vmax: spec.vmax},
        [0, 0, 0, 0],
      )
      : null;
    // Per dwelling, non-residential buildings keep their value per
    // area, on the palette's second ramp and their own bounds.
    const other = otherView(entry);
    if (other && spec.otherRgb) {
      const otherValues = featureValues(entry, true);
      const flags = entry.decoded.arrays.nonresidential;
      const scale = {vmin: other.vmin, vmax: other.vmax, alpha: spec.alpha};
      const fill = featureColors(otherValues, {...scale, rgb: spec.otherRgb}, palette.no_value);
      const edge = spec.otherLineRgb && lineColors
        ? featureColors(
          otherValues,
          {...scale, rgb: spec.otherLineRgb, alpha: spec.line_alpha},
          [0, 0, 0, 0],
        )
        : null;
      for (let i = 0; i < flags.length; i++) {
        if (flags[i]) {
          colors.set(fill.subarray(4 * i, 4 * i + 4), 4 * i);
          if (edge) {
            lineColors.set(edge.subarray(4 * i, 4 * i + 4), 4 * i);
          }
        }
      }
    }
    // A palette may draw every edge (`line`) and blend its fills
    // additively (`blend`): light adds up, no depth is written, so the
    // result does not depend on draw order (see `Palette.blend`).
    const additive = palette.blend === 'additive';
    // Ghosts (`ghost`): left out of the solid layer and drawn in a
    // layer of their own, their own color, mostly transparent, writing
    // no depth, so whatever stands behind them still shows. Switched
    // off, that layer is hidden. Both layers select by the same
    // per-feature flag (deck.gl's data filter, which discards the
    // left-out features' fragments; a zero alpha would still write
    // depth and hide what is behind).
    const ghosts = entry.decoded.arrays.ghost;
    const ghostFilter = (keep) => (ghosts
      ? {
        extensions: [new DataFilterExtension({filterSize: 1})],
        getFilterValue: (_, {index}) => ghosts[entry.objectFeature[index]],
        filterRange: keep === 'ghosts' ? [1, 1] : [0, 0],
      }
      : {});
    const shared = {
      data: entry.data,
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT_OFFSETS,
      coordinateOrigin: entry.origin,
      positionFormat: 'XYZ',
      _normalize: false,
      _windingOrder: 'CCW',
      extruded: true,
      visible: shown[entry.spec.name],
    };
    // Edges as a layer of their own, drawn after the faces, with the
    // faces pushed back a little in depth: drawn as the faces' own
    // wireframe they shared the faces' depth, and near buildings
    // showed broken, dotted outlines where the faces won the depth test
    // (maintainer, 2026-10-16). WebGL's polygon offset moves filled
    // triangles only, never lines, so it is the faces that move.
    // A single line color (Pastel's lot lines): one 1-pixel line along
    // each top edge, the same width everywhere. The wireframe also
    // draws bottom outlines and a strut at every corner, which ran
    // together into thick, dotted lines where lots stand tall
    // (maintainer, 2026-10-16). Value-colored edges (Neon) keep the
    // wireframe: there the struts are the look.
    // Lots (the bottom layer) take top-edge lines whether the line is
    // one color (Pastel) or follows each lot's value (Earth, Neon).
    const lotLines = entry.spec.name === bottom && lineColors;
    const topEdges = (spec.line && !lineColors) || lotLines
      ? new PathLayer({
        id: `${entry.id}-edges`,
        data: topOutlines(entry),
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT_OFFSETS,
        coordinateOrigin: entry.origin,
        positionFormat: 'XYZ',
        _pathType: 'loop',
        getColor: lotLines
          ? (_, {index}) => {
            const f = entry.objectFeature[index];
            return lineColors.subarray(4 * f, 4 * f + 4);
          }
          : spec.line,
        getWidth: 1,
        widthUnits: 'pixels',
        visible: shown[entry.spec.name],
        pickable: false,
        updateTriggers: {getColor: [palette.id, valueView, withEstimates]},
      })
      : null;
    const outline = topEdges || (lineColors
      ? new SolidPolygonLayer({
        ...shared,
        id: `${entry.id}-edges`,
        filled: false,
        wireframe: true,
        pickable: false,
        getLineColor: lineColors
          ? (_, {index}) => {
            const f = entry.objectFeature[index];
            return lineColors.subarray(4 * f, 4 * f + 4);
          }
          : spec.line,
        updateTriggers: {getLineColor: [palette.id, valueView, withEstimates]},
      })
      : null);
    const faces = new SolidPolygonLayer({
      ...(palette.material ? {material: palette.material} : {}),
      ...(outline
        ? {getPolygonOffset: ({layerIndex}) => [1, 4 - layerIndex * 100]}
        : {}),
      wireframe: false,
      parameters: additive
        ? {
          blend: true,
          blendColorOperation: 'add',
          blendColorSrcFactor: 'src-alpha',
          blendColorDstFactor: 'one',
          blendAlphaOperation: 'add',
          blendAlphaSrcFactor: 'one',
          blendAlphaDstFactor: 'one-minus-src-alpha',
          depthWriteEnabled: false,
        }
        : {},
      id: entry.id,
      data: entry.data,
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT_OFFSETS,
      coordinateOrigin: entry.origin,
      positionFormat: 'XYZ',
      _normalize: false,
      _windingOrder: 'CCW',
      extruded: true,
      pickable: true,
      visible: shown[entry.spec.name],
      getFillColor: (_, {index}) => {
        const f = entry.objectFeature[index];
        return colors.subarray(4 * f, 4 * f + 4);
      },
      updateTriggers: {
        getFillColor: [palette.id, valueView, withEstimates],
      },
      ...ghostFilter('solid'),
    });
    const out = outline ? [faces, outline] : [faces];
    if (ghosts) {
      out.push(new SolidPolygonLayer({
        ...shared,
        id: `${entry.id}-ghosts`,
        visible: shown[entry.spec.name] && shown.ghosts !== false,
        pickable: true,
        wireframe: true,
        ...(palette.material ? {material: palette.material} : {}),
        getFillColor: (_, {index}) => {
          const f = entry.objectFeature[index];
          const c = colors.subarray(4 * f, 4 * f + 4);
          return [c[0], c[1], c[2], GHOST_ALPHA];
        },
        getLineColor: (_, {index}) => {
          const f = entry.objectFeature[index];
          const c = (lineColors || colors).subarray(4 * f, 4 * f + 4);
          return [c[0], c[1], c[2], 140];
        },
        parameters: {depthWriteEnabled: false},
        updateTriggers: {
          getFillColor: [palette.id, valueView, withEstimates],
          getLineColor: [palette.id, valueView, withEstimates],
        },
        ...ghostFilter('ghosts'),
      }));
    }
    return out;
  };
  // Landmarks: a dot on the roof (or ground) of the place, its name
  // above it, drawn over everything else so a tower never hides its
  // own label. With the terrain off they drop by their ground height.
  const landmarkData = () => landmarks.map((mark) => {
    const [x, y, z] = mark.position;
    return {...mark, at: [x, y, z - (flatGround ? mark.ground || 0 : 0) + 2]};
  });
  const makeLandmarks = () => {
    if (!landmarks.length) {
      return [];
    }
    const data = landmarkData();
    const light = palette.theme === 'light';
    return [
      new ScatterplotLayer({
        id: 'landmark-dots',
        data,
        getPosition: (d) => d.at,
        getRadius: 25,
        radiusUnits: 'meters',
        radiusMinPixels: 3,
        radiusMaxPixels: 6,
        getFillColor: light ? [30, 34, 48, 230] : [255, 255, 255, 235],
        stroked: true,
        getLineColor: light ? [255, 255, 255, 230] : [10, 14, 24, 200],
        lineWidthMinPixels: 2,
        pickable: true,
        parameters: {depthTest: false},
        updateTriggers: {getPosition: flatGround, getFillColor: palette.id},
      }),
      new TextLayer({
        id: 'landmark-labels',
        data,
        getPosition: (d) => d.at,
        getText: (d) => d.name,
        // Sized on the ground, so labels shrink with distance: about
        // 9 px across a whole town, 15 px down among the buildings.
        getSize: 70,
        sizeUnits: 'meters',
        sizeMinPixels: 9,
        sizeMaxPixels: 15,
        getPixelOffset: [0, -13],
        getColor: light ? [20, 24, 36, 255] : [255, 255, 255, 255],
        background: true,
        getBackgroundColor: light ? [255, 255, 255, 215] : [8, 11, 20, 200],
        backgroundPadding: [6, 3],
        fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
        fontWeight: 600,
        pickable: true,
        parameters: {depthTest: false},
        updateTriggers: {getPosition: flatGround, getColor: palette.id},
      }),
    ];
  };
  const layers = [];
  const redraw = () => {
    deckgl.setProps({
      layers: [
        // Each entry is its faces and, under some palettes, its edges.
        ...layers.flatMap((l) => l.layer),
        boundaries,
        ...(shown.landmarks ? makeLandmarks() : []),
      ],
    });
  };

  const setPalette = (next) => {
    palette = next;
    try {
      localStorage.setItem('openplaces-palette', palette.id);
    } catch (error) {
      // Private windows may refuse storage; the choice then lasts
      // for this visit only.
    }
    document.body.style.background = palette.background;
    document.body.classList.toggle('light-page', palette.theme === 'light');
    for (const button of switcher.children) {
      button.classList.toggle('active', button.dataset.palette === palette.id);
    }
    drawLegends();
    boundaries = makeBoundaries();
    for (const entry of layers) {
      entry.layer = makeLayer(entry);
    }
    redraw();
  };
  // The bottom layer (land) lifts the layers stacked on it; with it
  // switched off they are rebuilt resting on the ground.
  const bottom = scene.layers[0].name;
  setChoice = (group, id) => {
    if (group === 'view') {
      valueView = id;
    } else {
      withEstimates = id === 'estimates';
      // Estimated features change height, so the pin and the geometry
      // follow.
      if (pin) {
        unpin();
      }
    }
    for (const entry of layers) {
      if (group === 'values') {
        entry.data = buildBinary(entry.decoded, !shown[bottom], flatGround, !withEstimates).data;
      }
      entry.layer = makeLayer(entry);
    }
    markViewSwitch();
    drawLegends();
    redraw();
  };
  toggleLayer = (name) => {
    shown[name] = !shown[name];
    // A pinned tooltip goes with its layer, and with any switch that
    // moves the buildings (land lowers them, terrain flattens all).
    if (pin) {
      unpin();
    }
    if (name === 'terrain') {
      flatGround = !shown.terrain;
      boundaries = makeBoundaries();
    }
    for (const entry of layers) {
      const moves =
        name === 'terrain' || (name === bottom && entry.decoded.arrays.stack);
      if (moves) {
        entry.data = buildBinary(entry.decoded, !shown[bottom], flatGround, !withEstimates).data;
      }
      entry.layer = makeLayer(entry);
    }
    markLayerSwitch();
    drawLegends();
    redraw();
  };

  // Display options only: they change colors, never which features
  // are shown.
  for (const option of palettes) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = option.label;
    button.dataset.palette = option.id;
    button.classList.toggle('active', option.id === palette.id);
    button.addEventListener('click', (event) => {
      // The chips sit in the legend's summary line: a click picks a
      // palette and must not also fold the legend.
      event.preventDefault();
      setPalette(option);
    });
    switcher.appendChild(button);
  }

  const jobs = [];
  for (const spec of scene.layers) {
    for (const [unit, paths] of Object.entries(spec.files)) {
      jobs.push({spec, unit, paths});
    }
  }
  // The tour's first stop's unit loads first, all its layers, so the
  // descent can start as soon as that unit is complete; the others
  // follow bottom layer first, each drawn as soon as it arrives.
  const firstUnit = scene.tour && scene.tour.length ? scene.tour[0].admin3_id : null;
  const first = jobs.filter((j) => j.unit === firstUnit);
  const rest = jobs.filter((j) => j.unit !== firstUnit);
  let firstLeft = first.length;
  if (!firstLeft) {
    tourControl.ready();
  }
  let done = 0;
  setStatus(`Loading 0 of ${jobs.length} files...`);
  // Units dropped while their files were loading; their layers are
  // discarded when the files arrive.
  const dropped = new Set();
  const load = async ({spec, unit, paths}) => {
    const [geometry, tooltip] = await Promise.all([
      fetchGzip(paths.geometry),
      fetchGzip(paths.tooltip),
    ]);
    if (dropped.has(unit)) {
      return;
    }
    const decoded = decodeScene(geometry);
    const values = JSON.parse(new TextDecoder().decode(tooltip));
    const {objectFeature, data} = buildBinary(
      decoded,
      !shown[bottom],
      flatGround,
      !withEstimates,
    );
    const id = `${spec.name}-${unit}`;
    layerInfo.set(id, {spec, values, objectFeature});
    // Ghosts are drawn in a layer of their own with the same features.
    layerInfo.set(`${id}-ghosts`, layerInfo.get(id));
    const entry = {
      id,
      unit,
      spec,
      decoded,
      data,
      objectFeature,
      featureValue: decoded.arrays.value,
      origin: [...decoded.header.origin, 0],
      order: scene.layers.indexOf(spec),
    };
    entry.layer = makeLayer(entry);
    layers.push(entry);
    layers.sort((a, b) => a.order - b.order);
    redraw();
    if (unit === firstUnit && --firstLeft === 0) {
      tourControl.ready();
    }
    done += 1;
    setStatus(done < jobs.length ? `Loading ${done} of ${jobs.length} files...` : '');
  };
  // The first file (the first stop's ground) loads alone, with all the
  // bandwidth, so something is on screen as early as possible.
  const queue = [...first, ...rest];
  if (!constrained(scene)) {
    if (queue.length) {
      await load(queue.shift());
    }
    await inBatches(queue, load);
    window.openplacesScene.loaded = true;
    return;
  }

  // A device short of memory holds only the units in and near view:
  // what the camera looks at loads, what it has left far behind is
  // dropped. Units follow the camera; nothing chooses them.
  const bounds = scene.unit_bounds;
  const loaded = new Set();
  let busy = false;
  const visibleUnits = () => {
    // deck.gl asserts when asked for its viewports before its first
    // frame; until then the scene's own starting view stands in.
    let v = null;
    try {
      v = deckgl.getViewports()[0] || null;
    } catch (error) {
      v = null;
    }
    v = v || {
      width: window.innerWidth,
      height: window.innerHeight,
      ...scene.view,
    };
    // The ground under the camera, as if looking straight down, with a
    // margin: a tilted view's far horizon would take in everything.
    const flat = new deck.WebMercatorViewport({
      width: v.width,
      height: v.height,
      longitude: v.longitude,
      latitude: v.latitude,
      zoom: v.zoom,
      pitch: 0,
      bearing: 0,
    });
    const [west, south, east, north] = flat.getBounds();
    const mx = (east - west) * VIEW_MARGIN;
    const my = (north - south) * VIEW_MARGIN;
    const inView = [];
    for (const [unit, [w, s, e, n]] of Object.entries(bounds)) {
      if (w <= east + mx && e >= west - mx && s <= north + my && n >= south - my) {
        // Distance from the view's center to the unit's box, in
        // degrees: 0 for the unit the camera is over.
        const dx = Math.max(w - v.longitude, 0, v.longitude - e);
        const dy = Math.max(s - v.latitude, 0, v.latitude - n);
        inView.push([Math.hypot(dx, dy), unit]);
      }
    }
    // A wide view takes in every unit; the nearest ones fill the
    // budget, and the rest wait for the camera to come closer.
    inView.sort((a, b) => a[0] - b[0]);
    return new Set(inView.slice(0, MAX_UNITS_CONSTRAINED).map(([, unit]) => unit));
  };
  const update = async () => {
    if (busy) {
      return;
    }
    busy = true;
    try {
      const near = visibleUnits();
      if (firstUnit) {
        near.add(firstUnit);
      }
      // Drop far units first, so their memory is free before more load.
      for (const unit of [...loaded]) {
        if (!near.has(unit)) {
          loaded.delete(unit);
          dropped.add(unit);
          for (let i = layers.length - 1; i >= 0; i--) {
            if (layers[i].unit === unit) {
              layerInfo.delete(layers[i].id);
              layers.splice(i, 1);
            }
          }
          redraw();
        }
      }
      const order = [...near].sort((a, b) => (a === firstUnit ? -1 : b === firstUnit ? 1 : 0));
      for (const unit of order) {
        if (!loaded.has(unit)) {
          loaded.add(unit);
          dropped.delete(unit);
          for (const job of jobs.filter((j) => j.unit === unit)) {
            await load(job);
          }
        }
      }
      // Counting files is meaningless when only some ever load.
      setStatus('');
      window.openplacesScene.loaded = true;
    } finally {
      busy = false;
    }
  };
  await update();
  setInterval(update, 1000);
}

// Margin around the view, as a share of its width and height, within
// which units load ahead of the camera on a constrained device.
const VIEW_MARGIN = 0.35;
// Most units a constrained device holds at once. Measured 2026-10-16
// on boston-inner-core and boston-outer-rings (phone-sized Edge, 4x CPU
// throttle): Boston plus one neighbor took 233-382 MB of JS heap, all
// 17 units in an overview 556 MB, about what crashed the maintainer's
// phone before.
const MAX_UNITS_CONSTRAINED = 6;

// A device that may not hold every unit at once: little memory
// reported, or a touch screen of phone size. ?all=1 forces loading
// everything, ?all=0 forces viewport loading (for testing).
function constrained(scene) {
  const flag = new URLSearchParams(location.search).get('all');
  if (flag === '1' || !scene.unit_bounds) {
    return false;
  }
  if (flag === '0') {
    return true;
  }
  const memory = navigator.deviceMemory;
  const phone = window.matchMedia('(pointer: coarse)').matches && screen.width < 900;
  return (memory !== undefined && memory <= 4) || phone;
}

main().catch((error) => {
  setStatus(`Could not load the scene: ${error.message}`);
  console.error(error);
});
