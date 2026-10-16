// Viewer of an openplaces value scene, written by
// openplaces.viz.web_scene. It reads scene.json and the binary files it
// names, and draws them with deck.gl on a plain background. Every unit
// in the manifest loads at start, and hovering a feature shows the
// fields shipped with it.

'use strict';

const AREA_UNIT_LABELS = {ac: 'acre', ha: 'ha', sqft: 'sq ft', m2: 'm²'};
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
function buildBinary({arrays}, dropStack = false) {
  const {xy, ring_offsets: rings, part_ring_offsets: partRings} = arrays;
  // A draped layer carries each vertex's own ground height in `z`;
  // a flat one stands every vertex at its feature's `base`.
  const {part_feature: partFeature, base, z: vertexZ, triangles} = arrays;
  // `stack`: the lift a feature gets from the layer below it; dropped
  // when that layer is switched off, so buildings rest on the ground.
  const {height, stack} = arrays;
  const lower = dropStack && stack ? stack : null;
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
    for (let v = rings[partRings[p]]; v < rings[partRings[p + 1]]; v++) {
      positions[3 * v] = xy[2 * v];
      positions[3 * v + 1] = xy[2 * v + 1];
      positions[3 * v + 2] = vertexZ
        ? vertexZ[v]
        : base[feature] - (lower ? lower[feature] : 0);
      elevations[v] = height[feature];
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

function choosePalette(palettes) {
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
    palettes[0]
  );
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
  // The label switches its whole layer on and off (display only: it
  // never selects features by their attributes).
  const label = document.createElement('button');
  label.type = 'button';
  label.className = 'legend-label';
  label.dataset.layer = legend.name;
  label.setAttribute('aria-pressed', String(legend.shown !== false));
  label.title = legend.shown === false ? 'Show' : 'Hide';
  label.textContent =
    `${legend.label} per ${AREA_UNIT_LABELS[legend.unit] || legend.unit}`;
  if (legend.onToggle) {
    label.addEventListener('click', legend.onToggle);
  }
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

async function main() {
  const scene = await (await fetch('scene.json', {cache: 'no-cache'})).json();
  document.title = scene.title;
  document.getElementById('title').textContent = scene.title;
  const notices = document.getElementById('notices');
  for (const [recipe, text] of Object.entries(scene.notices)) {
    const pre = document.createElement('pre');
    pre.textContent = `${recipe}\n\n${text}`;
    notices.appendChild(pre);
  }
  const palettes = scene.palettes.map(preparePalette);
  let palette = choosePalette(palettes);
  const legendPanel = document.getElementById('legend-panel');
  const switcher = document.createElement('div');
  switcher.id = 'palette-switch';
  const legends = document.createElement('div');
  legendPanel.append(switcher, legends);
  // Which layers are switched on; `toggleLayer` is set once deck.gl
  // exists.
  const shown = Object.fromEntries(scene.layers.map((l) => [l.name, true]));
  let toggleLayer = () => {};
  const drawLegends = () => {
    legends.replaceChildren();
    for (const layer of scene.layers) {
      const spec = palette.layers[layer.name];
      drawLegend(legends, {
        name: layer.name,
        shown: shown[layer.name],
        onToggle: () => toggleLayer(layer.name),
        label: layer.label,
        unit: layer.unit,
        vmin: spec.vmin,
        vmax: spec.vmax,
        // 32 stops are plenty for a gradient bar.
        colors: spec.lut.filter((_, i) => i % 8 === 0 || i === spec.lut.length - 1),
      });
    }
  };
  document.body.style.background = palette.background;
  drawLegends();

  await loadScript(scene.deck_gl_url);
  const {Deck, SolidPolygonLayer, PathLayer, COORDINATE_SYSTEM} = deck;

  // The tour runs while the files load, so the layers rise into a
  // moving view; ?tour=0 in the address, or a system setting for
  // reduced motion, opens a still view instead.
  const tour =
    new URLSearchParams(location.search).get('tour') !== '0' &&
    !window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const layerInfo = new Map();
  const deckgl = new Deck({
    parent: document.getElementById('scene'),
    initialViewState: tour
      ? {
        ...scene.view,
        zoom: scene.view.zoom + TOUR.farZoomOffset,
        pitch: TOUR.farPitch,
        maxPitch: 85,
      }
      : {...scene.view, maxPitch: 85},
    // On touch screens two fingers rotate (twist) and tilt (drag up
    // or down together) as well as zoom (pinch); deck.gl leaves the
    // twist off by default.
    controller: {touchRotate: true, dragRotate: true},
    layers: [],
    // Binary layers report the picked part's index, not an object.
    getTooltip: ({index, layer}) => {
      if (index < 0 || !layer || !layerInfo.has(layer.id)) {
        return null;
      }
      const info = layerInfo.get(layer.id);
      return {html: tooltipHtml(info, info.objectFeature[index])};
    },
  });

  // A device that runs out of graphics memory drops the drawing
  // context: say so, rather than leave a black screen.
  const canvas = deckgl.getCanvas && deckgl.getCanvas();
  if (canvas) {
    canvas.addEventListener('webglcontextlost', () => {
      setStatus('This device ran out of graphics memory. Reload to try again.');
    });
  }

  const tourControl = tour
    ? startTour(deckgl, scene.view, scene.tour, document.getElementById('scene'))
    : {stop: () => {}, ready: () => {}};

  // For measuring the page (frame rate, load time) from a test browser.
  window.openplacesScene = {deck: deckgl, loaded: false, stopTour: tourControl.stop};

  const makeBoundaries = () => new PathLayer({
    id: 'boundaries',
    data: scene.boundaries,
    getPath: (d) => d.path,
    getColor: palette.boundary || BOUNDARY_COLOR,
    getWidth: 1.5,
    widthUnits: 'pixels',
    // Drawn over the scene, so terrain and extrusions never hide it.
    parameters: {depthTest: false},
    updateTriggers: {getColor: palette.id},
  });
  let boundaries = makeBoundaries();

  // One loaded file: its deck.gl layer is rebuilt (same id, so deck.gl
  // only recomputes the colors) whenever the palette changes.
  const makeLayer = (entry) => {
    const colors = featureColors(
      entry.featureValue,
      palette.layers[entry.spec.name],
      palette.no_value,
    );
    return new SolidPolygonLayer({
      id: entry.id,
      data: entry.data,
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT_OFFSETS,
      coordinateOrigin: entry.origin,
      positionFormat: 'XYZ',
      _normalize: false,
      _windingOrder: 'CCW',
      extruded: true,
      wireframe: false,
      pickable: true,
      visible: shown[entry.spec.name],
      getFillColor: (_, {index}) => {
        const f = entry.objectFeature[index];
        return colors.subarray(4 * f, 4 * f + 4);
      },
      updateTriggers: {getFillColor: palette.id},
    });
  };
  const layers = [];
  const redraw = () => {
    deckgl.setProps({layers: [...layers.map((l) => l.layer), boundaries]});
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
  toggleLayer = (name) => {
    shown[name] = !shown[name];
    for (const entry of layers) {
      if (name === bottom && entry.decoded.arrays.stack) {
        entry.data = buildBinary(entry.decoded, !shown[bottom]).data;
      }
      entry.layer = makeLayer(entry);
    }
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
    button.addEventListener('click', () => setPalette(option));
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
    const {objectFeature, data} = buildBinary(decoded, !shown[bottom]);
    const id = `${spec.name}-${unit}`;
    layerInfo.set(id, {spec, values, objectFeature});
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
