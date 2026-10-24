// Viewer of an openplaces value scene, written by
// openplaces.viz.web_scene. It reads scene.json and the binary files it
// names, and draws them with deck.gl on a plain background. Every unit
// in the manifest loads at start, and hovering a feature shows the
// fields shipped with it.

'use strict';

// Units as legends name them; a dwelling is a "home" to visitors.
const AREA_UNIT_LABELS = {ac: 'acre', ha: 'ha', sqft: 'sq ft', m2: 'm²', dwelling: 'home'};
// Names of the layer switches (their tooltips); a layer not listed
// shows its label.
const LAYER_SWITCH_LABELS = {parcels: 'Land', buildings: 'Buildings'};
// Small symbols on the layer switches instead of words (maintainer,
// 2026-10-17), drawn in the text color: a lot seen at an angle, two
// blocks, the same blocks dashed and empty (ghosts), a pin.
const LAYER_ICONS = {
  parcels: '<path d="M1 10.5 7 6.5l10 2.5-6 5.5z" fill="currentColor" fill-opacity=".35"'
    + ' stroke="currentColor" stroke-linejoin="round"/>',
  buildings: '<path d="M3 15V7h5v8M8 15V2h7v13z" fill="currentColor" fill-opacity=".35"'
    + ' stroke="currentColor" stroke-linejoin="round"/>',
  ghosts: '<path d="M3 15V7h5v8M8 15V2h7v13z" fill="none" stroke="currentColor"'
    + ' stroke-dasharray="2 1.6" stroke-linejoin="round"/>',
  landmarks: '<path d="M9 15s-5-5-5-8.5a5 5 0 0 1 10 0C14 10 9 15 9 15z" fill="currentColor"'
    + ' fill-opacity=".35" stroke="currentColor"/><circle cx="9" cy="6.5" r="1.6"'
    + ' fill="currentColor"/>',
};
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
// The dotted top edge of a feature whose value openplaces imputed: a
// dot one line width long, a gap three (deck.gl's dash units are line
// widths), dark on light pages and light on dark ones. Full strength
// from zoom 16.5 in, fading out to nothing at zoom 14.5 and beyond
// (`styleEdges`), so they never stand out over a whole county
// (maintainer, 2026-10-23).
const IMPUTED_DASH = [1, 3];
const IMPUTED_LINE_WIDTH = 1.5;
const IMPUTED_FULL_ZOOM = 16.5;
const IMPUTED_FAR_ZOOM = 14.5;
const IMPUTED_LINE_LIGHT = [40, 44, 56, 230];
const IMPUTED_LINE_DARK = [235, 238, 245, 230];

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

// A recorded zero (`zeroValue`) is drawn apart from a missing value
// (NaN, `noValue`).
function featureColors(values, spec, noValue, zeroValue = noValue) {
  const out = new Uint8Array(values.length * 4);
  const lo = Math.log1p(spec.vmin);
  const span = Math.log1p(spec.vmax) - lo;
  const last = spec.rgb.length - 1;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === 0) {
      out.set(zeroValue, 4 * i);
      continue;
    }
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

// The Sources document (scene.json `sources`): per layer, each source
// with its share of the outlines, its terms, its credit and a link.
function renderSources(container, scene) {
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text) {
      node.textContent = text;
    }
    return node;
  };
  container.replaceChildren();
  container.appendChild(el('p', 'src-intro',
    'Values are as recorded by the assessors named below, with gaps filled by openplaces ' +
    '(openplaces.io), which harmonized and curated them. The terms are the sources\' own.'));
  // Known issues first: what we know is wrong in this page, and that
  // we look for it (maintainer, 2026-10-17).
  if (scene.known_issues) {
    container.appendChild(el('h3', 'src-layer', 'Known issues'));
    if (scene.known_issues.length) {
      const list = el('ul', 'src-issues');
      for (const issue of scene.known_issues) {
        const item = el('li', null, issue.text);
        item.appendChild(el('span', 'src-found', ` Found ${issue.found}.`));
        list.appendChild(item);
      }
      container.appendChild(list);
    } else {
      container.appendChild(el('p', 'src-note', 'None known for this place at the moment.'));
    }
  }
  // Values a model filled in (the building imputation): what it is,
  // where its code lives, and that "Tax assessor" leaves it out.
  if (scene.value_notes && scene.value_notes.length) {
    container.appendChild(el('h3', 'src-layer', 'Imputed values'));
    const notes = el('ul', 'src-list');
    for (const note of scene.value_notes) {
      const item = el('li', 'src-item');
      item.appendChild(el('div', 'src-head')).appendChild(el('span', 'src-name', note.name));
      item.appendChild(el('div', 'src-credit', note.credit));
      notes.appendChild(item);
    }
    container.appendChild(notes);
  }
  // A team page's models: what they are and whom they credit.
  if (scene.model_sources && scene.model_sources.length) {
    container.appendChild(el('h3', 'src-layer', 'Models'));
    const models = el('ul', 'src-list');
    for (const source of scene.model_sources) {
      const item = el('li', 'src-item');
      item.appendChild(el('div', 'src-head')).appendChild(el('span', 'src-name', source.name));
      item.appendChild(el('div', 'src-credit', source.credit));
      models.appendChild(item);
    }
    const estimated = Object.entries(scene.model_dollars || {}).filter(([, c]) => c.estimated);
    if (estimated.length) {
      models.appendChild(el('li', 'src-item', `Estimated: where a county's FHFA index ends ` +
        `before the latest year (${estimated.map(([fips, c]) => `${fips}, to ${c.to_year}`)
          .join('; ')}), its state's index carries it on.`));
    }
    container.appendChild(models);
  }
  const sections = scene.sources || [];
  const shareOf = (source) => {
    const percent = source.share * 100;
    const share = el('span', 'src-share', percent < 0.5 ? '<1%' : `${Math.round(percent)}%`);
    const bar = el('span', 'src-bar');
    bar.style.setProperty('--w', `${Math.max(2, source.share * 100)}%`);
    return [bar, share];
  };
  for (const section of sections) {
    container.appendChild(el('h3', 'src-layer', section.layer));
    // Who recorded the values, apart from who drew the outlines: the
    // outline shares read as the values' own (maintainer, 2026-10-17).
    if (section.values && section.values.length) {
      container.appendChild(el('h4', 'src-group', 'Values'));
      const values = el('ul', 'src-list');
      for (const source of section.values) {
        const item = el('li', 'src-item');
        const head = el('div', 'src-head');
        head.append(el('span', 'src-name', source.name), ...shareOf(source));
        item.appendChild(head);
        if (source.credit) {
          item.appendChild(el('div', 'src-credit', source.credit));
        }
        values.appendChild(item);
      }
      container.appendChild(values);
      container.appendChild(el('h4', 'src-group', 'Outlines'));
    }
    const list = el('ul', 'src-list');
    for (const source of section.sources) {
      const item = el('li', 'src-item');
      const head = el('div', 'src-head');
      head.appendChild(el('span', 'src-name', source.name));
      if (source.share) {
        head.append(...shareOf(source));
      }
      item.appendChild(head);
      if (source.credit) {
        item.appendChild(el('div', 'src-credit', source.credit));
      }
      if (source.terms) {
        const terms = el('div', 'src-terms', source.terms);
        if (source.url) {
          const link = el('a', null, 'terms');
          link.href = source.url;
          link.target = '_blank';
          link.rel = 'noopener';
          terms.append(' · ', link);
        }
        item.appendChild(terms);
      }
      list.appendChild(item);
    }
    container.appendChild(list);
    for (const note of section.share_alike || []) {
      container.appendChild(el('p', 'src-note', note));
    }
  }
  if (!sections.length && scene.notices) {
    // Scenes built before the Sources document: the raw notices.
    for (const [recipe, text] of Object.entries(scene.notices)) {
      container.appendChild(el('pre', null, `${recipe}\n\n${text}`));
    }
  }
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

// `extra`: rows the page adds from the feature's own arrays (a model
// view's value and area of applicability).
function tooltipHtml(layerInfo, index, extra = '') {
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
    rows.join('') + extra;
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

function rampCanvas(colors) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 1;
  const context = canvas.getContext('2d');
  const gradient = context.createLinearGradient(0, 0, 256, 0);
  colors.forEach((color, i) => {
    gradient.addColorStop(i / (colors.length - 1), color);
  });
  context.fillStyle = gradient;
  context.fillRect(0, 0, 256, 1);
  return canvas;
}

// One legend: its label, its color ramp and ticks. `legend.other` (per
// dwelling) adds the non-residential ramp as a thin strip right under
// the main one, with its range in one line, instead of a full legend of
// its own (maintainer, 2026-10-17).
function drawLegend(container, legend) {
  const block = document.createElement('div');
  block.className = legend.shown === false ? 'legend hidden' : 'legend';
  const label = document.createElement('div');
  label.className = 'legend-label';
  label.textContent =
    `${legend.label} per ${AREA_UNIT_LABELS[legend.unit] || legend.unit}`;
  // Which values the bar shows ("imputed", "assessed", later a model's
  // name), behind the title (maintainer, 2026-10-17).
  if (legend.source) {
    const source = document.createElement('span');
    source.className = 'legend-source';
    source.textContent = legend.source;
    label.appendChild(source);
  }
  const canvas = rampCanvas(legend.colors);
  const strip = legend.other ? rampCanvas(legend.other.colors) : null;
  if (strip) {
    strip.className = 'legend-other';
  }
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
  block.append(label, canvas);
  if (strip) {
    block.append(strip);
  }
  block.append(ticks);
  if (legend.other) {
    const {other} = legend;
    const note = document.createElement('div');
    note.className = 'legend-other-note';
    const swatch = document.createElement('span');
    swatch.style.background = `linear-gradient(to right, ${other.colors.join(', ')})`;
    const unit = AREA_UNIT_LABELS[other.unit] || other.unit;
    note.append(
      swatch,
      `Lower strip: non-residential, ${compactMoney(other.vmin)} to `
        + `${compactMoney(other.vmax)} per ${unit}`,
    );
    block.append(note);
  }
  container.appendChild(block);
}

// The type view's legend: one swatch per type the scene draws, the
// same list for land and buildings (scene.json `types`), the most
// common first: by count of lots, types on no lot (outbuildings)
// after them, and "Unknown" last (maintainer, 2026-10-19).
function drawTypeLegend(container, types) {
  types = [...types].sort((a, b) => (a.key === 'unknown') - (b.key === 'unknown')
    || (b.count || 0) - (a.count || 0));
  const block = document.createElement('div');
  block.className = 'legend type-legend';
  const label = document.createElement('div');
  label.className = 'legend-label';
  label.textContent = 'Land and building type';
  const source = document.createElement('span');
  source.className = 'legend-source';
  // Height is value per area, so a bar's volume is its value
  // (`render_terrain`), on one scale for land and buildings.
  source.textContent = 'volume: value';
  label.appendChild(source);
  const list = document.createElement('div');
  list.className = 'type-swatches';
  for (const type of types) {
    if (!type.present) {
      continue;
    }
    const item = document.createElement('span');
    // The lot's tint inside, the building's color around it.
    const swatch = document.createElement('i');
    swatch.style.background = `rgb(${typeRgb(type.color, TYPE_LOT_TINT)})`;
    swatch.style.border = `2px solid rgb(${typeRgb(type.color, 0, TYPE_BUILDING_SHADE)})`;
    item.append(swatch, type.label);
    list.appendChild(item);
  }
  block.append(label, list);
  container.appendChild(block);
}

// A pastel type view (maintainer, 2026-10-19): lots in a light tint of
// their type's color, buildings in the full color, a little darker, so
// they stand out from their lot.
const TYPE_LOT_TINT = 0.4;
const TYPE_BUILDING_SHADE = 0.85;

// One type color as drawn: mixed toward white by `tint`, then scaled
// by `shade`.
function typeRgb(hex, tint = 0, shade = 1) {
  return hexToRgb(hex).map((c) => Math.round((c + (255 - c) * tint) * shade));
}

// Each feature's color in the type view, from its `type_code`.
function typeColors(codes, types, alpha, tint = 0, shade = 1) {
  const rgb = types.map((t) => typeRgb(t.color, tint, shade));
  const out = new Uint8Array(codes.length * 4);
  for (let i = 0; i < codes.length; i++) {
    out.set(rgb[codes[i]] || rgb[0], 4 * i);
    out[4 * i + 3] = alpha;
  }
  return out;
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
// The tour stays high for longer and circles higher (maintainer,
// 2026-10-17): at least `highHold` seconds at the overview, a slower
// descent, stops seen from farther up. One stop (`tour_contrast`)
// shows the per-dwelling view over the sharpest contrast between
// costly and inexpensive homes, from higher still, for longer.
// It starts sooner and looks down a quarter less steeply (maintainer,
// 2026-10-18: `highHold` 14 to 9 s; pitches 55 and 48 to 42 and 36).
// The first stop is downtown, held longer (`downtownOrbit`), where the
// buildings go; the camera then travels slowly (`slowFlight`) to the
// contrast of costly and modest homes, where the view switches between
// value per ground area and per home twice.
const TOUR = {
  farZoomOffset: -1.5,
  farPitch: 0,
  highHold: 9,
  focusZoom: 14.3,
  focusPitch: 42,
  contrastZoom: 13.5,
  contrastPitch: 36,
  contrastOrbit: 46,
  downtownOrbit: 26,
  slowFlight: 1.6,
  // Share of the descent in which the center reaches the first stop.
  centerShare: 0.35,
  // A contrast farther than this from downtown is shown from higher up
  // rather than flown to (maintainer, 2026-10-18: zoom out instead).
  nearKm: 4,
  degreesPerSecond: 2.5,
  descent: 22,
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

// What the tour does at each stop, by seconds into the segment: a
// hint above the control first (`target`: the buildings switch, or a
// value view's button), which stays, and the switch when the countdown
// reaches 0 (maintainer, 2026-10-17: a hint shown only briefly was
// easy to miss). Switches count down from five (`countdown`;
// maintainer, 2026-10-19).
// Visitors read "home"; the data's term is "dwelling" (one house,
// apartment or condo).
// Two places for words (maintainer, 2026-10-18). The main messages,
// centered at about a quarter of the map's height: the autopilot line,
// what is about to switch (`text`), and afterwards what the visitor
// can now see (an insight, naming no figures from the map). A bubble
// above the switching button keeps the definition (`tip`) and the
// countdown.
// Downtown, the first stop: the buildings go, and land speaks alone.
const TOUR_DOWNTOWN_ACTIONS = [
  {
    at: 4,
    hint: {
      target: 'buildings',
      text: 'Hiding the buildings to show land value alone',
      countdown: 5,
      linger: 9,
    },
  },
  {at: 9, buildings: false},
  {
    at: 13,
    hint: {
      // Maintainer's wording, 2026-10-19.
      text: 'Land value per area gives you an idea of where humans most want to ' +
        'secure land rights to govern, do commerce, and to live and sleep.',
      linger: 10,
    },
  },
];
// Without a contrast stop, the buildings return before downtown is
// left.
const TOUR_DOWNTOWN_ONLY_ACTIONS = [
  ...TOUR_DOWNTOWN_ACTIONS,
  {at: 23, hint: {target: 'buildings', text: 'Buildings back', linger: 3}},
  {at: 25, buildings: true},
];
// Leaving downtown for the contrast: the switch to value per home
// comes 6 s after leaving (maintainer, 2026-10-18: no more than about
// 7 s), while the camera glides on: one second for the message, then
// the countdown from five. The flight lasts at least 16 s
// (`flightBase` x `slowFlight`), so the insight at 10 s always runs.
const TOUR_TO_CONTRAST_ACTIONS = [
  {
    at: 0,
    hint: {
      target: 'dwelling',
      text: 'You can visualize wealth distributions by switching to value *per home*.',
      tip: 'A home is one apartment, house, or condo.',
      countdown: 5,
      countdownAfter: 1,
      linger: 9,
    },
  },
  {at: 6, view: 'dwelling'},
  {
    at: 10,
    hint: {
      text: 'This gives you a first idea of where the rich and the poor tend to live.',
      linger: 10,
    },
  },
];
// At the contrast: each view once (maintainer, 2026-10-19). After
// value per home, the buildings come back, then the type view; once
// its text has run, and a few seconds more, back to value per ground
// area, which the tour keeps from there on (maintainer, 2026-10-23).
// The contrast orbit lasts `TOUR.contrastOrbit` (46 s), so the last
// switch, at 44 s, still happens there.
const TOUR_CONTRAST_ACTIONS = [
  {at: 4, hint: {target: 'buildings', text: 'Buildings back', countdown: 5, linger: 7}},
  {at: 9, buildings: true},
  {
    at: 14,
    hint: {
      target: 'type',
      text: 'You can also color land and buildings by their *type*.',
      tip: 'Single-family, multi-family, commercial, industrial, and more.',
      countdown: 5,
      countdownAfter: 1,
      linger: 9,
    },
  },
  {at: 20, view: 'type'},
  {
    at: 24,
    hint: {text: 'Homes, shops, industry, and public land, each in its own color.', linger: 10},
  },
  {
    at: 39,
    hint: {target: 'area', text: 'Back to value per ground area', countdown: 5, linger: 7},
  },
  {at: 44, view: 'area'},
];
// Leaving the contrast: the view the tour last showed stays.
const TOUR_LEAVE_ACTIONS = [];

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
// A segment may carry `valueView` ('area' or 'dwelling'): the coloring
// the tour switches to as the segment begins.
function buildTour(view, foci, contrast, downtown = null) {
  const far = {
    longitude: view.longitude,
    latitude: view.latitude,
    zoom: view.zoom + TOUR.farZoomOffset,
    pitch: TOUR.farPitch,
  };
  // A stop: where it is, and how high and steep it is seen from.
  const stop = (at, zoom = TOUR.focusZoom, pitch = TOUR.focusPitch) => ({
    longitude: at.longitude,
    latitude: at.latitude,
    zoom,
    pitch,
  });
  // The center reaches the first stop in the first third of the
  // descent and stays on it while the camera comes down, so downtown
  // sits near the middle of the screen until arrival (maintainer,
  // 2026-10-18).
  const descend = (to) => ({
    duration: TOUR.descent,
    view: (t) => {
      const s = ease(t / TOUR.descent);
      const c = ease(t / (TOUR.descent * TOUR.centerShare));
      return {
        longitude: lerp(far.longitude, to.longitude, c),
        latitude: lerp(far.latitude, to.latitude, c),
        zoom: lerp(far.zoom, to.zoom, s),
        pitch: lerp(far.pitch, to.pitch, s),
      };
    },
  });
  const orbit = (at, duration = TOUR.orbit) => ({duration, view: () => ({...at})});
  // `slow`: longer and lower, for a glide rather than a hop.
  const fly = (from, to, slow = 1) => {
    const km = distanceKm(from, to);
    const duration =
      slow * Math.min(TOUR.flightMax, TOUR.flightBase + TOUR.flightPerKm * km);
    const climb = Math.min(TOUR.climbMax, TOUR.climbPerKm * km) / slow;
    return {
      duration,
      view: (t) => {
        const s = ease(t / duration);
        const arc = Math.sin(Math.PI * s);
        return {
          longitude: lerp(from.longitude, to.longitude, s),
          latitude: lerp(from.latitude, to.latitude, s),
          zoom: lerp(from.zoom, to.zoom, s) - climb * arc,
          pitch: lerp(from.pitch, to.pitch, s) - TOUR.pitchDip * arc,
        };
      },
    };
  };

  // Circle slowly at the high view until the first stop's buildings
  // are on screen (`ready`), so the descent never arrives at bare
  // ground; its duration is set when they are.
  const hold = {duration: Infinity, view: () => ({...far})};

  // Downtown first, always (maintainer, 2026-10-18); without a
  // downtown, the densest stop, or the area's center.
  const stops = (foci || []).map((f) => stop(f));
  const first = downtown
    ? stop(downtown)
    : stops.shift() || stop({longitude: view.longitude, latitude: view.latitude});
  const loop = [];
  let at = first;
  if (contrast) {
    // Downtown, buildings off; a slow glide to the contrast, turning
    // to its angle, land alone; there the switching between per
    // ground area and per home; then on to the other stops.
    // A contrast far from downtown is seen from higher up, so the
    // glide stays short and both ends fit the view.
    const km = distanceKm(first, contrast);
    const zoom = km > TOUR.nearKm
      ? TOUR.contrastZoom - Math.log2(km / TOUR.nearKm)
      : TOUR.contrastZoom;
    const high = stop(contrast, zoom, TOUR.contrastPitch);
    loop.push(
      {...orbit(first, TOUR.downtownOrbit), actions: TOUR_DOWNTOWN_ACTIONS},
      {...fly(first, high, TOUR.slowFlight), actions: TOUR_TO_CONTRAST_ACTIONS},
      {...orbit(high, TOUR.contrastOrbit), actions: TOUR_CONTRAST_ACTIONS},
    );
    at = high;
  } else {
    loop.push({...orbit(first, TOUR.downtownOrbit), actions: TOUR_DOWNTOWN_ONLY_ACTIONS});
  }
  for (const [i, next] of stops.entries()) {
    loop.push(i === 0 && contrast ? {...fly(at, next), actions: TOUR_LEAVE_ACTIONS} : fly(at, next));
    loop.push(orbit(next));
    at = next;
  }
  // Each round ends back downtown.
  if (at !== first) {
    loop.push(fly(at, first));
  }
  return {hold, intro: [hold, descend(first)], loop};
}

// `onAction` carries out a segment's timed `actions` (hints and
// switches); `onStop` runs once when the visitor takes control.
function startTour(
  deckgl,
  view,
  foci,
  element,
  contrast = null,
  onAction = () => {},
  onStop = () => {},
  downtown = null,
  from = null,
) {
  // `from`: the camera as the visitor left it, for a replay, which
  // then glides out from there instead of jumping to the overview.
  const {hold, intro, loop} = buildTour(view, foci, contrast, downtown);
  // The segment entered last, and how many of its actions have run.
  let entered = -1;
  let fired = 0;
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
    if (index !== entered) {
      entered = index;
      fired = 0;
    }
    const actions = segments[index].actions || [];
    while (fired < actions.length && t >= actions[fired].at) {
      onAction(actions[fired]);
      fired += 1;
    }
    const target = segments[index].view(t);
    // The turn is continuous for the whole tour, so it needs no
    // smoothing: it never changes speed.
    const bearing =
      (from ? from.bearing : 0) + (TOUR.degreesPerSecond * (now - start)) / 1000;
    if (stage1 === null) {
      const first = from || target;
      stage1 = Object.fromEntries(keys.map((key) => [key, first[key]]));
      stage2 = {...stage1};
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
    if (running) {
      running = false;
      onStop();
    }
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
      // At least `highHold` seconds at the overview.
      hold.duration = Math.max(
        TOUR.highHold,
        index === 0 && segmentStart !== null
          ? (performance.now() - segmentStart) / 1000
          : 0,
      );
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
// layers are off; and the choices of what is drawn (`choices`: the
// view, the values, the local or nationwide scale; maintainer,
// 2026-10-19). The palette carries over by itself (stored per site).
function buildPlaceNav(places, deckgl, isTouring, shown, choices = () => ({})) {
  if (!places.length) {
    return;
  }
  // A phone page (`<slug>-phone`) is the same place.
  const here = location.pathname.split('/').filter(Boolean).pop().replace(/-phone$/, '');
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
    // What is off travels as `hide`; ghosts, off by default, as `show`.
    const hidden = Object.keys(shown).filter((name) => name !== 'ghosts' && !shown[name]);
    if (hidden.length) {
      query.set('hide', hidden.join(','));
    }
    if (shown.ghosts) {
      query.set('show', 'ghosts');
    }
    // Only what differs from a page's opening, so a plain visit stays
    // a plain link. `carry` lets the next page open on the type view,
    // which a bare address never does.
    const {view, values, range} = choices();
    if (view && view !== 'area') {
      query.set('view', view);
      query.set('carry', '1');
    }
    if (values === 'assessed') {
      query.set('values', 'assessed');
    }
    if (range === 'national') {
      query.set('range', 'national');
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
  // A phone gets the place's lighter page (`phone_scene`, a smaller
  // circle) with the same address parameters; ?full=1 stays here.
  if (scene.phone_scene && params.get('full') !== '1' && lowCapacity()) {
    location.replace(`${scene.phone_scene}${location.search}${location.hash}`);
    return;
  }
  // "[place]: Land and building values" in the browser's tab and
  // bookmarks (maintainer, 2026-10-18).
  document.title = scene.place_title
    ? `${scene.place_title}: Land and building values`
    : scene.title;
  document.getElementById('title').textContent = scene.title;
  // No subtitle on the scene: what the values are follows from the
  // controls (with estimates, or the assessment alone).
  // The other places of the site (`places.json`, written with the
  // landing page beside the scenes); absent for a lone scene.
  const places = await fetch('../places.json', {cache: 'no-cache'})
    .then((r) => (r.ok ? r.json() : []))
    .catch(() => []);
  renderSources(document.getElementById('notices'), scene);
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
  // On a desktop the controls stand in the bottom bar, right after the
  // legend, always shown (index.html); on a phone they open above the
  // bar from their chip.
  const desktop = window.matchMedia('(min-width: 641px)');
  const dockControls = () => {
    if (desktop.matches) {
      document.getElementById('legend-box').after(legendPanel);
    } else {
      bar.before(legendPanel);
    }
    placeControls();
  };
  desktop.addEventListener('change', dockControls);
  dockControls();
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
  // Lots can be ghosts too (`mark_land_ghosts`): slivers carrying a
  // whole account's value. The switch then names what it holds.
  const ghostKinds = scene.layers
    .filter((l) => l.n_ghosts > 0)
    .map((l) => (l.name === 'buildings' ? 'buildings' : 'lots'));
  const ghostWhat = ghostKinds.includes('lots')
    ? (ghostKinds.includes('buildings') ? 'values' : 'lots')
    : 'buildings';
  if (hasGhosts) {
    // Off by default (maintainer, 2026-10-17); `?show=ghosts` opens them.
    shown.ghosts = (params.get('show') || '').split(',').includes('ghosts');
    switches.push(['ghosts', `Unexplained ${ghostWhat}?`]);
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
    const icon = LAYER_ICONS[name];
    if (icon) {
      // A checkbox and a symbol; the name is the tooltip. A name ending
      // in "?" gets a question mark of its own after the button (the
      // ghosts' link to the methods page, below).
      const plain = label.replace(/\?$/, '');
      button.title = plain;
      button.setAttribute('aria-label', plain);
      button.innerHTML = `<svg viewBox="0 0 18 16" aria-hidden="true">${icon}</svg>`;
    } else {
      button.textContent = label;
    }
    button.addEventListener('click', () => toggleLayer(name));
    layerSwitch.appendChild(button);
  }
  if (hasGhosts) {
    // The question mark right after the switch is a link of its own to
    // the methods page's section on these buildings, with a tooltip
    // that says what they are (maintainer, 2026-10-17). A new tab, so
    // the map stays where it is; outside the button, so a click on it
    // never switches the layer.
    const note = document.createElement('a');
    note.className = 'ghost-note';
    note.href = '../methods.html#unexplained';
    note.target = '_blank';
    note.rel = 'noopener';
    note.textContent = '?';
    const why = {
      buildings: 'Unexplained buildings: their value per square foot is higher than '
        + 'their measured floors explain.',
      lots: 'Unexplained lots: tiny outlines carrying far more value per acre than '
        + 'any other land here.',
    };
    const kinds = [...new Set(ghostKinds)];
    note.title = `${kinds.map((k) => why[k]).join(' ')} Drawn see-through when `
      + 'switched on. Click for how they are found.';
    note.setAttribute(
      'aria-label', `What are unexplained ${ghostWhat}? (methods page)`,
    );
    layerSwitch.querySelector('[data-layer="ghosts"]').after(note);
  }
  const markLayerSwitch = () => {
    for (const button of layerSwitch.querySelectorAll('button')) {
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
  // A layer's other colorings (`views`): per dwelling, and on a team
  // page the model land values (`model`), each a button of its own.
  const viewOf = (layer, key = valueView) => (layer.views && layer.views[key]) || null;
  const modelViews = [];
  for (const layer of scene.layers) {
    for (const [key, view] of Object.entries(layer.views || {})) {
      const plain = !view.per_home && !view.dollars;
      if (view.model && plain && !modelViews.some(([k]) => k === key)) {
        modelViews.push([key, view.label]);
      }
    }
  }
  // A model's views: per ground area (`<key>`) or per home
  // (`<key>__home`), chosen with the same toggle as the assessor's;
  // scaled to the assessments, or in today's dollars (`<key>__2026`).
  const HOME_SUFFIX = '__home';
  const DOLLARS_SUFFIX = '__2026';
  const modelBase = (key) => key.replace(HOME_SUFFIX, '').replace(DOLLARS_SUFFIX, '');
  const isModel = (key) => modelViews.some(([k]) => k === modelBase(key));
  const isPerHome = (key) => key === 'dwelling' || (isModel(key) && key.endsWith(HOME_SUFFIX));
  const hasDollars = scene.layers.some((l) =>
    Object.values(l.views || {}).some((v) => v.dollars));
  // The year the dollars reach, and whether a county's series was
  // carried on by its state's (an estimate).
  const dollarCounties = Object.values(scene.model_dollars || {});
  const dollarsYear = Math.max(...dollarCounties.map((c) => c.to_year), 0);
  const dollarsEstimated = dollarCounties.some((c) => c.estimated);
  const viewKeys = new Set(['area', 'dwelling']);
  for (const layer of scene.layers) {
    Object.keys(layer.views || {}).forEach((key) => viewKeys.add(key));
  }
  // A page opens on a value view, value per ground area unless the
  // address names another; on the type view only when the visitor
  // carried it from another place (`carry`, the places menu;
  // maintainer, 2026-10-19). A view this page lacks (another page's
  // model) falls back to the same kind here.
  const asked = params.get('view');
  const carried = params.get('carry') === '1';
  let valueView = viewKeys.has(asked) && (asked !== 'type' || carried)
    ? asked
    : asked && asked.endsWith('__home') && viewKeys.has('dwelling')
      ? 'dwelling'
      : 'area';
  let withEstimates = params.get('values') !== 'assessed';
  // Local color bounds (the scene's own) or nationwide ones
  // (`national_bounds`, per display unit).
  let colorRange = params.get('range') === 'national' ? 'national' : 'local';
  const boundsFor = (unit, vmin, vmax) => {
    const national = colorRange === 'national' && scene.national_bounds &&
      scene.national_bounds[unit];
    return national ? {vmin: national[0], vmax: national[1]} : {vmin, vmax};
  };
  // The model a "Modeled" choice shows: the one in the address, else
  // the first.
  let modelKey = isModel(valueView) ? modelBase(valueView) : (modelViews[0] || [null])[0];
  // Models in today's dollars already (`model_year` null) open in
  // their own dollars; PLACES-FMV's 2010 ones scaled to assessments.
  let modelDollars = valueView.includes(DOLLARS_SUFFIX) ||
    (scene.model_year === null && !isModel(valueView));
  const modelView = (perHome) =>
    `${modelKey}${modelDollars ? DOLLARS_SUFFIX : ''}${perHome ? HOME_SUFFIX : ''}`;
  const hasDwellingView = scene.layers.some((l) => viewOf(l, 'dwelling'));
  // Rows: per area or per dwelling; then where the values come from
  // (maintainer, 2026-10-18): "Tax assessor", "+ imputed" (the
  // default) and, on a team page, "Modeled", which alone shows a menu
  // of the models.
  const viewSwitch = document.createElement('div');
  viewSwitch.id = 'view-switch';
  const valuesSwitch = document.createElement('div');
  valuesSwitch.id = 'values-switch';
  const choices = [];
  if (hasDwellingView || modelViews.length) {
    // Per ground area first (the default), then per dwelling. "Ground"
    // covers both a lot's area and a building's footprint, the two
    // areas the bars divide by.
    choices.push(['view', 'area', 'Value per ground area']);
  }
  if (hasDwellingView) {
    // "Home", not "dwelling", for visitors (maintainer, 2026-10-17).
    choices.push(['view', 'dwelling', 'Value per home']);
  }
  // The type view (maintainer, 2026-10-19): land and buildings colored
  // by one list of types; the heights still show the value.
  const hasTypeView = scene.layers.some((l) => viewOf(l, 'type')) && scene.types;
  if (hasTypeView) {
    if (!choices.some(([, id]) => id === 'area')) {
      choices.push(['view', 'area', 'Value per ground area']);
    }
    choices.push(['view', 'type', 'Type']);
  }
  if (scene.layers.some((l) => l.n_estimated > 0) || modelViews.length) {
    choices.push(['values', 'assessed', 'Tax assessor'], ['values', 'estimates', '+ imputed']);
  }
  if (modelViews.length) {
    choices.push(['values', 'model', 'Modeled']);
  }
  // The year the models describe (PLACES-FMV: 2010); none for models
  // handed over in today's dollars (`model_year` null).
  const modelYear = scene.model_year === undefined ? 2010 : scene.model_year;
  const titles = {
    assessed: 'Values the assessor recorded; values openplaces filled in are drawn as no value',
    estimates: "The assessor's values, with openplaces' fills where the roll has none",
    model: modelYear
      ? `Land value from a ${modelYear} model, scaled to the place's assessments`
      : 'Land value from a model',
  };
  let setChoice = () => {};
  for (const [group, id, label] of choices) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.group = group;
    button.dataset.choice = id;
    button.textContent = label;
    button.title = titles[id] || '';
    button.setAttribute('role', 'radio');
    button.addEventListener('click', () => setChoice(group, id));
    (group === 'view' ? viewSwitch : valuesSwitch).appendChild(button);
  }
  // The models' menu: a folded list of buttons under "Modeled", shown
  // only while "Modeled" is chosen. A display choice, like the
  // palettes: it selects no features.
  const modelMenu = document.createElement('details');
  modelMenu.id = 'model-menu';
  const modelSummary = document.createElement('summary');
  modelMenu.append(modelSummary);
  const modelList = document.createElement('div');
  modelList.setAttribute('role', 'radiogroup');
  modelList.setAttribute('aria-label', 'Model shown');
  for (const [key, label] of modelViews) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.group = 'model';
    button.dataset.choice = key;
    button.textContent = label;
    button.setAttribute('role', 'radio');
    button.addEventListener('click', () => {
      modelMenu.open = false;
      setChoice('model', key);
    });
    modelList.append(button);
  }
  modelMenu.append(modelList);
  // Under "Modeled": the models scaled to the assessments, or in
  // today's dollars (maintainer, 2026-10-18).
  const scaleSwitch = document.createElement('div');
  scaleSwitch.id = 'scale-switch';
  scaleSwitch.setAttribute('role', 'radiogroup');
  scaleSwitch.setAttribute('aria-label', 'Model scale');
  if (hasDollars) {
    const scales = [
      ['scaled', 'Scaled to assessments',
        'Each model times one number: the median of recorded over predicted land value'],
      ['dollars', `${dollarsYear} dollars`,
        modelYear
          ? 'Each model brought to today with consumer and house price indexes (CPI-U, FHFA)'
          : `Each model's own prediction, in ${dollarsYear} dollars${
            scene.model_basis ? ` (${scene.model_basis})` : ''}`],
    ];
    for (const [id, label, title] of scales) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.group = 'scale';
      button.dataset.choice = id;
      button.textContent = label;
      button.title = title;
      button.setAttribute('role', 'radio');
      button.addEventListener('click', () => setChoice('scale', id));
      scaleSwitch.append(button);
    }
  }
  // Each pair is one segmented toggle, one option always chosen
  // (maintainer, 2026-10-17: clearly a toggle, not two buttons).
  viewSwitch.setAttribute('role', 'radiogroup');
  viewSwitch.setAttribute('aria-label', 'Value shown');
  valuesSwitch.setAttribute('role', 'radiogroup');
  valuesSwitch.setAttribute('aria-label', 'Values used');
  layerSwitch.after(viewSwitch, valuesSwitch, modelMenu, scaleSwitch);
  const markViewSwitch = () => {
    const modeled = isModel(valueView);
    const source = modeled ? 'model' : withEstimates ? 'estimates' : 'assessed';
    const buttons = [
      ...viewSwitch.children,
      ...valuesSwitch.children,
      ...modelList.children,
      ...scaleSwitch.children,
    ];
    for (const button of buttons) {
      const {group, choice} = button.dataset;
      const on = group === 'values'
        ? choice === source
        : group === 'model'
          ? choice === modelKey
          : group === 'scale'
            ? (choice === 'dollars') === modelDollars
            : choice === (valueView === 'type'
              ? 'type'
              : isPerHome(valueView) ? 'dwelling' : 'area');
      button.classList.toggle('active', on);
      button.setAttribute('aria-checked', String(on));
    }
    modelMenu.hidden = !modeled;
    scaleSwitch.hidden = !modeled || !hasDollars;
    const current = modelViews.find(([key]) => key === modelKey);
    modelSummary.textContent = current
      ? `Model${modelYear ? `, ${modelYear}` : ''}: ${current[1]}`
      : '';
  };
  markViewSwitch();
  // A layer's coloring under the current choices: its scale (bounds,
  // unit) and the value of each feature.
  // A model view keeps the layer's own bounds, so its colors read on
  // the same scale as the assessed values.
  const layerScale = (layer, spec) => {
    const view = viewOf(layer);
    const unit = view ? view.unit : layer.unit;
    const local = view
      ? {vmin: view.vmin ?? spec.vmin, vmax: view.vmax ?? spec.vmax}
      : {vmin: spec.vmin, vmax: spec.vmax};
    return {...spec, ...boundsFor(unit, local.vmin, local.vmax), unit};
  };
  // The non-residential part of a layer's per-dwelling view, if shown.
  const otherView = (entry) => {
    const view = viewOf(entry.spec);
    return view && view.other && entry.decoded.arrays.nonresidential ? view.other : null;
  };
  const featureValues = (entry, perArea = false) => {
    const {arrays} = entry.decoded;
    const view = viewOf(entry.spec);
    const own = view && arrays[`value_${valueView}`];
    let values = own && !perArea ? own : entry.featureValue;
    // A model's values are the model's: the assessment choices below
    // do not apply to them. Per area (the non-residential lots of a
    // per-home view), the model's own per-area values.
    if (own && view.model) {
      return perArea ? arrays[`value_${view.base || valueView}`] || values : values;
    }
    if (!withEstimates && arrays.estimated) {
      // Not a recorded zero: drawn as no value.
      values = values.map((v, i) => (arrays.estimated[i] ? NaN : v));
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

  // Under the legend, while imputed values are drawn: what the dotted
  // top edges mean (`IMPUTED_DASH`).
  const drawImputedNote = () => {
    if (!withEstimates || isModel(valueView) || !scene.layers.some((l) => l.n_estimated > 0)) {
      return;
    }
    const note = document.createElement('div');
    note.className = 'legend-imputed';
    const sample = document.createElement('span');
    note.append(sample, 'Dotted edge: value imputed by openplaces');
    legends.appendChild(note);
  };
  const drawLegends = () => {
    legends.replaceChildren();
    // Which values the bars show, named small behind each title.
    const valuesShown = withEstimates ? 'imputed' : 'assessed';
    // The type view has no ramp, so no local or nationwide scale.
    const rangeChoice = document.getElementById('range-switch');
    if (rangeChoice) {
      rangeChoice.hidden = valueView === 'type';
    }
    if (valueView === 'type' && hasTypeView) {
      drawTypeLegend(legends, scene.types);
      drawImputedNote();
      placeControls();
      return;
    }
    for (const layer of scene.layers) {
      const spec = layerScale(layer, palette.layers[layer.name]);
      // Per dwelling: the non-residential ramp, drawn as a strip under
      // the main one.
      const view = viewOf(layer);
      const ramp = spec.other_line_lut || spec.other_lut;
      const other = view && view.other && ramp
        ? {
          unit: view.other.unit,
          ...boundsFor(view.other.unit, view.other.vmin, view.other.vmax),
          colors: ramp.filter((_, i) => i % 8 === 0 || i === ramp.length - 1),
        }
        : null;
      drawLegend(legends, {
        shown: shown[layer.name],
        label: view && view.model ? `${layer.label}, model ${view.label}` : layer.label,
        unit: spec.unit,
        vmin: spec.vmin,
        vmax: spec.vmax,
        // 32 stops are plenty for a gradient bar.
        // Where the edges carry the value, the legend shows theirs.
        colors: (spec.line_lut || spec.lut).filter(
          (_, i) => i % 8 === 0 || i === spec.lut.length - 1,
        ),
        other,
        source: !(view && view.model)
          ? valuesShown
          : view.dollars
            ? `${modelYear ? `${modelYear} model` : 'model'} in ${dollarsYear} dollars`
              + (dollarsEstimated ? ', estimated' : '')
            : `${modelYear ? `${modelYear}, ` : ''}scaled to assessments`,
      });
    }
    // No row for features without a value (maintainer, 2026-10-17).
    // A third bar changes the legend's height.
    drawImputedNote();
    placeControls();
  };
  document.body.style.background = palette.background;
  document.body.classList.toggle('light-page', palette.theme === 'light');
  drawLegends();

  await loadScript(scene.deck_gl_url);
  const {Deck, SolidPolygonLayer, PathLayer, COORDINATE_SYSTEM} = deck;
  const {LightingEffect, AmbientLight, DirectionalLight} = deck;
  const {ScatterplotLayer, TextLayer, DataFilterExtension, PathStyleExtension} = deck;

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
    : scene.tour_downtown || (scene.tour && scene.tour.length ? scene.tour[0] : scene.view);
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
  let hoverAt = () => {};
  let placePin = () => {};
  let turnCompass = () => {};
  let fadeEdges = () => {};
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
    // The hover tooltip is the page's own (`showHover`), not deck.gl's.
    onHover: (picked) => hoverAt(picked),
    // Deferred: both are set once the layers they read exist.
    onClick: (picked) => togglePin(picked),
    onAfterRender: () => {
      placePin();
      turnCompass();
      fadeEdges();
    },
  });

  // The hover tooltip, steady while the camera moves (maintainer,
  // 2026-10-17: deck.gl's own flickered as the view turned, because a
  // pick between frames often found nothing). Three rules keep it
  // still: its text changes only when the feature under the cursor
  // changes; an empty pick hides it only after `HOVER_GRACE_MS`, so a
  // miss between two hits never shows; and while the view moves under a
  // resting cursor, the page picks again a few times a second, so the
  // tooltip follows what is now under it.
  const HOVER_GRACE_MS = 350;
  const HOVER_REPICK_MS = 150;
  const hoverBox = document.createElement('div');
  hoverBox.className = 'hover-tip deck-tooltip';
  hoverBox.hidden = true;
  document.body.appendChild(hoverBox);
  let hoverKey = null;
  let hoverTimer = null;
  let pointer = null;
  // Under a model view: the model's value of the feature and whether it
  // lies inside the model's area of applicability. Shown as text only;
  // nothing is hidden or dimmed by it (design limit L4).
  const modelRows = (layerId, feature) => {
    const entry = layers.find((e) => e.id === layerId || `${e.id}-ghosts` === layerId);
    const view = entry && viewOf(entry.spec);
    // Under the type view, the type the feature's color stands for.
    const codes = view && view.categorical && entry.decoded.arrays.type_code;
    if (codes) {
      const type = scene.types[codes[feature]] || scene.types[0];
      return `<div><b>Type:</b> ${escapeHtml(type.label)}</div>`;
    }
    const values = view && view.model && entry.decoded.arrays[`value_${valueView}`];
    if (!values) {
      return '';
    }
    const value = values[feature];
    const text = Number.isFinite(value)
      ? formatValue(value, 'money_per_area', view.unit)
      : 'none';
    const aoa = entry.decoded.arrays[`aoa_${valueView}`];
    const inside = aoa ? ['unknown', 'outside', 'inside'][aoa[feature]] : null;
    const scale = view.dollars ? `${dollarsYear} dollars` : 'scaled';
    return `<div><b>Model ${escapeHtml(view.label)} (${scale}):</b> ${text}</div>` +
      (inside ? `<div><b>Area of applicability:</b> ${inside}</div>` : '');
  };
  // What a pick shows: a key naming the feature, and its tooltip.
  const describe = ({index, layer, object}) => {
    // A landmark's own name and kind, written into the page.
    if (layer && layer.id.startsWith('landmark') && object) {
      return {
        key: `landmark:${object.name}`,
        html: `<b>${escapeHtml(object.name)}</b><br>${escapeHtml(object.kind)}`,
      };
    }
    // Binary layers report the picked part's index, not an object.
    if (index === undefined || index < 0 || !layer || !layerInfo.has(layer.id)) {
      return null;
    }
    const info = layerInfo.get(layer.id);
    const feature = info.objectFeature[index];
    // The pinned feature already shows its values.
    if (pin && pin.layerId === layer.id && pin.feature === feature) {
      return null;
    }
    return {
      key: `${layer.id}:${feature}:${valueView}`,
      html: () => tooltipHtml(info, feature, modelRows(layer.id, feature)),
    };
  };
  const hideHover = () => {
    clearTimeout(hoverTimer);
    hoverTimer = null;
    hoverKey = null;
    hoverBox.hidden = true;
  };
  const showHover = (shown, x, y) => {
    if (!shown) {
      if (!hoverBox.hidden && hoverTimer === null) {
        hoverTimer = setTimeout(hideHover, HOVER_GRACE_MS);
      }
      return;
    }
    clearTimeout(hoverTimer);
    hoverTimer = null;
    if (shown.key !== hoverKey) {
      hoverKey = shown.key;
      hoverBox.innerHTML = typeof shown.html === 'function' ? shown.html() : shown.html;
    }
    hoverBox.hidden = false;
    // Beside the cursor, kept inside the window.
    const width = hoverBox.offsetWidth;
    const height = hoverBox.offsetHeight;
    const left = x + 14 + width > window.innerWidth ? x - 14 - width : x + 14;
    const top = y + 14 + height > window.innerHeight ? y - 14 - height : y + 14;
    hoverBox.style.transform = `translate(${Math.max(0, left)}px, ${Math.max(0, top)}px)`;
  };
  hoverAt = (picked) => {
    pointer = picked.x >= 0 && picked.y >= 0 ? {x: picked.x, y: picked.y} : null;
    showHover(describe(picked), picked.x, picked.y);
  };
  document.getElementById('scene').addEventListener('pointerleave', () => {
    pointer = null;
    hideHover();
  });
  // The view moved under a resting cursor: pick again where it rests.
  let pickedView = '';
  setInterval(() => {
    if (!pointer) {
      return;
    }
    let viewport = null;
    try {
      viewport = deckgl.getViewports()[0];
    } catch (error) {
      return;
    }
    if (!viewport) {
      return;
    }
    const view = [viewport.longitude, viewport.latitude, viewport.zoom,
      viewport.pitch, viewport.bearing].map((v) => v.toFixed(6)).join();
    if (view === pickedView) {
      return;
    }
    pickedView = view;
    const picked = deckgl.pickObject({x: pointer.x, y: pointer.y, radius: 2});
    showHover(picked ? describe(picked) : null, pointer.x, pointer.y);
  }, HOVER_REPICK_MS);

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
    pinBox.innerHTML = tooltipHtml(info, feature, modelRows(layer.id, feature));
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

  // The tour's own messages (maintainer, 2026-10-17 and 2026-10-18).
  // The main ones, centered at about a quarter of the map's height and
  // above the legend: "Flying on autopilot" as it starts, then what is
  // about to switch, then what the visitor can now see. Beside them a
  // bubble above the button about to switch, pointing at it, with the
  // definition and the countdown 5 to 0. Both stay a while after
  // the switch, so a visitor learns what the controls do by watching;
  // the bubble's arrow flickers once as it appears.
  const banner = document.createElement('div');
  banner.id = 'autopilot';
  banner.hidden = true;
  document.body.appendChild(banner);
  const tip = document.createElement('div');
  tip.id = 'tour-tip';
  tip.hidden = true;
  document.body.appendChild(tip);
  let bannerTimers = [];
  // The button the bubble points at, or null.
  let tipAnchor = null;
  const controlFor = (target) => {
    if (!target) {
      return null;
    }
    const control = target === 'buildings'
      ? layerSwitch.querySelector('[data-layer="buildings"]')
      : viewSwitch.querySelector(`[data-choice="${target}"]`);
    // Folded controls (a phone): their chip stands in.
    return control && control.getClientRects().length > 0 ? control : restore;
  };
  const margin = 16;
  // A quarter of the way up the map, but never over the legend, the
  // bottom bar or the button bubble.
  const placeBanner = () => {
    const height = window.innerHeight;
    let floor = height * 0.25;
    for (const id of ['legend-box', 'bottom-bar']) {
      const element = document.getElementById(id);
      if (element && element.getClientRects().length) {
        floor = Math.max(floor, height - element.getBoundingClientRect().top + 12);
      }
    }
    if (!tip.hidden) {
      floor = Math.max(floor, height - tip.getBoundingClientRect().top + 12);
    }
    banner.style.left = `${Math.max(margin, (window.innerWidth - banner.offsetWidth) / 2)}px`;
    banner.style.bottom = `${floor}px`;
  };
  const placeTip = () => {
    const width = tip.offsetWidth;
    const rect = tipAnchor.getBoundingClientRect();
    const center = rect.left + rect.width / 2;
    const left = Math.min(Math.max(margin, center - width / 2), window.innerWidth - margin - width);
    tip.style.left = `${left}px`;
    tip.style.bottom = `${window.innerHeight - rect.top + 12}px`;
    tip.style.setProperty('--arrow-x', `${center - left}px`);
  };
  const placeAll = () => {
    if (!tip.hidden && tipAnchor) {
      placeTip();
    }
    if (!banner.hidden) {
      placeBanner();
    }
  };
  window.addEventListener('resize', placeAll);
  const hideHint = () => {
    bannerTimers.forEach(clearTimeout);
    bannerTimers = [];
    banner.hidden = true;
    tip.hidden = true;
  };
  const later = (ms, fn) => bannerTimers.push(setTimeout(fn, ms));
  // `countdown`: from that number, one a second, starting
  // `countdownAfter` seconds after the message appears; `linger`: how
  // long the message stays in all. `text` may be a list of lines, the
  // first one the message, the others smaller; a word in *asterisks*
  // is stressed.
  const showLines = (text, box = banner) => {
    box.replaceChildren(
      ...[].concat(text).map((words, i) => {
        const line = document.createElement('div');
        words.split(/\*([^*]+)\*/).forEach((part, j) => {
          if (j % 2) {
            const stressed = document.createElement('em');
            stressed.textContent = part;
            line.appendChild(stressed);
          } else if (part) {
            line.appendChild(document.createTextNode(part));
          }
        });
        if (i > 0) {
          line.className = 'tour-sub';
        }
        return line;
      }),
    );
  };
  // `text` goes in the main message, `tip` (the definition) and the
  // countdown in the bubble above `target`'s button; the bubble shows
  // only when it has something to say.
  const showHint = ({target, text, tip: definition, countdown, countdownAfter = 0, linger = 5}) => {
    hideHint();
    // Texts appear steadily; only the bubble's arrow flickers once,
    // pointing out the button (maintainer, 2026-10-18).
    banner.classList.remove('fading');
    tip.classList.remove('fading', 'flick');
    showLines(text);
    tipAnchor = controlFor(target);
    if (tipAnchor && (definition || countdown)) {
      showLines(definition || [], tip);
      if (countdown) {
        // 5 down to 1, and 0 at the switch; a second later the count goes,
        // and a bubble with nothing else to say goes with it, never
        // left empty.
        const count = document.createElement('div');
        count.className = 'tour-count';
        tip.appendChild(count);
        for (let n = countdown; n >= 0; n--) {
          later((countdownAfter + countdown - n) * 1000, () => {
            count.textContent = String(n);
            placeAll();
          });
        }
        later((countdownAfter + countdown + 1) * 1000, () => {
          if (definition) {
            count.remove();
            placeAll();
          } else {
            tip.hidden = true;
          }
        });
      }
      tip.hidden = false;
      void tip.offsetWidth;
      tip.classList.add('flick');
    }
    banner.hidden = false;
    placeAll();
    later(linger * 1000 - 1200, () => {
      banner.classList.add('fading');
      tip.classList.add('fading');
    });
    later(linger * 1000, hideHint);
  };
  // Buildings the tour switched off come back when the visitor takes
  // over; the coloring the tour chose stays.
  let tourHidBuildings = false;
  const tourAction = (action) => {
    // A page without the view a step names (no type view, no per-home
    // view) skips that step and its hint.
    const needs = action.view || (action.hint && action.hint.target);
    if (needs && needs !== 'buildings' && !viewSwitch.querySelector(`[data-choice="${needs}"]`)) {
      return;
    }
    if (action.hint) {
      showHint(action.hint);
    }
    if (action.view && action.view !== valueView) {
      setChoice('view', action.view);
    }
    if ('buildings' in action && 'buildings' in shown && shown.buildings !== action.buildings) {
      toggleLayer('buildings');
      tourHidBuildings = !action.buildings;
    }
  };
  const tourStopped = () => {
    banner.hidden = true;
    hideHint();
    if (tourHidBuildings && !shown.buildings) {
      toggleLayer('buildings');
    }
    tourHidBuildings = false;
  };
  const launchTour = (from = null) => startTour(
    deckgl,
    scene.view,
    scene.tour,
    document.getElementById('scene'),
    hasDwellingView ? scene.tour_contrast : null,
    tourAction,
    tourStopped,
    scene.tour_downtown,
    from,
  );
  let tourControl = tour ? launchTour() : {stop: () => {}, ready: () => {}};
  // The place's name at the top as the page opens, tour or not, for
  // about eight seconds (maintainer, 2026-10-18).
  if (scene.place_title) {
    const placeTitle = document.createElement('div');
    placeTitle.id = 'place-title';
    placeTitle.textContent = scene.place_title;
    document.body.appendChild(placeTitle);
    setTimeout(() => placeTitle.classList.add('fading'), 8000);
    setTimeout(() => placeTitle.remove(), 9500);
  }
  // The opening line says how to take over with what the visitor
  // holds: a mouse or a finger (maintainer, 2026-10-18).
  if (tour) {
    const touch = window.matchMedia('(pointer: coarse)').matches;
    showLines([
      'Flying on autopilot. Take control anytime.',
      touch
        ? 'Drag a finger to pan, pinch to zoom, twist two fingers to rotate.'
        : 'Drag the mouse to pan, scroll to zoom, right-drag to rotate.',
    ]);
    banner.hidden = false;
    placeBanner();
    later(9000, () => banner.classList.add('fading'));
    later(10200, hideHint);
  }

  // For measuring the page (frame rate, load time) from a test browser.
  window.openplacesScene = {deck: deckgl, loaded: false, stopTour: () => tourControl.stop()};

  // Whether the tour still runs; it stops on the visitor's first touch,
  // and the replay button appears (maintainer, 2026-10-18).
  let touring = tour;
  const replayButton = document.createElement('button');
  replayButton.type = 'button';
  replayButton.id = 'replay-chip';
  replayButton.className = 'panel';
  replayButton.dataset.replay = 'tour';
  replayButton.textContent = '▶ Replay tour';
  replayButton.title = 'Start the guided flight again';
  replayButton.hidden = tour || !(scene.tour && scene.tour.length);
  const watchForTakeover = () => {
    for (const type of ['pointerdown', 'wheel', 'touchstart', 'keydown']) {
      (type === 'keydown' ? window : document.getElementById('scene')).addEventListener(
        type,
        () => {
          touring = false;
          replayButton.hidden = false;
        },
        {once: true, passive: true},
      );
    }
  };
  if (tour) {
    watchForTakeover();
  }
  // A replay starts from where the visitor left the camera, with value
  // per ground area and every layer as the tour expects them.
  replayButton.addEventListener('click', () => {
    if (touring) {
      return;
    }
    replayButton.hidden = true;
    if (pin) {
      unpin();
    }
    if (valueView !== 'area') {
      setChoice('view', 'area');
    }
    if ('buildings' in shown && !shown.buildings) {
      toggleLayer('buildings');
    }
    const v = deckgl.getViewports()[0];
    tourControl = launchTour({
      longitude: v.longitude,
      latitude: v.latitude,
      zoom: v.zoom,
      pitch: v.pitch,
      bearing: v.bearing,
    });
    // Everything is loaded: the overview needs no wait beyond its own.
    tourControl.ready();
    touring = true;
    watchForTakeover();
  });
  buildPlaceNav(places, deckgl, () => touring, shown, () => ({
    view: valueView,
    values: withEstimates ? 'imputed' : 'assessed',
    range: colorRange,
  }));

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
          replayButton.hidden = false;
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
  bar.insertBefore(replayButton, shareButton);

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
      replayButton.hidden = false;
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
      range: colorRange,
    });
    if (shown.ghosts) {
      query.set('show', 'ghosts');
    }
    const hidden = Object.keys(shown).filter((name) => name !== 'ghosts' && !shown[name]);
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
    const zeroValue = palette.zero_value || palette.no_value;
    const colors = featureColors(values, spec, palette.no_value, zeroValue);
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
      const scale = {...boundsFor(other.unit, other.vmin, other.vmax), alpha: spec.alpha};
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
        } else if (otherValues[i] === 0) {
          // A recorded zero keeps its own color per dwelling too.
          colors.set(zeroValue, 4 * i);
        }
      }
    }
    // A palette may draw every edge (`line`) and blend its fills
    // additively (`blend`): light adds up, no depth is written, so the
    // result does not depend on draw order (see `Palette.blend`).
    // The type view: every feature in its type's color, fills and
    // value-colored edges alike, the same in every palette. Lots take
    // a light tint and buildings the full color, darker: in the very
    // same color a house vanished into its single-family lot
    // (Wilmington, 2026-10-19).
    const codes = entry.decoded.arrays.type_code;
    if (valueView === 'type' && codes) {
      const isLot = entry.spec.name === scene.layers[0].name;
      const tint = isLot ? TYPE_LOT_TINT : 0;
      const shade = isLot ? 1 : TYPE_BUILDING_SHADE;
      colors.set(typeColors(codes, scene.types, spec.alpha, tint, shade));
      if (spec.lineSpec && lineColors) {
        lineColors = typeColors(codes, scene.types, spec.line_alpha, tint, shade);
      }
    }
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
        updateTriggers: {getColor: [palette.id, valueView, withEstimates, colorRange]},
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
        updateTriggers: {getLineColor: [palette.id, valueView, withEstimates, colorRange]},
      })
      : null);
    const faces = new SolidPolygonLayer({
      ...(palette.material ? {material: palette.material} : {}),
      // Pushed back behind their own edges, by the same amount in every
      // layer. deck.gl's default offset pulls each later layer forward
      // (`-layerIndex * 100`); kept in here, it pulled the buildings'
      // faces in front of their wireframe, so Neon's building edges
      // came and went as the camera moved (maintainer, 2026-10-17).
      ...(outline ? {getPolygonOffset: () => [1, 6]} : {}),
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
        getFillColor: [palette.id, valueView, withEstimates, colorRange],
      },
      ...ghostFilter('solid'),
    });
    const out = outline ? [faces, outline] : [faces];
    // Values openplaces imputed (`estimated`), shown with "+ imputed":
    // a dotted line along the feature's top edge, more gap than dot
    // (maintainer, 2026-10-21), so a county with many imputed building
    // values does not read as if its assessor had valued them all. Not
    // a translucent fill: translucent walls blend with whatever was
    // drawn before them and showed blocks' back walls through their
    // fronts (Soden St, Cambridge, 2026-10-16).
    const estimated = entry.decoded.arrays.estimated;
    // Under a model's view the colors are the model's own, so the
    // assessor's gaps are not marked there.
    if (withEstimates && !isModel(valueView) && estimated && estimated.some((v) => v)) {
      const light = palette.theme === 'light';
      out.push(new PathLayer({
        id: `${entry.id}-imputed`,
        data: topOutlines(entry),
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT_OFFSETS,
        coordinateOrigin: entry.origin,
        positionFormat: 'XYZ',
        _pathType: 'loop',
        getColor: light ? IMPUTED_LINE_LIGHT : IMPUTED_LINE_DARK,
        getWidth: IMPUTED_LINE_WIDTH,
        widthUnits: 'pixels',
        getDashArray: IMPUTED_DASH,
        dashJustified: true,
        visible: shown[entry.spec.name],
        pickable: false,
        extensions: [
          new PathStyleExtension({dash: true}),
          new DataFilterExtension({filterSize: 1}),
        ],
        getFilterValue: (_, {index}) => estimated[entry.objectFeature[index]],
        filterRange: [1, 1],
        updateTriggers: {getColor: palette.id},
      }));
    }
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
          getFillColor: [palette.id, valueView, withEstimates, colorRange],
          getLineColor: [palette.id, valueView, withEstimates, colorRange],
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
  // Edges thin out and fade as the camera pulls back, so that from
  // afar the fill colors carry the map (maintainer, 2026-10-17): full
  // strength from zoom 15 in, 40% of the width and a quarter of the
  // opacity from zoom 12 out. The zoom is read in steps of a quarter,
  // so a zoom gesture restyles the edges a few times, not every frame.
  const EDGE_FULL_ZOOM = 15;
  const EDGE_FAR_ZOOM = 12;
  let edgeZoom = Math.round(scene.view.zoom * 4) / 4;
  const styleEdges = (layer) => {
    if (layer.id.endsWith('-imputed')) {
      // The imputed values' dotted edges fade out entirely as the
      // camera pulls back: from afar, thousands of them piled up into
      // dark blotches (maintainer, 2026-10-23), even at sub-pixel
      // widths, which WebGL still draws as at least a pixel.
      const t = Math.max(0, Math.min(1,
        (edgeZoom - IMPUTED_FAR_ZOOM) / (IMPUTED_FULL_ZOOM - IMPUTED_FAR_ZOOM)));
      return layer.clone({opacity: t, visible: layer.props.visible && t > 0});
    }
    if (!layer.id.endsWith('-edges')) {
      return layer;
    }
    const t = Math.max(0, Math.min(1,
      (edgeZoom - EDGE_FAR_ZOOM) / (EDGE_FULL_ZOOM - EDGE_FAR_ZOOM)));
    const style = {opacity: 0.25 + 0.75 * t};
    if (layer instanceof PathLayer) {
      style.widthScale = 0.4 + 0.6 * t;
    }
    return layer.clone(style);
  };
  fadeEdges = () => {
    let viewport = null;
    try {
      viewport = deckgl.getViewports()[0];
    } catch (error) {
      return;
    }
    const zoom = viewport ? Math.round(viewport.zoom * 4) / 4 : edgeZoom;
    if (zoom !== edgeZoom) {
      edgeZoom = zoom;
      redraw();
    }
  };

  const redraw = () => {
    deckgl.setProps({
      layers: [
        // Each entry is its faces and, under some palettes, its edges.
        ...layers.flatMap((l) => l.layer).map(styleEdges),
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
    // "Modeled", or a model from its menu: that model's view, per
    // ground area or per home as the other toggle stands. The toggle
    // keeps working under a model; another source returns to the
    // assessor's values the same way.
    const perHome = isPerHome(valueView);
    if (group === 'scale') {
      modelDollars = id === 'dollars';
      valueView = modelView(perHome);
      group = 'view';
    } else if (group === 'model' || (group === 'values' && id === 'model')) {
      modelKey = group === 'model' ? id : modelKey;
      valueView = modelView(perHome);
      group = 'view';
    } else if (group === 'view') {
      valueView = isModel(valueView) && id !== 'type' ? modelView(id === 'dwelling') : id;
    } else {
      if (isModel(valueView)) {
        valueView = perHome ? 'dwelling' : 'area';
      }
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
  // The ramp's bounds: the place's own spread (local, the default) or
  // the same for every place (nationwide), so places compare
  // (maintainer, 2026-10-18). Display only, like the palettes.
  const rangeSwitch = document.createElement('div');
  rangeSwitch.id = 'range-switch';
  rangeSwitch.setAttribute('role', 'radiogroup');
  rangeSwitch.setAttribute('aria-label', 'Color scale');
  const markRange = () => {
    for (const button of rangeSwitch.children) {
      const on = button.dataset.range === colorRange;
      button.classList.toggle('active', on);
      button.setAttribute('aria-checked', String(on));
    }
  };
  if (scene.national_bounds) {
    const ranges = [
      ['local', 'Local scale', "Colors spread over this place's own values"],
      ['national', 'Nationwide scale', 'The same colors for the same values in every place'],
    ];
    for (const [id, label, title] of ranges) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.range = id;
      button.textContent = label;
      button.title = title;
      button.setAttribute('role', 'radio');
      button.addEventListener('click', () => {
        colorRange = id;
        markRange();
        drawLegends();
        for (const entry of layers) {
          entry.layer = makeLayer(entry);
        }
        redraw();
      });
      rangeSwitch.appendChild(button);
    }
    markRange();
    legends.before(rangeSwitch);
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
  const firstUnit = (scene.tour_downtown && scene.tour_downtown.admin3_id) ||
    (scene.tour && scene.tour.length ? scene.tour[0].admin3_id : null);
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
  return lowCapacity();
}

// Little memory reported, or a touch screen of phone size.
function lowCapacity() {
  const memory = navigator.deviceMemory;
  const phone = window.matchMedia('(pointer: coarse)').matches && screen.width < 900;
  return (memory !== undefined && memory <= 4) || phone;
}

main().catch((error) => {
  setStatus(`Could not load the scene: ${error.message}`);
  console.error(error);
});
