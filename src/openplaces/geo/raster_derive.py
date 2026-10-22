"""
Derive annual boolean rasters from an imported land-cover product.

A land-cover time series arrives as one multi-band file (one band per
year), one file per year, or a single "year of loss" layer. The analyses
downstream want the same handful of derived layers from any of them: is
this pixel forest in year y; did it change between y-1 and y; is it
forest in y without ever having regrown since a start year; was it
deforested in any of the n years before y. Each of those is a small
array operation, and each was, until now, a notebook cell that wrote a
dated folder nobody could regenerate.

This module makes the derivation declarative. A raster dataset recipe
lists ``derived_layers``, each naming an ``op`` from :data:`OPS` and the
layer it reads; :func:`derive_layers` executes the list in order and
writes one compressed GeoTIFF per (layer, year). Ops are pure functions
on numpy arrays so they can be tested without files; the file plumbing
(:func:`read_year`, :func:`write_year`) is shared and tiny.

Layers are addressed by ``(layer, year)``. The source product is the
layer ``'source'``: for a multi-band file, band ``year - first_year + 1``;
for per-year files, the file for that year; for a year-of-change file,
the one band, from which the year is selected by value.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio

NODATA = 255

_PROFILE = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'count': 1,
    'nodata': NODATA,
    'compress': 'deflate',
    'zlevel': 6,
    'tiled': True,
    'blockxsize': 256,
    'blockysize': 256,
}


def band_equals(band: np.ndarray, value, valid: np.ndarray | None = None):
    """1 where *band* equals *value*, 0 elsewhere, nodata where invalid."""
    out = (band == value).astype('uint8')
    return _apply_nodata(out, valid)


def annual_transition(
    before: np.ndarray, after: np.ndarray, from_value: int, to_value: int
):
    """1 where a pixel went from *from_value* in year y-1 to *to_value* in y.

    ``from_value=1, to_value=0`` on a forest layer is deforestation in
    year y; ``from_value=0, to_value=1`` is gain. A pixel that is nodata
    in either year is nodata in the result.
    """
    valid = (before != NODATA) & (after != NODATA)
    out = ((before == from_value) & (after == to_value)).astype('uint8')
    return _apply_nodata(out, valid)


def cumulative_no_gain(previous_no_gain: np.ndarray, current: np.ndarray):
    """Forest in year y only if forest in y and never regrown since the start.

    Given the no-gain layer of y-1 and the plain forest layer of y, a pixel
    is 1 when it is forest now and was forest under the no-gain rule last
    year. A pixel that was non-forest last year stays 0 however the
    current year classifies it, so regrowth never re-enters. The first
    year's no-gain layer is the plain forest layer of that year.

    A pixel with no observation last year takes this year's value: a gap
    in the series is missing information about regrowth, not evidence of
    it, and carrying nodata forward would make every later year unknown.
    """
    known = previous_no_gain != NODATA
    out = np.where(
        known, (previous_no_gain == 1) & (current == 1), current == 1
    ).astype('uint8')
    return _apply_nodata(out, current != NODATA)


def any_in_years(layers: list[np.ndarray]):
    """1 where any of *layers* is 1; nodata only where every layer is nodata."""
    if not layers:
        raise ValueError('any_in_years needs at least one layer.')
    stack = np.stack(layers)
    valid = (stack != NODATA).any(axis=0)
    out = ((stack == 1) & (stack != NODATA)).any(axis=0).astype('uint8')
    return _apply_nodata(out, valid)


def and_not(a: np.ndarray, b: np.ndarray):
    """1 where *a* is 1 and *b* is not; nodata where *a* is nodata.

    Natural forest in year y minus natural forest continuously present
    since the start year is forest that was non-forest in some earlier
    year: the regrowing forest of the NASA LCLUC proposal's legend
    ("a natural forest that was classified as a non-forest land cover at
    any prior time period").
    """
    out = ((a == 1) & (b != 1)).astype('uint8')
    return _apply_nodata(out, a != NODATA)


def lossyear_equals(lossyear: np.ndarray, year: int, offset: int = 2000):
    """1 where a year-of-loss band records loss in *year* (value = year - offset)."""
    return band_equals(lossyear, year - offset)


def lossyear_to_cover(
    lossyear: np.ndarray, cover_start: np.ndarray, year: int, offset: int = 2000
):
    """Forest in *year* from a start-year cover mask minus every loss up to *year*.

    *cover_start* is the boolean cover at the offset year (Hansen's 2000
    tree cover thresholded by the recipe); a pixel is forest in *year* if
    it was covered at the start and its loss year is 0 (none) or later
    than *year*.
    """
    lost_by_year = (lossyear > 0) & (lossyear <= year - offset)
    valid = cover_start != NODATA
    out = ((cover_start == 1) & ~lost_by_year).astype('uint8')
    return _apply_nodata(out, valid)


def _apply_nodata(out: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    if valid is None:
        return out
    out = out.copy()
    out[~valid] = NODATA
    return out


def read_band(path: str | Path, band: int = 1) -> tuple[np.ndarray, dict]:
    """Read one band as uint8 with nodata mapped to :data:`NODATA`."""
    with rasterio.open(path) as src:
        arr = src.read(band)
        profile = src.profile
        nodata = src.nodata
    if nodata is not None and not np.isnan(nodata):
        arr = np.where(arr == nodata, NODATA, arr)
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.where(np.isnan(arr), NODATA, arr)
    return arr.astype('uint8'), profile


def write_layer(path: str | Path, array: np.ndarray, profile: dict, tags=None):
    """Write a uint8 array as a compressed, tiled GeoTIFF beside its source."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        k: profile[k] for k in ('crs', 'transform', 'width', 'height') if k in profile
    }
    out.update(_PROFILE)
    with rasterio.open(path, 'w', **out) as dst:
        dst.write(array.astype('uint8'), 1)
        if tags:
            dst.update_tags(**tags)
    return path


# One registered function per `op` in a recipe's `derived_layers`. Each
# receives the executor and the spec and writes every year it covers, so
# the year range of an op is its own business (a transition needs y-1, a
# prior-years union needs y-n).
OPS: dict[str, Callable] = {}


def _op(name):
    def decorator(fn):
        OPS[name] = fn
        return fn

    return decorator


class LayerStore:
    """Read and write (layer, year) rasters for one derivation run.

    Parameters
    ----------
    source : str or Path
        Path of the imported product: a multi-band file or a year-of-loss
        band. Per-year source files are given as a ``{year}`` template.
    path_fn : callable
        ``path_fn(layer, year) -> Path`` naming each derived raster. The
        Ingester passes a closure over ``get_output_path`` so derived
        layers land beside the recipe's own output under its naming;
        :func:`derive_layers` builds a plain ``{layer}_{year}.tif`` one
        from an output directory for callers outside the Ingester.
    first_year : int, optional
        Year of band 1 in a multi-band source.
    """

    def __init__(self, source, path_fn: Callable, first_year: int | None = None):
        self.source = str(source)
        self.path_fn = path_fn
        self.first_year = first_year
        self._profile = None

    def path(self, layer: str, year: int) -> Path:
        return Path(self.path_fn(layer, year))

    def read(self, layer: str, year: int) -> np.ndarray:
        if layer == 'source':
            if '{year}' in self.source:
                arr, profile = read_band(self.source.format(year=year))
            elif self.first_year is not None:
                arr, profile = read_band(self.source, band=year - self.first_year + 1)
            else:
                arr, profile = read_band(self.source)
        else:
            arr, profile = read_band(self.path(layer, year))
        if self._profile is None:
            self._profile = profile
        return arr

    def write(self, layer: str, year: int, array: np.ndarray, spec: dict) -> Path:
        if self._profile is None:
            raise RuntimeError('LayerStore.write called before any read.')
        tags = {'OPENPLACES_OP': spec['op'], 'OPENPLACES_LAYER': layer}
        return write_layer(self.path(layer, year), array, self._profile, tags)


def _years(spec: dict) -> range:
    start, end = spec['years']
    return range(int(start), int(end) + 1)


@_op('band_equals')
def _band_equals(store: LayerStore, spec: dict) -> list[Path]:
    """``{layer, op: band_equals, value, years: [a, b], from: source}``."""
    src = spec.get('from', 'source')
    return [
        store.write(
            spec['layer'],
            year,
            band_equals(
                store.read(src, year), spec['value'], store.read(src, year) != NODATA
            ),
            spec,
        )
        for year in _years(spec)
    ]


@_op('annual_transition')
def _annual_transition(store: LayerStore, spec: dict) -> list[Path]:
    """``{layer, op: annual_transition, from, before, after, years: [a, b]}``.

    Year a of the output needs year a-1 of the input; the range is the
    output's.
    """
    return [
        store.write(
            spec['layer'],
            year,
            annual_transition(
                store.read(spec['from'], year - 1),
                store.read(spec['from'], year),
                spec['before'],
                spec['after'],
            ),
            spec,
        )
        for year in _years(spec)
    ]


@_op('cumulative_no_gain')
def _cumulative_no_gain(store: LayerStore, spec: dict) -> list[Path]:
    """``{layer, op: cumulative_no_gain, from, years: [a, b]}``.

    Year a copies *from*; every later year carries the no-gain rule forward.
    """
    years = _years(spec)
    written = []
    previous = store.read(spec['from'], years[0])
    written.append(store.write(spec['layer'], years[0], previous, spec))
    for year in years[1:]:
        previous = cumulative_no_gain(previous, store.read(spec['from'], year))
        written.append(store.write(spec['layer'], year, previous, spec))
    return written


@_op('any_in_prior_years')
def _any_in_prior_years(store: LayerStore, spec: dict) -> list[Path]:
    """``{layer, op: any_in_prior_years, from, n_years, years: [a, b]}``.

    Output year y covers y-n_years .. y-1 of *from*; the notebook's
    ``prev_def_{y-3}_{y-1}_{y}``.
    """
    n = int(spec['n_years'])
    return [
        store.write(
            spec['layer'],
            year,
            any_in_years([store.read(spec['from'], y) for y in range(year - n, year)]),
            spec,
        )
        for year in _years(spec)
    ]


@_op('and_not')
def _and_not(store: LayerStore, spec: dict) -> list[Path]:
    """``{layer, op: and_not, from, minus, years: [a, b]}``."""
    return [
        store.write(
            spec['layer'],
            year,
            and_not(store.read(spec['from'], year), store.read(spec['minus'], year)),
            spec,
        )
        for year in _years(spec)
    ]


@_op('lossyear_equals')
def _lossyear_equals(store: LayerStore, spec: dict) -> list[Path]:
    """``{layer, op: lossyear_equals, years: [a, b], offset: 2000}``."""
    band = store.read('source', _years(spec)[0])
    return [
        store.write(
            spec['layer'],
            year,
            lossyear_equals(band, year, spec.get('offset', 2000)),
            spec,
        )
        for year in _years(spec)
    ]


@_op('lossyear_to_cover')
def _lossyear_to_cover(store: LayerStore, spec: dict) -> list[Path]:
    """``{layer, op: lossyear_to_cover, cover_start, years: [a, b], offset}``.

    *cover_start* is a path to the boolean start-year cover raster.
    """
    band = store.read('source', _years(spec)[0])
    cover, _ = read_band(spec['cover_start'])
    return [
        store.write(
            spec['layer'],
            year,
            lossyear_to_cover(band, cover, year, spec.get('offset', 2000)),
            spec,
        )
        for year in _years(spec)
    ]


def derive_layers(
    source,
    out_dir=None,
    specs: list[dict] | None = None,
    first_year: int | None = None,
    verbose=False,
    path_fn: Callable | None = None,
) -> dict[tuple[str, int], Path]:
    """Execute a recipe's ``derived_layers`` list against one source product.

    Parameters
    ----------
    source : str or Path
        Imported product (multi-band file, ``{year}`` template, or a
        year-of-loss band).
    out_dir : str or Path, optional
        Where the derived ``{layer}_{year}.tif`` files go. Ignored when
        *path_fn* is given.
    specs : list of dict
        The ``derived_layers`` entries, in order; a later entry may read
        an earlier one's layer.
    first_year : int, optional
        Year of band 1 in a multi-band source.
    verbose : bool, optional
        Print one line per layer written.
    path_fn : callable, optional
        ``path_fn(layer, year) -> Path``; overrides *out_dir* naming.

    Returns
    -------
    dict
        ``{(layer, year): path}`` for everything written.
    """
    if path_fn is None:
        if out_dir is None:
            raise ValueError('derive_layers needs out_dir or path_fn.')
        out = Path(out_dir)

        def path_fn(layer, year):
            return out / f'{layer}_{year}.tif'

    store = LayerStore(source, path_fn, first_year=first_year)
    written: dict[tuple[str, int], Path] = {}
    for spec in specs or []:
        op = spec.get('op')
        if op not in OPS:
            raise ValueError(
                f'derived_layers: unknown op {op!r}; expected one of {sorted(OPS)}.'
            )
        years = _years(spec)
        paths = OPS[op](store, spec)
        for year, path in zip(years, paths, strict=True):
            written[(spec['layer'], year)] = path
        if verbose:
            print(f'  derived_layers: {spec["layer"]} ({op}), {len(paths)} year(s)')
    return written
