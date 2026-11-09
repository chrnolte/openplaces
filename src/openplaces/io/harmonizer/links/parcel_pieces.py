"""Cut building outlines at parcel lines: one building per parcel piece.

A footprint is an outline as a source draws it, and a source draws a
townhome row, or a side-by-side duplex, as one outline. The building is
one structure (`core.schema.ENTITY_DEFINITIONS`), so the building
geospine cuts such an outline where the cadastre draws the lot lines
between its units, and each piece becomes a building of its own
(`split_outlines_at_parcel_lines`, plans/buildings-from-parcel-pieces.md).

**The rule** (that plan, section 3). The input is the footprint
geospine's parcel overlay as `load_links` restored it, after the
overlay's own trimming and chain snapping.

1. Candidates: outlines with two or more pairs the trimmed overlay
   labels 'multi-parcel footprint'. Never cut: a condo-cluster row (one
   building by construction), a row inferred from a parcel (one parcel),
   and an outline a chain snap moved to one parcel (it keeps one pair).
2. Pieces: the kept pairs in decreasing intersection area; piece k is
   the outline intersected with parcel k, less pieces 1 to k-1, because
   lot layers overlap. The parcel polygons are the ones the overlay was
   computed against (the reference rows and preparation `load_links`
   restored), so the parcel ids are the overlay's.
3. Fold: a piece that falls below either of the overlay's trimming
   thresholds after the subtraction, and the remainder (the outline
   less every kept piece: the part over a street or no parcel), join
   the dominant piece. The pieces partition the outline.
4. Refusal (decision D2 of the plan, 2026-11-06): a non-dominant piece
   stays a building of its own only when all three hold, and otherwise
   folds into the dominant piece:

   - a building or dwelling point (NSI, Overture) lies inside it;
   - its lot has an improvement value of at least the recipe's floor,
     or has no value and a structure flag the recipe names;
   - it is not round (compactness, 4 pi area over perimeter squared,
     at most the recipe's ceiling, in a local metric projection).

   A side-by-side duplex half is a building of its own (maintainer,
   2026-11-06). The rule was chosen against 190 labeled pieces over
   the plan's thirteen test units (its section 6.2); the floor and the
   ceiling live in the recipe, not here. A width test was measured and
   dropped. The rule errs toward fewer buildings: a piece with no
   evidence keeps today's single building.
5. Ids: every piece of a cut outline takes its centroid's code
   (`geo.ids.distinct_centroid_codes`), a collision broken by a finer
   cell and never by a sequence number (design limit L3). An uncut
   outline keeps `building_id == footprint_id`, so its links and
   enrichments stay valid.
6. Re-keying at the pair: every overlay row of a cut outline is moved
   to one piece, `(footprint_id, parcel_id) -> building_id`: a kept
   pair to its piece; a trimmed pair, a refused piece's lot and the
   off-parcel remainder to the dominant piece. No row is added or
   removed, so no link is lost. The areas of a moved row are measured
   again on its piece, and the labels of a cut outline's rows are
   rebuilt by the overlay's own trimming (`_build_crosswalk`). Rows of
   an outline no longer on the spine (removed by a later geometry step
   of the footprint geospine) keep their id.
7. Marker: `footprint_id` stays on every row; `building_geometry` says
   `outline` (uncut) or `parcel_piece`; `geometry_source` keeps the
   outline's token, which the delivery terms and the condo-cluster and
   inferred-row rules read.

**Why parcel lines and not another source's outlines.** Splitting an
outline into a later source's per-lot outlines was tried and removed
(`spine_resolution`): those outlines are often older drawings of the
same row. Measured 2026-10-16 on US-NC-CAR: Microsoft's global layer
matched 19.4% of the area such per-lot pieces covered, against 98.7%
of the outlines they replaced. Lot lines are the cadastre's own
statement of where one unit's land ends, and the roll describes each
lot, which is what the refusal rule reads.

**Where a lot's value comes from.** This step runs in the building
geospine, before the parcel geospine and parcel spine exist, so it
cannot read the parcel spine's `improvement_value` and `has_structure`.
The overlay's reference lots (the parcel ingest rows `load_links`
restored) exist, and so do the ingest-level parcel and property tables
the parcel geospine joins onto its lots. `lot_structure_evidence`
therefore runs the parcel geospine's own `link_by_id` passes (every
`auto_discover` pass, with that recipe's configs and the unit's
patches) on those lots, which is how the parcel spine gets the same two
columns. Nothing about the join is restated here. Where the parcel
geospine merges several parcel layers, its lots and these can differ
(the overlay links one layer unless `kept_by` names the parcel
geospine), and a sum the parcel geospine withholds from lots sharing a
key is withheld here over the overlay's lots.

**Why points are tested by containment, not through the point joins.**
The NSI and dwelling joins run after this step, on the buildings it
makes, so they cannot decide those buildings; and their proximity
passes attach a point outside a polygon to the nearest building, which
is no evidence of a structure on a piece. The rule was measured with
points inside the piece, and a plain containment test is that.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from openplaces.geo.ids import distinct_centroid_codes
from openplaces.geo.link import get_entity_link_path
from openplaces.geo.polygon import get_areas, local_metric_crs
from openplaces.io import to_parquet
from openplaces.io.harmonizer import HarmonizeState, _record_source, _register
from openplaces.io.harmonizer.links._shared import _LINK_METADATA_KEY
from openplaces.io.harmonizer.links.by_id import link_by_id
from openplaces.io.harmonizer.links.sidecars import (
    _fingerprint_safe_step,
    _link_fingerprint,
    _with_source_hashes,
)
from openplaces.io.harmonizer.links.spatial import (
    _build_crosswalk,
    _resolve_reference_recipe,
    reference_rows,
)
from openplaces.io.harmonizer.written import note_written
from openplaces.io.readers import get_entities
from openplaces.recipe import (
    apply_recipe_patches,
    get_recipe_by_id,
    get_recipe_id,
    source_id_from_recipe_id,
)

#: Column naming what a building's polygon is.
BUILDING_GEOMETRY_COLUMN = 'building_geometry'
#: `building_geometry` of an uncut outline.
OUTLINE = 'outline'
#: `building_geometry` of a piece of an outline cut at the parcel lines.
PARCEL_PIECE = 'parcel_piece'

#: The overlay label of a pair kept on an outline over several parcels.
MULTI_PARCEL_LINK = 'multi-parcel footprint'

#: `geometry_source` tokens (before the first '.') never cut: a
#: condo-cluster row is one building by construction, a row inferred
#: from a parcel stands on that one parcel.
NEVER_CUT_TOKENS = frozenset({'condo_cluster', 'parcel'})

#: The lot columns the refusal rule reads.
IMPROVEMENT_VALUE_COLUMN = 'improvement_value'
STRUCTURE_FLAG_COLUMN = 'has_structure'

#: Footer key naming the geospine whose overlay the cut sidecar re-keys.
CUT_FROM_KEY = 'openplaces:link_cut_from'

_KEEP_PIECE_KEYS = ('improvement_value_min', 'structure_flag_values', 'compactness_max')


def parse_keep_piece(spec: dict | None) -> dict:
    """Validate the refusal rule's thresholds as the recipe states them.

    Parameters
    ----------
    spec : dict
        ``improvement_value_min`` (number: a lot's improvement value at
        or above it counts as a structure), ``structure_flag_values``
        (list: values of the lot's structure flag that count as a
        structure where the lot has no improvement value; compared as
        upper-case text, so a boolean True reads 'TRUE') and
        ``compactness_max`` (number in (0, 1]: a piece rounder than this
        is refused).

    Returns
    -------
    dict
        The three thresholds, typed.

    Raises
    ------
    ValueError
        When a key is missing or unknown, or a threshold is out of
        range. The thresholds have no defaults: they are the
        maintainer's decision and belong in the recipe.
    """
    spec = dict(spec or {})
    unknown = set(spec) - set(_KEEP_PIECE_KEYS)
    missing = [k for k in _KEEP_PIECE_KEYS if k not in spec]
    if unknown or missing:
        raise ValueError(
            'split_outlines_at_parcel_lines: keep_piece needs exactly '
            f'{list(_KEEP_PIECE_KEYS)}; unknown {sorted(unknown)}, missing {missing}.'
        )
    compactness_max = float(spec['compactness_max'])
    if not 0 < compactness_max <= 1:
        raise ValueError(
            f'keep_piece: compactness_max must lie in (0, 1], got {compactness_max}.'
        )
    return {
        'improvement_value_min': float(spec['improvement_value_min']),
        'structure_flag_values': [
            str(v).strip().upper() for v in spec['structure_flag_values'] or []
        ],
        'compactness_max': compactness_max,
    }


@dataclass
class SplitContext:
    """What the split step reads from the recipes around it.

    Parameters
    ----------
    links_recipe : dict
        The geospine whose parcel overlay `load_links` restored (the
        footprint geospine), with the unit's patches applied.
    links_key : str
        Column of this spine naming the row's id in that geospine.
    overlay_step : dict
        That geospine's overlay step for the entity type.
    min_fraction : float
        The overlay's `min_fraction_of_largest`.
    area_min_m2 : float
        The overlay's `area_intersection_m2_min`.
    values_recipe : dict or None
        The recipe whose `link_by_id` passes give a lot its values (the
        parcel geospine), patched; None when the step names none.
    fingerprint_config : dict
        The step configuration the cut sidecar's fingerprint records.
    """

    links_recipe: dict
    links_key: str
    overlay_step: dict
    min_fraction: float
    area_min_m2: float
    values_recipe: dict | None
    fingerprint_config: dict


def _as_run_for(recipe: dict, admin_id) -> dict:
    """*recipe* with the unit's patches applied, as its run saw it."""
    if admin_id is None:
        return recipe
    patched, _ = apply_recipe_patches(recipe, admin_id)
    return patched


def _lot_value_steps(values_recipe: dict | None) -> tuple[dict | None, list[dict]]:
    """The `resolve_spine` step and the `auto_discover` `link_by_id`
    passes of *values_recipe*, in pipeline order."""
    if values_recipe is None:
        return None, []
    resolve = None
    passes = []
    for step in values_recipe.get('pipeline') or []:
        if not isinstance(step, dict):
            continue
        if step.get('step') == 'resolve_spine' and resolve is None:
            resolve = step
        elif step.get('step') == 'link_by_id' and step.get('auto_discover'):
            passes.append(step)
    return resolve, passes


def split_context(
    recipe: dict, step_index: int | None, step_cfg: dict, admin_id
) -> SplitContext:
    """Locate the overlay and the value passes a split step reads.

    The overlay is the one the last `load_links` step before
    *step_index* restored: that step's geospine, and in it the
    `spatial_overlay` link to the split's entity type. Its thresholds
    are read from there, never repeated in the building recipe.

    Raises
    ------
    ValueError
        When no `load_links` step precedes the split or its geospine
        has no overlay link to the entity type.
    """
    entity_type = step_cfg.get('entity_type', 'parcel')
    pipeline = recipe.get('pipeline') or []
    upto = len(pipeline) if step_index is None else step_index
    load_cfg = next(
        (
            s
            for s in reversed(pipeline[:upto])
            if isinstance(s, dict) and s.get('step') == 'load_links'
        ),
        None,
    )
    if load_cfg is None:
        raise ValueError(
            'split_outlines_at_parcel_lines needs a load_links step before it: '
            'the cut reads the overlay that step restores.'
        )
    links_recipe = _as_run_for(get_recipe_by_id(load_cfg['links_recipe_id']), admin_id)
    links_key = load_cfg.get('links_key')
    if links_key is None:
        entity = links_recipe.get('entity')
        links_key = f'{entity.entity_type}_id' if entity is not None else 'entity_id'
    overlay_step = None
    for other in links_recipe.get('pipeline') or []:
        if not isinstance(other, dict) or other.get('step') != 'link_to_reference':
            continue
        if other.get('join', 'spatial_overlay') != 'spatial_overlay':
            continue
        if other.get('save_link') is False:
            continue
        # The declared type, or the one a named recipe's id carries; no
        # discovery, so the match does not depend on what is built.
        other_type = other.get('entity_type')
        if other_type is None and other.get('recipe_id'):
            _, other_type = _resolve_reference_recipe(
                other['recipe_id'], None, admin_id
            )
        if other_type == entity_type:
            overlay_step = other
            break
    if overlay_step is None:
        raise ValueError(
            f'split_outlines_at_parcel_lines: {get_recipe_id(links_recipe)} has '
            f'no saved spatial_overlay link to {entity_type!r}.'
        )
    thresholds = overlay_step.get('thresholds') or {}
    values_id = step_cfg.get('lot_values_from')
    values_recipe = (
        _as_run_for(get_recipe_by_id(values_id), admin_id) if values_id else None
    )
    resolve, passes = _lot_value_steps(values_recipe)
    own = {k: v for k, v in step_cfg.items() if k not in ('step', 'checkpoint')}
    fingerprint_config = {
        'step': _fingerprint_safe_step(own),
        'overlay': _fingerprint_safe_step(overlay_step),
        'lot_values': {
            'resolve_spine': (
                None
                if resolve is None
                else {
                    'keep_columns': resolve.get('keep_columns'),
                    'track_provenance': resolve.get('track_provenance'),
                }
            ),
            'passes': [_fingerprint_safe_step(p) for p in passes],
        },
    }
    return SplitContext(
        links_recipe=links_recipe,
        links_key=links_key,
        overlay_step=overlay_step,
        min_fraction=float(thresholds.get('min_fraction_of_largest', 1 / 6)),
        area_min_m2=float(thresholds.get('area_intersection_m2_min', 10)),
        values_recipe=values_recipe,
        fingerprint_config=fingerprint_config,
    )


def candidate_pairs(
    flat: pd.DataFrame, spine: pd.DataFrame, links_key: str
) -> pd.Series:
    """Outlines to cut and their kept parcels, largest overlap first.

    Parameters
    ----------
    flat : DataFrame
        The overlay sidecar's rows: *links_key*, `parcel_id`,
        `area_intersection_m2` and the `link` label.
    spine : DataFrame
        The spine, carrying *links_key* and `geometry_source`.
    links_key : str
        The outline id column.

    Returns
    -------
    pandas.Series
        Per outline id, the list of its parcel ids labeled
        'multi-parcel footprint', ordered by intersection area (largest
        first, ties by parcel id), for outlines with two or more such
        parcels that are on the spine once and are not of a never-cut
        source.
    """
    multi = flat[(flat['link'] == MULTI_PARCEL_LINK) & flat['parcel_id'].notna()]
    multi = multi[[links_key, 'parcel_id', 'area_intersection_m2']]
    counts = multi.groupby(links_key).size()
    ids = set(counts.index[counts >= 2])
    keys = spine[links_key]
    on_spine = keys[keys.notna()]
    once = on_spine[~on_spine.duplicated(keep=False)]
    ids &= set(once)
    if 'geometry_source' in spine.columns:
        token = spine['geometry_source'].astype('string').str.split('.').str[0]
        never = set(keys[token.isin(NEVER_CUT_TOKENS).fillna(False).to_numpy()])
        ids -= never
    multi = multi[multi[links_key].isin(ids)]
    multi = multi.sort_values(
        [links_key, 'area_intersection_m2', 'parcel_id'],
        ascending=[True, False, True],
        kind='stable',
    )
    return multi.groupby(links_key, sort=True)['parcel_id'].agg(list)


def _polygonal(geom):
    """The polygonal part of *geom* (an intersection can add lines)."""
    if geom is None or geom.is_empty:
        return shapely.Polygon()
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        return geom
    parts = [
        g for g in shapely.get_parts(geom) if g.geom_type in ('Polygon', 'MultiPolygon')
    ]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def _valid(geom):
    return geom if geom.is_valid else _polygonal(shapely.make_valid(geom))


def _areas_m2(geoms, crs) -> np.ndarray:
    """Areas in m2, measured as the overlay measures them (`get_areas`)."""
    if len(geoms) == 0:
        return np.zeros(0)
    frame = gpd.GeoDataFrame(geometry=list(geoms), crs=crs)
    return get_areas(frame, 'm2').to_numpy()


def cut_outlines(
    outlines: gpd.GeoSeries,
    pairs: pd.Series,
    parcels: gpd.GeoSeries,
    min_fraction: float,
    area_min_m2: float,
) -> gpd.GeoDataFrame:
    """Cut each outline at its kept parcels and fold what falls short.

    Steps 2 and 3 of the rule (module docstring). For every outline in
    *pairs*: piece k is the outline on parcel k less the pieces before
    it; a piece under *area_min_m2* or under *min_fraction* of the
    largest piece folds into the dominant piece (rank 0), which also
    takes the remainder, so the pieces of an outline partition it.

    Parameters
    ----------
    outlines : GeoSeries
        Outline geometry by outline id.
    pairs : Series
        Per outline id, its kept parcel ids, largest overlap first
        (`candidate_pairs`).
    parcels : GeoSeries
        Parcel geometry by parcel id, in the outlines' CRS. A parcel
        missing here contributes no piece.
    min_fraction, area_min_m2 : float
        The overlay's trimming thresholds.

    Returns
    -------
    GeoDataFrame
        One row per piece: `outline_id`, `rank` (0 the dominant piece),
        `parcel_id` and the geometry, in the outlines' CRS.
    """
    crs = outlines.crs
    raw = []
    for outline_id, parcel_ids in pairs.items():
        outline = _valid(outlines.at[outline_id])
        taken = None
        for parcel_id in parcel_ids:
            parcel = parcels.get(parcel_id)
            if parcel is None or parcel.is_empty:
                continue
            piece = _polygonal(shapely.intersection(outline, _valid(parcel)))
            if taken is not None and not piece.is_empty:
                piece = _polygonal(shapely.difference(piece, taken))
            if piece.is_empty:
                continue
            raw.append((outline_id, parcel_id, piece))
            taken = piece if taken is None else shapely.union(taken, piece)
    columns = ['outline_id', 'rank', 'parcel_id']
    if not raw:
        return gpd.GeoDataFrame(
            columns=[*columns, 'geometry'], geometry='geometry', crs=crs
        )
    areas = _areas_m2([r[2] for r in raw], crs)
    rows = []
    start = 0
    while start < len(raw):
        outline_id = raw[start][0]
        stop = start
        while stop < len(raw) and raw[stop][0] == outline_id:
            stop += 1
        largest = areas[start:stop].max()
        kept = [
            (raw[i][1], raw[i][2])
            for i in range(start + 1, stop)
            if areas[i] >= area_min_m2 and areas[i] >= min_fraction * largest
        ]
        outline = _valid(outlines.at[outline_id])
        dominant = _fold_into_dominant(outline, [g for _, g in kept])
        rows.append((outline_id, 0, raw[start][1], dominant))
        rows.extend(
            (outline_id, rank, parcel_id, geom)
            for rank, (parcel_id, geom) in enumerate(kept, start=1)
        )
        start = stop
    frame = pd.DataFrame([r[:3] for r in rows], columns=columns)
    return gpd.GeoDataFrame(frame, geometry=[r[3] for r in rows], crs=crs)


def _fold_into_dominant(outline, others):
    """The dominant piece: the outline less every other kept piece."""
    if not others:
        return outline
    return _polygonal(shapely.difference(outline, shapely.union_all(others)))


def compactness(geoms: gpd.GeoSeries) -> pd.Series:
    """4 pi area over perimeter squared, in a local metric projection.

    1 for a circle, about 0.785 for a square, lower for a strip.
    """
    if len(geoms) == 0:
        return pd.Series(dtype=float, index=geoms.index)
    metric = geoms.to_crs(local_metric_crs(geoms))
    perimeter = metric.length
    return (4 * np.pi * metric.area / perimeter.pow(2)).where(perimeter > 0)


def points_inside(pieces: gpd.GeoSeries, points: list[gpd.GeoSeries]) -> pd.Series:
    """Whether any of *points* lies inside each piece (containment)."""
    found = pd.Series(False, index=pieces.index)
    if len(pieces) == 0:
        return found
    for layer in points:
        if layer is None or len(layer) == 0:
            continue
        layer = layer[layer.notna()]
        if layer.crs is not None and pieces.crs is not None:
            layer = layer.to_crs(pieces.crs)
        tree = shapely.STRtree(layer.to_numpy())
        hits, _ = tree.query(pieces.to_numpy(), predicate='contains')
        found.iloc[np.unique(hits)] = True
    return found


def keep_pieces(evidence: pd.DataFrame, rule: dict) -> pd.Series:
    """The refusal rule (step 4): which non-dominant pieces stay.

    Parameters
    ----------
    evidence : DataFrame
        Per piece: `has_point` (bool), `improvement_value` (number, NaN
        where the lot has none), `structure_flag` (the lot's flag, any
        type) and `compactness`.
    rule : dict
        From `parse_keep_piece`.

    Returns
    -------
    pandas.Series
        True where the piece is kept as a building of its own.
    """
    value = pd.to_numeric(evidence['improvement_value'], errors='coerce')
    flag = (
        evidence['structure_flag']
        .astype('string')
        .str.strip()
        .str.upper()
        .isin(rule['structure_flag_values'])
        .fillna(False)
    )
    structure = value.ge(rule['improvement_value_min']).fillna(False) | (
        value.isna() & flag
    )
    round_ = evidence['compactness'].gt(rule['compactness_max']).fillna(False)
    return (
        evidence['has_point'].fillna(False).astype(bool) & structure & ~round_
    ).astype(bool)


def apply_refusals(
    pieces: gpd.GeoDataFrame, outlines: gpd.GeoSeries, keep: pd.Series
) -> gpd.GeoDataFrame:
    """Fold refused pieces into the dominant piece.

    *keep* is True for the non-dominant pieces that stay (indexed like
    *pieces*; the dominant pieces are always kept). The dominant piece
    is the outline less the kept others, so the partition holds. An
    outline left with its dominant piece alone is dropped from the
    result: it is not cut.
    """
    stays = (pieces['rank'] == 0) | keep.reindex(pieces.index, fill_value=False)
    pieces = pieces[stays.to_numpy()].copy()
    n_pieces = pieces.groupby('outline_id')['rank'].transform('size')
    pieces = pieces[(n_pieces > 1).to_numpy()]
    if pieces.empty:
        return pieces
    for outline_id, group in pieces.groupby('outline_id', sort=False):
        others = group.loc[group['rank'] > 0, 'geometry'].tolist()
        dominant_index = group.index[group['rank'] == 0][0]
        pieces.at[dominant_index, 'geometry'] = _fold_into_dominant(
            _valid(outlines.at[outline_id]), others
        )
    # Ranks stay as cut: a refused piece leaves a gap, which is harmless
    # and keeps a piece's rank tied to its parcel's place in the cut.
    return pieces


def mint_piece_ids(pieces: gpd.GeoDataFrame, taken: set) -> pd.Series:
    """Ids for every piece of a cut outline (step 5), by centroid code."""
    if pieces.empty:
        return pd.Series(dtype=object, index=pieces.index)
    return distinct_centroid_codes(
        gpd.GeoDataFrame(geometry=pieces.geometry, crs=pieces.crs), taken
    )


def split_spine(
    spine: gpd.GeoDataFrame,
    pieces: gpd.GeoDataFrame,
    links_key: str,
) -> gpd.GeoDataFrame:
    """Replace each cut outline's row by one row per piece.

    *pieces* carries `outline_id`, `rank`, `building_id` and geometry.
    A piece copies its outline's columns (provenance, geographic ids,
    `envelope_id`), takes its own geometry and id, and is marked
    `parcel_piece`; every other row is marked `outline`. The rows keep
    the spine's order, each outline's pieces in rank order where the
    outline stood.
    """
    spine = spine.copy()
    spine[BUILDING_GEOMETRY_COLUMN] = OUTLINE
    if pieces.empty:
        return spine
    index_name = spine.index.name
    position = pd.Series(np.arange(len(spine)), index=spine[links_key].to_numpy())
    position = position[~position.index.duplicated()]
    order = pd.Series(np.arange(len(spine)), index=spine.index, dtype=float)
    cut = spine[links_key].isin(set(pieces['outline_id'])).to_numpy()
    sources = spine.iloc[position.loc[pieces['outline_id']].to_numpy()].copy()
    sources.index = pd.Index(pieces['building_id'].to_numpy(), name=index_name)
    sources[spine.geometry.name] = gpd.GeoSeries(
        pieces.geometry.to_crs(spine.crs).to_numpy(), index=sources.index, crs=spine.crs
    )
    sources[BUILDING_GEOMETRY_COLUMN] = PARCEL_PIECE
    piece_order = pd.Series(
        position.loc[pieces['outline_id']].to_numpy()
        + pieces['rank'].to_numpy() / 1000.0,
        index=sources.index,
    )
    out = pd.concat([spine[~cut], sources])
    out_order = pd.concat([order[~cut], piece_order])
    out = out.iloc[np.argsort(out_order.to_numpy(), kind='stable')]
    return gpd.GeoDataFrame(out, geometry=spine.geometry.name, crs=spine.crs)


def rekey_at_pairs(
    flat: pd.DataFrame,
    pieces: gpd.GeoDataFrame,
    parcels: gpd.GeoSeries,
    links_key: str,
    spine_id_col: str,
    mapping: pd.Series,
    min_fraction: float,
    area_min_m2: float,
) -> pd.DataFrame:
    """The overlay re-keyed at the pair (step 6).

    Parameters
    ----------
    flat : DataFrame
        The other geospine's overlay sidecar: *links_key*, `parcel_id`,
        the area columns and the `link` label (`link_chain` where the
        overlay snapped chains).
    pieces : GeoDataFrame
        The kept pieces of every cut outline: `outline_id`, `rank`,
        `parcel_id`, `building_id`, geometry.
    parcels : GeoSeries
        Parcel geometry by parcel id, in the pieces' CRS.
    links_key, spine_id_col : str
        The outline id column and this spine's id column.
    mapping : Series
        Outline id to building id for the uncut rows on the spine; an
        id it lacks keeps its value.
    min_fraction, area_min_m2 : float
        The overlay's trimming thresholds, for the cut outlines' labels.

    Returns
    -------
    DataFrame
        The same rows in the same order, *spine_id_col* first holding
        the building each pair now belongs to, *links_key* last.
    """
    flat = flat.reset_index(drop=True).copy()
    outline = flat[links_key]
    building = outline.map(mapping)
    building = building.where(building.notna(), outline).astype(object)
    if not pieces.empty:
        cut = outline.isin(set(pieces['outline_id']))
        dominant = pieces[pieces['rank'] == 0].set_index('outline_id')['building_id']
        own = pieces.loc[pieces['rank'] > 0, ['outline_id', 'parcel_id', 'building_id']]
        rows = pd.DataFrame(
            {
                'outline_id': outline[cut].to_numpy(),
                'parcel_id': flat.loc[cut, 'parcel_id'].to_numpy(),
            }
        )
        # A kept pair finds its piece; every other row of the outline
        # (its dominant lot, a trimmed lot, a refused piece's lot, the
        # off-parcel remainder) goes to the dominant piece.
        found = rows.merge(own, on=['outline_id', 'parcel_id'], how='left')
        moved = pd.Series(found['building_id'].to_numpy(), index=flat.index[cut])
        moved = moved.where(moved.notna(), outline[cut].map(dominant))
        building[cut] = moved
        flat = _measure_moved_rows(flat, cut, moved, pieces, parcels)
        flat = _relabel_moved_rows(
            flat, cut, moved, spine_id_col, min_fraction, area_min_m2
        )
    flat.insert(0, spine_id_col, building.to_numpy())
    columns = [c for c in flat.columns if c != links_key] + [links_key]
    return flat[columns]


def _measure_moved_rows(flat, cut, moved, pieces, parcels):
    """Areas of the moved rows, measured on their piece."""
    geometry = pieces.set_index('building_id').geometry
    piece_geoms = geometry.reindex(moved.to_numpy()).to_numpy()
    piece_areas = _areas_m2(piece_geoms, pieces.crs)
    if 'area_spine_m2' in flat.columns:
        flat.loc[cut, 'area_spine_m2'] = piece_areas
    parcel_ids = flat.loc[cut, 'parcel_id']
    parcel_geoms = parcels.reindex(parcel_ids.to_numpy()).to_numpy()
    has_parcel = np.array([g is not None and not g.is_empty for g in parcel_geoms])
    if 'area_intersection_m2' in flat.columns and has_parcel.any():
        inter = np.full(len(piece_geoms), np.nan)
        geoms = [
            _polygonal(shapely.intersection(p, _valid(q)))
            for p, q in zip(piece_geoms[has_parcel], parcel_geoms[has_parcel])
        ]
        inter[has_parcel] = _areas_m2(geoms, pieces.crs)
        rows = flat.index[cut]
        flat.loc[rows[has_parcel], 'area_intersection_m2'] = inter[has_parcel]
        if 'iou' in flat.columns and 'area_ref_m2' in flat.columns:
            ref_area = pd.to_numeric(flat.loc[rows, 'area_ref_m2'], errors='coerce')
            union = piece_areas + ref_area.to_numpy() - inter
            iou = np.where(union > 0, inter / np.where(union > 0, union, 1), np.nan)
            flat.loc[rows[has_parcel], 'iou'] = iou[has_parcel]
    return flat


def _relabel_moved_rows(flat, cut, moved, spine_id_col, min_fraction, area_min_m2):
    """Labels of the moved rows, rebuilt by the overlay's own trimming."""
    if 'link' not in flat.columns:
        return flat
    rows = flat.loc[cut].drop(columns=['link', 'link_chain'], errors='ignore')
    rows = rows.assign(**{spine_id_col: moved.to_numpy()})
    keep = [c for c in rows.columns if c in ('area_intersection_m2', 'iou')]
    indexed = rows.set_index([spine_id_col, 'parcel_id'])[keep]
    # _build_crosswalk drops rows equal in every column as copies of one
    # pair, comparing the columns and not the index; two pieces of
    # equal size on lots of equal size would compare equal. The pair
    # itself, as a column, keeps them apart.
    indexed = indexed.assign(
        _pair=[f'{b}|{p}' for b, p in zip(rows[spine_id_col], rows['parcel_id'])]
    )
    crosswalk = _build_crosswalk(indexed, spine_id_col, min_fraction, area_min_m2)
    labels = crosswalk.reset_index()[[spine_id_col, 'parcel_id', 'link']]
    # A merge, not a reindex: the off-parcel remainder's parcel id is
    # missing, and a merge pairs missing keys with each other.
    found = (
        pd.DataFrame(
            {
                spine_id_col: moved.to_numpy(),
                'parcel_id': flat.loc[cut, 'parcel_id'].to_numpy(),
            }
        )
        .merge(labels, on=[spine_id_col, 'parcel_id'], how='left')['link']
        .to_numpy()
    )
    flat['link'] = flat['link'].astype(object)
    flat.loc[cut, 'link'] = found
    return flat


def lot_structure_evidence(
    state: HarmonizeState, lots: pd.DataFrame, reference_id: str, values_recipe: dict
) -> pd.DataFrame:
    """A lot's improvement value and structure flag, as the parcel spine
    states them, before the parcel geospine exists.

    Runs every `auto_discover` `link_by_id` pass of *values_recipe* (the
    parcel geospine), in its order and with its configs, on *lots*: the
    overlay's reference rows, one per lot id (the first, as
    `resolve_spine` keeps it), reduced to the columns that recipe's
    `resolve_spine` keeps (each row's `geometry_source` is the
    reference's source, tracked columns carry it in their `_source`).
    See the module docstring for why.

    The rows are the raw reference, not the prepared one the overlay
    attributes from: preparing sums a lot's rows per id, and a sum of
    nothing is 0, so a lot whose row states no value would start at 0
    and a later pass could no longer fill it. Measured 2026-11-08 on
    US-WI-VI: starting from the prepared reference left 146 lots at 0
    that the parcel spine values at 5,000 or more.

    Parameters
    ----------
    state : HarmonizeState
        The run (its admin unit).
    lots : DataFrame
        The reference rows, with their lot id in `geo_id`.
    reference_id : str
        The reference recipe the lots came from.
    values_recipe : dict
        The recipe whose passes are run, patched for the unit.

    Returns
    -------
    DataFrame
        `improvement_value` and `has_structure` per lot id (missing
        where no source states one).
    """
    resolve, passes = _lot_value_steps(values_recipe)
    keep_columns = list((resolve or {}).get('keep_columns') or [])
    tracked = list((resolve or {}).get('track_provenance') or [])
    lots = lots[lots['geo_id'].notna()].drop_duplicates('geo_id')
    lots = pd.DataFrame(lots).set_index('geo_id')
    lots.index.name = 'parcel_id'
    spine = pd.DataFrame(lots[[c for c in keep_columns if c in lots.columns]]).copy()
    label = source_id_from_recipe_id(reference_id)
    spine['geometry_source'] = label
    for column in tracked:
        if column in spine.columns:
            _record_source(spine, column, spine[column].notna(), label)
    shim = HarmonizeState(
        recipe=values_recipe,
        admin_id=state.admin_id,
        verbose=False,
        timer=None,
        spine=spine,
    )
    shim.metadata['spine_source_recipe_ids'] = {reference_id}
    shim.metadata['spine_keep_columns'] = set(keep_columns)
    with warnings.catch_warnings():
        # The parcel geospine raises these same warnings for the unit.
        warnings.simplefilter('ignore')
        for step in passes:
            params = {k: v for k, v in step.items() if k not in ('step', 'checkpoint')}
            shim = link_by_id(shim, **params)
    return shim.spine.reindex(columns=[IMPROVEMENT_VALUE_COLUMN, STRUCTURE_FLAG_COLUMN])


def _load_points(specs, admin_id) -> list[gpd.GeoSeries]:
    """The point layers the rule tests, geometry only."""
    layers = []
    for spec in specs or []:
        recipe_id = spec.get('recipe_id') if isinstance(spec, dict) else spec
        if not recipe_id:
            continue
        frame = get_entities(recipe_id, admin_id, geom=True, missing='warn')
        if frame is None or len(frame) == 0 or 'geometry' not in frame:
            continue
        layers.append(frame.geometry)
    return layers


def cut_sidecar_path(recipe: dict, reference_id: str, admin_id):
    """Where the cut overlay is written: beside this geospine's output,
    the path the curate readers resolve for the building's parcels."""
    return get_entity_link_path(get_recipe_id(recipe), reference_id, admin_id)


def _write_cut_sidecar(path, flat, fingerprint, cut_from, verbose=False) -> None:
    to_parquet(
        flat,
        path,
        file_metadata={
            _LINK_METADATA_KEY: json.dumps(_with_source_hashes(fingerprint)),
            CUT_FROM_KEY: cut_from,
        },
        index=False,
    )
    note_written(path)
    if verbose:
        print(f'  Split: wrote link sidecar {path.name}')


def split_overlay(
    spine: gpd.GeoDataFrame,
    flat: pd.DataFrame,
    parcels: gpd.GeoSeries,
    context: SplitContext,
    evidence_for,
    rule: dict,
    taken: set,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict]:
    """The cut, without reading or writing anything.

    Parameters
    ----------
    spine : GeoDataFrame
        This geospine's rows, carrying `context.links_key`.
    flat : DataFrame
        The other geospine's overlay sidecar rows.
    parcels : GeoSeries
        Parcel geometry by parcel id.
    context : SplitContext
        Thresholds and key names.
    evidence_for : callable
        Given the non-dominant pieces (GeoDataFrame with `outline_id`,
        `rank`, `parcel_id`), returns the evidence frame
        `keep_pieces` reads, indexed like them.
    rule : dict
        From `parse_keep_piece`.
    taken : set
        Ids a new piece id must not take.

    Returns
    -------
    tuple
        The new spine, the re-keyed overlay rows, and counts (outlines
        cut, buildings minted, pieces folded by the thresholds and
        refused by the rule).
    """
    links_key = context.links_key
    spine_id_col = spine.index.name
    keys = spine[links_key]
    mapping = pd.Series(spine.index[keys.notna().to_numpy()], index=keys.dropna())
    mapping = mapping[~mapping.index.duplicated(keep=False)]
    pairs = candidate_pairs(flat, spine, links_key)
    outlines = gpd.GeoSeries(
        spine.geometry.to_numpy(), index=keys.to_numpy(), crs=spine.crs
    )
    outlines = outlines[~outlines.index.duplicated(keep=False)]
    parcels = parcels.to_crs(spine.crs) if parcels.crs != spine.crs else parcels
    pieces = cut_outlines(
        outlines, pairs, parcels, context.min_fraction, context.area_min_m2
    )
    n_kept_pairs = int(pairs.map(len).sum()) if len(pairs) else 0
    others = pieces[pieces['rank'] > 0]
    if len(others):
        evidence = evidence_for(others)
        keep = keep_pieces(evidence, rule)
    else:
        keep = pd.Series(dtype=bool)
    final = apply_refusals(pieces, outlines, keep)
    final = final.assign(building_id=mint_piece_ids(final, taken))
    new_spine = split_spine(spine, final, links_key)
    new_flat = rekey_at_pairs(
        flat,
        final,
        parcels,
        links_key,
        spine_id_col,
        mapping,
        context.min_fraction,
        context.area_min_m2,
    )
    stats = {
        'candidates': len(pairs),
        'outlines_cut': int(final['outline_id'].nunique()) if len(final) else 0,
        'buildings_added': int(len(final) - final['outline_id'].nunique())
        if len(final)
        else 0,
        'pieces_folded': n_kept_pairs - len(pieces),
        'pieces_refused': int((~keep).sum()) if len(keep) else 0,
    }
    return new_spine, new_flat, stats


@_register('split_outlines_at_parcel_lines', phase='geometry')
def split_outlines_at_parcel_lines(
    state: HarmonizeState,
    entity_type: str = 'parcel',
    lot_values_from: str | None = None,
    points: list | None = None,
    keep_piece: dict | None = None,
    save_link: bool = True,
) -> HarmonizeState:
    """Cut outlines into one building per parcel piece (see the module).

    Runs in the building geospine after `load_links` and before the
    geometry attributes and the point joins, which then work on the
    pieces. Rewrites the restored overlay's crosswalk and overlay at
    the pair, and with *save_link* writes them beside this geospine's
    output, where the curate readers and the building spine find the
    building's parcel pairs; its fingerprint records the step's
    configuration, the overlay's thresholds, the value passes, the
    ingest inputs and the content hash of the overlay it re-keys.

    Parameters
    ----------
    entity_type : str
        The overlay to cut at (default 'parcel'); resolved as the
        footprint geospine resolves it.
    lot_values_from : str
        The recipe whose `link_by_id` passes give a lot its improvement
        value and structure flag (`US_parcel-geospine-2026`); a key that
        does not end in 'recipe_id', because the building geospine runs
        before that recipe and reads its configuration only, never its
        output.
    points : list of dict
        Point layers whose containment in a piece is evidence of a
        structure, each `{recipe_id: ...}`.
    keep_piece : dict
        The refusal rule's thresholds (`parse_keep_piece`).
    save_link : bool
        Write the cut overlay sidecar (default True).
    """
    spine = state.spine
    if spine is None or len(spine) == 0:
        return state
    rule = parse_keep_piece(keep_piece)
    if not lot_values_from:
        raise ValueError(
            'split_outlines_at_parcel_lines needs lot_values_from: the recipe '
            'whose link_by_id passes give a lot its values.'
        )
    reference_id, _ = _resolve_reference_recipe(None, entity_type, state.admin_id)
    if reference_id is None or reference_id not in state.crosswalks:
        # No overlay was restored (a unit without parcels): every
        # building is its outline.
        state.spine = split_spine(spine, gpd.GeoDataFrame(), '')
        if state.verbose:
            print(
                f'  Split: no {entity_type} overlay for {state.admin_id}; nothing cut.'
            )
        return state
    pipeline = state.recipe.get('pipeline') or []
    step_cfg = (
        pipeline[state.step_index]
        if state.step_index is not None and state.step_index < len(pipeline)
        else {
            'step': 'split_outlines_at_parcel_lines',
            'entity_type': entity_type,
            'lot_values_from': lot_values_from,
            'points': points,
            'keep_piece': keep_piece,
            'save_link': save_link,
        }
    )
    context = split_context(state.recipe, state.step_index, step_cfg, state.admin_id)
    source_path = get_entity_link_path(
        get_recipe_id(context.links_recipe), reference_id, state.admin_id
    )
    flat = pd.read_parquet(source_path)
    references = state.references.get(reference_id)
    parcels = references.geometry if references is not None else gpd.GeoSeries()

    def evidence_for(others):
        # The same rows the overlay was computed against, read again
        # raw (see lot_structure_evidence); only a unit with something
        # to cut pays for the read and the value passes.
        raw, _ = reference_rows(
            reference_id, state.admin_id, context.overlay_step.get('kept_by')
        )
        lots = lot_structure_evidence(state, raw, reference_id, context.values_recipe)
        values = lots.reindex(others['parcel_id'].to_numpy())
        return pd.DataFrame(
            {
                'has_point': points_inside(
                    others.geometry, _load_points(points, state.admin_id)
                ).to_numpy(),
                'improvement_value': values[IMPROVEMENT_VALUE_COLUMN].to_numpy(),
                'structure_flag': values[STRUCTURE_FLAG_COLUMN].to_numpy(),
                'compactness': compactness(others.geometry).to_numpy(),
            },
            index=others.index,
        )

    taken = set(spine.index.astype(str)) | set(
        flat[context.links_key].dropna().astype(str)
    )
    new_spine, new_flat, stats = split_overlay(
        spine, flat, parcels, context, evidence_for, rule, taken
    )
    spine_id_col = spine.index.name
    if save_link:
        fingerprint = _link_fingerprint(state, reference_id, context.fingerprint_config)
        _write_cut_sidecar(
            cut_sidecar_path(state.recipe, reference_id, state.admin_id),
            new_flat,
            fingerprint,
            get_recipe_id(context.links_recipe),
            verbose=state.verbose,
        )
    labeled = new_flat[new_flat['link'].notna()].set_index([spine_id_col, 'parcel_id'])
    previous = state.crosswalks[reference_id]
    state.crosswalks[reference_id] = labeled[
        [c for c in previous.columns if c in labeled.columns]
    ]
    state.overlays[reference_id] = new_flat.drop(
        columns=['link', 'link_chain', context.links_key], errors='ignore'
    ).set_index([spine_id_col, 'parcel_id'])
    state.spine = new_spine
    if state.verbose:
        print(
            f'  Split: {stats["outlines_cut"]:,d} of {stats["candidates"]:,d} '
            f'candidate outlines cut, {stats["buildings_added"]:,d} buildings '
            f'added; {stats["pieces_folded"]:,d} pieces folded below the '
            f'thresholds, {stats["pieces_refused"]:,d} refused by the rule'
        )
    if state.timer:
        state.timer.mark('Split (parcel pieces)')
    return state
