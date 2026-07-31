"""
Pipeline steps that create and use relationships between the spine and
reference datasets:
- link_to_reference: load a reference and build a spine ↔ reference crosswalk
- infer_spine_additions: add new spine entries inferred from a reference
- resolve_overlaps: remove remaining geometry overlaps from the spine
"""

from __future__ import annotations

import json
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

from openplaces.core.attribute_registry import (
    get_agg_func,
    get_attributes,
    load_registry,
)
from openplaces.core.schema import AdminId, SourceGeometryType
from openplaces.diagnostics import find_recipes
from openplaces.geo.ids import add_openlocationcode_index, get_geo_ids
from openplaces.geo.link import get_entity_link_path
from openplaces.geo.polygon import (
    clean_polygons,
    get_areas,
    overlay_polygons,
    resolve_overlapping_polygons,
)
from openplaces.io import to_parquet
from openplaces.io.aggregate import aggregate_rows, read_file_metadata
from openplaces.io.cleanup import read_receipt
from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.readers import get_entities
from openplaces.io.transform import make_index_unique, remap
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    get_save_admin_level,
    resolve_attribute_name,
    source_id_from_recipe_id,
)

# Columns carried in spine-reference crosswalk tables (index levels excluded).
_CROSSWALK_COLS = [
    'area_intersection_m2',
    'iou',
    'area_intersection_m2_inner',
    'fraction_of_largest',
]

# Parquet footer key holding a link sidecar's validity fingerprint.
_LINK_METADATA_KEY = 'openplaces:link'


@_register('link_to_reference')
def link_to_reference(
    state: HarmonizeState,
    join: str = 'spatial_overlay',
    entity_type: str | None = None,
    recipe_id: str | None = None,
    thresholds: dict | None = None,
    remap_id: str | None = None,
    source_geometry_type: str | None = None,
    aggregation_function=None,
    sort_by: str | None = None,
    list_columns: list[str] | None = None,
    save_link: bool = False,
) -> HarmonizeState:
    """Load a reference dataset and build a spine ↔ reference crosswalk.

    Populates ``state.references[recipe_id]``,
    ``state.crosswalks[recipe_id]``, ``state.overlays[recipe_id]`` (for
    ``spatial_overlay`` joins), ``state.reference_types[recipe_id]``, and
    ``state.source_geometry_types[recipe_id]`` (when *source_geometry_type*
    is provided).

    Parameters
    ----------
    join : str
        How to join the reference to the spine:

        ``'spatial_overlay'``
            Polygon-on-polygon identity overlay.  Produces a crosswalk table
            with IoU and area-intersection columns.  Populates
            ``state.overlays[recipe_id]`` with the full geometry-bearing
            overlay result for use by later steps.
        ``'spatial_point'``
            Point-in-polygon sjoin.  Joins reference points to the spine
            entities, and unlinked points to any polygon reference already
            in ``state.references`` matching the reference's entity type.

    entity_type : str, optional
        Auto-discover the best ingest recipe of this entity type for the
        current ``admin_id``.  Ignored when ``recipe_id`` is given.
    recipe_id : str, optional
        Explicit reference recipe ID.  Takes precedence over ``entity_type``.
    thresholds : dict, optional
        For ``spatial_overlay``:
        ``min_fraction_of_largest`` (float, default 1/6) — minimum fraction
        of the largest spine-reference intersection to keep a secondary link.
        ``area_intersection_m2_min`` (float, default 10) — minimum intersection
        area in m² to keep a link.
        For ``spatial_point``:
        ``proximity_m`` (float, default 10) — radius for inner proximity pass.
        ``far_proximity_m`` (float, default 100) — radius for outer proximity
        pass (same-parcel constraint applied).  Set to 0 to disable.
    remap_id : str, optional
        Recipe ID for a column-remap table applied to the reference after
        loading (see :func:`openplaces.io.transform.remap`).
    source_geometry_type : str, optional
        :class:`~openplaces.core.schema.SourceGeometryType` value describing
        what this source represents spatially (e.g. ``'single_building_point'``).
        Stored in ``state.source_geometry_types`` for use by downstream steps
        such as ``classify_footprint_priority``.
    aggregation_function : None, callable, or dict, optional
        Controls how duplicate ``geo_id`` rows in the reference are reduced to
        one row before joining.  ``None`` (default) applies the aggregation
        function from the attribute registry.  A dict maps column names to
        callables; columns absent from the dict fall back to the registry
        default.  Only used for ``spatial_overlay`` joins.
    sort_by : str, optional
        Column to sort reference rows by descending before aggregation.
        Falls back to geometry area when the column is absent and the
        reference is a GeoDataFrame.  Only used for ``spatial_overlay`` joins.
    list_columns : list of str, optional
        Column names for which an extra ``{col}_list`` column is added to the
        aggregated reference, collecting all values per ``geo_id`` into a list.
        Normal scalar aggregation for each column still applies alongside.
        Only used for ``spatial_overlay`` joins.
    save_link : bool, optional
        Persist the full many-to-many identity overlay (geometry-free,
        every spine-reference pair including sub-threshold slivers, with
        the crosswalk's link label joined on) as a sidecar parquet at the
        canonical entity-link path. On later runs the sidecar is reloaded
        instead of recomputing the overlay — the single most expensive
        harmonize step — iff its footer fingerprint (step config plus
        size/mtime of the ingest inputs) still matches; a deleted input
        with a tombstone receipt stays verifiable. After a reload,
        ``state.overlays[recipe_id]`` carries no geometry column (only the
        area/IoU columns are consumed downstream). Only used for
        ``spatial_overlay`` joins.
    """
    if state.spine is None:
        warnings.warn('link_to_reference: spine is None; skipping.')
        return state

    resolved_recipe_id, resolved_entity_type = _resolve_reference_recipe(
        recipe_id, entity_type, state.admin_id
    )
    if resolved_recipe_id is None:
        if state.verbose:
            print(
                f'  Link ({join}): no reference recipe found for '
                f'entity_type={entity_type!r} and {state.admin_id}. '
                'Skipping.'
            )
        return state

    if source_geometry_type is not None:
        state.source_geometry_types[resolved_recipe_id] = SourceGeometryType(
            source_geometry_type
        )

    if join == 'spatial_overlay':
        return _link_spatial_overlay(
            state,
            resolved_recipe_id,
            resolved_entity_type,
            thresholds or {},
            aggregation_function,
            sort_by,
            list_columns,
            save_link,
        )
    elif join == 'spatial_point':
        return _link_spatial_point(
            state,
            resolved_recipe_id,
            resolved_entity_type,
            remap_id,
            thresholds or {},
        )
    else:
        raise ValueError(
            f"Unknown join mode: '{join}'. "
            "Expected 'spatial_overlay' or 'spatial_point'."
        )


def _resolve_reference_recipe(
    recipe_id: str | None,
    entity_type: str | None,
    admin_id: AdminId | None,
) -> tuple[str | None, str | None]:
    """Return (resolved_recipe_id, entity_type) for a reference step."""
    if recipe_id is not None:
        derived_type = entity_type
        if derived_type is None:
            base = recipe_id.split('_')[-1]
            derived_type = base.split('-')[0]
        return recipe_id, derived_type
    if entity_type is not None and admin_id is not None:
        found_id = _find_reference_recipe(entity_type, admin_id)
        return found_id, entity_type
    return None, None


def _find_reference_recipe(entity_type: str, admin_id: AdminId) -> str | None:
    """Auto-discover the best ingest recipe for *entity_type* and *admin_id*.

    Scans all stage=``'ingest'`` recipes of the given entity type and returns
    the recipe_id of the one with the most-specific ``admin_id`` prefix that
    is a parent of (or equal to) *admin_id*.
    """
    df = find_recipes(entity_type, stage='ingest')
    if df.empty:
        return None
    admin_str = str(admin_id)
    best_level = -1
    best_row = None
    for _, row in df.iterrows():
        rid_str = row['admin_id']
        if rid_str == '' or admin_str.startswith(rid_str):
            level = rid_str.count('-') + 1 if rid_str else 0
            if level > best_level:
                best_level = level
                best_row = row
    if best_row is None:
        return None
    prefix = f'{best_row["admin_id"]}_' if best_row['admin_id'] else ''
    return (
        f'{prefix}{best_row["entity_type"]}'
        f'-{best_row["source_id"]}-{best_row["version"]}'
    )


def _link_spatial_overlay(
    state: HarmonizeState,
    recipe_id: str,
    entity_type: str | None,
    thresholds: dict,
    aggregation_function=None,
    sort_by: str | None = None,
    list_columns: list[str] | None = None,
    save_link: bool = False,
) -> HarmonizeState:
    """Polygon-on-polygon identity overlay; builds spine-reference crosswalk."""
    min_fraction = thresholds.get('min_fraction_of_largest', 1 / 6)
    area_min_m2 = thresholds.get('area_intersection_m2_min', 10)
    spine_id_col = state.spine.index.name

    ref_raw = get_entities(recipe_id, state.admin_id, geom=True)
    if state.verbose:
        print(
            f'  Link (overlay): {len(ref_raw):,d} {entity_type or "ref"} ({recipe_id})'
        )

    ref = ref_raw.copy()

    # geo_id is generated at ingest only for parcels; footprint/building
    # references (e.g. FEMA) arrive without it, so derive the same stable
    # geometry-hash id here to dedup identical geometries below.
    if 'geo_id' not in ref.columns:
        ref['geo_id'] = get_geo_ids(ref, handle_duplicates=False)

    registry = load_registry()
    numeric_attrs = set(registry.index[registry['data_type'].isin(['float', 'int'])])
    for numeric_col in numeric_attrs:
        if numeric_col in ref.columns:
            ref[numeric_col] = pd.to_numeric(ref[numeric_col], errors='coerce')

    # Build the combined group label for whichever land-use vocabulary the
    # reference carries: parcels use use_group ("what it is used for"),
    # buildings/footprints use purpose_group ("what it was built for").
    for _base in ('use_group', 'purpose_group'):
        if _base in ref.columns:
            _sub = _base.replace('_group', '_subgroup')
            _label = ref[_base].astype(str).fillna('n/a')
            if _sub in ref.columns:
                _label = _label + ' | ' + ref[_sub].astype(str).fillna('n/a')
            ref[f'{_base}_combined'] = pd.Categorical(_label)
    ref['has_duplicate_geometry'] = ref['geo_id'].duplicated(keep=False)
    ref['ha'] = get_areas(ref, 'ha')

    ref_polys = ref[~ref['geo_id'].duplicated()][
        [
            c
            for c in ['geometry', 'geo_id', 'ha', 'has_duplicate_geometry']
            if c in ref.columns
        ]
    ].copy()
    if 'geo_id' in ref_polys.columns:
        ref_polys.index = ref_polys['geo_id'].rename('parcel_id')

    ref_agg = aggregate_rows(
        ref,
        by='geo_id',
        aggregation_function=aggregation_function,
        sort_by=sort_by,
        list_columns=list_columns,
    )
    if ref_agg is not None:
        ref_agg['n_parcels'] = ref.groupby('geo_id').size()
        ref_polys = ref_polys.join(ref_agg)
    if 'improvement_value' in ref_polys.columns and 'ha' in ref_polys.columns:
        ref_polys['improvement_value_per_ha'] = (
            ref_polys['improvement_value'] / ref_polys['ha']
        )

    sidecar_path = None
    fingerprint = None
    footprints_on_ref = None
    if save_link:
        sidecar_path = get_entity_link_path(
            get_recipe_id(state.recipe), recipe_id, state.admin_id
        )
        fingerprint = _link_fingerprint(
            state,
            recipe_id,
            {
                'min_fraction_of_largest': min_fraction,
                'area_intersection_m2_min': area_min_m2,
                'sort_by': sort_by,
                'list_columns': list_columns,
                'aggregation_function': (
                    None if aggregation_function is None else str(aggregation_function)
                ),
            },
        )
        if not state.reprocess:
            footprints_on_ref = _load_link_sidecar(
                sidecar_path, fingerprint, spine_id_col, verbose=state.verbose
            )
    computed_fresh = footprints_on_ref is None

    if computed_fresh:
        footprints_on_ref = overlay_polygons(
            state.spine,
            ref_polys,
            suffixes=('_spine', '_ref'),
            how='identity',
            iou=True,
            geom=True,
        )
    if state.verbose:
        print(
            f'  Link (overlay): {len(footprints_on_ref):,d} '
            f'spine-{entity_type or "ref"} overlaps'
        )
    if state.timer:
        state.timer.mark('Link')

    crosswalk = _build_crosswalk(
        footprints_on_ref, spine_id_col, min_fraction, area_min_m2
    )

    snapped_links = None
    if thresholds.get('snap_chains'):
        crosswalk, snapped_links = snap_chained_links(
            crosswalk,
            spine_id_col,
            fraction_max=float(thresholds.get('chain_fraction_max', 0.75)),
        )
        if state.verbose and len(snapped_links):
            n_snapped = snapped_links.index.get_level_values(spine_id_col).nunique()
            print(
                f'  Link (overlay): snapped {n_snapped:,d} chain-displaced '
                'footprints to their dominant parcel'
            )

    # Written on the reload path too (not just computed_fresh): the link and
    # chain labels are rebuilt from the current crosswalk every run, so the
    # sidecar must be rewritten to keep its stored labels in sync (the raw
    # overlay rows and the fingerprint are carried over unchanged).
    if save_link:
        _write_link_sidecar(
            sidecar_path,
            footprints_on_ref,
            crosswalk,
            fingerprint,
            snapped=snapped_links,
            verbose=state.verbose,
        )

    state.references[recipe_id] = ref_polys
    state.crosswalks[recipe_id] = crosswalk
    state.overlays[recipe_id] = footprints_on_ref
    if entity_type:
        state.reference_types[recipe_id] = entity_type
    return state


def _build_crosswalk(
    footprints_on_ref,
    spine_id_col: str,
    min_fraction: float,
    area_min_m2: float,
) -> pd.DataFrame:
    """Build the trimmed spine-reference crosswalk from the identity overlay.

    Pure function shared by the fresh-overlay and sidecar-reload paths, so
    the sliver trimming and link labeling can never diverge between them.
    Tolerates a geometry-free overlay (the reloaded sidecar).
    """
    crosswalk_cols = [v for v in _CROSSWALK_COLS if v in footprints_on_ref.columns]
    mask_multi = footprints_on_ref.index.get_level_values(spine_id_col).duplicated(
        keep=False
    )

    footprints_single = (
        footprints_on_ref[~mask_multi]
        .reset_index()
        .set_index(spine_id_col)[['parcel_id'] + crosswalk_cols]
    )
    footprints_single.insert(
        1,
        'link',
        np.where(
            footprints_single['parcel_id'].notnull(),
            'unique parcel',
            'no parcel',
        ),
    )

    footprints_multi = footprints_on_ref[mask_multi].copy()
    footprints_multi['fraction_of_largest'] = (
        footprints_multi['area_intersection_m2']
        / footprints_multi.groupby(spine_id_col)['area_intersection_m2'].transform(
            'max'
        )
    ).round(3)

    mask_identified = footprints_multi.index.get_level_values('parcel_id').notnull()
    footprints_multi_identified = footprints_multi[mask_identified]
    footprints_multi_unidentified = footprints_multi[~mask_identified]

    footprints_multi_identified_trimmed = footprints_multi_identified.query(
        f'fraction_of_largest >= {min_fraction} '
        f'and area_intersection_m2 >= {area_min_m2}'
    )
    footprints_multi_trimmed = pd.concat(
        [footprints_multi_identified_trimmed, footprints_multi_unidentified]
    )
    mask_still_multi = footprints_multi_trimmed.index.get_level_values(
        spine_id_col
    ).duplicated(keep=False)

    multi_crosswalk_cols = [v for v in _CROSSWALK_COLS if v in footprints_multi.columns]
    footprints_single_from_multi = (
        footprints_multi_trimmed[~mask_still_multi]
        .reset_index()
        .set_index(spine_id_col)[['parcel_id'] + multi_crosswalk_cols]
    )
    footprints_single_from_multi.insert(
        1,
        'link',
        np.where(
            footprints_single_from_multi['parcel_id'].notnull(),
            'unique parcel (dropping small neighbor)',
            'no parcel',
        ),
    )

    footprints_single = pd.concat(
        [
            footprints_single.drop(
                set(footprints_single.index) & set(footprints_single_from_multi.index)
            ),
            footprints_single_from_multi,
        ]
    ).sort_index()

    split_cols = [v for v in _CROSSWALK_COLS if v in footprints_multi_trimmed.columns]
    footprints_to_split = footprints_multi_trimmed[mask_still_multi][
        split_cols + (['geometry'] if 'geometry' in footprints_multi_trimmed else [])
    ].copy()
    footprints_to_split.insert(0, 'link', 'multi-parcel footprint')

    return pd.concat(
        [
            footprints_single.reset_index().set_index([spine_id_col, 'parcel_id']),
            footprints_to_split.drop(columns='geometry', errors='ignore'),
        ]
    ).sort_index()


def snap_chained_links(
    crosswalk: pd.DataFrame,
    spine_id_col: str,
    fraction_max: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Snap chain-displaced multi-parcel links to their dominant parcel.

    A footprint layer displaced relative to the parcel layer makes each
    footprint straddle its own parcel and the next one over, chaining
    footprint-parcel-footprint-parcel down the block and inflating
    n_parcels_per_footprint for every home on it. Such a footprint is snapped
    to its dominant (largest-intersection) parcel when

    - every minor link's parcel is a *different* footprint's dominant or
      unique parcel — the neighbor demonstrably has its own building. A
      genuine shared row-house footprint never satisfies this: the
      neighboring parcels' only building is the shared footprint itself, so
      real multi-parcel buildings keep their multi links; and
    - every minor link's fraction_of_largest is at most *fraction_max* —
      a near-equal split leaves the dominant side genuinely ambiguous, so it
      is left alone.

    Ownership is computed from the pre-snap crosswalk (dominants never move),
    so one pass resolves whole chains deterministically regardless of row
    order. Uses no geometry, so it works identically on a reloaded
    geometry-free link sidecar.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The adjusted crosswalk — each snapped footprint collapses to a single
        link relabeled ``'unique parcel (snapped from chain)'`` — and the
        removed minor rows (empty when nothing was snapped).
    """
    multi = crosswalk[crosswalk['link'] == 'multi-parcel footprint']
    if multi.empty:
        return crosswalk, crosswalk.iloc[0:0]

    flat = multi.reset_index().sort_values(
        'area_intersection_m2', ascending=False, kind='stable'
    )
    is_dominant = ~flat.duplicated(subset=spine_id_col).to_numpy()

    single_links = crosswalk['link'].astype('string').str.startswith('unique parcel')
    owned = set(
        crosswalk.index.get_level_values('parcel_id')[
            single_links.fillna(False).to_numpy()
        ].dropna()
    )
    owned |= set(flat.loc[is_dominant, 'parcel_id'].dropna())

    if 'fraction_of_largest' in flat.columns:
        fraction = pd.to_numeric(flat['fraction_of_largest'], errors='coerce')
    else:
        fraction = flat['area_intersection_m2'] / flat.groupby(spine_id_col)[
            'area_intersection_m2'
        ].transform('max')

    # A minor link's parcel in `owned` is necessarily another footprint's home:
    # a footprint is either a multi or a single link (never both), and within
    # one footprint the dominant and minor parcels are distinct index entries.
    # The dominant side must be a real parcel — a footprint whose largest
    # overlap is the unmatched (null-parcel) identity remainder has nothing to
    # snap to.
    minor_ok = flat['parcel_id'].isin(owned) & (fraction <= fraction_max)
    row_ok = pd.Series(
        np.where(is_dominant, flat['parcel_id'].notna(), minor_ok),
        index=flat.index,
    )
    snap_fp = row_ok.groupby(flat[spine_id_col]).transform('all').to_numpy()

    minor_pairs = pd.MultiIndex.from_frame(
        flat.loc[snap_fp & ~is_dominant, [spine_id_col, 'parcel_id']]
    )
    dominant_pairs = pd.MultiIndex.from_frame(
        flat.loc[snap_fp & is_dominant, [spine_id_col, 'parcel_id']]
    )
    snapped = crosswalk.loc[crosswalk.index.isin(minor_pairs)]
    out = crosswalk.drop(index=snapped.index)
    out.loc[out.index.isin(dominant_pairs), 'link'] = (
        'unique parcel (snapped from chain)'
    )
    return out, snapped


def _link_fingerprint(
    state: HarmonizeState, ref_recipe_id: str, step_config: dict
) -> dict:
    """Validity fingerprint stored in (and checked against) a link sidecar.

    Records the step configuration and the size/mtime of every resolvable
    ingest-stage input of the harmonize recipe for this admin unit (the
    reference parquet among them). The mid-pipeline spine itself is
    deliberately not fingerprinted. A source that was deliberately deleted
    stays verifiable through its tombstone receipt's recorded size/mtime;
    a missing source with no receipt yields nulls, which no longer match
    once the file reappears (fail safe: recompute).
    """
    from openplaces.io.cleanup import _relative_posix

    upstream_ids = {ref_recipe_id}
    try:
        edges = get_recipe_dependencies(state.recipe, admin_id=state.admin_id)
        upstream_ids |= {e.upstream_recipe_id for e in edges if e.upstream_recipe_id}
    except Exception:
        pass

    sources = []
    for upstream_id in sorted(upstream_ids):
        try:
            upstream = get_recipe_by_id(upstream_id)
            if upstream.get('stage', 'ingest') != 'ingest':
                continue
            source_admin = (
                state.admin_id.truncate_to_level(get_save_admin_level(upstream))
                if state.admin_id
                else None
            )
            path = get_output_path(upstream, admin_id=source_admin)
        except Exception:
            continue
        entry = {'path': _relative_posix(path), 'size': None, 'mtime': None}
        if path.exists():
            stat = path.stat()
            entry['size'] = stat.st_size
            entry['mtime'] = round(stat.st_mtime, 3)
        else:
            receipt = read_receipt(path)
            if receipt is not None:
                entry['size'] = receipt.get('source_size_bytes')
                mtime = receipt.get('source_mtime')
                entry['mtime'] = round(mtime, 3) if mtime is not None else None
        sources.append(entry)

    return {
        'format': 1,
        'spine_recipe_id': get_recipe_id(state.recipe),
        'ref_recipe_id': ref_recipe_id,
        'admin_id': str(state.admin_id) if state.admin_id is not None else None,
        'step_config': step_config,
        'sources': sources,
    }


def _load_link_sidecar(
    sidecar_path, fingerprint: dict, spine_id_col: str, verbose: bool = False
):
    """Reload the persisted identity overlay iff its fingerprint matches.

    Returns the geometry-free overlay (MultiIndex [spine_id, parcel_id])
    or None when the sidecar is absent or invalid (recompute, fail safe).
    Only the footer is read for the validity check.
    """
    if sidecar_path is None or not sidecar_path.exists():
        return None
    stored_raw = read_file_metadata(sidecar_path).get(_LINK_METADATA_KEY)
    if stored_raw is None:
        return None
    try:
        stored = json.loads(stored_raw)
    except json.JSONDecodeError:
        return None
    if stored != fingerprint:
        if verbose:
            print(
                '  Link (overlay): sidecar fingerprint mismatch; recomputing overlay.'
            )
        return None
    overlay = pd.read_parquet(sidecar_path)
    overlay = overlay.set_index([spine_id_col, 'parcel_id'])
    overlay = overlay.drop(columns=['link', 'link_chain'], errors='ignore')
    if verbose:
        print(f'  Link (overlay): reloaded link sidecar {sidecar_path.name}')
    return overlay


def _write_link_sidecar(
    sidecar_path,
    footprints_on_ref,
    crosswalk: pd.DataFrame,
    fingerprint: dict,
    snapped: pd.DataFrame | None = None,
    verbose: bool = False,
) -> None:
    """Persist the geometry-free full identity overlay with link labels.

    The sidecar is a superset of the trimmed crosswalk: every raw overlay
    pair (including sub-threshold slivers and unmatched spine rows) with
    the crosswalk's link label left-joined on (null = trimmed-out pair).
    With chain snapping enabled (*snapped* not None, see
    :func:`snap_chained_links`), a ``link_chain`` column records the
    adjustment: ``'snapped minor'`` on the removed minor pairs (whose
    ``link`` is null, like any other pair excluded from attribution) and
    ``'snapped dominant'`` on the promoted 1-1 link, so the full physical
    overlap stays queryable even though it no longer drives attribution.
    """
    flat = pd.DataFrame(footprints_on_ref.drop(columns='geometry', errors='ignore'))
    if 'link' in crosswalk.columns:
        flat = flat.join(crosswalk['link'])
    if snapped is not None:
        chain = pd.Series(pd.NA, index=flat.index, dtype=object)
        chain[flat.index.isin(snapped.index)] = 'snapped minor'
        promoted = crosswalk.index[
            crosswalk['link'] == 'unique parcel (snapped from chain)'
        ]
        chain[flat.index.isin(promoted)] = 'snapped dominant'
        flat['link_chain'] = chain
    to_parquet(
        flat.reset_index(),
        sidecar_path,
        file_metadata={_LINK_METADATA_KEY: json.dumps(fingerprint)},
    )
    if verbose:
        print(f'  Link (overlay): wrote link sidecar {sidecar_path.name}')


def _rename_right_index(
    gdf: pd.DataFrame,
    right_index_name: str | None,
    target: str,
) -> pd.DataFrame:
    """Rename the right-index column added by ``gpd.sjoin`` to *target*.

    geopandas names the right-index column ``'{name}_right'`` when there is a
    column-name conflict, ``'{name}'`` when there is no conflict (geopandas
    1.x), and ``'index_right'`` when the right index has no name.  This helper
    handles all three conventions.
    """
    candidates = (
        [f'{right_index_name}_right', right_index_name, 'index_right']
        if right_index_name
        else ['index_right']
    )
    for col in candidates:
        if col in gdf.columns:
            return gdf.rename(columns={col: target})
    return gdf


def _dedup_address_points(
    ref: gpd.GeoDataFrame,
    unit_col: str,
    address_col: str,
    housenumber_col: str,
) -> gpd.GeoDataFrame:
    """Deduplicate Overture-style address points for multi-dwelling buildings.

    Groups by (address_col, housenumber_col) — the base address without unit.

    *  When a group contains **unit-specific** records (e.g. "Apt 1", "Apt 2")
       **and** a no-unit record for the same base address, the no-unit record is
       dropped as a redundant aggregate.  Each unit-specific record is kept and
       tagged ``n_dwellings = 1``; downstream :func:`_aggregate_multipoint`
       then sums these into the total per footprint.
    *  When a group contains **only** a no-unit record (single address, no
       unit breakdown), it is kept unchanged.

    Disabled automatically when *unit_col* is absent from *ref*.
    """
    if unit_col not in ref.columns:
        return ref

    key_cols = [c for c in [address_col, housenumber_col] if c in ref.columns]
    if not key_cols:
        return ref

    ref = ref.copy()
    has_unit = ref[unit_col].notna() & (ref[unit_col].astype(str).str.strip() != '')

    ref['_has_unit'] = has_unit
    ref['_group_has_unit'] = ref.groupby(key_cols)['_has_unit'].transform('any')

    mask_drop = ~has_unit & ref['_group_has_unit']
    ref = ref[~mask_drop]

    if 'n_dwellings' not in ref.columns:
        ref['n_dwellings'] = 1.0
    else:
        ref.loc[ref['n_dwellings'].isna(), 'n_dwellings'] = 1.0

    return ref.drop(columns=['_has_unit', '_group_has_unit'])


def _build_size_limit_dict(
    within: pd.DataFrame,
    spine: gpd.GeoDataFrame,
    spine_id_col: str,
    min_samples: int = 10,
) -> dict[str, tuple[float, float]]:
    """Build per-occupancy-class footprint area bounds from Pass 1 (within) matches.

    Computes mean ± 2σ of footprint area (m²) for each ``occupancy_type``
    class in *within*.  Classes with fewer than *min_samples* observations are
    excluded (no size constraint for those classes).

    Parameters
    ----------
    within : DataFrame
        Pass 1 crosswalk (point index, spine_id_col column, occupancy_type column).
    spine : GeoDataFrame
        Spine GeoDataFrame (used to look up footprint areas).
    spine_id_col : str
        Column in *within* that holds the matched footprint ID.
    min_samples : int
        Minimum observations required to compute bounds for a class.
    """
    if 'occupancy_type' not in within.columns or spine_id_col not in within.columns:
        return {}

    areas_m2 = get_areas(spine, unit='m2')
    fp_area = within[spine_id_col].map(areas_m2)

    limits: dict[str, tuple[float, float]] = {}
    for occ, grp_areas in fp_area.groupby(within['occupancy_type']):
        vals = grp_areas.dropna()
        if len(vals) < min_samples:
            continue
        mu, sigma = float(vals.mean()), float(vals.std())
        limits[str(occ)] = (max(0.0, mu - 2 * sigma), mu + 2 * sigma)
    return limits


def _filter_by_size_limit(
    near: pd.DataFrame,
    size_limit_dict: dict[str, tuple[float, float]],
    spine: gpd.GeoDataFrame,
    spine_id_col: str,
) -> pd.DataFrame:
    """Drop proximity-matched pairs whose footprint area falls outside class bounds."""
    if not size_limit_dict or 'occupancy_type' not in near.columns:
        return near
    if spine_id_col not in near.columns:
        return near

    areas_m2 = get_areas(spine, unit='m2')
    fp_area = near[spine_id_col].map(areas_m2)
    lo = near['occupancy_type'].map({k: v[0] for k, v in size_limit_dict.items()})
    hi = near['occupancy_type'].map({k: v[1] for k, v in size_limit_dict.items()})
    no_limit = lo.isna()
    in_range = (fp_area >= lo) & (fp_area <= hi)
    return near[no_limit | in_range]


def _aggregate_multipoint(
    linked: pd.DataFrame,
    spine_id_col: str,
    source_geometry_type,
    verbose: bool = False,
) -> pd.DataFrame:
    """Aggregate multiple points linked to the same footprint into one row.

    Mirrors Lochhead et al. (2026) ``merge_into_group`` / ``merge_occ_type``
    logic.

    For ``single_building_point`` sources (e.g. NSI): sums unit counts across
    the matched occupancy classes via ``_OCC_UNITS`` and re-classifies the
    total using :func:`~openplaces.io.harmonizer.attributes.reverse_occ_units`.
    For ``single_dwelling_point`` sources (address points): sums
    ``n_dwellings`` across all matched points per footprint.

    The highest-quality representative row (by existing sort order) is kept as
    the output row; its ``purpose_subgroup`` and ``n_dwellings`` are
    updated in-place.
    """
    from openplaces.core.schema import SourceGeometryType as _SGT
    from openplaces.io.harmonizer.attributes import _OCC_UNITS, reverse_occ_units

    if spine_id_col not in linked.columns:
        return linked

    dup_fp = linked[spine_id_col].duplicated(keep=False)
    if not dup_fp.any():
        return linked

    singles = linked[~dup_fp]
    multis = linked[dup_fp]

    _is_nsi = source_geometry_type == _SGT.single_building_point
    _is_addr = source_geometry_type == _SGT.single_dwelling_point

    agg_idx: list = []
    agg_rows: list = []
    n_aggregated = 0

    for _fp_id, group in multis.groupby(spine_id_col, sort=False):
        sort_cols = [c for c in ['source', 'structure_value'] if c in group.columns]
        if sort_cols:
            group = group.sort_values(
                sort_cols, ascending=[True, False][: len(sort_cols)]
            )
        rep = group.iloc[0].copy()

        if len(group) > 1:
            n_aggregated += 1
            if _is_nsi and 'occupancy_type' in group.columns:
                total = group['occupancy_type'].map(_OCC_UNITS).fillna(0.0).sum()
                if total > 0:
                    rep['occupancy_type'] = reverse_occ_units(total)
                    rep['n_dwellings'] = float(round(total))
            elif _is_addr:
                if 'n_dwellings' in group.columns:
                    total = (
                        pd.to_numeric(group['n_dwellings'], errors='coerce')
                        .fillna(1.0)
                        .sum()
                    )
                else:
                    # each single_dwelling_point represents one unit
                    total = float(len(group))
                if total > 0:
                    rep['n_dwellings'] = float(total)
                for _col in group.columns:
                    if (
                        not pd.api.types.is_numeric_dtype(group[_col])
                        and not pd.api.types.is_bool_dtype(group[_col])
                        and not pd.api.types.is_datetime64_any_dtype(group[_col])
                    ):
                        _seen = dict.fromkeys(
                            str(v) for v in group[_col] if pd.notna(v)
                        )
                        if _seen:
                            rep[_col] = '; '.join(_seen)

        agg_idx.append(group.index[0])
        agg_rows.append(rep)

    agg_df = pd.DataFrame(agg_rows, index=agg_idx)
    result = pd.concat([singles, agg_df])

    if verbose and n_aggregated > 0:
        print(
            f'  Aggregate: {n_aggregated:,d} footprints with >1 point; units aggregated'
        )
    return result


def flag_duplicate_points(
    ref: pd.DataFrame,
    key_col: str,
    ignore_sources: list[str],
) -> pd.Series:
    """Flag colocated duplicate points from low-rank sources.

    Within groups of two or more points sharing *key_col* (e.g. NSI's
    ``building_id_ubid``, or the ``_olc`` location cell), rows whose
    ``source`` is in *ignore_sources* are labeled
    ``'colocated low-rank source'`` when the group also contains at least one
    source outside that set — a higher-level record to defer to. A group made
    up entirely of ignorable sources stays unflagged (nothing better exists),
    as does any point at a unique location. Returns an object Series aligned
    to *ref* (null = kept); rows are never dropped here — the exclusion is
    applied where the evidence is merged onto a spine
    (:func:`~openplaces.io.harmonizer.attributes.reconcile_attributes`).
    """
    resolution = pd.Series(pd.NA, index=ref.index, dtype=object)
    if not ignore_sources or 'source' not in ref.columns or key_col not in ref.columns:
        return resolution

    key = ref[key_col].astype('string')
    in_group = key.notna() & key.duplicated(keep=False)
    if not in_group.any():
        return resolution

    source = ref['source'].astype(object)
    ignorable = source.isin(list(ignore_sources))
    # Rows with a null key fall out of the groupby; treat them as having no
    # better sibling (they are not in a group anyway).
    has_better = (~ignorable).groupby(key).transform('any').fillna(False).astype(bool)
    flagged = in_group & ignorable & has_better
    resolution[flagged] = 'colocated low-rank source'
    return resolution


def _link_spatial_point(
    state: HarmonizeState,
    recipe_id: str,
    entity_type: str | None,
    remap_id: str | None,
    thresholds: dict,
) -> HarmonizeState:
    """Point-in-polygon join: reference points → spine entities.

    Four-pass attribution (Lochhead et al. 2026, Table 3):

    1. ``predicate='within'`` — direct containment (Step 2).
    2. ``sjoin_nearest`` up to *proximity_m* (default 10 m) — inner proximity
       fallback for near-miss points (Step 5).
    3. ``sjoin_nearest`` up to *far_proximity_m* (default 100 m), constrained
       to the same parcel as the point (Step 6).  Set to 0 to disable.
    4. ``sjoin_nearest`` up to *unbounded_proximity_m* (default 0 = disabled)
       — nearest-footprint fallback with no parcel constraint (Step 7).

    Parcel-derived footprints (``geometry_source`` starting with ``'parcel'``, the
    parcel-shaped fallbacks added by :func:`infer_spine_additions`) participate in
    Pass 1 only: they acquire points by strict containment and are excluded from
    the proximity passes, since a parcel-shaped polygon is not a real building
    outline and must not grab nearby points.

    Optional pre-processing and post-processing steps controlled via
    *thresholds*:

    ``dedup_addresses`` (bool)
        Run :func:`_dedup_address_points` before spatial linking.  Groups by
        base address (street + housenumber), keeps the unit-less record as the
        spatial representative, and counts unit siblings as ``n_dwellings``.
        Column name overrides: ``dedup_unit_number_col`` (default ``'unit_number'``),
        ``dedup_address_street_col`` (default ``'address_street'``),
        ``dedup_address_number_col`` (default ``'address_number'``).
    ``use_size_limit`` (bool)
        After Passes 2–3, drop pairs whose footprint area is outside the
        mean ± 2σ bounds derived from Pass 1 matches per occupancy class
        (:func:`_build_size_limit_dict` / :func:`_filter_by_size_limit`).
    ``aggregate_multipoint`` (bool)
        After linking, aggregate multiple points per footprint into one row
        (:func:`_aggregate_multipoint`) rather than silently dropping
        lower-priority duplicates.

    After linking, joins all points to the first polygon reference in
    ``state.overlays`` to attach a polygon reference ID (e.g. parcel_id).
    """
    _EA_CRS = 'EPSG:6933'
    proximity_m: float = thresholds.get('proximity_m', 10.0)
    far_proximity_m: float = thresholds.get('far_proximity_m', 100.0)
    unbounded_m: float = thresholds.get('unbounded_proximity_m', 0.0)
    dedup_addresses: bool = bool(thresholds.get('dedup_addresses', False))
    use_size_limit: bool = bool(thresholds.get('use_size_limit', False))
    aggregate_mp: bool = bool(thresholds.get('aggregate_multipoint', False))

    ref = get_entities(recipe_id, state.admin_id, geom=True)
    if remap_id:
        ref = remap(ref, remap_id)
    if state.verbose:
        print(f'  Load: {len(ref):,d} {entity_type or "point"} ({recipe_id})')

    # Location key for detecting two points at (near-)identical coordinates
    # (e.g. an ESRI-sourced point duplicating a Parcel-sourced one at the same
    # address) regardless of which linking pass matches each one. Computed here,
    # before geometry is dropped by Pass 1's sjoin below, as a plain non-geometry
    # column so it rides through every pass the same way `source` does.
    # codelength=11 gives a ~2.5-3 m snapping cell (avoids float/rounding
    # false-negatives); handle_duplicates=False keeps colocated points' codes
    # equal (the whole point of using this as a grouping key).
    from openplaces.geo.ids import get_openlocationcodes

    ref = ref.copy()
    ref['_olc'] = get_openlocationcodes(ref, codelength=11, handle_duplicates=False)

    # Duplicate resolution BEFORE any merging to footprints/parcels: flag (not
    # drop) colocated duplicate points per the recipe-chosen rule. The label
    # rides through every linking pass onto state.crosswalks; the actual
    # exclusion is applied where NSI evidence is merged onto the spine
    # (_attribute_point_reference), so the resolution stays inspectable and
    # the method swappable per recipe.
    resolve_dup = thresholds.get('resolve_duplicates')
    if resolve_dup:
        key_col = resolve_dup.get('key', 'building_id_ubid')
        key_col = '_olc' if key_col == 'olc' else key_col
        ignore_sources = resolve_dup.get('ignore_sources', [])
        ref['duplicate_resolution'] = flag_duplicate_points(
            ref, key_col, ignore_sources
        )
        if state.verbose:
            n_flagged = int(ref['duplicate_resolution'].notna().sum())
            if n_flagged:
                print(
                    f'  Dedup (colocated): {n_flagged:,d} low-rank duplicate '
                    'point(s) flagged'
                )

    # Address deduplication: keep building-level representative, count unit siblings
    if dedup_addresses:
        n_before = len(ref)
        ref = _dedup_address_points(
            ref,
            unit_col=thresholds.get('dedup_unit_number_col', 'unit_number'),
            address_col=thresholds.get('dedup_address_street_col', 'address_street'),
            housenumber_col=thresholds.get(
                'dedup_address_number_col', 'address_number'
            ),
        )
        if state.verbose and n_before - len(ref) > 0:
            print(
                f'  Dedup (address): {len(ref):,d} after merging '
                f'{n_before - len(ref):,d} duplicates'
            )

    spine_id_col = state.spine.index.name

    # Pass 1 — within (Lochhead Step 2)
    within = gpd.sjoin(ref, state.spine[['geometry']]).drop(columns='geometry')
    within = _rename_right_index(within, spine_id_col, spine_id_col)
    attributed_idx: set = set(within.index)
    n_pass1 = len(attributed_idx)

    # Parcel-derived footprints (geometry_source like 'parcel.<source>', set by
    # infer_spine_additions) are parcel-shaped fallbacks, not true building
    # outlines, so they link points by strict containment only (Pass 1). Exclude
    # them from the proximity passes below so a parcel-shaped polygon never grabs
    # a nearby point — the only valid criterion for them is 'within'.
    if 'geometry_source' in state.spine.columns:
        parcel_derived = (
            state.spine['geometry_source']
            .astype('string')
            .str.startswith('parcel')
            .fillna(False)
        )
        proximity_spine = state.spine.loc[~parcel_derived, ['geometry']]
    else:
        proximity_spine = state.spine[['geometry']]

    # Build per-class size limits from Pass 1 for use in Passes 2–3
    size_limit_dict: dict[str, tuple[float, float]] = {}
    if use_size_limit:
        size_limit_dict = _build_size_limit_dict(within, state.spine, spine_id_col)

    # Pass 2 — inner proximity, default 10 m (Step 5)
    if proximity_m > 0:
        unlinked = ref[~ref.index.isin(attributed_idx)]
        if not unlinked.empty:
            spine_proj = proximity_spine.to_crs(_EA_CRS)
            unlinked_proj = unlinked.to_crs(_EA_CRS)
            near = gpd.sjoin_nearest(
                unlinked_proj,
                spine_proj,
                how='left',
                max_distance=proximity_m,
                distance_col='_dist',
            )
            near = _rename_right_index(near, spine_id_col, spine_id_col)
            near = near[near[spine_id_col].notna()].drop(columns='_dist')
            near = near.set_crs(ref.crs, allow_override=True)
            if size_limit_dict:
                near = _filter_by_size_limit(
                    near, size_limit_dict, state.spine, spine_id_col
                )
            within = pd.concat([within, near])
            attributed_idx |= set(near.index)

    # Pass 3 — outer proximity, default 100 m, same-parcel constraint (Step 6)
    overlay_ids = list(state.overlays.keys())
    poly_ref: gpd.GeoDataFrame | None = None
    if far_proximity_m > 0 and overlay_ids:
        poly_ref = state.references.get(overlay_ids[0])
        unlinked = ref[~ref.index.isin(attributed_idx)]
        if not unlinked.empty and poly_ref is not None:
            poly_ref_id_col = poly_ref.index.name
            pts_parcel = gpd.sjoin(
                unlinked[['geometry']], poly_ref[['geometry']], how='left'
            ).drop(columns='geometry')
            pts_parcel = _rename_right_index(pts_parcel, poly_ref_id_col, '_pt_parcel')

            fp_parcel = (
                state.crosswalks[overlay_ids[0]]
                .reset_index()[[spine_id_col, poly_ref_id_col]]
                .drop_duplicates(spine_id_col)
                .set_index(spine_id_col)[poly_ref_id_col]
                .rename('_fp_parcel')
            )

            spine_proj = proximity_spine.to_crs(_EA_CRS)
            unlinked_proj = unlinked.to_crs(_EA_CRS)
            far = gpd.sjoin_nearest(
                unlinked_proj,
                spine_proj,
                how='left',
                max_distance=far_proximity_m,
                distance_col='_dist',
            )
            far = _rename_right_index(far, spine_id_col, spine_id_col)
            far = far[far[spine_id_col].notna()].drop(columns='_dist')
            far = far.set_crs(ref.crs, allow_override=True)
            far = far.join(pts_parcel[['_pt_parcel']])
            far = far.join(fp_parcel, on=spine_id_col)
            far = far[
                far['_pt_parcel'].notna() & (far['_pt_parcel'] == far['_fp_parcel'])
            ].drop(columns=['_pt_parcel', '_fp_parcel'])
            if size_limit_dict:
                far = _filter_by_size_limit(
                    far, size_limit_dict, state.spine, spine_id_col
                )
            within = pd.concat([within, far])
            attributed_idx |= set(far.index)

    # Pass 4 — unbounded nearest-footprint fallback, no parcel constraint (Step 7)
    if unbounded_m > 0:
        unlinked = ref[~ref.index.isin(attributed_idx)]
        if not unlinked.empty:
            spine_proj_p4 = proximity_spine.to_crs(_EA_CRS)
            unlinked_proj_p4 = unlinked.to_crs(_EA_CRS)
            far2 = gpd.sjoin_nearest(
                unlinked_proj_p4,
                spine_proj_p4,
                how='left',
                max_distance=unbounded_m,
                distance_col='_dist',
            )
            far2 = _rename_right_index(far2, spine_id_col, spine_id_col)
            far2 = far2[far2[spine_id_col].notna()].drop(columns='_dist')
            far2 = far2.set_crs(ref.crs, allow_override=True)
            within = pd.concat([within, far2])
            attributed_idx |= set(far2.index)

    linked = within

    # Deduplicate: one spine entity per point (keep highest-quality source first)
    sort_cols = [c for c in ['source', 'structure_value'] if c in linked.columns]
    if sort_cols:
        linked = linked.sort_values(
            sort_cols,
            ascending=[True, False][: len(sort_cols)],
        )
    if linked.index.duplicated().any():
        linked = linked[~linked.index.duplicated()].copy()

    # Filter: for footprints that already have a same-parcel dwelling point,
    # drop dwelling points that are on a different parcel.
    _poly_ref_filter = poly_ref
    if _poly_ref_filter is None and overlay_ids:
        _poly_ref_filter = state.references.get(overlay_ids[0])
    if _poly_ref_filter is not None and not linked.empty:
        _prf_id = _poly_ref_filter.index.name
        _ref_sub = ref.loc[ref.index.isin(linked.index), ['geometry']]
        _pts_poly = gpd.sjoin(
            _ref_sub, _poly_ref_filter[['geometry']], how='left'
        ).drop(columns='geometry')
        _pts_poly = _rename_right_index(_pts_poly, _prf_id, '_pt_parcel')
        if _pts_poly.index.duplicated().any():
            _pts_poly = _pts_poly[~_pts_poly.index.duplicated()].copy()
        _fp_parcel_sets = (
            state.crosswalks[overlay_ids[0]]
            .reset_index()[[spine_id_col, _prf_id]]
            .groupby(spine_id_col)[_prf_id]
            .agg(set)
        )
        _pt_parcel = linked.join(_pts_poly[['_pt_parcel']])['_pt_parcel']
        _fp_parcel_set = linked[spine_id_col].map(_fp_parcel_sets)
        _is_same_parcel = pd.Series(
            [
                (pd.notna(pt) and isinstance(fps, set) and pt in fps)
                for pt, fps in zip(_pt_parcel, _fp_parcel_set)
            ],
            index=linked.index,
            dtype=bool,
        )
        _is_cross_parcel = pd.Series(
            [
                (pd.notna(pt) and isinstance(fps, set) and pt not in fps)
                for pt, fps in zip(_pt_parcel, _fp_parcel_set)
            ],
            index=linked.index,
            dtype=bool,
        )
        _fp_has_same = _is_same_parcel.groupby(linked[spine_id_col]).transform('any')
        _mask_drop = _fp_has_same & _is_cross_parcel
        if _mask_drop.any():
            n_cross = int(_mask_drop.sum())
            linked = linked[~_mask_drop].copy()
            if state.verbose:
                print(
                    f'  Filter (cross-parcel): {n_cross:,d} cross-parcel link(s) '
                    f'dropped ({len(linked):,d} remain)'
                )

    # Aggregate multiple points per footprint (Lochhead merge_into_group)
    if aggregate_mp:
        sgt = state.source_geometry_types.get(recipe_id)
        linked = _aggregate_multipoint(linked, spine_id_col, sgt, verbose=state.verbose)

    # Attach polygon reference ID (e.g. parcel_id) to each linked point
    if poly_ref is None and overlay_ids:
        poly_ref = state.references.get(overlay_ids[0])
    if poly_ref is not None:
        poly_ref_id_col = poly_ref.index.name
        ref_on_poly = gpd.sjoin(ref, poly_ref[['geometry']], how='left').drop(
            columns='geometry'
        )
        ref_on_poly = _rename_right_index(ref_on_poly, poly_ref_id_col, poly_ref_id_col)
        if ref_on_poly.index.duplicated().any():
            ref_on_poly = ref_on_poly[~ref_on_poly.index.duplicated()].copy()
        if poly_ref_id_col in ref_on_poly.columns:
            linked = linked.join(ref_on_poly[[poly_ref_id_col]])

    # Drop low-quality NSI duplicates when a higher-quality source
    # covers the same entity
    if 'source' in linked.columns and overlay_ids:
        poly_ref_id_col = (
            state.references[overlay_ids[0]].index.name if overlay_ids else None
        )
        group_cols = [
            c for c in [spine_id_col, poly_ref_id_col] if c and c in linked.columns
        ]
        if group_cols:
            low_quality = {'ESRI', 'HAZUS/NSI-2015'}
            mask_dup = linked[group_cols].duplicated(keep=False)
            first_source = linked.groupby(group_cols, sort=False)['source'].transform(
                'first'
            )
            mask_to_drop = (
                mask_dup
                & linked['source'].isin(low_quality)
                & first_source.eq('Parcel')
            )
            linked = linked[~mask_to_drop]

    # Flag ESRI points that share a location with a differently-sourced point
    # (e.g. a home-office duplicate of a Parcel-sourced record at the same
    # address) -- broader and more targeted than the entity-membership drop
    # above: it fires purely on coordinates, regardless of whether the two
    # points happen to link to the same footprint/parcel, and regardless of
    # what the other source is (not just 'Parcel'). Consumers that sum/count
    # NSI evidence into an upward Single->Multi-Family correction (e.g.
    # n_dwellings in _attribute_point_reference) should exclude flagged rows;
    # other uses of `linked` are unaffected -- rows are flagged, not dropped.
    if 'source' in linked.columns and '_olc' in linked.columns:
        colocated_sources = linked.groupby('_olc')['source'].transform('nunique') > 1
        linked['exclude_from_upward_correction'] = colocated_sources & linked[
            'source'
        ].eq('ESRI')

    if state.verbose:
        n_linked = (
            linked[spine_id_col].notna().sum()
            if spine_id_col in linked.columns
            else len(linked)
        )
        n_proximity = len(attributed_idx) - n_pass1
        print(
            f'  Link (point): {n_linked:,d} points linked'
            + (f' ({n_proximity:,d} via proximity)' if n_proximity > 0 else '')
        )
    if state.timer:
        state.timer.mark('Link (point)')

    state.references[recipe_id] = ref
    state.crosswalks[recipe_id] = linked
    if entity_type:
        state.reference_types[recipe_id] = entity_type
    return state


def _find_admin_scoped_recipe_ids(state: HarmonizeState, entity_type: str) -> list[str]:
    """Ingest recipes of *entity_type* whose admin scope covers ``state.admin_id``.

    One recipe id per (admin_id, source_id), keeping the newest version when
    several exist for the same source (mirrors the specificity/version
    precedence ``find_entity_recipe_id`` uses, ``recipe.py:444-463``).

    Returned oldest-version-first: :func:`link_by_id`'s auto-discover mode
    joins sources in this order, so the most recent source's attributes are
    the ones applied last (see its column-priority rule).
    """
    if state.admin_id is None:
        return []
    best: dict[tuple[str, str], tuple[str, str]] = {}
    for _, row in find_recipes(entity_type, stage='ingest').iterrows():
        admin_id_str = row['admin_id']
        if not admin_id_str or not AdminId(admin_id_str).is_parent_or_equal_of(
            state.admin_id
        ):
            continue
        key = (admin_id_str, row['source_id'])
        recipe_id = f'{admin_id_str}_{entity_type}-{row["source_id"]}-{row["version"]}'
        if key not in best or row['version'] > best[key][0]:
            best[key] = (row['version'], recipe_id)
    return [
        recipe_id for _version, recipe_id in sorted(best.values(), key=lambda vr: vr[0])
    ]


def _discover_link_sources(state: HarmonizeState, entity_type: str) -> list[dict]:
    """Find ingest sources covering ``state.admin_id`` and how to join each.

    A standalone roll (the candidate's own primary entity) joins on the
    standardized cross-source key ``parcel_id_local``. A bundled
    ``additional_layers`` entry joins on its declared ``layer_key`` if
    present (a same-source key shared with its primary entity, e.g.
    MassGIS's ``parcel_id_admin2``), else also falls back to
    ``parcel_id_local``.

    Ordered oldest-version-first (see :func:`_find_admin_scoped_recipe_ids`):
    :func:`link_by_id` joins matches in this order and, for any column
    covered by more than one match, prefers whichever source is applied last
    (i.e. most recent) as long as it covers a majority of parcels — so
    recency, not admin specificity, decides which source's attributes win by
    default.
    """
    from openplaces.recipe import get_recipe_by_id

    matches = []
    for recipe_id in _find_admin_scoped_recipe_ids(state, entity_type):
        recipe = get_recipe_by_id(recipe_id)
        matches.append(
            {'recipe_id': recipe_id, 'layer': None, 'key': 'parcel_id_local'}
        )
        for layer_spec in recipe.get('additional_layers') or []:
            if 'entity' not in layer_spec:
                continue
            matches.append(
                {
                    'recipe_id': recipe_id,
                    'layer': str(layer_spec['entity'].entity_type),
                    'key': layer_spec.get('layer_key', 'parcel_id_local'),
                }
            )
    return matches


def _apply_remap_csvs(state: HarmonizeState, recipe_id: str) -> HarmonizeState:
    """Auto-apply any ``{recipe_id}_*-remap.csv`` crosswalk found beside *recipe_id*.

    Each match's filename (the part between ``{recipe_id}_`` and
    ``-remap.csv``, dashes replaced by underscores) names the spine column it
    remaps; applied only when that column is present. Output columns are
    whichever non-key columns the crosswalk's own header defines (e.g.
    ``use_group``/``use_subgroup`` for a use-code crosswalk). The crosswalk's
    own key length determines how much to truncate codes before lookup
    (handles e.g. 3- vs 4-digit codes sharing one 3-digit crosswalk).
    """
    if state.spine is None:
        return state
    from openplaces.path import recipe_path
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id(recipe_id)
    recipe_dir = recipe_path(recipe['admin_id'], recipe['entity'], as_dir=True)
    if not recipe_dir.exists():
        return state

    spine = state.spine
    prefix = f'{recipe_id}_'
    for csv_path in sorted(recipe_dir.glob(f'{prefix}*-remap.csv')):
        stem = csv_path.stem
        if not stem.endswith('-remap'):
            continue
        column = stem[len(prefix) : -len('-remap')].replace('-', '_')
        if column not in spine.columns:
            continue
        table = pd.read_csv(csv_path, dtype=str)
        key_col = table.columns[0]
        table = table.drop_duplicates(subset=[key_col]).set_index(key_col)
        key_lengths = table.index.to_series().astype(str).str.len()
        key_length = int(key_lengths.mode().iat[0])
        codes = spine[column].astype('string').str.slice(0, key_length)
        for target in table.columns:
            spine[target] = codes.map(table[target])
        if state.verbose:
            matched = codes.isin(table.index).sum()
            print(
                f'  link_by_id: applied {csv_path.name} '
                f'({matched:,d}/{len(spine):,d} {column!r} matched)'
            )

    state.spine = spine
    return state


def _write_prioritized(
    spine: gpd.GeoDataFrame,
    name: str,
    new_vals: pd.Series,
    majority_coverage: float = 0.5,
) -> None:
    """Write *new_vals* into ``spine[name]``, applying the recency/coverage rule.

    A column already on the spine came from an earlier (older, per
    :func:`_find_admin_scoped_recipe_ids`'s ordering) source. *new_vals* only
    overwrites it outright when *new_vals* itself covers at least
    *majority_coverage* of the spine; otherwise it only fills the existing
    column's gaps, so a sparse newer source can't blank out a more complete
    older one.
    """
    if name not in spine.columns:
        spine[name] = new_vals
        return
    coverage = new_vals.notna().mean() if len(new_vals) else 0.0
    if coverage >= majority_coverage:
        spine[name] = new_vals.combine_first(spine[name])
    else:
        spine[name] = spine[name].combine_first(new_vals)


def _warn_if_duplicate_key(key_series: pd.Series, key_name: str, context: str) -> None:
    """Warn when *key_series* has duplicate values.

    openplaces indices (e.g. ``parcel_id``, a geo_id) are unique by design.
    Joining on a column that isn't -- ``parcel_id_local`` is the common case --
    is fine as long as it's *known* to be a many-to-one key (and, for
    'aggregate'/'count' modes, is actually aggregated); this makes that
    non-uniqueness visible instead of a silent assumption.
    """
    valid = key_series.dropna()
    dup_mask = valid.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows:
        n_dup_values = int(valid[dup_mask].nunique())
        warnings.warn(
            f'link_by_id ({context}): {key_name!r} is not unique -- '
            f'{n_dup_rows:,d} rows share a value with at least one other row, '
            f'across {n_dup_values:,d} duplicated values.'
        )


@_register('link_by_id')
def link_by_id(
    state: HarmonizeState,
    recipe_id: str | None = None,
    auto_discover: bool = False,
    entity_type: str = 'parcel',
    mode: str = 'attributes',
    spine_key: str = 'parcel_id_local',
    ref_key: str = 'parcel_id_local',
    columns: list[str] | None = None,
    suffix: str | None = None,
    count_as: str = 'n_transactions',
    flag_as: str = 'is_transacted',
    layer: str | None = None,
) -> HarmonizeState:
    """Link a reference entity to the spine by a precomputed id key (non-spatial).

    Joins on the standardized matching key (``parcel_id_local``) that data
    ingestion already computed on both sides, so no re-conversion happens here.

    Column priority (``'attributes'`` and ``'aggregate'`` modes): when a
    column is already on the spine (from an earlier call, e.g. an earlier
    *auto_discover* match), the new source only overwrites it outright if the
    new source covers a majority of spine rows for that column; otherwise it
    only fills the existing column's gaps (see :func:`_write_prioritized`).
    Combined with *auto_discover*'s oldest-to-newest join order, this makes
    the most recent source the default winner for each column, without
    letting a sparse recent source blank out a more complete older one.

    Parameters
    ----------
    recipe_id : str, optional
        Reference entity recipe (e.g. an assessment roll or a transaction
        table). Required unless *auto_discover* is set.
    auto_discover : bool
        When set, ignore *recipe_id* and instead discover every ingest
        recipe of *entity_type* (plus any bundled ``additional_layers``)
        whose admin scope covers the admin unit being processed
        (:func:`_discover_link_sources`), and recurse into this function once
        per match. Each match always joins via ``mode='aggregate'`` (a
        correct generalization of ``'attributes'`` for 1:1 data too) on its
        resolved key (``parcel_id_local`` for a standalone roll; a layer's
        own ``layer_key`` for a bundled ``additional_layers`` entry), with
        *columns* defaulting to the attribute registry's canonical columns
        for that entity type (:func:`~openplaces.core.attribute_registry.
        get_attributes`) — pass an explicit *columns* to override this
        default for every discovered match. Any ``*-remap.csv`` crosswalk
        found beside a matched source is applied automatically (see
        :func:`_apply_remap_csvs`). A standalone match that is also one of
        the spine's own geometry sources has its ``resolve_spine``
        ``keep_columns`` dropped from the join (already correct on the
        spine; re-deriving them via a non-unique local key would pool
        values across every row sharing it).
    entity_type : str
        Entity type to discover when *auto_discover* is set (default
        ``'parcel'``).
    mode : {'attributes', 'count', 'aggregate'}
        ``'attributes'`` joins *columns* from the reference onto the spine
        (1:1 on the key, keeping the first reference row per key). ``'count'``
        aggregates a 1:many reference into a per-spine count (*count_as*) and a
        boolean presence flag (*flag_as*) — used to track which parcels have
        been transacted. ``'aggregate'`` reduces a 1:many reference onto the
        spine by grouping on the key and applying each column's attribute-
        registry aggregation (e.g. sum land_value/n_dwellings, max year_built),
        falling back to the first non-null value for columns without a registry
        rule; it also emits a per-key record count. Use it when several
        reference rows share one spine key, such as MassGIS L3_ASSESS condominium
        records stacked on one parcel polygon.
    spine_key, ref_key : str
        Key columns on the spine and reference (default ``parcel_id_local``).
        Unlike an entity's own index (unique by design), this key is not
        guaranteed unique; see :func:`_warn_if_duplicate_key`, which warns
        when either side turns out to have duplicates, so that risk is
        visible rather than silently assumed away.
    columns : list of str, optional
        Reference columns to attach in ``'attributes'`` mode.
    suffix : str, optional
        Suffix appended to attached column names (``'attributes'`` mode).
    count_as, flag_as : str
        Output column names in ``'count'`` mode.
    layer : str, optional
        Secondary layer (entity type or full entity string) of an
        ``additional_layers`` entity to load from *recipe_id*, e.g. the
        ``property`` assessor table bundled inside a MassGIS parcel recipe.
    """
    if auto_discover:
        # A standalone roll that is also one of the spine's own geometry
        # sources (state.metadata['spine_source_recipe_ids'], set by
        # resolve_spine) would otherwise re-derive its keep_columns
        # attributes (e.g. use_group/use_subgroup) by aggregating across
        # every spine row sharing its join key -- overwriting an
        # already-correct per-geometry value with one pooled from unrelated
        # rows. Those columns are already on the spine directly from the
        # same source's own row, so drop them from this match; every other
        # attribute (improvement_value, ...) still needs the join.
        spine_source_ids = state.metadata.get('spine_source_recipe_ids', set())
        spine_keep_columns = state.metadata.get('spine_keep_columns', set())
        for match in _discover_link_sources(state, entity_type):
            match_columns = columns or list(
                get_attributes(match['layer'] or entity_type).index
            )
            if match['layer'] is None and match['recipe_id'] in spine_source_ids:
                match_columns = [
                    c for c in match_columns if c not in spine_keep_columns
                ]
                if not match_columns:
                    continue
            state = link_by_id(
                state,
                recipe_id=match['recipe_id'],
                mode='aggregate',
                spine_key=match['key'],
                ref_key=match['key'],
                columns=match_columns,
                layer=match['layer'],
            )
            state = _apply_remap_csvs(state, match['recipe_id'])
        return state

    if recipe_id is None:
        warnings.warn('link_by_id: no recipe_id and auto_discover is False; skipping.')
        return state

    if state.spine is None:
        warnings.warn('link_by_id: spine is None; skipping.')
        return state
    if spine_key not in state.spine.columns:
        warnings.warn(
            f'link_by_id: spine has no {spine_key!r}; skipping {recipe_id}. '
            'Was the spine source ingested with a parcel_id_local directive '
            'and resolve_spine keep_columns?'
        )
        return state

    try:
        ref = get_entities(recipe_id, state.admin_id, layer=layer)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        # The reference is not available for this admin (e.g. a source-specific
        # roll listed in a shared pipeline that does not apply here, including
        # one scoped to a different state) or has not been ingested; skip.
        if state.verbose:
            print(f'  link_by_id: no {recipe_id} for {state.admin_id}; skipping.')
        return state
    if ref is None or ref_key not in ref.columns:
        warnings.warn(
            f'link_by_id: reference {recipe_id} has no {ref_key!r}; skipping.'
        )
        return state

    spine = state.spine
    skey = spine[spine_key].astype('string')
    rkey = ref[ref_key].astype('string')
    _warn_if_duplicate_key(skey, spine_key, 'spine key')

    if mode == 'attributes':
        cols = [c for c in (columns or []) if c in ref.columns]
        # 'attributes' keeps one arbitrary row per key (no aggregation, unlike
        # 'aggregate'/'count') -- a duplicate ref_key here is silently resolved
        # by drop_duplicates below, so flag it before that happens.
        _warn_if_duplicate_key(rkey, ref_key, 'attributes reference key')
        ref_unique = ref.dropna(subset=[ref_key]).drop_duplicates(ref_key).copy()
        ref_unique.index = ref_unique[ref_key].astype('string')
        for col in cols:
            name = f'{col}{suffix}' if suffix else col
            _write_prioritized(spine, name, skey.map(ref_unique[col]))
        if state.verbose:
            matched = skey.isin(set(rkey.dropna())).sum()
            print(
                f'  Link by id (attributes): {matched:,d}/{len(spine):,d} spine '
                f'rows matched {recipe_id} ({len(cols)} columns)'
            )
    elif mode == 'count':
        counts = rkey.dropna().value_counts()
        spine[count_as] = skey.map(counts).fillna(0).astype('int64')
        spine[flag_as] = spine[count_as] > 0
        if state.verbose:
            print(
                f'  Link by id (count): {int(spine[flag_as].sum()):,d}/'
                f'{len(spine):,d} spine rows linked to {recipe_id} '
                f'({count_as}, {flag_as})'
            )
    elif mode == 'aggregate':
        cols = [c for c in (columns or []) if c in ref.columns and c != ref_key]
        ref_valid = ref.dropna(subset=[ref_key]).copy()
        ref_valid[ref_key] = ref_valid[ref_key].astype('string')
        grouped = ref_valid.groupby(ref_key, sort=False)

        # Registry-driven reduction (sum values/dwellings, mean year, etc.);
        # columns without a usable registry rule fall back to the first value.
        reducible = {'sum', 'mean', 'max', 'min', 'first', 'last', 'median'}
        for col in cols:
            fname = get_agg_func(resolve_attribute_name(col))
            func = fname if fname in reducible else 'first'
            name = f'{col}{suffix}' if suffix else col
            _write_prioritized(spine, name, skey.map(grouped[col].agg(func)))
        count_col = count_as if count_as != 'n_transactions' else 'n_records_per_key'
        spine[count_col] = skey.map(grouped.size()).fillna(0).astype('int64')
        if state.verbose:
            matched = skey.isin(set(rkey.dropna())).sum()
            print(
                f'  Link by id (aggregate): {matched:,d}/{len(spine):,d} spine '
                f'rows matched {recipe_id} ({len(cols)} columns, {count_col})'
            )
    else:
        raise ValueError(
            f'link_by_id: unknown mode {mode!r}; expected '
            "'attributes', 'count', or 'aggregate'."
        )

    state.spine = spine
    return state


@_register('infer_spine_additions')
def infer_spine_additions(
    state: HarmonizeState,
    entity_type: str | None = None,
    recipe_id: str | None = None,
    thresholds: dict | None = None,
) -> HarmonizeState:
    """Add spine entries inferred from a reference crosswalk.

    For each reference polygon that has no existing spine coverage and
    exceeds the improvement-value threshold, creates a new spine geometry
    equal to the reference polygon geometry.

    The inferred GeoDataFrame is stored in
    ``state.metadata['inferred_from_<recipe_id>']`` for use by
    ``reconcile_attributes``.

    Parameters
    ----------
    entity_type : str, optional
        Selects all crosswalks matching this type via
        ``state.reference_types``.
    recipe_id : str, optional
        Explicit crosswalk key to use.  Takes precedence over entity_type.
    thresholds : dict, optional
        ``n_per_group_min`` (float, default 0.2) — minimum mean spine count
        per purpose group to be eligible for inference.
        ``value_per_ha_quantile`` (float, default 0.05) — quantile of
        improvement_value_per_ha used as the lower inference bound.
    """
    if state.spine is None:
        return state

    thresholds = thresholds or {}
    n_min: float = thresholds.get('n_per_group_min', 0.2)
    q: float = thresholds.get('value_per_ha_quantile', 0.05)

    if recipe_id is None and entity_type is not None:
        candidates = list(state.get_crosswalks_by_type(entity_type).keys())
        if not candidates:
            if state.verbose:
                print(
                    f'  Infer: no crosswalk for entity_type={entity_type!r}; skipping.'
                )
            return state
        recipe_id = candidates[0]

    if recipe_id is None:
        warnings.warn('infer_spine_additions: no recipe_id resolved.')
        return state

    crosswalk = state.crosswalks.get(recipe_id)
    ref_polys = state.references.get(recipe_id)
    if crosswalk is None or ref_polys is None:
        warnings.warn(
            f'infer_spine_additions: crosswalk or reference for '
            f'{recipe_id!r} not found in state.'
        )
        return state

    if 'improvement_value_per_ha' not in ref_polys.columns:
        if state.verbose:
            print(
                '  Infer: improvement_value_per_ha not available; '
                'skipping parcel-based inference.'
            )
        return state

    # Parcels carry the use_* land-use vocabulary; fall back to purpose_* for a
    # building/footprint reference.
    group_col = next(
        (c for c in ('use_group', 'purpose_group') if c in ref_polys.columns),
        'use_group',
    )

    ref_stat_cols = [
        c
        for c in [
            group_col,
            'improvement_value_per_ha',
            'has_duplicate_geometry',
        ]
        if c in ref_polys.columns
    ]

    has_dup = 'has_duplicate_geometry' in ref_polys.columns
    ref_no_dup = (
        ref_polys[~ref_polys['has_duplicate_geometry']] if has_dup else ref_polys
    )

    footprint_ref_data = crosswalk[['link']].join(
        ref_no_dup[ref_stat_cols],
        on='parcel_id',
        how='inner',
    )

    n_footprints_per_group = (
        (
            (
                footprint_ref_data[group_col].value_counts()
                / ref_polys[group_col].value_counts()
            )
            .fillna(0)
            .rename('n_footprints_mean')
        )
        if group_col in footprint_ref_data.columns
        else pd.Series(dtype=float)
    )

    imp_val_q_col = f'improvement_value_per_ha_q{q}'
    has_imp = 'improvement_value_per_ha' in footprint_ref_data.columns
    if has_imp and group_col in footprint_ref_data.columns:
        imp_val_q_by_group = (
            footprint_ref_data.sample(frac=1)
            .reset_index()
            .drop_duplicates('parcel_id', keep=False)
            .groupby(group_col)['improvement_value_per_ha']
            .quantile((q, 0.5))
            .unstack()
            .rename(
                columns={
                    q: imp_val_q_col,
                    0.5: 'improvement_value_per_ha_median',
                }
            )
        )
    else:
        imp_val_q_by_group = None

    parcel_ids_with_footprint = crosswalk.index.get_level_values('parcel_id').unique()
    mask_without_footprint = ~ref_polys.index.isin(parcel_ids_with_footprint)

    candidate_cols = [
        c
        for c in [
            group_col,
            'improvement_value',
            'improvement_value_per_ha',
            'has_duplicate_geometry',
            'geometry',
        ]
        if c in ref_polys.columns
    ]
    ref_candidates = ref_polys[mask_without_footprint][candidate_cols].copy()

    if imp_val_q_by_group is not None:
        ref_candidates = ref_candidates.join(imp_val_q_by_group, on=group_col)
    if not n_footprints_per_group.empty:
        ref_candidates = ref_candidates.join(n_footprints_per_group, on=group_col)

    imp_val_floor = 0.0
    if has_imp:
        imp_val_floor = (
            ref_polys[ref_polys.index.isin(parcel_ids_with_footprint)]
            .query('improvement_value_per_ha > 0')['improvement_value_per_ha']
            .quantile(q)
        )
    if state.verbose:
        print(f'  Infer: footprint if improvement_value > $ {imp_val_floor:,.0f}/ha')

    n_col_ok = (
        ref_candidates['n_footprints_mean'].gt(n_min)
        if 'n_footprints_mean' in ref_candidates.columns
        else pd.Series(True, index=ref_candidates.index)
    )
    if imp_val_q_col in ref_candidates.columns and has_imp:
        val_col_ok = ref_candidates['improvement_value_per_ha'].gt(
            ref_candidates[imp_val_q_col].div(2).clip(lower=imp_val_floor)
        )
    else:
        val_col_ok = pd.Series(True, index=ref_candidates.index)

    mask_inferred = n_col_ok & val_col_ok
    ref_inferred = ref_candidates[mask_inferred]

    footprints_from_ref = add_openlocationcode_index(
        ref_inferred[['geometry']].reset_index(), name=state.spine.index.name
    )
    footprints_from_ref = footprints_from_ref[
        ~footprints_from_ref.index.isin(state.spine.index)
    ]
    _source_id = source_id_from_recipe_id(recipe_id)
    _et = entity_type or recipe_id.rsplit('_', 1)[-1].split('-', 1)[0]
    footprints_from_ref['geometry_source'] = f'{_et}.{_source_id}'

    state.spine = pd.concat(
        [state.spine, footprints_from_ref[['geometry', 'geometry_source']]]
    ).sort_index()

    if state.spine.index.duplicated().any():
        n_dup = state.spine.index.duplicated().sum()
        warnings.warn(f'{n_dup} duplicate spine IDs for {state.admin_id}.')
        state.spine = make_index_unique(state.spine, sort_duplicates_by_area=True)

    if state.verbose:
        print(
            f'  Infer: +{len(footprints_from_ref):,d} reference-inferred spine entries'
        )
    if state.timer:
        state.timer.mark('Infer')

    state.metadata[f'inferred_from_{recipe_id}'] = footprints_from_ref
    return state


@_register('resolve_overlaps')
def resolve_overlaps(
    state: HarmonizeState,
    **_params,
) -> HarmonizeState:
    """Resolve remaining geometry overlaps in the spine.

    Calls :func:`~openplaces.geo.polygon.resolve_overlapping_polygons` on
    ``state.spine`` (with ``keep=False``).
    """
    if state.spine is None:
        return state

    state.spine = clean_polygons(state.spine)
    state.spine = resolve_overlapping_polygons(state.spine, keep=False)
    if state.verbose:
        print(f'  Resolve: {len(state.spine):,d} after resolving overlaps')
    if state.timer:
        state.timer.mark('Resolve')
    return state
