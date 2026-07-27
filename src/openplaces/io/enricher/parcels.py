"""
Registered enrich step: attach a reference parcel dataset's attributes onto
current parcels via a fractional area-weighted crosswalk. Works with any
parcel entity recipe that carries a `geo_id` column -- not tied to any one
source -- so long as its raw columns are documented in a sidecar
`{recipe_id}_column-notes.csv` next to the recipe YAML (raw name, canonical
attribute-registry name, aggregation kind, unit; see
`US_parcel-placeslab-fmv2026_column-notes.csv` for the reference example).
"""

from __future__ import annotations

import warnings
from functools import cache

import geopandas as gpd
import pandas as pd

from openplaces.geo.crosswalk import (
    build_id_or_overlay_crosswalk,
    warn_on_geo_id_area_mismatch,
)
from openplaces.io import save_parquet
from openplaces.io.aggregate import aggregate_rows_weighted
from openplaces.io.enricher import EnrichState, _register
from openplaces.io.readers import get_admin, get_entities
from openplaces.path import recipe_path
from openplaces.recipe import get_output_path, get_recipe_by_id

_ADMIN4_GROUP_COL = 'admin4_id'


def _resolve_calibrate_group_col(
    state: EnrichState, old: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, str | None]:
    """Attach an `admin4_id` (town) grouping column onto *old* for calibration.

    A systematic geometry offset between two parcel vintages is set by
    whichever GIS system digitized a given town, not by the county it
    happens to sit in -- grouping `build_id_or_overlay_crosswalk`'s
    calibration stage by town instead of pooling the whole county
    substantially tightens per-group noise (verified empirically on real MA
    data: per-town offset dispersion is roughly half the county-pooled
    dispersion, with genuinely different medians between towns, not just
    noise around one shared shift). `state.spine` already carries
    `admin4_id` when the current-parcel harmonize recipe opted it into
    `resolve_spine`'s `keep_columns` (true for
    `US_parcel-spine-2026`/MassGIS); *old* is mapped onto the same admin4
    units, cheapest option first:

    1. If *old* already has a `csd_id` (Census county-subdivision id)
       column with at least half its rows populated, map it through the
       last 5 digits of each town's `admin4_id_admin1` (a standard Census
       COUSUB code) -- no geometry needed. The coverage check matters: a
       `csd_id` column can be *present but useless* (e.g. an upstream
       ingest regression that leaves it entirely null) without that being
       visible from column presence alone -- trusting it anyway would
       silently produce an all-null `admin4_id`, which `groupby(...,
       dropna=True)` in `_calibrate_unmatched` would then just as silently
       drop every row from, degrading calibration back to ungrouped-quality
       results with no error raised anywhere.
    2. Otherwise, spatially join *old*'s centroids to the admin4 boundaries.
    3. If `state.spine` itself has no `admin4_id` (e.g. a non-MA recipe
       whose source hasn't opted into `keep_columns`) or admin4 boundaries
       aren't available for `state.admin_id`, calibration falls back to the
       previous ungrouped behavior -- this never raises.

    Returns
    -------
    old : GeoDataFrame
        *old*, with an `admin4_id` column added if resolution succeeded
        (same object if `admin4_id` was already present or nothing could be
        resolved).
    group_col : str or None
        `'admin4_id'` if resolution succeeded, else `None`.
    """
    if _ADMIN4_GROUP_COL not in state.spine.columns:
        return old, None
    if _ADMIN4_GROUP_COL in old.columns:
        return old, _ADMIN4_GROUP_COL

    try:
        towns = get_admin(state.admin_id, 4, geom=True)
    except Exception:
        return old, None
    if towns.empty:
        return old, None

    if 'csd_id' in old.columns and old['csd_id'].notna().mean() >= 0.5:
        cousub_to_admin4 = pd.Series(
            towns.index, index=towns['admin4_id_admin1'].astype(str).str[-5:]
        )
        old = old.copy()
        csd_numeric = pd.to_numeric(old['csd_id'], errors='coerce').astype('Int64')
        old[_ADMIN4_GROUP_COL] = (
            csd_numeric.astype(str).str.zfill(5).map(cousub_to_admin4)
        )
        return old, _ADMIN4_GROUP_COL

    with warnings.catch_warnings():
        # Coarse "which town is this parcel in" -- a geographic-CRS centroid
        # is imprecise by at most a few meters, irrelevant next to a town's
        # own size, so this is safe to silence rather than reproject first.
        warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
        old_centroids = old.geometry.centroid
    old_points = gpd.GeoDataFrame(
        geometry=old_centroids, index=old.index, crs=old.crs
    ).to_crs(towns.crs)
    joined = gpd.sjoin(old_points, towns[['geometry']], how='left', predicate='within')
    town_col = (
        _ADMIN4_GROUP_COL if _ADMIN4_GROUP_COL in joined.columns else 'index_right'
    )
    old = old.copy()
    old[_ADMIN4_GROUP_COL] = joined[town_col].reindex(old.index)
    return old, _ADMIN4_GROUP_COL


@cache
def _load_column_notes(reference_recipe_id: str) -> pd.DataFrame:
    """Load `{reference_recipe_id}_column-notes.csv`, the recipe's raw-column crosswalk.

    Columns: `name` (raw source column), `canonical_name` (target
    attribute-registry name; blank when the column is deliberately excluded
    from canonicalization), `suffix` (extra qualifier text -- radius, year,
    model, CV type -- carried into the evidence column name; blank when the
    raw name has none), `aggregation` (`mean`/`sum`/`first`), `raw_unit`
    (only set when it differs from the documented, final `unit` -- e.g. `%`
    for a `_share` attribute, which this step rescales to a 0-1 fraction).
    """
    recipe = get_recipe_by_id(reference_recipe_id)
    recipe_dir = recipe_path(recipe['admin_id'], recipe['entity'], as_dir=True)
    return pd.read_csv(recipe_dir / f'{reference_recipe_id}_column-notes.csv')


def _default_columns(reference_recipe_id: str, aggregation: str) -> list[str]:
    notes = _load_column_notes(reference_recipe_id)
    return notes.loc[notes['aggregation'] == aggregation, 'name'].tolist()


def _percent_columns(reference_recipe_id: str, columns: list[str]) -> list[str]:
    """Raw columns whose real values are 0-100 and need rescaling to 0-1."""
    notes = _load_column_notes(reference_recipe_id).set_index('name')
    return [c for c in columns if notes.loc[c, 'raw_unit'] == '%']


def _source_suffix(reference_recipe: dict) -> str:
    """Provenance suffix for evidence columns: the reference recipe's version.

    Mirrors the image-enrichment convention (`roof_shape_brails`,
    `n_stories_brails`) but derived from the recipe rather than hardcoded, so
    this generic step works for any reference source without editing code.
    Uses `version` rather than `entity.source.source_id` deliberately:
    `source_id` correctly names who produced the data (e.g. a lab), which
    isn't what should appear on every downstream evidence column -- `version`
    names the specific data product instead.
    """
    return reference_recipe['entity'].version


def _canonical_names(
    reference_recipe_id: str, columns: list[str], source_suffix: str
) -> dict[str, tuple]:
    """Map each raw column with a non-blank `canonical_name` to
    `(bare_canonical_name, full_evidence_name)`; `full_evidence_name` is
    `canonical_name[_suffix]_{source_suffix}`. Columns with a blank
    `canonical_name` (deliberately excluded, e.g. the ZTRAX-sourced columns)
    are omitted, so callers fall back to leaving those columns under their
    raw name.
    """
    notes = _load_column_notes(reference_recipe_id).set_index('name')
    result = {}
    for col in columns:
        canonical = notes.loc[col, 'canonical_name']
        if not (isinstance(canonical, str) and canonical):
            continue
        suffix = notes.loc[col, 'suffix']
        full = canonical
        if isinstance(suffix, str) and suffix:
            full += f'_{suffix}'
        full += f'_{source_suffix}'
        result[col] = (canonical, full)
    return result


def _save_published_crosswalk(crosswalk, old, join_key_name, crosswalk_path):
    """Persist *crosswalk* as a publishable sidecar (the inverse of
    `_load_published_crosswalk`). `parcel_id_old` (the reference entity's
    internal, ingest-time-regenerated index) is translated back to the
    reference source's own native id via *join_key_name* when available, so
    an external consumer holding only the raw reference data can join on it
    directly -- the internal id alone would be meaningless to them.
    """
    published = crosswalk[
        ['parcel_id_new', 'parcel_id_old', 'area_ha', 'fraction_of_old', 'match_type']
    ].rename(columns={'parcel_id_new': 'parcel_id'})
    if join_key_name and join_key_name in old.columns:
        published[join_key_name] = published['parcel_id_old'].map(old[join_key_name])
        published = published.drop(columns='parcel_id_old')
    else:
        published = published.rename(columns={'parcel_id_old': 'reference_parcel_id'})
    save_parquet(published, crosswalk_path)


def _load_published_crosswalk(crosswalk_path, old, join_key_name):
    """Reload a sidecar written by `_save_published_crosswalk`, reconstructing
    the internal shape (`parcel_id_old` as *old*'s actual index) the rest of
    `enrich_parcels_from_reference_crosswalk` merges against. A published id
    that no longer maps onto *old* (e.g. the reference source changed
    underneath it) yields NaN -- downstream inner merges against *old* simply
    drop those rows rather than mismatching.
    """
    published = pd.read_parquet(crosswalk_path)
    if join_key_name and join_key_name in published.columns:
        reverse_map = pd.Series(old.index, index=old[join_key_name])
        parcel_id_old = published[join_key_name].map(reverse_map).to_numpy()
    else:
        parcel_id_old = published['reference_parcel_id'].to_numpy()
    return pd.DataFrame(
        {
            'parcel_id_old': parcel_id_old,
            'parcel_id_new': published['parcel_id'].to_numpy(),
            'area_ha': published['area_ha'].to_numpy(),
            'fraction_of_old': published['fraction_of_old'].to_numpy(),
            'match_type': published['match_type'].to_numpy(),
        }
    )


@_register('enrich_parcels_from_reference_crosswalk')
def enrich_parcels_from_reference_crosswalk(
    state: EnrichState,
    reference_parcel_recipe_id: str | None = None,
    min_overlap_m2: float = 10.0,
    area_ratio_tolerance: float = 0.01,
    silent_qa: bool = False,
    mean_columns: list[str] | None = None,
    sum_columns: list[str] | None = None,
    id_columns: list[str] | None = None,
    save_crosswalk: bool = False,
) -> EnrichState:
    """Attach a reference parcel dataset's attributes as area-weighted evidence.

    Builds an ID-first/overlay-fallback crosswalk (`openplaces.geo.crosswalk`)
    between `state.spine` (current parcels; the recipe must set `spine_geom:
    true`) and the reference parcel entity named by `reference_parcel_recipe_id`
    (or the recipe's own `reference_parcel_recipe_id` field), runs the QA
    area-ratio check, then area-weight-aggregates the reference dataset's
    value columns onto `state.evidence`: `mean_columns` (weighted by overlap
    area), `sum_columns` (fraction-of-old apportioned, so splitting an old
    parcel across several new ones conserves its total), and `id_columns`
    (unweighted "first" pick -- for stable reference-geography identifiers
    like a census tract/block-group/ZIP code, not floating values). Missing
    reference coverage for an admin unit is tolerated (returns `state`
    unchanged), matching how image-based steps tolerate a missing
    `image_recipe` admin unit (`io/enricher/attributes.py:_build_image_set`).

    When `mean_columns`/`sum_columns` default (are not passed explicitly),
    the column list, aggregation kind, `%`-to-fraction rescaling, and the
    canonical evidence-column renaming
    (`{canonical_name}[_{suffix}]_{reference_version}`) are all driven by
    the reference recipe's `{recipe_id}_column-notes.csv` sidecar and its own
    `entity.version`. Passing `mean_columns`/`sum_columns`
    explicitly bypasses all of that -- rescaling and renaming both need
    crosswalk-CSV metadata this function has no way to infer for an
    arbitrary caller-chosen column, so those columns aggregate under their
    raw name, unscaled.

    Parameters
    ----------
    state : EnrichState
    reference_parcel_recipe_id : str, optional
        Recipe ID of the reference parcel entity; defaults to the enrich
        recipe's own `reference_parcel_recipe_id` field.
    min_overlap_m2 : float, default 10.0
        Passed to `build_id_or_overlay_crosswalk`.
    area_ratio_tolerance : float, default 0.01
        Passed to `warn_on_geo_id_area_mismatch`.
    silent_qa : bool, default False
        Passed to `warn_on_geo_id_area_mismatch`; set True for unattended
        batch runs.
    mean_columns, sum_columns, id_columns : list of str, optional
        Reference dataset columns to aggregate by weighted mean / apportioned
        sum / unweighted first pick. Default to the reference recipe's
        column-notes CSV classification (see above).
    save_crosswalk : bool, default False
        Persist the built id-or-overlay crosswalk as a `..._crosswalk.parquet`
        sidecar next to this step's evidence output (`parcel_id`, the
        reference recipe's `join_partitions_by.join_key_name` if set else
        `reference_parcel_id`, `area_ha`, `fraction_of_old`, `match_type`) --
        a publishable artifact letting an external consumer holding only the
        raw reference data join it against current parcels without
        re-running this pipeline. When the sidecar already exists and
        `state.reprocess` is False, it is reloaded and reused instead of
        recomputing the crosswalk (the expensive part of this step).
    """
    reference_recipe_id = reference_parcel_recipe_id or state.recipe.get(
        'reference_parcel_recipe_id'
    )
    if not reference_recipe_id:
        raise ValueError(
            'enrich_parcels_from_reference_crosswalk requires '
            "'reference_parcel_recipe_id'."
        )
    reference_recipe = get_recipe_by_id(reference_recipe_id)

    try:
        old = get_entities(reference_recipe, state.admin_id, geom=True)
    except FileNotFoundError:
        return state
    if state.timer:
        state.timer.mark('load reference parcels')

    explicit = mean_columns is not None or sum_columns is not None
    if explicit:
        mean_columns = mean_columns or []
        sum_columns = sum_columns or []
        id_columns = id_columns or []
    else:
        mean_columns = _default_columns(reference_recipe_id, 'mean')
        sum_columns = _default_columns(reference_recipe_id, 'sum')
        id_columns = _default_columns(reference_recipe_id, 'first')

    mean_columns = [c for c in mean_columns if c in old.columns]
    sum_columns = [c for c in sum_columns if c in old.columns]
    id_columns = [c for c in id_columns if c in old.columns]

    if not explicit:
        source_suffix = _source_suffix(reference_recipe)
        percent_columns = _percent_columns(
            reference_recipe_id, mean_columns + sum_columns
        )
        if percent_columns:
            old = old.copy()
            for col in percent_columns:
                old[col] = old[col] / 100

    crosswalk_path = None
    join_key_name = None
    if save_crosswalk:
        join_key_name = (reference_recipe.get('join_partitions_by') or {}).get(
            'join_key_name'
        )
        out_path = get_output_path(
            state.recipe, state.admin_id, entity_recipe_id=state.entity_recipe
        )
        crosswalk_path = out_path.with_stem(out_path.stem + '_crosswalk')

    reloaded = (
        crosswalk_path is not None and not state.reprocess and crosswalk_path.exists()
    )
    if reloaded:
        crosswalk = _load_published_crosswalk(crosswalk_path, old, join_key_name)
        if state.verbose:
            print(f'  reloaded crosswalk sidecar {crosswalk_path.name}')
    else:
        old, calibrate_group_col = _resolve_calibrate_group_col(state, old)
        if state.timer:
            state.timer.mark('assign admin4')

        crosswalk = build_id_or_overlay_crosswalk(
            state.spine,
            old,
            min_overlap_m2=min_overlap_m2,
            calibrate_group_col=calibrate_group_col,
            verbose=state.verbose,
            timer=state.timer,
        )

    if state.verbose:
        counts = crosswalk['match_type'].value_counts()
        n_new_matched = crosswalk['parcel_id_new'].nunique()
        n_old_matched = crosswalk['parcel_id_old'].nunique()
        print(
            f'  crosswalk: {counts.get("geo_id", 0):,} geo_id matches, '
            f'{counts.get("overlay", 0):,} overlay matches '
            f'({len(crosswalk):,} pairs total)'
        )
        print(
            f'  new parcels matched: {n_new_matched:,}/{len(state.spine):,} '
            f'({n_new_matched / len(state.spine):.1%}), '
            f'reference parcels matched: {n_old_matched:,}/{len(old):,} '
            f'({n_old_matched / len(old):.1%})'
        )

    warn_on_geo_id_area_mismatch(
        crosswalk, state.spine, old, tolerance=area_ratio_tolerance, silent=silent_qa
    )
    if state.timer:
        state.timer.mark('QA area mismatch check')

    if crosswalk_path is not None and not reloaded:
        _save_published_crosswalk(crosswalk, old, join_key_name, crosswalk_path)
        if state.verbose:
            print(f'  wrote crosswalk sidecar {crosswalk_path.name}')

    parts = []
    if mean_columns:
        joined = crosswalk.merge(
            old[mean_columns], left_on='parcel_id_old', right_index=True
        )
        result = aggregate_rows_weighted(
            joined,
            by='parcel_id_new',
            wcol='area_ha',
            aggregation_function=dict.fromkeys(mean_columns, 'mean'),
        )
        if result is not None:
            parts.append(result)

    if sum_columns:
        joined = crosswalk.merge(
            old[sum_columns], left_on='parcel_id_old', right_index=True
        )
        result = aggregate_rows_weighted(
            joined,
            by='parcel_id_new',
            wcol='fraction_of_old',
            aggregation_function=dict.fromkeys(sum_columns, 'sum'),
        )
        if result is not None:
            parts.append(result)

    if id_columns and not explicit:
        # aggregate_rows_weighted's unweighted 'first' path re-checks
        # attribute_registry membership internally (delegates to
        # aggregate_rows), so these columns must already carry a registered
        # canonical name -- with no qualifier suffix yet -- before
        # aggregating, unlike mean/sum columns, which rename after.
        id_names = _canonical_names(reference_recipe_id, id_columns, source_suffix)
        bare_rename = {raw: canonical for raw, (canonical, _full) in id_names.items()}
        if bare_rename:
            joined = crosswalk.merge(
                old[list(bare_rename)].rename(columns=bare_rename),
                left_on='parcel_id_old',
                right_index=True,
            )
            result = aggregate_rows_weighted(
                joined,
                by='parcel_id_new',
                wcol='area_ha',
                aggregation_function=dict.fromkeys(bare_rename.values(), 'first'),
            )
            if result is not None:
                full_from_bare = {
                    canonical: full for canonical, full in id_names.values()
                }
                parts.append(result.rename(columns=full_from_bare))

    if state.timer:
        state.timer.mark('aggregate columns')

    if not parts:
        return state

    evidence = pd.concat(parts, axis=1)
    if not explicit:
        rename = {
            raw: full
            for raw, (_canonical, full) in _canonical_names(
                reference_recipe_id, mean_columns + sum_columns, source_suffix
            ).items()
        }
        evidence = evidence.rename(columns=rename)
    for col in evidence.columns:
        state.evidence[col] = evidence[col].reindex(state.evidence.index)
    state.metadata.setdefault('attempted_keys', set()).update(evidence.index)
    return state
