"""Keep a condominium's footprint whole over its unit parcels
(consolidate_condo_cluster_footprints).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from openplaces.geo.link import get_entity_link_path
from openplaces.geo.polygon import local_metric_crs
from openplaces.io import to_parquet
from openplaces.io.aggregate import read_file_metadata
from openplaces.io.harmonizer import (
    HarmonizeState,
    _register,
)
from openplaces.io.harmonizer.links._shared import _LINK_METADATA_KEY
from openplaces.io.harmonizer.written import note_written
from openplaces.recipe import get_recipe_id

#: Link label of a pair the step adds between a footprint kept whole
#: and one of its lots.
CLUSTER_LINK = 'condo cluster'

#: Prefix of the geometry source of a footprint kept whole. The building
#: geospine's cut (`parcel_pieces.NEVER_CUT_TOKENS`) and the footprint
#: curate read it; the source after it still names the outline's layer
#: for the delivery terms.
CLUSTER_SOURCE_PREFIX = 'condo_cluster.'


def condo_ruleset(recipe_id: str, filename: str) -> list[dict]:
    """The rules of a ruleset CSV kept beside another recipe.

    Parameters
    ----------
    recipe_id : str
        The recipe the ruleset sits beside (for the land-use keywords,
        the parcel curate ``US_parcel-openplaces-2026``).
    filename : str
        The ruleset's name as that recipe gives it
        (``land-use-keywords.csv``).

    Returns
    -------
    list of dict
        From :func:`openplaces.core.rulesets.read_ruleset`.
    """
    from openplaces.core.rulesets import read_ruleset
    from openplaces.path import recipe_path
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id(recipe_id)
    return read_ruleset(
        recipe_path(recipe['admin_id'], recipe['entity'], filename=filename)
    )


def lots_under_footprints(
    footprints: gpd.GeoSeries, lots: gpd.GeoSeries, min_lot_share: float
) -> pd.DataFrame:
    """Each footprint's lots: those with *min_lot_share* of their area under it.

    One spatial join and one array intersection over the unit, in a
    local metric projection; a pair counts only where the polygons
    intersect (strict, no tolerance).

    Parameters
    ----------
    footprints, lots : GeoSeries
        Outlines and lot polygons, each with a unique index, in one CRS.
    min_lot_share : float
        Smallest share of a lot's area under the footprint.

    Returns
    -------
    DataFrame
        One row per (footprint, lot) pair kept: ``footprint`` and ``lot``
        (index labels), ``share`` and ``area_intersection_m2``.
    """
    columns = ['footprint', 'lot', 'share', 'area_intersection_m2']
    if footprints.empty or lots.empty:
        return pd.DataFrame(columns=columns)
    crs_m = local_metric_crs(gpd.GeoSeries(footprints.values, crs=footprints.crs))
    shapes = gpd.GeoSeries(footprints.values, crs=footprints.crs).to_crs(crs_m)
    parcels = gpd.GeoSeries(shapely.make_valid(lots.values), crs=lots.crs).to_crs(crs_m)
    left, right = parcels.sindex.query(shapes.values, predicate='intersects')
    if len(left) == 0:
        return pd.DataFrame(columns=columns)
    lot_area = shapely.area(parcels.values[right])
    overlap = shapely.area(
        shapely.intersection(shapes.values[left], parcels.values[right])
    )
    share = np.divide(overlap, lot_area, out=np.zeros_like(overlap), where=lot_area > 0)
    keep = share >= min_lot_share
    return pd.DataFrame(
        {
            'footprint': footprints.index.to_numpy()[left[keep]],
            'lot': lots.index.to_numpy()[right[keep]],
            'share': share[keep],
            'area_intersection_m2': overlap[keep],
        }
    )


@_register('consolidate_condo_cluster_footprints', phase='geometry')
def consolidate_condo_cluster_footprints(
    state: HarmonizeState,
    entity_type: str | None = 'parcel',
    recipe_id: str | None = None,
    ruleset_recipe_id: str = 'US_parcel-openplaces-2026',
    ruleset: str = 'land-use-keywords.csv',
    condo_class: str = 'Condominium',
    text_columns: tuple[str, ...] = ('use_group', 'use_subgroup', 'building_style'),
    min_lot_share: float = 0.5,
    min_condo_share: float = 0.5,
    min_condo_lots: int = 9,
    min_lots: int = 33,
) -> HarmonizeState:
    """Keep a footprint whole where its lots are condominium units.

    A footprint over several lots is normally cut at the lot lines in
    the building geospine: a townhome row is one building per lot. A
    condominium drawn as one parcel per unit is the exception: its unit
    parcels are not real lots, and the building is one. A footprint
    stays whole when it lies over

    - at least *min_condo_lots* lots, at least *min_condo_share* of
      which the land-use ruleset classes *condo_class*, or
    - at least *min_lots* lots, whatever their use.

    A lot counts when at least *min_lot_share* of its area lies under
    the footprint. The second branch keeps large commercial and
    industrial condominiums whole whose roll does not say condo
    (warehouses, storage, offices).

    How the thresholds were set (plans/buildings-from-parcel-pieces.md,
    section 7.2, "D7 implementation design"): the maintainer labeled
    candidate groups on aerial imagery in three rounds, 2026-11-11, and
    37 of them were footprints over several lots, of which 17 should
    stay whole. Labeled townhome rows lay over at most 21 lots, and the
    two rows whose roll says condo over 2 and 7; every condominium kept
    whole lay over 12 or more. The rule keeps 12 of the 17 whole and no
    row; it misses commercial condominiums under 33 lots whose roll
    does not say condo, and single large buildings over several real
    lots, which are then cut, as before this step existed.
    Measured 2026-11-11 on US-NC-CAR: 149 footprints kept whole over 3,606 lots.

    The condominium signal is the parcel curate's own land-use keyword
    ruleset (*ruleset* beside *ruleset_recipe_id*), applied to the
    parcel reference's *text_columns* joined, so that one maintained
    list decides what a condominium label is in both stages. Where the
    parcel layer carries a use code, its recipe's remap crosswalk
    supplies the text first, as it does for the parcel spine. It matches
    "CONDO" and not abbreviations such as "CNDO". Among the labeled
    groups it agreed with a broader keyword on every group.

    For each footprint kept whole the step changes two things and
    removes nothing:

    - its ``geometry_source`` becomes ``condo_cluster.<source>``, which
      the building geospine never cuts;
    - each of its lots gets a ``condo cluster`` link, with the true
      intersection area, where the overlay has no link for the pair:
      the overlay's ``min_fraction_of_largest`` trim drops a unit
      parcel's small share of a large building, and a lot without a
      link would get a parcel-shaped fallback outline from
      :func:`infer_spine_additions`, which runs after this step.

    It no longer groups touching parcels, builds an outline from lot
    polygons or supersedes links (the maintainer, 2026-11-11: no
    outline without a real footprint).

    Parameters
    ----------
    entity_type : str, optional
        Selects the parcel crosswalk the way :func:`infer_spine_additions`
        does (default ``'parcel'``). Ignored when *recipe_id* is given.
    recipe_id : str, optional
        Explicit crosswalk key.
    ruleset_recipe_id, ruleset : str, optional
        Where the land-use keyword ruleset lives.
    condo_class : str, optional
        The ruleset class that marks a condominium unit.
    text_columns : tuple of str, optional
        Parcel reference columns joined into the label the ruleset
        reads; those absent are skipped.
    min_lot_share : float, optional
        Share of a lot's area under the footprint for the lot to count
        (default 0.5).
    min_condo_share : float, optional
        Share of the footprint's lots classed *condo_class* (default 0.5).
    min_condo_lots : int, optional
        Lots needed with the condominium signal (default 9).
    min_lots : int, optional
        Lots needed without it (default 33).
    """
    if state.spine is None:
        return state
    if recipe_id is None and entity_type is not None:
        candidates = list(state.get_crosswalks_by_type(entity_type).keys())
        if not candidates:
            return state
        recipe_id = candidates[0]
    if recipe_id is None:
        return state
    crosswalk = state.crosswalks.get(recipe_id)
    ref_polys = state.references.get(recipe_id)
    if crosswalk is None or ref_polys is None:
        return state
    spine = state.spine
    spine_id_col = spine.index.name
    if spine_id_col is None or spine_id_col not in crosswalk.index.names:
        return state

    from openplaces.core.rulesets import match_ruleset
    from openplaces.io.harmonizer.spine import in_unit_extent

    # Only parcels within the unit's extent: one outside it is a stray
    # the parcel geospine drops by the same rule.
    lots = ref_polys[in_unit_extent(ref_polys, state.admin_id)]
    lots = lots[~lots.index.duplicated(keep='first') & lots.geometry.notna()]
    source = spine.get('geometry_source', pd.Series('', index=spine.index))
    source = source.astype('string').fillna('')
    # Outlines only: a dotted source was derived from parcels.
    outline = ~source.str.contains('.', regex=False)
    pairs = lots_under_footprints(
        spine.geometry[outline.to_numpy()], lots.geometry, min_lot_share
    )
    if state.timer:
        state.timer.mark('Condo clusters: lots under footprints', n_pairs=len(pairs))
    if pairs.empty:
        return state

    # The parcel spine reads a code-only layer (Florida's DOR use code)
    # through the recipe's remap crosswalk; this step runs before that
    # join, so it applies the same crosswalk here. Otherwise the rule
    # sees no use text in such units and only its lot-count branch
    # fires (pass B, 2026-11-15: US-FL-LA 5 kept whole, 13 expected).
    from openplaces.io.harmonizer.links.discovery import remap_frame

    attributes = remap_frame(
        pd.DataFrame(lots.drop(columns=lots.geometry.name)), recipe_id, state.admin_id
    )
    present = [c for c in text_columns if c in attributes.columns]
    if present:
        label = (
            attributes[present]
            .astype('string')
            .fillna('')
            .agg(' | '.join, axis=1)
            .str.strip(' |')
            .replace('', pd.NA)
        )
        klass, _ = match_ruleset(label, condo_ruleset(ruleset_recipe_id, ruleset))
        is_condo = (klass == condo_class).fillna(False).astype(bool)
    else:
        is_condo = pd.Series(False, index=lots.index)
    pairs['condo'] = is_condo.reindex(pairs['lot']).to_numpy()
    per = pairs.groupby('footprint').agg(n=('lot', 'size'), condo=('condo', 'mean'))
    whole = per.index[
        ((per['n'] >= min_condo_lots) & (per['condo'] >= min_condo_share))
        | (per['n'] >= min_lots)
    ]
    if whole.empty:
        return state

    spine = spine.copy()
    spine.loc[whole, 'geometry_source'] = CLUSTER_SOURCE_PREFIX + source.loc[whole]
    state.spine = spine

    # Links added only where the pair has none: nothing written is
    # removed, and a pair is never recorded twice.
    members = pairs[pairs['footprint'].isin(whole)]
    added = pd.DataFrame(
        {
            spine_id_col: members['footprint'].to_numpy(),
            'parcel_id': members['lot'].to_numpy(),
            'link': CLUSTER_LINK,
            'area_intersection_m2': members['area_intersection_m2'].to_numpy(),
        }
    ).set_index([spine_id_col, 'parcel_id'])
    added = added[~added.index.isin(crosswalk.index)]
    state.crosswalks[recipe_id] = pd.concat([crosswalk, added]).sort_index()
    overlay = state.overlays.get(recipe_id)
    if overlay is not None:
        new = added[~added.index.isin(overlay.index)]
        state.overlays[recipe_id] = pd.concat(
            [overlay, new[['area_intersection_m2']]]
        ).sort_index()
    _append_to_sidecar(state, recipe_id, spine_id_col, added)

    if state.verbose:
        print(
            f'  consolidate_condo_cluster_footprints: {len(whole):,} footprints kept '
            f'whole over {len(members):,} lots ({len(added):,} links added).'
        )
    if state.timer:
        state.timer.mark('Condo clusters: mark footprints and add links')
    return state


def _append_to_sidecar(
    state: HarmonizeState, recipe_id: str, spine_id_col: str, added: pd.DataFrame
) -> None:
    """Add *added* to link_to_reference's overlay sidecar, if one exists.

    The curate readers (apportion_curated_values, collect_link_ids) and
    the building geospine read the sidecar, not the in-memory crosswalk;
    without these rows a footprint kept whole would reach only the lots
    the overlay kept. The sidecar's stored fingerprint is reused as is.

    The sidecar holds every raw overlay pair, and a pair the overlay's
    trim dropped is already there with a null label, which its readers
    take as no link. Such a pair takes the label of *added*; a pair
    with a label keeps it, and a pair not yet there is appended.
    """
    sidecar_path = get_entity_link_path(
        get_recipe_id(state.recipe), recipe_id, state.admin_id
    )
    if added.empty or not sidecar_path.exists():
        return
    stored_raw = read_file_metadata(sidecar_path).get(_LINK_METADATA_KEY)
    if stored_raw is None:
        return
    existing = pd.read_parquet(sidecar_path).set_index([spine_id_col, 'parcel_id'])
    if 'link' not in existing.columns:
        existing['link'] = pd.Series(pd.NA, index=existing.index, dtype=object)
    # Before 2026-11-16 these pairs were skipped as present: measured
    # on pass B, US-NC-CAR, 2,140 of 4,086 pairs of the 149 footprints
    # kept whole stayed unlabeled, and 1,315 condo lots reached no
    # building.
    unlabeled = existing['link'].isna() & existing.index.isin(added.index)
    if unlabeled.any():
        existing['link'] = existing['link'].astype(object)
        existing.loc[unlabeled, 'link'] = (
            added['link'].reindex(existing.index[unlabeled]).to_numpy()
        )
    new = added[~added.index.isin(existing.index)]
    merged = pd.concat([existing, new[['area_intersection_m2', 'link']]]).sort_index()
    to_parquet(
        merged.reset_index(),
        sidecar_path,
        file_metadata={_LINK_METADATA_KEY: stored_raw},
    )
    note_written(sidecar_path)
