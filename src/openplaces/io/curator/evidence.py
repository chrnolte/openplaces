"""Curation steps for incorporating enrichment evidence."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openplaces.core.constants import M2_TO_SQFT
from openplaces.geo.link import get_entity_link_path, get_link_owner_recipe_id
from openplaces.io import read_parquet
from openplaces.io.curator import CurateState, _register
from openplaces.io.harmonizer.apportion import (
    APPORTIONED_VALUE_COLUMNS,
    apportion_reference_values,
)
from openplaces.recipe import get_output_path, get_recipe_by_id


def _field(obj, name):
    """Read *name* from *obj*, whether it is a dict or a schema object."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


@_register('link_curated_entity', phase='gather')
def link_curated_entity(
    state: CurateState,
    recipe_id: str,
    columns: dict,
    entity_key: str = 'parcel_id',
    ref_key: str = 'parcel_id',
    allow_missing: bool = False,
) -> CurateState:
    """Join another curated entity's attributes onto the current one by a shared id.

    Reads the output of a curate-stage recipe and maps each named column onto
    the current entity (1:many ref -> entity) on the id key. This is how, e.g.,
    the footprint lane consumes the parcel curation lane's land-use decisions
    (the refined ``use_group_combined`` and the ``manufactured_home_community``
    flag), so the referenced entity's classification is finalized before the
    current one's is inferred. Existing columns are overwritten — the
    referenced curated value supersedes the raw harmonized one.

    Parameters
    ----------
    recipe_id : str
        A stage ``curate`` recipe whose output supplies the attributes.
    columns : dict
        Mapping of ``{ref_column: entity_column}``. The entity column is the
        name written onto ``state.curated`` (suffix it to mirror a harmonized
        evidence column, e.g. ``_parcel``).
    entity_key : str, optional
        Id column on the current entity (default ``parcel_id``). Unlike other
        cross-attributed columns, an id column keeps its bare name rather than
        the usual ``_parcel``-style suffix (see ``_attributed_name`` in the
        harmonizer) — the entity's own ``parcel_id`` already names the
        referenced row, so a suffix would be redundant. This is the
        referenced entity's globally-unique ``parcel_id`` (geo_id), not
        ``parcel_id_local``: the latter is only locally cross-comparable and
        can collide within a single admin unit, which would silently
        misattribute the joined columns to an unrelated parcel.
    ref_key : str, optional
        Id column on the referenced entity's own curated output (default
        ``parcel_id`` — that entity's own key, not a cross-attributed one).
    allow_missing : bool, optional
        When True, a mapped column the reference's curated output does
        not carry is written all-missing instead of raising. For an
        attribute only some admin units' sources publish (a structure
        description on one county's roll), where absence is expected
        rather than a recipe error. Default False, so a misspelled
        column still fails loudly.
    """
    ref_recipe = get_recipe_by_id(recipe_id)
    if ref_recipe.get('stage') != 'curate':
        raise ValueError(f"Reference recipe '{recipe_id}' must have stage 'curate'.")

    curated = state.curated
    if entity_key not in curated.columns:
        if state.verbose:
            print(f"  link_curated_entity: entity has no '{entity_key}'; skipping.")
        return state

    ref = read_parquet(get_output_path(ref_recipe, state.admin_id))
    if ref_key not in ref.columns and ref.index.name == ref_key:
        ref = ref.reset_index()
    if ref_key not in ref.columns:
        raise ValueError(f"Curated reference '{recipe_id}' has no '{ref_key}' column.")

    lookup = ref.dropna(subset=[ref_key]).drop_duplicates(ref_key)
    lookup.index = lookup[ref_key].astype('string')
    key = curated[entity_key].astype('string')
    for ref_col, entity_col in columns.items():
        if ref_col not in lookup.columns and allow_missing:
            curated[entity_col] = pd.Series(pd.NA, index=curated.index, dtype=object)
            continue
        if ref_col not in lookup.columns:
            raise ValueError(
                f"Column '{ref_col}' missing from curated reference '{recipe_id}'."
            )
        curated[entity_col] = key.map(lookup[ref_col])

    if state.verbose:
        matched = int(key.isin(set(lookup.index)).sum())
        print(
            f'  link_curated_entity: {matched:,}/{len(curated):,} rows '
            f'matched {recipe_id} ({len(columns)} columns).'
        )
    state.curated = curated
    return state


def _apportioned_sources(
    pairs, ref, ref_key, ref_cols, spine_id_col, index, lane='parcel'
):
    """Map each entity's apportioned value back to its references' provenance.

    Apportionment used to drop provenance entirely: an entity received a
    share of a reference's dollars with no record of where they came from,
    which silently ended the chain for every ``{column}_source`` sidecar
    downstream of it. This rebuilds it from the same ``pairs`` the
    apportionment ran on.

    Two rules, and one covers both apportionment styles. The token is the
    **dominant** reference's, or the largest-overlap reference's that has
    one where the dominant reference has none. That is exact for a
    :data:`~openplaces.io.harmonizer.apportion.WHOLE_VALUE_COLUMNS` column
    drawn from one reference, and the best available label for a sum over
    several: a proportional split, or a whole value that also draws on
    references no principal entity took (the fallback in
    :func:`~openplaces.io.harmonizer.apportion.apportion_reference_values`).
    The derived marker is then applied if **any** contributing
    reference's value was derived, because a sum containing one estimated
    dollar is itself an estimate. Erring toward marking is the safe
    direction: a value wrongly called original is a false claim, a value
    wrongly called derived is only a lost guarantee.

    The open marker (an estimate learned only from values a public
    bundle may carry, `impute_land_value`'s *open_only*) is kept only
    where **every** contributing reference that was derived carries it,
    and added where they all do: one estimate learned from restricted
    values makes the sum one too.

    Parameters
    ----------
    lane : str, optional
        Entity type of the reference, used as the provenance lane prefix
        (default ``'parcel'``). The reference key inside *pairs* is
        normalized to ``'parcel_id'`` by the caller, but the lane a token
        names must be the reference's own.

    Returns
    -------
    dict of {ref_column: Series}
        Per-entity token for every *ref_cols* entry that has a sidecar on
        *ref*, aligned to *index*. Columns with no sidecar are absent.
    """
    from openplaces.io.curator.provenance import is_imputed, mark_imputed, source_column

    sides = {col: source_column(col) for col in ref_cols}
    sides = {col: side for col, side in sides.items() if side in ref.columns}
    if not sides or not len(pairs):
        return {}

    lookup = (
        ref.dropna(subset=[ref_key])
        .drop_duplicates(ref_key)
        .set_index(ref_key)[list(sides.values())]
    )
    ranked = pairs.sort_values('area_intersection_m2', ascending=False)
    dominant = ranked.drop_duplicates(spine_id_col).set_index(spine_id_col)['parcel_id']

    out = {}
    for col, side in sides.items():
        first_known = (
            ranked.assign(_token=ranked['parcel_id'].map(lookup[side]))
            .dropna(subset=['_token'])
            .drop_duplicates(spine_id_col)
            .set_index(spine_id_col)['_token']
        )
        tokens = (
            dominant.map(lookup[side])
            .reindex(index)
            .astype(object)
            .fillna(first_known.reindex(index))
        )
        # Record the entity-crossing hop the same way geometry_source
        # does (`parcel.pamlicocounty`): the token names the lane and
        # the dataset it carried, so `parcel.floridagio` and the flat
        # `parcel` fallback read as one lane instead of two vocabularies
        # in one column. The `+imputed` marker splits on `+`, so the dot
        # is invisible to is_imputed.
        tokens = tokens.map(
            lambda t: t if pd.isna(t) or str(t).startswith(lane) else f'{lane}.{t}'
        )
        pair_tokens = pairs['parcel_id'].map(lookup[side])
        contributed = is_imputed(pair_tokens)
        closed = contributed & ~_has_open_marker(pair_tokens)

        def _by_entity(flags):
            return (
                pd.Series(flags.to_numpy())
                .groupby(pairs[spine_id_col].to_numpy())
                .any()
                .reindex(index)
                .fillna(False)
                .astype(bool)
            )

        any_imputed = _by_entity(contributed)
        any_closed = _by_entity(closed)
        tokens = tokens.where(~any_imputed, tokens.map(_strip_open_marker))
        marked = tokens.map(mark_imputed)
        out[col] = tokens.where(
            ~any_imputed, marked.where(any_closed, marked.map(_with_open_marker))
        )
    return out


def _token_parts(token) -> list[str]:
    from openplaces.core.provenance import TOKEN_SEPARATOR

    return [] if pd.isna(token) else str(token).split(TOKEN_SEPARATOR)


def _has_open_marker(tokens) -> pd.Series:
    """True where a token carries the open marker as a whole part."""
    from openplaces.io.delivery.redaction import OPEN

    return pd.Series(
        [OPEN in _token_parts(t) for t in pd.Series(tokens).astype(object)],
        index=pd.Series(tokens).index,
        dtype=bool,
    )


def _strip_open_marker(token):
    """*token* without the open marker, or as it is."""
    from openplaces.core.provenance import TOKEN_SEPARATOR
    from openplaces.io.delivery.redaction import OPEN

    if pd.isna(token):
        return token
    return TOKEN_SEPARATOR.join(p for p in _token_parts(token) if p != OPEN)


def _with_open_marker(token):
    """An imputed *token* with the open marker before the imputed one
    (`parcel.tier+open+imputed`)."""
    from openplaces.core.provenance import IMPUTED_MARKER, TOKEN_SEPARATOR
    from openplaces.io.delivery.redaction import OPEN

    parts = [p for p in _token_parts(token) if p not in (OPEN, IMPUTED_MARKER)]
    return TOKEN_SEPARATOR.join([*parts, OPEN, IMPUTED_MARKER])


def _readmittable_trimmed_links(links, id_col, ref_key, min_area_m2, orphaned_refs=()):
    """Trimmed overlaps larger than everything still linked to their parcel.

    The harmonize crosswalk drops a footprint's link to a parcel when the
    overlap is below 1/6 of the footprint's largest overlap, on the
    premise that the parcel has a building of its own and the overlap is
    an edge or a displacement. A large building standing on several
    lots breaks that premise: 740 m2 of a 9,000 m2 building on a lot
    whose only linked footprint is 14 m2 is trimmed, and the lot's whole
    building value then goes to the 14 m2 piece. A trimmed pair is
    readmitted when its overlap exceeds the total overlap of the
    parcel's linked footprints, which is where the premise fails;
    displaced neighbors (the trim's real target) overlap less than the
    neighbor's own building and stay out. Pairs that snap_chained_links
    removed ('snapped minor') stay out too, since that step established
    that their parcel holds its own building, unless the parcel is in
    *orphaned_refs*: that building's outline was linked and then left
    the spine (`resolve_overlaps` drops a roof tier nested in the lower
    tier around it, and the link sidecar is written before that), so
    the evidence the snap rested on is gone. Measured 2026-10-16 on
    US-MA-BOS: a 13-story office lot whose 1,264 m2 tower roof was
    dropped in favor of the 3,084 m2 lower tier around it (which the
    snap had assigned to the neighboring lot) held $52.4M on a 185 m2
    remnant. So does every pair on a
    parcel with no linked footprint at all: such a parcel carries the
    harmonize stage's whole-parcel stand-in where it is improved, and
    a neighbor's edge readmitted there took the stand-in's value away
    (measured 2026-10-15 on US-TX-GAL: 191 stand-ins left at zero).

    Measured 2026-10-15 on boston-inner-core: 61 of 1,154 footprints
    above $3,500 of structure value per sq ft owed it to a trimmed
    neighbor of this kind.
    """
    is_linked = links['link'].notna()
    linked_area = links[is_linked].groupby(ref_key)['area_intersection_m2'].sum()
    trimmed = links[
        ~is_linked
        & links[ref_key].notna()
        & (links['area_intersection_m2'] >= min_area_m2)
    ]
    if 'link_chain' in trimmed.columns:
        trimmed = trimmed[
            trimmed['link_chain'].ne('snapped minor').fillna(True)
            | trimmed[ref_key].isin(orphaned_refs)
        ]
    largest = linked_area.reindex(trimmed[ref_key]).fillna(0.0).to_numpy()
    return trimmed[
        (largest > 0) & (trimmed['area_intersection_m2'].to_numpy() > largest)
    ]


def _stated_built_area(
    ref, ref_key, floor_area_columns, stories_column, lot_area_refs=None
):
    """Building ground area (m2) a reference's own record states.

    Floor area over story count, from the first of *floor_area_columns*
    that holds a positive, non-imputed value, capped at the reference's
    own area. A record with no original floor area or story count states
    nothing and gets no value, so the apportionment leaves it alone.

    A reference in *lot_area_refs* that states no floor area in any of
    the columns gets its own area instead: its improvement value is
    read as spread over the lot, so outlines covering a small part of
    the lot keep only that part (see the *lot_area_when_unstated*
    option of :func:`apportion_curated_values`). A record that states
    floor area but no story count still states nothing.
    """
    from openplaces.core.provenance import is_imputed
    from openplaces.io.curator.provenance import source_column

    r = ref.dropna(subset=[ref_key]).drop_duplicates(ref_key).set_index(ref_key)

    def _original(col):
        if col not in r.columns:
            return pd.Series(float('nan'), index=r.index)
        values = pd.to_numeric(r[col], errors='coerce').astype(float)
        side = source_column(col)
        if side in r.columns:
            values = values.mask(is_imputed(r[side]).to_numpy())
        return values.where(values > 0)

    floor = pd.Series(float('nan'), index=r.index)
    for col in floor_area_columns:
        floor = floor.fillna(_original(col))
    stories = _original(stories_column).where(lambda s: s >= 1)
    area = floor / M2_TO_SQFT / stories
    if 'area_ha' in r.columns:
        lot = pd.to_numeric(r['area_ha'], errors='coerce') * 1e4
        area = area.clip(upper=lot)
        if lot_area_refs is not None:
            use_lot = floor.isna() & r.index.isin(list(lot_area_refs))
            area = area.where(~use_lot, lot)
    return area.dropna()


def _measured_floors(frame, stories_column, height_column, story_height_m):
    """Floors from a stated count or a height over *story_height_m*.

    The larger of the two where both are known: a measured height is
    the firmer of the two, and a stated count can be a tag left at its
    default (an outline of 103.8 m by Lidar stated at 1 floor, US-MA-BOS
    `3|470`, 2026-10-16). A height counts the nearest whole number of
    floors (half up), at least one; a stated fraction rounds up. Missing
    where neither is known.

    Rounding a height up counted a roof as a floor: a one-story house
    with a pitched roof measured at 4 m read as two floors, and on a lot
    split by floor area its value moved toward the pitched or taller
    roofs beside a flat annex of 3.4 m (maintainer, 2026-10-17).
    """
    stated = pd.Series(np.nan, index=frame.index, dtype=float)
    if stories_column and stories_column in frame.columns:
        stated = pd.to_numeric(frame[stories_column], errors='coerce').astype(float)
    from_height = pd.Series(np.nan, index=frame.index, dtype=float)
    if height_column and height_column in frame.columns:
        height = pd.to_numeric(frame[height_column], errors='coerce').astype(float)
        ratio = (height / float(story_height_m)).where(height > 0)
        from_height = np.floor(ratio + 0.5).clip(lower=1)
    floors = pd.concat([stated, from_height], axis=1).max(axis=1)
    return np.ceil(floors.where(floors > 0)).clip(lower=1)


@_register('flag_unconfirmed_height', phase='infer')
def flag_unconfirmed_height(
    state: CurateState,
    output_column: str = 'height_unconfirmed',
    value_column: str = 'structure_value',
    area_column: str = 'area_m2',
    min_value_per_sqft: float = 3_500.0,
    max_value_per_floor_sqft: float = 3_000.0,
    stories_column: str | None = None,
    height_column: str | None = None,
    story_height_m: float = 3.5,
    reconciled_stories_column: str | None = None,
    lot_column: str | None = None,
    lot_release_min_m2: float = 100.0,
) -> CurateState:
    """Flag outlines whose value per area no measured height explains.

    A value page draws a building's height from its value per square
    foot of outline, so a tall building is drawn tall because it is
    tall. Divided by its measured floors the same value comes to a
    value per square foot of floor, which has a plausible range; an
    outline drawn above *min_value_per_sqft* that still comes to more
    than *max_value_per_floor_sqft* per square foot of floor is drawn
    as a needle no building there explains: the value of a building the
    outline does not hold (a lot's other buildings, a structure the
    footprint layer lacks), or a box no other layer maps. Floors are
    the measured story count, else the measured height over
    *story_height_m*; an outline with neither counts as one floor, so
    an unconfirmed box is flagged. The value itself is unchanged; the
    flag only says the drawn height is not confirmed.

    Measured 2026-10-16 on boston-inner-core (BOS, CAM, SOM, EVE,
    floors from Overture): value per sq ft of floor had median $210 and
    p99.9 $2,067 over all outlines; the 62-story `4|1149|10` comes to
    $2,110, below the threshold.

    Parameters
    ----------
    output_column : str
        Boolean column written (default ``height_unconfirmed``); missing
        where the outline has no value or no area.
    value_column, area_column : str
        Value and outline area (m2) columns.
    min_value_per_sqft : float
        Value per sq ft of outline above which an outline counts as
        drawn tall (default 3,500, about the 99th percentile).
    max_value_per_floor_sqft : float
        Value per sq ft of floor above which the height is unexplained
        (default 3,000).
    stories_column, height_column : str, optional
        Measured story count and height (m) columns on the curated
        table.
    story_height_m : float
        Meters per story when only a height is known (default 3.5).
    reconciled_stories_column : str, optional
        The curated story count (the roll's or NSI's), used where it
        exceeds the measured floors. Overture measures no height for
        about one outline in nine, and without this an 8-story condo
        building with no Overture height was flagged as a one-floor
        box. Measured 2026-10-17 on boston-inner-core: flagged outlines
        fall from 39 to 28; the 11 released are buildings of 2 to 8
        stories, 6 of them over 400 m2 of outline.
    lot_column : str, optional
        The outline's lot id. Where given, an outline whose floor count
        is known and whose area is at least *lot_release_min_m2* is not
        flagged when its lot's value, over the floor area of all the
        lot's outlines, stays within *max_value_per_floor_sqft*: its own
        excess then comes from a value split by outline area among
        buildings of different heights, not from a doubtful outline. An
        outline with no floor count, or a small box drawn as a needle,
        is still judged alone.
        Measured 2026-10-17 on boston-inner-core: flagged outlines fall from 36 to 32.
        The four released are three Old Colony public-housing buildings
        of 2 stories (279 to 516 m2) on a lot that comes to $2,628 per
        sq ft of floor, and a 211 m2 one-story building on a hospital
        lot that comes to $196.
    lot_release_min_m2 : float
        Smallest outline (m2) the lot test can release (default 100,
        the size below which the floor split treats an unmeasured
        outline as one floor).
    """
    curated = state.curated
    value = pd.to_numeric(curated.get(value_column), errors='coerce')
    area = pd.to_numeric(curated.get(area_column), errors='coerce')
    if value is None or area is None:
        return state
    per_sqft = value / (area * M2_TO_SQFT)
    floors = _measured_floors(curated, stories_column, height_column, story_height_m)
    if reconciled_stories_column and reconciled_stories_column in curated.columns:
        reconciled = pd.to_numeric(
            curated[reconciled_stories_column], errors='coerce'
        ).clip(lower=1)
        floors = pd.concat([floors, reconciled], axis=1).max(axis=1)
    per_floor = per_sqft / floors.fillna(1.0)
    flag = (per_sqft > min_value_per_sqft) & (per_floor > max_value_per_floor_sqft)
    known = (value > 0) & (area > 0)
    if lot_column and lot_column in curated.columns:
        lot = curated[lot_column]
        floor_sqft = (area * M2_TO_SQFT * floors.fillna(1.0)).where(known)
        lot_value = value.where(known).groupby(lot).transform('sum')
        lot_floor_sqft = floor_sqft.groupby(lot).transform('sum')
        lot_fits = (lot_value / lot_floor_sqft) <= max_value_per_floor_sqft
        released = (
            lot_fits.fillna(False)
            & floors.notna()
            & lot.notna()
            & (area >= lot_release_min_m2)
        )
        flag &= ~released
    curated[output_column] = flag.where(known).astype('boolean')
    if state.verbose:
        print(
            f'  flag_unconfirmed_height: {int(flag[known].sum()):,} of '
            f'{int(known.sum()):,} outlines with a value'
        )
    return state


def _floors_per_entity(state, index, floor_weight: dict) -> pd.Series | None:
    """Floors per entity from an enrichment's measured heights.

    Read from the enrich recipe's own evidence file rather than from
    ``state.curated``: the apportionment runs before merge_enrichments
    gathers evidence into the curated table. Floors are the larger of
    the stated count and the height over *story_height_m* rounded to
    the nearest floor, at least one (`_measured_floors`). None when the
    unit has no such evidence. As in `merge_enrichments`, an
    ``entity_recipe`` names the spine an outline enrichment was computed
    on and ``key`` the curated column holding that spine's id (the
    building curate reads Overture's outline heights through each
    building's ``footprint_id``).
    """
    unknown = set(floor_weight) - {
        'recipe_id',
        'stories_column',
        'height_column',
        'story_height_m',
        'unmeasured_max_m2',
        'entity_recipe',
        'key',
    }
    if unknown or 'recipe_id' not in floor_weight:
        raise ValueError(
            'apportion_curated_values: floor_weight takes recipe_id, '
            'stories_column, height_column, story_height_m, '
            'unmeasured_max_m2, entity_recipe and key; got '
            f'{sorted(floor_weight)}.'
        )
    recipe = get_recipe_by_id(floor_weight['recipe_id'])
    path = get_output_path(
        recipe,
        state.admin_id,
        entity_recipe_id=floor_weight.get('entity_recipe') or state.entity_recipe,
    )
    if not path.exists():
        if state.verbose:
            print(
                f'  floor_weight: no {floor_weight["recipe_id"]} evidence; area split.'
            )
        return None
    evidence = read_parquet(path)
    if index.name in evidence.columns:
        evidence = evidence.set_index(index.name)
    floors = _measured_floors(
        evidence,
        floor_weight.get('stories_column'),
        floor_weight.get('height_column'),
        floor_weight.get('story_height_m', 3.5),
    )
    floors = floors[~floors.index.duplicated()]
    key = floor_weight.get('key')
    if key is None:
        floors = floors.reindex(index)
    else:
        floors = state.curated[key].reindex(index).map(floors)
    # An unmeasured outline too small to hold a tower's floor area counts
    # as one floor instead of returning its whole lot to the area split:
    # a lot with a measured tower and a few unmeasured kiosks or roof
    # pieces then still splits by floor area (US-MA-BOS `3|470`: 15
    # outlines, 8 of them under 50 m2 and unmeasured, 2026-10-16).
    # Larger unmeasured outlines still return the lot to the area split.
    small_m2 = float(floor_weight.get('unmeasured_max_m2', 100.0))
    curated = state.curated
    # area_m2 is written later in the pipeline; the geospine's area_ha
    # is already there when the apportionment runs.
    if 'area_m2' in curated.columns:
        area = pd.to_numeric(curated['area_m2'], errors='coerce')
    elif 'area_ha' in curated.columns:
        area = pd.to_numeric(curated['area_ha'], errors='coerce') * 1e4
    else:
        return floors
    small = area.reindex(index) < small_m2
    return floors.where(floors.notna() | ~small, 1.0)


def _linked_pairs(
    state,
    *,
    entity_type,
    link_recipe_id,
    ref_key,
    readmit_trimmed_links,
    min_trimmed_area_m2,
    step,
):
    """Footprint-reference pairs from the harmonize link sidecar.

    Only link-labeled pairs (kept by the harmonize crosswalk's sliver
    thresholds) take part, matching what the harmonize attribution saw,
    plus, with *readmit_trimmed_links*, the trimmed pairs
    :func:`_readmittable_trimmed_links` selects. Pairs whose entity is
    not in ``state.curated`` are dropped. Shared by every step that
    needs a parcel's mapped area, so they all count the same overlaps.

    Returns
    -------
    tuple of (pandas.DataFrame, str, str)
        The pairs (entity id column, ``parcel_id``,
        ``area_intersection_m2``), the entity id column's name, and the
        resolved reference recipe id.

    Raises
    ------
    FileNotFoundError
        When the sidecar is missing: curation cannot recompute the
        overlay, so re-run harmonize with ``save_link: true`` first.
    """
    from openplaces.io.harmonizer.links import _resolve_reference_recipe

    resolved_id, _ = _resolve_reference_recipe(
        link_recipe_id, entity_type, state.admin_id
    )
    if resolved_id is None:
        raise ValueError(
            f'{step}: no reference recipe found for '
            f'entity_type={entity_type!r} and {state.admin_id}.'
        )
    # The sidecar is keyed by the recipe that ran the link steps -- the
    # geospine when the spine is split into geometry and attribute
    # recipes.
    entity_recipe_id = get_link_owner_recipe_id(state.entity_recipe)
    sidecar_path = get_entity_link_path(entity_recipe_id, resolved_id, state.admin_id)
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f'Link sidecar not found: {sidecar_path}. Re-run harmonize for '
            f'{entity_recipe_id} with save_link enabled on the overlay step '
            'so curation can apportion the curated values.'
        )

    links = pd.read_parquet(sidecar_path)
    curated = state.curated
    id_col = curated.index.name
    if id_col not in links.columns:
        # The sidecar records the harmonizer's working spine id column;
        # fall back to the first (index) column
        id_col = links.columns[0]

    present = links[id_col].isin(curated.index)
    all_links = links[present]
    if 'link' in links.columns:
        # Parcels that lost a linked outline between the overlay and
        # the spine (see _readmittable_trimmed_links).
        orphaned = links.loc[~present & links['link'].notna(), ref_key].unique()
        links = all_links[all_links['link'].notna()]
        if readmit_trimmed_links:
            links = pd.concat(
                [
                    links,
                    _readmittable_trimmed_links(
                        all_links,
                        id_col,
                        ref_key,
                        min_trimmed_area_m2,
                        orphaned_refs=orphaned,
                    ),
                ]
            )
    else:
        links = all_links
    pairs = links[[id_col, ref_key, 'area_intersection_m2']].rename(
        columns={ref_key: 'parcel_id'}
    )
    return pairs, id_col, resolved_id


@_register('apportion_curated_values', phase='gather')
def apportion_curated_values(
    state: CurateState,
    recipe_id: str,
    columns: dict,
    entity_type: str = 'parcel',
    link_recipe_id: str | None = None,
    ref_key: str = 'parcel_id',
    entity_key: str = 'parcel_id',
    priority_column: str = 'priority_on_parcel',
    dwelling_column: str = 'n_dwellings_overture',
    land_use_column: str | None = None,
    non_residential_classes: list[str] | None = None,
    readmit_trimmed_links: bool = False,
    min_trimmed_area_m2: float = 10.0,
    floor_area_columns: list[str] | None = None,
    stories_column: str | None = None,
    lot_area_when_unstated: dict | None = None,
    floor_weight: dict | None = None,
) -> CurateState:
    """Apportion a curated reference entity's values over the n:m link sidecar.

    The value-column counterpart of :func:`link_curated_entity`: instead of
    mapping each entity's dominant reference's undivided value, this re-uses
    the n:m overlay persisted by the harmonize stage (``link_to_reference``
    with ``save_link: true``) and the shared apportionment implementation
    (:func:`~openplaces.io.harmonizer.apportion.apportion_reference_values`)
    to distribute the *curated* reference's values exactly the way the
    harmonize stage distributes raw reference values: overlap-area shares
    for ``improvement_value``/``n_dwellings``, dwelling-linked suppression,
    ``land_value``/``address`` whole on the dominant reference of principal
    entities only, ``year_built`` as the linked references' mean.

    Synthetic reference-derived fallback geometries (``geometry_source``
    starting with ``'{entity_type}.'``, added by ``infer_spine_additions``
    *after* the overlay was persisted, so never present in the sidecar) are
    appended as full-weight single links via their ``entity_key`` column.

    Parameters
    ----------
    recipe_id : str
        A stage ``curate`` recipe whose output supplies the values.
    columns : dict
        Mapping of ``{ref_column: entity_column}``. Every ref column must be
        one the shared apportionment knows
        (``APPORTIONED_VALUE_COLUMNS``); use :func:`link_curated_entity`
        for plain dominant-reference attributes.
    entity_type : str, optional
        Reference entity type; resolves the link sidecar the same way the
        harmonize overlay did (default ``parcel``). Ignored when
        *link_recipe_id* is given.
    link_recipe_id : str, optional
        Explicit reference recipe id of the link's other side.
    ref_key : str, optional
        Id column on the referenced curated output (default ``parcel_id``).
    entity_key : str, optional
        Id column on the current entity holding its dominant reference id
        (default ``parcel_id``); used only to link synthetic fallback rows,
        which are absent from the sidecar.
    priority_column : str, optional
        Column holding each entity's role on its reference (default
        ``priority_on_parcel``); drives the secondary/principal rules.
    dwelling_column : str, optional
        Column whose positive value marks an entity as dwelling-linked
        (default ``n_dwellings_overture``); drives the suppression rule.
    land_use_column : str, optional
        Classifier column on the curated reference (e.g. ``land_use_class``)
        used to widen the split for non-residential references -- see
        *non_residential_classes*. Read separately from *columns*: it is a
        classifier input, not one of ``APPORTIONED_VALUE_COLUMNS``, and is
        never itself apportioned. Ignored unless *non_residential_classes* is
        also given; when given but absent from the reference, skipped with a
        verbose message rather than raising (an entity recipe with no
        land-use classification just gets the existing residential-style
        behavior).
    non_residential_classes : list of str, optional
        *land_use_column* values identifying non-residential references,
        passed as ``equal_area_ref_ids`` to
        :func:`~openplaces.io.harmonizer.apportion.apportion_reference_values`
        -- every entity linked to one of these references keeps its full
        overlap-area share of
        :data:`~openplaces.io.harmonizer.apportion.PROPORTIONAL_SPLIT_COLUMNS`
        (``improvement_value``, ``land_value_imputed``,
        ``improvement_value_imputed``), instead of the residential
        dwelling-linked/secondary-structure exclusions: a warehouse plus its
        loading dock, or a retail strip's several units, are all real
        value-bearing structures with no single dwelling to anchor a
        residential-style split to.
    readmit_trimmed_links : bool, optional
        Let a trimmed overlap (a footprint's link to a parcel that the
        harmonize crosswalk dropped as below 1/6 of the footprint's
        largest overlap) take part in that parcel's split when it is
        larger than the overlap of all the parcel's linked footprints
        together. Default False. See :func:`_readmittable_trimmed_links`.
    min_trimmed_area_m2 : float, optional
        Smallest trimmed overlap that can be readmitted (default 10, the
        crosswalk's own minimum).
    floor_area_columns : list of str, optional
        Floor-area columns (square feet) on the curated reference, in
        order of preference, from which the reference's stated building
        ground area is derived together with *stories_column*. Passed to
        :func:`~openplaces.io.harmonizer.apportion.apportion_reference_values`
        as *ref_built_area*. Values marked imputed are not used.
    stories_column : str, optional
        Story-count column on the curated reference. Both this and
        *floor_area_columns* are needed to derive the stated area.
    lot_area_when_unstated : dict, optional
        Read a reference that states no floor area at all as stating
        its own area, so its improvement value is shared over the lot
        and its outlines keep only the part they cover (the rest stays
        unallocated, through *ref_built_area*). Keys: ``column`` and
        ``pattern`` (a regular expression its values must match in
        full) select the references, and ``max_mapped_m2`` limits it to
        references whose linked outlines total less than that area.
        Default None (off). Set only in a county patch recipe, for a
        roll that states the floor area of every building it records,
        so that a record without one is an account the roll records no
        building on: an exempt hospital account with $33.5M of
        improvements whose only mapped outline is 12 m2. The size cap
        is what keeps campuses out: there the outlines, thousands of
        m2, are the buildings the value belongs to even though the
        roll states no floor area for them, and the lot share would
        strip it from them.
    floor_weight : dict, optional
        Split a reference's value among its entities by floor area
        (footprint area times floors) instead of footprint area, so a
        tower takes the tower's value and its one-story podium a
        podium's. Keys: ``recipe_id``, an enrich recipe whose evidence
        carries per-building heights; ``stories_column`` and
        ``height_column`` on that evidence (floors are the larger of the
        story count and the height over ``story_height_m``, default 3.5,
        rounded to the nearest floor). ``unmeasured_max_m2`` (default
        100) counts an unmeasured outline below that area as one floor.
        A reference with any larger entity of unknown height keeps the
        area split (see ``volume_weight`` in
        :func:`~openplaces.io.harmonizer.apportion.apportion_reference_values`).
        Default None (off).

    Raises
    ------
    FileNotFoundError
        When the sidecar is missing: curation cannot recompute the overlay,
        so re-run harmonize with ``save_link: true`` first.
    """
    unknown = [c for c in columns if c not in APPORTIONED_VALUE_COLUMNS]
    if unknown:
        raise ValueError(
            f'apportion_curated_values: no apportionment semantics for '
            f'{unknown}; supported: {list(APPORTIONED_VALUE_COLUMNS)}. '
            'Use link_curated_entity for dominant-reference attributes.'
        )

    ref_recipe = get_recipe_by_id(recipe_id)
    if ref_recipe.get('stage') != 'curate':
        raise ValueError(f"Reference recipe '{recipe_id}' must have stage 'curate'.")

    ref = read_parquet(get_output_path(ref_recipe, state.admin_id))
    if ref_key not in ref.columns and ref.index.name == ref_key:
        ref = ref.reset_index()
    if ref_key not in ref.columns:
        raise ValueError(f"Curated reference '{recipe_id}' has no '{ref_key}' column.")
    # A ref column can be legitimately absent when its source never provided
    # that attribute (e.g. a parcel source with no improvement_value split);
    # skip it -- apportion_reference_values already only produces columns
    # present in ref_values, and the final assignment loop below already
    # defaults an unapportioned column to missing.
    present = [c for c in columns if c in ref.columns]
    missing = [c for c in columns if c not in ref.columns]
    if missing and state.verbose:
        print(
            f'  apportion_curated_values: {missing} missing from curated '
            f"reference '{recipe_id}'; skipping."
        )
    ref_values = (
        ref.dropna(subset=[ref_key])
        .drop_duplicates(ref_key)
        .set_index(ref_key)[present]
    )

    equal_area_ref_ids: set | None = None
    if land_use_column and non_residential_classes:
        if land_use_column in ref.columns:
            land_use_lookup = (
                ref.dropna(subset=[ref_key])
                .drop_duplicates(ref_key)
                .set_index(ref_key)[land_use_column]
            )
            equal_area_ref_ids = set(
                land_use_lookup.index[land_use_lookup.isin(non_residential_classes)]
            )
        elif state.verbose:
            print(
                f'  apportion_curated_values: {land_use_column!r} missing from '
                f"curated reference '{recipe_id}'; skipping equal-area rule."
            )

    curated = state.curated
    pairs, id_col, _ = _linked_pairs(
        state,
        entity_type=entity_type,
        link_recipe_id=link_recipe_id,
        ref_key=ref_key,
        readmit_trimmed_links=readmit_trimmed_links,
        min_trimmed_area_m2=min_trimmed_area_m2,
        step='apportion_curated_values',
    )
    # A link naming a parcel the curated table lacks passes it no value.
    # Some are expected (parcels a curate filter dropped); a large count
    # means the overlay and the parcel table name parcels differently,
    # which until 2026-10-17 dropped 31,585 links in US-TX-HAR silently.
    unknown = ~pairs['parcel_id'].isin(ref_values.index)
    if unknown.any():
        warnings.warn(
            f'apportion_curated_values: {int(unknown.sum()):,d} of {len(pairs):,d} '
            f'links in {state.admin_id} name '
            f'{pairs.loc[unknown, "parcel_id"].nunique():,d} parcels absent from '
            'the curated reference; their values are not apportioned.',
            stacklevel=2,
        )

    lot_area_refs = None
    if lot_area_when_unstated:
        rule = lot_area_when_unstated
        unknown_keys = set(rule) - {'column', 'pattern', 'max_mapped_m2'}
        if unknown_keys or 'column' not in rule or 'pattern' not in rule:
            raise ValueError(
                'apportion_curated_values: lot_area_when_unstated takes '
                "'column', 'pattern' and optionally 'max_mapped_m2'; got "
                f'{sorted(rule)}.'
            )
        if rule['column'] not in ref.columns:
            raise ValueError(
                f'apportion_curated_values: lot_area_when_unstated column '
                f'{rule["column"]!r} is not on the curated reference.'
            )
        selector = ref.dropna(subset=[ref_key]).drop_duplicates(ref_key)
        matches = (
            selector[rule['column']]
            .astype('string')
            .str.fullmatch(rule['pattern'])
            .fillna(False)
            .to_numpy()
        )
        lot_area_refs = set(selector.loc[matches, ref_key])
        if rule.get('max_mapped_m2') is not None:
            mapped = pairs.groupby('parcel_id')['area_intersection_m2'].sum()
            large = set(mapped.index[mapped >= float(rule['max_mapped_m2'])])
            lot_area_refs -= large

    stated_area = (
        _stated_built_area(
            ref,
            ref_key,
            floor_area_columns,
            stories_column,
            lot_area_refs=lot_area_refs,
        )
        if floor_area_columns and stories_column
        else None
    )

    # Synthetic fallback rows postdate the sidecar; add them as their
    # reference's sole full-weight link.
    if 'geometry_source' in curated.columns and entity_key in curated.columns:
        is_synthetic = (
            curated['geometry_source']
            .astype('string')
            .str.startswith(f'{entity_type}.', na=False)
        )
        synthetic = curated.loc[
            is_synthetic
            & curated[entity_key].notna()
            & ~curated.index.isin(set(pairs[id_col])),
            [entity_key],
        ]
        if len(synthetic):
            pairs = pd.concat(
                [
                    pairs,
                    pd.DataFrame(
                        {
                            id_col: synthetic.index,
                            'parcel_id': synthetic[entity_key].to_numpy(),
                            'area_intersection_m2': 1.0,
                        }
                    ),
                ],
                ignore_index=True,
            )

    result = apportion_reference_values(
        pairs,
        ref_values,
        spine_id_col=id_col,
        priority=curated.get(priority_column),
        dwelling_linked_ids=(
            set(curated.index[curated[dwelling_column] > 0])
            if dwelling_column in curated.columns
            else None
        ),
        equal_area_ref_ids=equal_area_ref_ids,
        ref_built_area=stated_area,
        volume_weight=(
            _floors_per_entity(state, curated.index, floor_weight)
            if floor_weight
            else None
        ),
    )

    from openplaces.io.curator.provenance import record_sources

    sources = _apportioned_sources(
        pairs, ref, ref_key, present, id_col, curated.index, lane=entity_type
    )
    for ref_col, entity_col in columns.items():
        attributed = (
            result[ref_col] if ref_col in result.columns else pd.Series(dtype='float64')
        )
        curated[entity_col] = attributed.reindex(curated.index)
        if ref_col in sources:
            record_sources(
                curated, entity_col, sources[ref_col], mask=curated[entity_col].notna()
            )

    if state.verbose:
        n_linked = curated.index.isin(set(pairs[id_col])).sum()
        print(
            f'  apportion_curated_values: {n_linked:,}/{len(curated):,} rows '
            f'linked to {recipe_id} ({len(columns)} columns).'
        )
    state.curated = curated
    return state


@_register('collect_link_ids', phase='gather')
def collect_link_ids(
    state: CurateState,
    entity_type: str | None = None,
    link_recipe_id: str | None = None,
    column: str = 'parcel_id_all',
    ref_key: str | None = None,
    include_below_threshold: bool = False,
) -> CurateState:
    """Materialize all linked reference ids per entity from a link sidecar.

    Reads the n:m link sidecar persisted by the harmonize overlay
    (``link_to_reference`` with ``save_link: true``) and writes a
    pipe-joined id column onto the curated entity, ordered by descending
    intersection area — the dominant reference (e.g. the entity's
    ``parcel_id``) comes first. Harmonized spines stay link-only; this is
    where multi-reference membership becomes a canonical column.

    Parameters
    ----------
    entity_type : str, optional
        Auto-discover the reference recipe of this entity type for the
        current admin unit, the same way the harmonize overlay resolved
        it (so the sidecar filename matches per state/county). Ignored
        when ``link_recipe_id`` is given.
    link_recipe_id : str, optional
        Explicit reference recipe ID of the link's other side.
    column : str, optional
        Output column written onto the curated entity (default
        ``parcel_id_all``). Left missing where the collected value is
        identical to the base id column (e.g. ``parcel_id``).
    ref_key : str, optional
        Sidecar column holding the reference's own id. Defaults to
        ``{entity_type}_id``, or ``parcel_id`` when only a
        ``link_recipe_id`` was given. Name it explicitly for a reference
        whose id column does not follow that pattern.
    include_below_threshold : bool, optional
        When False (default), only link-labeled pairs (those kept by the
        harmonize crosswalk's sliver thresholds) are collected. When
        True, every raw overlap in the sidecar counts.

    Raises
    ------
    FileNotFoundError
        When the sidecar is missing: curation cannot recompute the
        overlay, so re-run harmonize with ``save_link: true`` first.
    KeyError
        When the sidecar carries no *ref_key* column.
    """
    from openplaces.io.harmonizer.links import _resolve_reference_recipe

    resolved_id, _ = _resolve_reference_recipe(
        link_recipe_id, entity_type, state.admin_id
    )
    if resolved_id is None:
        if state.verbose:
            print(
                f'  collect_link_ids: no reference recipe for '
                f'entity_type={entity_type!r} and {state.admin_id}; skipping.'
            )
        return state

    entity_recipe_id = get_link_owner_recipe_id(state.entity_recipe)
    sidecar_path = get_entity_link_path(entity_recipe_id, resolved_id, state.admin_id)
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f'Link sidecar not found: {sidecar_path}. Re-run harmonize for '
            f'{entity_recipe_id} with save_link enabled on the overlay step '
            'so curation can collect the n:m link ids.'
        )

    links = pd.read_parquet(sidecar_path)
    curated = state.curated
    id_col = curated.index.name
    if id_col not in links.columns:
        # The sidecar records the harmonizer's working spine id column;
        # fall back to the first (index) column
        id_col = links.columns[0]

    # The reference id column, not the literal 'parcel_id': recipe
    # resolution here is generic over entity_type, so reading one
    # hardcoded name raised KeyError for every non-parcel reference.
    if ref_key is None:
        ref_key = f'{entity_type}_id' if entity_type else 'parcel_id'
    if ref_key not in links.columns:
        raise KeyError(
            f'Link sidecar {sidecar_path.name} has no {ref_key!r} column; '
            f'found {list(links.columns)}. Set ref_key to the reference '
            'id column the harmonize overlay wrote.'
        )

    links = links.dropna(subset=[ref_key])
    if not include_below_threshold and 'link' in links.columns:
        links = links[links['link'].notna()]
    if 'area_intersection_m2' in links.columns:
        links = links.sort_values('area_intersection_m2', ascending=False)

    collected = links.groupby(id_col)[ref_key].agg(
        lambda s: '|'.join(s.dropna().astype(str).unique())
    )
    curated[column] = collected.where(collected != '').reindex(curated.index)

    # Like the harmonizer's `_join_distinct` convention, leave the `_all`
    # column missing where it would just repeat the base id.
    base_column = column.removesuffix('_all')
    if base_column != column and base_column in curated.columns:
        same = (
            curated[column].astype('string') == curated[base_column].astype('string')
        ).fillna(False)
        curated[column] = curated[column].mask(same)

    if state.verbose:
        n_multi = int(curated[column].str.contains(r'\|', na=False).sum())
        print(
            f'  collect_link_ids: {column} set for '
            f'{int(curated[column].notna().sum()):,}/{len(curated):,} rows '
            f'({n_multi:,} multi-reference).'
        )
    state.curated = curated
    return state


@_register('merge_enrichments', phase='gather')
def merge_enrichments(
    state: CurateState,
    recipes: list[dict],
) -> CurateState:
    """Merge enrichment evidence into canonical columns.

    Each recipe specification must contain ``recipe_id`` and a ``columns``
    mapping from evidence-column names to canonical-column names. Existing
    canonical values take precedence; enrichment fills missing values.
    An optional ``record_source`` (default ``False``) writes a ``{column}
    _source`` provenance sidecar for that spec's columns -- reserve this
    for genuinely canonical, possibly-multi-sourced columns (e.g. a value
    later reconciled against a second source); leave it off for a bulk,
    single-source enrichment batch (e.g. dozens of physical/environmental
    predictors from one model), where a sidecar per column is pure
    bookkeeping bloat with no downstream reconciliation to inform.

    A ``recipe_spec`` whose enrichment output doesn't exist yet for this admin
    is skipped rather than raising -- e.g. an imagery-dependent enrichment
    intentionally left unrun this pass (see
    ``notebooks/examples/US_curate_footprints.ipynb``'s ``--no_streetview``/
    ``--no_googlesatellite``, which skip billing-costly image ingestion and
    the enrichment steps that depend on it). A single requested column
    missing from evidence that *does* exist is tolerated the same way and
    skipped on its own (warned when verbose) -- e.g. a per-source predictor
    that simply was never computed for this admin unit's geography (a legacy
    reference dataset's SSURGO-derived soil columns, present for the states
    it originally covered, can be genuinely absent for a newly added one).
    Every *other* column in the same ``recipe_spec`` still merges normally.

    Every column this step touches -- across every ``recipe_spec``, values
    and source sidecars alike -- is collected in memory and written to
    ``curated`` in a single batched ``pd.concat`` at the end, rather than
    one ``curated[col] = ...`` assignment per column: a wide, already
    highly-columned entity table (e.g. the curated parcel table, 170+
    columns) fragments its internal block manager under dozens of
    one-at-a-time inserts, tripping pandas' own
    ``PerformanceWarning: DataFrame is highly fragmented`` -- exactly the
    pattern a large ``merge_enrichments`` call (e.g. ~45 columns from one
    enrichment source) produces.

    An enrichment's output is found under the spine it enriched, which is
    the curated entity's own unless a spec names another with
    ``entity_recipe``. The building curate reads the enrichments of
    outlines that way (``entity_recipe: US_footprint-spine-2026``), and
    joins them on its ``key`` column, the building's ``footprint_id``,
    rather than on its index: an outline's roof shape is what every
    building under that outline stands under. Without ``entity_recipe``
    such an output is looked for under the building spine, found missing
    and skipped as if the enrichment had not run.
    """
    from openplaces.io.curator.provenance import _apply_source_mask, source_column

    curated = state.curated
    pending: dict[str, pd.Series] = {}

    for recipe_spec in recipes:
        recipe_id = recipe_spec.get('recipe_id')
        columns = recipe_spec.get('columns') or {}
        want_source = recipe_spec.get('record_source', False)
        if not recipe_id:
            raise ValueError("Enrichment specifications require 'recipe_id'.")
        if not columns:
            raise ValueError(
                f"Enrichment specification '{recipe_id}' requires 'columns'."
            )

        enrichment_recipe = get_recipe_by_id(recipe_id)
        if enrichment_recipe.get('stage') != 'enrich':
            raise ValueError(f"Evidence recipe '{recipe_id}' must have stage 'enrich'.")
        key = recipe_spec.get('key')
        if key is not None and key not in curated.columns:
            raise ValueError(
                f"merge_enrichments: '{recipe_id}' joins on '{key}', which the "
                'curated table does not have.'
            )
        evidence_path = get_output_path(
            enrichment_recipe,
            state.admin_id,
            entity_recipe_id=recipe_spec.get('entity_recipe') or state.entity_recipe,
        )
        if not evidence_path.exists():
            if state.verbose:
                print(
                    f"  merge_enrichments: '{recipe_id}' has no evidence for "
                    f'{state.admin_id} (enrichment not run this pass?); skipping.'
                )
            continue
        evidence = read_parquet(evidence_path)

        # Provenance token: 'source_id-version' (e.g. 'brails-2026'), an explicit
        # spec 'source_label', or the recipe id as a fallback. The loaded recipe's
        # dataset is a DataSet object (attributes), but tolerate a raw dict too.
        ds = enrichment_recipe.get('dataset')
        source_id = _field(_field(ds, 'source'), 'source_id')
        version = _field(ds, 'version')
        token = recipe_spec.get('source_label')
        if not token:
            token = f'{source_id}-{version}' if source_id and version else source_id
        token = token or recipe_id

        for evidence_column, canonical_column in columns.items():
            if evidence_column not in evidence:
                if state.verbose:
                    print(
                        f"  merge_enrichments: '{evidence_column}' missing from "
                        f'{evidence_path.name} (not computed for this admin unit?); '
                        'skipping.'
                    )
                continue
            if key is None:
                values = evidence[evidence_column].reindex(curated.index)
            else:
                values = curated[key].map(evidence[evidence_column])

            # A prior recipe_spec in this same call may have already
            # queued a value for this column -- read that first so
            # declaration order still acts as a priority chain (a later
            # spec only fills rows still missing), matching the
            # pre-batching combine_first behavior exactly.
            current = pending.get(canonical_column)
            if current is None and canonical_column in curated.columns:
                current = curated[canonical_column]
            if current is not None:
                was_null = current.isna()
                merged = current.combine_first(values)
            else:
                was_null = pd.Series(True, index=curated.index)
                merged = values
            pending[canonical_column] = merged

            if want_source:
                filled = was_null & merged.notna()
                if filled.any():
                    side = source_column(canonical_column)
                    side_existing = pending.get(side)
                    if side_existing is None and side in curated.columns:
                        side_existing = curated[side]
                    pending[side] = _apply_source_mask(
                        side_existing, curated.index, filled, token
                    )

    if pending:
        new_cols = pd.DataFrame(pending, index=curated.index)
        overlap = [c for c in new_cols.columns if c in curated.columns]
        if overlap:
            curated = curated.drop(columns=overlap)
        curated = pd.concat([curated, new_cols], axis=1)

    state.curated = curated
    return state


@_register('derive_admin_attribute', phase='gather')
def derive_admin_attribute(
    state: CurateState,
    output: str,
    attribute: str | None = None,
    segment: str | None = None,
    scheme: str = 'us-census-geoid',
    admin_id_column: str = 'admin3_id',
    admin_level: int | None = None,
) -> CurateState:
    """Copy a column from the admin spine onto every entity in its unit.

    The admin spine already knows things about a unit that an entity in it
    cannot state for itself -- its name, its type, the code some other
    authority assigns it. This joins one such column on, keyed by the
    admin id the entity already carries.

    The motivating use is a **stable external identifier**. openplaces'
    own ``admin3_id`` is internal and can be re-minted: a re-mint may hand
    ``US-NC-CM`` to a different county than it named before, so an id
    written into a shipped file is only as durable as the mint behind it.
    A federal county FIPS code is not openplaces' to change, so an
    inventory that carries one stays joinable to Census, FEMA and every
    other federal product across a re-mint that renames everything else.

    Reads the spine through :func:`~openplaces.io.readers.get_admin`
    rather than any hard-coded table, so which attribute is surfaced is a
    recipe's decision and no geography lives in this function.

    Two ways to name what to copy, and only one of them is safe for a
    code whose meaning varies by geography. Pass *attribute* to copy a
    spine column verbatim. Pass *segment* to take one named part of the
    unit's national code, resolved through
    :mod:`~openplaces.io.admin_codes.segments`, which is what a federal
    identifier needs: ``admin3_id_admin1`` holds a 5-digit county FIPS
    in 44 states and a 10-digit county-subdivision GEOID in the six New
    England states, so copying it verbatim writes two different things
    into one column. ``segment='county'`` returns the county FIPS in
    both.

    Parameters
    ----------
    output : str
        Column to write onto the entity.
    attribute : str, optional
        Column on the admin spine to copy verbatim. Mutually exclusive
        with *segment*; exactly one of the two is required.
    segment : str, optional
        Name of a part of the unit's own national code, in the source's
        vocabulary (e.g. ``'county'``, ``'state'``). Mutually exclusive
        with *attribute*.
    scheme : str, optional
        Coding scheme *segment* belongs to, default
        ``'us-census-geoid'``.
    admin_id_column : str, optional
        Entity column holding the admin id to look up (default
        ``admin3_id``).
    admin_level : int, optional
        Spine level to read. Inferred from *admin_id_column* when it is
        named ``admin{N}_id``; required otherwise.
    """
    from openplaces.io.readers import get_admin

    if (attribute is None) == (segment is None):
        raise ValueError(
            'derive_admin_attribute: pass exactly one of attribute= or '
            'segment=. Use segment= for a federal code whose width varies '
            "by geography (e.g. segment='county')."
        )

    curated = state.curated
    if admin_id_column not in curated.columns:
        if state.verbose:
            print(
                f'  derive_admin_attribute: {admin_id_column} absent, '
                f'{output} not derived.'
            )
        return state

    if admin_level is None:
        import re

        match = re.fullmatch(r'admin(\d+)_id', admin_id_column)
        if match is None:
            raise ValueError(
                f'derive_admin_attribute: cannot infer admin_level from '
                f'{admin_id_column!r}; pass admin_level explicitly.'
            )
        admin_level = int(match.group(1))

    try:
        spine = get_admin(state.admin_id, level=admin_level)
    except Exception as exc:  # noqa: BLE001 - a missing spine is not fatal
        if state.verbose:
            print(f'  derive_admin_attribute: no admin level {admin_level}: {exc}')
        return state
    if spine is None or not len(spine):
        return state

    spine = spine.reset_index()
    wanted = attribute if attribute is not None else f'{admin_id_column}_admin1'
    if admin_id_column not in spine.columns or wanted not in spine.columns:
        if state.verbose:
            print(
                f'  derive_admin_attribute: spine has no '
                f'{wanted!r}/{admin_id_column!r}; {output} not derived.'
            )
        return state

    lookup = (
        spine.dropna(subset=[admin_id_column])
        .drop_duplicates(admin_id_column)
        .set_index(admin_id_column)[wanted]
    )
    if segment is not None:
        from openplaces.io.admin_codes.segments import slice_segment

        # Not strict: the spine is global and a curate run may cover
        # units whose national code belongs to another scheme entirely.
        # Those come back missing rather than aborting the recipe.
        lookup = slice_segment(lookup, segment=segment, scheme=scheme, strict=False)
    curated[output] = (
        curated[admin_id_column].astype(object).map(lookup).astype('string')
    )
    state.curated = curated

    if state.verbose:
        n = int(curated[output].notna().sum())
        print(f'  derive_admin_attribute: {output} set for {n:,} rows.')
    return state
