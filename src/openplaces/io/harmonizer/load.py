"""
Pipeline step that restores a geospine recipe's results into a fresh
HarmonizeState, so an attribute-only recipe can run downstream of the
expensive geometry phase without recomputing any of it.

The geometry/attribute recipe split: a geospine recipe (e.g.
US_footprint-geospine-2026) runs the geometry-phase steps -- spine
resolution, spatial overlays and point joins, overlap resolution -- and
persists its spine plus one link sidecar per join. The attribute recipe
(keeping the established id, e.g. US_footprint-spine-2026) declares the
geospine as its `entity_recipe` and starts its pipeline with
`load_geospine`, which reloads the spine and rebuilds the crosswalks,
overlays, and prepared references from the persisted link tables --
never from a new spatial computation.
"""

from __future__ import annotations

import json

import pandas as pd

from openplaces.core.schema import SourceGeometryType
from openplaces.geo.link import get_entity_link_path
from openplaces.io import to_parquet
from openplaces.io.aggregate import read_file_metadata
from openplaces.io.harmonizer import (
    HARMONIZE_METADATA_KEY,
    HarmonizeState,
    _register,
)
from openplaces.io.harmonizer.links import (
    _build_crosswalk,
    _link_fingerprint,
    _load_link_sidecar,
    _load_point_link_sidecar,
    _point_step_config,
    _prepare_reference,
    _resolve_reference_recipe,
    overlay_step_config,
    reference_rows,
    snap_chained_links,
)
from openplaces.io.readers import get_entities
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_id,
    get_save_admin_level,
    raise_if_coverage_complete,
    source_id_from_recipe_id,
)

#: Footer key naming the geospine whose sidecar a persisted re-keyed
#: link table was made from (`load_geospine(persist_links=True)`).
LINK_REKEYED_FROM_KEY = 'openplaces:link_rekeyed_from'


@_register('load_geospine')
def load_geospine(
    state: HarmonizeState,
    entity_recipe_id: str | None = None,
    links_recipe_id: str | None = None,
    links_key: str | None = None,
    persist_links: bool = False,
):
    """Restore the geospine recipe's spine and link products into *state*.

    Reads everything from what the geospine recipe persisted -- its
    entity output (attributes + `_geo` sidecar), its footer-carried
    pipeline metadata, and one link sidecar per `link_to_reference`
    step -- and rebuilds the state the geometry phase would have left
    behind: `spine`, `crosswalks`, `overlays`, prepared `references`
    (overlay links only; nothing downstream reads a point reference),
    `reference_types`, `source_geometry_types`, and the
    `inferred_from_{recipe_id}` metadata frames `reconcile_attributes`
    uses to attribute parcel-inferred spine rows directly.

    Which links to restore, and with what configuration, is read from the
    geospine recipe's own pipeline -- deliberately not re-declared in the
    attribute recipe, so the two YAMLs cannot drift. Each sidecar's footer
    fingerprint is recomputed exactly as the geospine run computed it
    (same recipe, same step position, same config subset); a missing or
    mismatched sidecar raises with instructions to rerun the geospine
    recipe rather than silently recomputing geometry here.

    A spine established from another entity's geospine (the building
    geospine from the footprint geospine) computes its point links
    itself but takes its parcel overlay from the rows it was built from.
    *links_recipe_id* names that geospine, and its sidecars are restored
    first, the same way, and re-keyed to this spine through *links_key*, the
    column holding each row's id in that geospine (`footprint_id`). The
    re-keying is a rename while the relation is one to one; a row without
    a source id gets no links, and several rows sharing one source id are
    refused, because fanning a link out to them is a decision (slice b3
    of the building spine) and not a lookup. A row built from several
    source rows names only one of them, so the others' links would
    stay unattributed: merging rows needs a link table as its key, not
    this column. The spine's source ids must also be exactly the
    source geospine's ids, or this spine was built from another version
    of it and the step raises.

    With *persist_links* the re-keyed sidecars are also written beside
    this spine's own geospine output, under the paths
    `geo.link.get_entity_link_path(<this geospine>, <reference>)`
    resolves, so that the curate steps that read a link sidecar by the
    curated entity's geospine (`link_curated_entity`,
    `apportion_curated_values`, `collect_link_ids`) find the building's
    parcel pairs where they look for them. The written
    copy is the source sidecar's rows with the id column renamed and
    re-keyed, its footer fingerprint kept, and `LINK_REKEYED_FROM_KEY`
    naming the geospine it came from; the harmonize-side loader never
    reads these copies, so the source sidecars stay the only thing it
    validates (plans/building-spine-consumers-and-footprint-projection.md).

    Parameters
    ----------
    entity_recipe_id : str or dict, optional
        Geospine recipe (id or loaded dict) to load. Defaults to the
        recipe's own `entity_recipe` key.
    links_recipe_id : str, optional
        Another geospine whose link sidecars apply to this spine's rows
        through *links_key*.
    links_key : str, optional
        Column of this spine holding the row's id in *links_recipe_id*.
        Defaults to that recipe's `{entity_type}_id`.
    persist_links : bool, optional
        Write the re-keyed sidecars beside this spine's geospine output
        (needs *links_recipe_id*). Default False.
    """
    geospine = entity_recipe_id or state.recipe.get('entity_recipe')
    if not geospine:
        raise ValueError(
            'load_geospine needs an entity_recipe: declare the geospine '
            "recipe under the attribute recipe's `entity_recipe` key."
        )
    if isinstance(geospine, str):
        geospine = get_recipe_by_id(geospine)
    geospine = _as_run_for(geospine, state.admin_id)
    geospine_id = get_recipe_id(geospine)

    spine = get_entities(geospine, state.admin_id, geom=True, missing='raise')
    if state.verbose:
        print(f'  Load (geospine): {len(spine):,d} {geospine_id} entities')

    _restore_handoff_metadata(state, geospine)

    # Mirror resolve_spine's working-index rename: overlay crosswalks name
    # the reference level 'parcel_id', so a parcel spine's own index must
    # not use that name while the pipeline runs. The harmonizer's save
    # step restores the original from metadata['spine_index_name'].
    if spine.index.name == 'parcel_id':
        state.metadata['spine_index_name'] = spine.index.name
        spine.index = spine.index.rename('spine_id')
    state.spine = spine
    spine_id_col = spine.index.name

    # The other geospine's links first: they were computed first (that
    # geospine runs before this one), and the attribute steps iterate the
    # crosswalks in the order they arrive, so the parcel overlay keeps
    # its place ahead of the point links this geospine computed itself.
    if links_recipe_id is not None:
        state = _restore_foreign_links(
            state,
            links_recipe_id,
            links_key,
            spine_id_col,
            persist_to=geospine_id if persist_links else None,
        )
    elif persist_links:
        raise ValueError(
            "load_geospine: persist_links needs links_recipe_id; a geospine's "
            'own sidecars are already beside its output.'
        )

    for step_index, step_cfg in enumerate(geospine.get('pipeline') or []):
        if not isinstance(step_cfg, dict):
            continue
        if step_cfg.get('step') != 'link_to_reference':
            continue
        if step_cfg.get('save_link') is False:
            continue
        state = _restore_link(state, geospine, step_index, step_cfg, spine_id_col)

    if state.timer:
        state.timer.mark('Load (geospine)')
    return state


@_register('load_links', phase='geometry')
def load_links(
    state: HarmonizeState,
    links_recipe_id: str,
    links_key: str | None = None,
):
    """Restore another geospine's links onto this geospine's rows.

    The geometry-phase counterpart of `load_geospine`'s
    *links_recipe_id*, for a geospine built from another one's rows
    (the building geospine from the footprint geospine). It restores
    that geospine's link sidecars, re-keyed to this spine's index
    through *links_key*, as crosswalks, overlays and prepared
    references, so that a later `link_to_reference` step finds them
    where it would find a link this recipe computed: the point joins
    read the first polygon overlay for their same-parcel pass, their
    cross-parcel filter and the parcel id they attach. Reading the
    persisted link, rather than overlaying parcels again, keeps one
    parcel overlay per outline, computed where the outline geometry
    steps need it (plans/building-spine-consumers-and-footprint-projection.md).

    Nothing is written. A link computed after this step depends on what
    it restored, so its fingerprint records each restored sidecar's
    content hash (`restored_link_hashes`): a rerun of the other geospine
    makes it stale even where no ingest input of this recipe moved.

    Parameters
    ----------
    links_recipe_id : str
        The geospine whose link sidecars are restored.
    links_key : str, optional
        Column of this spine holding the row's id in *links_recipe_id*.
        Defaults to that recipe's `{entity_type}_id`.
    """
    # The labeled crosswalk, not the rebuilt one: the point joins ran
    # against the geospine's in-memory crosswalk, which keeps every
    # 'condo cluster' pair, and the rebuild trims the small ones by
    # min_fraction_of_largest. Measured 2026-10-26 on US-NC-CUR: the
    # rebuild drops 44 labeled pairs and adds 1, which moved one
    # building's Overture address aggregation.
    state = _restore_foreign_links(
        state,
        links_recipe_id,
        links_key,
        state.spine.index.name,
        labeled_crosswalk=True,
    )
    if state.timer:
        state.timer.mark('Load (links)')
    return state


def restored_link_hashes(recipe: dict, step_index: int, admin_id) -> dict:
    """Content hashes of the sidecars `load_links` steps restore.

    Covers the `load_links` steps of *recipe* before *step_index*, keyed
    by sidecar file name; empty when there are none, which is every
    recipe but a geospine built from another one. Computed from the
    recipe and the files alone, so the geospine loader recomputes the
    same value the writing run stored. A sidecar that does not exist
    hashes to None (a link the other geospine skipped for a coverage
    gap).
    """
    from openplaces.io.harmonizer.links.sidecars import _hash_file

    hashes = {}
    for step_cfg in (recipe.get('pipeline') or [])[:step_index]:
        if not isinstance(step_cfg, dict) or step_cfg.get('step') != 'load_links':
            continue
        other = _as_run_for(get_recipe_by_id(step_cfg['links_recipe_id']), admin_id)
        other_id = get_recipe_id(other)
        for other_cfg in other.get('pipeline') or []:
            if not isinstance(other_cfg, dict):
                continue
            if other_cfg.get('step') != 'link_to_reference':
                continue
            if other_cfg.get('save_link') is False:
                continue
            resolved_id, _ = _resolve_reference_recipe(
                other_cfg.get('recipe_id'), other_cfg.get('entity_type'), admin_id
            )
            if resolved_id is None:
                continue
            path = get_entity_link_path(other_id, resolved_id, admin_id)
            hashes[path.name] = _hash_file(path) if path.exists() else None
    return hashes


def _as_run_for(geospine: dict, admin_id) -> dict:
    """The geospine recipe as its run for *admin_id* saw it, patches applied.

    The links' fingerprints record the configs of the geospine's prior
    geometry steps as the unit ran them, so a unit patch of the
    geospine (DC's parcel ranking) has to be applied here too, or every
    sidecar it shaped reads as stale.
    """
    if admin_id is None:
        return geospine
    from openplaces.recipe import apply_recipe_patches

    patched, _ = apply_recipe_patches(geospine, admin_id)
    return patched


def _restore_foreign_links(
    state: HarmonizeState,
    links_recipe_id: str,
    links_key: str | None,
    spine_id_col: str,
    persist_to: str | None = None,
    labeled_crosswalk: bool = False,
) -> HarmonizeState:
    """Restore another geospine's links and re-key them to this spine.

    See `load_geospine`. The sidecars are read exactly as that geospine's
    own attribute recipe would read them (its recipe, its step positions,
    its fingerprints), keyed by *links_key*; every crosswalk and overlay
    that arrives is then re-keyed to this spine's index through the
    *links_key* column. The inferred-row frames need no re-keying: they
    are rebuilt from this spine's own provenance columns. With
    *labeled_crosswalk* each overlay's crosswalk is taken from the
    sidecar's link labels (see `_restore_link`).
    """
    other = _as_run_for(get_recipe_by_id(links_recipe_id), state.admin_id)
    other_id = get_recipe_id(other)
    if links_key is None:
        entity = other.get('entity')
        links_key = f'{entity.entity_type}_id' if entity is not None else 'entity_id'
    spine = state.spine
    if links_key not in spine.columns:
        raise ValueError(
            f'load_geospine: links_recipe_id={other_id!r} needs the column '
            f'{links_key!r} on the spine to re-key its links, and the spine has '
            'no such column.'
        )
    keys = spine[links_key]
    present = keys.notna()
    duplicated = keys[present].duplicated()
    if duplicated.any():
        raise ValueError(
            f'load_geospine: {int(duplicated.sum()):,d} spine rows share a '
            f'{links_key} with another row; re-keying {other_id} links onto '
            'several rows is a decision the slice that splits rows has to make.'
        )
    mapping = pd.Series(spine.index[present.to_numpy()], index=keys[present].to_numpy())
    _require_same_rows(state, other, keys[present])

    already = set(state.crosswalks)
    for step_index, step_cfg in enumerate(other.get('pipeline') or []):
        if not isinstance(step_cfg, dict):
            continue
        if step_cfg.get('step') != 'link_to_reference':
            continue
        if step_cfg.get('save_link') is False:
            continue
        state = _restore_link(
            state,
            other,
            step_index,
            step_cfg,
            links_key,
            labeled_crosswalk=labeled_crosswalk,
        )

    for recipe_id in list(state.crosswalks):
        if recipe_id in already:
            continue
        state.crosswalks[recipe_id] = _rekey_frame(
            state.crosswalks[recipe_id], links_key, spine_id_col, mapping
        )
        if recipe_id in state.overlays:
            state.overlays[recipe_id] = _rekey_frame(
                state.overlays[recipe_id], links_key, spine_id_col, mapping
            )
        if persist_to is not None:
            _persist_rekeyed_sidecar(
                get_entity_link_path(other_id, recipe_id, state.admin_id),
                get_entity_link_path(persist_to, recipe_id, state.admin_id),
                other_id,
                links_key,
                spine_id_col,
                mapping,
                verbose=state.verbose,
            )
    if state.verbose:
        n_new = len(set(state.crosswalks) - already)
        print(
            f'  Load (geospine): {n_new} {other_id} links re-keyed '
            f'{links_key} -> {spine_id_col}'
        )
    return state


def _require_same_rows(state: HarmonizeState, other: dict, keys: pd.Series) -> None:
    """Refuse links from a geospine this spine was not built from.

    This spine (the building geospine's rows) names each row's source by
    *keys*; the source geospine's ids are geometry hashes, so a rebuilt
    source with any changed outline has ids this spine does not carry,
    and this spine has ids the source dropped. Re-keying across that
    gap gave the dropped rows no evidence and the new outlines no row,
    silently. Ids rather than file times, because a sync tool touching
    a file changes its mtime and not its content.
    """
    source_ids = get_entities(other, state.admin_id, geom=False, columns=[]).index
    stale = keys[~keys.isin(source_ids)]
    new = source_ids.difference(pd.Index(keys))
    if len(stale) or len(new):
        raise ValueError(
            f'load_geospine: this spine was built from another version of '
            f'{get_recipe_id(other)} for {state.admin_id} ({len(stale):,d} of its '
            f'rows are gone there, {len(new):,d} rows there are new). Rerun '
            'the geospine this spine is built from (the building geospine) '
            'before this recipe.'
        )


def _persist_rekeyed_sidecar(
    source_path,
    target_path,
    source_recipe_id: str,
    old: str,
    new: str,
    mapping: pd.Series,
    verbose: bool = False,
) -> None:
    """Write one link sidecar re-keyed to another spine's ids.

    The file, not the in-memory frame, is what is re-keyed: the loader
    drops the `link` label and the chain column from the overlay it
    restores, and the curate readers of a sidecar need them (only
    link-labeled pairs take part in apportionment). Every row is kept
    and every footer entry copied; `LINK_REKEYED_FROM_KEY` is added so
    a reader can tell the copy from a sidecar a geospine computed. A
    source sidecar that does not exist (a link the geospine skipped for
    a coverage gap) writes nothing.
    """
    if source_path is None or not source_path.exists():
        return
    flat = pd.read_parquet(source_path)
    if old not in flat.columns:
        return
    flat = _rekey_frame(flat, old, new, mapping)
    metadata = {
        key: value
        for key, value in read_file_metadata(source_path).items()
        if isinstance(key, str) and key.startswith('openplaces:')
    }
    metadata[LINK_REKEYED_FROM_KEY] = source_recipe_id
    target_path.parent.mkdir(parents=True, exist_ok=True)
    to_parquet(flat, target_path, file_metadata=metadata, index=False)
    if verbose:
        print(f'  Load (geospine): wrote re-keyed link sidecar {target_path.name}')


def _rekey_frame(frame, old: str, new: str, mapping: pd.Series):
    """Replace ids named *old* on *frame* by their *mapping* image, as *new*.

    Works on the level of a MultiIndex, on a single index, or on a column,
    whichever carries *old*. An id with no image keeps its value under
    the new name, and no row is dropped: a link sidecar records pairs for
    rows a later geometry step removed from the spine (`resolve_overlaps`
    runs after the parcel overlay in the footprint geospine), the
    attribute steps already tolerate keys the spine lacks, and dropping
    those rows here changed what the steps compute over the crosswalk.
    Measured 2026-09-29 on US-WI-VI: 62 footprints in the parcel sidecar
    and not on the spine, 199 crosswalk rows, whose absence moved the
    tie-break between equal overlaps on 14 footprints and the per-parcel
    footprint count on 53. A frame carrying *old* nowhere is returned
    unchanged.
    """
    index = frame.index
    if old in (index.names or []):
        level = list(index.names).index(old)
        original = index.get_level_values(level)
        images = mapping.reindex(original)
        images = images.where(images.notna(), pd.Series(original, index=images.index))
        frame = frame.copy()
        if isinstance(index, pd.MultiIndex):
            arrays = [index.get_level_values(i) for i in range(index.nlevels)]
            arrays[level] = pd.Index(images.to_numpy())
            names = list(index.names)
            names[level] = new
            frame.index = pd.MultiIndex.from_arrays(arrays, names=names)
        else:
            frame.index = pd.Index(images.to_numpy(), name=new)
        return frame
    if old in frame.columns:
        frame = frame.copy()
        values = frame[old]
        images = pd.Series(
            mapping.reindex(values.to_numpy()).to_numpy(), index=frame.index
        )
        frame[old] = images.where(images.notna(), values)
        return frame.rename(columns={old: new})
    return frame


@_register('adopt_entity_attributes')
def adopt_entity_attributes(
    state: HarmonizeState,
    recipe_id: str,
    key: str | None = None,
    rename: dict[str, str] | None = None,
) -> HarmonizeState:
    """Copy another spine's attributes onto this one through a one-to-one key.

    The footprint spine is a projection of the building spine while one
    building is one footprint: every attribute the building carries
    (parcel evidence, addresses, permit counts) is the outline's too, and
    the consumers of the footprint spine (the parcel geospine's morphology
    and dwelling-address steps, the imagery recipes, the footprint curate)
    read it there until they read the building spine directly. That is
    why this step exists and why it copies wide columns onto another
    entity, which harmonize otherwise avoids: the copy is transitional,
    and the alternative, rerunning every consumer against the building
    spine in one move, would change six recipes' inputs at once.

    Every column of *recipe_id*'s output except *key* and the geometry is
    written onto the spine: a column the spine already has is overwritten
    in place (keeping its position), a new one is appended in the other
    spine's order, so the projected table has the column order the
    attribute steps produced when they ran on this spine. Count columns
    named for the other entity are renamed through *rename*
    (`n_parcels_per_building` to `n_parcels_per_footprint`); a rename
    naming a column the other spine lacks is ignored, because a count a
    link writes is absent wherever its source does not apply.

    Parameters
    ----------
    recipe_id : str
        The spine whose attributes are adopted (`US_building-spine-2026`).
    key : str, optional
        Column of that spine holding this spine's id. Defaults to this
        spine's index name. A row of the other spine without a key is
        ignored (a building with no outline); several rows sharing one
        key are refused, as is a spine row no other row names and a
        keyed row naming no spine row, because each means the relation
        is no longer one to one, or the two spines were built from
        different versions of the outlines.
    rename : dict, optional
        Column renames applied to the adopted columns.
    """
    spine = state.spine
    if spine is None or len(spine) == 0:
        return state
    own_id = spine.index.name
    key = key or own_id
    other = get_entities(recipe_id, state.admin_id, geom=False, missing='raise')
    if key not in other.columns:
        raise ValueError(
            f'adopt_entity_attributes: {recipe_id} has no column {key!r} naming '
            f"this spine's {own_id}."
        )
    other = other[other[key].notna()]
    duplicated = other[key].duplicated()
    if duplicated.any():
        raise ValueError(
            f'adopt_entity_attributes: {int(duplicated.sum()):,d} rows of '
            f'{recipe_id} share a {key}; the projection onto {own_id} needs '
            'one row per key.'
        )
    other = other.set_index(key)
    missing = spine.index.difference(other.index)
    if len(missing):
        raise ValueError(
            f'adopt_entity_attributes: {len(missing):,d} {own_id} rows have no '
            f'row in {recipe_id}; rerun it for {state.admin_id} first.'
        )
    # The other way round too: a keyed row this spine lacks means the
    # other spine was built from rows this one no longer has, and
    # dropping it in the reindex below hid that.
    extra = other.index.difference(spine.index)
    if len(extra):
        raise ValueError(
            f'adopt_entity_attributes: {len(extra):,d} rows of {recipe_id} name '
            f'a {own_id} this spine lacks; it was built from another version '
            f"of this spine's geospine. Rerun it for {state.admin_id} first."
        )
    other = other.reindex(spine.index).drop(columns='geometry', errors='ignore')
    # A rename naming a column the other spine lacks is not an error: a
    # count written by a link to a source that does not apply to the unit
    # (a permit recipe held out of a checkout, a county without permits)
    # is legitimately absent, and the projection copies what exists.
    other = other.rename(columns=dict(rename or {}))
    for column in other.columns:
        spine[column] = other[column]
    state.spine = spine
    if state.verbose:
        print(
            f'  adopt_entity_attributes: {other.shape[1]} columns of {recipe_id} '
            f'projected onto {len(spine):,d} {own_id} rows'
        )
    if state.timer:
        state.timer.mark('Adopt (attributes)')
    return state


def _restore_handoff_metadata(state: HarmonizeState, geospine: dict) -> None:
    """Restore the footer-carried pipeline metadata of the geospine output.

    The geospine's save step wrote spine_index_name,
    spine_source_recipe_ids, and spine_keep_columns into its attribute
    parquet's footer (in-memory HarmonizeState.metadata does not survive
    the recipe split); `link_by_id(auto_discover=True)` in the attribute
    recipe needs the two sets to protect per-geometry keep_columns from
    being pooled. An output written before the split carries no footer
    entry -- restore nothing and leave the defaults.
    """
    admin_id = state.admin_id
    if admin_id is not None:
        admin_id = admin_id.truncate_to_level(get_save_admin_level(geospine))
    path = get_output_path(geospine, admin_id)
    stored_raw = read_file_metadata(path).get(HARMONIZE_METADATA_KEY)
    if stored_raw is None:
        return
    try:
        handoff = json.loads(stored_raw)
    except json.JSONDecodeError:
        return
    if handoff.get('spine_index_name') is not None:
        state.metadata['spine_index_name'] = handoff['spine_index_name']
    for key in ('spine_source_recipe_ids', 'spine_keep_columns'):
        if handoff.get(key):
            state.metadata[key] = set(handoff[key])


def _restore_link(
    state: HarmonizeState,
    geospine: dict,
    step_index: int,
    step_cfg: dict,
    spine_id_col: str,
    labeled_crosswalk: bool = False,
) -> HarmonizeState:
    """Rebuild one link_to_reference step's state from its sidecar.

    An overlay's crosswalk is rebuilt from the overlay by the link
    step's own thresholds. With *labeled_crosswalk* it is instead the
    sidecar's labeled rows, in file order: the pairs the geospine's
    crosswalk held when it wrote the file, including the 'condo
    cluster' pairs that consolidation adds without the trim.
    """
    resolved_id, resolved_et = _resolve_reference_recipe(
        step_cfg.get('recipe_id'), step_cfg.get('entity_type'), state.admin_id
    )
    if resolved_id is None:
        if state.verbose:
            print(
                '  Load (geospine): no reference recipe for '
                f'entity_type={step_cfg.get("entity_type")!r}; skipping link.'
            )
        return state

    join = step_cfg.get('join', 'spatial_overlay')
    empty_reference = False
    thresholds = step_cfg.get('thresholds') or {}
    sidecar_path = get_entity_link_path(
        get_recipe_id(geospine), resolved_id, state.admin_id
    )

    # Recompute the fingerprint exactly as the geospine run computed it:
    # same recipe, same pipeline position (prior geometry-phase steps),
    # same step-config subset. A stale sidecar then fails closed below.
    shim = HarmonizeState(
        recipe=geospine, admin_id=state.admin_id, verbose=False, timer=None
    )
    shim.step_index = step_index

    if join == 'spatial_overlay':
        min_fraction = thresholds.get('min_fraction_of_largest', 1 / 6)
        area_min_m2 = thresholds.get('area_intersection_m2_min', 10)
        aggregation_function = step_cfg.get('aggregation_function')
        # The same reference rows the geospine run overlaid: with
        # kept_by, the rows another spine's resolve_spine keeps.
        ref_raw, kept_config = reference_rows(
            resolved_id, state.admin_id, step_cfg.get('kept_by')
        )
        fingerprint = _link_fingerprint(
            shim,
            resolved_id,
            overlay_step_config(
                min_fraction,
                area_min_m2,
                step_cfg.get('sort_by'),
                step_cfg.get('list_columns'),
                aggregation_function,
                kept_config,
            ),
        )
        overlay = _load_link_sidecar(
            sidecar_path, fingerprint, spine_id_col, verbose=state.verbose
        )
        if ref_raw is None or len(ref_raw) == 0:
            # The geospine run skipped this link for the same reason (an
            # expected admin-scoped coverage gap), so there is nothing to
            # restore -- unless a sidecar exists, which means the input
            # disappeared after the link ran. A reference declaring
            # complete coverage escalates instead (see
            # recipe.coverage_is_complete).
            if overlay is None:
                raise_if_coverage_complete(resolved_id, state.admin_id)
                return state
            raise RuntimeError(
                f'Link sidecar {sidecar_path.name} exists but its reference '
                f'{resolved_id} has no data for {state.admin_id}; rerun '
                f'{get_recipe_id(geospine)} for this admin unit.'
            )
        if overlay is None:
            raise RuntimeError(
                f'Missing or stale link sidecar {sidecar_path} for '
                f'{resolved_id}; rerun {get_recipe_id(geospine)} for '
                f'{state.admin_id} to recompute the geometry phase.'
            )
        ref_polys = _prepare_reference(
            ref_raw,
            resolved_id,
            resolved_et,
            state,
            aggregation_function=aggregation_function,
            sort_by=step_cfg.get('sort_by'),
            list_columns=step_cfg.get('list_columns'),
        )
        crosswalk = _build_crosswalk(overlay, spine_id_col, min_fraction, area_min_m2)
        if thresholds.get('snap_chains'):
            crosswalk, _ = snap_chained_links(
                crosswalk,
                spine_id_col,
                fraction_max=float(thresholds.get('chain_fraction_max', 0.75)),
            )
        if labeled_crosswalk:
            flat = pd.read_parquet(sidecar_path)
            flat = flat[flat['link'].notna()].set_index([spine_id_col, 'parcel_id'])
            crosswalk = flat[[c for c in crosswalk.columns if c in flat.columns]]
        state.references[resolved_id] = ref_polys
        state.crosswalks[resolved_id] = crosswalk
        state.overlays[resolved_id] = overlay
        _restore_inferred(state, resolved_id, resolved_et)
    elif join == 'spatial_point':
        fingerprint = _link_fingerprint(
            shim,
            resolved_id,
            _point_step_config(
                thresholds,
                step_cfg.get('remap_id'),
                step_cfg.get('source_geometry_type'),
            ),
        )
        linked = _load_point_link_sidecar(
            sidecar_path, fingerprint, verbose=state.verbose
        )
        # An empty file is what the geospine writes for a reference with
        # no points in the unit; it restores exactly as no file did, by
        # the reference check below.
        if linked is not None and linked.empty:
            linked = None
        if linked is None:
            probe = get_entities(
                resolved_id, state.admin_id, geom=False, missing='ignore', columns=[]
            )
            if probe is None or len(probe) == 0:
                # Same rule as the overlay branch above: the geospine
                # run skipped this link for an expected admin-scoped
                # coverage gap, so nothing is left to restore. One that
                # declares complete coverage and produced nothing is a
                # vanished input, not a gap, and escalates.
                raise_if_coverage_complete(resolved_id, state.admin_id)
                # Fall through to the type bookkeeping below rather than
                # returning: reconcile_attributes names this reference's
                # (all-null) evidence columns from it.
                linked = None
                empty_reference = True
            if not empty_reference:
                raise RuntimeError(
                    f'Missing or stale link sidecar {sidecar_path} for '
                    f'{resolved_id}; rerun {get_recipe_id(geospine)} for '
                    f'{state.admin_id} to recompute the geometry phase.'
                )
        if linked is not None:
            state.crosswalks[resolved_id] = linked
    else:
        raise ValueError(f'Unknown join mode in geospine pipeline: {join!r}')

    if resolved_et:
        state.reference_types[resolved_id] = resolved_et
    sgt = step_cfg.get('source_geometry_type')
    if sgt is not None:
        state.source_geometry_types[resolved_id] = SourceGeometryType(sgt)
    return state


def _restore_inferred(
    state: HarmonizeState, recipe_id: str, entity_type: str | None
) -> None:
    """Rebuild the inferred-footprint metadata frame for one overlay link.

    infer_spine_additions stamps each parcel-inferred spine row with
    `geometry_source = '{entity_type}.{source_id}'` and records the source
    entity's id in `geometry_source_id`; `reconcile_attributes` attributes
    those rows directly (no area weighting) from
    `metadata['inferred_from_{recipe_id}']`, which held the seeding
    reference ids in a 'parcel_id' column. Reconstruct exactly that frame
    from the two persisted spine columns.
    """
    spine = state.spine
    if 'geometry_source' not in spine.columns:
        return
    if 'geometry_source_id' not in spine.columns:
        return
    _et = entity_type or recipe_id.rsplit('_', 1)[-1].split('-', 1)[0]
    token = f'{_et}.{source_id_from_recipe_id(recipe_id)}'
    mask = spine['geometry_source'].astype('string').eq(token).fillna(False)
    if not mask.any():
        return
    inferred = pd.DataFrame(
        {'parcel_id': spine.loc[mask, 'geometry_source_id']}, index=spine.index[mask]
    )
    state.metadata[f'inferred_from_{recipe_id}'] = inferred
