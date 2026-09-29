"""Link sidecars: the validity fingerprint over step configs, prior
geometry-phase steps and ingest inputs, and reading and writing the
sidecar files for overlay and point links.
"""

from __future__ import annotations

import json

import pandas as pd

from openplaces.io import to_parquet
from openplaces.io.aggregate import read_file_metadata
from openplaces.io.cleanup import read_receipt
from openplaces.io.harmonizer import (
    _STEP_PHASES,
    HarmonizeState,
)
from openplaces.io.harmonizer.links._shared import (
    _LINK_INDEX_KEY,
    _LINK_METADATA_KEY,
)
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    get_save_admin_level,
)


def _fingerprint_safe_step(step_cfg: dict) -> dict:
    """A pipeline entry reduced to its fingerprint-relevant config.

    Drops ``save_link`` (toggling persistence must never force a
    recompute) and the chain-snapping thresholds (``snap_chains`` /
    ``chain_fraction_max`` are geometry-free relabeling applied after the
    overlay, deliberately excluded since format 1).
    """
    entry = {k: v for k, v in step_cfg.items() if k != 'save_link'}
    thresholds = entry.get('thresholds')
    if isinstance(thresholds, dict):
        entry['thresholds'] = {
            k: v
            for k, v in thresholds.items()
            if k not in ('snap_chains', 'chain_fraction_max')
        }
    return entry


def _link_fingerprint(
    state: HarmonizeState, ref_recipe_id: str, step_config: dict
) -> dict:
    """Validity fingerprint stored in (and checked against) a link sidecar.

    Records the step configuration, the ordered configs of every
    geometry-phase pipeline entry that ran before this step (format 2 --
    the spine reaching the join is shaped by those steps, so a changed
    spine threshold invalidates the sidecar even though the mid-pipeline
    spine itself is deliberately not fingerprinted), and the size/mtime
    of every resolvable ingest-stage input of the harmonize recipe for
    this admin unit (the reference parquet among them). A source that
    was deliberately deleted stays verifiable through its tombstone
    receipt's recorded size/mtime; a missing source with no receipt
    yields nulls, which no longer match once the file reappears (fail
    safe: recompute).

    Comparison is :func:`_fingerprints_match`, not raw equality: the
    written copy additionally carries a per-source content sha256
    (stamped by :func:`_with_source_hashes`), which validates a source
    whose mtime moved but whose bytes did not.
    """
    from openplaces.io.cleanup import _relative_posix
    from openplaces.io.harmonizer import _load_steps

    # Phase tags live on the @_register decorators, so every step module
    # must be imported before _STEP_PHASES is consulted -- a caller that
    # reached this function without going through the dispatch loop (the
    # geospine loader, a test) may not have triggered the module imports.
    _load_steps()

    prior_geometry_steps = []
    if state.step_index is not None:
        for prior in (state.recipe.get('pipeline') or [])[: state.step_index]:
            if not isinstance(prior, dict):
                continue
            if _STEP_PHASES.get(prior.get('step')) == 'geometry':
                prior_geometry_steps.append(_fingerprint_safe_step(prior))

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
        'format': 2,
        'spine_recipe_id': get_recipe_id(state.recipe),
        'ref_recipe_id': ref_recipe_id,
        'admin_id': str(state.admin_id) if state.admin_id is not None else None,
        'step_config': step_config,
        'prior_geometry_steps': prior_geometry_steps,
        'sources': sources,
    }


def _hash_file(path) -> str:
    """Streaming SHA-256 of a file's full content."""
    import hashlib

    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _with_source_hashes(fingerprint: dict) -> dict:
    """Return a copy of *fingerprint* with each source content-hashed.

    Called at sidecar write time only, so a load never pays for hashing
    unless an mtime actually moved. Measured cost of the full-content
    hashes at write: 8 ms for a typical county's 8 inputs, 223 ms for
    Harris (431 MB, dominated by the statewide parcel file); noise
    against the minutes the geometry phase just spent.
    """
    from openplaces.io.cleanup import _resolve_relative

    out = dict(fingerprint)
    sources = []
    for entry in fingerprint.get('sources') or []:
        entry = dict(entry)
        try:
            path = _resolve_relative(entry.get('path'))
            if entry.get('size') is not None and path.exists():
                entry['sha256'] = _hash_file(path)
        except OSError:
            pass
        sources.append(entry)
    out['sources'] = sources
    return out


def _fingerprints_match(stored: dict, fresh: dict) -> bool:
    """Decide whether a stored fingerprint still validates a sidecar.

    Exact equality passes. Otherwise every part must match except
    source mtimes, and each source whose mtime moved must hash to the
    sha256 the sidecar recorded at write time. This is what lets a
    sync-tool touch (Dropbox re-hydration bumped every cache mtime on
    2026-08-24 with byte-identical content, reading as region-wide
    staleness) cost one hash check instead of a full geometry rerun,
    while any actual content change still fails closed. A stored entry
    without a hash (pre-hash sidecars) keeps the old behavior: an
    mtime move alone marks it stale.
    """
    from openplaces.io.cleanup import _resolve_relative

    if stored == fresh:
        return True
    if {k: v for k, v in stored.items() if k != 'sources'} != {
        k: v for k, v in fresh.items() if k != 'sources'
    }:
        return False
    stored_sources = stored.get('sources')
    fresh_sources = fresh.get('sources')
    if not isinstance(stored_sources, list) or not isinstance(fresh_sources, list):
        return False
    if len(stored_sources) != len(fresh_sources):
        return False
    for old, new in zip(stored_sources, fresh_sources):
        if old.get('path') != new.get('path') or old.get('size') != new.get('size'):
            return False
        if old.get('mtime') == new.get('mtime'):
            continue
        digest = old.get('sha256')
        if not digest:
            return False
        try:
            path = _resolve_relative(old['path'])
            if not path.exists() or _hash_file(path) != digest:
                return False
        except OSError:
            return False
    return True


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
    if not _fingerprints_match(stored, fingerprint):
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
        file_metadata={
            _LINK_METADATA_KEY: json.dumps(_with_source_hashes(fingerprint))
        },
    )
    if verbose:
        print(f'  Link (overlay): wrote link sidecar {sidecar_path.name}')


def _load_point_link_sidecar(sidecar_path, fingerprint: dict, verbose: bool = False):
    """Reload a persisted point crosswalk iff its fingerprint matches.

    Returns the geometry-free flat crosswalk with the reference's native
    index restored (its name is stored beside the fingerprint, since a
    reference index may be unnamed), or None when the sidecar is absent
    or invalid (recompute, fail safe). Only the footer is read for the
    validity check.
    """
    if sidecar_path is None or not sidecar_path.exists():
        return None
    metadata = read_file_metadata(sidecar_path)
    stored_raw = metadata.get(_LINK_METADATA_KEY)
    if stored_raw is None:
        return None
    try:
        stored = json.loads(stored_raw)
    except json.JSONDecodeError:
        return None
    if not _fingerprints_match(stored, fingerprint):
        if verbose:
            print('  Link (point): sidecar fingerprint mismatch; recomputing links.')
        return None
    linked = pd.read_parquet(sidecar_path)
    index_name = metadata.get(_LINK_INDEX_KEY) or 'index'
    if index_name in linked.columns:
        linked = linked.set_index(index_name)
        if index_name == 'index':
            linked.index.name = None
    if verbose:
        print(f'  Link (point): reloaded link sidecar {sidecar_path.name}')
    return linked


def _write_point_link_sidecar(
    sidecar_path, linked: pd.DataFrame, fingerprint: dict, verbose: bool = False
) -> None:
    """Persist the flat point crosswalk, geometry-free.

    Unlike the overlay sidecar this is the *final* crosswalk (after every
    pass, filter, and aggregation), so nothing is recomputed on the
    reload path and the sidecar is only written when computed fresh.
    Proximity-pass rows can still carry the reference geometry; it is
    dropped here (no downstream consumer reads crosswalk geometry), so
    fresh and reloaded crosswalks differ only in that column.
    """
    flat = pd.DataFrame(linked.drop(columns='geometry', errors='ignore'))
    index_name = flat.index.name or 'index'
    to_parquet(
        flat.reset_index(),
        sidecar_path,
        file_metadata={
            _LINK_METADATA_KEY: json.dumps(_with_source_hashes(fingerprint)),
            _LINK_INDEX_KEY: index_name,
        },
    )
    if verbose:
        print(f'  Link (point): wrote link sidecar {sidecar_path.name}')
