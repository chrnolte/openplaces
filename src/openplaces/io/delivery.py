"""
Split a curated entity's region-wide output into a shareable file bundle.

A curated entity file carries everything: the canonical attributes a user
actually models with, the evidence columns those were reconciled from, and the
geometry. Pooled over a whole region that is one unwieldy file -- 2.6M
Eastern-NC footprints come to roughly a gigabyte -- and most consumers need
only a slice of it.

`export_delivery` writes that same content as four files sharing one index, so
a consumer loads only the part they need and can rejoin the rest at any time:

=================  =========================================================
`{base}`           canonical attributes only (plain table, no geometry)
`{base}_point`     the same attributes on centroid points (GeoParquet)
`{base}_geo`       entity boundary polygons only (GeoParquet)
`{base}_evidence`  every remaining column (plain table)
=================  =========================================================

Which columns count as canonical is declared per recipe under a `share` block,
not inferred here: the curated schema holds far more columns that are
technically canonical than belong in a compact delivery, and that editorial
choice is the recipe author's.
"""

import stat
from pathlib import Path

import geopandas as gpd
import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.geo.polygon import points_from_coords
from openplaces.io import parquet_columns, read_parquet, to_parquet
from openplaces.io.bundle_terms import bundle_terms, format_notice
from openplaces.io.readers import get_admin
from openplaces.recipe import (
    get_output_path,
    get_process_admin_level,
    get_recipe_by_id,
)

# Mirrors `openplaces.io.curator.provenance.SOURCE_SUFFIX`, which
# this module sits below in the layer hierarchy, and so cannot import.
SOURCE_SUFFIX = '_source'

# Written by the split-layout and covering-bbox machinery; meaningless
# outside the file that carries them.
INTERNAL_COLUMNS = ('_join_id', 'bbox')

# Registry attributes holding the entity centroid in WGS84. Kept as
# plain columns in the canonical table, and consumed into the point
# geometry (so dropped) in the `_point` file.
COORDINATE_COLUMNS = ('long', 'lat')


def delivery_regions(recipe) -> list[dict]:
    """Return one delivery spec per region the recipe ships, in declared order.

    A curation recipe is written once and shipped to more than one region:
    the CHEER footprint recipe covers both 45 Eastern NC counties and 42
    coastal Texas ones, from identical curation logic. The regions are
    therefore data, not recipe structure -- named in
    ``share: delivery: regions:`` and defined once in the shared registry
    that :func:`~openplaces.io.readers.get_regions` reads, so mapping and
    ad-hoc analysis group by the same county lists the bundle ships.

    A recipe that instead lists ``admin_ids`` inline still works and yields
    a single unnamed region, so a grouping used in exactly one place need
    not be registered.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.

    Returns
    -------
    list of dict
        Each with ``region_id`` (None when declared inline), ``admin_level``,
        ``admin_ids``, and ``admin_id`` (the bundle's own unit, None when it
        should be derived from the members).
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    block = dict((recipe.get('share') or {}).get('delivery') or {})
    if not block:
        return []

    default_level = block.get('admin_level', 2)
    names = block.get('regions')
    if not names:
        if not block.get('admin_ids'):
            return []
        return [
            {
                'region_id': None,
                'admin_level': default_level,
                'admin_ids': [str(a) for a in block['admin_ids']],
                'admin_id': block.get('admin_id'),
            }
        ]

    from openplaces.io.readers import get_regions

    if isinstance(names, str):
        names = [names]
    regions = []
    for name in names:
        rows = get_regions(name)
        anchor = rows.get('region_admin_id', pd.Series(dtype=str))
        anchor = anchor[anchor.astype(str).str.strip().ne('')]
        regions.append(
            {
                'region_id': str(name),
                'admin_level': default_level,
                'admin_ids': list(dict.fromkeys(rows['admin_id'].dropna())),
                'admin_id': str(anchor.iat[0]) if len(anchor) else None,
            }
        )
    return regions


def _select_region(recipe, region=None) -> dict:
    """Return one region's spec, by ``region_id`` or the only one there is.

    Raises when *region* is unknown, and when a multi-region recipe is
    asked for "the" region without naming which -- guessing there would
    silently ship the wrong region's file under the other's name.
    """
    regions = delivery_regions(recipe)
    if not regions:
        return {}
    if region is None:
        if len(regions) > 1:
            names = ', '.join(str(r['region_id']) for r in regions)
            raise ValueError(
                f'Recipe {recipe.get("recipe_id", recipe)!r} declares '
                f'{len(regions)} delivery regions ({names}); name one via '
                f'region=.'
            )
        return regions[0]
    for spec in regions:
        if spec['region_id'] == str(region):
            return spec
    names = ', '.join(str(r['region_id']) for r in regions)
    raise KeyError(f'Unknown delivery region {region!r}; declared: {names}.')


def delivery_spec(recipe, region=None) -> dict:
    """Return the recipe's `share: delivery:` block for one region.

    Declares which admin unit the region-wide bundle covers
    (``admin_level``) and which units feed it (``admin_ids``). Named in the
    recipe rather than passed per run so the orchestrator can build and ship
    a region unattended, and can recognize a narrower run as not covering it.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    region : str, optional
        Which declared region to return. Required when the recipe ships
        more than one; see :func:`delivery_regions`.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    return _select_region(recipe, region)


def delivery_admin_id(recipe, admin_level=None, region=None) -> AdminId:
    """Return the admin unit a recipe's bundle covers.

    Derived from the declared members rather than the recipe's own admin ID:
    a recipe is filed at the scope it can process (``US`` for the CHEER
    footprint recipe) while its bundle covers only the region actually
    delivered (``US-NC``). Falls back to the recipe's admin ID when no
    members are declared, and to an explicit ``bundle_admin_id`` when the
    region's members do not all sit under one unit at *admin_level*.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    admin_level : int, optional
        Level to truncate to. Defaults to the declared value, then 2.
    region : str, optional
        Which declared region to resolve; see :func:`delivery_regions`.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    spec = delivery_spec(recipe, region)
    if admin_level is None:
        admin_level = spec.get('admin_level', 2)
    if spec.get('admin_id'):
        return AdminId(str(spec['admin_id']))
    members = spec.get('admin_ids') or []
    anchor = AdminId(str(members[0])) if members else AdminId(str(recipe['admin_id']))
    return AdminId(*anchor.levels[:admin_level])


def _unlock(path: Path) -> None:
    """Restore write permission on a shipped file so it can be rewritten."""
    if path.exists():
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def unlock_delivery(recipe, **kwargs) -> None:
    """Clear the read-only bit on a recipe's bundle files, if they exist.

    `export_delivery` does this for itself. The separate entry point is for
    an orchestrator, which has to clear the bit *before* it plans the job:
    Snakemake refuses to run any rule whose output file is read-only --
    whatever set the bit, not only its own `protected()` -- and would
    otherwise abort with ProtectedOutputException on every reship.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    **kwargs
        Forwarded to `delivery_paths` (admin_id, admin_level, output_dir).
    """
    for path in delivery_paths(recipe, **kwargs).values():
        _unlock(path)


def _lock(path: Path) -> None:
    """Drop write permission on a shipped file.

    The bundle is what leaves this repository, so between runs it sits on
    disk read-only: a stray process, a hand edit, or a GIS client cannot
    overwrite it. `export_delivery` unlocks its own outputs before rewriting
    them, so this guards against everything except the step that owns them.
    Deliberately not Snakemake's `protected()`, which additionally refuses to
    ever regenerate the file and would abort the workflow on every reship.
    """
    if path.exists():
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def share_columns(recipe) -> tuple[list[str], list[str]]:
    """Return the canonical and point-only columns a recipe declares as shared.

    The `share: columns:` block is the recipe's own statement of which of
    its columns are canonical -- the compact set a reader of the delivery
    is meant to work from, as opposed to the far larger set of columns
    that merely resolve to a registry attribute. Diagnostics that want to
    report on "the canonical columns" should read it here rather than
    re-deriving a set from the registry, which cannot tell an attribute
    from the evidence column feeding it.

    Parameters
    ----------
    recipe : dict
        Loaded recipe.

    Returns
    -------
    tuple of (list of str, list of str)
        Canonical columns, then the extra columns carried by the point
        file alone. Both empty when the recipe declares no `share` block:
        not every recipe is delivered, so absence is a normal state here
        and callers that require the list raise on their own behalf (see
        :func:`_share_spec`).
    """
    share = recipe.get('share') or {}
    return list(share.get('columns') or []), list(share.get('point_columns') or [])


def _share_spec(recipe):
    """Return the recipe's `share` block, erroring when it declares no columns."""
    columns, point_columns = share_columns(recipe)
    if not columns:
        raise ValueError(
            f'Recipe {recipe.get("recipe_id", recipe)!r} has no `share: columns:` '
            'block, so there is no canonical column list to deliver. Add one to '
            'the recipe naming the attributes the shared file should carry.'
        )
    return columns, point_columns


def _source_columns(columns, available):
    """Return the `{column}_source` sidecars available for *columns*.

    `geometry_source` is included whenever present: geometry is delivered as
    its own file rather than as a canonical column, so it never appears in
    *columns*, but which source each outline came from is part of what makes
    the canonical table readable.
    """
    available = set(available)
    sidecars = [
        f'{column}{SOURCE_SUFFIX}'
        for column in columns
        if f'{column}{SOURCE_SUFFIX}' in available
    ]
    if f'geometry{SOURCE_SUFFIX}' in available:
        sidecars.append(f'geometry{SOURCE_SUFFIX}')
    return sidecars


def delivery_members(recipe, admin_id=None, admin_ids=None, region=None) -> list[str]:
    """Return the process-level admin IDs whose output the bundle pools.

    Resolution order: an explicit *admin_ids* argument, then the recipe's
    declared `share: delivery: admin_ids:`, then every process-level child of
    *admin_id*. The declared list has to win over the hierarchy walk: a
    region is rarely all of a state's counties (the CHEER region is 45 of
    North Carolina's 100), and silently pooling the rest would ship a
    different dataset than the one the recipe describes.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    admin_id : str or `AdminId`, optional
        Admin unit the bundle covers, used only for the hierarchy fallback.
    admin_ids : list of str, optional
        Explicit member list, overriding the recipe.
    region : str, optional
        Which declared region to resolve; see :func:`delivery_regions`.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    if admin_ids is None:
        admin_ids = delivery_spec(recipe, region).get('admin_ids')
    if admin_ids is None:
        if admin_id is None:
            return []
        process_level = get_process_admin_level(recipe)
        admin_ids = get_admin(AdminId(admin_id), process_level).index
    elif isinstance(admin_ids, str | AdminId):
        admin_ids = [admin_ids]

    return list(dict.fromkeys(str(aid) for aid in admin_ids))


def _resolve_inputs(recipe, admin_id, admin_ids, region=None):
    """Return the process admin level and the [(admin_id, path)] files to pool."""
    inputs = []
    for process_id in delivery_members(recipe, admin_id, admin_ids, region):
        path = get_output_path(recipe, process_id)
        if path.exists():
            inputs.append((process_id, path))
    return get_process_admin_level(recipe), inputs


def delivery_paths(
    recipe, admin_id=None, admin_level=None, output_dir=None, region=None
) -> dict:
    """Return the four bundle paths, keyed by their role.

    The single source of truth for where a delivery lands: `export_delivery`
    writes exactly these, and the orchestrator declares exactly these as the
    job's outputs, so the two cannot drift apart.

    Cannot go through `recipe.get_output_path` alone, which rejects an admin
    unit coarser than the recipe's own save level. The bundle is deliberately
    coarser (a state-level file pooling county-level ones), so the level is
    overridden on a recipe copy first, the same way `io.aggregate` does it.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    admin_id : str or `AdminId`, optional
        Admin unit the bundle covers. Defaults to `delivery_admin_id`.
    admin_level : int, optional
        Admin level of the bundle. Defaults to the recipe's declared
        `share: delivery: admin_level:`, then 2.
    output_dir : str, optional
        Directory to write into, as named in STANDARD_DIRS. Defaults to the
        recipe's declared value, then the recipe's own `save_to: data_dir:`.
    region : str, optional
        Which declared region to resolve; see :func:`delivery_regions`.

    Returns
    -------
    dict of str to pathlib.Path
        Keyed 'canonical', 'point', 'geo', 'evidence', 'terms'.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    spec = delivery_spec(recipe, region)

    if admin_level is None:
        admin_level = spec.get('admin_level', 2)
    if output_dir is None:
        output_dir = spec.get('output_dir')
    if admin_id is None:
        admin_id = delivery_admin_id(recipe, admin_level, region)

    output_recipe = dict(recipe)
    output_recipe['save_to'] = {
        **recipe.get('save_to', {}),
        'admin_level': admin_level,
    }
    if output_dir is not None:
        output_recipe['save_to']['data_dir'] = output_dir

    canonical = get_output_path(output_recipe, admin_id)

    # Two declared regions can roll up to the same admin unit (both
    # Boston test regions are Massachusetts towns), which would give
    # their bundles identical filenames in one directory - the second
    # delivery silently overwriting the first. Nest each colliding
    # region's bundle in a subdirectory named after it; regions with a
    # unit of their own (the CHEER NC/TX pair) keep their flat paths.
    if region is not None:
        siblings = [
            r['region_id']
            for r in delivery_regions(recipe)
            if r.get('region_id') is not None
        ]
        if len(siblings) > 1 and region in siblings:
            unit_of = {
                rid: str(delivery_admin_id(recipe, admin_level, rid))
                for rid in siblings
            }
            mine = unit_of[region]
            if sum(unit == mine for unit in unit_of.values()) > 1:
                canonical = canonical.parent / str(region) / canonical.name

    return {
        'canonical': canonical,
        'point': canonical.with_stem(f'{canonical.stem}_point'),
        'geo': canonical.with_stem(f'{canonical.stem}_geo'),
        'evidence': canonical.with_stem(f'{canonical.stem}_evidence'),
        # Ships with the data because the obligations do: whoever receives
        # the bundle needs to know what its sources require of them.
        'terms': canonical.with_stem(f'{canonical.stem}_LICENSE').with_suffix('.txt'),
    }


ACCURACY_DIR_NAME = 'accuracies'


def delivery_accuracy_dir(recipe, **kwargs) -> Path:
    """Return the directory holding a delivery's own accuracy reporting.

    A sibling of the four bundle files rather than a cache location, because
    how well the inventory scores is part of what is delivered: someone who
    receives the bundle and not this repository still needs to know what its
    columns are worth. It travels with the data or it is not really published.

    Only aggregate tables and figures belong here. Row-level validation
    artifacts stay in the cache -- linking survey points to entities carries
    their addresses through, and the bundle is shared.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    **kwargs
        Passed to :func:`delivery_paths` (``admin_id``, ``admin_level``,
        ``output_dir``) so the directory follows the bundle it describes.

    Returns
    -------
    pathlib.Path
        The ``accuracies`` directory. Not created; the writer does that.
    """
    return delivery_paths(recipe, **kwargs)['canonical'].parent / ACCURACY_DIR_NAME


def _deduplicate(frame, coverage_columns, admin_id_column):
    """Drop repeated index entries, keeping the best-covered copy of each.

    An entity sitting essentially on a county line is ingested and curated
    independently by both neighboring counties, and the two copies' centroids
    land in the same Open Location Code cell -- so they share an entity id.
    Keep whichever copy carries more non-null canonical attributes, breaking
    ties on the admin id so the result does not depend on read order.
    """
    if not frame.index.has_duplicates:
        return frame, 0

    coverage = frame[coverage_columns].notna().sum(axis=1)
    ordered = frame.assign(_coverage=coverage).sort_values(
        ['_coverage', admin_id_column], ascending=[False, True], kind='stable'
    )
    kept = ordered[~ordered.index.duplicated(keep='first')].drop(columns='_coverage')
    return kept, len(frame) - len(kept)


def export_delivery(
    recipe,
    admin_id=None,
    admin_ids=None,
    admin_level=None,
    output_dir=None,
    region=None,
    verbose=False,
):
    """Write a curated entity's region-wide output as a four-file bundle.

    Pools the per-process-unit curated files of the region, resolves entities
    that two neighboring units both curated, and writes the canonical table,
    its centroid-point twin, the boundary polygons, and the evidence
    supplement. All four are indexed by the entity id and share one row order,
    so any two of them rejoin 1:1 on that index.

    The written files are left read-only, since they are what leaves this
    repository; a later call unlocks and rewrites its own outputs.

    Every parameter but *recipe* falls back to the recipe's own
    ``share: delivery:`` block, so a fully declared recipe delivers with
    ``export_delivery(recipe_id)``.

    Parameters
    ----------
    recipe : str or dict
        Curation recipe ID (e.g. ``'US_footprint-openplaces-2026'``) or a
        pre-loaded recipe dict. Must declare a ``share: columns:`` block.
    admin_id : str or `AdminId`, optional
        Admin unit the bundle covers, e.g. ``'US-NC'``. Names the output
        files. Defaults to `delivery_admin_id`.
    admin_ids : list of str, optional
        Process-level admin IDs to pool. Defaults to the recipe's declared
        members, then every process-level child of *admin_id*. IDs with no
        curated file on disk are skipped.
    admin_level : int, optional
        Admin level of the output files. Defaults to the recipe's declared
        value, then 2 (state/region).
    output_dir : str, optional
        Directory the bundle is written to, as named in ``STANDARD_DIRS``.
        Defaults to the recipe's declared value, then its own output bucket.
    region : str, optional
        Which declared region to ship, by ``region_id``. Required when the
        recipe declares more than one; see :func:`delivery_regions`.
    verbose : bool, optional
        Print a line per written file.

    Returns
    -------
    dict of str to pathlib.Path
        The written paths, keyed ``'canonical'``, ``'point'``, ``'geo'``,
        ``'evidence'``.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    canonical_columns, point_columns = _share_spec(recipe)
    paths = delivery_paths(recipe, admin_id, admin_level, output_dir, region)
    if admin_id is None:
        admin_id = delivery_admin_id(recipe, admin_level, region)

    process_level, inputs = _resolve_inputs(recipe, admin_id, admin_ids, region)
    if not inputs:
        raise FileNotFoundError(
            f'No curated output found under {admin_id} for '
            f'{recipe.get("recipe_id", recipe)!r}; nothing to deliver.'
        )
    admin_id_column = f'admin{process_level}_id'
    for path in paths.values():
        _unlock(path)

    # Two read passes rather than one, so the wide evidence columns
    # are never in memory at the same time as the polygons: pooled
    # over a region, either one alone runs to several GB.
    declared = [*canonical_columns, *point_columns]
    frames = []
    for process_id, path in inputs:
        available = parquet_columns(path)
        wanted = [column for column in declared if column in available]
        wanted += _source_columns(canonical_columns, available)
        part = read_parquet(path, geom=True, columns=wanted)
        # A declared column a single unit happens to lack is
        # filled, not dropped, so every unit has the same schema.
        for column in declared:
            if column not in part.columns:
                part[column] = pd.NA
        part[admin_id_column] = process_id
        frames.append(part)

    pooled = gpd.GeoDataFrame(pd.concat(frames), crs=frames[0].crs)
    frames.clear()

    coverage_columns = [c for c in canonical_columns if c != admin_id_column]
    pooled, n_dropped = _deduplicate(pooled, coverage_columns, admin_id_column)
    # Sort by entity id so all four files share one row order, not
    # just one set of ids: deduplication reorders by coverage, and the
    # evidence pass rebuilds rows county by county, so neither lands
    # in file order on its own.
    pooled = pooled.sort_index()
    if n_dropped and verbose:
        print(
            f'Resolved {n_dropped} entity id(s) curated by two adjacent '
            f'{admin_id_column} units; kept the better-covered copy.'
        )

    source_columns = _source_columns(canonical_columns, pooled.columns)
    # Sources trail the values they explain, so the table opens on
    # the attributes and the provenance block stays out of the way.
    canonical = pd.DataFrame(pooled[[*canonical_columns, *source_columns]])
    to_parquet(canonical, paths['canonical'])

    point = points_from_coords(
        pooled[
            [
                *[c for c in canonical_columns if c not in COORDINATE_COLUMNS],
                *point_columns,
                *source_columns,
                *COORDINATE_COLUMNS,
            ]
        ]
    )
    to_parquet(point, paths['point'], schema_version='1.1.0')

    to_parquet(pooled[['geometry']], paths['geo'], schema_version='1.1.0')

    # Which copy of each entity survived deduplication, so pass two
    # keeps the matching evidence row rather than an arbitrary one.
    kept = pooled[admin_id_column]
    n_entities = len(pooled)
    # Copied out before `pooled` is released: the licence notice reports
    # each source's share of the geometry actually delivered.
    geometry_source = (
        pooled['geometry_source'].copy()
        if 'geometry_source' in pooled.columns
        else None
    )
    del pooled, canonical, point

    delivered = {*declared, *source_columns, *INTERNAL_COLUMNS, 'geometry'}
    evidence_frames = []
    for process_id, path in inputs:
        available = parquet_columns(path)
        wanted = [column for column in available if column not in delivered]
        # Sources ride along in both files: dictionary-encoded, they
        # cost almost nothing, and either file reads on its own.
        wanted += _source_columns(canonical_columns, available)
        part = read_parquet(path, geom=False, columns=wanted)
        evidence_frames.append(part[kept.reindex(part.index).eq(process_id)])

    evidence = pd.concat(evidence_frames).sort_index()
    evidence_frames.clear()
    to_parquet(evidence, paths['evidence'])

    terms = bundle_terms(recipe, geometry_source)
    paths['terms'].write_text(format_notice(recipe, terms, admin_id), encoding='utf-8')

    for path in paths.values():
        _lock(path)

    if verbose:
        for role in ('canonical', 'point', 'geo', 'evidence'):
            size_mb = paths[role].stat().st_size / 1024**2
            print(f'{role:>9}: {paths[role].name} ({size_mb:,.0f} MB)')
        print(f'{n_entities:,} entities from {len(inputs)} {admin_id_column} unit(s)')
        for license_text, share in sorted(
            terms['share_alike'].items(), key=lambda item: -item[1]
        ):
            print(
                f'{share:.1%} of the geometry is {license_text}: publishing '
                f'this bundle requires {license_text} and attribution. '
                f'See {paths["terms"].name}.'
            )
        if terms['unrecorded']:
            unchecked = ', '.join(e['source_id'] for e in terms['unrecorded'])
            print(f'Terms not recorded for: {unchecked}')

    return paths
