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

import os
import stat
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from openplaces.core.constants import ENVELOPE_ID_COLUMN
from openplaces.core.schema import AdminId
from openplaces.geo.polygon import points_from_coords
from openplaces.io import (
    coerce_mixed_object_columns,
    parquet_columns,
    read_parquet,
    to_parquet,
)
from openplaces.io.delivery.redaction import (
    _names_attribute,
    find_restricted,
    merge_counts,
    units_supplied,
    withhold,
)
from openplaces.io.delivery.slicing import SLICED_FROM, slice_at_unit_lines
from openplaces.io.delivery.terms import (
    bundle_terms,
    format_notice,
    known_source_ids,
    restricted_inputs,
    share_alike_conflict_text,
    share_alike_family_named,
    share_alike_inputs,
    value_source_cells,
)
from openplaces.io.readers import get_admin
from openplaces.io.tables import redacted_share_writes
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

# A team bundle's region id is its base region's plus this suffix, and
# its files sit in this subdirectory beside the base region's bundle.
TEAM_SUFFIX = '.team'
TEAM_DIR = 'team'

# Where bundles are filed. Curated per-unit files stay in their recipe's
# own, private bucket: they carry restricted sources' values, which
# only the bundle withholds.
DELIVERY_DIR = 'share'


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
        ``admin_ids``, ``admin_id`` (the bundle's own unit, None when it
        should be derived from the members), `whole_unit` (True for a
        region listed under `whole_unit_regions`, whose bundle is filed
        as its unit's own; see :func:`delivery_paths`) and
        `withhold_share_alike` (the share-alike families whose values the
        bundle leaves out, from `withhold_share_alike`; always empty for
        a team bundle, which is never published).
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
                # An inline grouping has no name to file it under, so
                # `delivery_paths` files it as its unit, and refuses one
                # that would land on a curated output.
                'whole_unit': False,
                'withhold_share_alike': _withheld_families(block, None),
            }
        ]

    from openplaces.io.readers import get_regions

    if isinstance(names, str):
        names = [names]
    whole_units = {str(name) for name in block.get('whole_unit_regions') or []}
    undeclared = sorted(whole_units - {str(name) for name in names})
    if undeclared:
        raise KeyError(
            f'whole_unit_regions names {", ".join(map(repr, undeclared))}, '
            'which is not one of the declared delivery regions.'
        )
    withheld = block.get('withhold_share_alike')
    if isinstance(withheld, dict):
        undeclared = sorted({str(n) for n in withheld} - {str(n) for n in names})
        if undeclared:
            raise KeyError(
                f'withhold_share_alike names {", ".join(map(repr, undeclared))}, '
                'which is not one of the declared delivery regions.'
            )
    # One registry read for the whole block, not one per named region:
    # the CSV was re-parsed for each of the seven declared regions, and
    # `delivery_paths` resolves regions repeatedly.
    registry = get_regions()
    regions = []
    for name in names:
        rows = registry[registry['region_id'] == str(name)]
        if rows.empty:
            # Ask the registry directly so its own error, which lists
            # what is registered, is the one the caller sees.
            rows = get_regions(name)
        anchor = rows.get('region_admin_id', pd.Series(dtype=str))
        anchor = anchor[anchor.astype(str).str.strip().ne('')]
        regions.append(
            {
                'region_id': str(name),
                'admin_level': default_level,
                'admin_ids': list(dict.fromkeys(rows['admin_id'].dropna())),
                'admin_id': str(anchor.iat[0]) if len(anchor) else None,
                'audience': 'public',
                'whole_unit': str(name) in whole_units,
                'withhold_share_alike': _withheld_families(block, str(name)),
            }
        )
    # A team bundle is a second delivery of a declared region, for the
    # research team only: it keeps the values of sources a person has
    # cleared for team sharing (`team_sharing_permitted`), which the
    # region's own bundle withholds. Its own id keeps it a separate job
    # and a separate file, so it can never overwrite the public one.
    for name in block.get('team_regions') or []:
        base = next((r for r in regions if r['region_id'] == str(name)), None)
        if base is None:
            raise KeyError(
                f'team_regions names {name!r}, which is not one of the '
                'declared delivery regions.'
            )
        regions.append(
            {
                **base,
                'region_id': f'{name}{TEAM_SUFFIX}',
                'audience': 'team',
                'base_region_id': str(name),
                # Share-alike binds on publication, and a team bundle is
                # never published (maintainer, 2026-10-23), so it keeps
                # every family its region's public bundle withholds.
                'withhold_share_alike': [],
            }
        )
    return regions


def _withheld_families(block, region_id) -> list[str]:
    """The share-alike families a region's bundle leaves out.

    `share: delivery: withhold_share_alike:` maps a region id to the
    families its bundle withholds (`{greenville-nc: [odbl]}`); a plain
    list applies to every region the recipe declares, an inline
    `admin_ids` grouping included. Names resolve through
    `share_alike_family_named`, so a misspelt family raises here rather
    than withholding nothing.
    """
    declared = block.get('withhold_share_alike') or []
    if isinstance(declared, dict):
        declared = (declared.get(region_id) or []) if region_id else []
    if isinstance(declared, str):
        declared = [declared]
    return list(dict.fromkeys(share_alike_family_named(name) for name in declared))


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
    members are declared.

    Every member has to sit under the resulting unit, and a region whose
    members do not raises here rather than shipping under the first
    member's state: the unit was read off `members[0]` alone, so a
    two-state region with no declared anchor was filed, and named, as
    though it were one state's. Declare the region's own unit to fix it,
    as `region_admin_id` in the region registry or `admin_id` in the
    recipe's inline `share: delivery:` block.

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
    if not members:
        anchor = AdminId(str(recipe['admin_id']))
        return AdminId(*anchor.levels[:admin_level])
    anchor = AdminId(str(members[0]))
    unit = AdminId(*anchor.levels[:admin_level])
    outside = [m for m in members if not unit.is_parent_or_equal_of(AdminId(str(m)))]
    if outside:
        raise ValueError(
            f'Delivery region {spec.get("region_id") or "(inline)"!r} of '
            f'{recipe.get("recipe_id", recipe)!r} has members outside '
            f'{unit}, which was derived from its first member: '
            f'{", ".join(str(m) for m in outside[:5])}. Declare the '
            "region's own unit (`region_admin_id` in the region registry, "
            "or `admin_id` in the recipe's `share: delivery:` block)."
        )
    return unit


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
        Forwarded to `delivery_paths` (admin_id, admin_level, output_dir,
        region).
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


PARTIAL_SUFFIX = '.partial'


def _staged(paths: dict) -> dict:
    """Return the temporary path each bundle file is written to first."""
    return {
        role: path.with_name(path.name + PARTIAL_SUFFIX) for role, path in paths.items()
    }


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


def share_is_spatial(recipe) -> bool:
    """Whether the bundle carries geometry, from `share: geometry:`.

    Most curated entities are places and ship four files. A transaction
    is not a place: it is an event whose location belongs to the parcel
    it names, and a deed covering several parcels has no one point at
    all. Such a recipe declares `share: geometry: false` and ships the
    canonical table and its evidence supplement only, rather than
    inventing a centroid to satisfy the point file.

    Default True, so every recipe written before this stays spatial.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary. Pass the dict where one
        is already in hand: `delivery_paths` and `_share_spec` both ask,
        and re-loading by id on each call re-reads the recipe tree.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    return bool((recipe.get('share') or {}).get('geometry', True))


def share_derived_from(recipe) -> dict[str, list[str]]:
    """Return the recipe's `share: derived_from:` block, validated.

    Names columns a curate step computes from other columns without
    writing a `_source` sidecar, so redaction cannot tell from the cell
    whose values it holds:

    .. code-block:: yaml

        share:
          derived_from:
            occupancy_type_parcel: [use_group_combined_parcel, group_parcel]

    A cell of such a column is withheld on every row where one of its
    inputs is (`io.delivery.redaction`, rule 7): the footprint's
    `occupancy_type_parcel` is the roll's use class in occupancy terms,
    and ships only where the roll's class may. Declared in the recipe,
    beside the step that computes the column, because nothing in the
    curated file records the dependency.

    Parameters
    ----------
    recipe : dict
        Loaded recipe.

    Returns
    -------
    dict of str to list of str
        Column to the columns it is computed from; empty when the recipe
        declares none.
    """
    block = (recipe.get('share') or {}).get('derived_from') or {}
    recipe_id = recipe.get('recipe_id', recipe)
    derived = {}
    for column, inputs in block.items():
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list | tuple) or not inputs:
            raise ValueError(
                f'Recipe {recipe_id!r}: `share: derived_from: {column}:` needs '
                'a list of the columns it is computed from.'
            )
        derived[str(column)] = [str(name) for name in inputs]
    return derived


def _units_supplied(inputs, sources, admin_id_column) -> dict[str, set]:
    """Units each restricted source supplied values in, over every file.

    Read once, before either pass, from the sidecars of the columns the
    sources can reach: the canonical pass and the evidence pass each see
    only some of them, and a footprint's estimate in one file can be
    derived from a roll whose token is only in the other.
    """
    supplied: dict[str, set] = {}
    wanted = {
        f'{column}{SOURCE_SUFFIX}'
        for source in sources
        for column in source.get('attributes') or ()
    }
    for process_id, path in inputs:
        columns = [c for c in parquet_columns(path) if c in wanted]
        if not columns:
            continue
        part = read_parquet(path, geom=False, columns=columns)
        part[admin_id_column] = process_id
        for source_id, units in units_supplied(
            pd.DataFrame(part), sources, admin_id_column
        ).items():
            supplied.setdefault(source_id, set()).update(units)
    return supplied


class RestrictedInputError(RuntimeError):
    """A delivery would pass on data whose terms forbid passing it on."""


def _refuse_restricted_cells(
    frame,
    sources,
    admin_id_column,
    known_sources=None,
    supplied=None,
    derived_from=None,
):
    """Raise if any value a restricted source supplied is still in *frame*.

    The backstop behind `io.redaction.withhold`. It runs on what is about
    to be written, so it fires only on restricted data actually present:
    a source that merely feeds the recipe, or whose values were withheld,
    does not stop the delivery. There is no override; shipping such
    values is a decision for the licensor, not for a flag.
    """
    found = find_restricted(
        frame, sources, admin_id_column, known_sources, supplied, derived_from
    )
    if not found['counts']:
        return
    lines = [
        f'  {source_id}: {count["rows"]:,} rows, {count["cells"]:,} values'
        for source_id, count in sorted(found['counts'].items())
    ]
    raise RestrictedInputError(
        'Values from sources whose terms restrict passing their data on '
        'are still in the delivery after redaction:\n'
        + '\n'.join(lines)
        + '\nThe delivery was not written.'
    )


class ShareAlikeConflictError(RuntimeError):
    """A delivery would carry share-alike licences of two families."""


def _refuse_share_alike_conflict(
    terms, recipe, admin_id, region=None, audience='public'
) -> None:
    """Raise if a published bundle carries two share-alike families.

    Maintainer's rule (2026-10-23): a bundle may carry share-alike data
    as long as it is internally compatible, so ODbL or CC BY-SA, never
    both. Each family requires the whole release to carry its licence,
    so a bundle holding two cannot be published under either, and a
    notice saying so is not enough once the files exist. There is no
    override flag: the remedies are a declaration that the region
    withholds one family (`share: delivery: withhold_share_alike:`),
    leaving one family's sources out of the recipe, or a licensor's
    permission to relicense.

    A team bundle is exempt: share-alike binds only on publication, and
    a team bundle is never published (maintainer, 2026-10-23; the CHEER
    team bundle carries Edgecombe and Pitt together, decision of
    2026-10-22, option A). Its notice still names the families.
    """
    if audience == 'team' or not terms.get('share_alike_conflict'):
        return
    name = region or str(admin_id)
    raise ShareAlikeConflictError(
        f'{name} ({recipe.get("recipe_id", "")}) would carry share-alike '
        'licences of more than one family, which no single release can '
        'satisfy:\n'
        + share_alike_conflict_text(terms)
        + '\nDeclare that the region withholds one family '
        f'(share: delivery: withhold_share_alike: {{{name}: [odbl]}}), leave '
        "one family's sources out of the recipe, or obtain permission to "
        'relicense them. The delivery was not written.'
    )


def _bundle_sources(recipe, region, process_ids) -> dict:
    """Who a bundle is for and which sources' values it withholds.

    Returns
    -------
    dict
        ``audience`` ('public' or 'team'); ``restricted``, the restricted
        sources to withhold; ``team_shared``, the sources a team bundle
        keeps; ``share_alike``, every share-alike input (with its
        ``family``); ``withheld_share_alike``, those of a family the
        region declares withheld; and ``withhold``, the entries
        `io.delivery.redaction.withhold` is given.
    """
    # Walked per pooled unit, so the county sources the pipeline
    # auto-discovered there are found too. Only candidates: whether any
    # of their data is in the bundle is decided on the rows.
    restricted = restricted_inputs(recipe, list(process_ids))
    spec = delivery_spec(recipe, region)
    # A team bundle keeps what a person cleared for team sharing; every
    # other bundle, and every other restricted source, is withheld.
    audience = spec.get('audience', 'public')
    team_shared = [
        entry
        for entry in restricted
        if audience == 'team' and entry.get('team_sharing_permitted')
    ]
    restricted = [entry for entry in restricted if entry not in team_shared]
    # A region declared to leave one share-alike family out has that
    # family's values withheld by the same rules as a restricted
    # source's, so what counts as "its values" is defined once.
    share_alike = share_alike_inputs(recipe, list(process_ids))
    # Never in a team bundle, which is not published, and share-alike
    # binds only on publication.
    families = set(spec.get('withhold_share_alike') or [])
    if audience == 'team':
        families = set()
    withheld_share_alike = [
        {**entry, 'share_alike_family': entry['family']}
        for entry in share_alike
        if entry['family'] in families
    ]
    return {
        'audience': audience,
        'restricted': restricted,
        'team_shared': team_shared,
        'share_alike': share_alike,
        'withheld_share_alike': withheld_share_alike,
        'withhold': [*restricted, *withheld_share_alike],
    }


def _share_alike_reach(frame, entries, admin_id_column) -> dict:
    """Rows and cells each share-alike source still has in *frame*.

    Counted by the redaction rules (`find_restricted`), one source id at
    a time, so a source two recipes record (a county's roll and its
    sales table) is counted once over the union of what both reach.
    """
    by_source: dict[str, list[dict]] = {}
    for entry in entries:
        by_source.setdefault(entry['source_id'], []).append(entry)
    reach = {}
    for source_id, group in by_source.items():
        found = find_restricted(frame, group, admin_id_column)
        if not found['counts']:
            continue
        rows = found['rows']
        cells = sum(int((mask & ~rows).sum()) for mask in found['cells'].values())
        reach[source_id] = {'rows': int(rows.sum()), 'cells': cells}
    return reach


def _withheld_report(withheld, entries) -> list[dict]:
    """The notice's withheld list: each source's counts and its terms."""
    by_source = {entry['source_id']: entry for entry in entries}
    return [
        {
            'source_id': source_id,
            'license': by_source[source_id]['license'],
            'terms_url': by_source[source_id].get('terms_url'),
            'share_alike_family': by_source[source_id].get('share_alike_family'),
            **count,
        }
        for source_id, count in sorted(withheld.items())
    ]


def _check_columns(available, entries) -> list[str]:
    """Columns the dry check reads: what the redaction rules test.

    Every `_source` sidecar with the value beside it (redaction empties
    a cell, and rewrites its sidecar, only where the value is), and every
    column named after an attribute one of *entries* maps, which rules 3
    and 4 test inside the source's scope. Never a personal column, as
    the export never reads one.
    """
    sidecars = [
        c for c in available if c.endswith(SOURCE_SUFFIX) and not _is_personal_column(c)
    ]
    wanted = {c[: -len(SOURCE_SUFFIX)] for c in sidecars}
    for entry in entries:
        for attribute in entry.get('attributes') or ():
            for column in available:
                if _names_attribute(column, attribute, entry.get('entity_type')):
                    wanted.add(column)
    values = [
        c
        for c in available
        if c in wanted and c != 'geometry' and not _is_personal_column(c)
    ]
    return list(dict.fromkeys([*values, *sidecars]))


def check_share_alike(
    recipe, region=None, admin_id=None, admin_ids=None, allow_missing=False
) -> dict:
    """Report a bundle's share-alike families without writing anything.

    Reads the curated files `export_delivery` would pool (their `_source`
    sidecars and the columns the redaction rules test), withholds what
    restricted sources supplied and, where the region declares it, what
    one share-alike family supplied, as the export does, and asks
    `bundle_terms` which share-alike licences the remaining geometry and
    cells carry. Call it on a recipe and region before exporting: the
    export raises `ShareAlikeConflictError` exactly when this reports
    ``conflict``.

    The counts are taken before entities curated by two neighboring
    units are resolved to one copy, so a share can differ slightly from
    the exported bundle's; the families do not.

    Parameters
    ----------
    recipe : str or dict
        Curation recipe ID or loaded recipe.
    region : str, optional
        Which declared region to check; required when the recipe
        declares more than one.
    admin_id : str or `AdminId`, optional
        Admin unit the bundle covers. Defaults to `delivery_admin_id`.
    admin_ids : list of str, optional
        Process-level units to pool, overriding the declared members.
    allow_missing : bool, optional
        Check the members that have a curated file and list the others
        under ``missing``, rather than raising as the export does.

    Returns
    -------
    dict
        ``region``, ``admin_id``, ``audience``, ``units`` (members read),
        ``missing`` (members without a curated file),
        ``withhold_share_alike`` (the families the region withholds),
        ``withheld`` (each withheld source's rows and cells, as the
        notice lists them), ``families`` (family to its sources, each
        with ``share``, ``rows``, ``cells`` and ``cell_share``),
        ``mixed_families`` (True when more than one family is present),
        ``conflict`` (True when the export would refuse: mixed families
        in a bundle that is not a team bundle), ``message`` (the
        export's refusal text, or None) and ``terms`` (the full
        `bundle_terms`).
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    if admin_id is None:
        admin_id = delivery_admin_id(recipe, region=region)
    missing = []
    if allow_missing:
        inputs = []
        process_level = get_process_admin_level(recipe)
        for process_id in delivery_members(recipe, admin_id, admin_ids, region):
            path = get_output_path(recipe, process_id)
            if path.exists():
                inputs.append((process_id, path))
            else:
                missing.append(process_id)
    else:
        process_level, inputs = _resolve_inputs(recipe, admin_id, admin_ids, region)
    process_ids = [process_id for process_id, _ in inputs]
    sources = _bundle_sources(recipe, region, process_ids)
    tested = [*sources['withhold'], *sources['share_alike']]
    # As the export redacts: the same source ids name sources, and the
    # same declared derivations follow their inputs.
    known = known_source_ids()
    derived_from = share_derived_from(recipe)

    unit_column = f'admin{process_level}_id'
    geometry_parts = []
    cells: dict[str, int] = {}
    reach: dict[str, dict] = {}
    withheld: dict[str, dict] = {}
    for process_id, path in inputs:
        wanted = _check_columns(parquet_columns(path), tested)
        if not wanted:
            continue
        part = read_parquet(path, geom=False, columns=wanted)
        part[unit_column] = process_id
        part, counts = withhold(
            part,
            sources['withhold'],
            unit_column,
            known_sources=known,
            derived_from=derived_from,
        )
        # Units hold different rows, so their counts add up.
        for into, tally in (
            (withheld, counts),
            (reach, _share_alike_reach(part, sources['share_alike'], unit_column)),
        ):
            for source_id, count in tally.items():
                total = into.setdefault(source_id, {'rows': 0, 'cells': 0})
                total['rows'] += count['rows']
                total['cells'] += count['cells']
        if 'geometry_source' in part.columns:
            geometry_parts.append(part['geometry_source'].astype('string'))
        for token, count in value_source_cells(part).items():
            cells[token] = cells.get(token, 0) + count
        del part

    geometry_source = pd.concat(geometry_parts) if geometry_parts else None
    terms = bundle_terms(
        recipe,
        geometry_source,
        process_ids,
        value_cells=cells,
        share_alike_reach=reach,
    )
    message = None
    try:
        _refuse_share_alike_conflict(
            terms, recipe, admin_id, region, sources['audience']
        )
    except ShareAlikeConflictError as error:
        message = str(error)
    return {
        'region': region,
        'admin_id': str(admin_id),
        'audience': sources['audience'],
        'units': process_ids,
        'missing': missing,
        'withhold_share_alike': delivery_spec(recipe, region).get(
            'withhold_share_alike', []
        ),
        'withheld': _withheld_report(withheld, sources['withhold']),
        'families': terms['share_alike_families'],
        'mixed_families': terms['share_alike_conflict'],
        'conflict': message is not None,
        'message': message,
        'terms': terms,
    }


def _share_spec(recipe):
    """Return the recipe's `share` block, validated as a deliverable set.

    Every problem here is one the export would otherwise hit halfway
    through, after the canonical file had already been written: a
    duplicated name fails deep inside pyarrow, a missing coordinate
    column raises a bare KeyError from the point pass, and a `_source`
    sidecar named explicitly is appended a second time automatically.

    Parameters
    ----------
    recipe : dict
        Loaded recipe.

    Returns
    -------
    tuple of (list of str, list of str)
        The canonical columns and the point-only columns.
    """
    columns, point_columns = share_columns(recipe)
    recipe_id = recipe.get('recipe_id', recipe)
    if not columns:
        raise ValueError(
            f'Recipe {recipe_id!r} has no `share: columns:` '
            'block, so there is no canonical column list to deliver. Add one to '
            'the recipe naming the attributes the shared file should carry.'
        )

    def _repeated(names):
        seen, repeated = set(), []
        for name in names:
            if name in seen and name not in repeated:
                repeated.append(name)
            seen.add(name)
        return repeated

    repeated = _repeated([*columns, *point_columns])
    if repeated:
        raise ValueError(
            f'Recipe {recipe_id!r} names {", ".join(repeated)} more than once '
            'across `share: columns:` and `share: point_columns:`; the '
            'delivered files would carry duplicate column names.'
        )

    # Only a name that duplicates a sidecar this module appends by
    # itself: a column may legitimately end in `_source` and be an
    # attribute of its own (`property_source` names how a unit count was
    # derived), so the test is whether the column it explains is
    # delivered, plus `geometry_source`, which is always appended.
    declared = {*columns, *point_columns}
    sidecars = [
        c
        for c in [*columns, *point_columns]
        if c.endswith(SOURCE_SUFFIX)
        and (c == f'geometry{SOURCE_SUFFIX}' or c[: -len(SOURCE_SUFFIX)] in declared)
    ]
    if sidecars:
        raise ValueError(
            f'Recipe {recipe_id!r} names the provenance sidecar(s) '
            f'{", ".join(sidecars)} explicitly. Every canonical column has '
            f'its own `{SOURCE_SUFFIX}` sidecar added automatically, so '
            'naming one here duplicates it.'
        )

    # A non-spatial bundle writes no point file, so it needs no
    # coordinates and may declare none: requiring them would force a
    # transaction recipe to invent a location for an event that has
    # none of its own.
    if not share_is_spatial(recipe):
        if point_columns:
            raise ValueError(
                f'Recipe {recipe_id!r} declares `share: geometry: false` '
                'but also names `share: point_columns:`. There is no point '
                'file for them to go in; move them to `share: columns:`.'
            )
        return columns, point_columns

    coordinates = [c for c in COORDINATE_COLUMNS if c in point_columns]
    if coordinates:
        raise ValueError(
            f'Recipe {recipe_id!r} names {", ".join(coordinates)} under '
            '`share: point_columns:`. The coordinates become the point '
            "file's geometry, so they belong in `share: columns:`."
        )
    missing = [c for c in COORDINATE_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            f'Recipe {recipe_id!r} does not declare {", ".join(missing)} in '
            '`share: columns:`, and the point file is built from those '
            'coordinates. Add them, or the export fails after the canonical '
            'file has already been written.'
        )
    return columns, point_columns


class PersonalColumnError(ValueError):
    """Raised when a recipe declares a personal column as deliverable."""


def _is_personal_column(column: str) -> bool:
    """Return whether *column* names or locates a person.

    Reads through a provenance suffix and a trailing `_source`, so that
    `owner_name_parcel` and `grantor_source` count as well as `grantor`.
    """
    from openplaces.core.attribute_registry import is_personal_attribute
    from openplaces.recipe import resolve_attribute_name

    return is_personal_attribute(
        resolve_attribute_name(column.removesuffix(SOURCE_SUFFIX))
    ) or is_personal_attribute(resolve_attribute_name(column))


def _refuse_personal_columns(columns) -> None:
    """Refuse a delivery whose declared columns include a personal one.

    A curated table may hold the parties to a deed or a property's
    owner where the installation chose to keep them
    (`config.get_keep_personal_columns`). That choice is about one
    installation's own working data. A delivery leaves it, so these
    columns are withheld from every delivered file whatever the choice
    was, and a recipe that lists one under `share: columns` is refused
    outright rather than quietly shipped without it: the bundle would
    not be what its recipe says.
    """
    personal = [column for column in columns if _is_personal_column(column)]
    if personal:
        raise PersonalColumnError(
            'A delivery never carries personal columns, and this recipe '
            f'declares {personal} under `share`. Remove them from the '
            'recipe; they can stay in the curated table on an '
            'installation that keeps them.'
        )


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


def _require_whole_unit(recipe, admin_id, members) -> None:
    """Raise unless *members* are every process-level unit of *admin_id*.

    A whole-unit region is filed at its unit's own path, the path a
    reader takes for the unit's complete bundle. Checked when the bundle
    is written rather than when its path is resolved, because the
    orchestrator resolves paths for every job it plans and the check
    reads the admin spine.
    """
    expected = {
        str(unit)
        for unit in get_admin(
            AdminId(str(admin_id)), get_process_admin_level(recipe)
        ).index
    }
    members = {str(member) for member in members}
    missing = sorted(expected - members)
    extra = sorted(members - expected)
    if missing or extra:
        raise ValueError(
            f'A whole-unit region of {recipe.get("recipe_id", recipe)!r} is '
            f'filed as the bundle of {admin_id}, but its members are not '
            f'every unit of it (missing: {", ".join(missing[:10]) or "none"}; '
            f'outside it: {", ".join(extra[:10]) or "none"}). Remove it from '
            '`whole_unit_regions`, which files it in a folder of its own, or '
            'complete its members in the region registry.'
        )


def _resolve_inputs(recipe, admin_id, admin_ids, region=None):
    """Return the process admin level and the [(admin_id, path)] files to pool.

    Raises when a declared member has no curated file. A bundle is
    published as the region it names, so quietly dropping the units that
    were not built ships a different dataset under that name, and the
    only trace used to be a unit count printed when verbose.

    Parameters
    ----------
    recipe : dict
        Loaded curation recipe.
    admin_id : str or `AdminId` or None
        Admin unit the bundle covers, used for the hierarchy fallback.
    admin_ids : list of str or None
        Explicit member list, overriding the recipe's declared one.
    region : str, optional
        Which declared region to resolve.

    Returns
    -------
    tuple of (int, list of (str, pathlib.Path))
        The process admin level, and one (admin_id, path) pair per member.
    """
    inputs = []
    missing = []
    for process_id in delivery_members(recipe, admin_id, admin_ids, region):
        path = get_output_path(recipe, process_id)
        if path.exists():
            inputs.append((process_id, path))
        else:
            missing.append(process_id)
    if missing and inputs:
        listed = ', '.join(missing[:10])
        more = f' (and {len(missing) - 10} more)' if len(missing) > 10 else ''
        raise FileNotFoundError(
            f'{len(missing)} of {len(missing) + len(inputs)} declared members '
            f'have no curated output: {listed}{more}. Curate them, or pass '
            '`admin_ids` naming exactly the units this bundle should pool; '
            'shipping the rest under the region name would publish a '
            'different dataset than the one declared.'
        )
    return get_process_admin_level(recipe), inputs


def _bundle_output_recipe(recipe, admin_level, output_dir=None) -> dict:
    """Return a copy of *recipe* saving at *admin_level*, into *output_dir*."""
    output_recipe = dict(recipe)
    output_recipe['save_to'] = {
        **recipe.get('save_to', {}),
        'admin_level': admin_level,
    }
    if output_dir is not None:
        output_recipe['save_to']['data_dir'] = output_dir
    return output_recipe


def _refuse_shared_flat_path(recipe, spec, admin_id, canonical) -> None:
    """Raise when a bundle filed as its unit would share a path.

    Only a whole-unit region and an inline grouping are filed as their
    unit; every other region has a folder of its own. Two such bundles
    can still meet: two whole-unit regions of one recipe on one unit,
    or a bundle whose unit is at the recipe's save level, whose path is
    then that unit's curated output. Either would overwrite a file.
    """
    try:
        curated = get_output_path(recipe, admin_id)
    except ValueError:
        # A unit coarser than the save level has no curated file.
        curated = None
    if curated is not None and canonical == curated:
        raise ValueError(
            f'The {spec.get("region_id") or "inline"} bundle of '
            f'{recipe.get("recipe_id", recipe)!r} would be written over the '
            f'curated output of {admin_id} ({curated}). Name the region in '
            'the region registry, so that it is filed in a folder of its own.'
        )
    region_id = spec.get('region_id')
    if region_id is None:
        return
    sharing = [
        other['region_id']
        for other in delivery_regions(recipe)
        if other.get('whole_unit')
        and other.get('audience') != 'team'
        and other['region_id'] != region_id
        and str(delivery_admin_id(recipe, region=other['region_id'])) == str(admin_id)
    ]
    if sharing:
        raise ValueError(
            f'{recipe.get("recipe_id", recipe)!r} lists {region_id!r} and '
            f'{", ".join(sharing)} under `whole_unit_regions`, all on '
            f'{admin_id}; one unit has one whole-unit bundle.'
        )


def delivery_paths(
    recipe, admin_id=None, admin_level=None, output_dir=None, region=None
) -> dict:
    """Return the five bundle paths, keyed by their role.

    The single source of truth for where a delivery lands: `export_delivery`
    writes exactly these, and the orchestrator declares exactly these as the
    job's outputs, so the two cannot drift apart.

    Cannot go through `recipe.get_output_path` alone, which rejects an admin
    unit coarser than the recipe's own save level. The bundle is deliberately
    coarser (a state-level file pooling county-level ones), so the level is
    overridden on a recipe copy first, the same way `io.aggregate` does it.

    Where a bundle is filed:

    - A named region sits in a folder named after it, inside its unit's
      folder: `share/US/FL/_all/transaction/openplaces/2026/lakeland-fl/
      US-FL_transaction-openplaces-2026.parquet`. Its path therefore
      never equals another region's, its state's, or a county's curated
      output, whichever unit it rolls up to.
    - A region the recipe lists under `share: delivery:
      whole_unit_regions:` is its unit (`florida-statewide`), and is
      filed as the unit's own bundle, with no folder.
      `export_delivery` refuses one whose members are not every unit
      of it.
    - An inline grouping (`admin_ids:`) has no name, and is filed as
      its unit too.
    - A team bundle (`<region>.team`) sits in a `team/` folder beside
      its region's bundle.

    Raises `ValueError` when a bundle filed as its unit would be written
    over a curated output, or over another whole-unit region's bundle.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    admin_id : str or `AdminId`, optional
        Admin unit the bundle covers, which names its files. Defaults to
        `delivery_admin_id`.
    admin_level : int, optional
        Level `delivery_admin_id` truncates the members to when the
        region declares no unit of its own. Defaults to the recipe's
        declared `share: delivery: admin_level:`, then 2.
    output_dir : str, optional
        Directory to write into, as named in STANDARD_DIRS. Defaults to the
        recipe's declared value, then `share`.
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

    if spec.get('audience') == 'team':
        # Beside its base region's bundle, never in place of it.
        base = delivery_paths(
            recipe, admin_id, admin_level, output_dir, spec['base_region_id']
        )
        return {role: path.parent / TEAM_DIR / path.name for role, path in base.items()}

    if admin_level is None:
        admin_level = spec.get('admin_level', 2)
    if output_dir is None:
        output_dir = spec.get('output_dir') or DELIVERY_DIR
    if admin_id is None:
        admin_id = delivery_admin_id(recipe, admin_level, region)
    admin_id = AdminId(str(admin_id))

    # The file is named after the unit, at the unit's own level: a
    # region anchored on a county (`region_admin_id` US-FL-PO) used to
    # be asked for at the declared level 2, which `get_output_path`
    # refuses for a level-3 id.
    output_recipe = _bundle_output_recipe(recipe, admin_id.get_level(), output_dir)
    canonical = get_output_path(output_recipe, admin_id)

    # A named region is filed in a folder of its own, named after it,
    # inside its unit's folder (maintainer's rule, 2026-10-23: regional
    # bundles do not sit in the state folder). Only a region declared
    # under `whole_unit_regions`, which is its unit, takes the unit's
    # own path. Before, a region was nested only when a sibling region
    # of the same recipe shared its unit, so adding a second Florida
    # region moved the statewide bundle, and a region filed flat on
    # its state's path read as that state's bundle.
    region_id = spec.get('region_id')
    if region_id is not None and not spec.get('whole_unit'):
        canonical = canonical.parent / str(region_id) / canonical.name
    else:
        _refuse_shared_flat_path(recipe, spec, admin_id, canonical)

    paths = {'canonical': canonical}
    # A non-spatial bundle has no point or boundary file, and this is
    # the single source of truth the orchestrator declares its outputs
    # from, so omitting them here is what keeps the job from waiting on
    # files nothing will write.
    if share_is_spatial(recipe):
        paths['point'] = canonical.with_stem(f'{canonical.stem}_point')
        paths['geo'] = canonical.with_stem(f'{canonical.stem}_geo')
    paths['evidence'] = canonical.with_stem(f'{canonical.stem}_evidence')
    # Ships with the data because the obligations do: whoever receives
    # the bundle needs to know what its sources require of them.
    paths['terms'] = canonical.with_stem(f'{canonical.stem}_LICENSE').with_suffix(
        '.txt'
    )
    return paths


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
        Passed to :func:`delivery_paths` (`admin_id`, `admin_level`,
        `output_dir`, `region`) so the directory follows the bundle it
        describes.

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

    The decision is made on two arrays and applied with a single
    positional take. Assigning a column, sorting the whole frame and
    dropping the column again held three copies of a region-wide
    GeoDataFrame at once, in the step this module is memory-bound in
    (4.1 million Texas footprints).

    Parameters
    ----------
    frame : geopandas.GeoDataFrame
        The pooled region, indexed by entity id.
    coverage_columns : list of str
        Columns whose non-null count decides which copy is better covered.
    admin_id_column : str
        Column holding the contributing unit, the tie-breaker.

    Returns
    -------
    tuple of (geopandas.GeoDataFrame, int)
        The deduplicated frame and the number of rows dropped.
    """
    if not frame.index.has_duplicates:
        return frame, 0

    coverage = frame[coverage_columns].notna().sum(axis=1).to_numpy()
    admin_rank = pd.factorize(frame[admin_id_column], sort=True)[0]
    # Primary key last: coverage descending, then admin id ascending,
    # then original position (lexsort is stable).
    order = np.lexsort((admin_rank, -coverage))
    first = ~pd.Index(frame.index.to_numpy()[order]).duplicated(keep='first')
    kept = frame.iloc[np.sort(order[first])]
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

    Each file is written beside its destination and moved into place only
    once all five exist, so a run that fails halfway leaves the previously
    shipped bundle intact. Without that, the canonical, point and geo
    files came from the new curation while the evidence file and licence
    notice were left over from the last ship, and the five no longer
    shared one index.

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
        Defaults to the recipe's declared value, then ``share``.
    region : str, optional
        Which declared region to ship, by ``region_id``. Required when the
        recipe declares more than one; see :func:`delivery_regions`.
    verbose : bool, optional
        Print a line per written file.

    Returns
    -------
    dict of str to pathlib.Path
        The written paths, keyed `'canonical'`, `'point'`, `'geo'`,
        `'evidence'` and `'terms'` (the licence notice).

    Raises
    ------
    ShareAlikeConflictError
        When the geometry and cells left after withholding carry
        share-alike licences of more than one family (ODbL and CC BY-SA,
        say), naming each source, its licence and its shares. Nothing is
        written. `check_share_alike` reports the same without exporting.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    canonical_columns, point_columns = _share_spec(recipe)
    spatial = share_is_spatial(recipe)
    paths = delivery_paths(recipe, admin_id, admin_level, output_dir, region)
    if admin_id is None:
        admin_id = delivery_admin_id(recipe, admin_level, region)

    if delivery_spec(recipe, region).get('whole_unit'):
        _require_whole_unit(
            recipe, admin_id, delivery_members(recipe, admin_id, admin_ids, region)
        )
    process_level, inputs = _resolve_inputs(recipe, admin_id, admin_ids, region)
    if not inputs:
        raise FileNotFoundError(
            f'No curated output found under {admin_id} for '
            f'{recipe.get("recipe_id", recipe)!r}; nothing to deliver.'
        )
    sources = _bundle_sources(recipe, region, [process_id for process_id, _ in inputs])
    restricted = sources['restricted']
    admin_id_column = f'admin{process_level}_id'
    # What tells a token naming a source from a method's name, and in
    # which units each restricted source's values are present: the two
    # things the rule on values derived from a source reads
    # (io.delivery.redaction, rule 5).
    known = known_source_ids()
    supplied = (
        _units_supplied(inputs, restricted, admin_id_column) if restricted else {}
    )
    derived_from = share_derived_from(recipe)
    redaction = {
        'known_sources': known,
        'supplied': supplied,
        'derived_from': derived_from,
    }
    # Unlocked up front because the finished files are moved onto these
    # names, which Windows refuses for a read-only destination.
    for path in paths.values():
        _unlock(path)
    staged = _staged(paths)
    # Leftovers from a failed run are cleared here rather than in a
    # finally clause, so a crash leaves its partial output to inspect.
    for path in staged.values():
        path.unlink(missing_ok=True)

    # Two read passes rather than one, so the wide evidence columns
    # are never in memory at the same time as the polygons: pooled
    # over a region, either one alone runs to several GB.
    declared = [*canonical_columns, *point_columns]
    _refuse_personal_columns(declared)
    frames = []
    for process_id, path in inputs:
        available = parquet_columns(path)
        wanted = [column for column in declared if column in available]
        wanted += _source_columns(canonical_columns, available)
        # Pieces the footprint geospine already cut carry their
        # envelope's id; the slicing below leaves them alone.
        if ENVELOPE_ID_COLUMN in available and ENVELOPE_ID_COLUMN not in wanted:
            wanted.append(ENVELOPE_ID_COLUMN)
        # A split (non-combined) curated output keeps its geometry in a
        # `_geo` sidecar joined on `_join_id`, or failing that `geo_id`.
        # Requesting only the declared columns dropped that key, and the
        # read then failed with 'Could not identify column to join
        # GeoParquet'; nothing here checks `save_to: combined`.
        # A non-spatial entity has no sidecar to join, and asking for
        # one makes the read look for geometry that was never written.
        if spatial and 'geometry' not in available:
            for join_key in ('_join_id', 'geo_id'):
                if join_key in available:
                    if join_key not in wanted:
                        wanted.append(join_key)
                    break
        part = read_parquet(path, geom=spatial, columns=wanted)
        # A declared column a single unit happens to lack is
        # filled, not dropped, so every unit has the same schema.
        for column in declared:
            if column not in part.columns:
                part[column] = pd.NA
        part[admin_id_column] = process_id
        frames.append(part)

    if spatial:
        pooled = gpd.GeoDataFrame(pd.concat(frames), crs=frames[0].crs)
    else:
        pooled = pd.concat(frames)
    frames.clear()

    # The de-duplication below trusts the index to identify an entity:
    # a footprint on a county line really is curated by both neighbours
    # and the two copies share an Open Location Code. An unnamed
    # integer index is never an entity id, and trusting one is silently
    # destructive rather than loud - the first Wisconsin transaction
    # bundle wrote 291,024 of 2,751,753 sales and reported success,
    # because each of the 72 counties numbered its rows 0, 1, 2 and the
    # pooled frame therefore collided with itself completely.
    if pooled.index.name is None or isinstance(pooled.index, pd.RangeIndex):
        raise ValueError(
            f'{recipe.get("recipe_id", recipe)!r} pooled '
            f'{len(pooled):,} rows under an index named '
            f'{pooled.index.name!r} of type '
            f'{type(pooled.index).__name__}, which does not identify an '
            'entity. A delivery de-duplicates on the index, so this '
            'would drop every row whose position repeats in another '
            'unit. Give the curated output an entity id first '
            '(io.harmonizer.entity_ids.mint_ids).'
        )

    # An outline two units both delivered is cut at the line between
    # them, each piece keeping its unit's values (io/delivery/slicing),
    # before the copies sharing one id could be resolved to one below.
    slice_provenance = None
    delivery_block = (recipe.get('share') or {}).get('delivery') or {}
    if spatial and delivery_block.get('slice_at_unit_lines'):
        units = get_admin(
            sorted(pooled[admin_id_column].dropna().unique()),
            level=process_level,
            geom=True,
        )
        pooled, slice_provenance = slice_at_unit_lines(pooled, admin_id_column, units)
        if SLICED_FROM not in pooled.columns:
            pooled[SLICED_FROM] = pd.NA
        canonical_columns = [*canonical_columns, SLICED_FROM]
        if verbose:
            print(
                f'Cut {len(slice_provenance):,} pieces from outlines two '
                f'{admin_id_column} units delivered.'
            )

    coverage_columns = [c for c in canonical_columns if c != admin_id_column]
    pooled, n_dropped = _deduplicate(pooled, coverage_columns, admin_id_column)
    # Sort by entity id so all four files share one row order, not
    # just one set of ids: deduplication reorders by coverage, and the
    # evidence pass rebuilds rows county by county, so neither lands
    # in file order on its own.
    pooled = pooled.sort_index()
    # A restricted source's values are withheld, not the delivery: one
    # county's terms must not hold every other county's rows back. The
    # check after it is the backstop, and fires only on values present.
    # A share-alike family the region declares withheld goes the same
    # way; what is left of every share-alike source is then measured for
    # the family check below.
    pooled, withheld_first = withhold(
        pooled, sources['withhold'], admin_id_column, **redaction
    )
    _refuse_restricted_cells(pooled, restricted, admin_id_column, **redaction)
    reach_first = _share_alike_reach(pooled, sources['share_alike'], admin_id_column)
    if n_dropped and verbose:
        print(
            f'Resolved {n_dropped} entity id(s) curated by two adjacent '
            f'{admin_id_column} units; kept the better-covered copy.'
        )

    source_columns = _source_columns(canonical_columns, pooled.columns)
    # Sources trail the values they explain, so the table opens on
    # the attributes and the provenance block stays out of the way.
    canonical = pd.DataFrame(pooled[[*canonical_columns, *source_columns]])
    # The one writer allowed into the share bucket, and only here,
    # after the withholding above (io.tables.redacted_share_writes).
    with redacted_share_writes():
        to_parquet(canonical, staged['canonical'])

    point = None
    if spatial:
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
        with redacted_share_writes():
            to_parquet(point, staged['point'], schema_version='1.1.0')
            to_parquet(pooled[['geometry']], staged['geo'], schema_version='1.1.0')

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
    # The evidence file repeats these sidecars; it must carry the
    # redacted tokens ('withheld'), not the ones pass one replaced.
    redacted_sources = pd.DataFrame(pooled[source_columns]).copy()
    del pooled, canonical, point

    delivered = {*declared, *source_columns, *INTERNAL_COLUMNS, 'geometry'}
    evidence_frames = []
    for process_id, path in inputs:
        available = parquet_columns(path)
        # The evidence file takes every remaining column, which is where
        # an owner or a party to a deed would ride out: a curated table
        # holds them on an installation that chose to keep them. Never
        # read, so never written.
        wanted = [
            column
            for column in available
            if column not in delivered and not _is_personal_column(column)
        ]
        # Sources ride along in both files: dictionary-encoded, they
        # cost almost nothing, and either file reads on its own.
        wanted += _source_columns(canonical_columns, available)
        part = read_parquet(path, geom=False, columns=wanted)
        rows = part[kept.reindex(part.index).eq(process_id)]
        if slice_provenance is not None and len(slice_provenance):
            # A piece's evidence is its unit's copy's row, under the
            # piece's id.
            mine = slice_provenance[slice_provenance[admin_id_column] == process_id]
            if len(mine):
                extra = part.loc[mine['source_id'].to_numpy()].copy()
                extra.index = pd.Index(
                    mine['piece_id'].to_numpy(), name=part.index.name
                )
                # A piece can get back the id its source row already
                # had (its centroid fell in the same cell), and that id
                # then also selected the source row above: keep it once.
                rows = pd.concat([rows.drop(index=extra.index, errors='ignore'), extra])
        evidence_frames.append(rows)

    evidence = pd.concat(evidence_frames).sort_index()
    evidence_frames.clear()
    for column in redacted_sources.columns:
        if column in evidence.columns:
            evidence[column] = redacted_sources[column].reindex(evidence.index)
    # Scope is tested on each row's unit, which this pass does not read;
    # it is borrowed from pass one under a name no curated column uses.
    unit_column = '_delivery_unit'
    evidence[unit_column] = kept.reindex(evidence.index)
    evidence, withheld_second = withhold(
        evidence, sources['withhold'], unit_column, **redaction
    )
    _refuse_restricted_cells(evidence, restricted, unit_column, **redaction)
    reach_second = _share_alike_reach(evidence, sources['share_alike'], unit_column)
    evidence = evidence.drop(columns=unit_column)
    # Counties disagree about a column's type more often than a single
    # county's file does, and pooling is where that surfaces: Florida's
    # `tax_neighborhood_code` is a string in some counties and a number
    # in others, and the concatenated object column stopped the whole
    # export with "Could not convert '500' with type str: tried to
    # convert to double" after the canonical file had already been
    # written. The canonical columns are registry-typed and so agree;
    # the evidence file takes whatever a source carried.
    evidence = coerce_mixed_object_columns(evidence)
    with redacted_share_writes():
        to_parquet(evidence, staged['evidence'])

    # Counted on the evidence after withholding: it carries every
    # shipped '_source' column, the canonical table's included, so a
    # source is named for exactly the cells that leave the build.
    terms = bundle_terms(
        recipe,
        geometry_source,
        [process_id for process_id, _ in inputs],
        value_cells=value_source_cells(evidence),
        share_alike_reach=merge_counts(reach_first, reach_second),
    )
    # Decided on what is about to ship, after withholding, so a
    # withheld source's licence no longer counts once its values are
    # out. The staged files are removed: a refusal is not a crash with
    # output to inspect, and the shipped bundle stays as it was.
    try:
        _refuse_share_alike_conflict(
            terms, recipe, admin_id, region, sources['audience']
        )
    except ShareAlikeConflictError:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise
    withheld = merge_counts(withheld_first, withheld_second)
    terms['withheld'] = _withheld_report(withheld, sources['withhold'])
    terms['audience'] = sources['audience']
    terms['team_shared'] = sources['team_shared']
    staged['terms'].write_text(format_notice(recipe, terms, admin_id), encoding='utf-8')

    # Every file exists; only now does the shipped bundle change.
    for role, path in paths.items():
        os.replace(staged[role], path)
    for path in paths.values():
        _lock(path)

    if verbose:
        for role in ('canonical', 'point', 'geo', 'evidence'):
            if role not in paths:
                continue
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
