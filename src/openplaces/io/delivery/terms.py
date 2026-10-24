"""
What a delivery bundle inherits from the sources that went into it.

A bundle is the one thing `openplaces` produces that leaves the machine
that made it. The engine does not decide whether to share it -- that is the
user's call -- but it does have to make the call an informed one, which
means saying what the bundle carries before the user hands it on.

Everything here is *derived*: which sources contributed geometry comes from
the bundle's own `geometry_source` column, and what each source permits
comes from that source recipe's `license` field. Nothing is typed by hand,
so a notice cannot drift from the data it describes. A source whose terms
nobody has recorded is reported as unrecorded rather than skipped -- an
absent licence is a gap in the notice, not a clean bill.

The counterpart of `io.consent`, which governs what openplaces agrees to on
the way in. This governs what it tells you on the way out.
"""

from __future__ import annotations

import re
from functools import cache

from openplaces.core.schema import admin_scope_covers
from openplaces.recipe import (
    apply_recipe_patches,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    source_id_from_recipe_id,
)

__all__ = [
    'bundle_terms',
    'format_notice',
    'share_alike_family',
    'share_alike_family_named',
    'share_alike_conflict_text',
    'share_alike_inputs',
    'SHARE_ALIKE_FAMILIES',
    'SHARE_ALIKE_LICENSES',
]

# Licence identifiers whose terms follow a derivative database and require
# the derivative itself to carry the same licence. Matched against the
# recorded `license` string case-insensitively by prefix, so 'ODbL-1.0' and
# 'ODbL-1.0 (with local amendment)' both count. Spaces count as hyphens,
# so 'CC BY-SA 4.0' is 'cc-by-sa-4.0'.
SHARE_ALIKE_LICENSES = ('odbl', 'cc-by-sa', 'gpl', 'ogl-sa')

# Licences that ask only to be credited. Prefix-matched as above.
ATTRIBUTION_LICENSES = ('cc-by', 'odbl', 'mixed-permissive', 'ogl')

_UNRECORDED = 'not recorded'


def _normalized(license_text) -> str:
    """A recorded licence, lower case, spaces and underscores as hyphens."""
    return re.sub(r'[\s_]+', '-', str(license_text).strip().lower())


def _is(license_text: str | None, prefixes: tuple[str, ...]) -> bool:
    """True when a recorded licence starts with one of *prefixes*."""
    if not license_text:
        return False
    text = _normalized(license_text)
    return any(text.startswith(prefix) for prefix in prefixes)


# Share-alike licences fall into families whose members can be released
# together under one of them. Every pair from two families conflicts,
# because each licence requires the whole derivative to carry it (or a
# licence it names as compatible) and neither names the other. Only
# compatibilities a licence states itself are used here:
#
# - ODbL 1.0, section 4.4: a Derivative Database must be under "this
#   License", "a later version of this License similar in spirit", or
#   "a compatible license" the licensor names. No later version exists
#   and the Open Data Commons names no compatible licence, so ODbL is a
#   family of its own (https://opendatacommons.org/licenses/odbl/1-0/).
# - CC BY-SA 4.0, section 3(b)(1), allows an adaptation under a "BY-SA
#   Compatible License", which includes later versions of BY-SA; 3.0,
#   2.5 and 2.0 (section 4(b)) allow "a later version of this License
#   with the same License Elements". 2.0 to 4.0 can therefore all be
#   released under 4.0. 1.0 has no such clause and stands alone
#   (https://creativecommons.org/compatiblelicenses/). Creative Commons
#   also declares BY-SA 4.0 one-way compatible with GPLv3; that is a
#   software licence, and the maintainer's rule (2026-10-23) is one
#   share-alike family per bundle, so it is not applied.
# - GPL: the FSF states that GPLv2-only and GPLv3 are incompatible,
#   while "v2 or later" may be used under v3
#   (https://www.gnu.org/licenses/gpl-faq.html#v2v3Compatibility).
#   Version 2 only is therefore its own family. A "v2 or later" source
#   beside a "v2 only" one is reported as a conflict although both
#   could be released under v2: a false alarm, never a false clearance.
# - OGL-SA: no compatibility recorded, a family of its own.
def share_alike_family(license_text: str | None) -> str | None:
    """The share-alike family a recorded licence belongs to, or None.

    Licences of one family can be combined in a single release; any two
    families cannot (the comment above this function gives the source
    of each rule).

    Parameters
    ----------
    license_text : str or None
        A licence as a source recipe records it, such as 'ODbL-1.0' or
        'CC BY-SA 4.0 (open data portal)'.

    Returns
    -------
    str or None
        'ODbL', 'CC BY-SA', 'CC BY-SA 1.0', 'GPL', 'GPL-2.0-only' or
        'OGL-SA'; None for a licence that is not share-alike.
    """
    if not _is(license_text, SHARE_ALIKE_LICENSES):
        return None
    text = _normalized(license_text)
    if text.startswith('odbl'):
        return 'ODbL'
    if text.startswith('cc-by-sa'):
        version = re.match(r'cc-by-sa-?(\d+(?:\.\d+)?)', text)
        if version and float(version.group(1)) < 2:
            return 'CC BY-SA 1.0'
        return 'CC BY-SA'
    if text.startswith('gpl'):
        version = re.match(r'gpl-?v?(\d+)', text)
        later = '+' in text or 'or-later' in text
        if version and version.group(1) == '2' and not later:
            return 'GPL-2.0-only'
        return 'GPL'
    return 'OGL-SA'


# Every name `share_alike_family` returns, in the order a notice lists
# them.
SHARE_ALIKE_FAMILIES = (
    'ODbL',
    'CC BY-SA',
    'CC BY-SA 1.0',
    'GPL',
    'GPL-2.0-only',
    'OGL-SA',
)


def share_alike_family_named(name) -> str:
    """The family a recipe names under `withhold_share_alike`.

    Matched like a recorded licence (case-insensitive, spaces as
    hyphens), so 'odbl', 'ODbL' and 'cc-by-sa' all resolve.

    Raises
    ------
    ValueError
        When *name* is not one of `SHARE_ALIKE_FAMILIES`.
    """
    wanted = _normalized(name)
    for family in SHARE_ALIKE_FAMILIES:
        if _normalized(family) == wanted:
            return family
    raise ValueError(
        f'{name!r} is not a share-alike family; expected one of '
        f'{", ".join(SHARE_ALIKE_FAMILIES)}.'
    )


def share_alike_inputs(recipe, admin_ids=None) -> list[dict]:
    """Upstream sources under a share-alike licence, each with its family.

    The share-alike counterpart of `restricted_inputs`, returning entries
    of the same shape, so `io.delivery.redaction` finds what each one
    supplied (and can withhold it) by the same four rules. Walked per
    unit like `restricted_inputs`. A source the walk does not reach but
    the catalog records for one of the units (an outline source a spine
    auto-discovered) is added with no attributes and `layer_source`
    False, so only the rows and cells whose `_source` names it count.
    A source of national or global scope keeps its layer but no
    attributes: what it filled is found by the tokens alone.

    Parameters
    ----------
    recipe : str or dict
        The recipe whose inputs are checked, usually a curation recipe.
    admin_ids : list of str, optional
        Admin units the output covers.

    Returns
    -------
    list of dict
        One entry per share-alike recipe, with `family` beside the keys
        `restricted_inputs` gives.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    found = []
    seen: set[tuple] = set()
    per_recipe = _source_terms(recipe, admin_ids, key=lambda e: e['recipe_id'])
    for entry in per_recipe.values():
        family = share_alike_family(entry.get('license'))
        if family is None:
            continue
        seen.add((entry['source_id'], str(entry['license'])))
        # Inside a county the roll is the parcel layer, so a column
        # named after what it maps is its value (redaction rule 3).
        # Nationally several sources share a layer and its names: OBM
        # maps `occupancy`, `source_date` and (its tiles) `n_buildings`,
        # and by name took FEMA's `occupancy_type_footprint_fema`, a
        # parcel's own `source_date` and `n_buildings_nsi` (measured
        # 2026-10-23 on US-MA-NTC: 37,719 parcel cells, none named obm).
        # The layer-only token rule stays: a value whose `_source`
        # says only `footprint` still counts for every footprint source.
        scope = str(entry.get('admin_id') or '')
        national = '-' not in scope
        found.append(
            {
                'source_id': entry['source_id'],
                'recipe_id': entry['recipe_id'],
                'license': entry['license'],
                'terms_url': entry.get('terms_url'),
                'family': family,
                'entity_type': entry.get('entity_type'),
                'admin_id': entry.get('admin_id'),
                'attributes': [] if national else entry.get('attributes') or [],
                'layer_source': entry.get('layer_source', True),
            }
        )
    for source_id, entries in _catalog_terms().items():
        for entry in entries:
            family = share_alike_family(entry.get('license'))
            key = (source_id, str(entry.get('license')))
            if family is None or key in seen or not _applies_to(entry, admin_ids):
                continue
            seen.add(key)
            found.append(
                {
                    'source_id': source_id,
                    'recipe_id': entry.get('recipe_id'),
                    'license': entry.get('license'),
                    'terms_url': entry.get('terms_url'),
                    'family': family,
                    'entity_type': None,
                    'admin_id': entry.get('admin_id'),
                    'attributes': [],
                    'layer_source': False,
                }
            )
    return sorted(
        found, key=lambda e: (e['family'], e['source_id'], e['recipe_id'] or '')
    )


def _share_alike_families(entries, cells_by_source, evidence, reach=None) -> dict:
    """Group contributing share-alike sources by family.

    With *evidence* (the bundle's geometry or value cells were given),
    a source counts only where it drew geometry, filled cells its
    `_source` names, or, with *reach*, left rows or cells the redaction
    rules attribute to it: a source that feeds the recipe and left
    nothing in the bundle passes no licence on, as in the rest of
    `bundle_terms`. Without evidence, every listed source counts.
    """
    reach = reach or {}
    total_cells = sum(cells_by_source.values())
    families: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    for entry in entries:
        family = share_alike_family(entry.get('license'))
        if family is None:
            continue
        key = (entry['source_id'], str(entry['license']))
        if key in seen:
            continue
        reached = reach.get(entry['source_id']) or {}
        # The redaction rules count every cell a `_source` names too,
        # so the larger of the two counts is the source's reach.
        cells = max(cells_by_source.get(entry['source_id'], 0), reached.get('cells', 0))
        rows = reached.get('rows', 0)
        share = entry.get('share')
        if evidence and not (share or cells or rows):
            continue
        seen.add(key)
        families.setdefault(family, []).append(
            {
                'source_id': entry['source_id'],
                'recipe_id': entry.get('recipe_id'),
                'license': entry['license'],
                'share': share,
                'rows': rows,
                'cells': cells,
                'cell_share': (
                    cells_by_source.get(entry['source_id'], 0) / total_cells
                    if total_cells
                    else None
                ),
            }
        )
    for members in families.values():
        members.sort(key=lambda e: (-(e['share'] or 0), -e['cells'], e['source_id']))
    return families


def share_alike_conflict_text(terms: dict) -> str:
    """Name each share-alike family of *terms* with its sources.

    One line per source: its licence, its share of the geometry and the
    cells it filled, as `bundle_terms` measured them.
    """
    lines = []
    for family, members in sorted(terms.get('share_alike_families', {}).items()):
        lines.append(f'  {family}:')
        for entry in members:
            parts = []
            if entry['share']:
                parts.append(f'{entry["share"]:.1%} of geometry')
            if entry.get('rows') and not entry['share']:
                parts.append(f'{entry["rows"]:,} rows')
            if entry['cells']:
                cell_share = entry.get('cell_share')
                share_text = (
                    f' ({cell_share:.1%} of sourced cells)' if cell_share else ''
                )
                parts.append(f'{entry["cells"]:,} cells{share_text}')
            reach = ', '.join(parts) or 'no measured share'
            lines.append(f'    {entry["source_id"]} ({entry["license"]}): {reach}')
    return '\n'.join(lines)


def _terms_key(entry: dict) -> tuple:
    """Identify one recorded set of terms.

    Keyed on what a recipe records, never on the source id alone. Two
    recipes can share a source id and record different licences: the
    Overture addresses theme is permissive while the Overture buildings
    theme is ODbL-1.0, both under the source id `overture`. Keying on the
    id let whichever recipe was reached first shadow the other, so a
    bundle built on Overture footprints reported the permissive licence
    and dropped the share-alike obligation entirely.

    Recipes that record the same terms collapse onto one entry, which is
    the common case for several versions of one source.
    """
    return (
        entry['source_id'],
        entry['license'],
        entry['terms_url'],
        entry['portal_url'],
        entry['redistribution_restricted'],
        entry['resale_restricted'],
    )


def _blank_entry(source_id: str) -> dict:
    """An entry for a source whose terms no recipe records."""
    return {
        'source_id': source_id,
        'recipe_id': None,
        'license': None,
        'terms_url': None,
        'portal_url': None,
        'redistribution_restricted': None,
        'resale_restricted': None,
        'usage_requirement': None,
    }


def _source_terms(recipe, admin_ids=None, key=_terms_key) -> dict:
    """Collect the terms every upstream recipe records, one entry each.

    Walks the recipe's real dependency graph rather than guessing from
    names, so a source only appears here if it actually feeds this recipe.

    The walk is **transitive**: `get_recipe_dependencies` reports one level,
    and a curated bundle sits several above the ingest recipes that hold the
    licences -- CHEER depends on a spine, which depends on a geospine, which
    depends on the footprint sources. Stopping at one level reports every
    source as unrecorded, which reads as "nobody checked" when the truth is
    "nobody looked far enough".

    Auto-discovered inputs (a spine's county parcel and property recipes)
    resolve only for a given admin unit, so without *admin_ids* they are
    skipped. With them, the walk runs once per unit and reaches every
    county source the pipeline would read there.

    Keyed by `_terms_key`, so two recipes sharing a source id but
    recording different licences both survive.
    """
    terms: dict[tuple, dict] = {}
    for admin_id in admin_ids or [None]:
        _walk_source_terms(recipe, admin_id, terms, key)
    return terms


def _mapped_attributes(recipe: dict) -> list[str]:
    """Attribute names a recipe writes: mapped columns and outputs."""
    names = set()
    for spec in [recipe, *(recipe.get('additional_layers') or [])]:
        if not isinstance(spec, dict):
            continue
        names.update(str(name) for name in (spec.get('columns') or {}))
        for step in spec.get('transformations') or []:
            if isinstance(step, dict) and step.get('output'):
                names.add(str(step['output']))
    return sorted(names)


def _referencing_specs(node, upstream_id: str) -> list[dict]:
    """Mappings in *node* that name *upstream_id* as their `recipe_id`."""
    found: list[dict] = []
    if isinstance(node, dict):
        if node.get('recipe_id') == upstream_id:
            found.append(node)
        for value in node.values():
            found.extend(_referencing_specs(value, upstream_id))
    elif isinstance(node, list):
        for item in node:
            found.extend(_referencing_specs(item, upstream_id))
    return found


def _explicit_outputs(recipe: dict, upstream_id: str) -> set[str] | None:
    """Column names *recipe* takes from *upstream_id*, or None for a layer.

    A step that names the upstream with an explicit `columns` mapping
    (a `link_by_id` pulling one field, a `merge_enrichments` entry) lets
    exactly those outputs into the recipe's table; `count_as` adds a
    count column. Any reference without such a list (an `entity_recipe`,
    a spine source, an auto-discovered layer) can carry every attribute
    the upstream maps, which None reports.
    """
    return _spec_outputs(_referencing_specs(recipe.get('pipeline'), upstream_id))


def _discovered_outputs(recipe: dict, edge, entity_type: str) -> set[str] | None:
    """Column names an auto-discovering step takes, or None for a layer.

    The parcel spine's `link_by_id` finds every transaction source of
    its unit by entity type and keeps only `last_sale_price`,
    `last_sale_date` and a count. Matched by `recipe_id` alone, the step
    was invisible, so a sales source reached only that way counted as a
    layer, and the redaction rules withheld every parcel column a
    footprint takes. Every discovering mapping of the edge's step and
    *entity_type* is read, a step's `sources` entries included, and one
    without a column list makes the source a layer.

    Measured 2026-10-24 on boston-inner-core: 2,754,940 LA3 cells before, 0 after.
    """
    if getattr(edge, 'kind', None) != 'auto_discover' or not entity_type:
        return None
    step_name = getattr(edge, 'step', None)
    entity = recipe.get('entity')
    default = str(getattr(entity, 'entity_type', '') or '') or None

    def discovers(node) -> bool:
        return (
            isinstance(node, dict)
            and not node.get('recipe_id')
            and bool(node.get('auto_discover') or node.get('entity_type'))
            and str(node.get('entity_type') or default) == entity_type
        )

    specs = []
    for step in recipe.get('pipeline') or []:
        if not isinstance(step, dict) or step.get('step') != step_name:
            continue
        sources = step.get('sources')
        if isinstance(sources, list):
            specs.extend(source for source in sources if discovers(source))
        elif discovers(step):
            specs.append(step)
    return _spec_outputs(specs)


def _spec_outputs(specs: list[dict]) -> set[str] | None:
    """Outputs of mappings that name their columns, or None if one does not."""
    if not specs:
        return None
    outputs: set[str] = set()
    for spec in specs:
        columns = spec.get('columns')
        if isinstance(columns, dict):
            outputs.update(str(name) for name in columns.values())
        elif isinstance(columns, list | tuple):
            outputs.update(str(name) for name in columns)
        else:
            return None
        if spec.get('count_as'):
            outputs.add(str(spec['count_as']))
    return outputs


def _as_run(recipe: dict, admin_id) -> dict:
    """*recipe* as *admin_id* runs it, its patch recipes applied.

    `get_recipe_dependencies` reads the patched pipeline, so an input a
    patch adds is reached. The columns it takes have to be read from the
    same pipeline: read from the unpatched one, the input counted as a
    layer mapping nothing, so the columns a patch merged from a
    restricted source were not withheld by name (measured 2026-10-18 on
    US-MA-NTC: the placeslab land values reached the public bundle, and
    the layer rules emptied land_use_class instead).
    """
    if admin_id is None or not recipe.get('pipeline'):
        return recipe
    try:
        patched, _ = apply_recipe_patches(recipe, admin_id)
    except Exception:  # noqa: BLE001 - as in get_recipe_dependencies
        return recipe
    return patched


def _recipe_entity_type(recipe_id: str) -> str | None:
    """Entity type of a recipe, or None where it cannot be read."""
    try:
        entity = get_recipe_by_id(recipe_id).get('entity')
    except Exception:  # noqa: BLE001 - as in _walk_source_terms
        return None
    if isinstance(entity, dict):
        return str(entity.get('entity_type') or '') or None
    return str(getattr(entity, 'entity_type', '') or '') or None


def _walk_source_terms(recipe, admin_id, terms: dict, key=_terms_key) -> None:
    """Add the terms of every recipe upstream of *recipe* for one unit.

    Each entry also records how far the source's data can reach: a
    source read as a layer (a spine or an auto-discovered county roll)
    can land any attribute it maps, and `io.redaction` withholds those by
    name inside the source's scope. A source reached only through steps
    that name their columns (a permit table linked for one evidence
    column) can land nothing else, so its entry carries just those
    columns and `layer_source` False. Before this distinction a
    national-scope permit source, mapping `year_built` and `county_fips`
    like any roll, emptied both across every county of a bundle.
    """
    # Each recipe reached, with the columns through which everything
    # upstream of it reaches the bundle, or None when it is read as a
    # layer. A recipe met again on a wider path (a layer path after a
    # column path) is walked again so its own inputs widen too; met
    # again on another column path, the columns are merged in place.
    #
    # Each path also carries the entity of the recipe the bundle's own
    # recipe reads it through (`via`): a county roll reaches the
    # footprint bundle inside the curated parcels, so a footprint cell
    # whose sidecar says only `parcel` may be the roll's, while a source
    # the recipe reads directly arrives under its own name.
    # io.delivery.redaction reads it for values derived from a source.
    reached: dict[str, tuple[set[str] | None, frozenset]] = {}
    frontier: list[tuple[dict, set[str] | None, str | None]] = [(recipe, None, None)]

    while frontier:
        current, cap, via = frontier.pop()
        as_run = _as_run(current, admin_id)
        for edge in get_recipe_dependencies(current, admin_id=admin_id):
            upstream_id = getattr(edge, 'upstream_recipe_id', None)
            if not upstream_id:
                # Unresolved auto-discovery cannot be attributed.
                continue
            explicit = _explicit_outputs(as_run, upstream_id)
            if explicit is None and getattr(edge, 'kind', None) == 'auto_discover':
                explicit = _discovered_outputs(
                    as_run, edge, _recipe_entity_type(upstream_id)
                )
            if explicit is not None:
                reach = explicit if cap is None else explicit & cap
            else:
                reach = cap
            vias = frozenset({via}) if via else frozenset()
            if upstream_id in reached:
                prior, prior_vias = reached[upstream_id]
                widened = prior is not None and reach != prior
                if not widened and vias <= prior_vias:
                    continue
                if prior is None:
                    reach = None
                elif reach is not None:
                    reach = prior | reach
                vias = vias | prior_vias
            reached[upstream_id] = (reach, vias)
            try:
                upstream = get_recipe_by_id(upstream_id)
            except Exception:
                continue

            entity = upstream.get('entity') or upstream.get('dataset') or {}
            upstream_entity = str(
                getattr(entity, 'entity_type', '')
                or (entity.get('entity_type') if isinstance(entity, dict) else '')
                or ''
            )
            # The first recipe below the bundle's names the layer for
            # everything beneath it; the bundle's recipe reads that
            # recipe itself directly.
            frontier.append((upstream, reach, via or upstream_entity or None))

            source = getattr(entity, 'source', None) or (
                entity.get('source') if isinstance(entity, dict) else None
            )
            if source is None:
                continue
            source_id = str(getattr(source, 'source_id', '') or '')
            if not source_id:
                source_id = source_id_from_recipe_id(upstream_id)
            entry = {
                'source_id': source_id,
                'recipe_id': upstream_id,
                'license': getattr(source, 'license', None),
                'terms_url': getattr(source, 'terms_url', None),
                'portal_url': getattr(source, 'portal_url', None),
                'redistribution_restricted': getattr(
                    source, 'redistribution_restricted', None
                ),
                'resale_restricted': getattr(source, 'resale_restricted', None),
                'usage_requirement': getattr(source, 'usage_requirement', None),
                # Where the source's data can land in an output: read by
                # io.redaction to withhold exactly its values.
                'entity_type': str(getattr(entity, 'entity_type', '') or '') or None,
                'admin_id': str(upstream.get('admin_id') or '') or None,
                'attributes': (
                    _mapped_attributes(upstream) if reach is None else sorted(reach)
                ),
                'layer_source': reach is None,
                # Entities of the layers it reaches the bundle through;
                # empty where the bundle's recipe reads it directly.
                'via_entities': sorted(vias),
                'team_sharing_permitted': getattr(
                    source, 'team_sharing_permitted', None
                ),
            }
            existing = terms.get(key(entry))
            if existing is None:
                terms[key(entry)] = entry
                continue
            existing['via_entities'] = sorted(
                set(existing.get('via_entities') or ()) | vias
            )
            if entry['layer_source'] or not existing['layer_source']:
                # Reached again by a wider or another column path: it
                # can land whatever either path lets through.
                existing['attributes'] = sorted(
                    set(existing['attributes']) | set(entry['attributes'])
                )
                existing['layer_source'] = (
                    existing['layer_source'] or entry['layer_source']
                )


def _usage_conditions(requirement) -> list[str]:
    """Plain-language conditions a `UsageRequirement` sets, or none."""
    if requirement is None:
        return []
    conditions = []
    if getattr(requirement, 'non_commercial', False):
        conditions.append('non-commercial use only')
    if getattr(requirement, 'environment', None):
        conditions.append('environment ' + ', '.join(requirement.environment))
    if getattr(requirement, 'admin_interest', False):
        conditions.append('jurisdictional interest')
    return conditions


def restricted_inputs(recipe, admin_ids=None) -> list[dict]:
    """Upstream sources whose terms forbid passing their data on.

    Unlike `bundle_terms`, which weighs sources by the geometry they
    contribute, this walks every source that feeds the recipe, geometry
    or attribute alike: a restricted table that only contributes a
    column (a county's room counts) would otherwise slip through. A
    source counts as restricted when its recipe records
    `redistribution_restricted: true` or a `usage_requirement` (a
    condition on who may use it). A no-resale clause alone does not:
    it forbids selling, not sharing, and stays a notice item.

    Parameters
    ----------
    recipe : str or dict
        The recipe whose inputs are checked, usually a curation recipe.
    admin_ids : list of str, optional
        Admin units the output covers. Auto-discovered inputs (county
        parcel and property recipes) resolve only per unit, so a check
        without them sees declared inputs alone.

    Returns
    -------
    list of dict
        One entry per restricted recipe, with `source_id`, `recipe_id`,
        `license`, `terms_url`, `reasons`, and where its data can land:
        `entity_type`, `admin_id` (the recipe's scope), `attributes`
        (what it maps), `layer_source` and `via_entities` (the entities
        of the layers it reaches the recipe through, empty where the
        recipe reads it directly). Empty when no input is restricted.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    blocked = []
    # Keyed per recipe, not per set of terms: two recipes of one source
    # (a roll and its buildings table) record the same terms but map
    # different attributes, and redaction needs both.
    per_recipe = _source_terms(recipe, admin_ids, key=lambda e: e['recipe_id'])
    for entry in per_recipe.values():
        reasons = []
        if entry.get('redistribution_restricted'):
            reasons.append('redistribution restricted')
        reasons += _usage_conditions(entry.get('usage_requirement'))
        if reasons:
            blocked.append(
                {
                    'source_id': entry['source_id'],
                    'recipe_id': entry['recipe_id'],
                    'license': entry['license'],
                    'terms_url': entry.get('terms_url'),
                    'reasons': reasons,
                    'entity_type': entry.get('entity_type'),
                    'admin_id': entry.get('admin_id'),
                    'attributes': entry.get('attributes') or [],
                    'layer_source': entry.get('layer_source', True),
                    'via_entities': entry.get('via_entities') or [],
                    'team_sharing_permitted': entry.get('team_sharing_permitted'),
                }
            )
    return sorted(blocked, key=lambda e: (e['source_id'], e['recipe_id'] or ''))


def known_source_ids() -> frozenset:
    """Every source id a recipe in the catalog declares.

    What tells a provenance token that names a source (`fairfaxcounty`,
    `nsi`) from one that names a method or a peer group
    (`city__is_residential`, `total_minus_improvement`, `rule`): only the
    first kind can say which source filled a cell. Redaction reads it so
    that an estimate's tier name is never mistaken for another source.
    """
    return frozenset(_catalog_terms())


@cache
def _catalog_terms() -> dict[str, list[dict]]:
    """Terms recorded by every recipe in the catalog, grouped by source id.

    Each source id maps to every *distinct* set of terms recorded for it,
    not to whichever recipe the directory walk reached first. One source
    id can carry more than one licence (`overture` covers both the
    permissive addresses theme and the ODbL buildings theme), and keeping
    only the first hid the other from the notice.

    A fallback for sources the dependency walk cannot reach. A spine
    auto-discovers its per-admin inputs (which county footprint source to
    use), and `get_recipe_dependencies` reports those as unresolved unless
    it is given an admin unit -- but a bundle pools dozens of units, so
    resolving them one at a time would mean dozens of walks. Reading the
    catalog once is cheaper and finds the same recipes.

    Parses the YAML directly rather than going through `get_recipe_dict`:
    all that is wanted is what a recipe *records* about its source, and the
    loader validates far more than that, so most recipes fail to load
    without the arguments they expect. A licence notice must not depend on
    a recipe being loadable in the abstract.

    The dependency graph stays authoritative where it resolves: this only
    answers for source ids it left unmatched.
    """
    import yaml

    from openplaces.config import cfg

    root = cfg.code_root.joinpath('src', 'openplaces', 'recipes')
    index: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    for path in sorted(root.rglob('*.yaml')):
        try:
            recipe = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(recipe, dict):
            continue
        entity = recipe.get('entity') or recipe.get('dataset') or {}
        source = entity.get('source') if isinstance(entity, dict) else None
        if not isinstance(source, dict):
            continue
        source_id = str(source.get('source_id') or '')
        if not source_id:
            continue
        entry = {
            'source_id': source_id,
            'recipe_id': path.stem,
            'license': source.get('license'),
            'terms_url': source.get('terms_url'),
            'portal_url': source.get('portal_url'),
            'redistribution_restricted': source.get('redistribution_restricted'),
            'resale_restricted': source.get('resale_restricted'),
            # The recipe's scope, so a bundle can tell a source of its
            # own units from one of another state (`_applies_to`).
            'admin_id': str(recipe.get('admin_id') or '') or None,
        }
        # First recipe wins per distinct set of terms and scope:
        # several versions of one source record the same terms, and a
        # later one should not replace an earlier. Different terms,
        # or the same terms recorded for another scope, are kept side
        # by side.
        key = (*_terms_key(entry), entry['admin_id'])
        if key in seen:
            continue
        seen.add(key)
        index.setdefault(source_id, []).append(entry)
    return index


def bundle_terms(
    recipe,
    geometry_source=None,
    admin_ids=None,
    value_cells=None,
    share_alike_reach=None,
) -> dict:
    """Describe the licence obligations a bundle inherits.

    Parameters
    ----------
    recipe : str or dict
        The curation recipe the bundle was exported from.
    geometry_source : pandas.Series, optional
        The bundle's own `geometry_source` column. When given, each
        contributing source is reported with its share of the geometry, so
        the notice states how much of the bundle each licence reaches.
    admin_ids : list of str, optional
        The units the bundle pools. The walk then runs as each unit runs
        the recipe: its patches applied and its auto-discovered county
        sources reached, as `restricted_inputs` does. Without them the
        notice sees declared inputs only, which missed a state's own
        sales feed and listed another state's roll that the national
        recipe named (the Wisconsin transaction bundle, 2026-10-19).
    value_cells : dict of str to int, optional
        Cells each source supplied, from the bundle's own `_source`
        columns (`value_source_cells`). Sources named here and not in
        the geometry are reported under ``values`` and take part in the
        attribution, restriction and unrecorded checks like the
        geometry's sources: a roll that supplies a county's values ships
        in the bundle as surely as an outline does. With *admin_ids*, a
        token counts only where the walk reached its source or a recipe
        of that source covers one of the units: a token a step wrote
        under another state's name is not that state's source.
    share_alike_reach : dict of str to dict, optional
        Rows and cells each share-alike source left in the bundle, as
        `io.delivery.redaction.find_restricted` counts them for the
        entries of `share_alike_inputs` (source id to `{'rows': n,
        'cells': n}`). A source counts toward its family where it drew
        geometry, a `_source` names it, or it reaches rows or cells
        here; a county roll whose values carry no sidecar of their own
        is found only this way.

    Returns
    -------
    dict
        ``sources``  -- one entry per set of terms a contributing source
        records, each with its recorded licence, terms URL and (when
        known) geometry share, sorted by share descending.
        ``share_alike`` -- licences requiring the bundle itself to carry
        them, with the combined share they reach.
        ``share_alike_families`` -- the share-alike sources that drew
        geometry or filled cells, grouped by `share_alike_family`, each
        with its ``share`` of the geometry, ``rows`` and ``cells`` it
        reaches, and ``cell_share`` (of all cells whose sidecar names a
        source).
        ``share_alike_conflict`` -- True when more than one family is
        present: no single release can satisfy them, and
        `export_delivery` refuses the bundle.
        ``attribution`` -- sources that must be credited.
        ``restricted`` -- sources whose recipe records
        `redistribution_restricted`.
        ``resale_restricted`` -- sources whose recipe records
        `resale_restricted`: selling is forbidden, free sharing is not.
        ``unrecorded`` -- sources whose terms nobody has checked yet.
        ``unknown_share`` -- the share of the bundle's geometry whose
        source the bundle itself does not record.
        ``values`` -- one entry per set of terms of a source that
        supplied values but no geometry, each with its ``cells`` count.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    known = _source_terms(recipe, admin_ids)
    known_ids = {entry['source_id'] for entry in known.values()}

    shares: dict[str, float] = {}
    unknown_share = 0.0
    if geometry_source is not None and len(geometry_source):
        # Shares are taken over every row, not only the attributed ones.
        # Normalizing over non-nulls made them sum to 1.0 whatever the
        # coverage, so geometry of unrecorded provenance vanished from
        # the notice instead of being reported as unaccounted for.
        total = len(geometry_source)
        recorded = geometry_source.dropna().astype(str).str.strip()
        counts = recorded[recorded != ''].value_counts()
        for value, count in counts.items():
            source_id = _resolve_source_id(str(value), known_ids)
            shares[source_id] = shares.get(source_id, 0.0) + float(count) / total
        unknown_share = max(0.0, 1.0 - sum(shares.values()))

    sources = []
    listed: dict[str, set] = {}
    for entry in known.values():
        if shares and entry['source_id'] not in shares:
            # Feeds the recipe but contributed no geometry to this bundle
            # (an attribute-only or reference input); not a licence the
            # geometry inherits.
            continue
        sources.append({**entry, 'share': shares.get(entry['source_id'])})
        listed.setdefault(entry['source_id'], set()).add(entry['license'])

    # One source id can carry more than one licence, and a
    # `geometry_source` value records only the id, never which of a
    # source's products a row came from (tracked as a follow-up on
    # the provenance column, not resolvable here). The dependency walk
    # reaches whichever recipe feeds this bundle by name, which for
    # `overture` was the permissive addresses recipe even where the
    # geometry came from the ODbL buildings recipe. Every distinct
    # licence the catalog records for a contributing id is therefore
    # reported, not only the one the walk happened to find.
    catalog = _catalog_terms()
    for source_id in shares:
        recorded = listed.get(source_id, set())
        fallbacks = catalog.get(source_id) or []
        if not fallbacks and not recorded:
            fallbacks = [_blank_entry(source_id)]
        for fallback in fallbacks:
            if fallback['license'] in recorded:
                continue
            if not fallback['license'] and recorded:
                # A recipe that recorded nothing does not unsettle one
                # that did.
                continue
            sources.append({**fallback, 'share': shares[source_id]})
            recorded.add(fallback['license'])
            listed[source_id] = recorded

    sources.sort(key=lambda entry: (-(entry['share'] or 0), entry['source_id']))

    # Sources that filled values but drew no geometry. A token is a
    # source only where a recipe declares that source id (the walk's or
    # the catalog's); the other tokens a sidecar carries name a method
    # (`group_statistic`, `rule`, `postal_code`), not a source whose
    # terms could be recorded.
    by_id: dict[str, list[dict]] = {}
    for entry in known.values():
        by_id.setdefault(entry['source_id'], []).append(entry)
    values = []
    value_counts: dict[str, int] = {}
    # The catalog answers only for recipes scoped to the bundle's units.
    # A provenance token is a step's label, and a label can name a
    # source whose data is not in the row: the transaction spine's
    # `reconcile_addresses` writes its first declared key, `nhcgov`
    # (New Hanover County NC), on every row with an `address` column,
    # so Florida's and Wisconsin's sales read `address_source = nhcgov`
    # and both bundles listed a North Carolina county's terms. The
    # scope test, not a fix to that one label, is what keeps any such
    # label out of the notice.
    in_scope = {
        source_id: [entry for entry in entries if _applies_to(entry, admin_ids)]
        for source_id, entries in catalog.items()
    }
    for token, count in (value_cells or {}).items():
        source_id = _resolve_source_id(token, known_ids)
        if source_id not in known_ids and not in_scope.get(source_id):
            continue
        value_counts[source_id] = value_counts.get(source_id, 0) + count
    for source_id, count in sorted(value_counts.items(), key=lambda kv: -kv[1]):
        if source_id in listed:
            continue
        entries = by_id.get(source_id) or in_scope.get(source_id)
        seen_licences = set()
        for entry in entries:
            if entry['license'] in seen_licences:
                continue
            seen_licences.add(entry['license'])
            values.append({**entry, 'share': None, 'cells': count})

    share_alike: dict[str, float] = {}
    attribution, restricted, unrecorded = [], [], []
    resale_restricted = []
    counted: set[tuple[str, str]] = set()
    for entry in [*sources, *values]:
        if entry['redistribution_restricted']:
            restricted.append(entry)
        if entry.get('resale_restricted'):
            resale_restricted.append(entry)
        license_text = entry['license']
        if not license_text:
            unrecorded.append(entry)
            continue
        if _is(license_text, SHARE_ALIKE_LICENSES):
            key = str(license_text)
            # One source id can record the same share-alike licence
            # under two recipes; its share is still counted once.
            if (entry['source_id'], key) in counted:
                share_alike.setdefault(key, 0.0)
            else:
                counted.add((entry['source_id'], key))
                share_alike[key] = share_alike.get(key, 0.0) + (entry['share'] or 0.0)
        if _is(license_text, ATTRIBUTION_LICENSES):
            attribution.append(entry)

    evidence = bool(shares) or bool(value_cells) or share_alike_reach is not None
    # The walk's own entries are offered too: a source that reaches the
    # bundle only through columns without a sidecar of their own is in
    # neither `sources` (geometry) nor `values` (named cells).
    families = _share_alike_families(
        [*sources, *values, *known.values()],
        value_counts,
        evidence,
        share_alike_reach,
    )

    return {
        'sources': sources,
        'share_alike': share_alike,
        'share_alike_families': families,
        'share_alike_conflict': len(families) > 1,
        'attribution': attribution,
        'restricted': restricted,
        'resale_restricted': resale_restricted,
        'unrecorded': unrecorded,
        'unknown_share': unknown_share,
        'values': values,
    }


def _applies_to(entry: dict, admin_ids) -> bool:
    """True when *entry*'s recipe scope relates to one of *admin_ids*.

    Related in either direction: a state recipe covers its counties,
    and a county recipe sits inside a state bundle. A recipe with no
    scope is national or global and applies everywhere, as does any
    entry when the units are not known.
    """
    if admin_ids is None:
        return True
    scope = entry.get('admin_id')
    if not scope:
        return True
    return any(
        admin_scope_covers(scope, unit) or admin_scope_covers(unit, scope)
        for unit in admin_ids
    )


def value_source_cells(frame) -> dict[str, int]:
    """Cells each source supplied, counted from the frame's `_source` columns.

    The provenance a bundle ships is the only record of which source
    filled which value: a roll that supplies a county's land values
    contributes no geometry, so the geometry shares alone never name it.
    Counted per specific token (`hcad` in `hcad+imputed`); tokens that
    name a layer or a method (`parcel`, `imputed`) and withheld cells are
    not sources. `geometry_source` is left to the geometry shares.
    Measured 2026-10-19 on houston-tx: hcad supplies 16,522,746 cells.
    """
    from openplaces.io.delivery.redaction import _UNSPECIFIC, WITHHELD, _tokens

    cells: dict[str, int] = {}
    for column in frame.columns:
        if not column.endswith('_source') or column == 'geometry_source':
            continue
        counts = frame[column].astype('string').value_counts()
        for value, count in counts.items():
            for token in _tokens(value):
                if token in _UNSPECIFIC or token == WITHHELD:
                    continue
                cells[token] = cells.get(token, 0) + int(count)
    return cells


def _resolve_source_id(value: str, known_ids: set[str]) -> str:
    """Map a `geometry_source` value onto a known upstream source id.

    Values are dotted where geometry was derived rather than taken whole
    (`parcel.somecounty`, `condo_cluster.parcel.somecounty`). Tries each
    dotted token against the sources that actually feed this recipe, most
    specific first, and falls back to the value itself so an unmatched
    source is still reported rather than silently dropped.
    """
    catalog = _catalog_terms()
    if value in known_ids or value in catalog:
        return value
    for token in reversed(value.split('.')):
        if token in known_ids or token in catalog:
            return token
    return value


def _withheld_lines(entries) -> list[str]:
    """Two notice lines per withheld source: its counts and its terms."""
    lines = []
    for entry in entries:
        lines.append(
            f'  {entry["source_id"]}: {entry["rows"]:,} rows left out, '
            f'{entry["cells"]:,} values emptied'
        )
        license_text = entry.get('license') or _UNRECORDED
        lines.append(f'    {license_text}  {entry.get("terms_url") or ""}')
    return lines


def format_notice(recipe, terms: dict, admin_id=None) -> str:
    """Render the notice text shipped alongside a bundle.

    Written for the person who receives the bundle, not for a machine: it
    names what is inside, what each part's licence is, and what they must
    do if they pass it on.
    """
    recipe_id = get_recipe_id(recipe)
    lines = [
        f'Data licence and attribution for {recipe_id}',
        '=' * 70,
        '',
        'This bundle was produced with openplaces (https://openplaces.io)',
        'from the sources listed below. openplaces neither hosts nor',
        'distributes data: these terms come from the upstream sources, and',
        'they travel with this bundle to whoever receives it.',
        '',
    ]
    if admin_id is not None:
        lines[0] = f'Data licence and attribution for {recipe_id} ({admin_id})'
        lines[1] = '=' * 70

    if terms.get('audience') == 'team':
        lines += [
            'TEAM-INTERNAL BUNDLE',
            '-' * 70,
            'For sharing within the research team only. It carries data from',
            'the sources below, whose terms restrict redistribution but which',
            'may be shared within the team. Do not publish it or pass it',
            'outside the team: the public bundle of the same region withholds',
            'these values.',
        ]
        for entry in terms.get('team_shared') or []:
            license_text = entry.get('license') or _UNRECORDED
            lines.append(f'  {entry["source_id"]}: {license_text}')
        lines.append('')

    lines += ['Sources', '-' * 70]
    # One source id can record more than one licence (a source shipping
    # two themes under one id). Naming the recipe on those lines is what
    # tells the reader which entry is which.
    repeated = {
        entry['source_id']
        for entry in terms['sources']
        if sum(
            1 for other in terms['sources'] if other['source_id'] == entry['source_id']
        )
        > 1
    }
    for entry in terms['sources']:
        share = entry['share']
        share_text = f'{share:6.1%}  ' if share is not None else ' ' * 8
        license_text = entry['license'] or _UNRECORDED
        label = entry['source_id']
        if label in repeated and entry['recipe_id']:
            label = f'{label} ({entry["recipe_id"]})'
        lines.append(f'{share_text}{label:<20} {license_text}')
        url = entry['terms_url'] or entry['portal_url']
        if url:
            lines.append(f'{" " * 8}{"":<20} {url}')
    if terms.get('unknown_share'):
        lines.append(
            f'{terms["unknown_share"]:6.1%}  '
            f'{"(source not recorded)":<20} {_UNRECORDED}'
        )
    lines.append('')

    if terms.get('values'):
        # Sources that filled values in this bundle but drew none of its
        # outlines: an assessment roll, read through the property spine.
        lines += [
            'Values from',
            '-' * 70,
            'These sources supplied values in this bundle (counted from its',
            "'_source' columns) but none of its geometry.",
        ]
        for entry in terms['values']:
            license_text = entry['license'] or _UNRECORDED
            cells = f'{entry["cells"]:>10,}  '
            lines.append(f'{cells}{entry["source_id"]:<20} {license_text}')
            url = entry['terms_url'] or entry['portal_url']
            if url:
                lines.append(f'{" " * 12}{"":<20} {url}')
        lines.append('')

    # Which recipe roots the build read from. Auto-discovery takes the
    # most specific recipe those roots hold, so the roots and their
    # versions are part of what produced this bundle. Named by
    # distribution, version and commit, never by folder: the folder is
    # a path on the builder's machine, user name included.
    from openplaces.path import recipe_root_records

    lines += ['Recipes', '-' * 70]
    for record in recipe_root_records():
        version = record.get('version') or 'unversioned'
        if record.get('commit'):
            version = f'{version} at commit {record["commit"]}'
        distribution = record.get('distribution') or record['provider']
        lines.append(f'{distribution} {version}')
    lines.append('')
    withheld = terms.get('withheld') or []
    restricted_withheld = [e for e in withheld if not e.get('share_alike_family')]
    family_withheld = [e for e in withheld if e.get('share_alike_family')]
    if restricted_withheld:
        lines += [
            'Withheld',
            '-' * 70,
            'These sources restrict passing their data on, so the values they',
            'supplied are not in this bundle. Rows whose outline came from them',
            "are left out; values they filled are empty, with their '_source'",
            "column set to 'withheld'.",
        ]
        lines += _withheld_lines(restricted_withheld)
        lines.append('')
    if family_withheld:
        names = ', '.join(sorted({e['share_alike_family'] for e in family_withheld}))
        lines += [
            'Withheld to keep one share-alike licence',
            '-' * 70,
            f'This bundle leaves out what sources licensed under {names}',
            'supplied, because their share-alike terms cannot be satisfied',
            "together with the share-alike licence of the bundle's other",
            'sources. Rows whose outline came from them are left out; values',
            "they filled are empty, with their '_source' column set to",
            "'withheld'.",
        ]
        lines += _withheld_lines(family_withheld)
        lines.append('')
    if terms.get('restricted'):
        lines += [
            'Redistribution restricted',
            '-' * 70,
            'The terms recorded for these sources restrict redistribution.',
            'Read each one before passing this bundle on.',
        ]
        for entry in terms['restricted']:
            url = entry['terms_url'] or entry['portal_url'] or ''
            license_text = entry['license'] or _UNRECORDED
            lines.append(f'  {entry["source_id"]}: {license_text}  {url}')
        lines.append('')

    if terms.get('resale_restricted'):
        # Its own section, not a line under "Redistribution restricted":
        # a no-resale clause leaves free sharing alone, and filing it as
        # a redistribution restriction would overstate it.
        lines += [
            'Resale restricted',
            '-' * 70,
            'The terms recorded for these sources forbid selling their data.',
            'Sharing this bundle at no charge is not what these clauses',
            'restrict; charging for it is. Read each one before you do.',
        ]
        for entry in terms['resale_restricted']:
            url = entry['terms_url'] or entry['portal_url'] or ''
            license_text = entry['license'] or _UNRECORDED
            lines.append(f'  {entry["source_id"]}: {license_text}  {url}')
        lines.append('')

    share_alike = terms['share_alike']
    present = terms.get('share_alike_families')
    if present is not None:
        # Only the licences of sources whose data is in the bundle: a
        # source that feeds the recipe, or whose values were withheld,
        # passes no share-alike obligation on.
        carried = {
            str(entry['license']) for members in present.values() for entry in members
        }
        share_alike = {k: v for k, v in share_alike.items() if k in carried}
        for license_text in sorted(carried - set(share_alike)):
            share_alike[license_text] = 0.0
    if share_alike:
        lines += ['Share-alike', '-' * 70]
        for license_text, share in sorted(
            share_alike.items(), key=lambda item: -item[1]
        ):
            if share:
                lines.append(
                    f'{share:.1%} of the geometry in this bundle is licensed '
                    f'{license_text},'
                )
            else:
                lines.append(
                    f'Values in this bundle come from a source licensed {license_text},'
                )
            lines.append(
                'which is a share-alike licence: publishing this bundle, or anything'
            )
            lines.append(
                f'derived from it that is still a database, requires '
                f'releasing it under {license_text} as well.'
            )
        # Two versions of one family (CC BY-SA 3.0 and 4.0) are one
        # release under the later version, so only families conflict.
        families = terms.get('share_alike_families')
        if families is None:
            families = {share_alike_family(text) for text in terms['share_alike']}
        if len(families) > 1:
            names = ', '.join(sorted(families))
            lines += [
                '',
                'THESE CANNOT ALL BE SATISFIED AT ONCE. A share-alike licence',
                'requires the whole release to carry that licence, so a bundle',
                f'inheriting more than one ({names}) cannot be published under',
                'any single one of them. Dual licensing does not resolve it.',
                '',
                'Options: leave out the sources carrying the minority licence,',
                'ask them for permission to relicense, or publish a Produced',
                'Work -- a figure, a map, aggregate statistics -- rather than',
                'the database itself, since both licences let a Produced Work',
                'carry its own licence, subject to attribution.',
            ]
        # The private-sharing sentence holds only where nothing above
        # restricts redistribution; a signed-agreement source is not
        # satisfied by carrying a notice, so it gets a pointer instead.
        if terms.get('restricted'):
            lines += [
                '',
                'One or more sources above restrict redistribution; see',
                'Redistribution restricted before passing this bundle on.',
            ]
        else:
            lines += [
                '',
                'Sharing it privately with named collaborators is a smaller act than',
                'publishing, and is generally satisfied by keeping this notice with',
                'the data.',
            ]
        lines += [
            'Publishing it is a decision for you as the distributor, not for',
            'openplaces, which is why this file states the terms rather than enforcing',
            'them.',
            '',
        ]

    if terms['attribution']:
        lines += ['Attribution required', '-' * 70]
        for entry in terms['attribution']:
            url = entry['terms_url'] or entry['portal_url'] or ''
            lines.append(f'  {entry["source_id"]}: {entry["license"]}  {url}')
        lines.append('')

    if terms['unrecorded']:
        lines += [
            'Terms not yet checked',
            '-' * 70,
            'Nobody has recorded terms for these sources. That is not the same as',
            'their being unrestricted -- treat this section as an open question, not',
            'as a clearance.',
        ]
        for entry in terms['unrecorded']:
            share = entry['share']
            if share is not None:
                share_text = f'{share:.1%} of geometry'
            elif entry.get('cells'):
                share_text = f'{entry["cells"]:,} values'
            else:
                share_text = ''
            lines.append(f'  {entry["source_id"]}  {share_text}')
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'
