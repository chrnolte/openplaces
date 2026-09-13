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

from functools import cache

from openplaces.recipe import (
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    source_id_from_recipe_id,
)

__all__ = ['bundle_terms', 'format_notice', 'SHARE_ALIKE_LICENSES']

# Licence identifiers whose terms follow a derivative database and require
# the derivative itself to carry the same licence. Matched against the
# recorded `license` string case-insensitively by prefix, so 'ODbL-1.0' and
# 'ODbL-1.0 (with local amendment)' both count.
SHARE_ALIKE_LICENSES = ('odbl', 'cc-by-sa', 'gpl', 'ogl-sa')

# Licences that ask only to be credited. Prefix-matched as above.
ATTRIBUTION_LICENSES = ('cc-by', 'odbl', 'mixed-permissive', 'ogl')

_UNRECORDED = 'not recorded'


def _is(license_text: str | None, prefixes: tuple[str, ...]) -> bool:
    """True when a recorded licence starts with one of *prefixes*."""
    if not license_text:
        return False
    text = str(license_text).strip().lower()
    return any(text.startswith(prefix) for prefix in prefixes)


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


def _walk_source_terms(recipe, admin_id, terms: dict, key=_terms_key) -> None:
    """Add the terms of every recipe upstream of *recipe* for one unit."""
    seen: set[str] = set()
    frontier = [recipe]

    while frontier:
        current = frontier.pop()
        for edge in get_recipe_dependencies(current, admin_id=admin_id):
            upstream_id = getattr(edge, 'upstream_recipe_id', None)
            if not upstream_id or upstream_id in seen:
                # Unresolved auto-discovery cannot be attributed; a repeat
                # is a diamond in the graph, not new information.
                continue
            seen.add(upstream_id)
            try:
                upstream = get_recipe_by_id(upstream_id)
            except Exception:
                continue
            frontier.append(upstream)

            entity = upstream.get('entity') or upstream.get('dataset') or {}
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
                'attributes': _mapped_attributes(upstream),
                'team_sharing_permitted': getattr(
                    source, 'team_sharing_permitted', None
                ),
            }
            terms.setdefault(key(entry), entry)


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
        `entity_type`, `admin_id` (the recipe's scope) and `attributes`
        (what it maps). Empty when no input is restricted.
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
                    'team_sharing_permitted': entry.get('team_sharing_permitted'),
                }
            )
    return sorted(blocked, key=lambda e: (e['source_id'], e['recipe_id'] or ''))


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
        }
        # First recipe wins per distinct set of terms: several
        # versions of one source record the same terms, and a later
        # one should not replace an earlier. Different terms are
        # kept side by side.
        key = _terms_key(entry)
        if key in seen:
            continue
        seen.add(key)
        index.setdefault(source_id, []).append(entry)
    return index


def bundle_terms(recipe, geometry_source=None) -> dict:
    """Describe the licence obligations a bundle inherits.

    Parameters
    ----------
    recipe : str or dict
        The curation recipe the bundle was exported from.
    geometry_source : pandas.Series, optional
        The bundle's own `geometry_source` column. When given, each
        contributing source is reported with its share of the geometry, so
        the notice states how much of the bundle each licence reaches.

    Returns
    -------
    dict
        ``sources``  -- one entry per set of terms a contributing source
        records, each with its recorded licence, terms URL and (when
        known) geometry share, sorted by share descending.
        ``share_alike`` -- licences requiring the bundle itself to carry
        them, with the combined share they reach.
        ``attribution`` -- sources that must be credited.
        ``restricted`` -- sources whose recipe records
        `redistribution_restricted`.
        ``resale_restricted`` -- sources whose recipe records
        `resale_restricted`: selling is forbidden, free sharing is not.
        ``unrecorded`` -- sources whose terms nobody has checked yet.
        ``unknown_share`` -- the share of the bundle's geometry whose
        source the bundle itself does not record.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    known = _source_terms(recipe)
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

    share_alike: dict[str, float] = {}
    attribution, restricted, unrecorded = [], [], []
    resale_restricted = []
    counted: set[tuple[str, str]] = set()
    for entry in sources:
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

    return {
        'sources': sources,
        'share_alike': share_alike,
        'attribution': attribution,
        'restricted': restricted,
        'resale_restricted': resale_restricted,
        'unrecorded': unrecorded,
        'unknown_share': unknown_share,
    }


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

    # Which recipe roots the build read from. Auto-discovery takes the
    # most specific recipe those roots hold, so the roots and their
    # versions are part of what produced this bundle.
    from openplaces.path import recipe_root_records

    lines += ['Recipes', '-' * 70]
    for record in recipe_root_records():
        version = record.get('version') or 'unversioned'
        if record.get('commit'):
            version = f'{version} at commit {record["commit"]}'
        distribution = record.get('distribution') or record['provider']
        lines.append(f'{distribution} {version}')
        lines.append(f'{" " * 8}{record["root"]}')
    lines.append('')
    if terms.get('withheld'):
        lines += [
            'Withheld',
            '-' * 70,
            'These sources restrict passing their data on, so the values they',
            'supplied are not in this bundle. Rows whose outline came from them',
            "are left out; values they filled are empty, with their '_source'",
            "column set to 'withheld'.",
        ]
        for entry in terms['withheld']:
            lines.append(
                f'  {entry["source_id"]}: {entry["rows"]:,} rows left out, '
                f'{entry["cells"]:,} values emptied'
            )
            license_text = entry.get('license') or _UNRECORDED
            lines.append(f'    {license_text}  {entry.get("terms_url") or ""}')
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

    if terms['share_alike']:
        lines += ['Share-alike', '-' * 70]
        for license_text, share in sorted(
            terms['share_alike'].items(), key=lambda item: -item[1]
        ):
            lines.append(
                f'{share:.1%} of the geometry in this bundle is licensed '
                f'{license_text},'
            )
            lines.append(
                'which is a share-alike licence: publishing this bundle, or anything'
            )
            lines.append(
                f'derived from it that is still a database, requires '
                f'releasing it under {license_text} as well.'
            )
        if len(terms['share_alike']) > 1:
            names = ', '.join(sorted(terms['share_alike']))
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
            share_text = f'{share:.1%} of geometry' if share is not None else ''
            lines.append(f'  {entry["source_id"]}  {share_text}')
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'
