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


def _source_terms(recipe) -> dict[str, dict]:
    """Map each upstream source id to the terms its recipe records.

    Walks the recipe's real dependency graph rather than guessing from
    names, so a source only appears here if it actually feeds this recipe.

    The walk is **transitive**: `get_recipe_dependencies` reports one level,
    and a curated bundle sits several above the ingest recipes that hold the
    licences -- CHEER depends on a spine, which depends on a geospine, which
    depends on the footprint sources. Stopping at one level reports every
    source as unrecorded, which reads as "nobody checked" when the truth is
    "nobody looked far enough".
    """
    terms: dict[str, dict] = {}
    seen: set[str] = set()
    frontier = [recipe]

    while frontier:
        current = frontier.pop()
        for edge in get_recipe_dependencies(current):
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
            terms.setdefault(
                source_id,
                {
                    'source_id': source_id,
                    'recipe_id': upstream_id,
                    'license': getattr(source, 'license', None),
                    'terms_url': getattr(source, 'terms_url', None),
                    'portal_url': getattr(source, 'portal_url', None),
                },
            )
    return terms


@cache
def _catalog_terms() -> dict[str, dict]:
    """Terms recorded by every recipe in the catalog, keyed by source id.

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
    index: dict[str, dict] = {}
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
        # First recipe wins: several versions of one source record the same
        # terms, and a later one should not silently replace an earlier.
        if not source_id or source_id in index:
            continue
        index[source_id] = {
            'source_id': source_id,
            'recipe_id': path.stem,
            'license': source.get('license'),
            'terms_url': source.get('terms_url'),
            'portal_url': source.get('portal_url'),
        }
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
        ``sources``  -- one entry per contributing source, each with its
        recorded licence, terms URL and (when known) geometry share, sorted
        by share descending.
        ``share_alike`` -- licences requiring the bundle itself to carry
        them, with the combined share they reach.
        ``attribution`` -- sources that must be credited.
        ``unrecorded`` -- sources whose terms nobody has checked yet.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    known = _source_terms(recipe)

    shares: dict[str, float] = {}
    if geometry_source is not None and len(geometry_source):
        counts = geometry_source.value_counts(normalize=True, dropna=True)
        for value, share in counts.items():
            shares[_resolve_source_id(str(value), known)] = shares.get(
                _resolve_source_id(str(value), known), 0.0
            ) + float(share)

    sources = []
    for source_id, entry in known.items():
        if shares and source_id not in shares:
            # Feeds the recipe but contributed no geometry to this bundle
            # (an attribute-only or reference input); not a licence the
            # geometry inherits.
            continue
        sources.append({**entry, 'share': shares.get(source_id)})

    catalog = _catalog_terms()
    for source_id in shares:
        if source_id in known:
            continue
        fallback = catalog.get(source_id)
        sources.append(
            {
                **(
                    fallback
                    or {
                        'source_id': source_id,
                        'recipe_id': None,
                        'license': None,
                        'terms_url': None,
                        'portal_url': None,
                    }
                ),
                'share': shares[source_id],
            }
        )

    sources.sort(key=lambda entry: (-(entry['share'] or 0), entry['source_id']))

    share_alike: dict[str, float] = {}
    attribution, unrecorded = [], []
    for entry in sources:
        license_text = entry['license']
        if not license_text:
            unrecorded.append(entry)
            continue
        if _is(license_text, SHARE_ALIKE_LICENSES):
            key = str(license_text)
            share_alike[key] = share_alike.get(key, 0.0) + (entry['share'] or 0.0)
        if _is(license_text, ATTRIBUTION_LICENSES):
            attribution.append(entry)

    return {
        'sources': sources,
        'share_alike': share_alike,
        'attribution': attribution,
        'unrecorded': unrecorded,
    }


def _resolve_source_id(value: str, known: dict) -> str:
    """Map a `geometry_source` value onto a known upstream source id.

    Values are dotted where geometry was derived rather than taken whole
    (`parcel.somecounty`, `condo_cluster.parcel.somecounty`). Tries each
    dotted token against the sources that actually feed this recipe, most
    specific first, and falls back to the value itself so an unmatched
    source is still reported rather than silently dropped.
    """
    catalog = _catalog_terms()
    if value in known or value in catalog:
        return value
    for token in reversed(value.split('.')):
        if token in known or token in catalog:
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

    lines += ['Sources', '-' * 70]
    for entry in terms['sources']:
        share = entry['share']
        share_text = f'{share:6.1%}  ' if share is not None else ' ' * 8
        license_text = entry['license'] or _UNRECORDED
        lines.append(f'{share_text}{entry["source_id"]:<20} {license_text}')
        url = entry['terms_url'] or entry['portal_url']
        if url:
            lines.append(f'{" " * 8}{"":<20} {url}')
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
        lines += [
            '',
            'Sharing it privately with named collaborators is a smaller act than',
            'publishing, and is generally satisfied by keeping this notice with the',
            'data. Publishing it is a decision for you as the distributor, not for',
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
