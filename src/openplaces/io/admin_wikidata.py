"""Match spine units to Wikidata items, for CC0 names and a stable join key.

The spine is CC0, so administrative names have to come from a public-domain
source. Wikidata is that source, and it also supplies the Q-number that
replaces the GADM foreign key: stable, language-neutral, and resolvable to
every other identifier system through Wikidata's own properties.

Two findings from piloting this on Colombia decided the approach, and both
went against the obvious design:

**Do not filter by type.** Restricting children to a country's municipality
class looks more precise and is actually worse -- 89.7% matched against
91.7% unfiltered -- because a spine unit is often typed as something else in
Wikidata (a capital district, a special unit). It also costs a manual class
lookup per country, which the unfiltered query avoids entirely.

**Do not walk the subclass tree.** A generic query using
`P31/P279* wd:Q56061` times out even scoped to a single country, so
harvesting by the parent's ISO code is not merely simpler but the only
version that runs.

**Do not resolve ambiguity automatically.** Where two Wikidata items share a
name under the same parent, this module reports the conflict rather than
choosing. For Colombia that is 5.3% of units, and the duplicates are mostly
natural features (Cerro Negro, Cuchilla Pena Negra) that happen to carry a
`P131` link -- but Otanche is a real municipality with two items, so picking
one silently would sometimes be wrong.
"""

import difflib
from collections import defaultdict

import pandas as pd

from openplaces.io.admin_codes.anchors import normalize_name

WIKIDATA_SPARQL = 'https://query.wikidata.org/sparql'

# Matching outcomes, in the order a reviewer should care about them.
MATCH_UNIQUE = 'unique'
MATCH_FUZZY = 'fuzzy'
MATCH_AMBIGUOUS = 'ambiguous'
MATCH_MISSING = 'missing'

# Similarity required before a near-miss is offered at all. Deliberately
# high: a fuzzy match is a suggestion for review, never a fact, and a loose
# threshold would bury genuine gaps under plausible-looking wrong answers.
FUZZY_CUTOFF = 0.90


def children_query(parent_prefix: str, limit: int = 40000) -> str:
    """Return SPARQL for every Wikidata child of a parent's ISO-coded units.

    Deliberately unfiltered by type; see the module docstring for why.

    Parameters
    ----------
    parent_prefix : str
        ISO 3166-2 prefix of the parents whose children are wanted, e.g.
        'CO-' for Colombia's departments.
    limit : int, optional
        Result cap. Colombia returns about 25,500 rows, so the default
        leaves room; a country that hits the cap needs paginating.

    Returns
    -------
    str
        A SPARQL query returning item, itemLabel, native and parentIso.

    Examples
    --------
    >>> 'wdt:P131' in children_query('CO-')
    True
    """
    return f'''SELECT ?item ?itemLabel ?native ?parentIso WHERE {{
  ?parent wdt:P300 ?parentIso .
  FILTER(STRSTARTS(?parentIso, "{parent_prefix}"))
  ?item wdt:P131 ?parent .
  OPTIONAL {{ ?item wdt:P1705 ?native }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }}
}}
LIMIT {limit}'''


def index_harvest(harvest: pd.DataFrame) -> dict:
    """Index a harvest by (parent ISO code, normalized name).

    Parameters
    ----------
    harvest : pandas.DataFrame
        Rows with item, itemLabel and parentIso columns.

    Returns
    -------
    dict
        Mapping from (parentIso, normalized name) to the set of item URIs
        carrying that name under that parent.
    """
    index = defaultdict(set)
    for row in harvest.itertuples(index=False):
        index[(row.parentIso, normalize_name(row.itemLabel))].add(row.item)
    return dict(index)


def match_units(
    units: pd.DataFrame,
    harvest: pd.DataFrame,
    parent_iso: dict,
    fuzzy_cutoff: float | None = None,
):
    """Match spine units to Wikidata items within their own parent.

    Matching inside the parent rather than across the country is what makes
    a bare name comparison safe: two departments may each hold a Rosario,
    but one department rarely does.

    Parameters
    ----------
    units : pandas.DataFrame
        Spine rows with admin_id, name and parent_admin_id columns.
    harvest : pandas.DataFrame
        Result of the children query.
    parent_iso : dict
        Mapping from a spine parent admin_id to that parent's ISO code, as
        used in the harvest.

    Returns
    -------
    pandas.DataFrame
        One row per unit with admin_id, name, wikidata_id (empty unless the
        match is unique), n_candidates and status.

    Notes
    -----
    Status is one of 'unique', 'ambiguous' or 'missing'. Only 'unique' is
    safe to adopt without review; 'ambiguous' names the conflict so a human
    can settle it, and 'missing' is the residual another source must cover.
    """
    index = index_harvest(harvest)
    # Names available under each parent, for the near-miss pass. Restricted
    # to the unit's own parent so a close spelling elsewhere in the country
    # can never be offered.
    by_parent: dict[str, list[str]] = defaultdict(list)
    for parent_code, name in index:
        by_parent[parent_code].append(name)

    rows = []
    for unit in units.itertuples(index=False):
        parent_code = parent_iso.get(unit.parent_admin_id, '')
        key = (parent_code, normalize_name(unit.name))
        candidates = index.get(key, set())
        if len(candidates) == 1:
            status, item = MATCH_UNIQUE, next(iter(candidates))
        elif candidates:
            status, item = MATCH_AMBIGUOUS, ''
        elif fuzzy_cutoff:
            near = difflib.get_close_matches(
                normalize_name(unit.name),
                by_parent.get(parent_code, []),
                n=1,
                cutoff=fuzzy_cutoff,
            )
            found = index.get((parent_code, near[0])) if near else None
            if found and len(found) == 1:
                status, item = MATCH_FUZZY, next(iter(found))
            else:
                status, item = MATCH_MISSING, ''
        else:
            status, item = MATCH_MISSING, ''
        rows.append(
            {
                'admin_id': unit.admin_id,
                'name': unit.name,
                'wikidata_id': item.rsplit('/', 1)[-1] if item else '',
                'n_candidates': len(candidates),
                'status': status,
            }
        )
    return pd.DataFrame(rows)


def match_summary(matches: pd.DataFrame) -> pd.Series:
    """Return the share of units in each match status.

    Parameters
    ----------
    matches : pandas.DataFrame
        Output of :func:`match_units`.

    Returns
    -------
    pandas.Series
        Share per status, indexed by status name.
    """
    if matches.empty:
        return pd.Series(dtype=float)
    return matches['status'].value_counts(normalize=True)
