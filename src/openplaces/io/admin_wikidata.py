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
    fuzzy_cutoff : float, optional
        Similarity a near-miss spelling must reach before it is offered
        at all. Unset disables the near-miss pass, so no row can come
        back 'fuzzy'.

    Returns
    -------
    pandas.DataFrame
        One row per unit with admin_id, name, wikidata_id (empty unless
        exactly one item was found), n_candidates and status.
        n_candidates counts the items the status was decided on, which
        for a fuzzy row is what the near-miss spelling found rather than
        the empty exact-key set, so filtering on a positive count does
        not silently drop every fuzzy row.

    Notes
    -----
    Status is one of 'unique', 'fuzzy', 'ambiguous' or 'missing'. Only
    'unique' is safe to adopt without review; 'fuzzy' carries an id found
    under a near-miss spelling and is a suggestion for review, never a
    fact; 'ambiguous' names the conflict so a human can settle it; and
    'missing' is the residual another source must cover.
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
        n_candidates = len(candidates)
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
                # What the near-miss spelling found, not the exact key's
                # empty set: a row reporting an id alongside zero
                # candidates is invisible to anyone filtering on a
                # positive count to list the matches.
                n_candidates = len(found)
            else:
                status, item = MATCH_MISSING, ''
        else:
            status, item = MATCH_MISSING, ''
        rows.append(
            {
                'admin_id': unit.admin_id,
                'name': unit.name,
                'wikidata_id': item.rsplit('/', 1)[-1] if item else '',
                'n_candidates': n_candidates,
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


# The class whose subclass tree separates an administrative unit from a
# lake, a hill or a company that also carries a "located in" link.
ADMINISTRATIVE_ENTITY = 'Q56061'
# Two subtrees inside it that are not the territorial hierarchy: an
# electoral district is drawn for voting, and a settlement is a place
# people live, which is why a village links to its state as readily as
# the state's municipalities do.
ELECTORAL_DISTRICT = 'Q192611'
HUMAN_SETTLEMENT = 'Q486972'
# Subtrees that reach the administrative tree and hold nothing that
# governs: a fort, an airbase, a diocese, an Antarctic claim.
EXCLUDED_TREES = {
    'Q473972': 'protected area',
    'Q97095925': 'military area',
    'Q1492823': 'ecclesiastical district',
    'Q20926517': 'religious administrative territorial entity',
    'Q15239622': 'disputed territory',
    'Q398141': 'school district',
}

#: Parents per SPARQL request. Measured 2026-09-12: 47 Kenyan counties in
#: one request take 3 s; 100 Thai districts 2 s. Larger batches risk the
#: endpoint's 60 s limit on a busy day.
PARENT_BATCH_SIZE = 50


def class_tree_query(root=ADMINISTRATIVE_ENTITY):
    """Return SPARQL for every subclass of the administrative-entity class.

    One request per process (9,045 classes, under 2 s), after which
    membership is a set lookup. Filtering items by P31/P279* inside each
    harvest query is what the pilot found times out per country.
    """
    return f'SELECT ?c WHERE {{ ?c wdt:P279* wd:{root} }}'


def _select(fields):
    return (
        f'SELECT {fields} ?itemLabel ?native ?iso '
        '(GROUP_CONCAT(DISTINCT ?cls; separator="|") AS ?classes) '
        '(GROUP_CONCAT(DISTINCT ?up; separator="|") AS ?parents) '
        '(GROUP_CONCAT(DISTINCT ?up2; separator="|") AS ?grandparents) '
        '(SAMPLE(?cc) AS ?country_code) WHERE {'
    )


def _tail(group):
    # A unit that was dissolved or replaced is history, not a unit: the
    # Soviet raions Armenia's marzer replaced still carry a "located in"
    # link and would outnumber them. "Replaced by" alone does not end a
    # unit that still carries an ISO code: Nigeria's Benue State is
    # "replaced by" the states carved out of it and is a state yet.
    return (
        ' FILTER NOT EXISTS { ?item wdt:P576 ?dissolved }'
        ' FILTER NOT EXISTS { ?item wdt:P1366 ?replaced .'
        ' FILTER NOT EXISTS { ?item wdt:P300 ?anycode } }'
        ' OPTIONAL { ?item wdt:P1705 ?native } OPTIONAL { ?item wdt:P31 ?cls }'
        ' OPTIONAL { ?item wdt:P300 ?iso } OPTIONAL { ?item wdt:P131 ?up }'
        ' OPTIONAL { ?item wdt:P131/wdt:P131 ?up2 } OPTIONAL { ?item wdt:P297 ?cc }'
        ' SERVICE wikibase:label { bd:serviceParam wikibase:language "en" } }'
        f' GROUP BY {group} ?itemLabel ?native ?iso'
    )


def country_children_query(alpha2):
    """Return SPARQL for a country's candidate second-level units.

    Two routes, unioned, because neither is complete on its own: 12 of
    Kenya's 47 counties reach the country only through a former province
    and are found by their ISO 3166-2 code, while Puerto Rico's
    municipios carry no ISO code and are found only as direct children.
    The `direct` column records which route found the item; the class
    the level is built from is chosen among direct children (Spain's
    provinces carry ISO codes too, but only its communities are direct
    children).

    Parameters
    ----------
    alpha2 : str
        ISO 3166-1 alpha-2 code, e.g. 'KE'.
    """
    return (
        _select('?item ?direct') + f' ?country wdt:P297 "{alpha2}" .'
        ' { ?item wdt:P131 ?country . BIND(true AS ?direct) }'
        ' UNION { ?item wdt:P300 ?code .'
        f' FILTER(STRSTARTS(?code, "{alpha2}-")) BIND(false AS ?direct) }}'
        + _tail('?item ?direct')
    )


def parent_children_query(parent_ids):
    """Return SPARQL for the units located in a batch of parent items.

    Parameters
    ----------
    parent_ids : iterable of str
        Q-numbers of the parents, at most `PARENT_BATCH_SIZE` at a time.
    """
    values = ' '.join(f'wd:{q}' for q in parent_ids)
    return (
        _select('?item ?parent')
        + f' VALUES ?parent {{ {values} }} ?item wdt:P131 ?parent .'
        + _tail('?item ?parent')
    )


def country_label_query(alpha2):
    """Return SPARQL for the English label of a country by alpha-2 code."""
    return (
        f'SELECT ?countryLabel WHERE {{ ?country wdt:P297 "{alpha2}" .'
        ' SERVICE wikibase:label { bd:serviceParam wikibase:language "en" } }'
    )


def class_labels_query(class_ids):
    """Return SPARQL for the English labels of a set of classes."""
    values = ' '.join(f'wd:{q}' for q in class_ids)
    return (
        f'SELECT ?c ?cLabel WHERE {{ VALUES ?c {{ {values} }}'
        ' SERVICE wikibase:label { bd:serviceParam wikibase:language "en" } }'
    )


def qid(uri):
    """Return the Q-number of an entity URI, or the value unchanged."""
    return str(uri).rsplit('/', 1)[-1] if isinstance(uri, str) else uri


def select_units(
    harvest,
    admin_classes,
    keep_classes=None,
    electoral_classes=(),
    settlement_classes=(),
    country_classes=(),
):
    """Keep the harvested items that are this level's administrative units.

    A "located in" link is carried by anything with a place: Kenya's 47
    counties come back among 4,680 children, beside settlements, hills
    and electoral constituencies. The unit list is cut in steps, each
    measured against a country it was wrong for. Only items of a class
    in the administrative-entity tree stay, and none that carries its
    own ISO 3166-1 code: Guadeloupe is located in France and coded
    FR-GP, and is a first-level unit of the spine all the same. Where
    two or more candidates carry an ISO 3166-2 code, the level is chosen
    among those (the standard's own list of principal subdivisions),
    settlements included: Korea's metropolitan cities are coded beside
    its provinces. Without codes, classes outside the settlement subtree
    are preferred (Mexico's 460 localities linked straight to the
    country outnumber its 32 states), the parent's direct children
    decide (`direct` True, or every row where the column is absent), and
    a country with a single uncoded child has no level at all (Aruba,
    Sint Maarten). A class whose items are mostly located in other
    candidates, one or two "located in" hops up, is a lower level and
    drops out (France's departments sit in its regions, both coded;
    Paris sits in Grand Paris in its region); an item the dominant
    class's items are located in is the level above and drops out too
    (Kenya's former provinces, still coded, hold 12 of its counties). At
    level 2 every remaining coded class is kept, which is what ISO lists
    (Myanmar's states beside its regions), but only its coded items: an
    uncoded item of a secondary class is a historical region of Spain,
    not a community. The dominant class keeps every candidate, whichever
    route found it. Each unit is labeled with the most common of its
    classes. Classes the sidecar names for this country and level are
    kept whole as well.

    Parameters
    ----------
    harvest : pandas.DataFrame
        Query result with item, itemLabel, native, iso, classes and,
        for level 2, direct.
    admin_classes : set of str
        Q-numbers in the administrative-entity subclass tree.
    keep_classes : iterable of str, optional
        Q-numbers to keep in addition to the dominant class.
    electoral_classes : iterable of str, optional
        Q-numbers in the electoral-district subtree; a class that ties
        with one of these labels the units (Sao Tome's districts are
        its electoral units too).
    settlement_classes : iterable of str, optional
        Q-numbers in the human-settlement subtree; a unit only where no
        other administrative class is on offer.
    country_classes : iterable of str, optional
        Q-numbers of classes whose label names the country; preferred
        as the type of a unit outside the dominant class.

    Returns
    -------
    tuple of (pandas.DataFrame, str)
        The kept rows with a `unit_class` column naming the class each
        row was kept for, and the dominant class. Empty and None when no
        candidate is administrative.
    """
    rows = harvest.copy()
    rows['class_list'] = (
        rows['classes']
        .fillna('')
        .astype(str)
        .str.split('|')
        .map(lambda cs: [qid(c) for c in cs if c])
    )
    rows['admin_list'] = rows['class_list'].map(
        lambda cs: [c for c in cs if c in admin_classes]
    )
    candidates = rows[rows['admin_list'].map(bool)]
    if 'country_code' in candidates:
        own = candidates['country_code'].fillna('').astype(str).str.strip() != ''
        candidates = candidates[~own]
    if candidates.empty:
        return candidates.assign(unit_class=pd.Series(dtype=str)), None
    settlement = set(settlement_classes)
    electoral = set(electoral_classes)
    coded = pd.DataFrame()
    if 'iso' in candidates:
        coded = candidates[candidates['iso'].fillna('').astype(str).str.strip() != '']
        # One coded item is a stray (a park, a former unit): a standard
        # that lists a country's subdivisions lists at least two.
        if coded['item'].nunique() < 2:
            coded = pd.DataFrame()
    if len(coded):
        pool = coded
    else:
        territorial = candidates[
            candidates['admin_list'].map(
                lambda cs: any(c not in settlement for c in cs)
            )
        ]
        pool = territorial if len(territorial) else candidates
        if 'direct' in pool:
            flagged = pool[pool['direct'].astype(str).str.lower() == 'true']
            pool = flagged if len(flagged) else pool
    counted = pool['admin_list'].map(
        lambda cs: [c for c in cs if c not in settlement] or cs
    )
    counts = counted.explode().value_counts()
    ranked = sorted(counts.index, key=lambda c: (-counts[c], c in electoral))
    members = {
        c: set(pool.loc[counted.map(lambda cs, c=c: c in cs), 'item'].map(qid))
        for c in ranked
    }

    def items_of(cls):
        return counted.map(lambda cs, c=cls: c in cs)

    def generic(cls):
        # True of a class that merely restates another: "first-level
        # administrative division" holds Romania's 41 counties and
        # Bucharest, "county of Romania" the 41. The narrower class is
        # the level; the wider one stays kept, for the capital.
        return any(
            members[o] < members[cls] and 2 * len(members[o]) >= len(members[cls])
            for o in ranked
            if o != cls
        )

    def lead(classes):
        first = next((c for c in classes if not generic(c)), classes[0])
        return [first, *(c for c in classes if c != first)]

    ranked = lead(ranked)
    above = set()
    if 'parents' in pool:
        pool_items = set(pool['item'].map(qid))
        parents_of = _links(pool['parents'])
        ancestors_of = parents_of
        if 'grandparents' in pool:
            ancestors_of = parents_of.combine(_links(pool['grandparents']), set.union)
        # A class nested in the pool is a lower level: most of its items
        # are located in another item of the pool, one or two hops up.
        nested = set()
        for cls in ranked:
            mask = counted.map(lambda cs, c=cls: c in cs)
            inside = ancestors_of[mask].map(lambda ps: bool(ps & pool_items))
            if len(inside) and inside.mean() >= 0.5:
                nested.add(cls)
        ranked = lead([c for c in ranked if c not in nested] or ranked)
        # An item holding the dominant class's items is the level above,
        # whatever code it still carries, unless it is of that class
        # itself (Tyumen Oblast holds two other federal subjects); and a
        # class with such an item among its own is the level above too
        # (Kenya's provinces no county points to go with the ones that
        # counties do).
        of_dominant = counted.map(lambda cs, c=ranked[0]: c in cs)
        above = set().union(*parents_of[of_dominant])
        above -= set(pool.loc[of_dominant, 'item'].map(qid))
        risen = [c for c in ranked[1:] if above & members[c]]
        ranked = [c for c in ranked if c not in risen]
        # Its other members go with it: Kenya's Coast Province is also
        # a bare "administrative territorial entity".
        above |= set().union(*(members[c] for c in risen))
    dominant = ranked[0]
    if not len(coded) and 'direct' in pool and len(members[dominant]) < 2:
        # One uncoded child of a country is not a level of it: Aruba's
        # single bare "administrative territorial entity".
        return candidates.iloc[:0].assign(unit_class=pd.Series(dtype=str)), None
    secondary = set(ranked[1:]) if len(coded) and 'direct' in pool else set()
    sidecar = set(keep_classes or ())
    whole = sidecar | {dominant}
    kept = candidates[
        ~candidates['item'].map(qid).isin(above)
        & candidates['admin_list'].map(lambda cs: bool(whole & set(cs)))
    ]
    if secondary:
        extra = coded[
            ~coded['item'].map(qid).isin(above)
            & coded['admin_list'].map(lambda cs: bool(secondary & set(cs)))
        ]
        kept = pd.concat([kept, extra.loc[~extra.index.isin(kept.index)]])
    wanted = whole | secondary
    # A unit is labeled with the dominant class where it has it, else
    # with a class named for the country ("municipality of Romania"
    # for Bucharest, not the "first-level administrative division" it
    # shares with every county), else the most common class that does
    # not merely restate another.
    named = set(country_classes)
    count_of = pool['admin_list'].explode().value_counts().to_dict()

    def label(cs):
        if dominant in cs:
            return dominant
        return min(
            (c for c in cs if c in wanted),
            key=lambda c: (
                c not in named,
                generic(c),
                -count_of.get(c, 0),
                c in settlement,
                int(c[1:] or 0),
            ),
        )

    kept = kept.assign(unit_class=kept['admin_list'].map(label))
    return kept.drop(columns=['class_list', 'admin_list']), dominant


def _links(column):
    """Return each row's set of Q-numbers from a |-joined URI column."""
    return (
        column.fillna('')
        .astype(str)
        .map(lambda ps: {qid(p) for p in ps.split('|') if p})
    )


def type_noun(type_label):
    """Return the head noun of a class label: 'county of Kenya' -> 'county'.

    Wikidata's class labels read "<noun> of <country>" or "<country>
    <noun>", and the noun is what a unit's own label repeats: "Busia
    County" is an instance of "county of Kenya".
    """
    label = str(type_label or '').strip().lower()
    if not label:
        return ''
    if ' of ' in label:
        return label.split(' of ')[0].strip()
    return label.split()[-1]


def strip_type_word(names, type_labels):
    """Drop a trailing or leading type noun from unit labels, where safe.

    "Busia County" becomes "Busia" and "Provincia de Cartago" stays as
    it is: only a bare trailing or leading word equal to the class's
    head noun is removed, and only where the result is not empty and no
    two siblings collapse onto one name, so a real "North County" next
    to a "North" keeps its word.

    Parameters
    ----------
    names : pandas.Series
        Unit labels.
    type_labels : pandas.Series
        Class label per unit, aligned with `names`.

    Returns
    -------
    pandas.Series
        The names, stripped where the rule applies.
    """
    import re

    def strip(name, type_label):
        noun = type_noun(type_label)
        if not noun:
            return name
        pattern = (
            rf'^(?:{re.escape(noun)}\s+(?:of\s+)?)?(.*?)(?:\s+{re.escape(noun)})?$'
        )
        match = re.fullmatch(pattern, str(name).strip(), flags=re.IGNORECASE)
        core = match.group(1).strip() if match else str(name).strip()
        return core or str(name).strip()

    stripped = pd.Series(
        [strip(n, t) for n, t in zip(names, type_labels)], index=names.index
    )
    collided = stripped.duplicated(keep=False) & ~names.duplicated(keep=False)
    return stripped.where(~collided, names)
