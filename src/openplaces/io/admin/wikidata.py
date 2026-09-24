"""Match spine units to Wikidata items, for CC0 names and a stable join key.

The spine is CC0, so administrative names have to come from a public-domain
source. Wikidata is that source, and it also supplies the Q-number that
replaces the GADM foreign key: stable, language-neutral, and resolvable to
every other identifier system through Wikidata's own properties.

Two findings from piloting this on Colombia decided the approach, and both
went against the obvious design:

**Do not filter by type.** Restricting children to a country's municipality
class looks more precise and is actually worse.
Measured 2026-08-22 on CO: 89.7% matched with the filter, 91.7% without.
A spine unit is often typed as something else in Wikidata (a capital
district, a special unit), and the filter costs a manual class lookup per
country, which the unfiltered query avoids entirely.

**Do not walk the subclass tree.** A generic query using
`P31/P279* wd:Q56061` times out even scoped to a single country, so
harvesting by the parent's ISO code is not merely simpler but the only
version that runs.

**Do not resolve ambiguity automatically.** Where two Wikidata items share a
name under the same parent, this module reports the conflict rather than
choosing. Measured 2026-08-22 on CO: 5.3% of units are ambiguous, and the
duplicates are mostly natural features (Cerro Negro, Cuchilla Pena Negra)
that happen to carry a
`P131` link -- but Otanche is a real municipality with two items, so picking
one silently would sometimes be wrong.
"""

import difflib
import re
from collections import defaultdict

import pandas as pd

from openplaces.io.admin_codes.anchors import normalize_name
from openplaces.io.admin_codes.languages import UNDETERMINED

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
        Result cap. Measured 2026-08-22 on CO: about 25,500 rows, so the
        default leaves room; a country that hits the cap needs paginating.

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
# What a class is called when it is drawn for voting. The subtree under
# ELECTORAL_DISTRICT cannot serve: real units are filed there too.
ELECTORAL_WORDS = re.compile(
    r'constituen|electoral|election|polling|wahlkreis|stimmkreis', re.I
)
# What a class is called when its members are not units at all. A unit
# that was abolished is history, and the query already drops an item
# marked dissolved (P576); what it cannot drop is an item whose only
# type is a class of abolished units, and those classes are numerous
# enough to win a level: Switzerland's level 3 came out as 113 "former
# municipality of Switzerland" and Poland's level 4 as 1,795 "former
# municipality". A list is not a unit either, and Zimbabwe's level 4
# was 56 items of "List of wards of Zimbabwe".
NON_UNIT_WORDS = re.compile(
    r'\bformer\b|\bhistoric|\bdefunct\b|\bdisestablish|\bancient\b|^list of\b', re.I
)
# Subtrees that reach the administrative tree and hold nothing that
# governs: a fort, an airbase, a diocese, an Antarctic claim.
EXCLUDED_TREES = {
    'Q473972': 'protected area',
    'Q97095925': 'military area',
    'Q1492823': 'ecclesiastical district',
    'Q20926517': 'religious administrative territorial entity',
    'Q15239622': 'disputed territory',
    'Q398141': 'school district',
    'Q23413': 'castle',
}

# Class-label nouns that name no kind of unit, so a class carrying one
# is never merged into the level for sharing it. These hold whatever a
# country did not type precisely, at any tier.
GENERIC_NOUNS = frozenset(
    {
        'administrative territorial entity',
        'administrative division',
        'administrative unit',
        'administrative region',
        'territorial entity',
        'division',
        'entity',
        'unit',
        'area',
        'region',
        'territory',
    }
)

#: Parents per SPARQL request. Measured 2026-09-12: 47 Kenyan counties in
#: one request take 3 s; 100 Thai districts 2 s. Larger batches risk the
#: endpoint's 60 s limit on a busy day.
PARENT_BATCH_SIZE = 50

#: Items per detail request in the two-pass fallback. The items are
#: named outright, so the cost is the per-item optionals rather than a
#: set to walk, and a few hundred is comfortably inside the budget.
DETAIL_BATCH_SIZE = 200


def class_tree_query(root=ADMINISTRATIVE_ENTITY):
    """Return SPARQL for every subclass of the administrative-entity class.

    One request per process, after which membership is a set lookup
    (Measured 2026-09-13 on wikidata: 9,045 classes, under 2 s). Filtering
    items by P31/P279* inside each harvest query is what the pilot found
    times out per country.
    """
    return f'SELECT ?c WHERE {{ ?c wdt:P279* wd:{root} }}'


def _select(fields):
    return (
        f'SELECT {fields} ?itemLabel ?enLabel ?native ?iso '
        '(GROUP_CONCAT(DISTINCT ?cls; separator="|") AS ?classes) '
        '(GROUP_CONCAT(DISTINCT ?up; separator="|") AS ?parents) '
        '(GROUP_CONCAT(DISTINCT ?up2; separator="|") AS ?grandparents) '
        '(SAMPLE(?cc) AS ?country_code) WHERE {'
    )


def label_languages(language=None):
    """Return the label-service language list for a country.

    Asking for English gives a unit the name an English encyclopedia
    uses, which is not the name it has: Spain's communities come back
    "Andalusia" and "Aragon" for Andalucia and Aragon, Italy's regions
    "Apulia" for Puglia, and Czechia's "Hradec Kralove" for
    Kralovehradecky kraj. That costs twice. The spine then publishes a
    name no source in the country uses, and the pin that decides whether
    a unit keeps its identifier is a name comparison, so the harvest
    stops matching both the present spine and the geoBoundaries polygons
    pinned to it (Spain pinned 4 of 18).

    The country's own language first and English behind it is what the
    label service is for. English still answers wherever the local label
    is missing, so no unit loses a name by this.

    Parameters
    ----------
    language : str, optional
        The country's language code, from the country-language table.

    Returns
    -------
    str
        A language list for `bd:serviceParam wikibase:language`.
    """
    if not language or language == UNDETERMINED:
        return 'en'
    return f'{language},en'


#: How an item's types are read: from its type *statements*, minus the
#: ones that have ended and the ones marked wrong.
#:
#: `wdt:P31` is the truthy predicate and keeps a statement whose own
#: qualifier says the membership ended, so a unit abolished decades
#: ago still reads as a current unit of its class. P576 does not catch
#: these: what was recorded is that the thing stopped being a
#: municipality, not that it stopped existing. The Netherlands is the
#: worked case, measured 2026-09-22: 1,317 items claim to be a
#: municipality of the Netherlands and 1,088 of those claims carry an
#: end date, leaving 344 against the 342 the country has.
#:
#: Reading statements also picks up a non-preferred type that `wdt:`
#: hides, which is why a count can rise by one or two.
#:
#: Stripping an ended type is not the same as dropping the item: a
#: former Dutch municipality keeps its "former municipality" type,
#: which the abolished-class rule in `select_units` then strikes, and
#: only an item left with no type at all falls out. That is why the
#: two mechanisms are separate.
LIVE_TYPES = (
    ' OPTIONAL { ?item p:P31 ?clsStatement . ?clsStatement ps:P31 ?cls .'
    ' FILTER NOT EXISTS { ?clsStatement pq:P582 ?clsEnd }'
    ' FILTER NOT EXISTS'
    ' { ?clsStatement wikibase:rank wikibase:DeprecatedRank } }'
)


def _tail(group, language=None):
    # A unit that was dissolved is history, not a unit.
    #
    # A "replaced by" (P1366) filter used to stand here too, for the
    # Soviet raions Armenia's marzer replaced, which still carry a
    # "located in" link and would have outnumbered them. It was a blunt
    # proxy: it dropped any replaced item without an ISO code, and most
    # units below level 2 have no ISO code. It cost Sweden twelve
    # current municipalities, the capital among them.
    #
    # Removed 2026-09-22 once `LIVE_TYPES` made it unnecessary, and
    # measured rather than assumed. Of the 28 Swedish municipalities it
    # dropped, 16 have no live type and are still excluded by the newer
    # rule; the 12 that do are the ones that were missing. Harvested
    # both ways: Armenia, the case it was written for, returns its 11
    # provinces either way, and Chile and the Netherlands do not move,
    # while Sweden goes from 278 municipalities to its correct 290.
    return (
        ' FILTER NOT EXISTS { ?item wdt:P576 ?dissolved }'
        ' OPTIONAL { ?item wdt:P1705 ?native }' + LIVE_TYPES + ''
        ' OPTIONAL { ?item wdt:P300 ?iso } OPTIONAL { ?item wdt:P131 ?up }'
        ' OPTIONAL { ?item wdt:P131/wdt:P131 ?up2 } OPTIONAL { ?item wdt:P297 ?cc }'
        # The English label as well as the local one. Asking the label
        # service for a language written in another script returns a
        # name the spine cannot carry: Japan's prefectures come back
        # as their kanji, which no admin code can be derived from, and
        # every prefecture would be re-minted as a placeholder with its
        # 1,809 municipalities orphaned under it.
        ' OPTIONAL { ?item rdfs:label ?enLabel FILTER(LANG(?enLabel) = "en") }'
        ' SERVICE wikibase:label { bd:serviceParam wikibase:language '
        f'"{label_languages(language)}" }} }}'
        f' GROUP BY {group} ?itemLabel ?enLabel ?native ?iso'
    )


def country_children_query(alpha2, language=None):
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
    language : str, optional
        The country's language, asked for ahead of English.
    """
    return (
        _select('?item ?direct') + f' ?country wdt:P297 "{alpha2}" .'
        ' { ?item wdt:P131 ?country . BIND(true AS ?direct) }'
        ' UNION { ?item wdt:P300 ?code .'
        f' FILTER(STRSTARTS(?code, "{alpha2}-")) BIND(false AS ?direct) }}'
        + _tail('?item ?direct', language)
    )


def parent_children_query(parent_ids, language=None):
    """Return SPARQL for the units located in a batch of parent items.

    Parameters
    ----------
    parent_ids : iterable of str
        Q-numbers of the parents, at most `PARENT_BATCH_SIZE` at a time.
    language : str, optional
        The country's language, asked for ahead of English.
    """
    values = ' '.join(f'wd:{q}' for q in parent_ids)
    return (
        _select('?item ?parent')
        + f' VALUES ?parent {{ {values} }} ?item wdt:P131 ?parent .'
        + _tail('?item ?parent', language)
    )


def parent_children_classes_query(parent_ids):
    """Return SPARQL for the children of a batch of parents, classes only.

    The first of two passes for a parent the endpoint will not answer
    for in one query. A German or Indian state, or an Australian one,
    has tens of thousands of "located in" children, and asking for their
    labels, codes, parents and grandparents at once runs past the
    endpoint's budget: it closes the connection rather than answering,
    and a smaller batch is not available below a single parent.

    This asks only what decides whether an item is a candidate at all,
    which is its class. No label service, no second "located in" hop, no
    optionals beyond `P31`. The caller keeps the items whose class is in
    the administrative tree, which for those states is hundreds out of
    tens of thousands, and fetches the rest for those alone.

    Parameters
    ----------
    parent_ids : iterable of str
        Q-numbers of the parents.

    Returns
    -------
    str
        A SPARQL query returning item and classes.
    """
    values = ' '.join(f'wd:{qid(q)}' for q in parent_ids)
    return (
        'SELECT ?item (GROUP_CONCAT(DISTINCT ?cls; separator="|") AS ?classes)'
        f' WHERE {{ VALUES ?parent {{ {values} }} ?item wdt:P131 ?parent .'
        ' FILTER NOT EXISTS { ?item wdt:P576 ?dissolved }' + LIVE_TYPES + ' }'
        ' GROUP BY ?item'
    )


def item_details_query(item_ids, parent_id, language=None):
    """Return SPARQL for named items' details, as the children query gives them.

    The second of the two passes. The items are named outright, so the
    endpoint has no set to walk, and the result has the same columns as
    `parent_children_query` so the selection rule cannot tell which
    route produced it. The parent is bound rather than matched, because
    it is already known and joining to it again would cost a walk.

    Parameters
    ----------
    item_ids : iterable of str
        Q-numbers of the items wanted, a batch at a time.
    parent_id : str
        The parent these items were found under.
    language : str, optional
        The country's language, asked for ahead of English.

    Returns
    -------
    str
        A SPARQL query with the columns `parent_children_query` returns.
    """
    values = ' '.join(f'wd:{qid(i)}' for i in item_ids)
    return (
        _select('?item ?parent')
        + f' VALUES ?item {{ {values} }} BIND(wd:{qid(parent_id)} AS ?parent)'
        + _tail('?item ?parent', language)
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
    class_labels=None,
    unit_classes=None,
    non_unit_classes=(),
):
    """Keep the harvested items that are this level's administrative units.

    A "located in" link is carried by anything with a place.
    Measured 2026-09-13 on KE: 47 counties come back among 4,680 children,
    beside settlements, hills and electoral constituencies. The unit list
    is cut in steps, each
    measured against a country it was wrong for. Only items of a class
    in the administrative-entity tree stay, and none that carries its
    own ISO 3166-1 code: Guadeloupe is located in France and coded
    FR-GP, and is a first-level unit of the spine all the same. Where
    two or more candidates carry an ISO 3166-2 code, the level is chosen
    among those (the standard's own list of principal subdivisions),
    settlements included: Korea's metropolitan cities are coded beside
    its provinces. Without codes, classes outside the settlement subtree
    and the constituency classes are preferred (Mexico's 460 localities
    linked straight to the country outnumber its 32 states; Kenya's 290
    constituencies its 195 sub-counties), the parent's direct children
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
        Q-numbers of classes that are constituencies by name. Without
        codes they rank below every other class, and a class that ties
        with one of these labels the units (Sao Tome's districts are
        its electoral units too). Not the electoral-district subtree:
        Wikidata files France's departments and the Netherlands'
        municipalities under it as well.
    settlement_classes : iterable of str, optional
        Q-numbers in the human-settlement subtree; a unit only where no
        other administrative class is on offer.
    country_classes : iterable of str, optional
        Q-numbers of classes whose label names the country; preferred
        as the type of a unit outside the dominant class.
    class_labels : dict, optional
        English label per class Q-number. Used to keep the classes that
        name the same kind of unit as the dominant one (Burundi's bare
        "province" beside its "province of Burundi"); without it those
        units are lost.
    non_unit_classes : iterable of str, optional
        Q-numbers of classes whose members are not units: abolished
        units and lists. Struck from every item's types before anything
        else is decided.
    unit_classes : iterable of str, optional
        Q-numbers that *are* this level, from a reviewed sidecar row.
        Given, they settle the level outright and every rule below is
        skipped, because the rules have already been read and found
        wrong for this country. `keep_classes` is the weaker
        instruction: keep these too, beside whatever the rules choose.

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
    # A class of abolished units, or a list, is struck from an item's
    # types rather than demoted: an item whose only remaining type is
    # one of those is not a unit and leaves with it, while an item that
    # is also typed as a current unit keeps that type and stays.
    usable = set(admin_classes) - set(non_unit_classes)
    rows['admin_list'] = rows['class_list'].map(
        lambda cs: [c for c in cs if c in usable]
    )
    candidates = rows[rows['admin_list'].map(bool)]
    if 'country_code' in candidates:
        own = candidates['country_code'].fillna('').astype(str).str.strip() != ''
        candidates = candidates[~own]
    if unit_classes:
        return _declared_units(candidates, list(unit_classes))
    if candidates.empty:
        return candidates.assign(unit_class=pd.Series(dtype=str)), None
    # A settlement class whose label names the country is that country's
    # own administrative class, not a place people happen to live in:
    # Guam's 19 villages and Taiwan's townships are its units and are
    # filed under human settlement all the same. Demoting them left Guam
    # with 7 generic municipalities and Taiwan with 25 indigenous areas.
    # This only reaches the uncoded branch below, so Mexico's 460
    # "locality of Mexico" items still never compete with its 32 coded
    # states.
    settlement = set(settlement_classes) - set(country_classes)
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
        # Without codes, a class drawn for voting or for living in is
        # the level only when nothing territorial is on offer: Kenya's
        # level 3 is its sub-counties, not its 290 constituencies, and
        # Mexico's 460 localities do not outrank its 32 states. A place
        # people live in still beats a voting map.
        pool = candidates
        for demoted in (settlement | electoral, electoral):
            kept_rows = candidates[
                candidates['admin_list'].map(
                    lambda cs, d=demoted: any(c not in d for c in cs)
                )
            ]
            if len(kept_rows):
                pool = kept_rows
                break
        if 'direct' in pool:
            flagged = pool[pool['direct'].astype(str).str.lower() == 'true']
            pool = flagged if len(flagged) else pool
    counted = pool['admin_list'].map(
        lambda cs: (
            [c for c in cs if c not in settlement | electoral]
            or [c for c in cs if c not in electoral]
            or cs
        )
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
        # A class kept for naming the same unit as the dominant one has
        # no members in the pool, so it restates nothing and is asked
        # about only when labeling a row.
        mine = members.get(cls, set())
        return any(
            members[o] < mine and 2 * len(members[o]) >= len(mine)
            for o in ranked
            if o != cls
        )

    def lead(classes):
        first = next((c for c in classes if not generic(c)), classes[0])
        return [first, *(c for c in classes if c != first)]

    ranked = lead(ranked)
    above = set()
    siblings = set()
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
        # A class sharing the dominant class's parents sits beside it,
        # not under it, and a country whose level is split across two
        # such classes loses the smaller one otherwise: Poland's 65
        # cities with powiat rights beside its 314 powiats, and the same
        # shape in Italy's metropolitan cities and Romania's towns.
        # Only a class the country named its own qualifies, so a generic
        # one cannot sweep a lower tier in, and the parent test keeps
        # out what really is lower (Poland's villages, located in its
        # gminas).
        #
        # Only where no ISO code is on offer. Where codes exist they
        # already decide which classes stand beside the level, and by
        # coded item rather than by class, so reaching past them let two
        # historical "region of Spain" items in beside the autonomous
        # communities.
        for cls in ranked[1:] if not len(coded) else ():
            if cls not in set(country_classes):
                continue
            mask = counted.map(lambda cs, c=cls: c in cs)
            beside = parents_of[mask].map(lambda ps: bool(ps & above))
            if len(beside) and beside.mean() >= 0.5:
                siblings.add(cls)
        risen = [c for c in ranked[1:] if above & members[c]]
        ranked = [c for c in ranked if c not in risen]
        # Its other members go with it: Kenya's Coast Province is also
        # a bare "administrative territorial entity".
        above |= set().union(*(members[c] for c in risen))
    dominant = ranked[0]
    # Every administrative class on offer, not only those of the pool:
    # the class that names the same unit as the dominant one may have no
    # coded item at all, which is what keeps it out of the pool.
    offered = set(candidates['admin_list'].explode().dropna())
    if not len(coded) and len(members[dominant]) < 2 and len(offered) > 1:
        # One uncoded item is not a level when the harvest offered other
        # classes and they were all ruled out: Aruba's single bare
        # "administrative territorial entity" beside its settlements,
        # Belize's one "administrative region" left over from eight
        # classes, Burundi's one commune. What this must not touch is a
        # parent that really holds one child, which arrives as a harvest
        # of one item in one class and keeps it.
        return candidates.iloc[:0].assign(unit_class=pd.Series(dtype=str)), None
    # Where codes are on offer, every remaining coded class stands
    # beside the level, by coded item rather than by class. At level 2
    # that is what ISO lists (Myanmar's states beside its regions); at
    # lower levels it is the same fact (Italy's metropolitan cities
    # beside its provinces, both coded, and without this the country
    # keeps 83 of its 107).
    secondary = set(ranked[1:]) if len(coded) else set()
    sidecar = set(keep_classes or ())
    whole = (
        sidecar
        | {dominant}
        | _bare_synonym(dominant, offered, class_labels)
        | (siblings - {dominant})
    )
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


def _declared_units(candidates, unit_classes):
    """Keep exactly the classes a reviewed sidecar row names as the level.

    Two countries need this and no rule can reach them. Kenya's
    sub-counties and Namibia's constituencies are the administrative
    unit below the county and the region, and Wikidata types both as
    constituencies, which the electoral demotion exists to remove. The
    demotion is right for the 19 countries it was added for and wrong
    for these two, and the difference is a fact about Kenya and Namibia,
    not a pattern in the data.

    Parameters
    ----------
    candidates : pandas.DataFrame
        The administrative candidates, with an `admin_list` column.
    unit_classes : list of str
        Q-numbers that are this level, most authoritative first.

    Returns
    -------
    tuple of (pandas.DataFrame, str)
        The kept rows and the first named class.
    """
    wanted = set(unit_classes)
    kept = candidates[candidates['admin_list'].map(lambda cs: bool(wanted & set(cs)))]
    if kept.empty:
        return kept.assign(unit_class=pd.Series(dtype=str)), None
    kept = kept.assign(
        unit_class=kept['admin_list'].map(
            lambda cs: next(c for c in unit_classes if c in cs)
        )
    )
    return kept.drop(columns=['class_list', 'admin_list']), unit_classes[0]


def _bare_synonym(dominant, classes, class_labels):
    """Return the unqualified class the dominant one refines, if it is on offer.

    Wikidata types a country's units under a country-specific class and,
    for some of them, under the plain class it refines: three of
    Burundi's five provinces are a "province of Burundi" and two are
    simply a "province". They are one tier, and taking only the larger
    class costs the country the rest of its units.

    Only the bare class qualifies, never another qualified one. Sharing
    the head noun is not enough: Vietnam's "province of South Vietnam"
    holds a province of a state that no longer exists, and merging it
    would put Kiến Hòa back on the map.

    A class whose noun is vague on its own ("administrative territorial
    entity", "division", "region") is not merged in either: the bare
    class then holds whatever a country did not type precisely, at any
    tier.

    Parameters
    ----------
    dominant : str
        Q-number of the class the level is built from.
    classes : iterable of str
        Every administrative class on offer among the candidates.
    class_labels : dict or None
        English label per class Q-number. Without it nothing is merged,
        because the noun cannot be read.

    Returns
    -------
    set of str
        Q-numbers to keep beside the dominant class.

    Examples
    --------
    >>> labels = {'Q1': 'province of Burundi', 'Q2': 'province', 'Q3': 'district'}
    >>> sorted(_bare_synonym('Q1', ['Q1', 'Q2', 'Q3'], labels))
    ['Q2']
    """
    if not class_labels:
        return set()
    noun = type_noun(class_labels.get(dominant, ''))
    if (
        not noun
        or noun in GENERIC_NOUNS
        or noun == str(class_labels.get(dominant, '')).strip().lower()
    ):
        return set()
    return {
        c
        for c in classes
        if c != dominant and str(class_labels.get(c, '')).strip().lower() == noun
    }


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


def is_latin(text):
    """True when a name is written in the Latin script.

    The spine's `name` has to be, because an admin code is derived from
    it and the code vocabulary is ASCII; a name in another script yields
    no code and the unit is minted as a placeholder instead. The native
    spelling is not lost by this: it is what `name_original` carries.

    A name with no letters at all (a number, a symbol) counts as Latin,
    since nothing about it argues for the English label instead.

    Parameters
    ----------
    text : str
        A name.

    Returns
    -------
    bool
        True when at least half its letters are ASCII.

    Examples
    --------
    >>> is_latin('Hokkaido')
    True
    >>> is_latin('北海道')
    False
    >>> is_latin('Ceuta')
    True
    """
    import unicodedata

    letters = [c for c in unicodedata.normalize('NFD', str(text)) if c.isalpha()]
    if not letters:
        return True
    return sum(1 for c in letters if c.isascii()) / len(letters) >= 0.5


def prefer_latin(labels, english):
    """Take the local label unless it is in another script.

    Parameters
    ----------
    labels : pandas.Series
        Labels as the label service returned them, in the country's own
        language where it has one.
    english : pandas.Series
        The English label per item, aligned with `labels`.

    Returns
    -------
    pandas.Series
        The local label where the spine can carry it, else the English
        one, else the local label after all.
    """
    english = english.reindex(labels.index).fillna('').astype(str).str.strip()
    usable = labels.map(is_latin) | (english == '')
    return labels.where(usable, english)


def _strip_pack_type_words(name, pack):
    """Drop a leading or trailing type word in the country's own language.

    Asking Wikidata for a local label brings the local type word with
    it: Chile's regions come back "Region de Antofagasta" where the
    spine and every Chilean source say "Antofagasta", and Czechia's
    "Jihomoravsky kraj" against "Jihomoravsky". The class label cannot
    strip these, because it is in English and says "region of Chile".
    The language pack already holds the vocabulary, for code
    abbreviation; matching folded lets an accented "Region" meet the
    unaccented entry.

    Parameters
    ----------
    name : str
        A unit label.
    pack : LanguagePack
        The country's vocabulary.

    Returns
    -------
    str
        The name without its type word, or unchanged when stripping
        would leave nothing.
    """
    from openplaces.io.admin_codes.languages import fold_diacritics

    def is_type(token):
        folded = fold_diacritics(token).lower()
        return pack.is_type_word(token) or pack.is_type_word(folded)

    def is_filler(token):
        folded = fold_diacritics(token).lower()
        return any(
            test(candidate)
            for test in (pack.is_article, pack.is_preposition)
            for candidate in (token, folded)
        )

    tokens = str(name).split()
    while tokens and is_type(tokens[0]):
        tokens = tokens[1:]
        # One binding word, not a run: "Region de Los Lagos" is Los
        # Lagos, and eating the article too leaves "Lagos".
        if tokens and is_filler(tokens[0]):
            tokens = tokens[1:]
    while tokens and is_type(tokens[-1]):
        tokens = tokens[:-1]
    return ' '.join(tokens) or str(name).strip()


def strip_type_word(names, type_labels, pack=None):
    """Drop a trailing or leading type noun from unit labels, where safe.

    "Busia County" becomes "Busia" and "Provincia de Cartago" stays as
    it is: only a bare trailing or leading word equal to the class's
    head noun is removed, and only where the result is not empty and no
    two siblings collapse onto one name, so a real "North County" next
    to a "North" keeps its word.

    With a language pack, the country's own type words are removed too,
    which is what a locally labeled harvest needs.

    Parameters
    ----------
    names : pandas.Series
        Unit labels.
    type_labels : pandas.Series
        Class label per unit, aligned with `names`.
    pack : LanguagePack, optional
        The country's vocabulary, for type words the English class label
        cannot name.

    Returns
    -------
    pandas.Series
        The names, stripped where the rule applies.
    """
    import re

    def strip(name, type_label):
        noun = type_noun(type_label)
        core = str(name).strip()
        if noun:
            pattern = (
                rf'^(?:{re.escape(noun)}\s+(?:of\s+)?)?(.*?)(?:\s+{re.escape(noun)})?$'
            )
            match = re.fullmatch(pattern, core, flags=re.IGNORECASE)
            core = (match.group(1).strip() if match else core) or core
        if pack is not None:
            core = _strip_pack_type_words(core, pack)
        return core or str(name).strip()

    stripped = pd.Series(
        [strip(n, t) for n, t in zip(names, type_labels)], index=names.index
    )
    collided = stripped.duplicated(keep=False) & ~names.duplicated(keep=False)
    return stripped.where(~collided, names)
