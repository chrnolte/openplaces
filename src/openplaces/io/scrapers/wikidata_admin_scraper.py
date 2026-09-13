"""Harvest one country's administrative units at one level from Wikidata.

Wikidata is CC0, so it is the one source whose names, types and parent
links the spine may publish for every country at once. The harvest is
not a single query: an item's "located in" (P131) link is carried by
anything with a place, and a query chained from the country item down
three levels times out for a large country. So each level is fetched
per country, one partition at a time, and a lower level is fetched per
parent, in batches, from the parent level's own ingested output. The
unit list is then cut to one administrative class per country and
level (see `admin_wikidata.select_units`), with a sidecar of reviewed
exceptions.

Nothing here is specific to a country: the level and the parent recipe
are recipe inputs (`scraper_options`), and the exceptions live in
`admin-wikidata-2026_class-overrides.csv` beside the recipes.
"""

from __future__ import annotations

import http.client
import io
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from openplaces.io import admin_wikidata as wd
from openplaces.io import request_headers
from openplaces.recipe import recipe_path

SPARQL = 'https://query.wikidata.org/sparql'
OVERRIDES_RECIPE = 'admin-wikidata-2026'

# Seconds between requests. The endpoint publishes no limit; this keeps
# a 700-request harvest polite rather than bursty.
REQUEST_INTERVAL_S = 0.5
RETRIES = 6
RETRY_WAIT_S = 5
RETRY_CODES = (429, 502, 503, 504)
TRANSIENT = (http.client.RemoteDisconnected, ConnectionError, TimeoutError)

_CLASS_TREE: dict[str, set[str]] = {}


def query(sparql: str, timeout: int = 180, retries: int = RETRIES) -> pd.DataFrame:
    """Run one SPARQL query and return its CSV result.

    Wikidata content-negotiates: the CSV form has to be asked for by
    header, or XML comes back regardless of the query string.
    """
    url = f'{SPARQL}?query={urllib.parse.quote(sparql)}'
    request = urllib.request.Request(
        url, headers={**request_headers(), 'Accept': 'text/csv'}
    )
    # The query service answers 502/503/504 under load and 429 when a
    # client is too eager, all of them transient: wait and retry, with
    # the wait doubling, before giving up on the partition.
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                frame = pd.read_csv(io.BytesIO(response.read()), dtype=str)
            break
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_CODES or attempt == retries:
                raise
            time.sleep(RETRY_WAIT_S * 2**attempt)
        except TRANSIENT:
            # A dropped connection is the same transient as a 502.
            if attempt == retries:
                raise
            time.sleep(RETRY_WAIT_S * 2**attempt)
    time.sleep(REQUEST_INTERVAL_S)
    return frame


def class_tree(root=wd.ADMINISTRATIVE_ENTITY) -> set[str]:
    """Return one class's subclass tree, fetched once per process."""
    if root not in _CLASS_TREE:
        _CLASS_TREE[root] = {wd.qid(c) for c in query(wd.class_tree_query(root))['c']}
    return _CLASS_TREE[root]


def admin_classes() -> set[str]:
    """Return the administrative-entity subclass tree, less what cannot govern.

    A national park reaches the tree through "protected area" (and
    Wikidata codes a few: Los Glaciares carries an Argentine ISO 3166-2
    code), a fort through "military area", a Catholic parish through
    "ecclesiastical district"; the British Virgin Islands' level 2 had
    come out as one fort. Every subtree in `EXCLUDED_TREES` is cut.
    """
    tree = class_tree(wd.ADMINISTRATIVE_ENTITY)
    for root in wd.EXCLUDED_TREES:
        tree = tree - class_tree(root)
    return tree


def load_overrides() -> pd.DataFrame:
    """Return the reviewed class exceptions per country and level."""
    path = recipe_path(None, OVERRIDES_RECIPE, filename='class-overrides.csv')
    if not path.exists():
        return pd.DataFrame(columns=['admin1_id', 'level', 'keep_classes', 'note'])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _kept_classes(alpha2: str, level: int) -> list[str]:
    table = load_overrides()
    rows = table[(table['admin1_id'] == alpha2) & (table['level'] == str(level))]
    return [c for cell in rows['keep_classes'] for c in cell.split('|') if c]


def _parents(parent_recipe_id: str, alpha2: str, level: int) -> list[str]:
    """Return the parent level's Q-numbers for one country."""
    from openplaces.io.readers import get_entities

    column = f'admin{level - 1}_id_wikidata'
    try:
        parents = get_entities(parent_recipe_id, alpha2, geom=False)
    except FileNotFoundError:
        return []
    if column not in parents:
        # An empty placeholder file (the country has no units at the
        # level above) carries no columns at all: nothing to harvest under.
        return []
    return sorted({str(q) for q in parents[column].dropna() if str(q).strip()})


def _has_national_recipe(alpha2: str, level: int) -> bool:
    """Return True when the country's own admin recipe covers this level.

    Those countries (the US Census, DANE, ONS, GISCO, IBGE, PSA layers)
    keep their national source; harvesting them would cost the most
    requests of any country and be discarded by the spine update.
    """
    from openplaces.recipe import find_admin_recipe_id

    found = find_admin_recipe_id(alpha2, level, silent=True)
    return bool(found) and not str(found).startswith('admin-')


def _harvest(alpha2: str, level: int, parent_recipe_id: str | None) -> pd.DataFrame:
    if level == 2:
        return query(wd.country_children_query(alpha2))
    if not parent_recipe_id:
        raise ValueError(f'level {level} needs parent_recipe_id in scraper_options.')
    parents = _parents(parent_recipe_id, alpha2, level)
    frames = []
    for start in range(0, len(parents), wd.PARENT_BATCH_SIZE):
        frames.append(_children_of(parents[start : start + wd.PARENT_BATCH_SIZE]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _children_of(parents: list[str]) -> pd.DataFrame:
    """Query one batch of parents, halving it when the server drops it.

    The endpoint closes the connection on a query that runs past its
    budget rather than answering with an error, and a batch of Yemen's
    governorates with their thousands of villages is such a query. A
    smaller batch is a different query, so it is asked at once instead
    of retrying the one that failed; a single parent gets the full
    retry schedule, since nothing smaller can be asked.
    """
    if len(parents) == 1:
        return query(wd.parent_children_query(parents))
    try:
        return query(wd.parent_children_query(parents), retries=1)
    except (urllib.error.HTTPError, *TRANSIENT) as error:
        if isinstance(error, urllib.error.HTTPError) and error.code not in RETRY_CODES:
            raise
    half = len(parents) // 2
    return pd.concat(
        [_children_of(parents[:half]), _children_of(parents[half:])],
        ignore_index=True,
    )


def fetch(
    partition_id=None,
    target_path=None,
    portal_url=None,
    admin_id_to_download=None,
    label=None,
    redownload=False,
    verbose=False,
    level=2,
    parent_recipe_id=None,
    **options,
):
    """Write one country's units at one level as CSV.

    Parameters
    ----------
    admin_id_to_download : AdminId or str
        The country, as its level-1 id (ISO 3166-1 alpha-2).
    target_path : pathlib.Path
        Where the CSV goes.
    level : int
        Admin level to harvest, 2 to 4.
    parent_recipe_id : str, optional
        The recipe whose output holds the level above, required from
        level 3 up.

    Returns
    -------
    pathlib.Path or None
        The written file, or None when the country has no units at
        this level, which the ingester records as a skipped partition.
    """
    alpha2 = str(admin_id_to_download)
    level = int(level)
    if _has_national_recipe(alpha2, level):
        if verbose:
            print(f'  {alpha2} level {level}: a national admin recipe covers it')
        return None
    harvest = _harvest(alpha2, level, parent_recipe_id)
    if harvest.empty:
        return None
    type_of = _class_labels(harvest)
    kept, dominant = wd.select_units(
        harvest,
        admin_classes(),
        _kept_classes(alpha2, level),
        # The electoral subtree only breaks ties: Wikidata files Kenya's
        # counties, Romania's counties and Brazil's states under it too,
        # so membership cannot tell a constituency from a county.
        electoral_classes=class_tree(wd.ELECTORAL_DISTRICT),
        settlement_classes=class_tree(wd.HUMAN_SETTLEMENT),
        country_classes=_country_classes(alpha2, type_of),
    )
    if kept.empty:
        return None
    out = pd.DataFrame(
        {
            'wikidata_id': kept['item'].map(wd.qid),
            'name': wd.strip_type_word(
                kept['itemLabel'].fillna('').astype(str).str.strip(),
                kept['unit_class'].map(type_of).fillna(''),
            ),
            'name_original': kept['native'].fillna('').astype(str).str.strip(),
            'type': kept['unit_class'].map(type_of).fillna(''),
            'unit_class': kept['unit_class'],
            'iso': kept['iso'].fillna(''),
        }
    )
    if level == 2:
        out['admin1_id'] = alpha2
    else:
        out['parent_wikidata_id'] = kept['parent'].map(wd.qid)
    # A label that is only the Q-number means Wikidata has no English
    # label; the native label stands in, and where there is none either
    # the row stays nameless and the mint gives it a placeholder code.
    unlabeled = out['name'].str.fullmatch(r'Q\d+')
    out.loc[unlabeled, 'name'] = out.loc[unlabeled, 'name_original']
    out = out.drop_duplicates('wikidata_id').sort_values('wikidata_id')
    if verbose:
        print(
            f'  {alpha2} level {level}: {len(harvest):,} candidates, '
            f'{len(out):,} units of class {dominant} ({type_of.get(dominant, "?")})'
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target_path, index=False, encoding='utf-8', lineterminator='\n')
    _write_class_report(target_path, harvest, kept, dominant)
    return target_path


def _class_labels(harvest) -> dict[str, str]:
    """Return English labels for the administrative classes in a harvest."""
    classes = (
        harvest['classes'].fillna('').astype(str).str.split('|').explode().map(wd.qid)
    )
    admin = admin_classes()
    wanted = sorted(c for c in set(classes) if c and c in admin)
    if not wanted:
        return {}
    labels = query(wd.class_labels_query(wanted))
    return dict(zip(labels['c'].map(wd.qid), labels['cLabel']))


def _country_classes(alpha2: str, type_of: dict[str, str]) -> set[str]:
    """Return the classes whose label names the country.

    Wikidata's country-specific classes read "<noun> of <country>" or
    "<Country> <noun>"; the country's own label is what they share.
    The last word of it is enough ("Korea" finds "Special City of
    Korea" for South Korea), and a word under four letters is not
    looked for at all.
    """
    country = query(wd.country_label_query(alpha2))
    if country.empty:
        return set()
    word = str(country['countryLabel'].iloc[0]).split()[-1].lower()
    if len(word) < 4:
        return set()
    return {c for c, label in type_of.items() if word in str(label).lower()}


def _write_class_report(target_path, harvest, kept, dominant):
    """Write the candidate classes beside the CSV, for review.

    The dominant-class rule decides the unit list without a person in
    the loop, so what it dropped has to be visible: Spain's two
    autonomous cities sit in a class of their own beside the
    communities, and only this report shows that they were left out.
    """
    classes = (
        harvest['classes'].fillna('').astype(str).str.split('|').explode().map(wd.qid)
    )
    counts = classes[classes != ''].value_counts()
    admin = admin_classes()
    labels = query(wd.class_labels_query(sorted(c for c in counts.index if c in admin)))
    label_of = dict(zip(labels['c'].map(wd.qid), labels['cLabel']))
    report = pd.DataFrame(
        {
            'class': counts.index,
            'label': [label_of.get(c, '') for c in counts.index],
            'candidates': counts.to_numpy(),
            'administrative': [c in admin for c in counts.index],
            'kept': [c in set(kept['unit_class']) for c in counts.index],
            'dominant': [c == dominant for c in counts.index],
        }
    )
    path = target_path.with_name(target_path.stem + '_classes.csv')
    report.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
