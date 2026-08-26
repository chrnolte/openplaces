"""Import the parcel-id conversion table from the grid search it came from.

`geo/parcel_id_links.csv` tells `parcel_id_local` how to normalize a raw
assessor parcel id into a locally comparable key. It was ported from the
predecessor `places` codebase, which had derived it by grid search over
every (pattern, conversion) pair that could join a county's parcel and
tax rolls, keeping the joint match rate of each.

Two things went wrong with that port, and this module fixes both.

**It kept the answer and dropped the evidence.** One winning pair per
county was committed; the alternatives, their measured match rates, the
id counts they were measured on, and which raw column the id came from
were all discarded. So there was no way to tell a link measured across
200,000 parcels from one fitted to 40, and nothing to fall back on when
a newly ingested source did not fit the one committed rule. Measured
2026-08-25, every instruction in the ported file appears in the grid
search or its manual overrides, so nothing is lost by importing afresh
and everything the port dropped is recovered.

**It was keyed on `admin_id`, which openplaces re-mints.** Three
retargets by identifier string left 163 duplicated ids (a hard
`ValueError` in `_resolve_instruction`, which indexes on `admin_id`), 54
ids naming no live unit, and twenty rows carrying a neighboring county's
rule after an initials-based code collision - Broward took Bradford's,
Champaign took Christian's. The key here is the unit's own national
code, which openplaces does not issue and cannot move.

Run it as::

    python -m openplaces.geo.import_parcel_id_links --auto <path> --manual <path>

After a re-mint moves the admin ids, the `admin_id` and `name` columns go
stale without anything breaking, because they are comments and the key is
the national code. Refresh them with::

    python -m openplaces.geo.import_parcel_id_links --refresh-ids

See `plans/revalidate-parcel-id-links-against-its-ztrax-source.md`.
"""

import argparse
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

from openplaces.geo.ids import _PARCEL_ID_LINK_LEVELS
from openplaces.path import spine_path

LINKS_PATH = Path(__file__).parent / 'parcel_id_links.csv'

# Vintage of county subdivision codes the grid search was keyed on. The
# Census re-codes a subdivision whenever its legal status changes, so a
# 2016 code cannot always be matched against a current one; this layer
# supplies the name that bridges the gap.
COUSUB_VINTAGE_RECIPE_ID = 'US_admin-census-2016_admin4'

# Commit whose committed rule each unit keeps where it can. The
# predecessor resolved a unit by reading its per-county search file and
# taking the first row, and those files are not what the pooled search
# table preserves: thousands of committed rules are a different member of
# a group of candidates *tied at the same match rate*. Preferring the rule
# in use keeps a county's parcel key still whenever the covering set has
# room for it.
DEFAULT_PRESERVE_REF = '89dc8d6'
LINKS_REPO_PATH = 'src/openplaces/geo/parcel_id_links.csv'
SPINE_REPO_PATH = (
    'src/openplaces/recipes/_all/admin/spine/2026/admin-spine-2026_admin{}.csv'
)
_PORTED_SIDES = {
    'parcel': ('pattern_parcel', 'conv_parcel'),
    'tax': ('pattern_tax', 'conv_tax'),
}

OUTPUT_COLUMNS = [
    'country_id',
    'admin_id_admin1',
    'admin_id',
    'name',
    'kind',
    'origin',
    'source_column',
    'pattern',
    'conv',
    'match_rate',
    'n_ids_parcel',
    'n_ids_tax',
]

# What identifies one conversion, independently of the unit it is applied
# to. The set of distinct values of this across the whole table is the
# conversion library: a few hundred rules serving thousands of counties,
# and the set worth trying when a newly ingested source does not fit the
# rule its unit was given.
RULE_COLUMNS = ['pattern', 'conv', 'source_column']

# The grid search names the two sides `pc` (parcel) and `za` (the tax
# roll). `kind` keeps openplaces' own words for them, which is what
# `_resolve_instruction` already takes.
SIDES = {
    'parcel': {
        'pattern': 'pattern_pc',
        'conv': 'apn_conv_pc',
        'source_column': 'apn_pc',
    },
    'tax': {
        'pattern': 'pattern_za',
        'conv': 'apn_conv_za',
        'source_column': 'apn_za',
    },
}
MATCH_RATE = 'success'
COUNT_COLUMNS = {'n_pc': 'n_ids_parcel', 'n_za': 'n_ids_tax'}

# A national code concatenates its levels, so a change at one level
# rewrites the code of every unit below it even when those units did not
# change: Connecticut replaced its counties with planning regions in
# 2022, and every town's ten-digit GEOID moved in its county segment
# alone. The outer segments are the stable ones.
_COUNTY_CODE_LENGTH = 5
_SUBDIVISION_CODE_LENGTH = 10
_TOP_SEGMENT_LENGTH = 2

# Status words a statistical agency appends to a unit's name to tell two
# of its own registers apart. They are stripped from both sides before
# comparing, because which side carries one is not stable: the grid
# search's vintage has 'Amesbury Town' against today's 'Amesbury', and
# 'North Attleborough' against today's 'North Attleborough Town'. A
# stripped name only ever resolves when it matches exactly one unit, so
# a real place called 'X City' cannot be swallowed by a neighbor 'X'.
_STATUS_WORDS = (
    ' TOWN',
    ' CITY',
    ' TOWNSHIP',
    ' VILLAGE',
    ' BOROUGH',
    ' PLANTATION',
)


def _strip_status_word(name: str) -> str:
    """Return a unit name without its trailing register-status word."""
    name = str(name).strip().upper()
    for word in _STATUS_WORDS:
        if name.endswith(word):
            return name[: -len(word)].strip()
    return name


def _region_of(admin_id: str) -> str:
    """Return an admin id's country and first-level-subdivision segments.

    The unit's own parent is not usable as a name scope: a re-mint can
    retire the parent (Connecticut's counties) while the unit survives,
    so the scope has to be coarse enough to outlive the change.
    """
    return '-'.join(admin_id.split('-')[:2])


def _read_live_spine() -> pd.DataFrame:
    """Return every unit the current spine names at the linked levels."""
    frames = []
    for level in _PARCEL_ID_LINK_LEVELS:
        column = f'admin{level}_id'
        spine = pd.read_csv(
            spine_path(level),
            dtype=str,
            keep_default_na=False,
            usecols=[column, f'{column}_admin1', 'name'],
        )
        frames.append(
            spine.rename(columns={column: 'admin_id', f'{column}_admin1': 'code'})[
                ['admin_id', 'code', 'name']
            ]
        )
    live = pd.concat(frames, ignore_index=True).drop_duplicates('admin_id')
    for column in live.columns:
        live[column] = live[column].fillna('').astype(str).str.strip()
    return live


def _read_vintage_names(recipe_id: str | None) -> dict[str, str]:
    """Return national code to unit name, from a historical admin layer.

    Returns an empty mapping when the layer is unavailable, because it
    only resolves a residue: measured 2026-08-25, 2,494 of the grid
    search's 2,524 subdivision keys resolve on their code alone.
    """
    if recipe_id is None:
        return {}
    from openplaces.api import get_entities
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id(recipe_id)
    level = _vintage_level(recipe_id)
    column = f'admin{level}_id_admin1'
    layer = get_entities(recipe, admin_id='US', geom=False)
    return dict(zip(layer[column].astype(str), layer['name'].astype(str)))


def _vintage_level(recipe_id: str) -> int:
    """Return the admin level a vintage recipe id names."""
    return int(recipe_id.rsplit('admin', 1)[-1])


class _UnitIndex:
    """Resolve a national code to the admin id that holds it today.

    Three tiers, tried in order of directness. Every one is a general
    rule about how national code schemes behave, so no state, county or
    code is named here.
    """

    def __init__(
        self,
        live: pd.DataFrame,
        vintage_names: dict[str, str],
        country_id: str,
    ):
        # Scoped to one country before anything else. A national code
        # means nothing outside the scheme that issued it: Colombia's
        # DANE codes are five digits, so an unscoped lookup pairs Greene
        # County, Arkansas with Argelia, Antioquia, and an unscoped
        # two-digit prefix reads '25' as Victoria, Australia rather than
        # Massachusetts.
        live = live[live['admin_id'].str.startswith(f'{country_id}-')]
        self.name_of = dict(zip(live['admin_id'], live['name']))
        self.code_of = dict(zip(live['admin_id'], live['code']))
        self.vintage_names = vintage_names
        self.by_code: dict[str, str] = {}
        self.by_subdivision: dict[str, list[str]] = {}
        self.by_name: dict[tuple[str, str], list[str]] = {}
        # A code's leading segment names the first-level subdivision the
        # unit sits in, but only the spine knows which admin id that is.
        self.region_of: dict[str, str] = {}
        for admin_id, code, name in zip(live['admin_id'], live['code'], live['name']):
            region = _region_of(admin_id)
            if code:
                self.by_code.setdefault(code, admin_id)
                self.region_of.setdefault(code[:_TOP_SEGMENT_LENGTH], region)
                if len(code) == _SUBDIVISION_CODE_LENGTH:
                    outer = code[:_TOP_SEGMENT_LENGTH] + code[_COUNTY_CODE_LENGTH:]
                    self.by_subdivision.setdefault(outer, []).append(admin_id)
            if name:
                key = (region, _strip_status_word(name))
                self.by_name.setdefault(key, []).append(admin_id)

    def resolve(self, code: str) -> tuple[str, str]:
        """Return (admin id, tier) for one national code, or ('', '')."""
        hit = self.by_code.get(code)
        if hit:
            return hit, 'code'

        if len(code) == _SUBDIVISION_CODE_LENGTH:
            outer = code[:_TOP_SEGMENT_LENGTH] + code[_COUNTY_CODE_LENGTH:]
            candidates = self.by_subdivision.get(outer, [])
            if len(candidates) == 1:
                return candidates[0], 'subdivision'

        name = self.vintage_names.get(code)
        region = self.region_of.get(code[:_TOP_SEGMENT_LENGTH])
        if name and region:
            candidates = self.by_name.get((region, _strip_status_word(name)), [])
            if len(candidates) == 1:
                return candidates[0], 'name'
        return '', ''


def _long_candidates(frame: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Split a grid-search table into one row per side and candidate.

    The search is a cross product of a parcel-side rule and a tax-side
    rule scored jointly, so a county with eight of each contributes 64
    rows saying very little. Splitting the sides apart recovers the
    distinct rules: 141,440 source rows hold 48,159 of them.

    Row order is the search's own ranking - each group arrives sorted by
    joint match rate, ties already broken - so a rule keeps the position
    of its first appearance rather than being re-sorted here. That order
    is what identifies the rule the predecessor resolved, which
    `_smallest_covering_set` prefers to keep.
    """
    frame = frame.reset_index(drop=True)
    rows = []
    for kind, columns in SIDES.items():
        keep = ['code', *columns.values()]
        if MATCH_RATE in frame.columns:
            keep += [MATCH_RATE, *COUNT_COLUMNS]
        side = frame[keep].copy()
        side = side.rename(columns={v: k for k, v in columns.items()} | COUNT_COLUMNS)
        side['kind'] = kind
        side['order'] = frame.index
        rows.append(side)
    long = pd.concat(rows, ignore_index=True)
    for column in ('pattern', 'conv', 'source_column'):
        long[column] = long[column].fillna('').astype(str).str.strip()
    if MATCH_RATE not in long.columns:
        long[MATCH_RATE] = pd.NA
        for column in COUNT_COLUMNS.values():
            long[column] = pd.NA
    long = long.rename(columns={MATCH_RATE: 'match_rate'})
    long['origin'] = origin
    # The same rule appears once per partner it was scored against. Its
    # first appearance carries the best joint rate it ever reached, since
    # the group arrives sorted, so keeping the first row keeps both the
    # rate and the search's ordering.
    long = long.sort_values('order', kind='stable')
    return long.drop_duplicates(['code', 'kind', 'pattern', 'conv', 'source_column'])


def _one_rule_per_unit(
    long: pd.DataFrame, preferred: dict[tuple[str, str], tuple]
) -> pd.DataFrame:
    """Reduce the candidates to a single rule per unit and side.

    The search offers a unit far more rules than it needs: 48,159 distinct
    rules across 10,972 (unit, side) pairs, most of them alternatives tied
    at the same match rate. Storing them all repeats the same conversion
    thousands of times and still leaves a reader to guess which to use, so
    each pair keeps exactly one - the hand-written override if it has one,
    otherwise the rule already committed for it, otherwise the search's
    own first choice.

    **Ties are not substitutions.** It is tempting to go further and pick,
    among the rules tied at a pair's best rate, whichever ones make the
    shared vocabulary smallest - that reduces 821 distinct conversions to
    600. Do not: a tie in `success` says two rules matched equally well on
    *the data the search ran over*, not that they are the same function.
    Dane County, Wisconsin is the counterexample. Its committed rule
    (`Dx`, no conversion) and a rule tied with it
    (`split_groups: 0|1 & drop_cols: 0_0 & skip_empty: 1`) score
    identically there, because that county's ids share a leading segment -
    but the second discards it, turning `1005` into `5`, and would collide
    with every other id ending in 5 the moment a differently formatted
    source arrived. Guarding against rules that add a discarding operation
    still leaves substitutions that drop a *different* column at the same
    cost, so there is no cheap static test for interchangeability. The
    rule that was measured for a unit is the rule the unit keeps.

    Nothing regresses. Every pair keeps a rule at its own best measured
    rate; where the preferred rule is somehow below it, the best-scoring
    candidate wins instead.

    Parameters
    ----------
    long : pandas.DataFrame
        One row per (unit, side, candidate), carrying `match_rate`.
    preferred : dict
        (admin id, kind) to the rule tuple that should win for it.
    """
    rule_of = list(zip(*[long[column] for column in RULE_COLUMNS]))
    pair_of = list(zip(long['admin_id'], long['kind']))
    rate_of = pd.to_numeric(long['match_rate'], errors='coerce').tolist()

    best: dict[tuple[str, str], float] = {}
    for pair, rate in zip(pair_of, rate_of):
        if pd.notna(rate) and rate > best.get(pair, float('-inf')):
            best[pair] = rate

    # The preferred rule wins outright unless it was scored and scored
    # below what the same pair reached elsewhere. An unscored rule is a
    # hand-written override with no search behind it, and a person
    # deciding beats a measurement that was never taken - dropping it for
    # want of a number is how the overrides went missing on the first
    # attempt at this.
    scored = {}
    for pair, rule, rate in zip(pair_of, rule_of, rate_of):
        scored[(pair, rule)] = rate

    chosen: dict[tuple[str, str], tuple] = {}
    for pair, rule in zip(pair_of, rule_of):
        if preferred.get(pair) != rule:
            continue
        rate = scored[(pair, rule)]
        if pair not in best or pd.isna(rate) or rate >= best[pair]:
            chosen[pair] = rule
    # Whatever is left had a preferred rule that a better-scoring
    # candidate beats, so take the best-scoring one instead.
    for pair, rule, rate in zip(pair_of, rule_of, rate_of):
        if pair in chosen or pd.isna(rate) or rate < best.get(pair, rate):
            continue
        chosen.setdefault(pair, rule)

    selected = [chosen.get(pair) == rule for pair, rule in zip(pair_of, rule_of)]
    return long[selected].drop_duplicates(['admin_id', 'kind'])


def _git_show(repo_root: Path, ref: str, repo_path: str) -> pd.DataFrame:
    """Read one CSV as of a git ref, without checking anything out."""
    blob = subprocess.run(
        ['git', 'show', f'{ref}:{repo_path}'],
        cwd=repo_root,
        capture_output=True,
        check=True,
    ).stdout.decode('utf-8-sig')
    return pd.read_csv(StringIO(blob), dtype=str, keep_default_na=False)


def _ported_rules(repo_root: Path, ref: str) -> set[tuple[str, str, str, str]]:
    """Return the (code, kind, pattern, conv) rules committed at a ref.

    The committed table names admin ids, so it is read together with the
    spine of the same commit: that is the only thing that says which unit
    each id meant when the rule was written.
    """
    ported = _git_show(repo_root, ref, LINKS_REPO_PATH)
    codes: dict[str, str] = {}
    for level in _PARCEL_ID_LINK_LEVELS:
        spine = _git_show(repo_root, ref, SPINE_REPO_PATH.format(level))
        column = f'admin{level}_id'
        codes.update(zip(spine[column], spine[f'{column}_admin1']))
    rules = set()
    for _, row in ported.iterrows():
        code = codes.get(row['admin_id'], '')
        if not code:
            continue
        for kind, (pattern, conv) in _PORTED_SIDES.items():
            rules.add((code, kind, row[pattern].strip(), row[conv].strip()))
    return rules


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the national code each grid-search row is keyed on."""
    frame = frame.copy()
    subdivision = frame['csd_id'].fillna('').astype(str).str.strip()
    frame['code'] = frame['fips'].astype(str).str.strip() + subdivision
    return frame


def build(
    auto_path: Path,
    manual_path: Path | None = None,
    vintage_recipe_id: str | None = COUSUB_VINTAGE_RECIPE_ID,
    country_id: str = 'US',
    repo_root: Path | None = None,
    preserve_ref: str | None = DEFAULT_PRESERVE_REF,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return the conversion table, keyed on each unit's national code.

    Parameters
    ----------
    auto_path : Path
        Grid-search table: one row per (unit, parcel rule, tax rule) with
        the joint match rate and the id counts it was measured on.
    manual_path : Path, optional
        Hand-written overrides, in the same column layout but unscored.
        They win over the search wherever both name a unit, which is the
        precedence the predecessor applied at read time.
    vintage_recipe_id : str, optional
        Admin layer of the vintage the grid search was keyed on, used to
        name a unit whose code has since been retired. Pass None to skip
        the name tier.
    country_id : str, optional
        Country whose scheme issued the codes. The grid search covers one
        country, and a code means nothing outside the scheme that issued
        it: Colombia's DANE codes are five digits, like a county FIPS.
    repo_root : Path, optional
        Repository or worktree to read the preserved commit from.
        Defaults to the installed package's own repository.
    preserve_ref : str, optional
        Commit whose committed rule keeps rank 0 for each unit it names.
        Pass None to rank purely by the search, which changes the parcel
        key of every county whose committed rule was one of several tied
        candidates.
    verbose : bool, optional
        Print the per-tier resolution and the resulting shape.

    Returns
    -------
    pandas.DataFrame
        One row per (unit, side, candidate), ranked best first within
        each unit and side. `rank` 0 is the rule `parcel_id_local` uses;
        the rest are what to try when a newly ingested source does not
        fit it.

    Raises
    ------
    ValueError
        If the resulting key is not unique. Two rows claiming one rank
        for one unit is the defect this rebuild exists to remove.
    """
    auto = _keyed(pd.read_parquet(auto_path))
    candidates = [_long_candidates(auto, 'auto')]
    if manual_path is not None:
        manual = _keyed(pd.read_csv(manual_path, dtype=str))
        candidates.append(_long_candidates(manual, 'manual'))
    long = pd.concat(candidates, ignore_index=True)

    if preserve_ref is None:
        ported_rules: set[tuple[str, str, str, str]] = set()
    else:
        root = repo_root or Path(__file__).resolve().parents[3]
        ported_rules = _ported_rules(root, preserve_ref)

    live = _read_live_spine()
    index = _UnitIndex(
        live[live['code'] != ''],
        _read_vintage_names(vintage_recipe_id),
        country_id,
    )
    resolved = {code: index.resolve(code) for code in long['code'].unique()}
    long['admin_id'] = [resolved[c][0] for c in long['code']]
    long['tier'] = [resolved[c][1] for c in long['code']]

    dropped = long[long['admin_id'] == '']
    long = long[long['admin_id'] != ''].copy()

    # Which rule each unit and side is already using, so the covering
    # set can keep it wherever keeping it costs nothing. A hand-written
    # override outranks the search, which is the precedence the
    # predecessor applied at read time; otherwise the committed rule
    # wins, and failing both, the search's own first choice.
    long['precedence'] = [
        0
        if origin == 'manual'
        else (1 if (code, kind, pattern, conv) in ported_rules else 2)
        for code, kind, pattern, conv, origin in zip(
            long['code'],
            long['kind'],
            long['pattern'],
            long['conv'],
            long['origin'],
        )
    ]
    long = long.sort_values(['precedence', 'order'], kind='stable')
    long = long.drop_duplicates(['admin_id', 'kind', *RULE_COLUMNS])
    # First row wins: the frame is already sorted by precedence, so a
    # dict comprehension here would silently keep the *last* rule for
    # each pair - the search's worst candidate rather than its best.
    preferred: dict[tuple[str, str], tuple] = {}
    for pair, rule in zip(
        zip(long['admin_id'], long['kind']),
        zip(*[long[column] for column in RULE_COLUMNS]),
    ):
        preferred.setdefault(pair, rule)
    before = long
    long = _one_rule_per_unit(long, preferred)

    long['country_id'] = country_id
    long['admin_id_admin1'] = long['admin_id'].map(index.code_of)
    long['name'] = long['admin_id'].map(index.name_of)
    out = long[OUTPUT_COLUMNS].sort_values(
        ['country_id', 'admin_id_admin1', 'kind'], ignore_index=True
    )

    collisions = out[
        out.duplicated(['country_id', 'admin_id_admin1', 'kind'], keep=False)
    ]
    if len(collisions):
        raise ValueError(
            f'{len(collisions)} rows share a (country, code, kind) key, '
            f'starting with {collisions.iloc[0].to_dict()}. Two rules for one '
            'unit and side is the defect this import exists to remove, so '
            'nothing was written.'
        )

    if verbose:
        library = out[RULE_COLUMNS].drop_duplicates()
        kept = sum(
            preferred.get(pair) == rule
            for pair, rule in zip(
                zip(out['admin_id'], out['kind']),
                zip(*[out[column] for column in RULE_COLUMNS]),
            )
        )
        print(f'{len(out):,} rules over {out["admin_id"].nunique():,} units')
        print(f'  chosen from {len(before):,} candidates')
        print(f'  conversion library: {len(library):,} distinct conversions')
        print(f'  keeping the rule already in use: {kept:,} of {len(out):,}')
        for tier, n in before['tier'].value_counts().items():
            print(f'  resolved by {tier}: {n:,}')
        if len(dropped):
            print(
                f'  dropped, naming no live unit: {len(dropped):,} rows over '
                f'{dropped["code"].nunique():,} codes'
            )
        measured = out['match_rate'].notna().sum()
        print(f'  carrying a measured match rate: {measured:,}')
    return out


def refresh_admin_ids(country_id: str = 'US', verbose: bool = True) -> pd.DataFrame:
    """Re-resolve the committed table's `admin_id` and `name` columns.

    Those two are carried for readability and joined on by nothing: the
    key is `(country_id, admin_id_admin1)`, and `geo.ids` resolves the
    admin id against the live spine every time it loads. So a re-mint
    leaves them stale without changing any behavior, and
    `tests/geo/test_parcel_id_links.py` fails on the drift rather than on
    a defect. This is the one-command answer to that failure.

    It needs no grid-search table, which is the point: refreshing a
    comment column should not depend on a file that is not in the
    repository.

    Parameters
    ----------
    country_id : str, optional
        Country whose scheme issued the codes.
    verbose : bool, optional
        Print how many rows moved and how many keys no longer resolve.

    Returns
    -------
    pandas.DataFrame
        The table with both columns re-resolved. Rows whose key names no
        live unit are left exactly as they were: the code is still the
        key, a later spine may name it again, and the test that every key
        resolves is the right place for that to surface.
    """
    table = pd.read_csv(LINKS_PATH, dtype=str, keep_default_na=False)
    live = _read_live_spine()
    index = _UnitIndex(live[live['code'] != ''], {}, country_id)
    resolved = [index.by_code.get(code) for code in table['admin_id_admin1']]

    moved = unresolved = 0
    admin_ids, names = [], []
    for was, now, name in zip(table['admin_id'], resolved, table['name']):
        if now is None:
            unresolved += 1
            admin_ids.append(was)
            names.append(name)
            continue
        moved += was != now
        admin_ids.append(now)
        names.append(index.name_of.get(now, name))
    table['admin_id'] = admin_ids
    table['name'] = names

    if verbose:
        print(f'{len(table):,} rows: {moved:,} admin ids refreshed')
        if unresolved:
            print(
                f'  {unresolved:,} keys name no live unit and were left '
                'alone; run the full import if that is not expected'
            )
    return table


def main(argv=None) -> None:
    """Import the table and write it in place."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--auto', type=Path, default=None, help='grid-search table (parquet)'
    )
    parser.add_argument(
        '--manual', type=Path, default=None, help='hand-written overrides (csv)'
    )
    parser.add_argument(
        '--vintage-recipe-id',
        default=COUSUB_VINTAGE_RECIPE_ID,
        help='admin layer naming the vintage the search was keyed on',
    )
    parser.add_argument(
        '--preserve-ref',
        default=DEFAULT_PRESERVE_REF,
        help='commit whose committed rule keeps rank 0 for each unit',
    )
    parser.add_argument(
        '--refresh-ids',
        action='store_true',
        help='re-resolve the admin_id/name comment columns and stop '
        '(needs no grid-search table)',
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='report and write nothing'
    )
    args = parser.parse_args(argv)

    if args.refresh_ids:
        table = refresh_admin_ids()
        if args.dry_run:
            print(f'dry run: would write {LINKS_PATH}')
            return
        table.to_csv(LINKS_PATH, index=False)
        print(f'wrote {LINKS_PATH}')
        return

    if args.auto is None:
        parser.error('--auto is required unless --refresh-ids is given')

    table = build(
        args.auto,
        args.manual,
        args.vintage_recipe_id,
        preserve_ref=args.preserve_ref,
    )
    if args.dry_run:
        print(f'dry run: would write {LINKS_PATH}')
        return
    table.to_csv(LINKS_PATH, index=False)
    print(f'wrote {LINKS_PATH} ({LINKS_PATH.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
