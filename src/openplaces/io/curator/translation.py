"""Registered curation step that translates a source's own words into one vocabulary.

Ingest keeps what a county calls a foundation or a frame (`CS`, `CONC
SLAB`, `FULL BSMT`) in a descriptive column such as
`foundation_description`; this step reads that column and writes the
shared class (`foundation_type`, NSI's `found_type` codes). The rules
are data, never code: one shared keyword table per variable beside the
curate recipe, and a per-source override beside a source's own recipe
for a source that uses codes rather than words. Geography lives in
those tables, not here.

A table row is a fixed statement about a word ("CRAWL means a crawl
space"), applied in order, first match wins. Nothing is learned, no
record is linked to another, and there is no fuzzy comparison: a value
that no row names stays missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

from openplaces.io.curator import CurateState, _register
from openplaces.io.curator.indicators import as_matchable_text

#: Method tokens this step writes into a `{output}_source` sidecar
#: beside the source ids: which kind of table decided the value.
KEYWORDS_TOKEN = 'keywords'
OVERRIDE_TOKEN = 'override'

_TRUE = frozenset({'true', '1', 'yes'})


@dataclass(frozen=True)
class TranslationRule:
    """One row of a translation table.

    Parameters
    ----------
    pattern : str
        Text the source value is matched against.
    match_type : str
        `contains` (a literal substring), `regex` (a regular expression
        searched anywhere in the value) or `exact` (the whole trimmed
        value). All three ignore case.
    target : str or None
        The class the row asserts. None is a stop row: the value is
        known to be ambiguous and says nothing.
    reviewed : bool
        Whether a decoder, the source's labels or a measurement states
        the reading.
    column : str or None
        The input column the row is limited to; None applies it to
        every input column.
    """

    pattern: str
    match_type: str
    target: str | None
    reviewed: bool
    column: str | None = None

    def matches(self, value: str) -> bool:
        """Whether *value* (already stripped) is named by this row."""
        if self.match_type == 'exact':
            return value.casefold() == self.pattern.strip().casefold()
        if self.match_type == 'regex':
            return re.search(self.pattern, value, flags=re.IGNORECASE) is not None
        return self.pattern.casefold() in value.casefold()


def _read_commented_csv(path: Path) -> pd.DataFrame:
    """Read a table whose leading `#` lines are a header comment.

    Only whole lines starting with `#` are comments. pandas' `comment`
    option would also cut a pattern at a `#` inside it.
    """
    from io import StringIO

    with open(path, encoding='utf-8') as f:
        lines = [line for line in f if not line.lstrip().startswith('#')]
    return pd.read_csv(StringIO(''.join(lines)), dtype=str, keep_default_na=False)


@cache
def load_translation_table(path: str, target: str) -> tuple[TranslationRule, ...]:
    """Load a shared keyword table or a per-source override.

    Parameters
    ----------
    path : str
        The CSV file.
    target : str
        The column holding the class (`foundation_type`).

    Returns
    -------
    tuple of TranslationRule
        In file order. Cached per process, like the recipe index: a
        session that edits a table must call `cache_clear()`.

    Raises
    ------
    KeyError
        If the table has no *target* or no `pattern` column.
    ValueError
        If a row names a match type other than contains, regex or
        exact, or a regular expression that does not compile.
    """
    table = _read_commented_csv(Path(path))
    for required in ('pattern', target):
        if required not in table.columns:
            raise KeyError(
                f'{Path(path).name} has no {required!r} column; found '
                f'{list(table.columns)}.'
            )
    rules = []
    for row in table.itertuples(index=False):
        values = row._asdict()
        match_type = (values.get('match_type') or 'contains').strip().lower()
        if match_type not in ('contains', 'regex', 'exact'):
            raise ValueError(
                f'{Path(path).name}: unknown match_type {match_type!r} for '
                f'pattern {values["pattern"]!r}.'
            )
        if match_type == 'regex':
            try:
                re.compile(values['pattern'])
            except re.error as exc:
                raise ValueError(
                    f'{Path(path).name}: pattern {values["pattern"]!r} is not '
                    f'a valid regular expression ({exc}).'
                ) from exc
        rules.append(
            TranslationRule(
                pattern=values['pattern'],
                match_type=match_type,
                target=values[target].strip() or None,
                reviewed=values.get('reviewed', '').strip().lower() in _TRUE,
                column=(values.get('column') or '').strip() or None,
            )
        )
    return tuple(rules)


def first_match(
    value: str, rules: tuple[TranslationRule, ...], column: str | None = None
) -> TranslationRule | None:
    """The first rule naming *value* in *column*, or None."""
    value = value.strip()
    if not value:
        return None
    for rule in rules:
        if rule.column is not None and column is not None and rule.column != column:
            continue
        if rule.matches(value):
            return rule
    return None


def split_source_label(label) -> list[str]:
    """The bare source ids in a property row's `source` label.

    `union_spine_sources` labels a row with its source id, adds
    `:units` for units split off a parcel layer, and
    `assign_entity_ids` joins the labels of merged rows with `+`
    (`galvestoncad+txgio:units`). Returns `['galvestoncad',
    'txgio:units']`, in label order.
    """
    if label is None or label is pd.NA or (isinstance(label, float) and pd.isna(label)):
        return []
    return [part for part in str(label).split('+') if part]


def _admin_covers(recipe_admin: str, admin_id: str) -> bool:
    """Whether a recipe scoped to *recipe_admin* covers *admin_id*."""
    return (
        not recipe_admin
        or admin_id == recipe_admin
        or admin_id.startswith(f'{recipe_admin}-')
    )


@cache
def resolve_source_recipe(label: str, admin_id: str) -> str | None:
    """The recipe a spine source label names, for one admin unit.

    A plain label is a property source id; `<id>:units` names the
    parcel recipe whose stacked units were split off at ingest. The
    most specific recipe covering *admin_id* wins, then the newest
    version. Detail tables (`supplements:`) and suffixed sibling
    recipes are not candidates: an override is named after the roll
    or parcel layer the row came from.

    Returns None when no recipe matches (a label from an explicit
    spine source entry, or a recipe removed since the spine was built).
    """
    from openplaces.diagnostics import find_recipes
    from openplaces.io.stacked_units import STACKED_UNITS_LABEL_SUFFIX

    if label.endswith(STACKED_UNITS_LABEL_SUFFIX):
        source_id = label[: -len(STACKED_UNITS_LABEL_SUFFIX)]
        entity_type = 'parcel'
    else:
        source_id, entity_type = label, 'property'
    recipes = find_recipes(entity_type, stage='ingest')
    if recipes.empty:
        return None
    candidates = recipes[
        (recipes['source_id'] == source_id)
        & (recipes['supplements'].fillna('') == '')
        & (recipes['filename_suffix'].fillna('') == '')
    ]
    candidates = candidates[
        candidates['admin_id'].map(lambda a: _admin_covers(str(a), admin_id))
    ]
    if candidates.empty:
        return None
    ranked = candidates.assign(
        _depth=candidates['admin_id'].map(lambda a: len(str(a).split('-')) if a else 0)
    ).sort_values(['_depth', 'version'], ascending=[False, False])
    return str(ranked['recipe_id'].iloc[0])


@cache
def source_override_path(recipe_id: str, suffix: str) -> Path | None:
    """`<recipe_id>_<suffix>.csv` beside *recipe_id*, or None if absent."""
    from openplaces.path import recipe_path
    from openplaces.recipe import get_recipe_by_id

    recipe = get_recipe_by_id(recipe_id)
    directory = recipe_path(recipe['admin_id'], recipe['entity'], as_dir=True)
    path = Path(directory) / f'{recipe_id}_{suffix}.csv'
    return path if path.exists() else None


def _keywords_path(state: CurateState, keywords: str) -> Path:
    """The shared table beside the curate recipe, named by its suffix.

    Resolved like `occupancy.load_ruleset`: `recipe_path` prefixes the
    curate recipe's own id, so `foundation-type-keywords` names
    `US_property-openplaces-2026_foundation-type-keywords.csv`.
    """
    from openplaces.path import recipe_path

    recipe = state.recipe
    filename = keywords if keywords.endswith('.csv') else f'{keywords}.csv'
    path = Path(
        recipe_path(
            recipe['admin_id'],
            recipe.get('entity') or recipe.get('dataset'),
            filename=filename,
        )
    )
    if not path.exists():
        raise FileNotFoundError(f'Translation table not found: {path}')
    return path


def _decide(
    value: str,
    column: str,
    overrides: list[tuple[TranslationRule, ...]],
    shared: tuple[TranslationRule, ...],
    reviewed_only: bool,
) -> tuple[str | None, str | None]:
    """The class one value of one column translates to, and the table kind.

    Override rows of each of the row's sources come first, in label
    order, then the shared table. The first matching row decides:
    a stop row, or an unreviewed row under *reviewed_only*, yields no
    class, and no later row is consulted.
    """
    for rules, kind in [*((o, OVERRIDE_TOKEN) for o in overrides), (shared, None)]:
        rule = first_match(value, rules, column)
        if rule is None:
            continue
        if rule.target is None or (reviewed_only and not rule.reviewed):
            return None, None
        return rule.target, kind or KEYWORDS_TOKEN
    return None, None


@_register('translate_descriptions')
def translate_descriptions(
    state: CurateState,
    output: str,
    columns: list[str],
    keywords: str,
    overrides: str | None = None,
    reviewed_only: bool = True,
    fill_only: bool = True,
    source_column: str = 'source',
) -> CurateState:
    """Translate a source's descriptive columns into one shared class.

    For each row, the input *columns* are tried in order, and for each
    one that holds a value, the per-source override rows are consulted
    before the shared keyword table. The first matching row decides the
    column: a reviewed row with a class writes it; a stop row (empty
    class) or an unreviewed row leaves the column silent, and the next
    input column may still speak. The first column yielding a class
    wins.

    `{output}_source` names the row's source ids and the kind of table
    that decided (`galvestoncad+override`, `fldor+keywords`). The ids
    are what lets a delivery withhold a restricted source's translated
    values; the value is never marked imputed, since it restates the
    source's own word.

    Parameters
    ----------
    output : str
        Column to write (`foundation_type`).
    columns : list of str
        Descriptive input columns, in priority order
        (`[construction_description, construction_class]`). Absent
        columns are skipped; with none present the step does nothing.
    keywords : str
        The shared table beside the curate recipe, by its suffix
        (`foundation-type-keywords`) or file name. Its class column is
        named *output*.
    overrides : str, optional
        Suffix of the per-source overrides (`foundation-type-override`):
        `<source recipe id>_<suffix>.csv` beside each source's recipe,
        found through the row's *source_column* label. Omit to use the
        shared table alone.
    reviewed_only : bool, optional
        Write only classes from rows marked reviewed (default). An
        unreviewed first match leaves the cell missing rather than
        falling through to a later row that may assert another class.
    fill_only : bool, optional
        Keep a value *output* already holds (default).
    source_column : str, optional
        Column holding each row's spine source label (default `source`).
    """
    from openplaces.io.curator.provenance import record_sources

    curated = state.curated
    present = [c for c in columns if c in curated.columns]
    if not present:
        if state.verbose:
            print(f'  translate_descriptions: no input column for {output}')
        return state

    shared = load_translation_table(str(_keywords_path(state, keywords)), output)
    admin_id = str(state.admin_id)
    # A row with no label consults the shared table only; an empty
    # string keeps it a plain dictionary key below.
    labels = (
        curated[source_column].astype('string').fillna('')
        if source_column in curated.columns
        else pd.Series('', index=curated.index, dtype='string')
    )

    def overrides_for(label) -> list[tuple[TranslationRule, ...]]:
        if overrides is None:
            return []
        tables = []
        for part in split_source_label(label):
            recipe_id = resolve_source_recipe(part, admin_id)
            if recipe_id is None:
                continue
            path = source_override_path(recipe_id, overrides)
            if path is not None:
                tables.append(load_translation_table(str(path), output))
        return tables

    override_cache = {label: overrides_for(label) for label in labels.unique()}

    result = pd.Series(pd.NA, index=curated.index, dtype=object)
    kinds = pd.Series(pd.NA, index=curated.index, dtype=object)
    for column in present:
        open_rows = result.isna()
        text = as_matchable_text(curated[column])
        pairs = pd.DataFrame({'label': labels, 'value': text}, index=curated.index).loc[
            open_rows & text.notna()
        ]
        if pairs.empty:
            continue
        # A county has a few hundred distinct words at most, so each
        # (label, value) pair is decided once and mapped back.
        decisions = {
            (label, value): _decide(
                str(value), column, override_cache[label], shared, reviewed_only
            )
            for label, value in pairs.drop_duplicates().itertuples(index=False)
        }
        keys = list(zip(pairs['label'], pairs['value'], strict=True))
        decided = pd.Series([decisions[k] for k in keys], index=pairs.index)
        values = decided.map(lambda d: d[0])
        written = values.notna()
        result.loc[values.index[written]] = values[written]
        kinds.loc[values.index[written]] = decided[written].map(lambda d: d[1])

    translated = result.notna()
    if fill_only and output in curated.columns:
        translated &= curated[output].isna()
    if translated.any():
        if output in curated.columns:
            existing = curated[output].astype(object)
            curated[output] = existing.where(~translated, result)
        else:
            curated[output] = result.where(translated)

        def token(label, kind) -> str:
            ids = [part.split(':', 1)[0] for part in split_source_label(label)]
            return '+'.join([*dict.fromkeys(ids), kind])

        tokens = pd.Series(
            [
                token(label, kind)
                for label, kind in zip(
                    labels[translated], kinds[translated], strict=True
                )
            ],
            index=curated.index[translated],
            dtype=object,
        )
        record_sources(curated, output, tokens.reindex(curated.index))
    elif output not in curated.columns:
        curated[output] = pd.Series(pd.NA, index=curated.index, dtype=object)

    state.curated = curated
    if state.verbose:
        described = curated[present].notna().any(axis=1).sum()
        print(
            f'  translate_descriptions: {output} on {int(translated.sum()):,} '
            f'of {int(described):,} described rows'
        )
    return state
