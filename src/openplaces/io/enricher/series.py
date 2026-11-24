"""Series enrichment: extract a variable only where a request asks.

A series is a variable a source gives per period (land cover by year).
The wide mode (`expand_year_steps`) extracts every year for every
entity, one column per year. A recipe that sets `series: true` instead
writes a long table, one row per entity and period and one column per
variable, and only for the (entity, period) pairs its `requests` name:
a sale asks for the period before it, a prediction for the latest one.

Why this shape, rather than the wide table with fewer columns:

- The uses need few cells.
  Measured 2026-11-24 on CO: linked transactions ask for 2,894 (parcel,
  period) pairs; with every parcel at 'now' the requests are 0.036 of
  parcels times periods (411 municipalities, 4.4 million parcels).
- A new period adds rows, never columns, so no registry row and no
  recipe line per year.
- The rows are the record of what is done. A rerun reads the requests
  again, computes only the pairs the file lacks and appends them, so a
  new sale costs its own pair and nothing else.

Why the steps are not changed: each period a missing pair names runs
the existing step (`zonal_stats`, `vicinity_coverage`, ...) once, on
the unit's whole spine, exactly the frame the wide mode samples, and
keeps the values of the pairs asked for. Not on the requested entities
alone, because a value then depends on which other entities share the
run: the rasterized back end gives a pixel under two overlapping
cadastral polygons to one of them, and a frame without the other gives
it to the first.
Measured 2026-11-24 on CO-AN-CV: sampled on its own entities, a rerun
of 56 pairs differed from the wide table in 97 of 392 cells; sampled
on the whole spine, in none. On the whole spine a pair's
value is the same whatever else was requested, which is what lets the
stored rows stand as the record of done. The cost stays small: the
spine is burned onto a grid once per unit and reused by every period
(`EnrichState.grid_cache`), and a further period is one pass over the
window's pixels, nearly independent of how many entities it holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openplaces.io import read_parquet, save_parquet
from openplaces.io.aggregate import COVERAGE_ALL, read_partition_coverage
from openplaces.io.readers import get_entities
from openplaces.io.tables import (
    PERIOD_RULES,
    SERIES_METADATA_KEY,
    answering_periods,
    request_years,
)
from openplaces.recipe import get_output_path, get_recipe_by_id, get_recipe_id

#: Keys of a series step that the series loop reads and the step never
#: receives: the step's period rule and the length of its periods.
SERIES_CONTROL_KEYS = ('period_rule', 'period_length_years')

#: The value of `date` that asks for the latest period.
NOW = 'now'


class SeriesRecipeError(ValueError):
    """A series enrich recipe that cannot run as written."""


@dataclass(frozen=True)
class SeriesLayer:
    """One series step of a recipe: a variable and the periods it has.

    Parameters
    ----------
    index : int
        Position of the step in the recipe's pipeline.
    step : dict
        The step's entry without `years` and the series control keys.
    column : str
        The variable's column in the long table.
    periods : tuple of int
        The periods the step can extract, labeled by their end year.
    rule : str
        The period rule of `io.tables.PERIOD_RULES`.
    period_length_years : int
        Years one period spans (1 for an annual product).
    """

    index: int
    step: dict
    column: str
    periods: tuple
    rule: str
    period_length_years: int


@dataclass(frozen=True)
class RequestSource:
    """One entry of a recipe's `requests` block.

    Parameters
    ----------
    recipe_id : str or None
        Recipe whose output supplies the requests; None for the spine.
    date : str
        A date or year column of that output, or 'now'.
    offsets_years : tuple of int
        Years added to each date before the period rule applies.
    entity_key : str or None
        The enriched entity's id column in the link table; the spine's
        index name when None.
    """

    recipe_id: str | None
    date: str
    offsets_years: tuple = (0,)
    entity_key: str | None = None

    @property
    def label(self) -> str:
        source = self.recipe_id or 'spine'
        offsets = ','.join(str(o) for o in self.offsets_years)
        return f'{source}:{self.date}:{offsets}'


def is_series_recipe(recipe) -> bool:
    """Whether an enrich recipe writes a long series table."""
    return bool(recipe.get('series'))


def series_layers(recipe) -> list[SeriesLayer]:
    """The series steps of a recipe, validated.

    Raises
    ------
    SeriesRecipeError
        For an enabled step without `years` (a constant layer belongs
        in a wide recipe), a column that is missing, carries `{year}`
        or repeats, an unknown period rule, or steps whose periods span
        different lengths (one row holds one period).
    """
    layers = []
    for index, cfg in enumerate(recipe.get('pipeline') or []):
        if cfg.get('enabled', True) is False:
            continue
        name = cfg.get('step')
        if 'years' not in cfg:
            raise SeriesRecipeError(
                f'Step {index} ({name}) has no years; a series recipe holds '
                'series steps only. Keep a constant layer in a wide recipe.'
            )
        column = cfg.get('column')
        if not column or '{year}' in column:
            raise SeriesRecipeError(
                f'Step {index} ({name}) needs a column without {{year}}: the '
                'period is a row of the long table, not part of the name.'
            )
        rule = cfg.get('period_rule', 'before')
        if rule not in PERIOD_RULES:
            raise SeriesRecipeError(
                f'Step {index} ({name}): period_rule {rule!r} is not one of '
                f'{PERIOD_RULES}.'
            )
        first, last = cfg['years']
        step = {
            k: v for k, v in cfg.items() if k not in ('years', *SERIES_CONTROL_KEYS)
        }
        layers.append(
            SeriesLayer(
                index=index,
                step=step,
                column=column,
                periods=tuple(range(int(first), int(last) + 1)),
                rule=rule,
                period_length_years=int(cfg.get('period_length_years', 1)),
            )
        )
    if not layers:
        raise SeriesRecipeError('A series recipe needs at least one series step.')
    columns = [layer.column for layer in layers]
    if len(set(columns)) != len(columns):
        raise SeriesRecipeError(f'Series columns repeat: {columns}.')
    if len({layer.period_length_years for layer in layers}) > 1:
        raise SeriesRecipeError(
            'Series steps of one recipe must share period_length_years: a row '
            'is one period, with one period_start.'
        )
    return layers


def request_sources(recipe) -> list[RequestSource] | str:
    """The recipe's `requests` block, validated.

    Returns
    -------
    list of RequestSource, or 'all'
        'all' asks for every entity at every period.

    Raises
    ------
    SeriesRecipeError
        Without a `requests` block (a series recipe refuses to run
        rather than extract everything by default; maintainer,
        2026-11-15, "ok for now"), or for a malformed entry.
    """
    raw = recipe.get('requests')
    if raw is None:
        raise SeriesRecipeError(
            'A series recipe refuses to run without a requests block. Name '
            "where its requests come from, or write 'requests: all' to "
            'extract every entity at every period.'
        )
    if raw == 'all':
        return 'all'
    if not isinstance(raw, list) or not raw:
        raise SeriesRecipeError("requests is a list of sources or 'all'.")
    sources = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise SeriesRecipeError(f'A request source is a mapping: {entry!r}.')
        recipe_id = entry.get('recipe_id')
        from_spine = entry.get('from') == 'spine'
        if (recipe_id is None) == (not from_spine):
            raise SeriesRecipeError(
                "A request source names exactly one of recipe_id and 'from: "
                f"spine': {entry!r}."
            )
        date = entry.get('date')
        if not isinstance(date, str):
            raise SeriesRecipeError(f'A request source needs a date: {entry!r}.')
        offsets = entry.get('offsets_years', [0])
        if not isinstance(offsets, list) or not all(
            isinstance(o, int) for o in offsets
        ):
            raise SeriesRecipeError(f'offsets_years is a list of integers: {entry!r}.')
        if date == NOW and list(offsets) != [0]:
            raise SeriesRecipeError("date: now takes no offsets: 'now' has no year.")
        sources.append(
            RequestSource(
                recipe_id=recipe_id,
                date=date,
                offsets_years=tuple(offsets),
                entity_key=entry.get('entity_key'),
            )
        )
    return sources


def _request_rows(source: RequestSource, enricher, admin_id, spine) -> pd.DataFrame:
    """One source's requests as (entity, year), before the offsets."""
    entity_key = source.entity_key or spine.index.name
    if source.recipe_id is None:
        ids = spine.index
        dates = spine[source.date] if source.date != NOW else None
        frame = pd.DataFrame({'entity': ids})
        frame['year'] = (
            request_years(dates).to_numpy() if dates is not None else float('nan')
        )
        return frame

    from openplaces.geo.link import get_entity_link_path, get_link_owner_recipe_id

    request_recipe = get_recipe_by_id(source.recipe_id)
    columns = [source.date] if source.date != NOW else None
    table = get_entities(request_recipe, admin_id, columns=columns)
    if source.date != NOW and source.date not in table.columns:
        raise SeriesRecipeError(
            f'{source.recipe_id} has no column {source.date!r} for its requests.'
        )
    request_key = table.index.name
    table = table.reset_index()
    if table.empty:
        return pd.DataFrame({'entity': pd.Series(dtype='string'), 'year': []})
    request_type = str(request_recipe['entity'].entity_type)
    entity_type = str(enricher.entity_recipe['entity'].entity_type)
    if request_type == entity_type:
        entity_ids = table[request_key]
        dates = table[source.date] if columns else None
    else:
        # Another entity's requests reach this one through the link
        # table its harmonize wrote (a sale names its parcels), one row
        # per pair, so a deed of three parcels asks for three.
        link_path = get_entity_link_path(
            get_link_owner_recipe_id(request_recipe),
            get_recipe_id(enricher.entity_recipe),
            admin_id,
        )
        links = pd.read_parquet(link_path)
        if request_key not in links.columns and links.index.name == request_key:
            links = links.reset_index()
        if entity_key not in links.columns and links.index.name == entity_key:
            links = links.reset_index()
        linked = links[[request_key, entity_key]].merge(
            table[[request_key, *(columns or [])]], on=request_key, how='inner'
        )
        entity_ids = linked[entity_key]
        dates = linked[source.date] if columns else None
    frame = pd.DataFrame({'entity': entity_ids.to_numpy()})
    frame['year'] = (
        request_years(dates).to_numpy() if dates is not None else float('nan')
    )
    return frame


def collect_requests(enricher, admin_id, spine, sources) -> tuple[pd.DataFrame, list]:
    """Every request of the unit as (entity, year, now), offsets applied.

    Returns
    -------
    requests : DataFrame
        Columns `entity`, `year` (float, NaN for 'now' and undated) and
        `now` (bool). Entities outside the spine are left out.
    summary : list of dict
        Per source: its label and its request rows, for the footer.
    """
    frames = []
    summary = []
    for source in sources:
        rows = _request_rows(source, enricher, admin_id, spine)
        now = source.date == NOW
        for offset in source.offsets_years:
            shifted = rows.assign(year=rows['year'] + offset, now=now)
            frames.append(shifted)
        summary.append({'source': source.label, 'rows': int(len(rows))})
    if not frames:
        return pd.DataFrame({'entity': [], 'year': [], 'now': []}), summary
    requests = pd.concat(frames, ignore_index=True)
    requests = requests[requests['entity'].isin(spine.index)]
    return requests, summary


def requested_pairs(requests: pd.DataFrame, layers, verbose=False) -> pd.DataFrame:
    """The distinct (entity, period) pairs the requests ask for.

    Each layer applies its own rule to the dated requests and 'now' to
    the undated 'now' requests. A request no layer answers (no date, or
    a date before every period) asks for no row. Two requests that one
    period answers cost one row.
    """
    pieces = []
    answered = pd.Series(False, index=requests.index)
    for layer in layers:
        periods = pd.Series(pd.NA, index=requests.index, dtype='Int64')
        now = requests['now'].astype(bool)
        if now.any():
            periods[now] = answering_periods(
                requests.loc[now, 'year'], layer.periods, 'now'
            )
        dated = ~now
        if dated.any():
            periods[dated] = answering_periods(
                requests.loc[dated, 'year'], layer.periods, layer.rule
            )
        hit = periods.notna()
        answered |= hit
        pieces.append(
            pd.DataFrame(
                {'entity': requests.loc[hit, 'entity'], 'period': periods[hit]}
            )
        )
    if verbose:
        print(
            f'  series: {len(requests):,} requests, '
            f'{int((~answered).sum()):,} answered by no period'
        )
    if not pieces:
        return pd.DataFrame({'entity': [], 'period': []})
    pairs = pd.concat(pieces, ignore_index=True).drop_duplicates()
    pairs['period'] = pairs['period'].astype('int64')
    return pairs.reset_index(drop=True)


def all_pairs(spine, layers) -> pd.DataFrame:
    """Every entity at every period any layer has ('requests: all')."""
    periods = sorted({p for layer in layers for p in layer.periods})
    index = pd.MultiIndex.from_product(
        [spine.index, periods], names=['entity', 'period']
    )
    return index.to_frame(index=False)


def _existing(out_path, reprocess: bool):
    """The stored long table, or None; refuses a wide one."""
    if reprocess or not out_path.exists():
        return None
    if SERIES_METADATA_KEY.encode() not in (pq.read_metadata(out_path).metadata or {}):
        raise SeriesRecipeError(
            f'{out_path} holds no series table (a wide enrichment written '
            'before the recipe became a series?). Rerun with reprocess to '
            'replace it.'
        )
    return read_parquet(out_path)


def _call(fn, layer: SeriesLayer, period: int, state):
    """Run one series step for one period on *state*."""
    params = {
        k: v.replace('{year}', str(period)) if isinstance(v, str) else v
        for k, v in layer.step.items()
        if k not in ('step', 'enabled')
    }
    params['year'] = period
    return fn(state, **params)


def extract_pairs(enricher, state, missing: pd.DataFrame, layers) -> pd.DataFrame:
    """The long table of the *missing* pairs, one column per layer."""
    from openplaces.io.enricher import lookup_step

    spine = state.spine
    entity_name = spine.index.name or 'entity'
    index = pd.MultiIndex.from_frame(
        missing[['entity', 'period']], names=[entity_name, 'period']
    )
    by_period = {
        period: group['entity'].to_numpy()
        for period, group in missing.groupby('period', sort=True)
    }
    # Row positions of each period's pairs in the table, found once.
    positions = {
        period: index.get_indexer(
            pd.MultiIndex.from_arrays(
                [entities, np.full(len(entities), period)], names=index.names
            )
        )
        for period, entities in by_period.items()
    }
    columns = {}
    for layer in layers:
        fn = lookup_step(layer.step['step'])
        values = np.full(len(index), np.nan)
        available = set(layer.periods)
        periods = [p for p in by_period if p in available]
        for period in periods:
            # The whole spine, never the requested entities alone (see
            # the module docstring): a value must not depend on what
            # else was requested.
            state.evidence = pd.DataFrame(index=spine.index)
            state = _call(fn, layer, period, state)
            column = pd.to_numeric(state.evidence[layer.column], errors='coerce')
            values[positions[period]] = column.reindex(by_period[period]).to_numpy(
                dtype='float64', na_value=np.nan
            )
        columns[layer.column] = values
        if state.timer:
            state.timer.mark(
                f'step {layer.index:02d} {layer.step["step"]} ({len(periods)} periods)'
            )
    table = pd.DataFrame(columns, index=index)
    length = layers[0].period_length_years
    if length > 1:
        starts = table.index.get_level_values('period') - length + 1
        table.insert(0, 'period_start', starts.astype('int64'))
    return table


def run_series(enricher, state, sub_admin_ids, reprocess: bool) -> None:
    """Enrich one unit in series mode: compute missing pairs, append.

    Parameters
    ----------
    enricher : Enricher
        The running enricher (its recipe, entity recipe and verbosity).
    state : EnrichState
        The unit's state, its spine loaded and restricted.
    sub_admin_ids : list of str or None
        Sub-units the run is restricted to.
    reprocess : bool
        Recompute every requested pair and replace the stored table.
    """
    from openplaces.io.ingester.empty_outputs import empty_table_metadata

    recipe = enricher.recipe
    layers = series_layers(recipe)
    sources = request_sources(recipe)
    spine = state.spine
    admin_id = state.admin_id
    if sources == 'all':
        pairs = all_pairs(spine, layers)
        summary = [{'source': 'all', 'rows': int(len(pairs))}]
        # Everything is the costly default the requests block exists to
        # avoid, so the run says what it asked for.
        print(f'[series] {admin_id}: requests: all, {len(pairs):,} rows')
    else:
        requests, summary = collect_requests(enricher, admin_id, spine, sources)
        pairs = requested_pairs(requests, layers, verbose=enricher.verbose)

    out_path = get_output_path(
        recipe, admin_id, entity_recipe_id=enricher.entity_recipe
    )
    existing = _existing(out_path, reprocess)
    entity_name = spine.index.name or 'entity'
    missing = pairs
    if existing is not None and len(existing):
        done = pd.MultiIndex.from_frame(existing.index.to_frame(index=False))
        asked = pd.MultiIndex.from_frame(pairs[['entity', 'period']])
        missing = pairs[~asked.isin(done)]
    if enricher.verbose:
        print(f'  series: {len(pairs):,} pairs requested, {len(missing):,} missing')
    if existing is not None and missing.empty:
        # Nothing to add: the stored rows already answer every request.
        return

    if missing.empty:
        new = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=[entity_name, 'period']),
            columns=[layer.column for layer in layers],
            dtype='float64',
        )
    else:
        new = extract_pairs(enricher, state, missing, layers)
    table = new if existing is None else pd.concat([existing, new])
    table = table.sort_index()

    footer = {
        SERIES_METADATA_KEY: json.dumps(
            {
                'requests': summary,
                'periods': sorted(
                    int(p) for p in table.index.get_level_values('period').unique()
                ),
                'layers': {
                    layer.column: {
                        'periods': [layer.periods[0], layer.periods[-1]],
                        'rule': layer.rule,
                        'period_length_years': layer.period_length_years,
                    }
                    for layer in layers
                },
            }
        ),
        'openplaces:partitions': json.dumps(sorted(_coverage(out_path, sub_admin_ids))),
    }
    if table.empty:
        # An empty saved table says why it is empty (readers.empty_reason).
        footer.update(
            empty_table_metadata(
                'nothing_requested',
                f'{len(summary)} request source(s) named no pair the series '
                f'answers in {admin_id}.',
            )
        )
    save_parquet(table, out_path, file_metadata=footer)
    if enricher.verbose:
        print(f'  series: saved {len(table):,} rows ({len(new):,} new)')


def _coverage(out_path, sub_admin_ids) -> list[str]:
    """The footer coverage after this run, as the wide mode keeps it."""
    if sub_admin_ids is None:
        return [COVERAGE_ALL]
    if not out_path.exists():
        return list(sub_admin_ids)
    coverage = read_partition_coverage(out_path)
    if not coverage or COVERAGE_ALL in coverage:
        return [COVERAGE_ALL]
    return sorted(set(coverage) | set(sub_admin_ids))


__all__ = [
    'NOW',
    'SERIES_CONTROL_KEYS',
    'RequestSource',
    'SeriesLayer',
    'SeriesRecipeError',
    'all_pairs',
    'collect_requests',
    'extract_pairs',
    'is_series_recipe',
    'request_sources',
    'requested_pairs',
    'run_series',
    'series_layers',
]
