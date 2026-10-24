"""
Functions to aggregate per-process-unit intermediate files
into final save-level output files.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.core.schema import AdminId
from openplaces.geo.ids import get_geo_ids
from openplaces.io import (
    coerce_mixed_object_columns,
    delete_data,
    parquet_columns,
    read_parquet,
    save_parquet,
)
from openplaces.io.readers import get_admin
from openplaces.recipe import (
    get_output_path,
    get_partition_ids,
    get_process_admin_level,
    get_recipe_by_id,
    get_save_admin_level,
    resolve_attribute_name,
)

# Re-exported: these moved to openplaces.table to break the
# geo/crosswalk <-> io/aggregate import cycle.
from openplaces.table import (  # noqa: F401
    _AGG_ALIASES,
    _agg_func_for,
    add_unique_suffix,
    aggregate_rows,
    join_nonnull_addresses,
    join_nonnull_strings,
)


def _delete_parquet_pair(path) -> None:
    """Delete an attribute parquet and its `_geo` sidecar.

    Uses `delete_data` rather than `io.delete_parquet`, which unlinks
    bare: the deletions here should also clean up an emptied directory
    and report a locked file. The sidecar goes first so the attribute
    file is the one that triggers that cleanup.

    Parameters
    ----------
    path : pathlib.Path
        Attribute parquet path.
    """
    geo_path = path.with_stem(path.stem + '_geo')
    if geo_path.exists():
        delete_data(geo_path)
    delete_data(path)


def _has_geometry(path) -> bool:
    """True if *path* carries geometry, in a `_geo` sidecar or in itself.

    Parameters
    ----------
    path : pathlib.Path
        Attribute parquet path.
    """
    if not path.exists():
        return False
    if path.with_stem(path.stem + '_geo').exists():
        return True
    return 'geometry' in parquet_columns(path)


def _has_default_index(df) -> bool:
    """True if *df* has a default, unnamed RangeIndex (0..n-1, step 1)."""
    idx = df.index
    return (
        isinstance(idx, pd.RangeIndex)
        and idx.start == 0
        and idx.step == 1
        and idx.name is None
    )


def aggregate_rows_weighted(
    df: pd.DataFrame,
    by: str | list[str],
    wcol: str,
    aggregation_function=None,
) -> pd.DataFrame | None:
    """Weighted-aggregate rows of *df* using per-column functions from the registry.

    Weighted counterpart to :func:`aggregate_rows`, for join tables carrying a
    per-row weight (e.g. an area-based crosswalk's ``area_ha`` or
    ``fraction_of_old``). Only the two aggregation kinds that benefit from a
    weight get weighted treatment; all other registry kinds (``'first'``,
    ``'max'``, ``'join_nonnull'``, ...) are delegated to :func:`aggregate_rows`
    unweighted, identical to its own behavior.

    Semantics
    ---------
    ``'mean'`` columns
        Weighted mean per group: ``sum(value * wcol) / sum(wcol * value.notnull())``.
        *wcol* should reflect physical overlap magnitude (e.g. ``area_ha``) -
        weight is normalized within each group implicitly by the division.
    ``'sum'`` columns
        Fraction-weighted (apportioned) sum per group: ``sum(value * wcol)``,
        with **no further normalization**. The caller must pass an
        already-source-normalized weight (e.g. a crosswalk's
        ``fraction_of_old``, which sums to 1.0 per source row) so that
        summing a value across every group sharing a source recovers exactly
        that source's original total. Passing a non-normalized weight (e.g.
        raw ``area_ha``) for a 'sum' column silently over- or under-counts -
        this is a caller responsibility, easy to get backwards. Because
        'mean' and 'sum' columns typically need *different* weight columns,
        call this function once per weight column (on the relevant column
        subset) when a table mixes both kinds.

    Parameters
    ----------
    df : pd.DataFrame
        Input rows, one per (target, source) link - e.g. one row per
        crosswalk (parcel_id_new, parcel_id_old) pair, with value columns to
        aggregate plus *wcol*.
    by : str or list of str
        Column(s) to group by (e.g. ``'parcel_id_new'``).
    wcol : str
        Weight column. See 'Semantics' above for what it must represent.
    aggregation_function : None, callable, or dict, optional
        ``None`` or callable: same semantics as :func:`aggregate_rows`
        (registry-driven default, or one function applied to every
        aggregatable column). A dict, unlike :func:`aggregate_rows`, is not
        limited to *overriding* registry-known columns: a column present as a
        dict key is aggregated with that function whether or not it is in the
        attribute registry, letting one-off columns from an external dataset
        (not meant to join the shared registry) be weight-aggregated too.

    Returns
    -------
    pd.DataFrame or None
        Aggregated DataFrame with *by* as the index, or ``None`` when no
        aggregatable columns are found in *df*.
    """
    if not (
        aggregation_function is None
        or callable(aggregation_function)
        or isinstance(aggregation_function, dict)
    ):
        raise ValueError(
            'aggregation_function must be None, a callable, or a dict; '
            f'got {type(aggregation_function)}'
        )

    by_cols = [by] if isinstance(by, str) else list(by)

    agg_cols: dict = {}
    for col in df.columns:
        if col in by_cols or col == wcol:
            continue
        if isinstance(aggregation_function, dict) and col in aggregation_function:
            # An explicit dict entry fully specifies the column, whether or
            # not it is in the shared attribute registry -- unlike
            # aggregate_rows, this lets one-off columns outside the registry
            # (e.g. a legacy external dataset's raw variable names) be
            # weight-aggregated without adding registry rows for them.
            agg_cols[col] = aggregation_function[col]
            continue
        fname = get_agg_func(resolve_attribute_name(col))
        if fname is None:
            continue
        agg_cols[col] = (
            aggregation_function if callable(aggregation_function) else fname
        )

    if not agg_cols:
        return None

    weighted_mean_cols = [c for c, f in agg_cols.items() if f == 'mean']
    weighted_sum_cols = [c for c, f in agg_cols.items() if f == 'sum']
    # Registry names ('join_nonnull') are resolved to the callables
    # groupby.agg needs. aggregate_rows resolves them on its own path, but
    # not for the entries of a dict it is handed, so an unresolved name
    # reached pandas and raised AttributeError.
    plain_cols = {
        c: (_agg_func_for(resolve_attribute_name(c), f) if isinstance(f, str) else f)
        for c, f in agg_cols.items()
        if f not in ('mean', 'sum')
    }

    group_keys = df[by] if isinstance(by, str) else [df[c] for c in by_cols]
    w = pd.to_numeric(df[wcol], errors='coerce')
    parts = []

    if weighted_mean_cols:
        numer = df[weighted_mean_cols].mul(w, axis=0).groupby(group_keys).sum()
        denom = (
            df[weighted_mean_cols]
            .notna()
            .mul(w, axis=0)
            .groupby(group_keys)
            .sum()
            .replace(0, np.nan)
        )
        parts.append(numer / denom)

    if weighted_sum_cols:
        parts.append(df[weighted_sum_cols].mul(w, axis=0).groupby(group_keys).sum())

    if plain_cols:
        # Grouped here rather than through aggregate_rows, which keeps only
        # columns the attribute registry knows and so silently dropped a
        # dict-declared column that is not registered, the opposite of what
        # this function's dict contract promises.
        parts.append(df.groupby(group_keys)[list(plain_cols)].agg(plain_cols))

    if not parts:
        return None

    return pd.concat(parts, axis=1)


def _strip_save_admin_level(recipe):
    """Return a shallow copy of recipe with save_to.admin_level removed.

    Without an explicit save_to.admin_level, get_output_path resolves to the
    process-level path - the intermediate per-chunk file written by
    TableIngester in aggregate mode.
    """
    temp = dict(recipe)
    if temp.get('save_to') and 'admin_level' in temp['save_to']:
        temp['save_to'] = {
            k: v for k, v in temp['save_to'].items() if k != 'admin_level'
        }
    return temp


def _to_id_list(ids):
    """Normalize a str, AdminId, or list thereof to list[str]."""
    if ids is None:
        return None
    if isinstance(ids, str | AdminId):
        ids = [ids]
    return [str(a) for a in ids]


# pyarrow's default maximum row group length for `pq.write_table`.
_STREAM_ROW_GROUP_ROWS = 1024 * 1024


class _NotStreamable(Exception):
    """Raised when a streamed aggregate cannot establish one schema."""


def _unified_input_schema(inputs, columns):
    """One arrow schema covering every input, or None if unobtainable.

    The streaming writer fixes its schema before it sees data, so that
    schema must already be the widest of the inputs. Taking the first
    input's instead fails on a real panel: Miami-Dade's 2002 vintage
    has columns that are entirely empty, which arrive as arrow type
    `null`, and a later vintage's `large_string` cannot be cast down to
    that. `unify_schemas` promotes the other way, and null casts to
    anything.

    A column that is text in one input and numeric in another has no
    arrow promotion, and is written as text (the second return value
    names those columns). Every Florida roll has such a column: a CSV
    vintage whose `address` cells are all empty is read as float, and
    columns with no empty cell in a vintage come out as integers there.
    Returning None for them sent every county to the in-memory path,
    which is what held all of US-FL-LA's 23 vintages in one frame. The
    in-memory path ends with the same text column: `pd.concat` makes
    the column object, and `coerce_mixed_object_columns` casts a mixed
    one to pandas 'string'; the caller applies that cast input by input,
    which yields the same text (a float 1.0 is '1.0' either way).

    Measured 2026-10-23 on US-FL-SA: 24 vintages, 6,539,606 rows, ingest peak 2.7 GB.

    Returns None when the post-transform columns are not the ones on
    disk (a transform derived or renamed some), since their types
    cannot be read from a footer; the caller then falls back.

    Returns
    -------
    tuple of (pyarrow.Schema, set of str) or None
        The schema, in the order of *columns*, and the columns to cast
        to text before writing.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    def unify(schemas):
        try:
            return pa.unify_schemas(schemas, promote_options='permissive')
        except TypeError:
            # pyarrow < 14 has no promote_options.
            return pa.unify_schemas(schemas)

    def textlike(t):
        return (
            pa.types.is_null(t)
            or pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_integer(t)
            or pa.types.is_floating(t)
            or pa.types.is_boolean(t)
        )

    try:
        schemas = [pq.read_schema(p) for _, p in inputs]
    except Exception:  # noqa: BLE001
        return None
    present = {name for s in schemas for name in s.names}
    if set(columns) != present:
        return None

    fields = []
    text_columns = set()
    for name in columns:
        column_fields = [s.field(name) for s in schemas if name in s.names]
        try:
            fields.append(unify([pa.schema([f]) for f in column_fields]).field(name))
        except (pa.ArrowTypeError, pa.ArrowInvalid, TypeError):
            if not all(textlike(f.type) for f in column_fields):
                return None
            fields.append(pa.field(name, pa.large_string()))
            text_columns.add(name)
    # The first input's footer metadata, as `unify_schemas` keeps it.
    return pa.schema(fields, metadata=schemas[0].metadata), text_columns


def _streamable_columns(inputs, transform):
    """The post-transform column union of *inputs*, or None if unknowable.

    The streaming writer fixes one schema before it sees any data, so it
    needs every column up front. Parquet footers give the columns on
    disk, but a `transform` (the legacy-column upgrader) renames and
    derives columns, so the footer list is the wrong one: taking it
    directly reinstated `consideration_raw` and dropped the `price` the
    upgrader had just derived.

    Applying the transform to a zero-row frame of the footer's columns
    yields the post-transform names without reading any data. A
    transform that cannot survive that returns None, and the caller
    falls back to the in-memory path rather than guess.
    """
    columns: list[str] = []
    for _input_id, p in inputs:
        cols = list(parquet_columns(p))
        if transform is not None:
            try:
                probe = pd.DataFrame({c: pd.Series(dtype='object') for c in cols})
                cols = list(transform(probe).columns)
            except Exception:  # noqa: BLE001
                return None
        for col in cols:
            if col not in columns:
                columns.append(col)
    return columns


def _can_stream_aggregate(*, how, final_path, replace_by, has_geo, reset_index):
    """Whether *final_path* can be written one input at a time.

    Streaming holds one input in memory instead of every input plus the
    existing file, which is what a large panel needs: Miami-Dade's 22
    yearly vintages are 19.1 million rows, and concatenating them beside
    a reloaded copy of themselves needs roughly twice that in one frame.

    It is only possible when nothing has to be compared across the whole
    result: no geometry to give one CRS, no `replace_by` row surgery, no
    meaningful index to sort, and no existing file to de-duplicate
    against.
    """
    if has_geo or replace_by is not None or not reset_index:
        return False
    return how == 'replace' or not final_path.exists()


def _stream_aggregate_to_file(
    final_path,
    inputs,
    *,
    combined=False,
    file_metadata=None,
    transform=None,
    verbose=False,
):
    """Concatenate *inputs* into *final_path* one input at a time.

    Same result as the in-memory path for the cases
    `_can_stream_aggregate` allows, at the memory cost of a single
    input (plus at most one buffered row group of smaller ones) rather
    than all of them.

    Duplicate detection is by 64-bit row hash rather than by holding the
    rows: a hash collision would raise a spurious duplicate error, never
    merge two distinct rows, so the failure direction is safe. Rows are
    written to a temporary file and moved into place only once the whole
    batch has been checked, so a duplicate leaves no partial output.

    Parameters
    ----------
    final_path : pathlib.Path
        Output parquet path.
    inputs : list of (id, pathlib.Path)
        Files to concatenate, each paired with its partition id.
    combined : bool
        Accepted for signature parity with `_aggregate_to_file`; a
        streamed aggregate never carries geometry, so it has no effect.
    file_metadata : dict of str to str, optional
        Footer key-value metadata written to the output.
    transform : callable, optional
        Applied to every frame after reading.
    verbose : bool
        Print a one-line summary.

    Raises
    ------
    ValueError
        If the batch contains full-row duplicates across its inputs.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = _streamable_columns(inputs, transform)
    if columns is None:
        raise _NotStreamable(final_path.name)

    # The schema has to be the *widest* of the inputs, not the first
    # input's. A column that is empty in the earliest vintage arrives
    # as arrow type `null`, and the first attempt at this fixed the
    # schema from the first chunk and then tried to cast a later
    # vintage's `large_string` down to it, which arrow rightly refuses.
    # Unifying the parquet footers promotes null to the real type
    # instead, and a null-to-anything cast is always available.
    unified = _unified_input_schema(inputs, columns)
    if unified is None:
        raise _NotStreamable(final_path.name)
    target_schema, text_columns = unified

    # The recipe-roots stamp every attribute table carries
    # (`io.tables._write_parquet`). The streaming writer bypasses that
    # function, so without this a streamed roll-up was the one
    # attribute table that could not say which recipes built it. An
    # explicit key in *file_metadata* wins, as it does there.
    from openplaces.path import RECIPE_ROOTS_METADATA_KEY, recipe_roots_footer

    file_metadata = {
        RECIPE_ROOTS_METADATA_KEY: recipe_roots_footer(),
        **(file_metadata or {}),
    }

    tmp_path = final_path.with_name(final_path.name + '.building')
    writer = None
    schema = None
    hashes = []
    n_rows = 0
    # Small inputs are buffered into row groups of the size a single
    # `pq.write_table` call writes, which the in-memory path does. One
    # row group per input doubled a small county's file (US-FL-LF:
    # 23 vintages of about 7,000 rows, 12 MB in one group, 23 MB in
    # 23). The buffer is flushed before it would pass one group, so it
    # never holds more than one group or one input; `concat_tables`
    # only chains the buffered tables, it copies nothing.
    pending = []
    pending_rows = 0

    def _flush():
        nonlocal pending_rows
        if pending:
            writer.write_table(
                pa.concat_tables(pending), row_group_size=_STREAM_ROW_GROUP_ROWS
            )
            pending.clear()
            pending_rows = 0

    try:
        for _input_id, p in inputs:
            df = read_parquet(p, geom=False)
            if transform is not None:
                df = transform(df)
            df = df.reset_index(drop=True)
            df = df.reindex(columns=columns)
            df = coerce_mixed_object_columns(df)
            for col in text_columns:
                df[col] = df[col].astype('string')
            hashes.append(pd.util.hash_pandas_object(df, index=False).to_numpy())
            if '_join_id' in df.columns:
                # `read_parquet` drops the inputs' `_join_id`, so the
                # reindex above left it empty. `save_parquet`, which
                # the in-memory path writes through, numbers the rows
                # 1..n; numbering them here, after the duplicate
                # hash, writes the same column.
                df['_join_id'] = np.arange(n_rows + 1, n_rows + len(df) + 1)
            n_rows += len(df)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if not table.schema.equals(target_schema, check_metadata=False):
                table = table.cast(target_schema)
            if writer is None:
                # The footer metadata goes on the schema up front: setting
                # it afterwards would mean reading the finished file back
                # in full, which is the thing this path exists to avoid.
                merged_meta = dict(table.schema.metadata or {})
                merged_meta.update(
                    {
                        (k.encode() if isinstance(k, str) else k): (
                            v.encode() if isinstance(v, str) else v
                        )
                        for k, v in file_metadata.items()
                    }
                )
                schema = target_schema.with_metadata(merged_meta)
                writer = pq.ParquetWriter(tmp_path, schema)
            if pending_rows + len(table) > _STREAM_ROW_GROUP_ROWS:
                _flush()
            pending.append(table.replace_schema_metadata(schema.metadata))
            pending_rows += len(table)
            del df, table
        if writer is None:
            raise ValueError(f'No inputs to aggregate into {final_path.name}')
        _flush()

        all_hashes = np.concatenate(hashes) if hashes else np.empty(0, dtype=np.uint64)
        hashes.clear()
        n_unique = len(np.unique(all_hashes))
        if n_unique != n_rows:
            input_ids = ', '.join(str(input_id) for input_id, _ in inputs)
            raise ValueError(
                f'New data for {final_path.name} contains '
                f'{n_rows - n_unique} duplicate row(s) before aggregation; '
                'refusing to merge. This usually means the same rows were '
                f'saved into several input partitions. Inspect the input '
                f'files ({input_ids}); rows are not shown because they may '
                'contain personal data.'
            )

    except PermissionError as e:
        raise PermissionError(
            f'Cannot write to {final_path.name}.\n\n'
            '\033[1m-> Close the file in QGIS / ArcGIS / Dropbox sync '
            'and re-run.\033[0m'
        ) from e
    except BaseException:
        # A partial temp file is never left behind to be mistaken for
        # a finished aggregate.
        if writer is not None:
            writer.close()
            writer = None
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    # Move into place only now that the whole batch has been checked.
    tmp_path.replace(final_path)

    if verbose:
        print(
            f'Aggregated {len(inputs)} chunk(s) -> {final_path.name} '
            f'({n_rows:,} rows, streamed)'
        )


def _partition_keys(column):
    """The partition id each value of an arrow *column* names, as text.

    A year partition is the id '2021' while its rows carry the year as
    an integer (or as a float, where a missing value forced one), so
    numbers are compared through their integer text. A fractional value
    raises rather than rounding into a neighboring partition.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if pa.types.is_dictionary(column.type):
        column = column.dictionary_decode()
    if pa.types.is_floating(column.type) or pa.types.is_decimal(column.type):
        column = pc.cast(column, pa.int64())
    return pc.cast(column, pa.string())


def _check_partition_column(inputs, partition_column):
    """Raise unless every input's rows all name that input's partition.

    Replacing a partition's rows in the combined file is exact only if
    the column says which partition each row came from; a vintage
    holding a row of another year would otherwise be dropped with the
    wrong partition or survive a re-run beside its replacement.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    for input_id, p in inputs:
        if partition_column not in parquet_columns(p):
            raise ValueError(
                f'{Path(p).name} has no {partition_column!r} column, so its '
                'rows cannot be matched to their partition in the combined file.'
            )
        keys = _partition_keys(
            pq.read_table(p, columns=[partition_column])[partition_column]
        )
        foreign = set(pc.unique(keys).to_pylist()) - {str(input_id)}
        if foreign:
            raise ValueError(
                f'Partition {input_id} ({Path(p).name}) has rows whose '
                f'{partition_column!r} names another partition or is missing '
                f'({len(foreign)} distinct value(s)); refusing to replace '
                'partitions by that column.'
            )


def _split_existing_by_partition(path, partition_column, drop_ids, work_dir):
    """Write the rows of *path* into one file per partition, batch by batch.

    Rows of the partitions in *drop_ids* are not written. Holds one
    record batch at a time, so splitting Miami-Dade's 19.1 million row
    file costs a batch, not the file. Row order inside each partition is
    the file's own.

    Returns
    -------
    dict of str to pathlib.Path
        Partition id to the file holding its kept rows.

    Raises
    ------
    ValueError
        If a row's partition column is missing or names no partition
        id, since such a row could be neither replaced nor kept with
        certainty.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    if partition_column not in pf.schema_arrow.names:
        raise ValueError(
            f'{Path(path).name} has no {partition_column!r} column; its rows '
            'cannot be assigned to partitions.'
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    pieces = {}
    try:
        for batch in pf.iter_batches(batch_size=200_000):
            keys = _partition_keys(batch.column(partition_column))
            if keys.null_count:
                raise ValueError(
                    f'{Path(path).name}: {keys.null_count} row(s) have no '
                    f'{partition_column!r}; refusing to replace partitions.'
                )
            for key in pc.unique(keys).to_pylist():
                if key in drop_ids:
                    continue
                part = batch.filter(pc.equal(keys, key))
                if key not in writers:
                    pieces[key] = work_dir / f'{key}.parquet'
                    writers[key] = pq.ParquetWriter(pieces[key], pf.schema_arrow)
                writers[key].write_batch(part)
    finally:
        for writer in writers.values():
            writer.close()
    return pieces


def _replace_partitions_in_file(
    final_path,
    inputs,
    *,
    partition_column,
    keep_original=False,
    file_metadata=None,
    transform=None,
    verbose=False,
):
    """Replace the rows of some partitions in a combined file, streamed.

    The existing file is split into one file per partition (rows of the
    partitions in *inputs* left out), and those pieces and the new
    inputs are streamed back into *final_path* in partition order. The
    result is the file a run of every partition would have written:
    same rows, same order, the replaced partitions holding only their
    new rows.

    This replaces the two choices `aggregate_partitions` had for a
    partial re-run, both wrong: a full-row `union` kept every old row
    the new vintage changed beside its new version, and `replace`
    dropped every other partition. Selecting rows by the partition
    column is what makes the replacement exact, which is why it is
    only offered when the recipe names that column. The split costs a
    second write of the file on disk, which buys holding one batch in
    memory rather than the file: the in-memory union of Miami-Dade's
    19.1 million rows held them beside a reloaded copy.

    Parameters
    ----------
    final_path : pathlib.Path
        The combined file. Must exist.
    inputs : list of (str, pathlib.Path)
        New partition files, each paired with its partition id.
    partition_column : str
        Column whose value names each row's partition.
    keep_original : bool
        If False, delete the new partition files afterward.
    file_metadata : dict of str to str, optional
        Footer metadata of the result (the merged coverage).
    transform : callable, optional
        Applied to every frame after reading, as in `_aggregate_to_file`.
    verbose : bool
        Print a one-line summary.

    Raises
    ------
    ValueError
        If a row of the existing file or of an input does not name its
        partition, or the combined rows contain full-row duplicates.
        The existing file is left as it was.
    """
    import shutil

    # Rows are split and rewritten as attribute tables; a `_geo` sidecar
    # would no longer line up with them.
    if any(_has_geometry(p) for p in [final_path, *(p for _, p in inputs)]):
        raise ValueError(
            f'{final_path.name}: replacing partitions by column is only '
            'supported for tables without geometry.'
        )
    _check_partition_column(inputs, partition_column)
    new_ids = {str(pid) for pid, _ in inputs}
    work_dir = final_path.with_name(final_path.name + '.split')
    if work_dir.exists():
        # A previous run killed mid-split; its pieces are incomplete.
        shutil.rmtree(work_dir)
    try:
        pieces = _split_existing_by_partition(
            final_path, partition_column, new_ids, work_dir
        )
        combined = dict(pieces)
        combined.update({str(pid): p for pid, p in inputs})
        # Partition ids are fixed-width (years, year-months), so text
        # order is time order and matches the order a full run reads
        # them in.
        ordered = [(pid, combined[pid]) for pid in sorted(combined)]
        _aggregate_to_file(
            final_path,
            ordered,
            how='replace',
            keep_original=True,
            reset_index=True,
            file_metadata=file_metadata,
            transform=transform,
            verbose=False,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if not keep_original:
        for _pid, p in inputs:
            _delete_parquet_pair(p)
    if verbose:
        print(
            f'Replaced partition(s) {", ".join(sorted(new_ids))} in '
            f'{final_path.name} ({len(pieces)} kept from the existing file)'
        )


def _aggregate_to_file(
    final_path,
    inputs,
    *,
    replace_by=None,
    how=None,
    combined=False,
    keep_original=False,
    reset_index=False,
    file_metadata=None,
    transform=None,
    verbose=False,
):
    """Concatenate intermediate parquet *inputs* into *final_path*.

    Shared merge core for admin-level and partition-level aggregation: reads the
    inputs (geo-aware), unions categorical category sets, concatenates, writes,
    and deletes the originals.

    Parameters
    ----------
    final_path : pathlib.Path
        Output parquet path.
    inputs : list of (id, pathlib.Path)
        Intermediate files to concatenate, each paired with the identifier of
        the chunk it represents (admin id, or partition id).
    replace_by : str, optional
        Column keyed on the input ids. When given and *final_path* exists, rows
        whose `replace_by` value matches an incoming id are dropped from the
        existing file before concatenation (a partial re-run replaces rather
        than duplicates); inputs missing the column are stamped with their id.
    how : {'union', 'replace'} or None
        Row-level merge policy for partition rollups (mutually exclusive with
        `replace_by`). ``'union'`` reloads the whole existing *final_path* and
        de-duplicates the combined rows (`drop_duplicates`), so re-adding an
        already-present partition is a no-op. ``'replace'`` overwrites
        *final_path* with only the current inputs. Both first verify the new
        batch has no internal full-row duplicates (a source/processing bug),
        raising `ValueError` if it does. Row identity includes the index
        whenever any frame carries a meaningful index (anything other than a
        default unnamed RangeIndex), so rows with equal values but distinct
        keys are not duplicates. ``None`` keeps the legacy behavior (concat
        inputs; honour `replace_by`).
    combined : bool
        Passed to `save_parquet` (single geoparquet vs. split layout).
    keep_original : bool
        If False, delete the input parquets (and `_geo` sidecars) afterward.
    reset_index : bool
        If True, drop the index after concatenation (row-union of records with
        no meaningful index, e.g. partition rollups); otherwise `sort_index`
        (entity tables keyed by geo_id/parcel_id).
    file_metadata : dict of str to str, optional
        Footer key-value metadata forwarded to `save_parquet` (e.g. the
        ``openplaces:partitions`` coverage list).
    transform : callable, optional
        Applied to every frame after reading (both the reloaded existing file
        and the new inputs), before duplicate checks and concatenation. Used
        by partition rollups to upgrade legacy columns so old and new files
        share a schema.
    verbose : bool
        Print a one-line summary.

    Raises
    ------
    ValueError
        If *final_path* is also one of *inputs*, which would rewrite an
        input from itself and then delete it.
    """
    input_paths = [p for _, p in inputs]
    # An output that is also one of its own inputs would be rewritten
    # from itself and then deleted by the keep_original loop below,
    # destroying the data it was meant to roll up. This happens when a
    # roll-up group key equals a partition id (rolling a year-partitioned
    # recipe up by year), where the roll-up is a no-op anyway. Refuse
    # rather than write the group under another name: a renamed output
    # is a file no reader looks for, and the caller's request was
    # meaningless, not merely misaddressed.
    if any(Path(p).resolve() == Path(final_path).resolve() for p in input_paths):
        raise ValueError(
            f'Refusing to aggregate {final_path.name} into itself: it is both '
            'an input and the output of this roll-up, so writing it would '
            'destroy the partition it was built from. This usually means the '
            'roll-up granularity matches the partition granularity (e.g. '
            "aggregate_partitions(by='year') on a year-partitioned recipe), "
            'which has nothing to combine.'
        )

    # Geometry lives either in a `_geo` sidecar (split layout) or in
    # the file itself (combined layout, `save_to: combined: true`).
    # Checking only for the sidecar silently read combined inputs
    # without their geometry, writing a geometry-less aggregate.
    # Decided per input: a chunk written without geometry beside chunks
    # that carry it used to make the whole aggregate geometry-less (when
    # the bare chunk came first) or raise (when it came later).
    input_has_geo = {p: _has_geometry(p) for p in input_paths}
    has_geo = any(input_has_geo.values())

    # A large panel rolled up in one frame is the memory ceiling of the
    # whole ingest: Miami-Dade's 22 vintages are 19.1 million rows, and
    # the in-memory path below holds them beside a reloaded copy of the
    # existing file. Stream instead wherever nothing has to be compared
    # across the whole result.
    if _can_stream_aggregate(
        how=how,
        final_path=final_path,
        replace_by=replace_by,
        has_geo=has_geo,
        reset_index=reset_index,
    ):
        try:
            _stream_aggregate_to_file(
                final_path,
                inputs,
                combined=combined,
                file_metadata=file_metadata,
                transform=transform,
                verbose=verbose,
            )
        except _NotStreamable:
            # Fall through to the in-memory path, which needs no schema
            # up front. Correctness first: a batch whose columns cannot
            # be known before reading is rare, and being slow on it
            # beats writing it wrong.
            pass
        else:
            if not keep_original:
                for _input_id, p in inputs:
                    _delete_parquet_pair(p)
            return

    try:
        dfs = []
        # 'union' reloads the whole existing file (de-duplicated below); the
        # legacy replace_by path reloads and drops only the rows being replaced
        # so a partial re-run doesn't discard prior chunks.
        if how == 'union' and final_path.exists():
            dfs.append(read_parquet(final_path, geom=_has_geometry(final_path)))
        elif replace_by is not None and final_path.exists():
            existing_df = read_parquet(final_path, geom=_has_geometry(final_path))
            replaced_ids = {input_id for input_id, _ in inputs}
            if replace_by in existing_df.columns:
                existing_df = existing_df[~existing_df[replace_by].isin(replaced_ids)]
            elif existing_df.index.name == replace_by:
                existing_df = existing_df[~existing_df.index.isin(replaced_ids)]
            if not existing_df.empty:
                dfs.append(existing_df)

        new_dfs = []
        for input_id, p in inputs:
            df = read_parquet(p, geom=input_has_geo[p])
            if (
                replace_by is not None
                and replace_by not in df.columns
                and df.index.name != replace_by
            ):
                df[replace_by] = input_id
            new_dfs.append(df)

        if transform is not None:
            dfs = [transform(df) for df in dfs]
            new_dfs = [transform(df) for df in new_dfs]

        # The freshly aggregated batch must not contain internal full-row
        # duplicates; those signal a source/processing bug rather than the
        # legitimate overlap that 'union' de-duplicates against the existing file.
        # Row identity includes the index when any frame carries a meaningful
        # one (e.g. entity tables keyed by geo_id), so equal-valued rows with
        # distinct keys are not duplicates.
        include_index = not reset_index and any(
            not _has_default_index(df) for df in [*dfs, *new_dfs]
        )
        if how is not None:
            new_df = pd.concat(new_dfs)
            new_df = (
                new_df.reset_index(drop=True) if reset_index else new_df.sort_index()
            )
            checked = new_df.reset_index() if include_index else new_df
            dup_mask = checked.duplicated(keep=False).to_numpy()
            if dup_mask.any():
                n_dups = int(checked.duplicated().sum())
                input_ids = ', '.join(str(input_id) for input_id, _ in inputs)
                # Do not print the rows: they may contain personal data
                # (e.g. transaction grantor/grantee names).
                raise ValueError(
                    f'New data for {final_path.name} contains {n_dups} duplicate '
                    'row(s) before aggregation; refusing to merge. This usually '
                    'means the same rows were saved into several input '
                    f'partitions. Inspect the input files ({input_ids}); rows '
                    'are not shown because they may contain personal data.'
                )
            dfs.append(new_df)
        else:
            dfs.extend(new_dfs)

        # Union category sets across chunks so pd.concat preserves them.
        cat_meta: dict[str, tuple[list, bool]] = {}
        for df in dfs:
            for col in df.columns:
                if not isinstance(df[col].dtype, pd.CategoricalDtype):
                    continue
                cats = list(df[col].cat.categories)
                ordered = df[col].cat.ordered
                if col not in cat_meta:
                    cat_meta[col] = (cats, ordered)
                else:
                    prev_cats, prev_ordered = cat_meta[col]
                    prev_set = set(prev_cats)
                    extra = [c for c in cats if c not in prev_set]
                    cat_meta[col] = (prev_cats + extra, prev_ordered)

        merged = pd.concat(dfs)
        if how == 'union':
            # Drop rows in the new batch that already exist in the reloaded file
            # (legitimate overlap); the new batch was already checked for
            # internal duplicates above.
            if include_index:
                index_names = merged.index.names
                flat = merged.reset_index()
                index_cols = list(flat.columns[: merged.index.nlevels])
                merged = flat.drop_duplicates().set_index(index_cols)
                merged.index.names = index_names
            else:
                merged = merged.drop_duplicates()
        merged = merged.reset_index(drop=True) if reset_index else merged.sort_index()
        if has_geo:
            # The CRS comes from whichever frame carried geometry: with a
            # mixed batch the first frame may have none.
            crs = next(
                (df.crs for df in dfs if getattr(df, 'crs', None) is not None), None
            )
            if 'geometry' in merged.columns:
                merged['geometry'] = gpd.GeoSeries(merged['geometry'], crs=crs)
            merged = gpd.GeoDataFrame(merged, crs=crs)

        for col, (cats, ordered) in cat_meta.items():
            if col in merged.columns:
                merged[col] = pd.Categorical(
                    merged[col], categories=cats, ordered=ordered
                )
        merged = coerce_mixed_object_columns(merged)
        save_parquet(merged, final_path, combined=combined, file_metadata=file_metadata)
    except PermissionError as e:
        raise PermissionError(
            f'Cannot write to {final_path.name}.\n\n'
            '\033[1m-> Close the file in QGIS / ArcGIS / Dropbox sync '
            'and re-run.\033[0m'
        ) from e
    finally:
        # Release DataFrames explicitly so pyarrow closes memory-mapped file
        # handles before we try to delete the temp files (Windows).
        dfs.clear()
        try:
            del merged
        except NameError:
            pass

    if not keep_original:
        for _input_id, p in inputs:
            _delete_parquet_pair(p)

    if verbose:
        print(f'Aggregated {len(input_paths)} chunk(s) -> {final_path.name}')


def aggregate_files(
    recipe,
    admin_level,
    output_dir=None,
    admin_ids_to_save=None,
    admin_ids_to_aggregate=None,
    keep_original=False,
    combined=False,
    verbose=False,
):
    """Aggregate per-process-unit intermediate files into save-level files.

    Used when the desired output level is coarser than the level at which
    files were written (``process_by.admin_level``).  Reads the intermediate
    parquet files, concatenates them into one file per save-level unit, and
    deletes the originals (unless *keep_original* is True).

    Parameters
    ----------
    recipe : str or dict
        Recipe ID string (e.g. ``'US_footprint-spine-2026'``) or a
        pre-loaded recipe dict.
    admin_level : int
        Target admin level for output files (e.g. ``2`` for state-level).
        Required explicitly because the recipe's own ``save_to.admin_level``
        may differ from the intended aggregation target.
    output_dir : str, optional
        Directory for the aggregated output files (e.g. ``'share'``).
        Does not affect where intermediate input files are looked up - those
        are always resolved from the recipe's original ``save_to.data_dir``.
        Uses the recipe default if omitted.
    admin_ids_to_save : str, AdminId, or list, optional
        Save-level admin ID(s) for which to write output files.  Accepts a
        single value or a list.  Defaults to all admin IDs at *admin_level*
        that are children of ``recipe['admin_id']``.
    admin_ids_to_aggregate : str, AdminId, or list, optional
        Process-level admin ID(s) whose intermediate files should be
        included as input.  Accepts a single value or a list.  Defaults to
        all process-level children of each *admin_ids_to_save* entry.
    keep_original : bool
        If True, do not delete the intermediate files after aggregation.
    combined : bool
        If True, write the aggregated output as a single geoparquet file
        (attributes and geometry together) rather than the default split
        layout of an attribute table plus a ``_geo`` sidecar.  Passed
        through to :func:`save_parquet`.
    verbose : bool
        If True, print a summary line for each aggregated file.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    # temp_recipe resolves intermediate (process-level) paths using the
    # recipe's original data_dir - output_dir must not bleed into this.
    process_admin_level = get_process_admin_level(recipe)
    temp_recipe = _strip_save_admin_level(recipe)

    # output_recipe determines where the aggregated file is written.
    output_recipe = dict(recipe)
    output_recipe['save_to'] = {
        **recipe.get('save_to', {}),
        'admin_level': admin_level,
    }
    if output_dir is not None:
        output_recipe['save_to']['data_dir'] = output_dir

    admin_ids_to_save = _to_id_list(admin_ids_to_save)
    admin_ids_to_aggregate = _to_id_list(admin_ids_to_aggregate)

    if admin_ids_to_save is None:
        admin_ids_to_save = list(
            dict.fromkeys(
                str(aid) for aid in get_admin(recipe['admin_id'], admin_level).index
            )
        )

    if admin_ids_to_aggregate is None:
        admin_ids_to_aggregate = list(
            dict.fromkeys(
                str(aid)
                for save_id in admin_ids_to_save
                for aid in get_admin(AdminId(save_id), process_admin_level).index
            )
        )

    for admin_id_to_save in admin_ids_to_save:
        save_id = AdminId(admin_id_to_save)

        # Process-level IDs that belong to this save-level unit.
        process_ids_for_save = [
            pid
            for pid in admin_ids_to_aggregate
            if save_id.is_parent_or_equal_of(AdminId(pid))
        ]

        temp_paths = [get_output_path(temp_recipe, pid) for pid in process_ids_for_save]
        existing = [
            (pid, p) for pid, p in zip(process_ids_for_save, temp_paths) if p.exists()
        ]

        if not existing:
            continue

        final_path = get_output_path(output_recipe, admin_id_to_save)
        _aggregate_to_file(
            final_path,
            existing,
            replace_by=f'admin{process_admin_level}_id',
            combined=combined,
            keep_original=keep_original,
            verbose=verbose,
        )


def aggregate_to_admin_level(
    recipe,
    admin_ids_to_process=None,
    keep_intermediates=False,
    combined=False,
    verbose=False,
):
    """Aggregate per-process-unit intermediate files into save-level files.

    Wrapper around :func:`aggregate_files` that reads the save level from
    the recipe's ``save_to.admin_level`` field.

    Parameters
    ----------
    recipe : dict
        Loaded recipe dictionary.  Must have an explicit ``save_to:
        admin_level`` that is coarser than the ``process_by`` /
        ``download_by`` level.
    admin_ids_to_process : list of str, optional
        Admin IDs at the process level whose intermediate files should be
        read.  Defaults to all IDs at the process level under the recipe's
        save-level admin IDs.
    keep_intermediates : bool
        If True, do not delete the intermediate files after aggregation.
    combined : bool
        If True, write the aggregated output as a single geoparquet file.
        Passed through to :func:`aggregate_files`.
    verbose : bool
        If True, print a summary line for each aggregated file.
    """
    aggregate_files(
        recipe,
        admin_level=get_save_admin_level(recipe),
        admin_ids_to_aggregate=admin_ids_to_process,
        keep_original=keep_intermediates,
        combined=combined,
        verbose=verbose,
    )


def join_partitions_by_index(
    recipe,
    table_names,
    admin_ids=None,
    join_key_name=None,
    keep_original=False,
    sidecar_tables=None,
    verbose=False,
) -> list[str]:
    """Left-join per-table partition outputs into one entity file per admin unit.

    Column-wise counterpart to :func:`aggregate_partitions` (which only
    row-concatenates): for a recipe ingested with ``download_by: {partition:
    table, table_names: [...]}``, each table is saved as its own partition
    file (``get_output_path(recipe, admin_id, partition_id=table_name)``).
    This reads them back and merges them into the single, un-partitioned
    entity file (``get_output_path(recipe, admin_id)``) that the rest of the
    codebase expects to read.

    The first entry in `table_names` that is not a sidecar table is the
    base table (read with geometry); every later one is a plain attribute
    table (its own `geometry` column, if any, is dropped) left-joined onto
    the base by their shared index. Every table is expected to already
    carry the same join key as its index (e.g. via a recipe's
    `create_index` opting out of `TableIngester`'s automatic geometry-hash
    `geo_id` indexing; see `US_parcel-placeslab-fmv2026.yaml` for the
    reference example). Since that opt-out means none of the partitions
    carry openplaces's own `geo_id`/`parcel_id`, this computes it fresh on
    the joined result, replicating `TableIngester.process`'s own logic.

    A unit without a base-table partition is skipped (not yet published),
    unless one of its sidecar tables is on disk. That sidecar stands in
    for geometry the source withholds (the placeslab fmv source delivers
    a parcel crosswalk instead of its restricted boundaries), so the
    attribute tables are joined without geometry and indexed by the join
    key itself, as a string `parcel_id`; such a file has no `geo_id`.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID string or loaded recipe dict. Must be `entity_type:
        parcel` (or another entity type `geo.ids.get_geo_ids` supports) with
        geometry on its base table.
    table_names : list of str
        Partition ids to join, in join order; the first that is not a
        sidecar table is the base table.
    admin_ids : str, AdminId, or list, optional
        Admin ID(s) to join. Defaults to every admin unit that has a saved
        base-table or sidecar-table partition file.
    join_key_name : str, optional
        If given, the shared index the tables were joined on is kept as a
        column under this name in the output (for provenance, and as the
        key a sidecar crosswalk names reference parcels by). If omitted,
        the original join index is dropped once the real
        `geo_id`/`parcel_id` index is assigned.
    keep_original : bool
        If False (default), delete the joined per-table partition files
        after a successful join. Sidecar partitions are never deleted.
    sidecar_tables : list of str, optional
        Entries of `table_names` that are ingested as partitions but never
        joined: their rows do not describe one entity each (a crosswalk is
        many-to-many), so they stay on disk as their own partition files
        for a later stage to read.
    verbose : bool
        If True, print a summary line per joined admin unit, and a note for
        any admin unit skipped because its base-table partition is missing
        (treated as "not yet published", matching how the Ingester itself
        tolerates a missing partition).

    Returns
    -------
    list of str
        The admin units an entity file was written for.
    """

    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    sidecar_tables = [str(t) for t in sidecar_tables or []]
    unknown = sorted(set(sidecar_tables) - {str(t) for t in table_names})
    if unknown:
        raise ValueError(
            f'sidecar_tables {unknown} are not among table_names {list(table_names)}.'
        )
    joinable = [t for t in table_names if str(t) not in sidecar_tables]
    if not joinable:
        raise ValueError('join_partitions_by_index needs one table to join.')
    base_table, *attribute_tables = joinable

    def _exists(admin_id, table_name):
        return get_output_path(recipe, admin_id, partition_id=table_name).exists()

    if admin_ids is None:
        save_level = get_save_admin_level(recipe)
        admin_ids = [
            str(a)
            for a in get_admin(recipe['admin_id'], save_level).index
            if any(_exists(a, t) for t in [base_table, *sidecar_tables])
        ]
    else:
        admin_ids = _to_id_list(admin_ids)

    written = []
    for admin_id in admin_ids:
        base_path = get_output_path(recipe, admin_id, partition_id=base_table)
        if base_path.exists():
            joined = read_parquet(base_path, geom=True)
            partition_paths = [base_path]
            tables_to_join = attribute_tables
        elif any(_exists(admin_id, t) for t in sidecar_tables):
            present = [t for t in attribute_tables if _exists(admin_id, t)]
            if not present:
                if verbose:
                    print(f'{admin_id}: no attribute partition beside its sidecar')
                continue
            if verbose:
                print(
                    f'{admin_id}: no {base_table!r} partition; joining '
                    f'{present} without geometry beside its sidecar'
                )
            first_path = get_output_path(recipe, admin_id, partition_id=present[0])
            joined = read_parquet(first_path, geom=False)
            joined = joined.drop(columns='geometry', errors='ignore')
            partition_paths = [first_path]
            tables_to_join = present[1:]
        else:
            if verbose:
                print(f'{admin_id}: no {base_table!r} partition, skipping')
            continue

        for table_name in tables_to_join:
            table_path = get_output_path(recipe, admin_id, partition_id=table_name)
            if not table_path.exists():
                if verbose:
                    print(f'{admin_id}: no {table_name!r} partition, skipping it')
                continue
            table = read_parquet(table_path, geom=False)
            table = table.drop(columns='geometry', errors='ignore')
            overlap = set(table.columns) & set(joined.columns)
            if overlap:
                raise ValueError(
                    f'{table_name!r} shares column name(s) {sorted(overlap)} with '
                    'data already joined; join_partitions_by_index assumes joined '
                    'tables have disjoint column names.'
                )
            joined = joined.join(table, how='left')
            partition_paths.append(table_path)

        if join_key_name:
            joined = joined.rename_axis(join_key_name).reset_index()
        if isinstance(joined, gpd.GeoDataFrame):
            joined['geo_id'] = get_geo_ids(joined, handle_duplicates=False)
            joined.index = pd.Index(
                add_unique_suffix(joined['geo_id']), name='parcel_id'
            )
        else:
            # No geometry to hash: the source's own key is the only id.
            key = joined[join_key_name] if join_key_name else joined.index
            joined.index = pd.Index(key.astype(str).to_numpy(), name='parcel_id')

        output_path = get_output_path(recipe, admin_id)
        save_parquet(joined, output_path)
        written.append(str(admin_id))
        if verbose:
            print(f'{admin_id}: joined {len(joined):,} rows -> {output_path.name}')

        if not keep_original:
            for p in partition_paths:
                _delete_parquet_pair(p)
    return written


def _legacy_upgrader(recipe):
    """Return a frame transform that applies the recipe's legacy-column upgrade.

    The returned callable renames columns written before a recipe rename
    (`apply_legacy_columns`) and, when the rename feeds derived columns,
    re-runs the recipe transformations, so old aggregated files and new
    partitions share a schema.
    """
    from openplaces.io.transform import apply_legacy_columns, apply_transformations

    def upgrade(df):
        # Only a frame that actually carries a legacy column name needs the
        # transformations re-run: a freshly ingested partition already had
        # them applied, and a non-idempotent step (a prefix, a concat) would
        # apply a second time on every roll-up.
        renames = recipe.get('legacy_columns') or {}
        is_legacy = any(old in df.columns for old in renames)
        df = apply_legacy_columns(df, recipe)
        if is_legacy and recipe.get('transformations'):
            df = apply_transformations(df, recipe, silent=True)
        return df

    return upgrade


def _partition_group_key(partition_id: str, by: str) -> str:
    """Map a partition id to its roll-up group key (e.g. '202106' -> '2021')."""
    if by == 'year':
        return str(partition_id)[:4]
    raise ValueError(
        f"Unsupported aggregate_by partition granularity: {by!r} (expected 'year')."
    )


# Footer-coverage sentinel: the file covers all admin units / partitions.
# Written by the enricher's coverage tracking and honored by lifecycle
# completeness checks.
COVERAGE_ALL = '__all__'


def read_partition_coverage(path) -> set[str]:
    """Return the partition ids recorded in an aggregated parquet's footer.

    Reads the ``openplaces:partitions`` key from the Parquet file-level
    (footer) metadata written by :func:`aggregate_partitions` - no rows are
    scanned. Returns an empty set when the file is missing or carries no such
    key (e.g. files written before this metadata was introduced).

    Parameters
    ----------
    path : str or pathlib.Path
        Aggregated parquet path.

    Returns
    -------
    set of str
        Partition ids (e.g. year-months) contained in the file.
    """
    raw = read_file_metadata(path).get('openplaces:partitions')
    if raw is None:
        return set()
    return set(json.loads(raw))


def read_file_metadata(path) -> dict[str, str]:
    """Return a parquet file's footer key-value metadata as a string dict.

    Reads only the footer (no rows are scanned). Returns an empty dict when
    the file is missing or carries no key-value metadata. Pyarrow-internal
    keys (e.g. the pandas schema) are included as-is.

    Parameters
    ----------
    path : str or pathlib.Path
        Parquet file path.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.exists():
        return {}
    meta = pq.read_metadata(path).metadata or {}
    return {k.decode(): v.decode(errors='replace') for k, v in meta.items()}


def _existing_partition_ids(recipe, admin_id) -> list[str]:
    """Discover partition IDs from existing per-partition output files.

    Fallback for recipes without a declared partition range (no
    'download_by': 'partition'), e.g. scraped checkpoint partitions: globs the
    files next to the recipe's bare output path and extracts the partition
    suffix.

    Skipped: geometry sidecars, and any sibling that records its own
    partition coverage in its footer, which marks it as a previous
    roll-up's output rather than a partition. A sibling that is neither,
    but shares the stem for some other reason (a differently suffixed
    product of another stage), is still indistinguishable from a
    partition by name alone.
    """
    base = get_output_path(recipe, admin_id)
    if not base.parent.exists():
        return []
    pids = []
    for p in sorted(base.parent.glob(base.stem + '_*' + base.suffix)):
        pid = p.stem[len(base.stem) + 1 :]
        if pid == 'all' or pid.endswith(('_geo', '_geo_simplified')):
            continue
        # Merging a previous roll-up back in would duplicate every row it
        # already holds, and deleting it as an original would discard it.
        if read_partition_coverage(p):
            continue
        pids.append(pid)
    return pids


def aggregate_partitions(
    recipe,
    by='year',
    single_file=False,
    how='union',
    admin_ids=None,
    partition_ids=None,
    keep_original=False,
    combined=False,
    partition_column=None,
    verbose=False,
):
    """Roll up per-partition output files into a coarser-grained file.

    For a recipe partitioned by ``year_month``: when *single_file* is True,
    concatenate the monthly output files into one dataset-wide
    ``..._all.parquet`` (the non-redundant default); otherwise write one file
    per group (``by='year'`` -> ``..._2021.parquet``). The aggregated file's
    footer records the partition ids it contains (read via
    :func:`read_partition_coverage`) so re-runs can skip already-ingested
    partitions without keeping the per-partition files. Reuses
    :func:`_aggregate_to_file`.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID string or loaded recipe dict.
    by : str
        Roll-up granularity for the partition dimension when *single_file* is
        False. Currently ``'year'``.
    single_file : bool
        If True, write only one file combining every partition, named with the
        ``'all'`` partition suffix (not the bare path, which the Ingester
        reserves for its "already-ingested" check), and no per-group files.
    how : {'union', 'replace'}
        Merge policy when the aggregated file already exists (named after
        ``geopandas.overlay``). ``'union'`` (default) integrates the new
        partitions, de-duplicating by full row against the existing file.
        ``'replace'`` overwrites the file with only the partitions in this run
        (use with a full-range reprocess, or it narrows the file to those
        partitions). Both raise if the new batch has internal duplicate rows.
    admin_ids : str, AdminId, or list, optional
        Save-level admin ID(s) to aggregate. Defaults to all save-level IDs
        under ``recipe['admin_id']``.
    partition_ids : list, optional
        Partition IDs to consider. Defaults to all of the recipe's partitions;
        only those whose per-partition output exists are aggregated. For
        recipes without a declared partition range (e.g. scraped checkpoint
        partitions), the default falls back to the partition files found next
        to each admin unit's output path.
    keep_original : bool
        If False (default), delete the per-partition (monthly) output parquets
        after the roll-up. The downloaded source files are untouched, so a
        re-run rebuilds them. Deletion happens last, after the file is written.
    combined : bool
        Passed to `save_parquet` (single geoparquet vs. split layout).
    partition_column : str, optional
        Column whose value names the partition a row came from (the
        Florida rolls' `tax_year`, whose partitions are years). Given,
        a 'union' into an existing file replaces the rows of the
        partitions in this run and keeps every other partition's, one
        partition in memory at a time, so a re-run of some years is
        exact. Every row must name a partition id, or it raises.
        Without it, 'union' de-duplicates by full row in memory.
    verbose : bool
        Print a summary line per written file.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    save_level = get_save_admin_level(recipe)
    admin_ids = _to_id_list(admin_ids)
    if admin_ids is None:
        if save_level == 0:
            admin_ids = [None]
        else:
            admin_ids = list(
                dict.fromkeys(
                    str(a) for a in get_admin(recipe['admin_id'], save_level).index
                )
            )

    if partition_ids is None:
        partition_ids = [str(p) for p in get_partition_ids(recipe) if p is not None]
        discover_partitions = not partition_ids
    else:
        partition_ids = [str(p) for p in partition_ids]
        discover_partitions = False

    upgrade_legacy = _legacy_upgrader(recipe)

    def _write_group(out_partition_id, group_inputs):
        out_path = get_output_path(recipe, admin_id, partition_id=out_partition_id)
        existing = read_partition_coverage(out_path) if how == 'union' else set()
        coverage = sorted(existing | {pid for pid, _ in group_inputs})
        if partition_column and how == 'union' and out_path.exists():
            _replace_partitions_in_file(
                out_path,
                group_inputs,
                partition_column=partition_column,
                keep_original=keep_original,
                file_metadata={'openplaces:partitions': json.dumps(coverage)},
                transform=upgrade_legacy,
                verbose=verbose,
            )
            return
        _aggregate_to_file(
            out_path,
            group_inputs,
            how=how,
            combined=combined,
            keep_original=keep_original,
            reset_index=True,
            file_metadata={'openplaces:partitions': json.dumps(coverage)},
            transform=upgrade_legacy,
            verbose=verbose,
        )

    for admin_id in admin_ids:
        admin_partition_ids = (
            _existing_partition_ids(recipe, admin_id)
            if discover_partitions
            else partition_ids
        )
        month_inputs = [
            (pid, get_output_path(recipe, admin_id, partition_id=pid))
            for pid in admin_partition_ids
        ]
        month_inputs = [(pid, p) for pid, p in month_inputs if p.exists()]
        if not month_inputs:
            continue

        if single_file:
            # One non-redundant dataset-wide file; no per-group files.
            _write_group('all', month_inputs)
        else:
            groups: dict[str, list] = {}
            for pid, p in month_inputs:
                groups.setdefault(_partition_group_key(pid, by), []).append((pid, p))
            for key, group_inputs in groups.items():
                _write_group(key, group_inputs)
