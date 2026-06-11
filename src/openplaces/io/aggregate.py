"""
Functions to aggregate per-process-unit intermediate files
into final save-level output files.
"""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.core.schema import AdminId
from openplaces.io import delete_data, read_parquet, save_parquet
from openplaces.io.readers import get_admin
from openplaces.recipe import (
    get_output_path,
    get_partition_ids,
    get_process_admin_level,
    get_recipe_by_id,
    get_save_admin_level,
)


def join_nonnull_strings(x):
    """Join non-null values of *x* as strings with ' + '; None when all null."""
    parts = [str(v) for v in x if v is not None and pd.notna(v)]
    return ' + '.join(parts) if parts else None


_AGG_ALIASES = {'join_nonnull': join_nonnull_strings}


def _has_default_index(df) -> bool:
    """True if *df* has a default, unnamed RangeIndex (0..n-1, step 1)."""
    idx = df.index
    return (
        isinstance(idx, pd.RangeIndex)
        and idx.start == 0
        and idx.step == 1
        and idx.name is None
    )


def aggregate_rows(
    df: pd.DataFrame,
    by: str | list[str],
    aggregation_function=None,
    sort_by: str | None = None,
    list_columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """Aggregate rows of *df* using per-column functions from the attribute registry.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.  Columns that appear in the attribute registry with a
        non-null aggregation function are included in the result.
    by : str or list of str
        Column(s) to group by.
    aggregation_function : None, callable, or dict, optional
        Controls which aggregation function is applied to each column.

        ``None``
            Use the function recorded in the attribute registry for each column.
        callable
            Apply this single function to all aggregatable columns.
        dict
            Map column names to callables; columns absent from the dict fall
            back to the registry default.
    sort_by : str, optional
        Column to sort *df* by descending before grouping.  When the column is
        absent or omitted and *df* is a GeoDataFrame, rows are sorted by
        geometry area descending.
    list_columns : list of str, optional
        Column names for which an additional ``{col}_list`` column is added to
        the output, collecting all values per group into a Python list.  The
        normal scalar aggregation for each column is still applied; these are
        extra columns alongside the registry-aggregated ones.

    Returns
    -------
    pd.DataFrame or None
        Aggregated DataFrame with *by* as the index, or ``None`` when no
        aggregatable columns are found in *df*.

    Raises
    ------
    ValueError
        When *aggregation_function* is not ``None``, a callable, or a dict.
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

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    elif isinstance(df, gpd.GeoDataFrame):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
            df = df.loc[df.geometry.area.sort_values(ascending=False).index]

    agg_cols: dict = {}
    for col in df.columns:
        fname = get_agg_func(col)
        if fname is None:
            continue
        if callable(aggregation_function):
            agg_cols[col] = aggregation_function
        elif isinstance(aggregation_function, dict):
            agg_cols[col] = aggregation_function.get(
                col, _AGG_ALIASES.get(fname, fname)
            )
        else:
            agg_cols[col] = _AGG_ALIASES.get(fname, fname)

    for col in list_columns or []:
        if col not in df.columns:
            continue
        # Categorical dtype cannot hold list values; cast to object first.
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df = df.copy()
            df[col] = df[col].astype(object)
        agg_cols[f'{col}_list'] = pd.NamedAgg(column=col, aggfunc=list)

    if not agg_cols:
        return None

    return df.groupby(by).agg(agg_cols)


def _strip_save_admin_level(recipe):
    """Return a shallow copy of recipe with save_to.admin_level removed.

    Without an explicit save_to.admin_level, get_output_path resolves to the
    process-level path — the intermediate per-chunk file written by
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
    """
    input_paths = [p for _, p in inputs]
    has_geo = input_paths[0].with_stem(input_paths[0].stem + '_geo').exists()

    try:
        dfs = []
        # 'union' reloads the whole existing file (de-duplicated below); the
        # legacy replace_by path reloads and drops only the rows being replaced
        # so a partial re-run doesn't discard prior chunks.
        if how == 'union' and final_path.exists():
            dfs.append(read_parquet(final_path, geom=has_geo))
        elif replace_by is not None and final_path.exists():
            existing_df = read_parquet(final_path, geom=has_geo)
            replaced_ids = {input_id for input_id, _ in inputs}
            if replace_by in existing_df.columns:
                existing_df = existing_df[~existing_df[replace_by].isin(replaced_ids)]
            elif existing_df.index.name == replace_by:
                existing_df = existing_df[~existing_df.index.isin(replaced_ids)]
            if not existing_df.empty:
                dfs.append(existing_df)

        new_dfs = []
        for input_id, p in inputs:
            df = read_parquet(p, geom=has_geo)
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
            merged = gpd.GeoDataFrame(merged, crs=dfs[0].crs)

        for col, (cats, ordered) in cat_meta.items():
            if col in merged.columns:
                merged[col] = pd.Categorical(
                    merged[col], categories=cats, ordered=ordered
                )
        save_parquet(merged, final_path, combined=combined, file_metadata=file_metadata)
    except PermissionError as e:
        raise PermissionError(
            f'Cannot write to {final_path.name}.\n\n'
            '\033[1m→ Close the file in QGIS / ArcGIS / Dropbox sync '
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
        # geo companion first so the attribute file triggers the empty-directory
        # cleanup on its turn.
        for _input_id, p in inputs:
            geo_path = p.with_stem(p.stem + '_geo')
            if geo_path.exists():
                delete_data(geo_path)
            delete_data(p)

    if verbose:
        print(f'Aggregated {len(input_paths)} chunk(s) → {final_path.name}')


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
        Does not affect where intermediate input files are looked up — those
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
    # recipe's original data_dir — output_dir must not bleed into this.
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

    Wrapper around :func:`aggregate` that reads the save level from the
    recipe's ``save_to.admin_level`` field.

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
        Passed through to :func:`aggregate`.
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


def _legacy_upgrader(recipe):
    """Return a frame transform that applies the recipe's legacy-column upgrade.

    The returned callable renames columns written before a recipe rename
    (`apply_legacy_columns`) and, when the rename feeds derived columns,
    re-runs the recipe transformations, so old aggregated files and new
    partitions share a schema.
    """
    from openplaces.io.transform import apply_legacy_columns, apply_transformations

    def upgrade(df):
        df = apply_legacy_columns(df, recipe)
        if recipe.get('legacy_columns') and recipe.get('transformations'):
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


def read_partition_coverage(path) -> set[str]:
    """Return the partition ids recorded in an aggregated parquet's footer.

    Reads the ``openplaces:partitions`` key from the Parquet file-level
    (footer) metadata written by :func:`aggregate_partitions` — no rows are
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
    suffix, skipping the aggregated 'all' file and '_geo' sidecars.
    """
    base = get_output_path(recipe, admin_id)
    if not base.parent.exists():
        return []
    pids = []
    for p in sorted(base.parent.glob(base.stem + '_*' + base.suffix)):
        pid = p.stem[len(base.stem) + 1 :]
        if pid == 'all' or pid.endswith('_geo'):
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
