"""
Functions to aggregate per-process-unit intermediate files
into final save-level output files.
"""

import warnings

import geopandas as gpd
import pandas as pd

from openplaces.core.attribute_registry import get_agg_func
from openplaces.core.schema import AdminId
from openplaces.io import delete_data, read_parquet, save_parquet
from openplaces.io.readers import get_admin
from openplaces.recipe import (
    get_output_path,
    get_process_admin_level,
    get_recipe_by_id,
    get_save_admin_level,
)


def join_nonnull_strings(x):
    """Join non-null values of *x* as strings with ' + '; None when all null."""
    parts = [str(v) for v in x if v is not None and pd.notna(v)]
    return ' + '.join(parts) if parts else None


_AGG_ALIASES = {'join_nonnull': join_nonnull_strings}


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
        Recipe ID string (e.g. ``'US_footprint-cheer-2026'``) or a
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

        existing_temp_paths = [p for _, p in existing]
        final_path = get_output_path(output_recipe, admin_id_to_save)

        # Detect whether a geo companion exists for the first temp file.
        has_geo = (
            existing_temp_paths[0]
            .with_stem(existing_temp_paths[0].stem + '_geo')
            .exists()
        )

        process_col = f'admin{process_admin_level}_id'

        try:
            dfs = []
            # Reload existing county parquet, dropping rows that are being
            # replaced so a partial re-run doesn't discard prior towns.
            if final_path.exists():
                existing_df = read_parquet(final_path, geom=has_geo)
                replaced_ids = {pid for pid, _ in existing}
                if process_col in existing_df.columns:
                    existing_df = existing_df[
                        ~existing_df[process_col].isin(replaced_ids)
                    ]
                elif existing_df.index.name == process_col:
                    existing_df = existing_df[~existing_df.index.isin(replaced_ids)]
                if not existing_df.empty:
                    dfs.append(existing_df)
            for process_admin_id, p in existing:
                df = read_parquet(p, geom=has_geo)
                if process_col not in df.columns and df.index.name != process_col:
                    df[process_col] = process_admin_id
                dfs.append(df)

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

            if has_geo:
                merged = gpd.GeoDataFrame(pd.concat(dfs).sort_index(), crs=dfs[0].crs)
            else:
                merged = pd.concat(dfs).sort_index()

            for col, (cats, ordered) in cat_meta.items():
                if col in merged.columns:
                    merged[col] = pd.Categorical(
                        merged[col], categories=cats, ordered=ordered
                    )
            save_parquet(merged, final_path, combined=combined)
        except PermissionError as e:
            raise PermissionError(
                f'Cannot write to {final_path.name}.\n\n'
                '\033[1m→ Close the file in QGIS / ArcGIS / Dropbox sync '
                'and re-run.\033[0m'
            ) from e
        finally:
            # Release DataFrames explicitly so pyarrow closes memory-mapped
            # file handles before we try to delete the temp files (Windows).
            dfs.clear()
            try:
                del merged
            except NameError:
                pass

        # Delete originals; geo companion first so the attribute file
        # triggers the empty-directory cleanup on its turn.
        if not keep_original:
            for temp_path in existing_temp_paths:
                geo_path = temp_path.with_stem(temp_path.stem + '_geo')
                if geo_path.exists():
                    delete_data(geo_path)
                delete_data(temp_path)

        if verbose:
            print(f'Aggregated {len(existing_temp_paths)} chunk(s) → {final_path.name}')


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
