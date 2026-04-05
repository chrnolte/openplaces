"""src/openplaces/io/aggregate.py

Functions to aggregate per-process-unit intermediate files
into final save-level output files.
"""

import geopandas as gpd
import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io import delete_data, read_parquet, save_parquet
from openplaces.io.readers import get_admin
from openplaces.recipe import (
    get_output_path,
    get_process_admin_level,
    get_save_admin_level,
)


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


def aggregate_to_admin_level(
    recipe,
    admin_ids_to_save=None,
    admin_ids_to_process=None,
    verbose=False,
):
    """Aggregate per-process-unit intermediate files into save-level files.

    Used when ``save_to: admin_level`` is coarser than ``process_by:
    admin_level``.  Reads the intermediate parquet files written at the
    process level, concatenates them into a single file at the save level,
    and deletes the intermediates (including any now-empty parent directories).

    Parameters
    ----------
    recipe : dict
        Loaded recipe dictionary.  Must have an explicit ``save_to:
        admin_level`` that is coarser than the ``process_by`` /
        ``download_by`` level.
    admin_ids_to_save : list of str, optional
        Admin IDs at the save level to aggregate.  Defaults to all IDs at
        ``save_to.admin_level`` that are children of ``recipe['admin_id']``.
    admin_ids_to_process : list of str, optional
        Admin IDs at the process level whose intermediate files should be
        read.  Defaults to all IDs at the process level under each
        ``admin_id_to_save``.
    verbose : bool
        If True, print a summary line for each aggregated file.
    """
    save_admin_level = get_save_admin_level(recipe)
    process_admin_level = get_process_admin_level(recipe)
    temp_recipe = _strip_save_admin_level(recipe)

    if admin_ids_to_save is None:
        admin_ids_to_save = list(get_admin(recipe['admin_id'], save_admin_level).index)

    if admin_ids_to_process is None:
        admin_ids_to_process = list(
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
            for pid in admin_ids_to_process
            if save_id.is_parent_or_equal_of(AdminId(pid))
        ]

        temp_paths = [get_output_path(temp_recipe, pid) for pid in process_ids_for_save]
        existing = [
            (pid, p) for pid, p in zip(process_ids_for_save, temp_paths) if p.exists()
        ]

        if not existing:
            continue

        existing_temp_paths = [p for _, p in existing]
        final_path = get_output_path(recipe, admin_id_to_save)

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
            if has_geo:
                merged = gpd.GeoDataFrame(pd.concat(dfs).sort_index(), crs=dfs[0].crs)
            else:
                merged = pd.concat(dfs).sort_index()
            save_parquet(merged, final_path)
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

        # Delete intermediates; geo companion first so the attribute file
        # triggers the empty-directory cleanup on its turn.
        for temp_path in existing_temp_paths:
            geo_path = temp_path.with_stem(temp_path.stem + '_geo')
            if geo_path.exists():
                delete_data(geo_path)
            delete_data(temp_path)

        if verbose:
            print(f'Aggregated {len(existing_temp_paths)} chunk(s) → {final_path.name}')
