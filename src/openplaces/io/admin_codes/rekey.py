"""Re-key an admin id column in data files after a re-mint, by national code.

Only the present spine is kept, so a file written before a re-mint cannot
be migrated by its old ids: a re-mint recycles them, and an old id may be
live and name another unit. What a file can be re-keyed on is the code
the unit's own country assigns it, which openplaces never issues and a
re-mint never moves. A US township file carries the Census GEOID beside
its admin4 id, and the present spine carries the same GEOID beside the
present admin4 id; the join between them is the whole repair.

Files are rewritten through pyarrow with their schema metadata intact,
because the harmonizer's fingerprint footers do not survive a round trip
through pandas, and a shipped bundle's read-only bit is put back after.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openplaces.config import cfg
from openplaces.core.constants import STRING_SEPARATOR_WITHIN_IDS
from openplaces.path import spine_path

DEFAULT_ROOTS = ('share', 'core', 'cache')


def code_to_admin_id(level: int, country: str) -> dict[str, str]:
    """Return the present spine's national code to admin id mapping.

    Parameters
    ----------
    level : int
        Admin level of the ids, 2 to 4.
    country : str
        Level-1 id whose units are wanted, e.g. 'US'. A national code is
        unique only within its own country.

    Returns
    -------
    dict of str to str
        Code to the admin id that carries it today. Units without a code
        are absent, so a file row on such a unit is left as it is.
    """
    column = f'admin{level}_id'
    spine = pd.read_csv(spine_path(level), dtype=str, keep_default_na=False)
    own = spine[column].str.startswith(country + STRING_SEPARATOR_WITHIN_IDS)
    coded = own & (spine[f'{column}_admin1'] != '')
    mapping = dict(zip(spine.loc[coded, f'{column}_admin1'], spine.loc[coded, column]))
    if len(mapping) != int(coded.sum()):
        raise ValueError(
            f'{country} level {level}: a national code is carried by more than '
            'one unit, so the re-key would be ambiguous.'
        )
    return mapping


def _rekey_table(table: pa.Table, id_column: str, code_column: str, mapping):
    """Return the table with `id_column` re-keyed, and the cells changed."""
    codes = table.column(code_column).to_pandas().astype('string')
    current = table.column(id_column).to_pandas().astype('string')
    wanted = codes.map(mapping).astype('string')
    # A row whose code the spine does not carry keeps its id: the
    # spine may name that unit again later, and blanking would lose
    # the only key the row has.
    updated = wanted.where(wanted.notna(), current)
    changed = int((updated.fillna('') != current.fillna('')).sum())
    if not changed:
        return table, 0
    new_column = pa.array(updated.astype(object).where(updated.notna(), None))
    index = table.schema.get_field_index(id_column)
    return table.set_column(index, table.schema.field(index).name, new_column), changed


def rekey_files(
    level: int,
    country: str,
    code_column: str,
    roots=DEFAULT_ROOTS,
    apply: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-key one admin id column in every data file that carries its code.

    Parameters
    ----------
    level : int
        Admin level of the id column to re-key, e.g. 4 for `admin4_id`.
    country : str
        Level-1 id whose files and units are meant, e.g. 'US'.
    code_column : str
        Column entity files carry the national code in, e.g.
        'census_subdivision_id' for US level 4. Named by the caller
        because it is a fact about the country's sources, not about the
        spine.
    roots : iterable of str, optional
        Standard directories to scan, under `{root}/{country}`.
    apply : bool, optional
        Write. Default False reports what would change.
    verbose : bool, optional
        Print one line per file that changes.

    Returns
    -------
    pandas.DataFrame
        One row per file scanned that carries both columns: `path`,
        `cells` changed, and `unknown` codes the spine does not carry.
    """
    id_column = f'admin{level}_id'
    mapping = code_to_admin_id(level, country)
    rows = []
    for root in roots:
        base = Path(cfg.get_dir(root)) / country
        if not base.exists():
            continue
        for path in sorted(base.rglob('*.parquet')):
            schema = pq.read_schema(path)
            if id_column not in schema.names or code_column not in schema.names:
                continue
            table = pq.read_table(path)
            codes = table.column(code_column).to_pandas().astype('string')
            unknown = int((~codes.isin(list(mapping)) & codes.notna()).sum())
            table, changed = _rekey_table(table, id_column, code_column, mapping)
            rows.append({'path': str(path), 'cells': changed, 'unknown': unknown})
            if not changed:
                continue
            if verbose:
                print(f'{changed:>9,} cells  {path.relative_to(base.parent)}')
            if apply:
                mode = path.stat().st_mode
                path.chmod(mode | stat.S_IWUSR)
                pq.write_table(table, path)
                path.chmod(mode)
    report = pd.DataFrame(rows, columns=['path', 'cells', 'unknown'])
    if verbose:
        changed = int((report['cells'] > 0).sum())
        print(
            f'{len(report):,} files carry {id_column} and {code_column}; '
            f'{changed:,} change, {int(report["cells"].sum()):,} cells'
            + ('' if apply else ' (dry run)')
        )
    return report
