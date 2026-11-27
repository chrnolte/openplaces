"""Print the run record a table's footer carries.

A harmonize, enrich, curate or ingest output records how the stage ran
(`io.readers.run_record`): the recipe, its digest, the commit, and the
rows its main table held after each step, rows read first. This prints
that record, one line per step, and marks each step that changed the
count. Reads the footer only, never a row.

Usage::

    python -m openplaces.flow.run_record <path.parquet> [<path> ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def format_record(record: dict) -> str:
    """The run record as a header and a table of steps and rows.

    Parameters
    ----------
    record : dict
        A record as `io.readers.run_record` returns it.

    Returns
    -------
    str
        Lines for printing.
    """
    lines = [
        f'stage          {record.get("stage")}',
        f'recipe         {record.get("recipe_id")}',
        f'unit           {record.get("admin_id")}',
        f'version        {record.get("version")}',
        f'commit         {record.get("commit")}',
        f'recipe digest  {record.get("recipe_digest")}',
        '',
        f'{"step":<52} {"rows":>12}  change',
    ]
    last = None
    for step, rows in record.get('steps') or []:
        shown = '' if rows is None else f'{rows:,}'
        change = ''
        if rows is not None and last is not None and rows != last:
            change = f'{rows - last:+,}'
        if rows is not None:
            last = rows
        lines.append(f'{str(step):<52} {shown:>12}  {change}')
    return '\n'.join(lines)


def main(argv=None) -> int:
    """Print the run record of each file given; 1 if any has none."""
    from openplaces.io.readers import run_record

    parser = argparse.ArgumentParser(
        prog='python -m openplaces.flow.run_record',
        description='Print the run record in a parquet footer.',
    )
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args(argv)
    status = 0
    for path in args.paths:
        if len(args.paths) > 1:
            print(f'== {path}')
        record = run_record(path)
        if record is None:
            print(f'{path.name}: no run record in the footer')
            status = 1
            continue
        print(format_record(record))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
