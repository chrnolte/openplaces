"""A failing batch ingest must leave every unit's attribute file.

The Snakefile's `ingest_<recipe>_batch` rule owns each unit's file.
Snakemake deletes a job's existing outputs before running it, so a
batch that failed after that left the units' `_geo` sidecars without
their attribute files (685 OBM and Microsoft units on 2026-10-27). The
rule flags its outputs, sidecars included, `update()`; these tests pin
the Snakemake behavior that fix relies on, on a three-unit batch that
rewrites its first unit and then fails.
"""

from __future__ import annotations

import subprocess
import sys
import time

UNITS = ('a', 'b', 'c')

SNAKEFILE = """
from pathlib import Path

DATA = Path(config['data'])
UNITS = {units!r}
PAIRS = [str(DATA / f'{{u}}{{s}}.parquet') for u in UNITS for s in ('', '_geo')]

rule all:
    input:
        [str(DATA / f'{{u}}.parquet') for u in UNITS]

rule batch:
    input:
        str(DATA / 'admin.txt')
    output:
        {output}
    shell:
        '{python} {writer}'
"""

WRITER = """
import sys
from pathlib import Path

data = Path(__file__).parent / 'data'
(data / 'a.parquet').write_text('new a')
(data / 'a_geo.parquet').write_text('new a geo')
sys.exit('SSL EOF while downloading the tile for b')
"""


def _run_failing_batch(tmp_path, output: str) -> dict[str, str]:
    data = tmp_path / 'data'
    data.mkdir()
    for u in UNITS:
        (data / f'{u}.parquet').write_text(f'old {u}')
        (data / f'{u}_geo.parquet').write_text(f'old {u} geo')
    # Older pairs than the input, so the batch has to rerun
    time.sleep(1.1)
    (data / 'admin.txt').write_text('newer input')
    writer = tmp_path / 'writer.py'
    writer.write_text(WRITER, encoding='utf-8')
    (tmp_path / 'Snakefile').write_text(
        SNAKEFILE.format(
            units=UNITS,
            output=output,
            python=sys.executable.replace('\\', '/'),
            writer=writer.as_posix(),
        ),
        encoding='utf-8',
    )
    proc = subprocess.run(
        [sys.executable, '-m', 'snakemake', '--cores', '1',
         '--rerun-triggers', 'mtime', '--keep-incomplete', '--rerun-incomplete',
         '--config', f'data={data.as_posix()}'],
        cwd=tmp_path, capture_output=True, text=True,
    )  # fmt: skip
    assert proc.returncode != 0, 'the batch was meant to fail'
    return {p.name: p.read_text() for p in data.glob('*.parquet')}


def test_plain_outputs_of_a_failed_batch_are_deleted(tmp_path):
    """The premise: without the flag, b and c lose their attribute files."""
    files = _run_failing_batch(tmp_path, "[str(DATA / f'{u}.parquet') for u in UNITS]")

    assert 'b.parquet' not in files and 'c.parquet' not in files
    assert files['b_geo.parquet'] == 'old b geo'


def test_update_outputs_of_a_failed_batch_keep_every_pair(tmp_path):
    files = _run_failing_batch(tmp_path, '[update(p) for p in PAIRS]')

    # Unit a is restored too, so its pair matches again
    assert files == {
        f'{u}{s}.parquet': f'old {u}{" geo" if s else ""}'
        for u in UNITS
        for s in ('', '_geo')
    }
