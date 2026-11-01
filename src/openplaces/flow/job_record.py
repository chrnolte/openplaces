"""Skip a job whose inputs and code are unchanged since its last run.

Snakemake reruns a job whenever an input file is newer than its output,
and a rebuilt spine rewrites its file even when the columns a consumer
reads came out the same. A harmonize, enrich or curate job run by the
Snakefile therefore records, after it succeeds, what it read and wrote
(`write_record`): the commit the checkout was on, the content digest of
every column it read from every file (`io.tables.start_read_log`), and
the digests of its outputs. The next time Snakemake starts it, the job
exits without running (`can_skip`) when all of these hold:

- the run does not force it (`--forcerun`, `--forceall`, `--force`);
- its outputs still hold what it wrote;
- every file it read, and every input the graph declares for it, still
  holds the same values in the columns it read;
- no file its code depends on changed between the recorded commit and
  the working tree (`code_staleness.job_changes`, which also honors a
  change cleared by a sample rerun).

A read that bypasses `io.tables` is seen at the pyarrow call and counts
as reading every column; a declared input never seen read counts the
same way, and a file that is not parquet is compared by its bytes. Each
of these errs toward running the job.

The Snakefile passes each job its declared inputs and outputs, and
whether the run forces it, through one JSON file per admin unit in the
logs folder (`write_job_io`), written by the main process while it
plans: a job on a cluster node parses the Snakefile again without the
run's force settings. Ingests are never skipped this way.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# The stages a job record covers. An ingest reads raw sources outside
# io.tables; delivery and validation jobs have no recipe stage.
RECORD_STAGES = ('harmonize', 'enrich', 'curate')

GLOBAL_UNIT = 'global'
FILE = '__file__'


def _logs_dir() -> Path:
    from openplaces.config import cfg

    return Path(cfg.get_dir('logs'))


def job_io_path(admin_id) -> Path:
    """The JSON file holding the declared files of one unit's jobs."""
    return _logs_dir() / 'job_io' / f'{admin_id or GLOBAL_UNIT}.json'


def record_path(rule: str) -> Path:
    """Where the record of a rule's last successful run is kept."""
    return _logs_dir() / 'job_records' / f'{rule}.json'


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.tmp-{os.getpid()}')
    temp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding='utf-8')
    os.replace(temp, path)


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def write_job_io(jobs) -> list[Path]:
    """Write the declared inputs and outputs of a run's jobs.

    Merged into each unit's file, so two runs whose graphs share a unit
    keep each other's rules.

    Parameters
    ----------
    jobs : iterable of dict
        Keys: admin_id, rule, inputs, outputs (lists of paths) and
        forced (bool).

    Returns
    -------
    list of Path
        The files written.
    """
    by_unit = {}
    for job in jobs:
        entry = {
            'inputs': [str(p) for p in job['inputs']],
            'outputs': [str(p) for p in job['outputs']],
            'forced': bool(job['forced']),
        }
        by_unit.setdefault(job['admin_id'], {})[job['rule']] = entry
    written = []
    for admin_id, rules in by_unit.items():
        path = job_io_path(admin_id)
        current = _read_json(path) or {}
        if any(current.get(rule) != entry for rule, entry in rules.items()):
            _write_json(path, {**current, **rules})
        written.append(path)
    return written


def load_job_io(path, rule: str) -> dict | None:
    """Return one rule's declared files from a `write_job_io` file."""
    return (_read_json(Path(path)) or {}).get(rule)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_state(path, columns=None) -> dict[str, str] | None:
    """Return the digests of a file's *columns* (None: all of it).

    Parquet files are digested by column (`io.tables.file_digests`);
    any other file by its bytes, and a directory by its listing with
    sizes and mtimes. None when the path does not exist.
    """
    from openplaces.io.tables import file_digests

    path = Path(path)
    if not path.exists():
        return None
    if path.is_dir():
        listing = sorted(
            f'{p.relative_to(path)}:{p.stat().st_size}:{p.stat().st_mtime_ns}'
            for p in path.rglob('*')
            if p.is_file()
        )
        return {FILE: hashlib.sha256('\n'.join(listing).encode()).hexdigest()}
    if path.suffix == '.parquet':
        try:
            return file_digests(path, None if columns is None else sorted(columns))
        except Exception:
            pass
    return {FILE: _hash_file(path)}


def _key(path) -> str:
    return str(Path(path).resolve())


def write_record(stage, recipe_id, admin_id, rule, job_io, read_log, commit) -> Path:
    """Record what a job that just succeeded read and wrote.

    Parameters
    ----------
    stage, recipe_id, admin_id, rule : str
        The job (*admin_id* None for a global one).
    job_io : dict
        Its declared files (`load_job_io`).
    read_log : dict
        `io.tables.start_read_log`'s log of the run.
    commit : str or None
        The checkout's commit when the job started.

    Returns
    -------
    Path
        The record file.
    """
    outputs = {_key(p) for p in job_io['outputs']}
    reads = {_key(p): columns for p, columns in read_log.items()}
    for path in job_io['inputs']:
        reads.setdefault(_key(path), None)
    inputs = {}
    for path, columns in reads.items():
        if path in outputs:
            continue
        columns = None if columns is None else sorted(columns)
        inputs[path] = {'columns': columns, 'state': file_state(path, columns)}
    record = {
        'stage': stage,
        'recipe_id': recipe_id,
        'admin_id': admin_id,
        'commit': commit,
        'inputs': inputs,
        'outputs': {path: file_state(path) for path in sorted(outputs)},
    }
    path = record_path(rule)
    _write_json(path, record)
    return path


def discard_record(rule: str) -> None:
    """Remove a rule's record, before the job runs again."""
    record_path(rule).unlink(missing_ok=True)


def can_skip(stage, recipe_id, admin_id, rule, job_io) -> tuple[bool, str]:
    """Return whether a job may exit without running, and why.

    See the module docstring for the conditions.

    Parameters
    ----------
    stage, recipe_id, admin_id, rule : str
        The job (*admin_id* None for a global one).
    job_io : dict
        Its declared files (`load_job_io`).

    Returns
    -------
    tuple of (bool, str)
        Whether to skip, and the reason for the decision.
    """
    from openplaces.flow.code_staleness import job_changes

    if stage not in RECORD_STAGES:
        return False, f'{stage} jobs always run'
    if job_io.get('forced'):
        return False, 'forced'
    record = _read_json(record_path(rule))
    if not record:
        return False, 'no record of a previous run'
    outputs = {_key(p) for p in job_io['outputs']}
    if set(record['outputs']) != outputs:
        return False, 'its outputs changed'
    for path, state in record['outputs'].items():
        if state is None or file_state(path) != state:
            return False, f'output changed since it was written: {Path(path).name}'
    missing = {_key(p) for p in job_io['inputs']} - set(record['inputs'])
    if missing:
        return False, f'new input: {Path(sorted(missing)[0]).name}'
    for path, entry in record['inputs'].items():
        if file_state(path, entry['columns']) != entry['state']:
            return False, f'input changed: {Path(path).name}'
    commit = record.get('commit')
    if not commit:
        return False, 'no commit recorded'
    changed = job_changes(
        stage, recipe_id, admin_id, commit, [*outputs, *record['inputs']]
    )
    if changed is None:
        return False, f'commit {commit} unknown to git'
    if changed:
        return False, f'code changed since {commit}: {changed[0]}'
    return True, f'inputs and code unchanged since {commit}'
