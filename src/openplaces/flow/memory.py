"""Memory reservations for workflow jobs, sized from measured peaks.

The Snakefile asks :func:`MemoryModel.estimate` for each job's `mem_mb`
when the job becomes ready to run (Snakemake evaluates a resource
callable then, so the job's inputs are on disk). The answer comes from
the first of these that applies:

1. **The job was measured on this machine.** Its recorded peak, scaled
   up if its inputs have grown since, times a safety margin.
2. **Its family was measured on this machine** (same stage and recipe,
   at least `MIN_FAMILY_OBSERVATIONS` units). A linear upper envelope of
   peak against input size that covers every measured unit, evaluated
   at this job's input size, times the same margin.
3. **Its family is in the seed table** shipped with the code
   (`workflow/profiles/mem_seed.csv`): the same envelope, fitted once
   on the maintainer's machine. This is what a machine without its own
   measurements, such as the cluster, starts from.
4. **Nothing was measured.** The flat per-stage value the caller passes
   (the Snakefile's `mem_lookup.csv`), or a thinly measured family's
   largest peak if that is larger.

Under 2 and 3, a job whose input is larger than that of every unit the
envelope was fitted on is given the run's whole memory cap, so it runs
alone.

Peaks are specific to the platform that measured them. On Windows the
benchmark's `max_vms` is the committed private memory, which is what
the low-memory condition counts, and it runs about 10% above the
resident set; on Linux `max_rss` is the resident set and `max_vms` the
address space, which can be far larger. So each machine keeps its own
table of measured jobs, outside git, under its logs directory
(:func:`default_table_path`), and refreshes it after a wave with
``python -m openplaces.flow.memory refresh``. A row only ever grows: a
later, lower measurement never replaces a higher one, so a run that
resumed from a checkpoint and peaked low cannot shrink the next
reservation. Delete a row to forget it. The seed table is rebuilt from
a local table with ``python -m openplaces.flow.memory seed``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

# Why size jobs at all, rather than one flat value per stage: a flat
# value has to cover the worst job of its stage, and the stage's jobs
# differ by two orders of magnitude.
# Measured 2026-10-20 on US-FL: harmonize peaks 0.2-1.5 GB (median), 31.8 GB max.
# The flat 16 GB was 10 to 80 times what the typical job used, so an
# 18 GB cap ran one job at a time on 20 cores (11.0 h for 402 jobs),
# and half of what the largest jobs used: a heavy job reserved at half
# its peak lets the scheduler start another heavy job beside it, which
# is what pushes the machine into the low-memory kill of other
# sessions.
#
# Why measured peaks and not a model of the code: peak memory depends
# on the county's data in ways no single size explains.
# Measured 2026-10-20 on US-FL: property spine peak/input correlation 0.46.
# (A 242 MB input peaked at 31.8 GB while a 574 MB one stayed under
# 1 GB.) A job's own last peak is the best predictor of its next one,
# because the jobs are deterministic. The input-size envelope is only
# the fallback for a job that never ran, where it errs on the large
# side by construction.
#
# The margin is a judgment, not a fit: the benchmark samples memory
# rather than tracing it, so a short spike can fall between samples,
# and a rerun reads files that may have grown slightly.
# Measured 2026-10-20 on US-FL: 737 jobs reserved 1.50x summed peak.
# (The median job 1.86x: the small jobs carry most of the excess,
# through the fixed headroom.)
SAFETY_FACTOR = 1.2
# Added after the factor: covers the interpreter and the libraries
# for the smallest jobs, where 20% of 200 MB is no margin at all.
HEADROOM_MB = 512
# Reservations are rounded up to this, so tiny input changes do not
# change a job's reservation from one evaluation to the next.
ROUND_MB = 256
# Fewer units with a known input size than this and a family's slope
# is not trusted: the seed is used instead if it has the family, else
# the family's largest peak, or the flat value if that is larger.
MIN_FAMILY_OBSERVATIONS = 5

TABLE_COLUMNS = (
    'rule',
    'stage',
    'recipe_id',
    'admin_id',
    'peak_mb',
    'input_mb',
    'measured',
)
SEED_COLUMNS = (
    'stage',
    'recipe_id',
    'n_measured',
    'floor_mb',
    'slope_mb_per_input_mb',
    'max_peak_mb',
    'max_input_mb',
)

# An admin id at the end of a rule name, as `dag.rule_name` writes it.
# A recipe id never ends in one: its last part is a version or a file
# name, both lowercase or numeric.
_RULE_ADMIN = re.compile(r'_((?:[A-Z][A-Z0-9](?:-[A-Za-z0-9]+)*)|global)$')
_STAGES = ('ingest', 'harmonize', 'enrich', 'curate')


@dataclass(frozen=True)
class Observation:
    """The measured peak of one job.

    Parameters
    ----------
    rule : str
        Snakemake rule name of the job (`dag.rule_name`).
    stage : str
        Pipeline stage, e.g. 'harmonize'.
    recipe_id : str
        Recipe the job runs.
    admin_id : str or None
        Admin unit, None for a global job or a batched ingest.
    peak_mb : float
        Peak memory in MB: the larger of `max_rss` and `max_vms` in
        Snakemake's benchmark. What that means depends on the platform
        (see the module docstring).
    input_mb : float or None
        Total size of the job's inputs when it ran, None when unknown.
    measured : str
        ISO date of the measurement.
    """

    rule: str
    stage: str
    recipe_id: str
    admin_id: str | None
    peak_mb: float
    input_mb: float | None
    measured: str


@dataclass(frozen=True)
class Envelope:
    """A family's upper bound of peak memory against input size.

    The bound is `floor_mb + slope * input_mb`. `floor_mb` is the
    family's smallest peak (the interpreter and libraries), raised to
    the largest peak of any unit that read no files; `slope` is the
    largest `(peak - floor_mb) / input_mb` of any unit, so the line
    passes on or above every point it was fitted on. It is a bound,
    not a fit: a least-squares line would put half the units above it.

    Parameters
    ----------
    stage, recipe_id : str
        The family.
    n_measured : int
        Units with a known input size the bound was fitted on.
    floor_mb : float
        Bound at zero input, in MB.
    slope : float
        MB of peak per MB of input, at least 0.
    max_peak_mb : float
        Largest peak among those units.
    max_input_mb : float
        Largest input among those units; a larger input is outside the
        range the bound was fitted on.
    """

    stage: str
    recipe_id: str
    n_measured: int
    floor_mb: float
    slope: float
    max_peak_mb: float
    max_input_mb: float

    def need(self, input_mb: float | None) -> float:
        """Bound in MB at *input_mb*; the largest peak when unknown."""
        if input_mb is None:
            return max(self.max_peak_mb, self.floor_mb)
        return self.floor_mb + self.slope * input_mb


def fit_envelope(observations: Iterable[Observation]) -> Envelope | None:
    """Fit the upper envelope of one family's measured units.

    Parameters
    ----------
    observations : iterable of Observation
        Units of one family. Those without a known input size are left
        out.

    Returns
    -------
    Envelope or None
        None when fewer than `MIN_FAMILY_OBSERVATIONS` units have a
        known input size.
    """
    sized = [o for o in observations if o.input_mb is not None]
    if len(sized) < MIN_FAMILY_OBSERVATIONS:
        return None
    floor = min(o.peak_mb for o in sized)
    floor = max([floor, *(o.peak_mb for o in sized if o.input_mb == 0)])
    slope = max(
        [0.0, *((o.peak_mb - floor) / o.input_mb for o in sized if o.input_mb > 0)]
    )
    return Envelope(
        stage=sized[0].stage,
        recipe_id=sized[0].recipe_id,
        n_measured=len(sized),
        floor_mb=floor,
        slope=slope,
        max_peak_mb=max(o.peak_mb for o in sized),
        max_input_mb=max(o.input_mb for o in sized),
    )


def parse_rule_name(rule: str) -> tuple[str, str, str | None] | None:
    """Split a rule name into stage, recipe id and admin id.

    Parameters
    ----------
    rule : str
        A name `dag.rule_name` produced, or a batched ingest's
        `ingest_<recipe>_batch`.

    Returns
    -------
    tuple or None
        (stage, recipe_id, admin_id), admin_id None for a global job or
        a batch; None for a name of another shape (a deliver or validate
        job, whose name ends in a region).
    """
    stage, _, rest = rule.partition('_')
    if stage not in _STAGES or not rest:
        return None
    if stage == 'ingest' and rest.endswith('_batch'):
        return stage, rest[: -len('_batch')], None
    match = _RULE_ADMIN.search(rest)
    if not match or match.start() == 0:
        return None
    admin = match.group(1)
    return stage, rest[: match.start()], None if admin == 'global' else admin


def input_megabytes(paths: Iterable[str | os.PathLike]) -> float | None:
    """Total size of a job's inputs in MB, counting `_geo` sidecars.

    A geometry sidecar (`<name>_geo.parquet`) is read with its attribute
    file but is not a declared input, so it is added here; for a parcel
    table it is most of the bytes.

    Parameters
    ----------
    paths : iterable of path-like
        The job's declared input files.

    Returns
    -------
    float or None
        Size in MB (10**6 bytes), or None if any declared input is
        missing, so a job evaluated before its inputs exist is not
        sized as if it had none.
    """
    total = 0
    for path in paths:
        path = str(path)
        try:
            total += os.path.getsize(path)
        except OSError:
            return None
        if path.endswith('.parquet'):
            sidecar = path[: -len('.parquet')] + '_geo.parquet'
            try:
                total += os.path.getsize(sidecar)
            except OSError:
                pass
    return total / 1e6


def read_benchmark(path: str | os.PathLike) -> float | None:
    """Peak memory in MB from one Snakemake benchmark file.

    Parameters
    ----------
    path : path-like
        A benchmark `.tsv` with `max_rss` and `max_vms` columns.

    Returns
    -------
    float or None
        The largest `max_rss` or `max_vms` over its rows, None if the
        file has no readable value.
    """
    peak = None
    try:
        with open(path, encoding='utf-8') as fh:
            for row in csv.DictReader(fh, delimiter='\t'):
                for column in ('max_rss', 'max_vms'):
                    try:
                        value = float(row.get(column) or 'nan')
                    except ValueError:
                        continue
                    if not math.isnan(value):
                        peak = value if peak is None else max(peak, value)
    except OSError:
        return None
    return peak


def load_table(path: str | os.PathLike) -> dict[str, Observation]:
    """Read a machine's measured-peak table, keyed by rule name.

    Parameters
    ----------
    path : path-like
        CSV with the columns in `TABLE_COLUMNS`. A missing file is an
        empty table.

    Returns
    -------
    dict
        Rule name to :class:`Observation`.
    """
    table = {}
    if not Path(path).exists():
        return table
    with open(path, encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh):
            rule = (row.get('rule') or '').strip()
            if not rule or rule.startswith('#'):
                continue
            table[rule] = Observation(
                rule=rule,
                stage=row['stage'],
                recipe_id=row['recipe_id'],
                admin_id=row.get('admin_id') or None,
                peak_mb=float(row['peak_mb']),
                input_mb=float(row['input_mb']) if row.get('input_mb') else None,
                measured=row.get('measured') or '',
            )
    return table


def write_table(path: str | os.PathLike, table: dict[str, Observation]) -> None:
    """Write a measured-peak table, sorted by rule name.

    Parameters
    ----------
    path : path-like
        Destination CSV; its folder is created if missing.
    table : dict
        Rule name to :class:`Observation`.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        writer.writerow(TABLE_COLUMNS)
        for rule in sorted(table):
            o = table[rule]
            writer.writerow(
                [
                    o.rule,
                    o.stage,
                    o.recipe_id,
                    o.admin_id or '',
                    f'{o.peak_mb:.0f}',
                    '' if o.input_mb is None else f'{o.input_mb:.1f}',
                    o.measured,
                ]
            )


def load_seed(path: str | os.PathLike) -> dict[tuple[str, str], Envelope]:
    """Read the seed table of family envelopes.

    Parameters
    ----------
    path : path-like
        CSV with the columns in `SEED_COLUMNS`. A missing file is an
        empty seed.

    Returns
    -------
    dict
        (stage, recipe_id) to :class:`Envelope`.
    """
    seed = {}
    if not Path(path).exists():
        return seed
    with open(path, encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh):
            stage = (row.get('stage') or '').strip()
            if not stage or stage.startswith('#'):
                continue
            env = Envelope(
                stage=stage,
                recipe_id=row['recipe_id'],
                n_measured=int(row['n_measured']),
                floor_mb=float(row['floor_mb']),
                slope=float(row['slope_mb_per_input_mb']),
                max_peak_mb=float(row['max_peak_mb']),
                max_input_mb=float(row['max_input_mb']),
            )
            seed[(env.stage, env.recipe_id)] = env
    return seed


def write_seed(path: str | os.PathLike, envelopes: Iterable[Envelope]) -> None:
    """Write a seed table, sorted by stage and recipe.

    Parameters
    ----------
    path : path-like
        Destination CSV.
    envelopes : iterable of Envelope
        One per family.
    """
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        writer.writerow(SEED_COLUMNS)
        for e in sorted(envelopes, key=lambda e: (e.stage, e.recipe_id)):
            writer.writerow(
                [
                    e.stage,
                    e.recipe_id,
                    e.n_measured,
                    f'{e.floor_mb:.0f}',
                    f'{e.slope:.4f}',
                    f'{e.max_peak_mb:.0f}',
                    f'{e.max_input_mb:.1f}',
                ]
            )


def build_seed(
    table: dict[str, Observation], scope: str | None = None
) -> list[Envelope]:
    """Fit one envelope per family of a measured-peak table.

    Parameters
    ----------
    table : dict
        Rule name to :class:`Observation`.
    scope : str, optional
        Keep only units whose admin id is this id or lies inside it
        (`US-FL` keeps `US-FL-OR`), e.g. to fit on one wave's state.

    Returns
    -------
    list of Envelope
        Families with enough units of known input size.
    """
    families: dict[tuple[str, str], list[Observation]] = {}
    for obs in table.values():
        if scope is not None and not (
            obs.admin_id == scope or (obs.admin_id or '').startswith(scope + '-')
        ):
            continue
        families.setdefault((obs.stage, obs.recipe_id), []).append(obs)
    fitted = (fit_envelope(members) for members in families.values())
    return [e for e in fitted if e is not None]


def ledger_path(benchmark_dir: str | os.PathLike, rule: str) -> Path:
    """Where the Snakefile records a job's input size, beside its benchmark."""
    return Path(benchmark_dir) / f'{rule}.input_mb'


def record_input_size(
    benchmark_dir: str | os.PathLike, rule: str, input_mb: float
) -> None:
    """Record a job's input size for the next :func:`refresh`.

    The benchmark Snakemake writes holds the peak but not what the job
    read, and the input size is what lets the family envelope fit an
    unmeasured unit. Written when the job is evaluated, before it runs;
    :func:`refresh` pairs it with the benchmark only if the benchmark is
    newer. Best effort: a failed write costs the size, not the run.

    Parameters
    ----------
    benchmark_dir : path-like
        Snakemake's benchmark directory.
    rule : str
        Rule name of the job.
    input_mb : float
        Its input size in MB.
    """
    try:
        path = ledger_path(benchmark_dir, rule)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{input_mb:.1f}\n', encoding='utf-8')
    except OSError:
        pass


def _round_up(mb: float) -> int:
    return int(math.ceil(mb / ROUND_MB) * ROUND_MB)


def _reserve(need_mb: float) -> int:
    return _round_up(need_mb * SAFETY_FACTOR + HEADROOM_MB)


class MemoryModel:
    """Per-job memory estimates from measured peaks and a seed."""

    def __init__(
        self,
        observations: dict[str, Observation],
        seed: dict[tuple[str, str], Envelope] | None = None,
    ):
        """Index the local measurements by job and fit their families.

        Parameters
        ----------
        observations : dict
            Rule name to :class:`Observation`, this machine's table as
            :func:`load_table` returns it.
        seed : dict, optional
            (stage, recipe_id) to :class:`Envelope`, the shipped seed
            as :func:`load_seed` returns it.
        """
        self.units = dict(observations)
        self.seed = dict(seed or {})
        self.families: dict[tuple[str, str], list[Observation]] = {}
        for obs in self.units.values():
            self.families.setdefault((obs.stage, obs.recipe_id), []).append(obs)
        self.envelopes = {
            key: env
            for key, members in self.families.items()
            if (env := fit_envelope(members)) is not None
        }

    @classmethod
    def from_files(
        cls, table_path: str | os.PathLike, seed_path: str | os.PathLike | None
    ) -> MemoryModel:
        """Load a model from a local table and a seed (missing: empty)."""
        seed = load_seed(seed_path) if seed_path is not None else {}
        return cls(load_table(table_path), seed)

    def envelope(self, stage: str, recipe_id: str) -> Envelope | None:
        """The family's local envelope, else its seed envelope, else None."""
        key = (stage, recipe_id)
        return self.envelopes.get(key) or self.seed.get(key)

    def estimate(
        self,
        rule: str,
        stage: str,
        recipe_id: str,
        input_mb: float | None,
        flat_mb: int,
        ceiling_mb: int | None = None,
    ) -> int:
        """Memory to reserve for one job, in MB.

        Parameters
        ----------
        rule : str
            The job's rule name, its key in the local table.
        stage, recipe_id : str
            Its family, for a job without its own measurement.
        input_mb : float or None
            Its current input size (:func:`input_megabytes`).
        flat_mb : int
            What to reserve when nothing of its family was measured.
        ceiling_mb : int, optional
            The run's `mem_mb` limit. An unmeasured job whose input is
            larger than that of every unit its envelope was fitted on
            is given all of it, so it runs alone: the envelope would be
            extrapolating, and the largest unit of a family is where it
            misses. Without a limit (a cluster run) the envelope stands,
            and a job that exceeds it is retried with more.
            Measured 2026-10-20 on US-FL-OR: geospine, 22.0 GB reserved, 31.8 GB used.

        Returns
        -------
        int
            Reservation in MB. Snakemake lowers a reservation above the
            run's `mem_mb` limit to that limit, so the largest jobs run
            alone rather than never.
        """
        own = self.units.get(rule)
        if own is not None:
            need = own.peak_mb
            # Inputs that have grown since the measurement (a new
            # vintage, a wider extract) scale the peak with them.
            # Proportional scaling overstates the growth of a peak that
            # is partly fixed overhead, which is the safe direction.
            if input_mb and own.input_mb:
                need *= max(1.0, input_mb / own.input_mb)
            return _reserve(need)
        env = self.envelope(stage, recipe_id)
        if env is not None:
            reserved = _reserve(env.need(input_mb))
            if ceiling_mb and input_mb is not None and input_mb > env.max_input_mb:
                reserved = max(reserved, int(ceiling_mb))
            return reserved
        members = self.families.get((stage, recipe_id))
        if members:
            # Too few sized units for a slope: the largest peak seen,
            # but never less than the flat value it replaces.
            return max(_reserve(max(o.peak_mb for o in members)), flat_mb)
        return flat_mb


def refresh(
    table_path: str | os.PathLike,
    benchmark_dir: str | os.PathLike,
    since: dt.date | None = None,
) -> tuple[int, int]:
    """Merge Snakemake benchmarks into a measured-peak table.

    A benchmark replaces a job's row only when its peak is higher, so
    the table keeps each job's largest measured peak. The input size
    recorded beside a benchmark (:func:`record_input_size`) is paired
    with it only when the benchmark is the newer file, that is, when
    the job ran after its inputs were measured.

    Parameters
    ----------
    table_path : path-like
        The table to update (created if missing).
    benchmark_dir : path-like
        Directory of `<rule>.tsv` benchmark files.
    since : datetime.date, optional
        Ignore benchmarks older than this date.

    Returns
    -------
    tuple of int
        (rows added, rows raised).
    """
    table = load_table(table_path)
    added = raised = 0
    for bench in sorted(Path(benchmark_dir).glob('*.tsv')):
        rule = bench.stem
        parts = parse_rule_name(rule)
        if parts is None:
            continue
        mtime = bench.stat().st_mtime
        measured = dt.date.fromtimestamp(mtime)
        if since is not None and measured < since:
            continue
        peak = read_benchmark(bench)
        if peak is None:
            continue
        input_mb = None
        ledger = ledger_path(benchmark_dir, rule)
        try:
            if ledger.stat().st_mtime <= mtime:
                input_mb = float(ledger.read_text(encoding='utf-8').strip())
        except (OSError, ValueError):
            pass
        stage, recipe_id, admin_id = parts
        new = Observation(
            rule, stage, recipe_id, admin_id, peak, input_mb, measured.isoformat()
        )
        old = table.get(rule)
        if old is None:
            table[rule] = new
            added += 1
        elif peak > old.peak_mb + 0.5:
            table[rule] = new
            raised += 1
        elif old.input_mb is None and input_mb is not None:
            # Same or lower peak: keep it, but learn the input size
            # when the kept row lacks one.
            table[rule] = replace(old, input_mb=input_mb)
    write_table(table_path, table)
    return added, raised


def default_table_path() -> Path:
    """This machine's measured-peak table: `<logs>/mem/mem_measured.csv`.

    Under the logs directory beside the benchmarks it is built from,
    outside git, because the peaks are specific to the platform that
    measured them.
    """
    from openplaces.config import cfg

    return Path(cfg.get_dir('logs')) / 'mem' / 'mem_measured.csv'


def default_seed_path() -> Path:
    """The seed shipped with the code: `workflow/profiles/mem_seed.csv`."""
    from openplaces.config import cfg

    return Path(cfg.code_root) / 'workflow' / 'profiles' / 'mem_seed.csv'


def main(argv: list[str] | None = None, *, _print: Callable = print) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m openplaces.flow.memory',
        description='Maintain the measured-peak tables behind job memory.',
    )
    sub = parser.add_subparsers(dest='command', required=True)
    ref = sub.add_parser('refresh', help="merge benchmarks into this machine's table")
    ref.add_argument('--table', type=Path, default=None)
    ref.add_argument('--benchmarks', type=Path, default=None)
    ref.add_argument('--since', type=dt.date.fromisoformat, default=None)
    sd = sub.add_parser('seed', help='fit the seed table from a local table')
    sd.add_argument('--table', type=Path, default=None)
    sd.add_argument('--out', type=Path, default=None)
    sd.add_argument('--scope', default=None, help='admin id to fit on, e.g. US-FL')
    args = parser.parse_args(argv)
    table = args.table or default_table_path()
    if args.command == 'refresh':
        if args.benchmarks is None:
            from openplaces.config import cfg

            benchmarks = Path(cfg.get_dir('logs')) / 'benchmarks'
        else:
            benchmarks = args.benchmarks
        added, raised = refresh(table, benchmarks, since=args.since)
        _print(f'{table}: {added} rows added, {raised} raised')
    elif args.command == 'seed':
        out = args.out or default_seed_path()
        envelopes = build_seed(load_table(table), scope=args.scope)
        write_seed(out, envelopes)
        _print(f'{out}: {len(envelopes)} families')
    return 0


if __name__ == '__main__':
    sys.exit(main())
