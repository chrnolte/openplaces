"""Disk-driven stage-chain worker pool for one region; run several at once.

The pattern that finished the 2026 CHEER builds after their Snakemake
parents died: workers coordinate through atomic claim files on disk, so
any number of them (started together or hours apart, surviving or not)
converge on one complete region without a live parent process. Disk
state is the source of truth: a county whose output already exists and
passes the bbox probe is skipped, everything else is claimed and built.

Two worker flavors, one file:

- A **striding worker** (``--offset K --stride N``) walks every Nth
  member and reclaims a claim older than ``--claim-ttl`` seconds,
  assuming its holder crashed.
- A **helper** (``--helper``) walks the whole list in reverse, never
  reclaims (so it cannot collide with a slow legitimate build), and
  refreshes its own claim's mtime from a thread so striding workers'
  TTL never fires on a county it still holds. Start one when a worker
  finishes its slice early and cores sit idle.

The stage chain is an argument, not code, because which recipes make up
a build is pipeline configuration::

    python -m openplaces.flow.region_pool US_footprint-openplaces-2026
        cheer-coastal-tx
        --chain "harmonize:US_property-spine-2026,
                 harmonize:US_footprint-geospine-2026,..."
        --offset 0 --stride 3
"""

import argparse
import os
import threading
import time
import warnings

MARGIN = 0.1


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def claim_dir(region_id):
    """Per-region claim directory beside the configured cache root."""
    from openplaces.config import cfg

    path = os.path.join(cfg.get_dir('cache'), '_claims', region_id)
    os.makedirs(path, exist_ok=True)
    return path


def claim(region_id, unit, ttl=None):
    """Atomically claim a unit; return its claim path or None.

    With ``ttl`` set, a claim file older than ttl seconds is treated as
    a crashed worker's and reclaimed. Without it (the helper), any
    existing claim is respected regardless of age.
    """
    path = os.path.join(claim_dir(region_id), unit)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return path
    except FileExistsError:
        if ttl is not None and time.time() - os.path.getmtime(path) > ttl:
            try:
                os.remove(path)
            except OSError:
                pass
            return claim(region_id, unit, ttl=ttl)
        return None


def parse_chain(raw):
    """Parse 'stage:recipe_id,stage:recipe_id,...' into a step list."""
    steps = []
    for part in raw.replace('\n', ',').split(','):
        part = part.strip()
        if not part:
            continue
        stage, _, recipe_id = part.partition(':')
        if stage not in ('harmonize', 'enrich', 'curate'):
            raise ValueError(f'Unknown stage in chain entry: {part!r}')
        steps.append((stage, recipe_id))
    return steps


def gate(op, adm, recipe_id, unit):
    """Row count and bbox overshoot of a unit's curated output."""
    g = op.get_entities(recipe_id, unit, geom=True)
    a = adm.loc[unit].geometry.bounds
    b = g.total_bounds
    over = max(a[0] - b[0], b[2] - a[2], a[1] - b[1], b[3] - a[3])
    return len(g), over


def run_worker(
    recipe_id,
    region_id,
    chain,
    offset=0,
    stride=1,
    helper=False,
    claim_ttl=3600,
):
    """Build every unclaimed, unfinished member of a region.

    Parameters
    ----------
    recipe_id : str
        The terminal curated recipe; its output existence and bbox gate
        decide whether a unit is already done.
    region_id : str
        Region from the shared registry.
    chain : list of (str, str)
        Ordered (stage, recipe_id) steps to run per unit.
    offset, stride : int, optional
        This worker's slice of the member list.
    helper : bool, optional
        Reverse-order, never-reclaiming, claim-refreshing mode.
    claim_ttl : int, optional
        Seconds before a striding worker treats a claim as crashed.
    """
    import openplaces as op
    from openplaces.core.schema import AdminId
    from openplaces.recipe import get_output_path, get_recipe_by_id

    members = op.get_region_admin_ids(region_id)
    mine = list(reversed(members)) if helper else members[offset::stride]
    parent = '-'.join(members[0].split('-')[:-1])
    level = len(members[0].split('-'))
    adm = op.get_admin(parent, level=level, geom=True)
    recipe = get_recipe_by_id(recipe_id)

    for unit in mine:
        exists = get_output_path(recipe, AdminId(*unit.split('-'))).exists()
        if exists:
            try:
                n, over = gate(op, adm, recipe_id, unit)
                if n and over < MARGIN:
                    continue
            except Exception:
                pass
        path = claim(region_id, unit, ttl=None if helper else claim_ttl)
        if path is None:
            continue
        stop = threading.Event()
        if helper:

            def keep_fresh(claim_path=path, stop_event=stop):
                while not stop_event.wait(300):
                    try:
                        os.utime(claim_path)
                    except OSError:
                        pass

            threading.Thread(target=keep_fresh, daemon=True).start()
        t0 = time.time()
        try:
            try:
                for stage, rid in chain:
                    getattr(op, stage)(rid, unit, reprocess=not exists)
                n, over = gate(op, adm, recipe_id, unit)
                if not (n and over < MARGIN):
                    raise RuntimeError(f'gate: rows={n} over={over:.3f}')
            except Exception:
                # One retry with a full reprocess: the cheap pass fails
                # legitimately when a stale intermediate survives from a
                # crashed builder.
                for stage, rid in chain:
                    getattr(op, stage)(rid, unit, reprocess=True)
                n, over = gate(op, adm, recipe_id, unit)
            verdict = 'pass' if n and over < MARGIN else f'GATE over={over:.3f}'
            log(f'POOL {verdict:9s} {unit} {time.time() - t0:.0f}s {n:,} rows')
        except Exception as exc:
            log(f'POOL FAIL {unit}: {type(exc).__name__}: {str(exc)[:110]}')
        finally:
            stop.set()
    log(f'worker {"helper" if helper else f"{offset}/{stride}"} done')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('recipe_id')
    parser.add_argument('region_id')
    parser.add_argument('--chain', required=True)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--helper', action='store_true')
    parser.add_argument('--claim-ttl', type=int, default=3600)
    args = parser.parse_args(argv)
    warnings.filterwarnings('ignore')
    run_worker(
        args.recipe_id,
        args.region_id,
        parse_chain(args.chain),
        offset=args.offset,
        stride=args.stride,
        helper=args.helper,
        claim_ttl=args.claim_ttl,
    )


if __name__ == '__main__':
    main()
