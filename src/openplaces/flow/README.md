<!-- Reference for `openplaces.flow`, moved verbatim from AGENTS.md on
2026-09-28. AGENTS.md's "Maintaining this file" rules apply
here too: depth proportional to surprise, verified against the
current code. -->

# openplaces.flow: reference

## Orchestration (`flow/dag.py`, `workflow/Snakefile`)

`RecipeDAG` derives one job per (stage, recipe, admin unit) from the recipe tree,
plus one terminal **`deliver`** job per delivery region the target recipe
declares (`dag.delivery_nodes`; the older `dag.delivery_node` still returns the
single node when there is exactly one, and raises when there are several).
`deliver` is not a recipe stage -- recipe stages are ranked
(`ingest < harmonize < enrich < curate`) and drive `find_entity_recipe_id`, so a
fifth would ripple into recipe resolution for nothing. It is a node kind this
graph derives, the way `extra_outputs` derives link sidecars from `save_link`.
An image recipe is the opposite case: it is configuration only (its imagery
is fetched in memory during enrichment and never written), so
`dag.writes_nothing` keeps it out of the graph's jobs and out of every
enrich job's inputs, while `exclude_recipe_ids` still prunes it through the
`image_recipe` edge. Until 2026-09-23 it got an ingest job whose output no
run could produce.

Scope decides whether each runs: an unscoped run builds and ships every declared
region, a run naming a region (or covering every member of it) ships that one,
and a narrower debug run stops at curate so a one-county rebuild cannot
overwrite a shipped regional file. `--config deliver=true|false` overrides
either way, `true` still limited to the regions the run touches.

A **`validate`** job follows each delivery, one per shipped region whose
recipe lists notebooks for it (`validation: notebooks:`, each naming the
reference it scores; `ground_truth` or a sidecar `references:` key resolves to
a region). It shares the delivery's recipe, unit and region, so `node_key`
adds the stage as a fourth part. `run_stage validate <recipe> <region>`
executes the notebooks with `jupyter nbconvert --execute`, passing arguments
through `OPENPLACES_NOTEBOOK_ARGS` (read in each notebook's test-arguments
cell), writes executed copies to the cache `_logs/validate/` tree, runs
`docs/_ext/generate_validation_tables.py`, and records a
`{recipe}_validation-run.json` manifest as its output. It follows `deliver`'s
scope; `--config validate=false|true` overrides it.

**A staging recipe's ingest is one job for all its units**
(`ingest_<recipe>_batch` in the Snakefile, for recipes that stage a
shared archive), and **the profiles keep a failed job's outputs**
(`keep-incomplete`, `rerun-incomplete`; `flow/submit.py` passes both
flags when it runs without a profile). Snakemake otherwise deletes every
output of a failed job, so one failing unit removed the whole recipe's
good files. What a failed batch leaves is whole, because
`io.tables.to_parquet` writes under a temporary name
(`<name>.parquet.tmp-<pid>`, which no `*.parquet` glob matches) and moves
the file into place, retrying for about 10 s while Dropbox or a reader
holds the target. Snakemake does not mark those kept files as
incomplete: a downstream job that needs only a good unit runs on it, and
the batch reruns when something needs a missing unit (tested 2026-10-17
on a two-unit batch). An attribute file and its `_geo` sidecar are moved
one after the other, so the pair is close to, not exactly, atomic. A
script that calls `snakemake` directly, without a profile, should pass
`--keep-incomplete --rerun-incomplete` too.

**A job's memory reservation is its own measured peak**
(`flow/memory.py`), not a flat value per stage. `mem_mb` comes from the
first that applies, times 1.2 plus 512 MB:

1. the peak this machine measured for the job, scaled up if its inputs
   have grown since;
2. this machine's envelope for its family (same stage and recipe, at
   least 5 measured units): a line of peak against input bytes (`_geo`
   sidecars counted) that covers every measured unit;
3. the seed envelope for the family in `workflow/profiles/mem_seed.csv`,
   in git, fitted on the maintainer's Windows machine (US-FL waves of
   2026-10-19/20);
4. the flat `profiles/scc/mem_lookup.csv` value (a recipe or unit row
   there is a floor under any of the above).

Under 2 and 3, a job whose input is larger than any unit the envelope
was fitted on gets the whole `--resources mem_mb` cap, so it runs alone.
Snakemake evaluates the callable when the job is ready, so its inputs
are on disk, and lowers a request above the cap to the cap.

**Peaks are platform-specific, so the measured table is per machine.**
On Windows the benchmark's `max_vms` is committed memory (what the
low-memory kill counts, about 10% above the resident set); on Linux
`max_rss` is the resident set and `max_vms` an address space that can
be far larger. The table lives outside git at
`<logs>/mem/mem_measured.csv` (`--config mem_table=<csv>` points a run
elsewhere), and each machine refreshes its own after a wave with
`python -m openplaces.flow.memory refresh`; a row only grows, so delete
it to forget a peak. The input size it pairs with a peak is the
`<rule>.input_mb` file the Snakefile writes beside the benchmark, also
in a dry run (it is used only when the benchmark is newer). `python -m
openplaces.flow.memory seed --scope <admin id>` refits the seed from a
local table. The cluster starts from the seed and passes the value as
`mem_per_core`; with no cap there, a job above its family's range keeps
the envelope and is retried with twice the memory if it fails.

**A footprint geospine depends on its neighbors' footprint ingests.** It
reconciles buildings crossing the unit line (harmonizer README,
`spine.py`), so the graph adds, for each touching unit, the ingest jobs
`io.harmonizer.neighbors.neighbor_inputs` names: the sources the unit reads
itself, less the step's `exclude_sources`, from recipes and admin polygons
only. A one-unit run therefore plans more jobs; measured 2026-10-18 on
US-MA-SOM: 15 more (OBM, Microsoft and MassGIS for 5 neighbors).
`OPENPLACES_NEIGHBOR_READS=0` in the environment of the `snakemake` call
drops the edges and the reads together for a debug run, and the geospine's
sidecar fingerprints record it, so a later full run rebuilds it.
