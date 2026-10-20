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

**A new source reaches built spines only through `stale_inputs`.** The
profiles rerun on mtime only, and Snakemake then reruns a job whose output
exists only when an input is newer; an input that does not exist yet is not
looked at, so its producer is never scheduled. Adding a source recipe for
units whose spines were built planned none of its ingests until the spines
were deleted by hand (2026-10-19, Cundinamarca's cadastre over the Andes).
`rule all` therefore lists `dag.stale_inputs()` beside `dag.target_paths()`:
every input of a built job that another job produces and that is not on disk.
The producers run, their outputs are newer than the consumers', and exactly
those consumers rerun. A consumed intermediate (absent, with a cleanup
receipt) is not returned, since `input_paths` already drops it for a job
whose output exists.
