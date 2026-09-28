<!-- Reference for `openplaces.io.delivery`, moved verbatim from AGENTS.md on
2026-09-28. AGENTS.md's "Maintaining this file" rules apply
here too: depth proportional to surprise, verified against the
current code. -->

# openplaces.io.delivery: reference

## Named regions (`_all/admin/regions/2026/admin-regions-2026.csv`)

A **region** is any named group of admin units the hierarchy cannot express: a
study area, a delivery footprint, a funder's geography. The registry is a flat
1:n CSV (`region_id`, `name`, `region_admin_id`, `admin_id`, one row per member)
read by `op.get_regions()` and `op.get_region_admin_ids(region_id)`, loaded
through `get_recipe_by_id` exactly like the admin spine CSVs beside it.

It exists because the same county list is wanted by delivery, by mapping, and
by ad-hoc analysis, and three copies drift. Anything needing "the CHEER Texas
counties" asks the registry rather than restating 42 ids. A grouping used in
exactly one place need not be registered -- `share: delivery: admin_ids:` still
takes an inline list. `region_admin_id` is the unit a region rolls up to
(`US-TX`); blank means callers derive it, which is what delivery does.

## Delivery bundles (`io/delivery/`, with `terms.py` and `redaction.py` beside the package file)

`export_delivery` pools a curation recipe's per-process-unit files into a
region-wide, shareable set of four files sharing one index: the canonical
attributes as a plain table, the same attributes on centroid points, the
boundary polygons alone, and an `_evidence` supplement holding every remaining
column. Which columns are canonical is declared per recipe in a `share:` block
(`columns`, plus `point_columns` for the point file only) -- not inferred, because
the curated schema holds far more technically-canonical columns than belong in a
compact delivery. Each canonical column's `{col}_source` sidecar is appended
automatically and written into *both* the canonical and the evidence file, so
either reads on its own.

**Four files, unless the entity is not a place.** The point file is built from
`long` and `lat`, which `_share_spec` therefore requires in `share: columns:`,
so a geometry-less entity could not be delivered at all until
`share: geometry: false`. A recipe declaring it ships the canonical table and
its evidence supplement only, and `delivery_paths` omits the `point` and `geo`
keys -- that dict is also the orchestrator's output declaration, so a delivery
job then waits on nothing that is never written. It is opt-in and defaults to
the four-file behavior. The transaction curation recipe is the case it exists
for: a sale is an event whose location belongs to the parcel it names (carried
as `parcel_id_local`, so a consumer joins geometry itself), and a deed covering
several parcels has no single point to put it on.

A `share: delivery:` block names the regions: `admin_level` (a bundle's own
level) plus either `admin_ids` (one grouping, listed inline) or `regions` (a
list of ids from the shared region registry, below). The declared member list
wins over walking the admin hierarchy, because a region is rarely all of a
state's children -- the CHEER regions are 45 of North Carolina's 100 counties
and 42 of Texas's 254.

One recipe ships as many regions as it declares: `US_footprint-cheer-2026`
produces a Carolina bundle and a Texas bundle from identical curation logic, so
the membership is data in the registry rather than structure in the YAML.
`delivery_regions(recipe)` is the resolver; `delivery_spec`, `delivery_members`,
`delivery_paths`, `delivery_admin_id` and `export_delivery` all take a `region=`
selector, and **raise rather than guess** when a multi-region recipe is asked
for "the" region -- silently picking one would ship a region's rows under
another's filename. A single-region recipe needs no selector, so
`export_delivery(recipe_id)` still works unchanged.

Sibling regions are told apart by containment, not by level: both CHEER bundles
sit at admin level 2, so `RecipeDAG._delivery_in_scope` asks whether a requested
unit is the bundle's own unit or an ancestor of it (`US`, `US-TX`), not merely
whether it is coarse enough. `--config deliver=true` likewise only forces the
regions the run actually touches.

**It withholds what restricted sources supplied, and ships the rest.** A
source recording `redistribution_restricted: true` or a `usage_requirement`
may feed the recipe; `bundle_terms.restricted_inputs` lists every such input,
walked per pooled unit because county parcel and property recipes reach a
spine only through auto-discovery, which resolves per admin unit (an unscoped
walk never saw Edgecombe's parcels feeding the NC bundle).
`io.delivery.redaction.withhold` then removes that source's values from both read
passes: a row whose `geometry_source` names it is left out, a cell whose
`{col}_source` names it is emptied, and inside the source's county, columns
named after an attribute its recipe maps (or with its entity suffix), or whose
sidecar names only its layer (`parcel`), are emptied unless a sidecar names
another specific source. Emptied cells' sidecars read `withheld`, and the
notice lists each source's counts. `RestrictedInputError` is only the
backstop: it fires if restricted values are still in the frames about to be
written, never because a source merely feeds the recipe, so one county's
terms cannot hold the rest of a bundle hostage. The rules over-withhold where
a layer's sidecar cannot say which source filled it; that costs values, not
a leak. A no-resale clause withholds nothing; the notice reports it.
**Team bundles are the one exception, and never published.** A region listed
under `share: delivery: team_regions:` also ships as `<region>.team`, in a
`team/` subdirectory beside the region's own bundle (whose path does not
move). A team bundle keeps the values of sources a person has cleared with
`team_sharing_permitted: true` (Edgecombe's parcels, decided by the user
2026-09-13), and its notice opens "TEAM-INTERNAL BUNDLE"; every other bundle
withholds them, and uncleared sources are withheld from team bundles too.
Asked for a delivery by admin unit alone, the DAG and `delivery_node` resolve
to the public bundle: a team twin is only ever shipped by naming it.

Three behaviors are worth knowing. It reads each unit twice (canonical+geometry,
then evidence) so the wide evidence columns are never in memory alongside the
polygons. It deduplicates the index: an entity on a county line is curated by
both neighbors and the two copies share an Open Location Code, so the
better-covered copy wins, ties broken on admin id. And it leaves its four
outputs read-only, unlocking them itself on the next run -- deliberately not
Snakemake's `protected()`, which additionally refuses to ever regenerate a file
and would abort the workflow on every reship.

The QGIS side is `qgis/load_joined_parquet.py`, which resolves the whole set from
any one of its files. It picks the join key at load time (`_join_id`, then
`geo_id`, then whichever entity id the two files share), so one algorithm reads
both a core split-layout file and a delivery bundle -- the only difference
between them is that key.

`viz/qgis_map` renders a delivery as a project rather than a layer. Asked for
the admin unit a recipe delivers, `resolve_layers` returns two standalone
output layers instead of the per-unit curated file: the `_point` centroids
carrying the canonical attributes (`RENDER_POINTS`, template fills rewritten as
markers) and the `_geo` polygons as plain outlines (`RENDER_OUTLINE`). Neither
joins -- classifying millions of polygons to color them costs far more than
reading the centroids. `include_inputs=False` drops the ingest-stage layers for
a map of the product rather than of how it was built, and a style variant whose
classifying column is absent from the data is skipped rather than shipped
empty.
