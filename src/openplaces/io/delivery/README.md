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
another's filename. A single-region recipe needs no selector.

**It refuses curated files older than their inputs** (`freshness.py`;
maintainer's decision 102, 2026-10-24). A spine rebuilt in one run
leaves the consumers another run built from it on disk, older than the
spine, and an export that did not look shipped them: transaction curates
linked to a property spine that had since been fixed reached the
valuation lane this way (2026-10). Before reading anything,
`export_delivery` runs `out_of_date` (file modification times only) on
the dependency graph of the pooled files and raises `StaleOutputError`,
listing every out-of-date output with the newer input or stale upstream
file it was found by, when any pooled file is among them. The graph is
`flow.dag.RecipeDAG`, which sits above this package, so the caller
passes it: `graph=openplaces.flow.freshness.delivery_graph(recipe_id,
region)`. Without a graph, or with one that does not build every pooled
file, it refuses too. `allow_stale=True` (`run_stage deliver
--allow-stale`, `region_verify --allow-stale`) skips the check.

**A named region's bundle sits in a folder named after the region**, inside
its unit's folder (`share/US/FL/_all/transaction/openplaces/2026/lakeland-fl/`;
maintainer's rule, 2026-10-23: regional bundles do not sit in the state
folder). The file names still carry the unit (`US-FL_...`), so the folder is
what tells a Lakeland bundle from Florida's. Only a region the recipe lists
under `share: delivery: whole_unit_regions:` is filed as its unit's own
bundle, with no folder (`florida-statewide` is
`.../2026/US-FL_transaction-openplaces-2026.parquet`), and `export_delivery`
refuses one whose members are not every process-level unit of it. A
region's path therefore does not depend on which other regions its recipe
declares. `delivery_paths` raises where a bundle filed as its unit would
land on a curated output (an inline `admin_ids:` grouping on a county) or on
another whole-unit region's bundle. Consumers ask `delivery_paths` (or
`delivery_accuracy_dir`, whose `accuracies/` folder follows the bundle) and
never rebuild the path.

Sibling regions are told apart by containment, not by level: both CHEER bundles
sit at admin level 2, so `RecipeDAG._delivery_in_scope` asks whether a requested
unit is the bundle's own unit or an ancestor of it (`US`, `US-TX`), not merely
whether it is coarse enough. `--config deliver=true` likewise only forces the
regions the run actually touches.

**The share bucket takes bundles only.** Curated per-unit files are saved to
`core`, since they hold every source's values unredacted; a curated Fairfax
property file written into `share/` carried a restricted roll's attributes
(2026-10-21). `delivery_paths` files bundles in `share` whatever bucket the
recipe saves to, a recipe whose `save_to: data_dir:` is `share` fails to
load, and `io.tables` refuses any table write into the share bucket
(`UnredactedShareWriteError`) outside `redacted_share_writes()`, which
`export_delivery` enters only around its writes of redacted frames. Team
folders get no exemption: a team bundle still withholds uncleared sources.

**It withholds what restricted sources supplied, and ships the rest.** A
source recording `redistribution_restricted: true` or a `usage_requirement`
may feed the recipe; `bundle_terms.restricted_inputs` lists every such input,
walked per pooled unit because county parcel and property recipes reach a
spine only through auto-discovery, which resolves per admin unit (an unscoped
walk never saw Edgecombe's parcels feeding the NC bundle), and read with the
unit's patch recipes applied, so a source a patch merges by named columns is
withheld by those columns. A step that finds its sources by entity type and
names its columns (the parcel spine's `link_by_id` keeping
`last_sale_price`, `last_sale_date` and `n_transactions` from every
transaction source of the unit) is read the same way, so a statewide sales
table reaches those three columns and no other parcel value.
`io.delivery.redaction.withhold` then removes that source's values from both read
passes: a row whose `geometry_source` names it is left out, a cell whose
`{col}_source` names it is emptied, and inside the source's county, columns
named after an attribute its recipe maps (or with its entity suffix), or whose
sidecar names only its layer (`parcel`), are emptied unless a sidecar names
another specific source. Emptied cells' sidecars read `withheld`, and the
notice lists each source's counts.

**What openplaces derived from a withheld source goes with it, within
the unit** (maintainer's decisions, 2026-10-23 and 2026-10-24). In a
unit where a restricted source supplied values, a cell whose sidecar is
marked `imputed`, or names only the layer it came through (`parcel`: a
structure value apportioned from the roll, the roll's year copied onto
a footprint), is withheld too. An estimate looked up from a place
reference (a city from its ZIP code, `zipcodes+imputed`) is tied to a
place, not to the roll's values, and stays. So does an estimate marked
`open`, in either of two places. The delivery's building imputation
writes `<version>+open+imputed`: its models were fit on the bundle's
frame after every restricted and share-alike value had been withheld
from it, so none of them reached the cell. Curate's `impute_land_value`
with `open_only: true` writes `<tier>+open+imputed`: it learns only from
rows these rules withhold nothing from, and estimates only such rows,
so no restricted value reached the cell. A `{col}_per_area` ratio
goes with its `{col}`, since times the shipped area it is the value,
and a column the recipe declares under `share: derived_from:` (the
footprint's `occupancy_type_parcel`, the roll's use class in occupancy
terms) goes on the rows where one of its inputs goes. Which unit a
source supplied is read from the bundle itself (its own token on a
column it reaches, or its being the layer in scope), not from the
tokens' wording, which names an estimate's peer group
(`parcel.city__is_residential+imputed`) rather than its inputs, and not
from the dependency walk alone, under which a statewide source reached
through the curated parcels reaches every value column of every unit.
An open county's estimates ship. A token counts as naming a source only
where a recipe declares that source id (`terms.known_source_ids`); a
tier name used to pass for one and exonerate an estimate inside the
restricted county. The walk records, per source, the entity of the
layer it reaches the recipe through (`via_entities`, `parcel` for a
county roll reaching footprints). A fold that carries another curate
whole (`aggregate_from_entities` with `columns: all`, the footprint
curate taking the building curate) names no layer of its own: its cells
keep the tokens the folded recipe wrote, so the layer is named one level
further down (`terms._carries_whole`). Named `building` instead, a cell
saying only `parcel` no longer matched, and Arlington's restricted
structure values would have shipped. A withheld structure value stays
empty in a public bundle: no other source stands in for it, since
NSI's is not right at the building level (maintainer, 2026-10-24).
`RestrictedInputError` is only the
backstop: it fires if restricted values are still in the frames about to be
written, never because a source merely feeds the recipe, so one county's
terms cannot hold the rest of a bundle hostage. The rules over-withhold where
a layer's sidecar cannot say which source filled it; that costs values, not
a leak. A no-resale clause withholds nothing; the notice reports it.

**Share-alike is shipped only when it is one family** (maintainer's rule,
2026-10-23). `terms.share_alike_family` groups share-alike licences into
families that one release can satisfy: ODbL; CC BY-SA 2.0 to 4.0 (each
allows a later version); GPL; OGL-SA. The comment above it cites each
licence's own compatibility clause. A bundle whose remaining geometry or
cells, after withholding, come from two families (Overture's ODbL
outlines beside Pitt County's CC-BY-SA-4.0 parcels) cannot be published
under either, so `export_delivery` raises `ShareAlikeConflictError`, naming
each source with its licence, geometry share and cells, and writes
nothing. **What a share-alike source left in the bundle is found by the
withholding rules above**, applied to `terms.share_alike_inputs` (every
share-alike input, walked per unit like `restricted_inputs`): its
outlines, cells whose `_source` names it, and inside its scope the
columns named after what it maps. A county roll's values in a sales
bundle carry no sidecar of their own and are found the third way. A
source that feeds the recipe but left nothing in the bundle passes no
licence on. `check_share_alike(recipe, region)` gives the same verdict
without writing; with `allow_missing=True` it checks the members that
have a curated file and lists the rest.

**A region can resolve a conflict by withholding one family**, declared
under `share: delivery: withhold_share_alike:` as a map from region id to
families (`{greenville-nc: [odbl]}`; a plain list applies to every
region of the recipe). The family's sources are withheld exactly as a
restricted source is, their sidecars read `withheld`, and the notice
lists their counts under "Withheld to keep one share-alike licence".
`greenville-nc` in the transaction recipe withholds ODbL: Pitt County's
CC BY-SA sales, without the ODbL footprint values Pitt's spines keep for
internal use and CHEER (decision 29(b)). `cheer-eastern-nc` in the
footprint recipe withholds CC BY-SA: its public bundle ships ODbL
outlines without Pitt's values.

**Share-alike binds only on publication** (maintainer, 2026-10-23), so a
team bundle, never published, is exempt from both: it may carry two
families, `export_delivery` does not refuse it, and it does not inherit
its region's `withhold_share_alike`. The CHEER team bundle carries
Edgecombe and Pitt together (2026-10-22, option A). Its notice still
names the families, and `check_share_alike` reports them under
`mixed_families` with `conflict` False.
**Team bundles are the one exception, published only behind a password.** A
team bundle is shared within the research team, and reaches a web page only
as a team value page (`viz.web_scene`, `team: true` in its `places.yaml`),
which `deploy_web_scene` uploads behind a password file on the host and
refuses to upload without one (maintainer, 2026-10-18). A region listed
under `share: delivery: team_regions:` also ships as `<region>.team`, in a
`team/` subdirectory beside the region's own bundle (whose path does not
move): `.../2026/nantucket-ma/team/`. A team bundle keeps the values of sources a person has cleared with
`team_sharing_permitted: true` (Edgecombe's parcels, decided by the user
2026-09-13), and its notice opens "TEAM-INTERNAL BUNDLE"; every other bundle
withholds them, and uncleared sources are withheld from team bundles too.
Asked for a delivery by admin unit alone, the DAG and `delivery_node` resolve
to the public bundle: a team twin is only ever shipped by naming it.

Three behaviors are worth knowing. It reads each unit twice (canonical+geometry,
then evidence) so the wide evidence columns are never in memory alongside the
polygons. It deduplicates the index: an entity on a county line is curated by
both neighbors and the two copies share an Open Location Code, so the
better-covered copy wins, ties broken on admin id. **A recipe declaring
`share: delivery: slice_at_unit_lines: true` cuts such outlines first**
(`slicing.py`, the footprint recipe; maintainer's rule, 2026-10-17): where two
units delivered one building, as the same outline or two outlines overlapping
by half the smaller, one outline is kept (its representative point in its own
unit, else the larger), cut along the unit line, and each piece takes its
unit's copy's row and values under a new id (the piece centroid's Open
Location Code, lengthened on a collision) with `sliced_from` naming the kept
outline. A piece under 10 m2 or 5% of the outline is not cut: the majority
copy stays and the other copy's value stays on its parcel. An outline only
one unit delivered stays whole. Curated files are untouched. Measured
2026-10-17 on boston-inner-core: 150 outline pairs drawn in two towns became 87
cut outlines (174 pieces) and 63 kept whole, none lost. And it leaves its four
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
