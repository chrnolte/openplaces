<!-- Reference for `openplaces.io.harmonizer`. AGENTS.md points
contributors and coding agents here before changes to this package. -->

# openplaces.io.harmonizer — Reference

## Overview

Recipe-driven composable pipeline: each step is a registered function
`(state: HarmonizeState, **params) -> HarmonizeState`.  Steps are listed in
the recipe's `pipeline:` key and dispatched via `_STEP_REGISTRY`. Each
registration carries a `phase` tag (`_STEP_PHASES`): `'geometry'` for steps
that mutate spine rows/geometry or run spatial joins, `'attributes'`
(default) for steps that only read or annotate. The tag feeds the
link-sidecar fingerprint (a changed prior geometry step invalidates a
persisted overlay) and defines which recipe hosts a step under the
geospine split: a *geospine* recipe runs the geometry phase and persists
every link product (`save_link` default-on), and an *attribute* recipe
(`entity_recipe:` the geospine, `save_to: geometry: false`) starts with
`load_geospine` (load.py) to restore the spine, crosswalks, overlays, and
prepared references from those persisted tables -- never from a new
spatial computation. A stale or missing sidecar raises with instructions
to rerun the geospine (fail closed).

**Scope: the core structure, with minimal redundancy.** Harmonize builds
each entity's spine and the keys between entities. An attribute lands
once, on the entity it describes (`core/schema.ENTITY_DEFINITIONS`):
a property's room counts on the property spine, not also on parcels.
A preliminary derived column (an early `occupancy_type`) is allowed
only where an enrich step needs it to choose its rows; none does yet.
Enrichment adds external evidence to an entity; curation assembles one
dataset per entity type and does the cross-entity aggregation. See
AGENTS.md, "Entity model and stage roles", for the full rule and the one
known exception still to migrate (the parcel geospine's property join).

Entry points: `harmonize(recipe, admin_ids, ...)` in `__init__.py`.

---

## HarmonizeState (\_\_init\_\_.py)

Central mutable container passed through all steps.

| Field | Type | Purpose |
|-------|------|---------|
| `recipe` | dict | Loaded harmonization recipe |
| `admin_id` | AdminId \| None | Current admin unit being processed |
| `verbose` | bool | Per-step print output |
| `timer` | object \| None | Timing helper |
| **`spine`** | GeoDataFrame \| None | Primary entity being built (index name = spine_id_col, e.g. `footprint_id`) |
| **`references`** | dict[str, GDF] | Reference datasets keyed by resolved recipe_id |
| **`crosswalks`** | dict[str, GDF] | Spine ↔ reference join tables (MultiIndex for polygon overlays, flat for point joins) |
| **`overlays`** | dict[str, GDF] | Full geometry-bearing polygon overlay results |
| `reference_types` | dict[str, str] | recipe_id → entity_type string |
| `source_geometry_types` | dict[str, SGT] | recipe_id → SourceGeometryType (used to detect dwelling/building evidence) |
| `simplified_geometry` | GeoSeries \| None | Set by simplify_geometries step |
| `metadata` | dict | Arbitrary intermediates (inferred footprints, source assignments, …) |

Helper methods: `get_crosswalks_by_type(entity_type)`, `get_references_by_type(entity_type)`.

---

## Crosswalk Data Structures

### Polygon reference crosswalk (spatial_overlay)
- **Index**: MultiIndex `[spine_id_col, reference_id_col]` (e.g. `footprint_id`, `parcel_id`)
- **Columns**: `link` (quality label), `area_intersection_m2`, `iou`,
  `area_intersection_m2_inner`, `fraction_of_largest`

### Point reference crosswalk (spatial_point)
- **Index**: Point reference's native index
- **Columns**: All reference columns + `spine_id_col` showing which spine entity each point was matched to (null = unmatched)

---

## Suffix Naming

Each evidence column carries a provenance suffix. The suffix includes exactly the
components that disambiguate, which is **intentional, not an inconsistency**:

- **Point / building-level refs** include **entity-level + source**:
  `_point_suffix()` → `_{entity_type}_{source_id}` (e.g. `_building_nsi`,
  `_dwelling_overture`). The entity level is kept because the three building entities
  (footprint / building / dwelling) are routinely conflated; the source is kept
  because point sources differ in meaning.
- **Polygon / parcel refs** include **entity-level only**:
  `_resolve_suffix()` → `_{entity_type}` (e.g. `_parcel`). Parcel layers are
  interchangeable, so the source is deliberately omitted. (If the reference entity
  type equals the spine entity type it falls back to source_id.)

So the same feature from two sources yields distinct columns:
`occupancy_type_building_nsi`, `purpose_subgroup_parcel`.

**Relational counts** use `n_{counted}s_per_{grouping}` (totals, include self):
`n_parcels_per_footprint`, `n_footprints_per_parcel`. **Source counts** keep the
entity stem + source: `n_buildings_nsi`, `n_dwellings_overture`.

**Output ordering** is rule-based (no explicit list): the curate `order_columns`
step derives the order from the provenance suffix + the attribute registry `sort`
rank. Phases: identity → source-evidence (by source: count → id → attribute) →
canonical → metrics → inferred → occupancy finals.

---

## Pipeline Steps

### A. PREPROCESS

#### `resolve_spine` (spine.py)
Build the spine from prioritized sources via IoU deduplication.

**Recipe params** — `sources` list (each: `recipe_id`/`label` or `auto_discover: true` +
`entity_type`), `thresholds`:
- `min_area_m2` — drop geometries below this before merging
- `overlap_iou_max` — two footprints with IoU > this are considered duplicates
- `elongated_aspect_min/angle_tol/long_overlap_min/lateral_sep_ratio` — elongated-duplicate
  filter (catches sideways-displaced trailers)

**State**: writes `spine` (`geometry`, `source` columns).

---

### B. ATTRIBUTE SOURCES TO SPINE

#### `link_to_reference` (links/spatial.py, points.py, sidecars.py)
Load a reference dataset and build a spine ↔ reference crosswalk.

**join modes**:

**`spatial_overlay`** (polygon-on-polygon):
1. Load reference, aggregate by `geo_id` using attribute registry functions
2. Run `overlay_polygons(how='identity', iou=True)` — or, with `save_link`
   (default **on**; opt out with `save_link: false`), reload the persisted
   link sidecar instead when its footer fingerprint (format 2: step config
   + the ordered configs of every prior geometry-phase pipeline step +
   size/mtime of the recipe's ingest inputs; tombstone receipts stand in
   for deliberately deleted inputs) still matches and `state.reprocess` is
   False. A reloaded overlay is geometry-free (only area/IoU columns are
   consumed downstream).
3. Build the trimmed crosswalk via `_build_crosswalk` (shared by fresh and
   reload paths); thresholds:
   - `min_fraction_of_largest` (default 1/6): drop secondary links below this fraction
   - `area_intersection_m2_min` (default 10 m²): minimum intersection to keep
   - `snap_chains` (default false) + `chain_fraction_max` (default 0.75):
     apply `snap_chained_links` after the trim — a multi-parcel footprint
     collapses to its dominant parcel (link label
     `'unique parcel (snapped from chain)'`) when every minor link's parcel
     is a *different* footprint's dominant/unique parcel (chain-displaced
     footprint layers; a genuine shared row-house footprint never qualifies)
     and each minor `fraction_of_largest <= chain_fraction_max`. Geometry-free,
     so it runs identically on the reload path; deliberately NOT part of the
     sidecar fingerprint (toggling it never forces an overlay recompute).
4. With `save_link` (default on): write the geometry-free FULL n:m overlay (every
   pair incl. sub-threshold slivers, crosswalk `link` label left-joined,
   null = trimmed-out) to the canonical entity-link path
   (`get_entity_link_path`, beside the finer entity's output). Consumed by
   the curate stage's `collect_link_ids` (parcel_id_all) and
   `apportion_curated_values` (both filter `link.notna()`). Rewritten on the
   reload path too, so stored labels always match the current crosswalk.
   With snapping enabled, a `link_chain` column records the adjustment:
   `'snapped minor'` on removed pairs (their `link` is null, excluded from
   attribution/apportionment like slivers) and `'snapped dominant'` on the
   promoted 1-1 link.
5. Writes: `references`, `crosswalks` (MultiIndex), `overlays`, `reference_types`, `source_geometry_types`

**`spatial_point`** (point-in-polygon, Lochhead Table 3 four-pass; with
`save_link`, default on, the final flat crosswalk is persisted
geometry-free at the entity-link path and reloaded on later runs while
its fingerprint matches, skipping every pass below):
0. Optional pre-link duplicate resolution: `thresholds.resolve_duplicates`
   (`key`: any ref column, default `building_id_ubid`; `'olc'` = the computed
   ~3 m `_olc` location cell; `ignore_sources`: e.g. `[ESRI, HAZUS/NSI-2015]`)
   runs `flag_duplicate_points` — within groups sharing the key, low-rank
   sources colocated with a higher-level record get
   `duplicate_resolution = 'colocated low-rank source'`. Rows are flagged,
   never dropped; the label rides through every pass onto the crosswalk, and
   `_attribute_point_reference` excludes flagged rows from ALL aggregates at
   the merge (occupancy/group picks, value sums, counts, collected ids).
1. Pass 1 — `sjoin(predicate='within')`
2. Pass 2 — `sjoin_nearest()` up to `proximity_m` (default 10 m)
3. Pass 3 — `sjoin_nearest()` up to `far_proximity_m` (default 100 m), constrained to same parcel
4. Pass 4 — `sjoin_nearest()` up to `unbounded_proximity_m` (default 0 = disabled)
5. Optional: size-limit filter (drop area outliers by occupancy class)
6. Optional: `aggregate_multipoint` — collapse multiple points per footprint into one row
7. Writes: `references`, `crosswalks` (flat), `reference_types`, `source_geometry_types`

---

#### `infer_spine_additions` (links/additions.py)
Add spine entries from parcel-only coverage (no linked footprint, high improvement value).

Thresholds: `n_per_group_min` (mean footprints/parcel ≥ this), `value_per_ha_quantile`
(improvement_value_per_ha ≥ this quantile).

Writes: `spine` (appended inferred footprints with `source = '{entity_type}.{source_id}'`),
`metadata['inferred_from_{recipe_id}']`.

---

#### `resolve_overlaps` (links/overlaps.py)
`clean_polygons()` then `resolve_overlapping_polygons()`. Writes: `spine`.

---

#### `classify_footprint_priority` (attributes/priority.py)
Assign `priority_on_parcel` (primary / secondary / unknown) per parcel.

**Seed**: all parcel-linked footprints → `'primary'`; unlinked → `'unknown'`.

**Multi-footprint parcels** (Lochhead Table 4):
- If parcel has dwelling-linked footprints → those are `'primary'`, others `'secondary'`
- Else if parcel has building-linked footprints (NSI) → those are `'primary'`, others `'secondary'`
- Else → all `'secondary'`
- Unlinked footprints with dwelling evidence → promoted to `'primary'`

Writes: `spine['priority_on_parcel']` (Categorical).

---

### C. ATTRIBUTE EVIDENCE (no value selection)

#### `reconcile_attributes` (attributes/reconcile.py)
Aggregate reference columns to the spine as source-suffixed evidence columns.
Attribution only — between-source value selection moved to the curate stage
(`reconcile_values` in `openplaces.io.curator.reconcilers`).

**Recipe params**:
- `sources` — list of `{recipe_id or entity_type, columns}` dicts

Dispatches to `_attribute_polygon_reference` (MultiIndex crosswalk) or
`_attribute_point_reference` (flat crosswalk).

**Declared columns always appear.** A point reference writes every column
the source entry declares, null where the reference carried nothing for a
row or lacks the column entirely; a reference with zero rows for the admin
unit (no crosswalk at all) still gets its columns and a 0 match count, via
`_attribute_absent_point_reference`. This follows the enricher's contract:
a missing declared column reads as a recipe error, not as a coverage gap,
so one county's spine cannot ship a narrower schema than its neighbors'.
An absent *polygon* reference warns instead, since its evidence is derived
from the reference geometry and cannot be named without loading it.

---

##### `_attribute_polygon_reference` (attributes/polygon.py)

Key computed columns (suffix = e.g. `_parcel`):

| Column | Formula |
|--------|---------|
| `overlap_fraction{suffix}` | identified_area / footprint_area |
| `n_parcels_per_footprint` | count of parcels under the footprint (total) |
| `n_footprints_per_parcel` | count of footprints on the parcel (total, incl. self) |
| `address{suffix}` | from the largest-intersection row |
| `land_value{suffix}` | from the largest-intersection row, but **only for primary footprints** (`priority_on_parcel == 'primary'` if available, else `n_other == 0`) |
| `improvement_value{suffix}` | parcel value × area_fraction (sum) |
| `n_dwelling_units{suffix}` | parcel units × area_fraction (sum) |
| `year_built{suffix}` | mean across intersections |

The value rows above (`address`, `land_value`, `improvement_value`,
`n_dwellings`, `year_built`) are computed by the **shared apportionment**
`apportion_reference_values` in `apportion.py` (`area_fraction` =
intersection_m2 / sum per parcel, or volume-weighted when
`use_volume_weight=True` and an `n_stories*` column exists). The curate stage
(`apportion_curated_values` in `io/curator/evidence.py`) calls the same
function on the persisted link sidecar with *curated* reference values, so the
two stages cannot drift. The footprint spine recipe no longer requests parcel
value columns at harmonize (they'd be empty pre-roll-merge); curate attributes
them instead.

**Lochhead Table 4 (dwelling suppression)**: when a parcel has ≥1 dwelling-linked
footprint, zero out `area_fraction` for footprints **without** dwelling evidence so they
receive no `improvement_value`/`n_dwelling_units`. Also suppresses `land_value` for those
IDs. (Implemented in `apportion_reference_values`.)

**Inferred footprints**: attributed separately from `metadata['inferred_from_{rid}']`
with direct column assignment (no area weighting).

---

##### `_attribute_point_reference` (attributes/point.py)

Key columns (suffix = e.g. `_building_nsi`):
`n_{entity_type}s_{source_id}` (e.g. `n_buildings_nsi`; for dwelling/overture the
units column is `n_dwellings_overture`), `purpose_subgroup`, `purpose_subgroup_all`,
`group`, `group_all`, `structure_value`, `year_built`, `n_dwelling_units`,
`building_id` (when `collect_ids: true`).

Polygon-point bridge: when both polygon and point refs exist, infer `group{poly}`
(e.g. `group_parcel`) from parcel's `purpose_group_combined` using a majority-vote
lookup per purpose group.

---

### D. (moved to curation)

Gap-filling, derived metrics, and occupancy inference no longer run in harmonize.
They are curate steps now (`openplaces.io.curator`): `derive_metrics` (m2,
`*_per_area`), `infer_group_combined` (→ `group_parcel_building_nsi_inferred`),
`impute_n_dwelling_units`, `infer_occupancy_type`. `_OCC_UNITS`/`reverse_occ_units`
remain in `attributes/_shared.py` (still used by `links/points.py` `_aggregate_multipoint`); the
curate `impute_n_dwelling_units` imports `_OCC_UNITS` from there.

---

## CHEER Recipe Walkthrough

Harmonize spine (`US_footprint-spine-2026.yaml`) — evidence only:
```
A. resolve_spine             OBM → Microsoft → state-specific → FEMA (IoU dedup, min 10 m²)
B. link_to_reference         parcel, spatial_overlay (min_fraction=1/6, min_area=10 m²)
   infer_spine_additions     parcel-only → footprint geometry (n_per_group≥0.2, q5 value)
   resolve_overlaps          clean geometry overlaps
   link_to_reference         NSI, spatial_point, 3-pass (10 m / 100 m parcel-constrained)
   link_to_reference         Overture dwellings, spatial_point, aggregate_multipoint=True
   classify_footprint_priority  primary/secondary/unknown via dwelling > building evidence
C. reconcile_attributes      attribution only → *_parcel / *_building_nsi / *_overture cols
```

Curate (`US_footprint-cheer-2026.yaml`):
```
reconcile_values         priority pick (n_dwelling_units, year_built, improvement_value)
derive_metrics           m2, *_per_area
infer_group_combined     group_parcel + group_building_nsi → *_inferred
impute_n_dwelling_units  fill nulls from occupancy via _OCC_UNITS
infer_occupancy_type     occupancy_type cascade (NSI → group_parcel → geometry → units → role)
merge_enrichments        roof_shape, n_stories (BRAILS evidence)
resolve_occupancy        parcel keyword ruleset overrides NSI (reviewed only)
refine_occupancy_height  occupancy_type_cheer HAZUS bands
cast_categoricals        registry-driven Categorical dtypes
order_columns            final output schema/order
```

---

## State Mutation Summary

| Step | Writes |
|------|--------|
| resolve_spine | `spine` |
| link_to_reference (overlay) | `references`, `crosswalks` (MultiIndex), `overlays`, `reference_types`, `source_geometry_types` |
| link_to_reference (point) | `references`, `crosswalks` (flat), `reference_types`, `source_geometry_types` |
| infer_spine_additions | `spine` (appended), `metadata['inferred_from_{rid}']` |
| resolve_overlaps | `spine` |
| classify_footprint_priority | `spine['priority_on_parcel']` |
| reconcile_attributes | `spine[*_suffixed evidence columns]` |

---

## Key Design Patterns

1. **IoU deduplication** — footprints from lower-priority sources added only if no IoU overlap with the current spine.
2. **MultiIndex crosswalk** — supports 1:N spine-to-reference relationships for value distribution.
3. **Dwelling > building evidence hierarchy** (Lochhead Table 4) — dwelling points suppress parcel values for non-dwelling-linked siblings.
4. **Area-weighted distribution** — `improvement_value` and `n_dwelling_units` split proportionally; `land_value` and `address` assigned to the primary footprint only.
5. **Priority-based column selection** — `.bfill(axis=1)` on suffixed columns selects first non-null across sources (now the curate `reconcile_values` step).
6. **Inferred footprints** — parcel-only coverage creates synthetic spine entries, attributed directly (no area weighting) in `reconcile_attributes`.

---

## Stage overview, property ids and link tables

<!-- Moved verbatim from AGENTS.md on 2026-09-28. Parts overlap
the sections above; deduplicate when next editing either. -->

**Stage 2 — Harmonize** (`io/harmonizer/`):

`Harmonizer` runs a composable step pipeline, executing steps declared in the recipe's
`pipeline` list. Each step is a function registered via `@_register('step_name')` in one
of the sub-modules. Registrations carry a `phase` tag (`'geometry'` for steps that
mutate spine rows/geometry or run spatial joins; `'attributes'`, the default, for
steps that only read or annotate) — the tag drives the link-sidecar fingerprint and
defines the geometry/attribute recipe split below.

**The geospine split.** Each expensive spine is two recipes. A *geospine* recipe
(`US_footprint-geospine-2026`, `US_parcel-geospine-2026`) runs the geometry phase —
spine resolution, spatial overlays and point joins, group detections, morphology —
and persists its spine plus one n:m link sidecar per `link_to_reference` join
(`save_link` is default-on; a step opts *out* with `save_link: false`). The
*attribute* recipe keeps the established id (`US_footprint-spine-2026`,
`US_parcel-spine-2026`), declares the geospine as its `entity_recipe`, starts with
`load_geospine`, and writes an attribute-only table (`save_to: geometry: false`);
its geometry lives with the geospine and is resolved by `get_entities` through the
`entity_recipe` chain — by declaration, never by probing for a `_geo` sidecar, so a
stale pre-split sidecar is ignored. Enricher and curator load their entity spine
through `get_entities` for the same reason. Rerunning an attribute recipe involves
no spatial computation (`--reprocess attributes` in the driver). Each sidecar's
footer fingerprint (format 2) covers the step config, the configs of every *prior
geometry-phase* pipeline step, and size/mtime of the ingest inputs, plus a
per-source content sha256 stamped at write time: an input whose mtime moved but
whose bytes did not (a sync-tool touch; Dropbox re-hydration bumped every cache
mtime on 2026-08-24) revalidates through the hash instead of forcing a geometry
rerun (`_fingerprints_match`). Anything else fails closed — a stale sidecar raises with instructions
to rerun the geospine, never silently recomputing geometry. Curate readers resolve
the sidecar's owner through `geo/link.get_link_owner_recipe_id` (the geospine when
split, the recipe itself when not). Geospine recipes declare
`save_to: retention: keep`: their outputs and link tables are the normalized
geometry store, exempt even from aggressive cleanup (an explicit retention wins
over the aggressive core-bucket demotion). Bucket policy: `core` holds the
normalized store and intermediate evidence; terminal curate outputs ship
self-contained (attributes + geometry) in `share`.

**The building spine** (`US_building-geospine-2026`, `US_building-spine-2026`,
slices b1 and b2 of `plans/core-schema-and-stage-contracts-review.md`,
2026-09-29). The building is the structure entity; the footprint is one
geometry source for it and today the only one. The building geospine is
`resolve_spine` over the footprint geospine alone (`- auto_discover: false`,
`geometry_source` kept from the row rather than stamped), `adopt_source_entity_id`
(the footprint's id kept in `footprint_id`, the index renamed `building_id`,
equal values while one outline is one building) and the geometry attributes.
The building spine loads it with the footprint geospine's link sidecars
re-keyed to building ids (`load_geospine` with `links_recipe_id`), writes the
footprint-to-building link table (`link_entities_by_id` against the footprint
geospine, method `footprint_outline`), and runs the parcel priority, evidence,
address, postal and permit steps that ran on the footprint spine until b2; the
relational counts take the building's name (`n_parcels_per_building`). The
footprint spine is then `load_geospine` plus `adopt_entity_attributes` of the
building spine with the counts renamed back, so its output is unchanged cell
for cell and its consumers (the parcel geospine's morphology and
dwelling-address steps, the imagery recipes, the footprint curate, the
validation references) are untouched; the projection goes when they read the
building spine. Order per unit: footprint geospine, building geospine,
building spine, footprint spine, parcel geospine. Slice b3 (townhome splits,
condo-tower merges, NSI points with no outline as rows) is where a building
stops being one footprint; the re-keying and the projection both refuse rows
that share a source id, so that slice has to decide the fan-out explicitly.

The step sub-modules:
- `spine.py` — build/merge the primary entity GeoDataFrame (`resolve_spine`),
  and assign each row's containing polygon id from a space-partitioning
  reference layer such as an admin level or a Census statistical geography
  (`link_geographic_ids`; unlike `link_to_reference` below, every configured
  reference is assumed to tile space without overlaps, so the relationship
  is always exactly one containing polygon, not a many-to-many crosswalk).
  An `inherit_from` option rolls up an already-linked recipe's own output
  (e.g. a parcel spine inheriting from its footprint spine, which runs
  first) wherever every row in the group agrees, so the direct spatial join
  only runs on the residual.
- `links/` — join to reference datasets, one module per concern since
  2026-09-29 (until then one 4,259-line `links.py`): `spatial.py`
  (`link_to_reference`, the overlay crosswalk), `points.py` (the point
  join), `sidecars.py` (fingerprints and sidecar files), `by_id.py`
  (`link_by_id`), `discovery.py` (auto-discovered sources and remap
  sidecars), `combine.py` (prioritized writes onto the spine),
  `address_ranges.py`, `additions.py` (`infer_spine_additions`),
  `condo_clusters.py`, `overlaps.py` (`resolve_overlaps`) and `_shared.py`
  (constants). The package file re-exports every name, private ones
  included, and is a plain package: a test patches the submodule a step
  lives in, or every submodule binding the name through
  `tests/links_patching.patch_links` (the forwarding class that stood in
  for that between 2026-09-29 and 2026-09-30 is gone). The step loader
  imports the package explicitly, because the shared loader skips
  sub-packages.
- `load.py` — restore a geospine recipe's spine, crosswalks, overlays, and
  prepared references from its persisted output and link sidecars
  (`load_geospine`); which links to restore is read from the geospine
  recipe's own pipeline, so the two YAMLs cannot drift. A spine built
  from another entity's geospine has no links of its own:
  `links_recipe_id` names the geospine whose sidecars apply, and they
  are restored keyed by that entity's id and re-keyed to this spine
  through `links_key` (the building spine reads the footprint geospine's
  parcel, NSI and Overture links through `footprint_id`; a rename while
  the relation is one to one, refused where rows share a source id).
  `adopt_entity_attributes` is the reverse projection: every column of
  another spine's output copied onto this one through a one-to-one key,
  overwriting in place and appending the rest in the other's order, with
  explicit renames for counts named after the other entity. The
  footprint spine is such a projection of the building spine (below).
- `attributes/` — attribute source columns to the spine as suffixed evidence
  columns (`reconcile_attributes`, in `reconcile.py` with the polygon and
  point attribution in `polygon.py` and `point.py`), assign each
  footprint's parcel priority (`classify_footprint_priority`,
  `priority.py`), build the combined land-use label the parcel classifier
  votes on (`derive_use_classes`, `use_classes.py`), summarize footprint
  morphology (`morphology.py`), detect shared-land and condo clusters
  (`shared_land.py`, `condo_clusters.py`), estimate property counts and
  attribute dwelling addresses; one package since 2026-09-29 (until then
  a 2,615-line module), every name re-exported by the package file, the
  step loader importing it explicitly. `reconcile_attributes`'s `columns` list is
  ordered, and each entry is either a column or a list of alternatives
  **coalesced per row** — the default is `[use_group, use_group_code]` then
  `[use_subgroup, use_subgroup_code]`, so a source whose raw code has no
  crosswalk still contributes that code as grouping and voting evidence,
  while a source whose crosswalk did fire contributes only the vocabulary
  and adds no new label values (so cohort statistics keyed on
  `use_group_combined` are not fragmented). This matters more than it
  looks: 20 recipes map a `use_*_code` but only 5 ship a crosswalk for it.
  The parcel spine appends `building_style` last so a county whose
  land-use text carries no occupancy signal can contribute one from its
  structure description.
  Value selection, gap-filling, and occupancy inference now run in the
  curation stage, not here; the harmonized spine (`US_footprint-spine-2026`)
  is an evidence-only table.
- `addresses.py` — reconcile a canonical street address from any number of
  source inputs (`reconcile_addresses`); coalesce ZIP-code evidence from
  multiple columns by priority (`reconcile_postal_code` — e.g. an
  address-parsed ZIP, then the spatially-derived `zcta5_id`, so a state with
  no address-parsed coverage at all still gets a ZIP-like value); and derive
  the USPS-preferred city for a ZIP code (`impute_postal_city`).
  - These are a deliberate exception to the "gap-filling belongs in curate" rule: the
    dividing line is whether there is dispute about how to derive its output. A ZIP code
    has exactly one USPS-preferred city — nothing parameter-sensitive to defer — so it's
    resolved once, in harmonize, where it's available early for downstream linking.
- `diagnostics.py` — conflict-inspection reports (agreement rates, a
  crosstab of disagreeing value-pairs, a bounded sample of conflicting rows)
  written to the cache when `save_statistics` is set, mirroring
  `io.curator.diagnostics`'s established convention. Never raises, never
  changes the harmonized spine.
- `filter.py` — subset rows (`filter_entities`)
- `discover.py` — discover available data sources for an admin unit
- `admin_names.py` — `assign_admin_id_from_name`: the admin unit a record
  only *names*, as an id, matched exactly and uniquely against the units of
  its county or not at all. It exists because an address key is scoped by
  `admin4_id`: a source with an empty scope matches no parcel by address,
  silently (0 of Vilas County WI's keyed returns, 0.759 with the scope
  ignored). The source's spelling is a recipe-side regex; a no-op where the
  level does not exist (New England).
- `link_methods.py` — `record_link_method`: after each `link_by_id` pass of
  the transaction spine, writes `parcel_link_method`, the name of the first
  rule that reached the row (`parcel_id_local`, then an exact address key).
  A fixed label, written once, never a score and never revised (patent
  shape 4). It re-reads the pass's reference instead of instrumenting
  `link_by_id`, so a step added between a pass and its label would make the
  label describe the wrong pass. A sale that reached its parcel through a
  unit's lot reads `stacked_units` (`via_column`/`via_label`).
- `parcel_link_keys.py` — `derive_parcel_link_key`: the one column
  (`parcel_link_key`) every parcel join of the transaction spine uses. A
  deed for a condo unit states the unit's number, which after the
  ingest-time split is on a property row, not a parcel row; the step looks
  it up in the split's own (unit, lot) pairs and joins on the lot
  (`lot_id_local`), never rewriting the stated number. A unit seen on
  several lots keeps its own number: picking one would assert which parcel
  the sale conveyed, which the record does not say.
  Without the step a re-ingested county loses its condo sales silently
  (Vilas County WI: 1.1% of returns).
- `transactions.py` — the steps that make the transaction spine's rows the
  entity (one recorded sale) and mint their ids, at the end of
  `US_transaction-spine-2026` after every parcel link, in this order:
  `derive_sale_period`, `dedup_transactions` (exact repeats of one document
  from overlapping source windows; 15.4% of Lake County FL's rows, all
  deeds, measured 2026-09-29), `derive_document_id`,
  `count_parcels_per_document`, `aggregate_multi_parcel_sales` (one row per
  deed, extensive columns summed, the rest from the heaviest parcel) and
  `assign_transaction_ids` (the document scoped by the admin unit,
  `US-FL-LA_<document>`, a content fingerprint where no document is named,
  a suffix only where two rows are still named alike). The fold precedes
  the ids so a deed's id carries no suffix. The curate stage loads this
  index and never re-mints; the curate steps of the same names remain as
  wrappers over the shared frame functions in `io/sale_records.py`. Moved
  from `US_transaction-openplaces-2026` on 2026-09-29 (the audit's decision
  that transaction ids are minted in the spine, like property ids). Two
  curate indicators changed base by the move: the nominal price floor and
  the disclosure share are measured over deeds now, not over deed-parcel
  rows.
- `last_sales.py` — `append_last_sales`: turn the last-sale fields an
  assessment roll carries into transaction rows
  (`sale_record_kind = assessor_last_sale`; rows already on the spine read
  `deed`), so a state with no recorder's feed still has prices. It is a
  reshape, not a match: the row keeps the roll row's own `parcel_id_local`.
  Three things are not what the name suggests. **"Has a last sale" is never
  "non-null"**: Florida's parcel layer fills every last-sale column on every
  row and 0 is the placeholder, in the year as well as the price, so a row
  needs a price and a real date or year. **A roll that records the month
  only arrives as first-of-month dates** (all 73,253 of Pitt County NC's);
  the step keeps year and month and writes no day, so a deed and its
  last-sale echo compare at month precision. **A roll ingested for several
  years repeats each last sale once per year** (Florida's DOR roll holds 24);
  the same ids, date and price are one row, while different last sales of
  one property across years are kept, which is history a single roll year
  lacks. Property tables are read before parcel tables, and a later source
  adds a sale only for a `parcel_id_local` no earlier one supplied. Only an
  allow-list of columns is read, so owner fields cannot enter. A last-sale
  field is the most recent sale only: these rows support cross-sectional
  work, not a repeat-sales index.

All steps share a `HarmonizeState` dataclass (spine, references, crosswalks, overlays,
metadata). Each step receives state and returns the updated state.

Public entrypoint: `harmonize(recipe, admin_ids, reprocess, verbose)`.

**A property's id is the account number its assessor issued**, prefixed with
the admin unit that scopes it: `US-TX-VIC_000123`
(`io/harmonizer/entity_ids.py`, step `assign_entity_ids` in
`US_property-spine-2026`). The number comes from the column a source's
recipe names in `entity_id`, else whichever of `property_id_assessor`,
`property_id_admin2`, `parcel_id_assessor` repeats on the fewest rows: the
*account*, never the lot (New Hanover County NC's account number repeats on
no row, its PIN on 19,519). Nothing raises: a source whose best column still
repeats on most rows issues no account number and is named by content, with
a warning. A source holding several `tax_year` values keeps only its latest
roll before ids are minted (Florida's DOR roll is 24 yearly rolls per
county, kept at ingest as a panel; Lake County's 3,952,270 rows are 210,057
properties), the unit's latest year and not the latest row per account,
which would revive every account retired since. Case is folded and runs of separators become one hyphen, but their
positions are kept, because in a map-block-lot number they carry meaning
(dropping them made 426 Somerville MA accounts collide). Two sources
publishing the same accounts therefore mint the same ids, and **rows sharing
an id merge into one**, the first-loaded (most specific) source winning each
cell and `source` becoming `a+b`: Boston's city table and MassGIS's layer are
364,997 rows and 185,132 properties. The few differing rows under one number
get `_2`, `_3` ordered by content, and a source with no issued number is
named `{admin}_{source}:{content hash}`. A spatial entity's id comes from its
geometry; this is the equivalent for an entity that has none.

**Relationships between entities are link tables** (`geo/link.py`,
`io/harmonizer/entity_links.py`), one row per pair, stored beside the finer
entity's output (`get_entity_link_path`, finer by `ENTITY_LINK_ORDER`). The
spatial joins write theirs from an overlay; `link_entities_by_id` writes the
property-to-parcel table from an exact match on `parcel_id_local`, with
columns `property_id`, `parcel_id`, `link_method`, `link_source` and `share`
(a test pins that a link table holds nothing else, so it can never carry a
person). It runs in `US_parcel-spine-2026` because that is the first recipe
in which both sides exist. `link_method` is a fixed label naming the rule
that found the pair, never a score, and no link is removed once written
(patent shape 4). A key on more rows than `link_by_id`'s placeholder cutoff
keeps its pairs under `parcel_id_local_shared_key`, because a link table,
unlike a sum, need not decide whether it is a placeholder or a large stack:
a reader that sums values leaves that method out. The footer
(`openplaces:entity_link`) records both input files, so a reader can tell a
link that predates a rebuilt spine. The step sits **after** the parcel
spine's checkpointed step: a restored checkpoint skips every step before
it and validates against the geospine only, so anything placed earlier
that reads another recipe's output goes stale unnoticed (the transaction
`link_by_id` sat near the top of that recipe with this weakness until
2026-09-23; it now follows the link table, and
`tests/recipe/test_parcel_spine_step_order.py` pins both). Curate's
`aggregate_from_entities` reads the table; see the stacked-units paragraph
in `src/openplaces/io/ingester/README.md` for the passes the ingest-time split adds.
