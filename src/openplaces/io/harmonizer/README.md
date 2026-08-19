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

#### `link_to_reference` (links.py)
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

#### `infer_spine_additions` (links.py)
Add spine entries from parcel-only coverage (no linked footprint, high improvement value).

Thresholds: `n_per_group_min` (mean footprints/parcel ≥ this), `value_per_ha_quantile`
(improvement_value_per_ha ≥ this quantile).

Writes: `spine` (appended inferred footprints with `source = '{entity_type}.{source_id}'`),
`metadata['inferred_from_{recipe_id}']`.

---

#### `resolve_overlaps` (links.py)
`clean_polygons()` then `resolve_overlapping_polygons()`. Writes: `spine`.

---

#### `classify_footprint_priority` (attributes.py)
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

#### `reconcile_attributes` (attributes.py)
Aggregate reference columns to the spine as source-suffixed evidence columns.
Attribution only — between-source value selection moved to the curate stage
(`reconcile_values` in `openplaces.io.curator.reconcilers`).

**Recipe params**:
- `sources` — list of `{recipe_id or entity_type, columns}` dicts

Dispatches to `_attribute_polygon_reference` (MultiIndex crosswalk) or
`_attribute_point_reference` (flat crosswalk).

---

##### `_attribute_polygon_reference` (attributes.py)

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

##### `_attribute_point_reference` (attributes.py)

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
remain in `attributes.py` (still used by `links.py` `aggregate_multipoint`); the
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
