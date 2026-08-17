# Writing a harmonize recipe

`stage: harmonize` builds a **spine**: one row per entity, with evidence from
every source attached under provenance-suffixed names. A harmonized entity is
deliberately *evidence-only* — it does not pick winners. Selecting a canonical
value, filling a gap, or inferring an attribute all belong in curate.

Read [`../README.md`](../README.md) for naming and style. When changing
`src/openplaces/io/harmonizer/` itself rather than a recipe, also read the
harmonizer pipeline reference noted in `AGENTS.md`.

## Shape

A harmonize recipe is an ordered list of registered steps sharing a
`HarmonizeState`:

```yaml
stage: harmonize
admin_id: US
entity:
  entity_type: parcel
  source: {source_id: spine}
  version: 2026

spine_geom: true            # keep geometry for spatial steps and the sidecar
process_by: {admin_level: 3}
save_to: {data_dir: core}

pipeline:
  - step: resolve_spine
    sources:
      - auto_discover: true
        entity_type: parcel
    keep_columns: [parcel_id_local, use_group, land_value, ...]
    track_provenance: [use_group, land_value, ...]

  - step: link_by_id
    auto_discover: true
    entity_type: transaction
    mode: aggregate
    columns: {price: last_sale_price, recorded_date: last_sale_date}
    count_as: n_transactions
    ref_sort_by: recorded_date
    ref_sort_ascending: false
```

## The rules that matter

**`keep_columns` is a whitelist, and forgetting it is the classic bug.**
`resolve_spine` retains only the columns you list from the winning geometry
row. Everything else must arrive via a later `link_by_id`. This is exactly why
`last_sale_price` disappears when `parcel_id_local` is null: the value is not
in `keep_columns`, so the join is its only route, and a broken key drops the
whole roll with no error.

**Auto-discovery is by admin scope and entity type.** `auto_discover: true`
picks up every ingest recipe of that entity type scoped to a child admin unit
of the recipe's `admin_id`. Adding a county recipe therefore changes the spine
for that county automatically — and invalidates everything downstream. Mark a
reference-only source `exclude_from_auto_discover: true`.

**Order is a dependency, not a preference.** The property spine depends only on
property ingests; the footprint spine additionally on parcel *ingests* and the
property spine; the parcel spine on both spines. Derive the order from
`RecipeDAG`, do not guess.

**Evidence keeps its provenance suffix.** Point/building-level refs carry
entity-level + source (`group_building_nsi`, `address_street_dwelling_overture`)
because footprint/building/dwelling are easily confused; parcel refs carry the
entity level only (`_parcel`). Relational counts use
`n_{counted}s_per_{grouping}` (`n_parcels_per_footprint`).

**Do not gap-fill here.** Two documented exceptions exist, and the dividing
line is whether there is any dispute about how to derive the output:
`reconcile_addresses` and `impute_postal_city` — a ZIP has exactly one
USPS-preferred city, so there is nothing parameter-sensitive to defer.
Anything with a threshold, a priority order or a vote belongs in curate.

## Common steps

| step | purpose |
|---|---|
| `resolve_spine` | build the primary geometry, merging sources with IoU dedup |
| `link_by_id` | join a source by a shared key (`attributes`, `aggregate`, `count` modes) |
| `link_to_reference` | spatial join to a reference dataset (points, overlay) |
| `reconcile_attributes` | attach a reference's columns as suffixed evidence |
| `derive_geometry_attributes` | centroid lat/long and area, computed once |
| `rename_columns` | preserve a raw value before a later step overwrites it |
| `summarize_footprint_morphology` | per-parcel footprint shape counts |
| `filter_entities` | subset rows |

`track_provenance:` on `resolve_spine` and `link_by_id` records which source
supplied each cell in a `{column}_source` sidecar. Set it on both when the same
column can arrive from either, or the sidecar starts from a blank baseline.

## After changing one

A new or changed source invalidates the spines that auto-discover it, then
everything downstream of those. Re-run in dependency order and verify the
output against the ingest figures — a spine that silently lost a join looks
identical to one that never had the source.
