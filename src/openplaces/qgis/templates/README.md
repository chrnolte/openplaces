# QGIS map template

`openplaces.viz.qgis_map.export_qgis_map` generates a standardized `.qgz`
project for a curate-stage recipe by cloning layers out of a hand-authored
template project — `openplaces_template.qgz`, expected in this directory.
This file is not included in the repository yet; it must be built once in
the QGIS GUI and saved here before `export_qgis_map` can run.

## Authoring contract

For every row in
`src/openplaces/viz/qgis_map/qgis_map_style_registry.csv`, the template must
contain exactly one layer pair, loaded and joined the way
`load_joined_parquet.py` does it (Processing Toolbox → openplaces → "Load
joined openplaces parquet files"):

- a visible geometry layer named exactly the row's `template_layer_name`
- a hidden attribute layer named `{template_layer_name}_attr`, joined to the
  geometry layer on `_join_id`/`geo_id`

Style the geometry layer however you like (categorized/graduated renderer)
— the generator clones the renderer and all styling untouched; it only
rewrites the layer's `<id>`, `<datasource>`, and `<layername>` when
producing a run for a specific recipe + admin unit. Renaming a template
layer is a one-line edit to the registry's `template_layer_name` column,
not a code change.

Group layers under four top-level layer-tree groups — `Buildings`,
`Parcels`, `Admin`, `Basemaps` — mirroring the pre-standard per-county maps
this template supersedes. Within `Buildings`, split further into `inputs`
(ingest-stage per-source layers, registry `role: input`) and `openplaces`
(harmonize/curate output layers, registry `role: output`/`admin`)
subgroups.

Registry `role` values and what they mean for the template:

- `output` / `input` / `admin`: a matched, resolver-driven layer pair (see
  above).
- `basemap` / `static`: a layer that always passes through unpruned,
  regardless of what a given recipe run resolves (e.g. basemap tiles,
  satellite imagery, country-level context layers not tied to any specific
  recipe). These only need `template_layer_name` in the registry — no
  `_attr` sibling is required unless the layer happens to use the same
  join pattern.
- `_fallback` (reserved `style_key`): one generic, unstyled layer pair used
  whenever a resolved layer's `(entity_type, source)` has no registered
  style. Place it under an `Unstyled` group so it's easy to spot.

A row's `default_visible` also controls a `basemap`/`static` layer's checked
state in the generated project (not just matched/cloned layers) — the
generator syncs it at generation time, so changing which basemap is on by
default is a registry edit, not a template rebuild.

A row's `dynamic_categorize_attr`, when set, names a resolved-data column
whose *actual* unique values for a given run should replace this row's baked
categorized-symbol category list, with a random 50%-opacity fill color
stable per category label. Use this for columns whose value set is
data-dependent rather than a fixed enum — e.g. an evidence-conflict summary
column, or a classification crosswalked independently per source — where the
template author's one reference county's category list would otherwise be
wrong or incomplete for any other admin unit. Leave it blank (the default)
for columns with a genuinely fixed, shared vocabulary; the template's own
baked categories are used unchanged.

## Style variants

A single dataset is often worth reviewing through more than one attribute at
once — e.g. a curated footprint layer styled by `occupancy_type` by default,
but also worth flipping through by `roof_shape`, `n_stories`, or
`structure_value`. The registry supports this as **one base row plus any
number of variant rows**, not as separate style keys:

- A **base row** (`variant_of` blank) is the full geo+attr prototype pair
  described above — unchanged.
- A **variant row** (`variant_of` set to the base row's `style_key`,
  `variant_label` set to a short human label, e.g. `"Roof shape"`) is a
  **geometry-only** prototype: style it on a different column, then join it
  — in the template, via `load_joined_parquet.py` same as any other layer —
  to the *base* row's `_attr` layer, not a `_attr` layer of its own. Its own
  `template_layer_name` is whatever you name that prototype layer; there is
  no naming convention to follow for variants beyond being unique in the
  template.

At generation time, the generator clones the base pair as usual, then clones
each variant's geometry layer and repoints its join at the freshly-cloned
attr layer — so a run's variants all end up joined to the same clone the
base layer uses, exactly as they were joined to the same `_attr` prototype
in the template. Variants are inserted as flat siblings of the base layer in
the base row's `group_path` (no extra nesting), and should default to
`default_visible=false` (`checked` off) so a run opens showing just the
default view, with the other column views a click away in the layer tree.

Rows with no meaningful alternate column (e.g. `parcel-output` styled only
on `land_use_class`) simply have no variant rows — variants are opt-in per
style key, not required.

## Other requirements

- **Print layout**: add at least one, with a title expression that
  references the project variables the generator sets on every run —
  `@recipe_id`, `@admin_id`, `@generated_at` — plus a legend, scale bar, and
  north arrow.
- **Basemaps**: include an open, no-API-key basemap (CARTO Positron — flat
  light-gray buildings, unlike OSM standard's high-contrast footprint
  outlines, which visually compete with the recipe's own footprint layer;
  attribution "© OpenStreetMap contributors © CARTO" applies) and the Google
  Satellite/Roads layers. All three use plain key-free XYZ tile endpoints (no
  `cfg.get_credentials()` involved — Google's is the same informal
  `mt1.google.com/vt/lyrs=...` tile trick `io/scrapers/google_satellite.py`
  already relies on for image ingestion). Google Roads is checked on by
  default (`basemap-google-roads` row); the other two are off by default —
  see `default_visible` above.
- **Project CRS**: every layer prototype's `<srs>` and the project's own
  `<projectCrs>` must be set (not just `<mapcanvas><destinationsrs>`) or
  QGIS has no reliable working CRS and layers silently fail to line up with
  the basemap. `build_template_from_reference.py` handles this
  automatically; if a real QGIS GUI ever hand-edits and re-saves this file
  directly, QGIS itself always writes `<projectCrs>` on save, so this only
  matters for anyone scripting the `.qgs` XML directly.
- **One base layer per style key, plus deliberate variants**: don't leave
  ad hoc exploratory copies of a dataset lying around — every extra copy of
  a dataset styled by a different column must be a registered variant row
  (see Style variants above), joined to the base row's shared `_attr`
  layer, not a stray unregistered layer.
- **No one-off rasters**: leave out layers tied to a specific past county
  rather than a registry row or basemap requirement (e.g. farmland/land-value
  composites, Landsat/Sentinel tiles) — those belonged to the ad-hoc
  per-county maps this template supersedes, not the reusable template.
- Save as `.qgz` (not `.qgs`) so the bundled style database travels with
  the project.

## Regenerating after a QGIS upgrade

If layer XML structure changes across QGIS versions, re-save this file from
the newer QGIS and re-verify `generator.py`'s assumptions about
`<maplayer>`, `<vectorjoins>/<join>`, `<layer-tree-layer>`,
`<legendgroup>/<legendlayer>`, and `<mapcanvas>/<extent>`/`<destinationsrs>`
structure still hold (see that module's docstring).
