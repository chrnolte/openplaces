# QGIS map test fixtures

- `tiny_template.qgz` — a minimal, synthetic QGIS project used by
  `test_generator.py`. Not authored in the QGIS GUI; built by
  `build_tiny_template.py` so it's regeneratable and diffable. Regenerate
  with:

  ```
  python build_tiny_template.py
  ```

  It has one prototype per `style_registry.csv` row: a joined attr/geo pair
  for `test-output` (footprint), a single-file (no attr) prototype for
  `test-output-combined` (parcel, exercising `LayerSpec.combined`), a joined
  pair for `test-input` (footprint/obm) and `test-admin` (admin), a
  standalone `basemap-open` prototype (always kept, unpruned), and a joined
  `_fallback` prototype for the graceful-degradation path.

- `style_registry.csv` — a small registry matching `tiny_template.qgz`'s
  layer names, passed to `style_registry.get_style(..., registry=...)` in
  tests instead of the production
  `viz/qgis_map/qgis_map_style_registry.csv`, so fixture changes don't
  require touching production data and vice versa.

Both are deliberately minimal: most metadata QGIS itself writes (symbology,
CRS blocks, field definitions, ...) is omitted since
`openplaces.viz.qgis_map.generator` only reads/writes the elements
documented in its own module docstring (`<maplayer>`, `<vectorjoins>`,
`<layer-tree-group>`/`<layer-tree-layer>`,
`<legend>`/`<legendgroup>`/`<legendlayer>`, `<mapcanvas>`). They are not
representative of a real, QGIS-GUI-authored template and should not be used
as an example for authoring the real one — see
`src/openplaces/qgis/templates/README.md` for that.
