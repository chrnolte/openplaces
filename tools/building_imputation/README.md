# Building imputation tools

Run scripts for the delivery bundles that impute building attributes
(`share: delivery: impute:` in `US_footprint-cheer-2026`; the
models live in `src/openplaces/io/building_imputation.py`). Run them in
the `openplaces` environment from the repository root.

- `export_and_check.py`: export one or more regions and check each
  bundle (fill shares, NSI-free year and value, no region columns,
  optional check against a restricted source's column notes).
  `--capture` saves the inputs of each public bundle's imputation.
- `leave_county_out.py`: rebuild the delivered model from captured
  inputs and score it with whole counties held out.
- `publish_imputed.py`: copy a cached export to `<region>.imputed`
  beside the region's own share folder. Imputed bundles never go into
  the region's own folder, which main's exports and the value pages
  read. A bundle already there is replaced only with `--archived`,
  after `scripts/storage/snapshot.py` copied it to the archive root.
- `ship_to_drive.py`: copy the `.imputed` bundles to a shared drive
  with rclone, replacing the files of the same name (the drive holds
  one copy); a team bundle needs `--team`.

Order: `export_and_check.py <region> --output-dir cache`, then
`publish_imputed.py <region>`, then `ship_to_drive.py`.

A team bundle reuses its public region's imputation, so export the
public region first.
