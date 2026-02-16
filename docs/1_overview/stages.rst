.. openplaces

Stages
======

Data analyses are broken down into nine steps:

- **01_configure**: define folder structure, set preferences, lock administrative referencing
- **02_ingest**: download, unzip, and stage data in cache (no filtering, imputation, or edits leading to information loss)
- **03_harmonize**: align entity datasets from multiple sources, create spine of entities.
- **04_enrich**: create features for entity spine (geoprocessing, record linkage).
- **05_curate**: create analysis-ready dataset (select, aggregate, snapshot)
- **06_model**: train, cross-validate, optimize, score
- **07_infer**: create artifacts from models: predictions, aggregated coefficients
- **08_report**: creation of publication-ready figures, tables, and text
- **09_show**: interactive display of results, demonstrating quality, issues, or functionality
