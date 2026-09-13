.. openplaces

.. _package_state:

Package state
=============

Technical scope and versioning of the active ``openplaces`` codebase.

Package scope
~~~~~~~~~~~~~

The package implements the core **data construction pipeline**, from source download to a shareable curated dataset. It does not include downstream statistical analysis or modeling:

* **Implemented**:
  
  * **Ingest**: Automated downloading, partitioning, and standard parquet schema formatting of source datasets, driven by recipes that also record each source's terms.
  * **Harmonize**: Spine resolution per entity, spatial and key-based links between entities (footprints, parcels, properties, transactions, administrative units), address standardization, and the geospine split that keeps geometry apart from attributes.
  * **Enrich**: Evidence keyed to an entity from imagery classifiers, raster statistics (zonal and neighborhood coverage), and reference inventories.
  * **Curate**: One dataset per entity type: value reconciliation, rule-based inference (e.g., occupancy voting), imputation with provenance, aggregation across entities (a parcel's rooms from its properties, one row per recorded sale from deed records), and delivery bundles for named regions with a terms notice.
  * **Orchestration**: A recipe graph that derives one job per stage, recipe and administrative unit, run through Snakemake on a workstation or a cluster.
  * **Extension points**: A public API for reading curated entities, administrative units and recipe metadata, and recipe roots that an installed package can contribute, so that a downstream package builds on ``openplaces`` without forking it.

* **Not implemented**:
  
  * Downstream statistical analyses, valuation modeling, or academic reference simulations. No analysis pipelines are run on top of the curated database; a valuation package is developed separately against the public API.

Metadata
~~~~~~~~

* **Version**: |release|
* **License**: ``Apache-2.0``
