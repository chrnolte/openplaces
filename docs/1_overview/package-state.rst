.. openplaces

.. _package_state:

Package state
=============

Technical scope and versioning of the active ``openplaces`` codebase.

Package scope
~~~~~~~~~~~~~

The package implements the core **data construction pipeline**, from
source download to shareable curated datasets. It does not include
downstream statistical analysis or modeling:

* **Implemented**:
  
  * **Ingest**: Automated downloading, partitioning, and standard
    Parquet schema formatting of source datasets, driven by recipes that
    record each source's terms.
  * **Harmonize**: Spine resolution per entity, spatial and key-based
    links between entities (footprints, parcels, properties,
    transactions, and administrative units), address standardization,
    and the geospine split that keeps geometry apart from attributes.
  * **Enrich**: Evidence keyed to an entity from imagery classifiers,
    raster statistics (zonal and neighborhood coverage), and reference
    inventories.
  * **Curate**: One dataset per entity type: value reconciliation,
    rule-based inference (e.g., occupancy voting), imputation with
    provenance, aggregation across entities (such as a parcel's rooms
    from its properties, or one row per recorded sale from deed records),
    and delivery bundles for named regions with a terms notice.
  * **Orchestration**: A recipe graph that derives one job per stage,
    recipe, and administrative unit, run through Snakemake on a
    workstation or a cluster.
  * **Extension points**: A public API for reading curated entities,
    administrative units, and recipe metadata, along with recipe roots
    that an installed package can contribute so that downstream packages
    build on ``openplaces`` without forking it.

* **Not implemented**:
  
  * Downstream statistical analysis, valuation modeling, or academic
    reference simulations. No analysis pipelines are run on top of the
    curated database; a valuation package is developed separately against
    the public API.

Metadata
~~~~~~~~

* **Version**: |release|
* **License**: ``Apache-2.0``
