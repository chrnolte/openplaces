.. openplaces

.. _package_state:

Package state
=============

Technical scope and versioning of the active ``openplaces`` codebase.

Package scope
~~~~~~~~~~~~~

The package implements the core **data construction pipeline** (up to the curation stage). It does not include downstream statistical analysis or modeling:

* **Implemented**:
  
  * **Ingest**: Automated downloading, partitioning, and standard parquet schema formatting of source datasets.
  * **Harmonize**: Spatial point-in-polygon joins, overlay matches, and geospine splits.
  * **Enrich**: Apportionment of census attributes and raster statistical extraction.
  * **Curate**: Rule-based consistency checking, duplicate resolution, and attribute inference (e.g., occupancy voting).

* **Not implemented**:
  
  * Downstream statistical analyses, valuation modeling, or academic reference simulations. No analysis pipelines are run on top of the curated database.

Metadata
~~~~~~~~

* **Version**: |release|
* **License**: ``Apache-2.0``
