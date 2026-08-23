.. openplaces

.. _package_state:

Package state
=============

Technical status, scope, and configuration of the ``openplaces`` codebase.

Implementation scope
~~~~~~~~~~~~~~~~~~~~

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

* **Version**: ``0.1.0``
* **Python compatibility**: ``>=3.12``
* **License**: ``Apache-2.0``

Architecture layers
~~~~~~~~~~~~~~~~~~~

The codebase is organized as a layered architecture. Modules in higher-numbered layers may only import from lower-numbered layers:

* **Layer 0**: ``core`` (schema, constants)
* **Layer 1**: ``config``, ``path``, ``diagnostics``
* **Layer 2**: ``recipe`` (recipe definitions and parser)
* **Layer 3**: ``io/__init__``, ``io/consent``, ``geo/address``
* **Layer 4**: ``io/readers``, ``table``
* **Layer 5**: ``geo/*`` (spatial joins, IDs, geometry helpers)
* **Layer 6**: ``io/ingester/*``, ``io/scrapers/*``, ``io/aggregate``, ``io/admin``, ``io/delivery``, ``io/transform``, ``io/cleanup``
* **Layer 7**: ``io/harmonizer``
* **Layer 8**: ``io/enricher``
* **Layer 9**: ``io/curator``
* **Layer 10**: ``viz/*`` (visualization)
* **Layer 11**: ``api.py``
* **Layer 12**: ``flow/*`` (runners, submitters)

Testing
~~~~~~~

The test suite validates correctness across layers:

.. code-block:: bash

   conda activate openplaces
   pytest
