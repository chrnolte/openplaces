.. openplaces

.. _pipeline:

Pipeline
========

``openplaces`` orchestrates data processing jobs across single machines or high-performance computing clusters.


Stages
------

Data processing and analysis scripts are broken down into nine steps:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Stage
     - Description
   * - ``01_configure``
     - Define folder structure, set preferences, lock administrative referencing
   * - ``02_ingest``
     - Download, unzip, and stage data in cache (no filtering, imputation, or edits leading to information loss)
   * - ``03_harmonize``
     - Align entity datasets from multiple sources, create spine of entities
   * - ``04_enrich``
     - Create features for entity spine (geoprocessing, record linkage)
   * - ``05_curate``
     - Create analysis-ready dataset (select, aggregate, snapshot)
   * - ``06_model``
     - Train, cross-validate, optimize, score
   * - ``07_infer``
     - Create artifacts from models: predictions, aggregated coefficients
   * - ``08_report``
     - Creation of publication-ready figures, tables, and text
   * - ``09_show``
     - Interactive display of results, demonstrating quality, issues, or functionality