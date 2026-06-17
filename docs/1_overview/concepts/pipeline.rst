.. openplaces

.. _pipeline:

Pipeline: from setup to publication
===================================

Notebooks for data processing and analysis (:gh-file:`notebooks`) are grouped into task types:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Stage
     - Description
   * - ``01_setup``
     - Define folder structure, set preferences, lock administrative referencing
   * - ``02_ingest``
     - Download, unzip, and stage data in cache
   * - ``03_harmonize``
     - Align entity datasets from multiple sources, create spine of entities
   * - ``04_enrich``
     - Produce entity-keyed evidence from imagery, models, geoprocessing, or
       record linkage without selecting canonical values.
   * - ``05_curate``
     - Create canonical datasets by reconciling evidence, imputing and
       inferring attributes, and filtering records.
   * - ``06_model``
     - Fit models, cross-validate, derive standard errors, optimize.
   * - ``07_infer``
     - Create artifacts from models: predictions, aggregated coefficients
   * - ``08_report``
     - Creation of publication-ready figures, tables, and text
   * - ``09_show``
     - Interactive display of results, demonstrating quality, issues, or functionality

These notebooks can be converted to scripts and be orchestrated across large computing clusters (to process data entire world regions).

