.. openplaces

.. _cheer_footprints_annex:

Technical Annex: CHEER footprints
=================================

This data pipeline produces the canonical CHEER footprint inventory, spanning from raw data ingestion to the final curation stage.

.. contents:: Table of contents
   :local:
   :depth: 3


Stage 1: Ingestion
------------------

This stage downloads and extracts raw footprint, parcel, and reference point datasets:

* **Footprints**: Downloads raw building geometries from OpenBuildingsMap (OBM) 2025 (``footprint-obm-2025``), Microsoft v2 (``US_footprint-microsoft-v2``), and auto-discovered state/local GIS footprint layers.
* **Parcels**: Gathers assessor geometry and property tax rolls from state/local GIS agencies.
* **Secondary datasets**: Downloads the National Structure Inventory (NSI) 2022 point database (``US_building-nsi-2022``) and Overture 2025 dwelling address points (``dwelling-overture-2025``).


Stage 2: Harmonization
----------------------

This stage merges geometries and links datasets to build the core entities.

Footprint spine harmonization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Merge source footprints

   * **Explanation**: Combines raw footprint geometries from OBM, Microsoft, and local layers. It filters out shapes below the :input:`min_area_m2` threshold (10.0 m²), resolves duplicates using the :input:`overlap_iou_max` threshold (0.02), and filters parallel-shifted shapes using aspect ratio and angle tolerances via :input:`elongated_aspect_min` (2.5), :input:`elongated_angle_tol` (15.0), and :input:`elongated_long_overlap_min` (0.5) with lateral separation ratio :input:`elongated_lateral_sep_ratio` (2.0). To protect larger structures, it enforces keep-out buffers for features exceeding :input:`buffer_min_area_m2` (250.0 m²) with a baseline width of :input:`buffer_base_m` (2.0 m) scaled by :input:`buffer_area_scale` (0.5).
   * **Function**: :func:`openplaces.io.harmonizer.spine.resolve_spine`

2. Intersect footprints with parcels

   * **Explanation**: Performs a spatial identity overlay between footprints and parcels based on the :input:`join` method (``spatial_overlay``) and :input:`source_geometry_type` (``mixed_type_footprint``) for the :input:`entity_type` ``parcel``. It filters out minor intersections under :input:`area_intersection_m2_min` (10 m²) or slivers less than :input:`min_fraction_of_largest` (0.1667) of the largest parcel intersection, sorting by :input:`sort_by` (``area_intersection_m2``).
   * **Function**: :func:`openplaces.io.harmonizer.links.link_to_reference`

3. Infer synthetic fallback footprints

   * **Explanation**: Generates synthetic footprint geometries for the :input:`entity_type` ``parcel`` using parcel boundaries where tax assessor records indicate a structure exists but no footprint is detected by spatial sources. It uses the thresholds :input:`n_per_group_min` (0.2) and :input:`value_per_ha_quantile` (0.05).
   * **Function**: :func:`openplaces.io.harmonizer.links.infer_spine_additions`

4. Trim overlapping boundaries

   * **Explanation**: Adjusts and trims geometry boundaries to resolve spatial conflicts between synthetic parcel fallbacks and actual detected footprint polygons.
   * **Function**: :func:`openplaces.io.harmonizer.links.resolve_overlaps`

5. Link building structure points

   * **Explanation**: Connects structure-level point evidence from the NSI database using a tiered containment, :input:`proximity_m` (10 m) inner proximity, or :input:`far_proximity_m` (100 m) outer proximity join. The join is configured as :input:`join` (``spatial_point``) for the :input:`source_geometry_type` (``single_building_point``) using the :input:`recipe_id` (``US_building-nsi-2022``) with a remapping crosswalk specified by :input:`remap_id` (``US_building-nsi-2022_occupancy-type-remap``).
   * **Function**: :func:`openplaces.io.harmonizer.links.link_to_reference`

6. Link dwelling address points

   * **Explanation**: Integrates Overture geocoded residential address points from the :input:`recipe_id` (``dwelling-overture-2025``) using a tiered proximity join configured via :input:`join` (``spatial_point``) for the :input:`source_geometry_type` (``single_dwelling_point``). It uses a proximity range from :input:`proximity_m` (10 m) to :input:`far_proximity_m` (50 m), with :input:`aggregate_multipoint` (``true``) and address deduplication enabled via :input:`dedup_addresses` (``true``).
   * **Function**: :func:`openplaces.io.harmonizer.links.link_to_reference`

7. Classify structural role

   * **Explanation**: Determines whether a footprint represents a primary or secondary structure on multi-building parcels based on NSI and Overture matches, using the :input:`entity_type` ``parcel``.
   * **Function**: :func:`openplaces.io.harmonizer.attributes.classify_footprint_priority`

8. Package raw variables

   * **Explanation**: Aggregates all joined source evidence from NSI, Overture, and parcels into intermediate columns on the footprint spine using the configured list of :input:`sources`.
   * **Function**: :func:`openplaces.io.harmonizer.attributes.reconcile_attributes`

Parcel spine harmonization
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Establish parcel boundary baseline

   * **Explanation**: Merges discovered statewide and local parcel geometry layers into a unified spatial spine, automatically discovering sources with the :input:`auto_discover` (``true``) flag for the :input:`entity_type` ``parcel`` and keeping the configured list of :input:`keep_columns`.
   * **Function**: :func:`openplaces.io.harmonizer.spine.resolve_spine`

2. Merge assessor tax records

   * **Explanation**: Discovers and joins county/local assessment tables by matching local ID keys, applying custom attribute remapping crosswalks. It automatically discovers and links tables for the :input:`entity_type` ``parcel`` using :input:`auto_discover` (``true``).
   * **Function**: :func:`openplaces.io.harmonizer.links.link_by_id`

3. Standardize property use codes

   * **Explanation**: Constructs a combined property use description and maps it to normalized use classifications.
   * **Function**: :func:`openplaces.io.harmonizer.attributes.derive_use_classes`

4. Associate building points to parcels

   * **Explanation**: Joins NSI point data from the :input:`recipe_id` (``US_building-nsi-2022``) using a spatial overlay and proximity boundaries configured via :input:`join` (``spatial_point``), :input:`source_geometry_type` (``single_building_point``), :input:`proximity_m` (10 m), and :input:`far_proximity_m` (100 m).
   * **Function**: :func:`openplaces.io.harmonizer.links.link_to_reference`

5. Identify dominant building group

   * **Explanation**: Resolves and summarizes NSI building attributes to find the modal building group per parcel using the specified :input:`sources`.
   * **Function**: :func:`openplaces.io.harmonizer.attributes.reconcile_attributes`

6. Integrate FEMA footprint occupancy

   * **Explanation**: Links FEMA footprints to parcels via spatial overlay from the :input:`recipe_id` (``US_footprint-fema-2023``) using the :input:`join` (``spatial_overlay``) method for the :input:`source_geometry_type` (``mixed_type_footprint``) and filters out minor intersections under :input:`area_intersection_m2_min` (10 m²).
   * **Function**: :func:`openplaces.io.harmonizer.links.link_to_reference`

7. Extract dominant FEMA occupancy

   * **Explanation**: Reconciles FEMA occupancy types from the :input:`sources` using the remapping crosswalk specified by :input:`remap_id` (``US_footprint-fema-2023_occupancy-type-remap``).
   * **Function**: :func:`openplaces.io.harmonizer.attributes.reconcile_attributes`

8. Summarize footprint morphology

   * **Explanation**: Counts total, primary, and small elongated footprint features on each parcel to feed downstream land-use classification. It links footprints from :input:`footprint_recipe_id` (``US_footprint-spine-2026``) matching on :input:`on` (``parcel_id``), filtering by :input:`small_area_max_m2` (185 m²), :input:`elongated_aspect_min` (2.0), and :input:`min_overlap_m2` (10 m²).
   * **Function**: :func:`openplaces.io.harmonizer.attributes.summarize_footprint_morphology`


Stage 3: Image ingestion
------------------------

This stage fetches imagery required for deep-learning visual classification:

* **Satellite imagery** (``image-googlesatellite-z20.yaml``): Scrapes zoom-level 20 Google Satellite tiles using footprint geometries via BRAILS++.
* **Street View imagery** (``image-googlestreetview-2026.yaml``): Downloads street-level Google Street View photos and depth maps.


Stage 4: Enrichment
-------------------

This stage runs deep learning models (BRAILS++) to predict visual building attributes.

1. Infer roof shape

   * **Explanation**: Runs BRAILS++ deep learning classifiers on satellite imagery to predict roof shape, using visual models on the footprint spine.
   * **Function**: :func:`openplaces.io.enricher.attributes.classify_roof_shape`

2. Detect story counts

   * **Explanation**: Uses computer vision detectors on street-level photos to estimate floors of living area, predicting story height.
   * **Function**: :func:`openplaces.io.enricher.attributes.detect_n_stories`

3. Predict visual occupancy class

   * **Explanation**: Estimates building usage using BRAILS++ model classifiers on Street View imagery.
   * **Function**: :func:`openplaces.io.enricher.attributes.classify_occupancy`


Stage 5: Curation
-----------------

This stage curates the spines into clean, canonical datasets.

Parcel curation
~~~~~~~~~~~~~~~

This stage curates the parcel spine to produce clean assessor attributes:

1. Impute parcel occupancy group

   * **Explanation**: Assigns groups based on NSI modal counts per use code using :input:`group_column` (``use_group_combined``), :input:`value_column` (``group_building_nsi``), the Mode statistic via :input:`statistic` (``mode``), and saving to :input:`output` (``group_parcel``).
   * **Function**: :func:`openplaces.io.curator.imputers.impute_from_group_statistic`

2. Score relative footprint area

   * **Explanation**: Calculates log-space z-scores to assist Vacant and Townhome rule classification by measuring how anomalously small the largest footprint is relative to other parcels sharing the same assessor use code. It uses :input:`group_column` (``use_group_combined``), :input:`value_column` (``max_footprint_area_m2``), :input:`output` (``footprint_area_log_zscore``), :input:`transform` (``log1p``), and :input:`statistic` (``zscore``).
   * **Function**: :func:`openplaces.io.curator.inferers.score_relative_to_group`

3. Classify parcel land use

   * **Explanation**: Assigns parcel land-use classes via weighted voting. It runs multi-indicator voting rules to identify Manufactured Home Park, RV Park, Standalone Manufactured Home, Townhome, Vacant, etc., saving to :input:`output` (``land_use_class``) with flags mapped to :input:`flag_column` (``manufactured_home_park``) and :input:`flag_class` (``Manufactured Home Park``), scoring columns via :input:`score_columns` (Vacant: ``land_use_vacancy_score``), review tracking on :input:`review_column` (``land_use_review``) with a margin of :input:`review_margin` (1.0) and the configured :input:`rules`.
   * **Function**: :func:`openplaces.io.curator.inferers.classify_parcel_land_use`

4. Standardize data categories

   * **Explanation**: Casts string columns to pandas Categorical types to optimize storage footprint and query performance.
   * **Function**: :func:`openplaces.io.curator.formatters.cast_categoricals`

5. Format curated parcel schema

   * **Explanation**: Enforces a standard column order on the final curated parcel schema.
   * **Function**: :func:`openplaces.io.curator.formatters.order_columns`


Footprint curation
~~~~~~~~~~~~~~~~~~

This stage curates the footprint spine, integrating parcel, imagery, and point evidence into the final exposure dataset. The curation recipe executes the following steps chronologically, grouped by functional objective:

Initial evidence assembly
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Integrate clean assessor data

   * **Explanation**: Matches each footprint in the spine to its corresponding parcel in the curated parcel lane using :input:`recipe_id` (``US_parcel-openplaces-2026``) and joins the configured :input:`columns` (improvement_value, land_value, year_built, use_group_combined, group_parcel, manufactured_home_park, occupancy_type_footprint_fema, and land_use_class).
   * **Function**: :func:`openplaces.io.curator.evidence.link_curated_entity`

2. Correct address evidence

   * **Explanation**: Suppresses Overture dwelling unit counts on vacant parcels. It sets :input:`column` (``n_dwellings_overture``) to null if the :input:`condition_column` (``land_use_class_parcel``) matches the value :input:`condition_value` (``Vacant``).
   * **Function**: :func:`openplaces.io.curator.reconcilers.suppress_where`

3. Determine implied Overture occupancy

   * **Explanation**: Assigns a temporary occupancy class to :input:`target` (``occupancy_type_dwelling_overture``) based on the corrected Overture count using the configured :input:`decisions` rules.
   * **Function**: :func:`openplaces.io.curator.reconcilers.resolve_by_vote`

Value reconciliation and metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4. Select canonical values

   * **Explanation**: Resolves conflicts between competing source attributes by selecting the canonical value from the prioritized lists in the :input:`priority` mapping (e.g. for dwelling counts, year built, and financial valuation).
   * **Function**: :func:`openplaces.io.curator.reconcilers.reconcile_values`

5. Zero-fill address counts

   * **Explanation**: Fills missing or suppressed Overture dwelling unit counts listed in :input:`columns` (``[n_dwellings_overture]``) with ``0`` and casts the column to integer.
   * **Function**: :func:`openplaces.io.curator.imputers.fill_missing_numeric`

6. Compute footprint metrics

   * **Explanation**: Calculates structural indicators such as footprint area in square meters from geometry and computes the structural improvement value per unit area.
   * **Function**: :func:`openplaces.io.curator.inferers.derive_metrics`

Baseline attribute imputation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

7. Impute missing residential units

   * **Explanation**: Imputes residential unit counts when no matched source evidence exists based on the occupancy base class.
   * **Function**: :func:`openplaces.io.curator.imputers.impute_n_dwellings`

8. Establish baseline occupancy class

   * **Explanation**: Establishes a baseline occupancy class by selecting the first present value from a prioritized list of evidence columns.
   * **Function**: :func:`openplaces.io.curator.inferers.impute_occupancy_type`

9. Apply property-use keyword corrections

   * **Explanation**: Refines the baseline occupancy class by correcting classes using property-use keywords from the configured csv file in :input:`ruleset` (``parcel-occupancy-keywords.csv``).
   * **Function**: :func:`openplaces.io.curator.reconcilers.resolve_occupancy`

10. Merge visual model predictions

    * **Explanation**: Merges predicted building attributes from the configured :input:`recipes` (e.g. ``US_footprint_built-roof-shape-brails-2026`` and ``US_footprint_built-n-stories-brails-2026``) and their respective columns.
    * **Function**: :func:`openplaces.io.curator.evidence.merge_enrichments`

Occupancy voting and refinement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

11. Score manufactured home probability

    * **Explanation**: Computes a probability score for manufactured home classification based on the :input:`ruleset` (``parcel-occupancy-keywords.csv``) while setting the :input:`update_occupancy` parameter to ``false``.
    * **Function**: :func:`openplaces.io.curator.inferers.classify_manufactured_homes`

12. Resolve occupancy by weighted vote

    * **Explanation**: Resolves final occupancy class (specifically Manufactured Home vs. Multi-Family conflicts) to the :input:`target` column (``occupancy_type``) using the configured :input:`decisions` rules.
    * **Function**: :func:`openplaces.io.curator.reconcilers.resolve_by_vote`

13. Split height bands

    * **Explanation**: Splits the standard :input:`multi_family_class` (``Multi-Family``) into HAZUS height bands based on the reconciled number of stories using the configured :input:`bands`.
    * **Function**: :func:`openplaces.io.curator.inferers.refine_occupancy_height`

14. Flag manufactured home communities

    * **Explanation**: Re-evaluates mobile home park boundaries and flags parcels containing at least :input:`min_homes` (3) final Manufactured Home footprints.
    * **Function**: :func:`openplaces.io.curator.inferers.flag_manufactured_home_communities`

Schema standardization and formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

15. Standardize data categories

    * **Explanation**: Converts string columns to pandas Categorical types.
    * **Function**: :func:`openplaces.io.curator.formatters.cast_categoricals`

16. Cast year built to integer

    * **Explanation**: Rounds and casts the year of construction columns listed in :input:`columns` (``[year_built]``) to nullable integer data types.
    * **Function**: :func:`openplaces.io.curator.formatters.cast_integers`

17. Clean up and order columns

    * **Explanation**: Enforces standard column order and drops the transient helper columns specified in the :input:`drop` list.
    * **Function**: :func:`openplaces.io.curator.formatters.order_columns`
