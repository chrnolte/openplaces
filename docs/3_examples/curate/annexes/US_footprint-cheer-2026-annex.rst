.. openplaces

.. _cheer_footprints_annex:

Technical annex: CHEER footprints
=================================

This annex provides a step-by-step description of the data curation recipe implemented in :gh-file:`src/openplaces/recipes/US/_all/footprint/cheer/2026/US_footprint-cheer-2026.yaml`. It details how each processing step behaves and how it translates in practice to produce the canonical CHEER footprint inventory.

.. contents:: Table of Contents
   :local:
   :depth: 2


Step 1: Curated Parcel Linking (``link_curated_entity``)
--------------------------------------------------------

* **Purpose**: Integrates clean assessor data from the parcel curation lane.
* **How it translates in practice**:
  
  The step matches each footprint in the spine to its corresponding parcel in the curated parcel lane (``US_parcel-openplaces-2026``) using the globally-unique ``parcel_id``. This joins critical attributes including:
  
  * ``improvement_value_parcel`` and ``land_value_parcel``
  * ``year_built_parcel``
  * ``use_group_combined_parcel``
  * ``group_parcel``
  * ``manufactured_home_park`` status
  * ``occupancy_type_footprint_fema``
  * ``land_use_class_parcel`` (11-class classification)

  Joining these curated parcel attributes first ensures downstream curation steps work with cleaned and harmonized parcel data rather than raw, noisy assessor records.


Step 2: Address Evidence Correction (``suppress_where``)
--------------------------------------------------------

* **Purpose**: Suppresses Overture dwelling unit counts on vacant parcels.
* **How it translates in practice**:
  
  Overture geocoded address points can sometimes be placed on platted but unbuilt or vacant lots. This step sets ``n_dwellings_overture`` to null (suppressed) for any footprints intersecting parcels classified as ``Vacant`` by the land-use classification. This prevents pre-construction or empty parcel address points from being counted as physical dwellings.


Step 3: Implied Overture Occupancy Class (``resolve_by_vote``)
--------------------------------------------------------------

* **Purpose**: Assigns a temporary occupancy class based on Overture address counts.
* **How it translates in practice**:
  
  Determines a transient occupancy class (``occupancy_type_dwelling_overture``) based on the corrected Overture count:
  
  * If the footprint has 2 or more Overture address points, the implied class is ``Multi-Family``.
  * If the footprint has 1 or fewer Overture address points, the implied class is ``Single-Family``.

  This is a transient variable that serves as a lowest-priority fallback and is subsequently dropped from the final schema.


Step 4: Attribute Reconciliation (``reconcile_values``)
-------------------------------------------------------

* **Purpose**: Selects the canonical value from competing source attributes based on a strict order of preference.
* **How it translates in practice**:
  
  For fields where multiple data sources provide values, the recipe resolves conflicts using the following priority orders:
  
  * **Dwelling counts (``n_dwellings``)**: Prioritizes Overture geocoded address counts over NSI structure unit counts (``[n_dwellings_overture, n_dwellings_nsi]``).
  * **Construction year (``year_built``)**: Prioritizes the assessor parcel record over the NSI census-block median fallback (``[year_built_parcel, year_built_block_median_building_nsi]``).
  * **Financial valuation (``value``)**: Prioritizes the assessor improvement value (which is distributed among parcel footprints) over the NSI structure replacement value (``[improvement_value_parcel, structure_value_building_nsi]``).


Step 5: Address Count Zero-Filling (``fill_missing_numeric``)
-------------------------------------------------------------

* **Purpose**: Fills missing Overture dwelling unit counts.
* **How it translates in practice**:
  
  Replaces any missing or suppressed values in ``n_dwellings_overture`` with ``0`` and casts the column to standard integer format. This is intentionally executed after the reconciliation step so that the absence of evidence isn't treated as a confirmed "0 dwellings" count during the voting and priority picks.


Step 6: Metric Derivation (``derive_metrics``)
----------------------------------------------

* **Purpose**: Computes geometry-based variables and footprint value metrics.
* **How it translates in practice**:
  
  Calculates structural indicators such as footprint area in square meters (``m2``) from geometry and computes the structural improvement value per unit area (USD/m²). This enables automated checks of value-to-area consistency.


Step 7: Dwelling Unit Imputation (``impute_n_dwellings``)
---------------------------------------------------------

* **Purpose**: Imputes residential unit counts when no matched source evidence exists.
* **How it translates in practice**:
  
  For residential footprints that still lack a dwelling count after reconciliation, the step estimates ``n_dwellings`` using the occupancy-to-units lookup mapping defined in Lochhead et al. (2026, Table 3) (for instance, mapping a Single-Family or Manufactured Home footprint to 1.0 unit).


Step 8: Occupancy Base Class Inference (``infer_occupancy_type``)
-----------------------------------------------------------------

* **Purpose**: Establishes a baseline occupancy class by selecting the first present value from a prioritized list of evidence columns.
* **How it translates in practice**:
  
  Scans through the available sources in order:
  
  1. NSI building class (``occupancy_type_building_nsi``)
  2. FEMA parcel occupancy (``occupancy_type_footprint_fema``)
  3. Assessor land-use group (``group_parcel``)
  4. Implied Overture occupancy (``occupancy_type_dwelling_overture``)

  The first available value sets the baseline class. If a footprint is located on a manufactured home park parcel, it is kept as ``Manufactured Home`` rather than being classified as ``Secondary`` (accessory), provided it meets size thresholds (at least 50% of the average manufactured home size or $\ge 25\text{ m}^2$, whichever is greater).


Step 9: Key-based Occupancy Correction (``resolve_occupancy``)
--------------------------------------------------------------

* **Purpose**: Refines the baseline occupancy class using property-use keywords.
* **How it translates in practice**:
  
  Applies rules from ``parcel-occupancy-keywords.csv`` to correct the base occupancy class. For example, if a footprint is classified as residential but the assessor property description contains keywords indicating commercial use (e.g., "RETAIL" or "OFFICE"), the classification is updated to match.


Step 10: Image-derived Evidence Integration (``merge_enrichments``)
-------------------------------------------------------------------

* **Purpose**: Joins visual building attributes predicted by deep learning models.
* **How it translates in practice**:
  
  Merges image classifier outputs (e.g., BRAILS models trained on satellite or Street View imagery) into the footprint spine. This fills missing values in the canonical ``roof_shape`` and ``n_stories`` columns.


Step 11: Manufactured Home Classification (``classify_manufactured_homes``)
---------------------------------------------------------------------------

* **Purpose**: Computes a probability score for manufactured home classification.
* **How it translates in practice**:
  
  Evaluates each small residential footprint using geometry and visual models to compute ``p_manufactured_home`` (the probability of being a mobile/manufactured home). The geometric rule looks for narrow, elongated rectangles (aspect ratio $\ge 2.5$ and footprint area $\le 185\text{ m}^2$). This step only outputs the probability score and does not assign the occupancy class directly.


Step 12: Occupancy Resolution Vote (``resolve_by_vote``)
--------------------------------------------------------

* **Purpose**: Resolves final occupancy class (specifically Manufactured Home vs. Multi-Family conflicts) via weighted voting.
* **How it translates in practice**:
  
  A weighted voting system resolves conflicting evidence:
  
  * **Manufactured Home Vote** (requires a minimum score of 2):
    
    * $+1$ if the assessor improvement value share of total parcel value is low ($\le 2.5\%$).
    * $+1$ if the assessor use group description matches mobile home keywords.
    * $+1$ if the parcel-level classification is ``Manufactured Home``.
    * $+1$ if the footprint's morphological probability ``p_manufactured_home`` is $\ge 0.5$.
    
    *Constraint*: To prevent tiny sheds or accessory structures from being misclassified, the footprint must be at least $20\text{ m}^2$ to become a Manufactured Home.
    
  * **Multi-Family Vote** (requires a minimum score of 1):
    
    * $+1$ if the reconciled unit count is 2 or more (``n_dwellings >= 2``).

  This step ensures that manufactured home community parcels (which often list high dwelling counts) are not misclassified as standard multi-family structures.


Step 13: Height-band Split (``refine_occupancy_height``)
--------------------------------------------------------

* **Purpose**: Splits Multi-Family classes into standard HAZUS height sub-classes.
* **How it translates in practice**:
  
  Refines the final ``occupancy_type`` by splitting Multi-Family structures into height bands based on the reconciled ``n_stories``:
  
  * **Low-Rise Multi-Family**: 1 to 3 stories
  * **Mid-Rise Multi-Family**: 4 to 7 stories
  * **High-Rise Multi-Family**: 8 or more stories

  The pre-split class is preserved in the ``occupancy_type_base`` column.


Step 14: Manufactured Home Community Flagging (``flag_manufactured_home_communities``)
-----------------------------------------------------------------------------------------


* **Purpose**: Re-evaluates mobile home park boundaries using final footprint classifications.
* **How it translates in practice**:
  
  Identifies and flags parcels containing 3 or more final ``Manufactured Home`` footprints. This identifies manufactured home communities that may have been missed during the initial parcel-level classification pass.


Step 15: Categorical Casting (``cast_categoricals``)
----------------------------------------------------

* **Purpose**: Converts string columns to pandas Categorical types.
* **How it translates in practice**:
  
  Standardizes column data types and casts textual label columns (such as final occupancy classes and sources) to Categorical types to optimize storage footprint and query performance. While the data types are cast to categorical within pandas, they are written to disk as logical string type columns in the Parquet file to ensure compatibility with GIS tools like GDAL/QGIS.


Step 16: Integer Casting (``cast_integers``)
--------------------------------------------

* **Purpose**: Rounds and casts year of construction to a nullable integer dtype.
* **How it translates in practice**:
  
  Rounds the reconciled ``year_built`` column (which may contain non-integer values from census-block fallbacks) and casts it to pandas' nullable ``Int64`` format. This ensures it displays as a whole year (e.g., 1964 instead of 1964.0) while keeping missing values as null rather than forcing them to a dummy 0.


Step 17: Column Ordering and Cleanup (``order_columns``)
--------------------------------------------------------

* **Purpose**: Cleans up the schema and enforces a standard column order.
* **How it translates in practice**:
  
  Drops transient helper columns that were only needed during the curation pipeline (including ``group_parcel``, ``occupancy_type_dwelling_overture``, its source sidecar, ``value_per_area``, ``occupancy_type_base``, ``p_manufactured_home``, and ``manufactured_home_park``), leaving a clean, documented canonical schema.
