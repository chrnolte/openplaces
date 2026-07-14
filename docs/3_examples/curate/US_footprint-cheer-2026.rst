.. openplaces

.. _cheer_footprints:

U.S. footprint inventory (CHEER)
================================

.. image:: ../../1_overview/concepts/images/footprint_building_dwelling_urban.png
  :width: 400
  :alt: Illustration of footprints, buildings, and dwellings
  :align: right

This recipe creates a footprint-level building inventory for hurricane damage and exposure modeling in the U.S.

It resolves building :ref:`footprints <footprints>` from multiple geometry sources (OpenBuildingMap, Microsoft, and local datasets), then enriches each footprint with data from linked parcels, the National Structure Inventory (NSI), Overture :ref:`dwellings <dwellings>` (addresses), and deep-learning visual classifiers (BRAILS++). 

This work is supported by NSF's Coastal Hazards, Economic Prosperity & Resilience hub (`CHEER <https://www.drc.udel.edu/cheer/>`_). It is currently tested for Florida, Massachusetts, North Carolina, and Texas.

* **Companion notebook**: :gh-file:`notebooks/examples/US_curate_footprints.ipynb`
* **Curation recipe**: :gh-file:`src/openplaces/recipes/US/_all/footprint/cheer/2026/US_footprint-cheer-2026.yaml`


Output columns
~~~~~~~~~~~~~~

The output is produced as a :file:`.parquet` file per county. Each row represents a deduplicated footprint (or parcel fallback). Columns are grouped below by logical category.

Canonical attributes
--------------------

``footprint_id``
    Unique 11-digit `openlocationcode <https://github.com/google/open-location-code>`_ identifier.
``geometry``
    The spatial polygon outlining the footprint, or the parcel polygon for fallback records.
``geometry_source``
    Source of the geometry: ``obm`` (OpenBuildingMap), ``microsoft``, local state sources (e.g., ``nconemap``), or ``parcel.<source>`` for parcel-shaped fallbacks representing unlocated structures.
``occupancy_type``
    Canonical occupancy class. Multi-Family structures are split into HAZUS height bands (Low-Rise: 1-3 stories, Mid-Rise: 4-7 stories, High-Rise: 8+ stories) using ``n_stories`` (stilts/open ground floors excluded).
``value``
    Reconciled structure value in USD. Prioritizes the parcel's improvement value (split across parcel footprints) over the NSI structure replacement value.
``year_built``
    Reconciled construction year. Prioritizes assessor parcel records over NSI block-median fallbacks.
``n_stories``
    Reconciled number of stories from joined enrichments.
``n_dwellings``
    Reconciled count of dwelling units. Prioritizes Overture geocoded address counts over NSI structure counts, falling back to occupancy-class imputation.
``roof_shape``
    Reconciled roof structure classification (e.g., Gable, Hip, Flat) from visual models.
``m2``
    Calculated footprint area in square meters.
``priority_on_parcel``
    Structural role: ``primary`` (main structure), ``secondary`` (accessory structure), or ``unknown`` (unlinked to parcel).

Provenance sidecars
-------------------
Indicates which input dataset or curation rule set the final canonical value.

* ``occupancy_type_source``
* ``value_source``
* ``year_built_source``
* ``n_stories_source``
* ``n_dwellings_source``
* ``roof_shape_source``

Parcel-derived evidence (assessor)
----------------------------------
Attributes inherited from the primary parcel linked to the footprint.

``parcel_id``
    Globally unique ID of the primary linked parcel, used to join parcels back onto footprints.
``parcel_id_local`` / ``parcel_id_local_all``
    Locally cross-comparable assessor ID of the primary linked parcel, and combined IDs of all intersected parcels (populated only if more than one distinct ID exists). Not unique across administrative units, so it is not used as a join key.
``use_group_combined_parcel`` / ``use_group_combined_parcel_all``
    Primary and combined land-use description strings (populated only if more than one distinct use exists).
``land_use_class_parcel``
    Curated parcel land-use classification mapped into 11 potential classes (e.g., Vacant, Manufactured Home, Retail, Office).
``occupancy_type_parcel``
    Assessor-proposed occupancy based on property keywords/groups.
``land_value_parcel``
    Assessor land valuation (allocated to primary footprints only).
``improvement_value_parcel``
    Assessor improvement valuation (distributed among footprints).
``year_built_parcel``
    Construction year recorded in assessor tax records.
``address_parcel``
    Property street address from assessor records.
``overlap_fraction_parcel``
    Fraction of footprint area intersecting this parcel.

National Structure Inventory evidence (NSI)
--------------------------------------------
Structure-level point attributes matched to the footprint.

``n_buildings_nsi`` / ``building_id_nsi``
    Count and ID of matched NSI structure records.
``occupancy_type_building_nsi`` / ``occupancy_type_building_nsi_all``
    NSI occupancy class (single matches, and combined list of matches if more than one distinct class exists).
``group_building_nsi`` / ``group_building_nsi_all``
    NSI occupancy group mapped to openplaces vocabulary (single matches, and combined list of matches if more than one distinct group exists).
``structure_value_building_nsi``
    NSI-calculated replacement value of the structure.
``year_built_block_median_building_nsi``
    Median year built for the associated census block.
``source_building_nsi``
    Underlying source database of the NSI record (e.g., Model).

Overture dwelling and address evidence
--------------------------------------
Residential unit address points matched from Overture.

``n_dwellings_overture``
    Total count of geocoded residential unit points linked (unmatched footprints or those on vacant-use parcels are filled with 0).
``address_street_dwelling_overture`` / ``address_number_dwelling_overture``
    Geocoded street name and house number components.
``postal_code_dwelling_overture`` / ``city_dwelling_overture``
    Postal code and city components.

Diagnostics and special metrics
-------------------------------
Flags and intermediate calculations used for curation and quality control.

``n_parcels_per_footprint``
    Number of parcels intersected by the footprint.
``n_footprints_per_parcel``
    Number of footprints on the associated parcel (includes synthetic fallbacks).
``occupancy_type_conflict``
    Summary of conflicting occupancy labels across inputs (formatted ``"nsi: X | fema: Y | parcel: Z | overture: W"``).
``occupancy_type_review``
    Flag (``True``) for low improvement-value parcel shares indicating potential manufactured homes needing inspection.
``improvement_value_parcel_per_area`` / ``structure_value_building_nsi_per_area``
    Calculated values divided by footprint area (USD/m²).

Processing pipeline
~~~~~~~~~~~~~~~~~~~

The creation of the CHEER dataset proceeds chronologically through five core stages of the ``openplaces`` pipeline: **Ingest**, **Harmonize**, **Image Ingest**, **Enrich**, and **Curate**.

.. list-table:: Source datasets joined
   :header-rows: 1
   :widths: 20 40 40

   * - Dataset
     - Main contribution
     - Analytical role
   * - **Footprints**
     - Geometries from OBM, Microsoft, and state sources
     - Primary spatial exposure units
   * - **Parcels**
     - Assessor values, land-use groups, and addresses
     - Financial valuation and ownership boundaries
   * - **NSI**
     - Structure point records and replacement value
     - Building classifications and story heights
   * - **Overture**
     - Address point clusters
     - Dwelling-unit counts and address components

Stage 1: ingest
---------------

This stage downloads and extracts raw geometry and reference point datasets:

1. **Footprint datasets**: Downloads and unzips raw footprints from OpenBuildingMap (OBM), Microsoft, and state/local layers.
2. **Assessor parcels**: Gathers property tax assessor geometry and tax rolls from local/state agencies.
3. **Reference layers**: Downloads structure point databases (National Structure Inventory; NSI) and geocoded residential address points (Overture).

Stage 2: harmonize
------------------

This stage merges geometries and links datasets to build the core footprint spine (``US_footprint-spine-2026``) and parcel spine (``US_parcel-spine-2026``):

1. **Geometry deduplication**: Merges footprints in priority order (OBM first, then Microsoft, then state-specific sources). Footprints smaller than 10 m² are dropped. Candidate shapes are added to the spine only if their Intersection-over-Union (IoU) with existing shapes is < 0.02 and they do not fall within a size-scaled exclusion buffer around large existing spine footprints (area >= 250 m²; buffer distance scales with shape area).
2. **Elongated-footprint filter**: Aspect ratios >= 2.5 with aligned axes (within 15°), longitudinal overlap >= 50%, and lateral spacing < 2x width are deduplicated to prevent parallel-shifted footprint representations (common for manufactured homes) from appearing twice.
3. **Parcel spatial overlay**: Intersects footprints with parcels via a polygon identity overlay. Intersections under 10 m² or minor slivers (< 1/6 of the footprint's largest parcel intersection) are filtered.
4. **Synthetic fallbacks**: For parcels where assessor records indicate a structure exists but no footprint is detected, synthetic "footprint" rows are created using the parcel boundary geometry (labeled ``parcel.<source>``). Overlaps against detected footprints are spatial-trimmed. These fallback polygons are included in the parcel footprint count, are backfilled with the ID (``parcel_id``) and local ID (``parcel_id_local``) of their source parcel, and are excluded from morphology metrics.
5. **Reference point linking**: Associates structure-level point evidence with the footprint spine using a tiered proximity join:

   * *Containment*: Point is within the footprint.
   * *Inner Proximity*: Points within 10 meters of an edge are assigned to the closest footprint.
   * *Outer Proximity*: Points within 100 meters (NSI) or 50 meters (Overture) are assigned to the closest footprint *on the same parcel*, ensuring points aren't misaligned across street boundaries.
6. **Role classification**: Footprints on multi-structure parcels are classified to identify primary vs. accessory structures:

   * Footprints with Overture dwelling points are ``primary``.
   * Otherwise, footprints with NSI building points are ``primary``.
   * If no point evidence exists, all footprints on the parcel default to ``secondary``.
   * Sole footprints on a parcel are always classified as ``primary``.
7. **Raw attribution**: Compiles all raw joined variables into source-suffixed evidence columns in the intermediate spine.

Stage 3: image ingest
---------------------

This stage fetches external imagery required for deep-learning visual classification, querying the Google API using footprint geometries from the harmonized spine:

1. **Satellite imagery**: Scrapes per-building Google Satellite images at Zoom level 20 (``image-googlesatellite-z20``) via the BRAILS++ scraper.
2. **Street View imagery**: Downloads street-level Google Street View photos (``image-googlestreetview-2026``).

Stage 4: enrich
---------------

This stage runs deep learning models (BRAILS++) on the ingested imagery to predict visual building attributes:

1. **Roof shape prediction**: Runs neural network classifiers on the scraped satellite imagery to predict ``roof_shape``.
2. **Story count detection**: Runs detectors on the Street View photos to infer ``n_stories``.

Stage 5: curate
---------------

This stage curates the spines into clean, canonical datasets:

1. **Parcel curation** (``US_parcel-openplaces-2026``): Derives parcel-level occupancy groups, calculates relative footprint area z-scores, and classifies land use (e.g., Manufactured Home Park, Vacant, Retail, Office) using weighted voting.
2. **Footprint curation** (``US_footprint-cheer-2026``): Integrates curated parcels, corrected address counts, reconciled attribute priorities, base occupancy imputation, visual model prediction, and weighted voting to resolve final footprint occupancy classes. This footprint curation process executes the following steps:

   * **Curated parcel linking**: Curated attributes from the parcel curation lane (FEMA occupancy, assessor values, combined land use, and the 11-class ``land_use_class_parcel`` classification) are joined by matching the footprint's parcel reference (``parcel_id``) to the curated parcel's own globally unique ID (``parcel_id``).
   * **Address evidence correction (suppress_where)**: Suppresses (nulls) ``n_dwellings_overture`` counts on parcels classified as ``Vacant`` by the parcel land-use vote, filtering out pre-construction or platted address points that do not represent physical buildings.
   * **Implied Overture occupancy (resolve_by_vote)**: Resolves the transient implied occupancy type (``occupancy_type_dwelling_overture``) from corrected Overture counts alone (Single-Family for <= 1 dwelling, Multi-Family for >= 2).
   * **Attribute reconciliation**: Canonical attributes are resolved from competing evidence (such as prioritizing Overture dwelling counts over NSI, assessor construction year over NSI block-median, and assessor improvement value over NSI replacement value).
   * **Address count zero-filling (fill_missing_numeric)**: Fills missing or suppressed ``n_dwellings_overture`` counts with 0.
   * **Base occupancy imputation**: Assigns base occupancy class sequentially using NSI occupancy, FEMA parcel occupancy, parcel land-use mode fallbacks, and lastly, the implied Overture occupancy.
   * **Occupancy voting**: Resolves ambiguous Manufactured Home vs. Multi-Family cases using a weighted point vote scoring system.
   * **Height-band splits**: Splits Multi-Family classes into HAZUS height bands (Low-Rise: 1-3, Mid-Rise: 4-7, High-Rise: 8+) based on ``n_stories``.
   * **Diagnostics & formatting**: Computes final diagnostic columns (e.g., value-per-area), flags occupancy conflicts, casts categoricals, and orders output columns.

Technical Annex
~~~~~~~~~~~~~~~

.. toctree::

   annexes/US_footprint-cheer-2026-annex.rst


Key references
~~~~~~~~~~~~~~

.. _Lochhead et al. 2026:

Lochhead M, Zsarnoczay A, Deierlein G (2026) Exposure matters: a synthesis framework for high-resolution building inventory development. International Journal of Disaster Risk Reduction 139 (2026): 106148. `doi:10.1016/j.ijdrr.2026.106148 <https://doi.org/10.1016/j.ijdrr.2026.106148>`_

.. _Cetiner et al. 2025:

Cetiner, B., McKenna, F., Yi, S.-ri, Wang, B., & Manousakis, I. V. (2025). BRAILS++ (v4.2.0). Zenodo. `doi:10.5281/zenodo.17797364 <https://doi.org/10.5281/zenodo.17797364>`_


