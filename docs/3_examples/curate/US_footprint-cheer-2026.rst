.. openplaces

.. _cheer_footprints:

U.S. footprint inventory (CHEER)
================================

.. image:: ../../1_overview/concepts/images/footprint_building_dwelling_urban.png
  :width: 400
  :alt: Illustration of footprints, buildings, and dwellings
  :align: right

This recipe creates a footprint-level building inventory for U.S. hurricane damage and exposure modeling. 

It resolves building :ref:`footprints <footprints>` from multiple geometry sources (OpenBuildingsMap, Microsoft, and local datasets), then enriches each footprint with data from linked parcels, the National Structure Inventory (NSI), and Overture :ref:`dwellings <dwellings>` (addresses). 

This work is supported by NSF's Coastal Hazards, Economic Prosperity & Resilience hub (`CHEER <https://www.drc.udel.edu/cheer/>`_). It is currently tested for Florida, Massachusetts, North Carolina, and Texas.

* **Companion notebook**: :gh-file:`notebooks/examples/US_curate_footprints.ipynb`
* **Curation recipe**: :gh-file:`src/openplaces/recipes/US/_all/footprint/cheer/2026/US_footprint-cheer-2026.yaml`
* **Technical Annex**: :doc:`US_footprint-cheer-2026-annex`


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
    Source of the geometry: ``obm`` (OpenBuildingsMap), ``microsoft``, local state sources (e.g. ``nconemap``), or ``parcel.<source>`` for parcel-shaped fallbacks representing unlocated structures.
``occupancy_type``
    Canonical occupancy class. Multi-Family structures are split into HAZUS height bands (Low-Rise: 1–3 stories, Mid-Rise: 4–7 stories, High-Rise: 8+ stories) using ``n_stories`` (stilts/open ground floors excluded).
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

``parcel_id`` (bare — no ``_parcel`` suffix, unlike the other attributes below)
    Globally-unique id of the primary linked parcel, used to join the curated parcel lane back onto each footprint (``link_curated_entity``).
``parcel_id_local`` / ``parcel_id_local_all`` (bare)
    Locally cross-comparable assessor ID of the primary linked parcel, and combined IDs of all intersected parcels (populated only if more than one distinct ID exists). Not unique across admin units, so it is not used as a join key.
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
    Underlying source database of the NSI record (e.g. Parcel, Model).

Overture dwelling & address evidence
------------------------------------
Residential unit address points matched from Overture.

``n_dwellings_overture``
    Total count of geocoded residential unit points linked (unmatched footprints or those on vacant-use parcels are filled with 0).
``address_street_dwelling_overture`` / ``address_number_dwelling_overture``
    Geocoded street name and house number components.
``postal_code_dwelling_overture`` / ``city_dwelling_overture``
    Postal code and city components.

Diagnostics & special metrics
-----------------------------
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

The creation of the CHEER dataset proceeds chronologically through the four core stages of the ``openplaces`` pipeline: **Ingest**, **Harmonize**, **Enrich**, and **Curate**.

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

1. **Geometry ingestion**: Downloads and unzips raw footprints from multiple geometry providers: OpenBuildingsMap (OBM), Microsoft, and state/local footprint layers.
2. **Reference data ingestion**: Gathers secondary datasets including property tax parcels, the National Structure Inventory (NSI) point database, and Overture dwelling address points.

Stage 2: harmonize
------------------

1. **Geometry deduplication**: Merges footprints in priority order (OBM first, then Microsoft, then state-specific sources). Footprints smaller than 10 m² are dropped. Candidate shapes are added to the spine only if their Intersection-over-Union (IoU) with existing shapes is < 0.02 and they do not fall within a size-scaled exclusion buffer around large existing spine footprints (area ≥ 250 m²; buffer distance scales with shape area).
2. **Elongated-footprint filter**: Aspect ratios ≥ 2.5 with aligned axes (within 15°), longitudinal overlap ≥ 50%, and lateral spacing < 2x width are deduplicated to prevent parallel shifted footprint representations (common for manufactured homes) from appearing twice.
3. **Parcel spatial overlay**: Intersects footprints with parcels via a polygon identity overlay. Intersections under 10 m² or minor slivers (< 1/6 of the footprint's largest parcel intersection) are filtered.
4. **Synthetic fallbacks**: For parcels where assessor records indicate a structure exists but no footprint is detected, synthetic "footprint" rows are created using the parcel boundary geometry (labeled ``parcel.<source>``). Overlaps against detected footprints are spatial-trimmed. These fallback polygons are included in the parcel footprint count, are backfilled with the id (``parcel_id``) and local ID (``parcel_id_local``) of their source parcel, and are excluded from morphology metrics (such as maximum footprint area and small elongated footprint tallies).
5. **Raw attribution**: Compiles all raw joined variables into source-suffixed evidence columns in the intermediate spine (``US_footprint-spine-2026``).

Stage 3: enrich
---------------

1. **Point joining (NSI & Overture)**: Associates structure-level point evidence with the footprint spine using a tiered proximity join (`Lochhead et al. 2026`_):

   * *Containment*: Point is within the footprint.
   * *Inner Proximity*: Points within 10 meters of an edge are assigned to the closest footprint.
   * *Outer Proximity*: Points within 100 meters (NSI) or 50 meters (Overture) are assigned to the closest footprint *on the same parcel*, ensuring points aren't misaligned across street boundaries.
2. **Role classification**: Footprints on multi-structure parcels are classified to identify primary vs. accessory structures:

   * Footprints with Overture dwelling points are ``primary``.
   * Otherwise, footprints with NSI building points are ``primary``.
   * If no point evidence exists, all footprints on the parcel default to ``secondary``.
   * Sole footprints on a parcel are always classified as ``primary``.
3. **Visual model enrichment**: Downloads and caches pretrained neural network weights (e.g. from Zenodo for BRAILS++ classifiers) on demand, then runs image-based models (`Cetiner et al. 2025`_) on satellite or Street View imagery to infer visual attributes (such as ``roof_shape``, ``n_stories``, and the probability ``p_manufactured_home``).

Stage 4: curate
---------------

1. **Curated parcel linking**: Curated attributes from the parcel curation lane (FEMA occupancy, assessor values, combined land use, and the 11-class ``land_use_class_parcel`` classification) are joined by matching the footprint's parcel reference (``parcel_id``) to the curated parcel's own globally-unique id (``parcel_id``) — not the locally-scoped ``parcel_id_local``, which can collide within an admin unit.
2. **Address evidence correction (suppress_where)**: Suppresses (nulls) ``n_dwellings_overture`` counts on parcels classified as ``Vacant`` by the parcel land-use vote, filtering out pre-construction or platted address points that do not represent physical buildings.
3. **Implied Overture occupancy (resolve_by_vote)**: Resolves the transient implied occupancy type (``occupancy_type_dwelling_overture``) from corrected Overture counts alone (Single-Family for ≤ 1 dwelling, Multi-Family for ≥ 2). This serves as a last-resort fallback evidence source and is subsequently dropped from the final output.
4. **Attribute reconciliation**: Canonical attributes are resolved from competing evidence:

   * *dwellings*: Overture count → NSI count → occupancy class-based imputation.
   * *year built*: Assessor record → NSI block-median year built.
   * *value*: Assessor improvement value (allocated only to primary footprints) → NSI structure replacement value.
5. **Address count zero-filling (fill_missing_numeric)**: Fills missing or suppressed ``n_dwellings_overture`` counts with 0 and casts the column to an integer type (``int64``). This is executed after reconciliation to avoid treating "no matched evidence" as "zero confirmed dwellings" during prior voting and priority-pick steps.
6. **Base occupancy inference**: Assigns base occupancy class sequentially using NSI occupancy, FEMA parcel occupancy, parcel land-use mode fallbacks, and lastly, the implied Overture occupancy. Accessory structures on manufactured home park parcels are kept as ``Manufactured Home`` rather than ``Secondary`` if they meet size thresholds.
7. **Occupancy voting**: Resolves ambiguous Manufactured Home vs. Multi-Family cases using a weighted point vote scoring system (assessor keywords, low value share, NSI class, and visual probability ``p_manufactured_home`` vs. unit counts). To classify as a Manufactured Home, the footprint must also satisfy a minimum size threshold of 20 m²; otherwise, it retains its existing class (e.g., Secondary).
8. **Height-band splits**: Splits Multi-Family classes into HAZUS height bands (Low-Rise: 1-3, Mid-Rise: 4-7, High-Rise: 8+) based on ``n_stories``.
9. **Diagnostics & formatting**: Computes final diagnostic columns (e.g. value-per-area), flags occupancy conflicts, casts categoricals, and orders output columns.

Key references
~~~~~~~~~~~~~~

.. _Lochhead et al. 2026:

Lochhead M, Zsarnoczay A, Deierlein G (2026) Exposure matters: a synthesis framework for high-resolution building inventory development. International Journal of Disaster Risk Reduction 139 (2026): 106148. `doi:10.1016/j.ijdrr.2026.106148 <https://doi.org/10.1016/j.ijdrr.2026.106148>`_

.. _Cetiner et al. 2025:

Cetiner, B., McKenna, F., Yi, S.-. ri ., Wang, B., & Manousakis, I. V. (2025). BRAILS++ (v4.2.0). Zenodo. `doi:10.5281/zenodo.17797364 <https://doi.org/10.5281/zenodo.17797364>`_
