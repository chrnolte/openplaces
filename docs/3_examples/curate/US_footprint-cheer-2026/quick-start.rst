.. openplaces

.. _cheer_quick_start:

Quick start
===========

This page gets you from a downloaded bundle to a table you can work
with, and lists the five things most often misread.

Getting a bundle
~~~~~~~~~~~~~~~~

The CHEER team receives the inventory on the
`CHEER shared drive <https://drive.google.com/drive/u/0/folders/1x7Ns775WM7C5Gu7zqF-xggPqYEAXnmqu>`_,
one folder per region:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Region
     - File stem
     - Covers
   * - ``cheer-eastern-nc``
     - ``US-NC_footprint-cheer-2026``
     - 44 counties of eastern North Carolina
   * - ``cheer-coastal-tx``
     - ``US-TX_footprint-cheer-2026``
     - 42 counties of coastal Texas

Each region folder holds the publishable bundle. A ``team/``
subfolder beside it holds the team bundle (see rule 4 below). Bundles
for other uses are available on request.

The files
~~~~~~~~~

Every file name starts with the region's stem, written ``*`` below:
in eastern North Carolina, ``*.parquet`` is
``US-NC_footprint-cheer-2026.parquet``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Holds
   * - ``*.parquet``
     - The inventory: one row per footprint, the canonical attributes
       and their ``_source`` columns, no geometry. Most analyses need
       only this file.
   * - ``*_point.parquet``
     - The same rows on centroid points (GeoParquet), plus
       ``structure_value_per_area`` for quick maps.
   * - ``*_geo.parquet``
     - The footprint polygons (GeoParquet).
   * - ``*_evidence.parquet``
     - Every source's own value for every attribute: the evidence the
       canonical values were chosen from.
   * - ``*_map.qgz``
     - A QGIS project over the bundle, with styled layers for the
       canonical classes and the imputed values.
   * - ``*_admin3_geo.parquet``, ``*_admin4_geo.parquet``
     - County and county-subdivision footprints for the map.
   * - ``*_LICENSE.txt``
     - The sources, their licences, what was withheld and imputed. See
       :ref:`cheer_licence`.

The four tables share the index ``footprint_id`` and the same row
order, so any two join one to one.

Opening it in Python
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import geopandas as gpd
   import pandas as pd

   stem = 'US-NC_footprint-cheer-2026'

   # The inventory, indexed by footprint_id.
   inventory = pd.read_parquet(f'{stem}.parquet')

   # One county's single-family homes (Pender County, FIPS 37141).
   homes = inventory[
       (inventory['county_fips'] == '37141')
       & (inventory['occupancy_type'] == 'Single-Family')
   ]

   # Evidence columns are read by name and join on the index.
   evidence = pd.read_parquet(
       f'{stem}_evidence.parquet',
       columns=['year_built_parcel', 'year_built_block_median_building_nsi'],
   )
   homes = homes.join(evidence)

   # Polygons, for mapping or spatial joins.
   footprints = gpd.read_parquet(f'{stem}_geo.parquet')

Reading only the columns you need (``columns=``) keeps the evidence
file, the largest of the four, fast to load. Every column used here,
and every other column of the four tables, is described in the
:ref:`variable dictionary <cheer_columns>`.

Opening it in QGIS
~~~~~~~~~~~~~~~~~~

Open ``*_map.qgz``. It draws the centroid points at regional
scale and the footprints once you zoom in, with ready-made views of each
canonical class. To load the files yourself, use the
:gui:`Load joined openplaces parquet files` algorithm from the
openplaces QGIS toolbox: pick any file of the bundle, choose points or
polygons, and tick :gui:`Join evidence columns` if you need them.

Five things to know before you analyze
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **One row is a footprint**: a building's shape as drawn from above. Most
   footprints are one building. A townhome row is one footprint holding
   several buildings (``n_buildings_per_footprint`` in the evidence
   file); its sums (dwellings, values) add over them. A garage or shed
   is its own footprint, classed ``Secondary``. See
   :ref:`entities <entities>`.

2. **Structure value is a market value.** ``structure_value`` is the
   county assessor's value of the building, net of land, at the
   county's appraisal date (``appraisal_year``).
   ``structure_value_adjusted`` is the same value in 2025 dollars.
   ``replacement_cost`` and ``depreciated_replacement_cost`` are NSI's
   cost measures at August 2025 prices: what rebuilding would cost,
   new and after wear. The three answer different questions and are
   not interchangeable. See the
   :ref:`variable dictionary <cheer_columns>`.

3. **A 0 is a statement; an empty cell is not.** A ``structure_value``
   of 0 means the roll gives the building no value of its own, and
   ``structure_value_zero_reason`` says why (a shed valued with the
   house, a home held as personal property, an exempt building, ...).
   An empty cell means no source and no model gave a value.

4. **Publishable and team bundles.** The bundle in the region folder
   may be published under its licence. The ``team/`` bundle puts back
   values the publishable one leaves out: sources cleared for sharing
   within the CHEER team only, and share-alike values that cannot be
   published together with the footprints' licence. Keep it within the
   team, and build anything you publish from the publishable bundle.

5. **Every value says where it came from.** Each canonical column has a
   ``_source`` column. ``parcel.<county>`` is the county roll,
   ``nsi`` the National Structure Inventory, and a value ending in
   ``+imputed`` was filled by openplaces. ``withheld`` marks a value a
   source supplied that its terms keep out of this bundle. See
   :ref:`cheer_provenance`.
