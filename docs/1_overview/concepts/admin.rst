.. openplaces

.. _administrative_units:

Administrative units: geography
===============================

Administrative units are the geographic regions by which property data is organized.

``openplaces`` uses a globally resolvable, hierarchical (multi-level) set of identifiers for administrative units:

.. list-table::
   :header-rows: 1
   :widths: 5 20 10 60

   * - 
     - Description
     - In code
     - Examples of identifiers
   * - 0
     - Earth
     - 
     - :input:`None`
   * - 1
     - Countries & territories
     - ``admin1``
     - | :input:`US` United States
       | :input:`CO` Colombia
   * - 2
     - States, departments, etc.
     - ``admin2``
     - | :input:`US-MA` state of Massachusetts
       | :input:`CO-AN` department of Antioquia
   * - 3
     - Counties, municipalities, etc.
     - ``admin3``
     - | :input:`US-MA-MI` Middlesex county, Massachusetts, U.S.
       | :input:`CO-AN-ME` municipality of Medellín, Colombia
   * - 4
     - Cities, towns, subdivisions, etc.
     - ``admin4``
     - | :input:`US-MA-MI-SO` city of Somerville

       *Not all countries have level-4 IDs in the default recipe*

Administrative hierarchies of any depth are supported.


Default spine
~~~~~~~~~~~~~

``openplaces`` ships with a default "spine": a canonical set of identifiers for the world:

:gh-file:`src/openplaces/recipes/_all/admin/spine/2026`

It is resolved for the first three administrative levels (countries > states > counties) and for part of the fourth (towns).

Identifiers come mostly from the `Global Administrative Database <https://gadm.org/>`_ (GADM). A few are already updated with official country datasets from published recipes (:gh-file:`src/openplaces/recipes`).

Administrative identifiers are a hybrid of prior systems (ISO, HASC) and codes created by ``openplaces`` derived from names.

Example breakdown: ``US-MA-MI-SO`` (city of Somerville in the state of Massachusetts, U.S.)

-  ``US`` and ``MA``: from the International Organization for Standardization (ISO)
-  ``MI`` (Middlesex county): based on `HASC <https://statoids.com/ihasc.html>`_ codes obtained from GADM.
-  ``SO`` (Somerville): generated internally (no prior standard)

.. admonition:: Disclaimer

   The data mostly reflects the structure of GADM and is used for convenience, not political stance. The identifiers themselves need work to become more intuitive and reflective of global opinion and are therefore subject to change.


Spatial boundaries
~~~~~~~~~~~~~~~~~~

If you want to make maps of administrative boundaries or use them for geoprocessing (e.g., identifying all building footprints within a county), you need to import the boundaries from their original source.

Official boundaries
-------------------

To import official in-country geometries (e.g., as a reference for your geoprocessing), run this notebook with a country recipe:

:gh-file:`notebooks/02_ingest/admin/ingest_admin.ipynb`

Global boundaries
-----------------

To import global geometries (GADM, >2 GB), run this notebook:

:gh-file:`notebooks/01_setup/03_ingest_global_administrative_units.ipynb`

After that, you can make precise global maps.

.. code-block:: python

   from openplaces.api import get_admin

   get_admin(geom=True).plot()

.. image:: images/admin-gadm-4~1_admin1.png
  :width: 350
  :alt: Administrative units

Note that these are large geometry files and will take a while to render (~10 seconds).