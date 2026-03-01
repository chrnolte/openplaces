.. openplaces

.. _recipes:

Recipes
=======

Recipes are instructions for ingesting and transforming data.

They describe where the input data comes from and how it needs to be processed at every step (e.g. data ingestion, harmonization, curation, etc.).

Recipes are what makes ``openplaces`` scalable: the package provides an abstraction logic, and the recipes translate external data to that logic.

And each new recipe expands the analytical capacities of all other users.

Identifiers
~~~~~~~~~~~

Recipes are identified by their ``recipe_id``, a concatenation of:

-  :ref:`administrative_units` (optional)
-  an identifier of :ref:`entities` or :ref:`datasets`, and
-  also optionally: a filename.

Examples:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Recipe ID
     - Description
   * - ``admin-openplaces-2026_admin1``
     - Global identifiers of countries/territories (administrative units, level 1: ``admin1``).

       *This recipe is a table*
   * - ``admin-gadm-4~1_admin1``
     - Global administrative geometries from the Global Administrative Database (GADM)

       *This recipe contains instructions on how to access the data*
   * - ``US_admin-census-2021_admin3``
     - United States: official county (``admin3``) boundaries from the US census
   * - ``US_building-microsoft-v2``
     - US building footprints provided by Microsoft
   * - ``US-NC_parcel-nconemap-2023``
     - Parcel boundary data for North Carolina, United States, provided by NCOneMap

Location
~~~~~~~~

Recipes live in :file:`src/openplaces/recipes/`.

They are grouped by :ref:`administrative_units`, followed by :ref:`entities` or :ref:`datasets`.

Understanding recipes
~~~~~~~~~~~~~~~~~~~~~

See instructions on :ref:`writing_recipes` in the section for contributors.