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

-  :input:`admin-openplaces-2026_admin1` is a  recipe for identifiers of level-1 :ref:`administrative_units` (countries) as a table (:file:`.csv`).
-  :input:`admin-gadm-4~1_admin1` contains the recipe for ingesting global administrative geometries from the Global Administrative Database.
-  :input:`US_admin-nhgis-2020_admin3` contains the recipe for ingesting official county boundaries for the United States provided by NHGIS.
-  :input:`US_building-microsoft-v2` contains the recipe for ingesting US building footprints provided by Microsoft
-  :input:`US-NC_parcel-nconemap-2023` contains the recipe for ingesting parcel boundary data for North Carolina, United States, provided by NCOneMap.

Location
~~~~~~~~

All recipes are in :file:`/src/openplaces/recipes/`, grouped by :ref:`administrative_units`, then by :ref:`entities` or :ref:`datasets`.

Understanding recipes
~~~~~~~~~~~~~~~~~~~~~

See instructions on :ref:`writing-recipes` in the section for contributors.