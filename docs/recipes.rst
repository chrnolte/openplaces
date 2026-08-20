.. openplaces recipe catalog

.. _recipe_catalog:

Recipe catalog
==============

:ref:`Recipes <recipes>` provide the instructions for data processing in ``openplaces``.

They define where a dataset comes from and how it is processed.

Recipes are dictionaries of arguments, saved as a :input:`.yaml` file.

For an explanation of the arguments you can find in a recipe, see :ref:`writing recipes <writing_recipes>`.

The whole catalog is auto-generated from :gh-file:`src/openplaces/recipes`.

.. recipe-coverage::

Geographies
-----------

Pick a country to see its own recipes and, where available, drill down into
its states/regions and counties. A blank cell means no recipe of that kind
exists yet for that geography -- not that the underlying data doesn't exist.

.. recipe-children::

.. toctree::
   :hidden:
   :titlesonly:
   :glob:

   recipes/global
   recipes/*
