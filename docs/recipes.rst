.. openplaces recipe catalog

.. _recipe_catalog:

Recipe catalog
==============

:ref:`Recipes <recipes>` provide the instructions for data processing in ``openplaces``.

They define where a dataset comes from and how it is processed.

Recipes are dictionaries of arguments, saved as a :input:`.yaml` file.

For an explanation of the arguments you can find in a recipe, see :ref:`writing recipes <writing_recipes>`.

The recipe catalog is auto-generated from :gh-file:`src/openplaces/recipes`:

.. toctree::
   :titlesonly:
   :glob:

   recipes/global
   recipes/*
