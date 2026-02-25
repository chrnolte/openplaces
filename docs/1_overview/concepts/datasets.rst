.. openplaces

.. _datasets:

Datasets
========

Datasets are what we name any data sets that is not organized as a dataset of :ref:`entities`, but contains data that we might want to attribute to entities.

For instance, a user might have a **dataset** of elevation and flood risk and might want to attribute it to **buildings**, an :ref:`entity <entities>`.

Like :ref:`administrative_units` and entities, datasets have an identifier that determines where in the filesystem their recipes and data can be found.


Themes
~~~~~~

Datasets are grouped into themes. The top-level themes are pre-defined, so the data/folder space does not get cluttered. They are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Theme
     - Description
   * - ``climate``
     - Temperature, precipitation, change
   * - ``land``
     - Topography, geology, soils
   * - ``landcover``
     - Land cover (usually remotely sensed)
   * - ``water``
     - Rivers, lakes, coasts
   * - ``bio``
     - Vegetation, species, land cover
   * - ``built``
     - Buildings, infrastructure, roads & accessibility
   * - ``people``
     - Demographic data (aggregate)
   * - ``persons``
     - Data of persons, e.g. property owners (sensitive)
   * - ``risk``
     - Floods, storms, wildfires
   * - ``rules``
     - Zoning, conservation