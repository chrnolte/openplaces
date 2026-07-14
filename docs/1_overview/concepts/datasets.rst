.. openplaces

.. _datasets:

Datasets: the attributes
========================

:ref:`Entities <entities>` are the rows in a table. Datasets provide the columns: attributes linked to each entity.

Datasets can come in many formats: tables, vectors, rasters, XML. What distinguishes them from :ref:`entities <entities>` is that datasets are *not yet* organized by entity: the rows don't refer to the entity yet (and therefore require some linkage algorithm).

For instance, a user might have a dataset of elevation, flood risk, or hurricane wind fields and want to link it to buildings, a type of :ref:`entity <entities>`.


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
   * - ``population``
     - Demographic data (aggregate)
   * - ``persons``
     - Data of persons, e.g., property owners (sensitive)
   * - ``risk``
     - Floods, storms, wildfires
   * - ``rules``
     - Zoning, conservation