.. openplaces

Why openplaces?
===============

``openplaces`` offers open-source processing for land, building, and property studies.

It is designed to synthesize and analyze property data for scientific research, such as:

- Estimating `effects of conservation interventions <https://placeslab.org/research/#effectiveness>`_,
- Valuing `environmental amenities and risks <https://placeslab.org/research/#valuation>`_,
- Valuing land values for `conservation planning <https://placeslab.org/research/#cost>`_ and property taxation,
- Estimation of economic damage from natural hazards, e.g. from `hurricanes <https://www.drc.udel.edu/cheer/>`_.


Open source
~~~~~~~~~~~

.. image:: images/open_source.png
  :width: 58
  :alt: Open source logo
  :align: right
  :class: padded-image


``openplaces`` is open source.

It is `on Github <https://github.com/chrnolte/openplaces>`_, published under a permissive `Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0>`_.

Science can lift everyone's boat - especially when data and methods behind insights are shareable, reproducible, and equitably accessible.

Leading public funders - including the `US National Science Foundation <https://www.nsf.gov/public-access>`_, the `European Research Council <https://erc.europa.eu/manage-your-project/open-science>`_, `Japan’s Science and Technology Agency <https://www.jst.go.jp/EN/about/strategy.html>`_, and Brazil's `FAPESP <https://www.fapesp.br/openscience/en>`_ - increasingly require open access, data management plans, and public sharing of research outputs. Global frameworks like the `UNESCO Recommendation on Open Science <https://www.unesco.org/en/open-science/about>`_ and initiatives like `cOAlition S <https://www.coalition-s.org/>`_ further promote code and datasets as public infrastructure.

But property data is an outlier. It is sensitive, identifiable, and subject to privacy restrictions. Public records are fragmented: created by thousands of agencies using thousands of systems. Researchers running large-scale analyses often work with data licensed from commercial data aggregators, who discourage data sharing. As a result, property data is rarely published alongside scientific insights. Data sets that take analysts months or years to prepare become unavailable after use.

This worsens the credibility problem of the empirical social sciences, e.g. in `environmental <https://www.journals.uchicago.edu/doi/10.1093/reep/reaa011>`_ and `agricultural economics <https://onlinelibrary.wiley.com/doi/full/10.1002/aepp.13323>`_, where `data quality and methods affect results <https://le.uwpress.org/content/100/1/200>`_, professional incentives encourage questionable research practices, publication bias can inflate statistical significance, and re-analyses are rare.

Publishing full analytical workflows in an open framework allows others to revisit, interrogate, and improve prior findings. It makes it easier to climb up onto the shoulders of giants.


Global referencing
~~~~~~~~~~~~~~~~~~

Most empirical studies of property data produce insights for a given place: a city, a state, or a country.

How do their findings replicate in a different place?

Much of the benefit transfer literature suggests they won't. The literature finds large transfer errors (e.g. `a median of 39% <https://doi.org/10.1016/j.jeem.2013.03.001>`_).

Are they robust to variations in methodological approaches or offer opportunities for overstating significance?

Comparing results between locations and methods also promises deeper, more robust, and more context-sensitive insights.

However, because property data is sensitive and often licensed, it is rarely shared, e.g., across country borders.

A globally usable open-source framework allows analysts to build local data sets in their own secure computing environment, analyze them with shared methods, and publish replicable science without having to give each other access to the underlying data.

``openplaces`` ships with a global referencing system designed to harmonize :ref:`administrative hierarchies<administrative_units>`:

- :input:`US` -> United States
- :input:`CO-AN` -> Antioquia, a department of Colombia
- :input:`US-CA-LA` -> Los Angeles, a county in the state of California, United States
- :input:`US-MA-MI-CA` -> Cambridge, a city in Middlesex county, Massachusetts, United States


Cross-platform
~~~~~~~~~~~~~~

.. image:: images/cross_platform.png
  :width: 150
  :alt: Open source logo
  :align: right
  :class: padded-image

``openplaces`` is developed for three operating systems. It runs on:

- a government agency's desktop computer (Microsoft Windows)
- a student's Macbook (macOS)
- a university research cluster (Linux)

It is built in `Python <https://www.python.org/>`_ on top of the open-source geospatial Python ecosystem, including GDAL, GeoPandas, Rasterio, Xarray, Parquet, scikit-learn, and XGBoost.


Scalable
~~~~~~~~

``openplaces`` is meant to facilitate large-scale, international analyses.

You can run a city-wide analysis on your laptop - or deploy it on a research cluster to process satellite data for millions of locations.

In the US, we have attributed satellite-based data to 133 million parcels and used it for `land valuation <https://placeslab.org/research/#cost>`_ and `hedonic valuation <https://placeslab.org/research/#valuation>`_, producing tens of terabytes of trained models on Boston University's shared computing cluster.


Extensible
~~~~~~~~~~

``openplaces`` makes it easy to integrate new datasets (tables, vectors, rasters) into parcel-level analyses.

New users :ref:`write recipes<writing_recipes>` to ingest new data and to harmonize datasets.

They can :ref:`contribute <contribute>` recipes to the public repository, receive credit, and allow others to build on their work.