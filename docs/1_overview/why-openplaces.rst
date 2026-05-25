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

It is `published on Github <https://github.com/chrnolte/openplaces>`_ under a Apache 2.0 license.

Science benefits when data and methods behind insights are shareable, reproducible, and equitably accessible. Leading public funders, such as the `US National Science Foundation <https://www.nsf.gov/public-access>`_, the `European Research Council <https://erc.europa.eu/manage-your-project/open-science>`_, `Japan’s Science and Technology Agency <https://www.jst.go.jp/EN/about/strategy.html>`_, and Brazil's `FAPESP <https://www.fapesp.br/openscience/en>`_, increasingly require open-access publication, data management plans, and public sharing of research data and supporting materials to improve transparency and reproducibility.

Global frameworks like the `UNESCO Recommendation on Open Science <https://www.unesco.org/en/open-science/about>`_ and multilateral initiatives like `cOAlition S <https://www.coalition-s.org/>`_ actively drive policy reform that treat code and datasets as essential public infrastructure.

But property data is an outlier. It is sensitive, identifiable, and often subject to privacy restrictions. Public records are fragmented: created by thousands of agencies using thousands of systems. Researchers running large-scale analyses have to buy licenses from commercial data aggregators, who discourage data sharing.

As a result, property data is rarely published alongside scientific insights. Many data sets that take analysts months or years to prepare become unavailable after use. Data processing pipelines are treated as property value and rarely shared, sometimes because such pipelines can reveal information about the input data.

This worsens the credibility problem of the empirical social sciences, e.g. in `environmental <https://www.journals.uchicago.edu/doi/10.1093/reep/reaa011>`_ and `agricultural economics <https://onlinelibrary.wiley.com/doi/full/10.1002/aepp.13323>`_, where `data quality and methods affect results <https://le.uwpress.org/content/100/1/200>`_, but professional incentives pull researchers towards overreporting of statistical significance, and re-analyses are rare.

Publishing full analytical workflows in an open framework allows others to revisit, interrogate, and improve prior findings - a precondition for truly "standing" on the shoulders of giants.


Globally compatible
~~~~~~~~~~~~~~~~~~~

Most empirical studies of property data produce insights for a given locality, state, or country. Do their findings replicate across different geographies or variations of methodological approaches?

Comparative studies hold the promise of deeper, more generalizable insights. However, because property data is sensitive, it is rarely shared across country boundaries.

A globally usable open-source framework allows analysts to build on each other's methods, run analyses locally on their own data, and share reproducible results without sharing the underlying data.

``openplaces`` is designed with international applications in mind. It includes a global referencing system to seamlessly work across :ref:`administrative hierarchies<administrative_units>`:

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

``openplaces`` is designed to run on anyone's operating system:

- a government agency PC (Windows)
- a student's Macbook (OSx)
- a university research cluster (Linux)

It is built in ``python`` and takes advantages of its fantastic geospatial ecosystem (GDAL, geopandas, etc.).


Scalable
~~~~~~~~

``openplaces`` is meant to facilitate large-scale, international, comparative property analyses.

You can run a city-wide analysis on your laptop - or deploy it on a research cluster to process satellite data for millions of locations.

In the US, we have attributed satellite-based data to 133 million parcels and used it for `land valuation <https://placeslab.org/research/#cost>`_ and `hedonic valuation <https://placeslab.org/research/#valuation>`_, producing tens of terabytes of trained models.


Extensible
~~~~~~~~~~

``openplaces`` makes it easy to integrate new datasets (tables, vectors, rasters) into parcel-level analyses.

New users :ref:`write recipes<writing_recipes>` to ingest new data and to harmonize datasets.

They can :ref:`contribute <contribute>` recipes to the public repository, receive credit, and allow others to build on their work.