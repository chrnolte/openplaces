.. openplaces

Why openplaces?
===============

``openplaces`` is an open-source processing engine for land, building, and property analytics.

It is designed to generate research-grade property data for:

- assessment of natural hazard risks, such as hurricanes
- valuation of environmental amenities, risks, and shocks
- property taxation problems, such as land vs. structure valuation
- estimating effects of conservation investments on land cover outcomes


Open source
~~~~~~~~~~~

``openplaces`` is open source `on Github <https://github.com/chrnolte/openplaces>`_ (Apache 2.0 license).

Humanity benefits when data and methods behind scientific insights are shareable and replicable. That's why the top research funders, such as the US National Science Foundation, embrace `open science principles <https://www.nsf.gov/public-access>`_.

But property data has long been an outlier. It is sensitive, identifiable, and often subject to privacy restrictions. Public records data is fragmented across thousands of local government systems. Researchers running large-scale studies require licenses from commercial data aggregators, which restrict data sharing.

As a result, property data is rarely published alongside scientific articles. Datasets that took analysts months or years to prepare become unavailable after use. Data processing pipelines become valuable assets but are often tightly guarded to stay competitive for research funding.

This worsens the credibility problem of the empirical social sciences, where data quality and methods are known to affect results, but publication bias favors overreporting of statistical significance, e.g. in `environmental <https://www.journals.uchicago.edu/doi/10.1093/reep/reaa011>`_ and `agricultural economics <https://onlinelibrary.wiley.com/doi/full/10.1002/aepp.13323>`_.

Publishing full analytical workflows in an open framework allows others to revisit, interrogate, and improve prior findings: a precondition for truly "standing on the shoulders of giants".


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


Extensible
~~~~~~~~~~

``openplaces`` makes it easy to integrate new datasets (tables, vectors, rasters) into parcel-level analyses.

New users :ref:`write recipes<writing_recipes>` to ingest new data and to harmonize datasets.

They can :ref:`contribute <contribute>` recipes to the public repository, receive credit, and allow others to build on their work.


Cross-platform
~~~~~~~~~~~~~~

``openplaces`` is designed to run on anyone's operating system:

- a government agency PC (Windows)
- a student's Macbook (OSx)
- a university research cluster (Linux)

It is written mostly in Python. It uses open-source packages for geospatial processing (GDAL, geopandas) and machine learning.


Scalable
~~~~~~~~

``openplaces`` is meant to facilitate large-scale, international, comparative property analyses.

You can run a city-wide analysis on your laptop - or deploy it on a research cluster to process satellite data for millions of locations.

In the US, we have attributed satellite-based data to 133 million parcels and used it for `land valuation <https://placeslab.org/research/#cost>`_ and `hedonic valuation <https://placeslab.org/research/#valuation>`_, producing tens of terabytes of trained models.