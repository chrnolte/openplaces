.. openplaces

Why openplaces?
===============

``openplaces`` is designed for research that links land, buildings, rights, and values.

For instance, at the `placeslab <https://placeslab.org>`_, we use it to:

- Estimate `effects of conservation interventions <https://placeslab.org/research/#effectiveness>`_,
- Value `environmental amenities and risks <https://placeslab.org/research/#valuation>`_,
- Value land for `conservation planning <https://placeslab.org/research/#cost>`_ and property taxation,
- Estimate economic damage from natural hazards, e.g. `from hurricanes <https://www.drc.udel.edu/cheer/>`_.

What these analyses have in common is that they benefit from decision-grade data on land ownership and buildings, often requiring dozens of variables, ideally for tens of millions of properties.

Generating, maintaining, and analyzing such data is no simple feat. The ``openplaces`` project tries to make it easier.


Open source
~~~~~~~~~~~

.. image:: images/open_source.png
  :width: 58
  :alt: Open source logo
  :align: right
  :class: padded-image

``openplaces`` is open source.

It is published `on Github <https://github.com/chrnolte/openplaces>`_ under a permissive `Apache 2.0 license <https://www.apache.org/licenses/LICENSE-2.0>`_.

Science lift everyone's boat when data assembly and analysis are publicly accessible and reproducible. Leading public funders - the `US National Science Foundation <https://www.nsf.gov/public-access>`_, the `European Research Council <https://erc.europa.eu/manage-your-project/open-science>`_, `Japan’s Science and Technology Agency <https://www.jst.go.jp/EN/about/strategy.html>`_, and Brazil's `FAPESP <https://www.fapesp.br/openscience/en>`_ - expect open access, data management plans, and public sharing of research outputs. Global frameworks like the `UNESCO Recommendation on Open Science <https://www.unesco.org/en/open-science/about>`_ and initiatives like `cOAlition S <https://www.coalition-s.org/>`_  promote the idea of code and datasets as public infrastructure.

But land ownership and property data is sensitive, often identifiable, and subject to privacy protections. Public records are created by thousands of government agencies in thousands of formats. Researchers interested in large-scale analyses have to obtain licenses for data from commercial real estate data aggregators who discourage public sharing. Peer-reviewed journals in land, agricultural, environmental, or housing economics rarely expect publication of data.

Data sets that took analysts months or years to put together become unavailable after use - and the next person has to start from scratch.

This worsens the credibility problem of the empirical social sciences, e.g. in `environmental <https://www.journals.uchicago.edu/doi/10.1093/reep/reaa011>`_ and `agricultural economics <https://onlinelibrary.wiley.com/doi/full/10.1002/aepp.13323>`_. When `data quality and methods affect results <https://le.uwpress.org/content/100/1/200>`_, publication bias favors novel findings of statistical significance, and re-analyses are rare, professional competition creates incentives for questionable research practices.

Easy access to replicable methods can counter this trend. Publishing the codebase of research workflows allows others to revisit results, interrogate findings, estimate new uncertainties, share bug reports, and contribute improved methods.

Open source code creates shortcuts onto the shoulders of giants.


Scalable
~~~~~~~~

``openplaces`` is meant to facilitate large-scale analyses.

You can develop a city-wide analysis on your laptop.

You can then parallelize your script on a research cluster to process satellite data and analyze patterns for millions of locations.

At Boston University's `placeslab <https://placeslab.org>`_, we have attributed satellite-based data to 133 million parcels in the US. We used the data for `land valuation <https://placeslab.org/research/#cost>`_ and `hedonic valuation <https://placeslab.org/research/#valuation>`_, producing tens of terabytes of trained models on our research cluster.


Global referencing
------------------

Most empirical studies of property data produce insights for a given place:

- a city, town, or municipality
- a state, province, or department
- a country

How generalizable are these insights? Do they replicate elsewhere?

Often, they do not. For instance, economists found that transferring environmental benefit estimates from one location to another is subject to large errors (e.g., a `median of 39% <https://doi.org/10.1016/j.jeem.2013.03.001>`_).

Replicating studies in multiple locations can generate insights that are more context-sensitive, statistically robust, and representative.

Important from a privacy perspective: this shared analysis can happen without sharing the underlying, sensitive property data. Any analyst can use ``openplaces`` to build data sets for their home location in their own secure computing environment, analyze them with shared methods, and publish replicable findings without ever providing access to the underlying data.

To simplify data integration and exchange, ``openplaces`` ships with a globally consistent, hierarchical, referencing system of :ref:`administrative units<administrative_units>`:

- :input:`US` -> United States
- :input:`CO-AN` -> Antioquia, a department of Colombia
- :input:`US-CA-LA` -> Los Angeles, a county in the state of California, United States
- :input:`US-MA-MI-CA` -> Cambridge, a city in Middlesex county, Massachusetts, United States

All data in the ``openplaces`` :ref:`directory structure <directory_structure>` is organized by geography. This makes it easy to restrict or expand access to sensitive data on different locations to different groups of collaborators.


Cross-platform
--------------

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


Extensible
----------

``openplaces`` makes it easy to integrate new datasets through :ref:`recipes <recipes>`.

Recipes hold machine-readable instructions for data ingestion (download & file handling, column renaming, partitioning, etc.) and data harmonization (e.g. merging multiple parcel or building datasets into one).

New users :ref:`write recipes<writing_recipes>` to ingest new datasets or create new harmonized versions.

Recipes can be :ref:`contributed <contribute>` to the public repository. Contributors receive badges and extra traffic to their published analyses.


