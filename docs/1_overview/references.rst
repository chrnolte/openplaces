.. _references:

References
==========

``openplaces`` builds on published methods, public datasets, and open-source
software. This page collects what the project reuses, so that work can be
cited correctly and its terms of use checked.

If you use ``openplaces`` in research, cite the methods below alongside the
tool, and cite the individual data sources your recipes actually ingest.

Methods
-------

.. _lochhead_et_al_2026:

**Lochhead M, Zsarnóczay A, Deierlein G (2026).** Exposure matters: a
synthesis framework for high-resolution building inventory development.
*International Journal of Disaster Risk Reduction* 139: 106148.
`doi:10.1016/j.ijdrr.2026.106148 <https://doi.org/10.1016/j.ijdrr.2026.106148>`_

  Four procedures in the harmonize and curate stages are adopted from this
  paper, each citing the table or section it comes from at the point of use:

  - the four-pass point attribution and the point-to-footprint aggregation
    rules (:mod:`openplaces.io.harmonizer.links`, Table 3);
  - the occupancy-class to dwelling-unit mapping
    (:mod:`openplaces.io.harmonizer.attributes` and
    :mod:`openplaces.io.curator.imputers`, Table 3);
  - the land- and improvement-value apportionment cases
    (:mod:`openplaces.io.harmonizer.apportion`, Table 4);
  - the source-geometry classification vocabulary
    (:mod:`openplaces.core.schema`, §2.3).

.. _cetiner_et_al_2025:

**Cetiner B, McKenna F, Yi S, Wang B, Manousakis I V (2025).** BRAILS++
(v4.2.0). Zenodo.
`doi:10.5281/zenodo.17797364 <https://doi.org/10.5281/zenodo.17797364>`_

  Source of the image-based enrichment lane. ``openplaces`` adapts the
  BRAILS++ Google Satellite and Street View scrapers
  (:mod:`openplaces.io.scrapers`) and reproduces the story-count
  post-processing pipeline
  (:mod:`openplaces.io.enricher.detectors.n_stories`). Both are redistributed
  under BSD 3-Clause; see the ``NOTICE`` file at the repository root.

.. _khanal_et_al_2025:

**Khanal K, Kaza N, Hino M, Sebastian A (2025).** Characterizing manufactured
home parks in North Carolina: a computer vision based approach. *EPB: Urban
Analytics and City Science*.
`doi:10.1177/23998083251395471 <https://doi.org/10.1177/23998083251395471>`_

  Independent support for how ``openplaces`` identifies manufactured housing.
  Khanal et al. detect individual units from aerial imagery with an object
  detector, then filter and group them using building footprints and parcel
  records; ``openplaces`` reaches the same classes from footprint morphology
  and assessor records, without imagery. Three of their findings bear
  directly on choices made here:

  - **Footprint dimensions identify manufactured units.** They filter
    candidates to a length of 40--80 ft and a width of 12--18 ft, and screen
    on the diagonal of the minimum-rotated bounding box (37--96 ft
    single-wide, 41--82 ft double-wide). That size envelope implies an aspect
    ratio of roughly 2.2--6.7 and a footprint area of roughly 45--134 m²,
    which brackets the ``aspect_ratio >= 2.5`` and ``area <= 185 m²`` rule
    used by ``classify_manufactured_homes``.
  - **Three units is the threshold for a park.** They validated parcels
    holding at least three units, "based on the observation that many parcels
    with three or fewer detections were primarily used for agricultural and
    industrial purposes, and not as a manufactured housing park".
    ``flag_manufactured_home_communities`` arrives at the same cutoff.
  - **Parcel records materially improve footprint-based classification.**
    They cite Durst et al. (2021), where landscape metrics derived from
    footprints classified mobile homes at 91% accuracy, rising to 99% once
    parcel information was added --- the pairing ``openplaces`` relies on
    when it votes footprint geometry together with assessor land-use.

  They also report that cadastral datasets "are neither comprehensive nor
  current and are often inconsistent across counties", and find roughly three
  times as many parks in North Carolina as the largest public registry lists.
  Both are reasons this pipeline infers the class rather than trusting a
  single source, and are consistent with manufactured homes being the class
  for which ``openplaces`` finds the least corroborating evidence.

.. _tackie_otoo_et_al_2026:

**Tackie-Otoo N O, Askari M, Hadinata P, Davidson R A, Taciroglu E, Hardy G
(2026).** Hurricane wind loss modeling using insurance claims data. *Natural
Hazards* 122: 300.
`doi:10.1007/s11069-026-08059-z <https://doi.org/10.1007/s11069-026-08059-z>`_

  Shows what a wind-loss model consumes, and why these particular attributes
  are curated. Its exposure and vulnerability features are square footage,
  building age, occupancy, construction type, units in structure, number of
  stories and building value --- which correspond to ``area_m2``,
  ``year_built``, ``occupancy_type``, ``construction_type``, ``n_dwellings``,
  ``n_stories`` and ``structure_value`` in the curated output.

  Manufactured housing is not incidental to such a model: "Mobile home" is
  one of six construction-type levels, covering some 11,000 insured buildings
  per hurricane in their data. Misclassifying a manufactured home therefore
  moves it into the wrong vulnerability class, which is the practical reason
  the occupancy vote treats it as a distinct decision rather than a variant
  of single-family.

Models
------

The story-count detector uses an EfficientDet-D4 architecture. Pretrained
weights are downloaded at run time from
`Zenodo record 4421613 <https://zenodo.org/record/4421613>`_ and are not
redistributed with this package. The inference engine itself is not bundled
--- see :mod:`openplaces.io.enricher.detectors.n_stories`.

Data sources
------------

Every ingest recipe declares its own source, and the recipe tree at
:gh-file:`src/openplaces/recipes` is the authoritative per-source record:
each recipe's ``source`` block carries the portal URL, and where one exists,
the DOI. At the time of writing the tree spans **81 distinct sources across
124 recipes**.

They fall into three groups, which differ in how they should be cited:

**Published datasets with a DOI or canonical citation.** Cite the dataset as
you would a paper. Example: HISDAC-US, the historical settlement data
compilation, recorded in its recipe as
`doi:10.7910/DVN/PKJ90M <https://doi.org/10.7910/DVN/PKJ90M>`_.

**Institutional and programme datasets.** National and international
products --- among them the National Structure Inventory, FEMA USA
Structures, Overture Maps, Microsoft Building Footprints, OpenBuildingsMap,
the USGS 3DEP elevation programme, the U.S. Census Bureau's TIGER and ACS
products, IPUMS NHGIS, Eurostat/GISCO, the JRC Global Human Settlement Layer,
and GADM. Cite the product and version; each recipe records the portal it was
retrieved from.

**Government open-data portals.** The majority: state and county assessor,
GIS, and land-records offices. These are public records rather than
publications, so the citation is the portal URL and the date retrieved, both
of which the recipe records.

.. note::

   Source metadata is incomplete. Of the 81 sources in the recipe tree, one
   declares a DOI and none declares a licence. Portal URLs are recorded
   almost everywhere, so provenance is traceable, but terms of use are not
   yet captured in a form that can be checked automatically. Two sources are
   known to carry conditions that matter --- GADM is licensed for
   non-commercial use only, and Google imagery has caching restrictions ---
   and a reader should not assume the remainder are unrestricted.
   Contributors adding a recipe should record the source's licence and
   citation alongside its URL.

Software
--------

``openplaces`` depends on the open-source geospatial and scientific Python
stack, principally GeoPandas, Shapely, pyproj, pyogrio, rasterio, DuckDB,
pandas, NumPy, SciPy and PyArrow, together with ``exactextract`` and
``rasterstats`` for zonal statistics, ``usaddress`` and ``rapidfuzz`` for
address parsing and matching, and ``openlocationcode`` for the Open Location
Code identifiers used as stable entity keys. Declared versions are in
:gh-file:`pyproject.toml` and :gh-file:`environment.yml`.

Third-party code redistributed inside this repository, with its licence, is
listed in the ``NOTICE`` file at the repository root.
