.. openplaces

.. _cheer_research:

Research
========

What the inventory takes from the work of others.

From the CHEER team
~~~~~~~~~~~~~~~~~~~

.. _Askari 2026:

Askari 2026
^^^^^^^^^^^

Askari M (2026). Imputation model selection for CoPe-CHEER: North
Carolina; and Imputing all nine building attributes in CHEER Places
North Carolina. Technical memos, 2026-09-10 and 2026-10-15.

- The imputation's learner (gradient-boosted trees) and its evaluation
  on whole counties held out.
- Validity repairs and gap-trained boosting, with their code.

:gh-file:`Code <src/openplaces/io/building_imputation_gaps.py>`

.. _Tackie-Otoo et al. 2026:

Tackie-Otoo et al. 2026
^^^^^^^^^^^^^^^^^^^^^^^

Tackie-Otoo N O, Askari M, Hadinata P, Davidson R A, Taciroglu E,
Hardy G (2026). Hurricane wind loss modeling using insurance claims
data. *Natural Hazards* 122: 300.
`doi:10.1007/s11069-026-08059-z <https://doi.org/10.1007/s11069-026-08059-z>`_

- The attributes a wind-loss model takes, which the inventory delivers.
- Manufactured homes as a class of their own.

.. _CHEER Inventory v0:

CHEER Inventory v0
^^^^^^^^^^^^^^^^^^

Soleimani H et al. CHEER Inventory v0, for North Carolina and Texas.

- Roof shape, foundation, construction, garage and story count, as
  evidence.

:gh-file:`Code <src/openplaces/recipes/US/NC/_all/building/cheer/v0/US-NC_building-cheer-v0.yaml>`

.. _CHEER Summer Scholars survey:

CHEER Summer Scholars survey
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Buildings in eastern North Carolina classified by hand.

- The reference for scoring occupancy (see the
  :ref:`validation annex <cheer_footprints_validation>`).

:gh-file:`Code <notebooks/02_ingest/footprints/prepare_cheer_ground_truth.ipynb>`

From outside CHEER
~~~~~~~~~~~~~~~~~~

.. _Lochhead et al. 2026:

Lochhead et al. 2026
^^^^^^^^^^^^^^^^^^^^

Lochhead M, Zsarnoczay A, Deierlein G (2026). Exposure matters: a
synthesis framework for high-resolution building inventory development.
*International Journal of Disaster Risk Reduction* 139: 106148.
`doi:10.1016/j.ijdrr.2026.106148 <https://doi.org/10.1016/j.ijdrr.2026.106148>`_

- Matching NSI and address points to footprints, in passes of
  increasing distance.
- Telling main buildings from outbuildings on a parcel.
- Splitting a parcel's value among its buildings by area and floors.
- Dwellings per occupancy class, where no source counts them.
- Collecting every source's value before choosing one.

:gh-file:`Code <src/openplaces/io/harmonizer/>`

.. _Cetiner et al. 2025:

Cetiner et al. 2025
^^^^^^^^^^^^^^^^^^^

Cetiner B, McKenna F, Yi S, Wang B, Manousakis I V (2025). BRAILS++
(v4.2.0). Zenodo.
`doi:10.5281/zenodo.17797364 <https://doi.org/10.5281/zenodo.17797364>`_

- Image scrapers, story-count detection and roof-type classification,
  for the optional recognition from imagery (BSD 3-Clause).

:gh-file:`Code <src/openplaces/io/enricher/detectors/>`

.. _Khanal et al. 2025:

Khanal et al. 2025
^^^^^^^^^^^^^^^^^^

Khanal K, Kaza N, Hino M, Sebastian A (2025). Characterizing
manufactured home parks in North Carolina: a computer vision based
approach. *EPB: Urban Analytics and City Science*.
`doi:10.1177/23998083251395471 <https://doi.org/10.1177/23998083251395471>`_

- Independent support for the footprint shapes used to find
  manufactured homes, and for pairing shapes with parcel records.

.. _Oostwegel et al. 2025:

Oostwegel et al. 2025
^^^^^^^^^^^^^^^^^^^^^

Oostwegel, Schorlemmer, Lingner, Evaz Zadeh (2025). OpenBuildingMap.
GFZ Data Services.
`doi:10.5880/GFZ.LKUT.2025.002 <https://doi.org/10.5880/GFZ.LKUT.2025.002>`_

- The first footprint layer (ODbL 1.0).

:gh-file:`Code <src/openplaces/recipes/_all/footprint/obm/2025/footprint-obm-2025.yaml>`
