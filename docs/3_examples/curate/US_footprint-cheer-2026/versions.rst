.. openplaces

.. _cheer_versions:

Versions
========

A bundle's version is the imputation version in its filled cells'
``_source`` (``building-imputation-v0.8+open+imputed``) and in its
licence notice. Changes to the inventory itself (sources, votes, value
rules) ship with the next imputation version.

v0.8 (from 2026-10-28)
~~~~~~~~~~~~~~~~~~~~~~

- The delivery is named
  :gh-file:`US_footprint-cheer-2026 <src/openplaces/recipes/US/_all/footprint/cheer/2026/US_footprint-cheer-2026.yaml>`;
  files are
  ``US-NC_footprint-cheer-2026*`` and ``US-TX_footprint-cheer-2026*``.
- NSI's replacement cost and depreciated replacement cost ship in their
  own columns, filled where NSI has no point.
- ``structure_value_zero_reason`` says why a structure value is 0, and
  the models treat each reason as described on :doc:`imputation`.
- A recorded 0 on the roll passes through to the building.
- A dwelling on a lot where another building holds the roll's value is
  not filled, so the lot's value is not counted twice.
- Small single-family and multi-family buildings with a stated zero
  (under 120 m2) are filled.
- Each filled attribute draws its own random values.

v0.7 (2026-10-26)
~~~~~~~~~~~~~~~~~

- Donor counties are weighted by the distance between county centers
  alone (250 km bandwidth, none beyond 400 km).

v0.6 (2026-10-24)
~~~~~~~~~~~~~~~~~

- Pender County's appraisal year set to its 2026 reappraisal.
- The models no longer use land-market inputs (regional land values),
  whose source is held pending review.
- Every delivery compares each county's single-family roll level with
  the Census median home value and warns outside 0.3 to 1.3.
- The random seed is recorded; the same inputs and code give the same
  values.
