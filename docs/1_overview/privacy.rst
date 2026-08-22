.. _privacy:

Privacy
=======

``openplaces`` is a general-purpose, open-source data processing library. It is designed to run entirely on your own local computer or private cluster.

Data control
------------

The project maintainers and contributors do not host, store, access, or distribute any personal data. Any property records, parcel boundaries, or other datasets that you process remain on your own storage infrastructure.

If you use ``openplaces`` to process datasets containing information about identifiable individuals, you are the sole data controller. You must ensure you have a lawful basis to obtain and process this data under the privacy laws of your jurisdiction. These frameworks include the European Union's GDPR, the U.S. CCPA/CPRA, and comparable state laws.

Refer to the project's :gh-file:`DISCLAIMER.md` for the authoritative terms regarding data control, user responsibilities, and legal disclaimers.

Repository safety
-----------------

To maintain a secure and compliant project, the public ``openplaces`` repository must never contain real personal information of individuals. This includes:

* Real names
* Real street addresses or house numbers
* Phone numbers or email addresses
* National or state identification numbers
* Cadastral numbers, land registry codes, or parcel IDs that map directly to real individuals

All examples, test fixtures, docstrings, and recipes in this repository use fabricated values. If you contribute to this project, you must ensure that no real personal records are included in your pull requests. Refer to the :ref:`no_personal_data` guide for detailed instructions on fabricating test data and scanning your commits.
