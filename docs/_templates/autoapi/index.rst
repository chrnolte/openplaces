Reference
=========

This is the reference documentation of:

- the ``openplaces`` package and API

  `Auto-generated <https://github.com/readthedocs/sphinx-autoapi>`_ from the source code in :gh-file:`src/openplaces`

- the catalog of **recipes** for data ingestion and processing

  Auto-generated from the recipe files in :gh-file:`src/openplaces/recipes`

.. toctree::
   :titlesonly:
   :maxdepth: 2

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}
   Recipe catalog </recipes>

