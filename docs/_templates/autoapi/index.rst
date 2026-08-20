Reference
=========

This is the reference documentation of:

- the catalog of **recipes** for data ingestion and processing

  Auto-generated from the recipe files in :gh-file:`src/openplaces/recipes`

- the ``openplaces`` package and API

  `Auto-generated <https://github.com/readthedocs/sphinx-autoapi>`_ from the source code in :gh-file:`src/openplaces`

.. toctree::
   :titlesonly:
   :maxdepth: 2

   Recipe catalog </recipes>
   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

