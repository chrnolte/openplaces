Reference
=========

Documentation of the ``openplaces`` source code.

On overview of modules, classes, and functions.

Based on inline documentation in :gh-file:`src/openplaces`.

.. toctree::
   :titlesonly:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

.. rst-class:: op-autoapi-credit

   Created with `autoapi <https://github.com/readthedocs/sphinx-autoapi>`_.
