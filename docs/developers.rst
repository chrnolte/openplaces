Developers
==========

Install your environment
~~~~~~~~~~~~~~~~~~~~~~~~

``openplaces`` comes with a handy setup script that installs the ``openplaces`` environment with ``mamba`` or ``conda``. You need to have at least one of them installed.

.. code-block:: bash

   python dev.py setup

To remove your environment, use:

.. code-block:: bash

   python dev.py clean


Code style
~~~~~~~~~~

Jupyter notebooks
-----------------

Format with ``black`` and ``isort``

-  :gui:`Menu` > :gui:`Settings` > :gui:`Settings Editor`
-  Click on :gui:`JSON Settings Editor` (Advanced Settings) in the upper right corner.
-  Find :gui:`Jupyterlab Code Formatter` (search for :input:`code format`).
-  Make sure the :gui:`User Preferences` contains:

.. code-block:: json

   {   

       "preferences": {
           "default_formatter": {
               "python": [
                   "black",
                   "isort"
               ]
           }
       },
       "black": {
           "line_length": 88,
           "string_normalization": false
       },
       "isort": {
           "multi_line_output": 3,
           "include_trailing_comma": true,
           "force_grid_wrap": 0,
           "use_parentheses": true,
           "ensure_newline_before_comments": true,
           "line_length": 88,
           "known_first_party": ["openplaces"],
       },
       "formatOnSave": true,

   }

-  Click on the "Save" (disc) symbol in the upper right corner
-  Now you can format Jupyter cells with :input:`Ctrl+Shift+I`


Python files
------------

Run `python dev.py format` (uses ``ruff``)
