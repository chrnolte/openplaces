Getting started
~~~~~~~~~~~~~~~

Code on your own branch
-----------------------

If you want to make edits to the ``openplaces`` source code:

1. Create your own branch for development purposes (use a shorthand that identifies you, e.g. your Github username, or the left part of your professional / BU email address).

2. Change your code.

3. Test your code.

4. Submit a pull request to have your code change reviewed. If they pass review, they'll become part of the ``main`` branch of the repository.


Code style
----------

``openplaces`` code is formatted using the ``black`` and ``isort`` code formatters.

The only exception to their rulesets is that ``openplaces`` uses single quotes for strings (``'hello'`` not ``"hello"``), as it's cleaner & faster to type on US keyboards.


Jupyter notebooks
^^^^^^^^^^^^^^^^^

``black`` and ``isort`` come preinstalled with the ``openplaces`` ``conda`` environment (see :ref:`install`):

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
-  Now you can format Jupyter cells with :input:`Ctrl + Shift + I`


Python files
^^^^^^^^^^^^

Run ``python dev.py format`` (uses ``ruff``)
