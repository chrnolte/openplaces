Contributor workflow
====================

``openplaces`` is a `public repository on Github <https://github.com/chrnolte/openplaces>`_.

Understanding the contribution workflow helps you make clean and well-tested contributions to the ``main`` branch with `git <https://git-scm.com/install/windows>`_.

We'll assume you already :ref:`installed  <install>` ``openplaces``, so you have a local clone of the repository and a functioning ``conda`` environment. Otherwise, :ref:`do that first <install>`.


Develop on your own branch
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to make edits to the ``openplaces`` source code:

1. Create your own branch for development purposes (use a shorthand that identifies you, e.g. your Github username, or the left part of your professional / BU email address).

2. Edit your code.

3. Test your code.

4. Commit your code to your branch. ``git commit`` triggers code style checks.

5. Submit a pull request to have your edits reviewed. If they pass, they'll become part of the ``main`` branch.


Activate your environment
~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure your ``openplaces`` environment (see :ref:`install`) is active (``conda activate openplaces``) before using ``git`` (e.g., ``git add``, ``git diff``, ``git commit``).

Otherwise, you'll get errors if the Python packages ``nbstripout`` or ``pre-commit`` can't be found.


Format Python code
~~~~~~~~~~~~~~~~~~

Clean code makes contributions more readable and interpretable (e.g., when using ``git diff``).

``openplaces`` uses ``ruff`` to format and lint code


Jupyter notebooks
-----------------

``jupyter-ruff`` makes ``ruff`` formatting available in Jupyter notebooks:

- Right-click to the left of the cell
- Select :gui:`Format Cell using Ruff`.

You can choose to automatically format notebooks when saving:

- Selecting :gui:`Settings` from the menu of your Jupyter homepage (directory view)
- :gui:`Settings Editor`
- Find :gui:`Jupyter Ruff` in the left-hand menu.
- Check the checkbox :gui:`Format on Save`.

.. note::

   Turning on :gui:`Format On Save` can trigger unwanted error popups if automated saving occurs while you are editing a cell and ``ruff`` fails to format the incomplete code.


Python scripts
--------------

Before you ``git add`` any changes to Python scripts, check whether your code is ``ruff`` compliant in your :gui:`Terminal` or :gui:`Anaconda Prompt`.

.. code-block:: bash

   # Check a file
   ruff check yourfolder/yourfilnename.ext

   # Check a folder
   ruff check yourfolder


Remove outputs from notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``openplaces`` uses ``nbstripout`` to remove outputs from Jupyter notebooks (e.g., figures, printed text, warnings, errors, timestamps) before adding them to the repository.

This should occur automatically: ``.gitattributes`` in the repository root is configured to run ``nbstripout`` for all notebook files (``.ipynb``) when using ``git diff`` or ``git add``.


Pass the pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~~

``openplaces`` uses pre-commit hooks to make sure your code is compliant with the ``ruff`` code style before it becomes part of the repository.

These hooks are triggered when you try to commit changes to the repository:

.. code-block::

   git add filenames.here
   git commit -m "Your commit message here."

``pre-commit`` will install a temporary environment and test your code against ``ruff`` code style.

``ruff`` will implement non-risky fixes. It will also tell you about any issues that weren't fixed and that you have to edit manually.

If fixes were implemented (automatically or by you), you need to re-add the edited files and re-commit them, until you pass all checks.


Edit the code style settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``openplaces`` code is formatted and linted with ``ruff``.

Code style settings are defined in:

- :file:`pyproject.toml` for command-line ``ruff`` and ``pre-commit`` hooks.
- :file:`notebooks/ruff.toml` to format Jupyter notebook cells with ``jupyter-ruff`` 

We use the defaults, but change two additional settings:

1. we use single quotes for strings (``'hello'`` not ``"hello"``), as it looks cleaner and is faster to type on US keyboards.

2. we identify ``openplaces`` as a "known first party" in the sorting of ``import`` statements, such that these imports appear in a third section after standard libraries and installed external packages. This adds clarity to the sections.