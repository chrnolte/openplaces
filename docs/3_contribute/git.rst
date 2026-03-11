Contributor workflow
====================

``openplaces`` is a `public repository on Github <https://github.com/chrnolte/openplaces>`_.

Understanding the contribution workflow helps you make clean and well-tested contributions to the ``main`` branch with `git <https://git-scm.com/install/windows>`_.

We'll assume you already :ref:`installed  <install>` ``openplaces``, so you have a local clone of the repository and a functioning ``conda`` environment. Otherwise, :ref:`do that first <install>`.


Develop on your own branch
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to make contributions to the ``openplaces`` source code:

1. Create your own branch for development purposes.

   As the name for your new branch, pick a shorthand that identifies you to collaborators and that you are okay with sharing with the world, e.g., your Github username or the left side of your professional email address.

2. Edit your code to create your contribution.

3. Test your code.

4. Format your code (see below).

5. Commit your code to your branch.

   ``git commit`` will trigger final code style checks.

6. Submit a pull request to have your edits reviewed.

   Once your contributions pass review, they become part of the ``main`` branch, and your Github badge will appear on the public list of collaborators.


Activate your environment
~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure your ``openplaces`` environment (see :ref:`install`) is active (``conda activate openplaces``) before using ``git``.

Commands that need ``nbstripout`` to strip notebook outputs (see below):

- ``git status``
- ``git add``
- ``git diff``

Commands that need ``pre-commit`` and ``ruff`` for code style checks and fixes:

- ``git commit``


Format code in scripts and notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clean code makes contributions more readable and interpretable.

``openplaces`` adopts `ruff <https://docs.astral.sh/ruff/>`_ to format and lint Python code in both scripts and notebooks before committing them to the repository (any branch).

We adopt ``ruff``'s default specifications and add two more:

1. We use single quotes for strings (``'hello'``, not ``"hello"``).
   
   It looks cleaner and is faster to type on many keyboards.

2. we identify ``openplaces`` as a "known first party" in the sorting of ``import`` statements.

   Imports from the repository appear in the last (third) section, after standard libraries and external packages. This adds clarity to the import sections.

To edit these and other settings, find and edit these two files:

- :file:`notebooks/ruff.toml` to format Jupyter notebook cells with ``jupyter-ruff``, see next section.
- :file:`pyproject.toml` for command-line ``ruff`` and ``pre-commit`` hooks.

Jupyter notebooks
-----------------

``jupyter-ruff`` makes ``ruff`` formatting available in Jupyter notebooks:

- Right-click to the left of the cell
- Select :gui:`Format Cell using Ruff`.

You can choose to automatically format notebooks when saving:

- Select :gui:`Settings` from the menu of your Jupyter homepage (directory view). You can access it from any notebook by clicking the Jupyter icon on the top left.
- Click :gui:`Settings Editor`.
- Find :gui:`Jupyter Ruff` in the left-hand menu.
- Check the checkbox :gui:`Format on Save`.

.. note::

   Turning on :gui:`Format On Save` can trigger unwanted error popups if automated saving occurs while you are editing a cell and ``ruff`` fails to format the incomplete code.


Python scripts
--------------

Before you ``git add`` any changes to Python scripts and Jupyter notebooks, make sure your code is compliant with the style format by running ``ruff check`` or ``ruff format`` in your :gui:`Terminal` or :gui:`Anaconda Prompt`.

.. code-block:: bash

   # Check a file
   ruff check yourfolder/yourfilnename.ext

   # Check a folder
   ruff check yourfolder

   # Format a file (fixes many issues)
   ruff format yourfolder/yourfilnename.ext

This avoids


Strip outputs from notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``openplaces`` uses ``nbstripout`` to remove outputs from Jupyter notebooks (e.g., figures, printed text, warnings, errors, timestamps) before adding them to the repository.

This should occur automatically when using ``git status``, ``git diff``, or ``git add`` on notebooks. It is configured in ``.gitattributes`` in the repository root.

You just need to make sure your ``openplaces`` environment is active.

.. note::

   User switching between machines or operating systems can lose their connection to ``nbstripout``. If you encounter ``nbstripout``-related errors in spite of your ``openplaces`` environment being active, re-running ``nbstripout --install`` frequently resolves the issue.


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

