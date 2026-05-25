Contributor workflow
====================

``openplaces`` is `public on Github <https://github.com/chrnolte/openplaces>`_.

Understanding the contributor workflow helps you make clean and well-tested contributions.

We assume you have :ref:`installed  <install>` ``openplaces``: you have a local clone of the repository and a ``conda`` environment on your machine. Otherwise, :ref:`do that first <install>`.

We also assume prior knowledge of `git <https://git-scm.com/install/windows>`_.

Best practices
~~~~~~~~~~~~~~

We adopt the `Git version control best practices <https://about.gitlab.com/topics/version-control/version-control-best-practices/>`_ from Gitlab:

* `Make incremental, small changes <https://about.gitlab.com/topics/version-control/version-control-best-practices/#make-incremental-small-changes>`_
* `Keep commits atomic <https://about.gitlab.com/topics/version-control/version-control-best-practices/#keep-commits-atomic>`_
* `Develop using branches <https://about.gitlab.com/topics/version-control/version-control-best-practices/#develop-using-branches>`_
* `Write descriptive commit messages <https://about.gitlab.com/topics/version-control/version-control-best-practices/#write-descriptive-commit-messages>`_



Develop using branches
~~~~~~~~~~~~~~~~~~~~~~

We use the `feature branching <https://about.gitlab.com/topics/version-control/version-control-best-practices/>`_ strategy: one branch per feature.

If you wish to make a contribution to ``openplaces``, follow these steps:

1. Create your own branch:

   .. code-block:: bash

      git pull origin main
      git switch -c create-new-feature  # pick a name

   Pick a short name that identifies the feature to collaborators.

2. Edit your code.

3. Test your code.

4. :ref:`Format your code <format_your_code>`.

5. Commit your changes to your branch:

   .. code-block:: bash

      git add <filename>
      git commit -m "Create new feature"

   This will trigger :ref:`automatic code style checks <pre_commit_hooks>`.

6. Push your changes to create a pull request.

   Before you push, pull recent changes to the repository. This ensures your commits reflect the most recent version of the repository.

   .. code-block::

      git pull --rebase origin main

   The first time you push, you have to register the branch upstream:

   .. code-block::

      git push -u origin create-new-feature

   The next time you push, the branch is already registered:

   .. code-block::

      git push

7. If your contributions pass review, they become part of the ``main`` branch. Your Github badge will appear on the public list of collaborators.


Activate your environment
~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure your ``openplaces`` environment is active before using ``git``:

.. code-block::

   conda activate openplaces

This is important because ``git`` needs:

- ``nbstripout`` to :ref:`strip notebook outputs <nbstripout>` 
- ``pre-commit`` and ``ruff`` for enforcing :ref:`code style checks <pre_commit_hooks>`.


.. _format_your_code:

Format your code
~~~~~~~~~~~~~~~~

Clean code makes contributions more readable and interpretable.

``openplaces`` adopts `ruff <https://docs.astral.sh/ruff/>`_ to format and lint Python code in both scripts and notebooks before committing them to the repository (any branch).

We adopt ``ruff``'s default specifications and add two more:

1. Use single quotes for strings (``'hello'``, not ``"hello"``).
   
   It looks cleaner and is faster to type on many keyboards.

2. Identify ``openplaces`` as a "known first party" to ``isort``.

   This ensures that imports from ``openplaces`` appear in a separate (third) section after standard Python libraries (1st) and external packages (2nd).

To edit these and other settings, find and edit these two files:

- :gh-file:`notebooks/ruff.toml` to format Jupyter notebook cells with ``jupyter-ruff``, see next section.
- :gh-file:`pyproject.toml` for command-line ``ruff`` and ``pre-commit`` hooks.


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

Before you ``git add`` any changes to Python scripts and Jupyter notebooks, make sure your code is compliant with the style format.

Do do so, run ``ruff check`` or ``ruff format`` in your :gui:`Terminal`:

.. code-block:: bash

   # Check a file
   ruff check yourfolder/yourfilename.ext

   # Check a folder
   ruff check yourfolder

   # Format a file (fixes many issues)
   ruff format yourfolder/yourfilename.ext

Once the code passes these checks, it will also pass the :ref:`pre-commit hooks <pre_commit_hooks>`.


.. _nbstripout:

Strip outputs from notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``openplaces`` uses ``nbstripout`` to remove outputs from Jupyter notebooks (e.g., figures, prints, warnings, errors) before adding them to the repository.

This will occur automatically: it is set in ``.gitattributes`` in the repository root. You just need to make sure your ``openplaces`` environment is active.

.. note::

   User switching between machines or operating systems can lose their connection to ``nbstripout``. If you encounter errors related to ``nbstripout`` while your ``openplaces`` environment is active, re-running ``nbstripout --install`` frequently resolves the issue.


.. _pre_commit_hooks:

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

