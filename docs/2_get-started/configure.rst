.. _configure:

Configure
=========

First-time setup
~~~~~~~~~~~~~~~~

Upon you first ``import openplaces``, you will be asked to configure your installation.

This is a required step. It's about where your data is stored and who will work with it.

We have prepared a notebook that walks you through the configuration and showcases functions for updating and removing it:

   :file:`notebooks/01_setup/01_first_steps.ipynb` `see on Github <https://github.com/chrnolte/openplaces/blob/main/notebooks/01_setup/01_first_steps.ipynb>`_

Open and run this notebook after :ref:`installing<install>` and activating your environment:

   .. code-block:: bash

      conda activate openplaces
      cd notebooks
      jupyter notebook

.. _directory_structure:

Directory structure
~~~~~~~~~~~~~~~~~~~

``openplaces`` uses different directories for different stages of the analytical pipeline: input data (external downloads, own raw data), scratch directories for intermediate data, canonical, analysis-ready data, output data, shared data, fitted models, and reports / publications.

This simplifies data sharing across different machines and cloud services and the setting of team and user-specific permissions. 

.. list-table::
   :widths: 15 25 15 45
   :header-rows: 1

   * - Name
     - Default
     - Shared/User
     - Description
   * - ``data_root``
     - :input:`None`
     - 🌍 Shared
     - Root directory for data, models, reports.

       If :input:`None`, the project code directory is used as the data root.
   * - ``core``
     - :file:`data/core`
     - 👤 User (multi-user)
     - Processed, analysis-ready data
   * - ``external``
     - :file:`data/external`
     - 🌍 Shared
     - Downloaded data from third party sources
   * - ``raw``
     - :file:`data/raw`
     - 🌍 Shared
     - Raw data from own data collection efforts
   * - ``cache``
     - :file:`data/cache`
     - 👤 User (multi-user)
     - Intermediate files generated during processing
   * - ``heap``
     - :file:`data/cache/_heap`
     - 👤 User (multi-user)
     - Freshly unzipped data, not yet with standard prefixes
   * - ``logs``
     - :file:`data/cache/_logs`
     - 👤 User (multi-user)
     - Logs from script runs with timing and metadata
   * - ``out``
     - :file:`data/out`
     - 👤 User (multi-user)
     - Output and results data
   * - ``share``
     - :file:`data/share`
     - 🌍 Shared
     - Shared data between users
   * - ``models``
     - :file:`models`
     - 👤 User (multi-user)
     - Trained and serialized models, model predictions, or model summaries
   * - ``reports``
     - :file:`reports`
     - 👤 User (multi-user)
     - Reports and figures

.. note::
   In single-user mode, all directories are at the same level (no user subfolders).
   In multi-user mode, user-specific directories are in ``data/<username>/``.

Credits to `Cookiecutter Data Science <https://cookiecutter-data-science.drivendata.org/>`_ (Carl Boettiger's lab @ Berkeley) for inspiring this directory structure.



Single vs. multi-user mode
~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration script will ask you to choose between single vs. multi-user mode for your data directories.

Single-user mode
----------------

Best when you're the only user.

**Directory structure:**

.. code-block:: text

   data_root/
   ├── data/
   │   ├── cache/         # Intermediate files (reproducible, deletable)
   │   │   ├── _heap/     # Freshly unzipped data
   │   │   └── _logs/     # Logs from runs with timing and arguments
   │   ├── core/          # Processed, analysis-ready data
   │   ├── external/      # External data (downloaded)
   │   ├── out/           # Outputs and results data
   │   ├── raw/           # Raw data
   │   └── share/         # Shared data
   ├── models/            # Models, trained and serialized, predictions
   └── reports/           # Reports and figures

* No user subfolders created
* Minimal config file created (commented template)
* Uses project defaults from ``openplaces.yaml``

Multi-user mode
---------------

Best for team projects where multiple people work on the same codebase.

**Setup process:**

1. Choose multi-user mode (option ``b``)
2. Accept default folder name (from your system username) or enter custom name
3. User-specific folders created for outputs

**Directory structure:**

.. code-block:: text

   data_root/
   ├── data/
   │   ├── external/              # 🌍 Shared
   │   ├── raw/                   # 🌍 Shared
   │   ├── share/                 # 🌍 Shared
   │   └── YourUsername/          # 👤 Yours
   │       ├── cache/             # 👤 Yours
   │       │   ├── _heap/         # 👤 Yours
   │       │   └── _logs/         # 👤 Yours
   │       ├── core/              # 👤 Yours
   │       └── out/               # 👤 Yours
   ├── models/                    # 🌍 Shared
   │   └── YourUsername/          # 👤 Yours
   └── reports/                   # 🌍 Shared
       └── YourUsername/          # 👤 Yours

* User subfolders for: ``cache``, ``heap``, ``logs``, ``core``, ``out``
* Shared folders for: ``external``, ``raw``, ``share``
* User subfolders in ``models``, ``reports``


Location of configuration files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``openplaces`` uses a hierarchical configuration system to customize data directories and settings.

Configuration files are used in priority order: **user > project > defaults**.

1. **User configuration** (highest priority)
   
   * A user configuration file is created interactively upon the first time a new user runs ``import openplaces``.
     
     Its location depends on your operating system:

     * Windows: ``%APPDATA%\openplaces\config.yaml``
     * macOS: ``~/Library/Application Support/openplaces/config.yaml``
     * Linux: ``~/.config/openplaces/config.yaml``
   * It contains user-specific overrides to the project configuration and is not committed to version control (``git``).

2. **Project configuration** (default values)
   
   * Location: ``/openplaces.yaml`` (root directory of repository).
   * Project-wide defaults committed to version control.
   * Shared by all users of an installation.

3. **Built-in defaults** (fallback)
   
   * Hardcoded in ``config.py``.

