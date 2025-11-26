Configure your file system
==========================

Before you start using ``openplaces``, you need to configure your installation.

After :ref:`installing your environment <install>`, the first call of ``import openplaces`` (in a Jupyter notebook or directly in Python) will launch an interactive configuration script.

The first th that allows you to choose your directory structure.


Standard directories
~~~~~~~~~~~~~~~~~~~~

It is usually helpful to have different directories for different stages in your analytical pipeline: input data (external downloads, own raw data), scratch directories for intermediate data, canonical, analysis-ready data, output data, shared data, fitted models, and reports / publications. This simplifies data sharing across different machines and cloud services and the setting of team and user-specific permissions. 

Loosely following `Cookiecutter Data Science <https://cookiecutter-data-science.drivendata.org/>`_ (Boettinger @ Berkeley), ``openplaces`` works with these directories:

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


Configuration files
~~~~~~~~~~~~~~~~~~~

``openplaces`` uses a flexible, hierarchical configuration system that makes it easy to set up and customize your data directories and settings.

Configuration files are located in priority order: **user > project > defaults**.

1. **User configuration** (highest priority)
   
   * File location: ``~/.config/openplaces/config.yaml``
     
     The file is created interactively upon the first time a new user runs ``import openplaces``.
     
     * macOS: ``~/Library/Application Support/openplaces/config.yaml``
     * Linux: ``~/.config/openplaces/config.yaml``
     * Windows: ``%APPDATA%\openplaces\config.yaml``
   * User-specific overrides to the project configuration.
   * Not committed to version control (``git``).

2. **Project configuration** (default values)
   
   * Location: ``./openplaces.yaml`` in project root directory
   * Project-wide defaults
   * Committed to version control
   * Shared by all users of an installation

3. **Built-in defaults** (fallback)
   
   * Hardcoded in ``config.py`` with platform-appropriate paths


First time setup
~~~~~~~~~~~~~~~~

Upon your first use of ``import openplaces``, you'll get an interactive setup prompt that asks you to define your directories and whether you're installing in single or multi-user mode.

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

* User-specific subfolders for: ``cache``, ``heap``, ``logs``, ``core``, ``out``
* Shared folders for: ``external``, ``raw``, ``share``
* User subfolders in ``models``, ``reports``
* Full config file created with your directory settings

Basic usage
~~~~~~~~~~~

After executing the places setup scripts and creating the ``openplaces`` environment, open and run this Jupyter notebook:

   :file:`notebooks/01_setup/01_first_steps.ipynb`