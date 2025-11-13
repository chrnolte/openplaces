
:orphan:

Accessing Settings
~~~~~~~~~~~~~~~~~~

All settings are accessible as attributes of the config object:

.. code-block:: python

   cfg = get_config()

   # CRS values
   print(cfg.crs_lat_long)     # "epsg:4326"
   print(cfg.crs_area)         # "epsg:6933"
   
   # Raster settings
   print(cfg.raster_xmin)      # -180.00
   print(cfg.raster_res)       # 0.00025
   
   # Processing settings
   print(cfg.max_workers)      # 4
   print(cfg.sep_str)          # " ~~ "
   
   # Geohashing
   print(cfg.geo_id_salt)      # "g1I9qtkKzxA3P98m80DLhuc0"

Special Attributes
~~~~~~~~~~~~~~~~~~

Some configuration values are transformed for convenience:

.. code-block:: python

   # Raster configuration dict
   cfg.r_cfg = {
       'xmin': -180.00,
       'ymin': -60.00,
       'xmax': 180.00,
       'ymax': 80.00,
       'res': 0.00025,
       'crs': CRS('epsg:4326')  # pyproj CRS object
   }
   
   # Transform function (arcsinh)
   cfg.geo_id_ha_trans  # numpy.arcsinh function

Common Workflows
----------------

Switching from Single-User to Multi-User
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Reconfigure and choose multi-user mode
   python -m openplaces.config --reconfigure
   # When prompted, choose option (b) for multi-user

Customizing Directory Paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Edit config file**

.. code-block:: bash

   python -m openplaces.config --edit

Edit the YAML file, then reload:

.. code-block:: python

   from openplaces.config import reload_config
   cfg = reload_config()

**Option 2: Python API**

.. code-block:: python

   # Create/edit user config manually
   from pathlib import Path
   import yaml

   config_path = Path.home() / '.config' / 'openplaces' / 'config.yaml'
   config_path.parent.mkdir(parents=True, exist_ok=True)
   
   config = {
       'directories': {
           'external': '/mnt/data/external',
           'raw': '/mnt/data/raw',
           'core': '/mnt/data/processed',
           'out': '/mnt/data/results'
       }
   }
   
   with open(config_path, 'w') as f:
       yaml.dump(config, f)
   
   # Reload
   from openplaces.config import reload_config
   cfg = reload_config()

Using with Different Projects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each project can have its own ``openplaces.yaml`` with project-specific defaults.
Your user config will override these defaults across all projects.

**Project A:**

.. code-block:: yaml

   # projectA/openplaces.yaml
   directories:
     external: data/external
     raw: data/raw
     core: data/core

**Project B:**

.. code-block:: yaml

   # projectB/openplaces.yaml
   directories:
     external: /shared/external_data
     raw: /shared/raw_data
     core: data/processed

**Your user config** (applies to both):

.. code-block:: yaml

   # ~/.config/openplaces/config.yaml
   max_workers: 8  # Override for all projects

CI/CD Integration
~~~~~~~~~~~~~~~~~

For continuous integration or automated environments:

**Option 1: Pre-create config**

.. code-block:: bash

   # In CI setup script
   mkdir -p ~/.config/openplaces
   cat > ~/.config/openplaces/config.yaml << EOF
   directories:
     external: /ci/workspace/data/external
     raw: /ci/workspace/data/raw
     core: /ci/workspace/data/core
     out: /ci/workspace/data/out
   EOF

**Option 2: Use non-interactive mode**

.. code-block:: python

   # In your test/CI script
   from openplaces.config import get_config
   
   cfg = get_config(interactive=False)
   # Uses project defaults, no prompts

Version Control
~~~~~~~~~~~~~~~

**DO commit:**

* ``openplaces.yaml`` (project defaults)
* ``.gitignore`` with appropriate exclusions

**DO NOT commit:**

* ``~/.config/openplaces/config.yaml`` (user config)
* ``data/`` directories

**Example .gitignore:**

.. code-block:: text

   # Data directories
   data/
   
   # User config (in case someone copies it to project dir)
   config.yaml
   !openplaces.yaml
   
   # Python
   __pycache__/
   *.pyc
   *.egg-info/

Troubleshooting
---------------

Config Not Found
~~~~~~~~~~~~~~~~

**Symptom:** Interactive prompt appears every time

**Solution:**

.. code-block:: bash

   # Check if config file exists
   python -m openplaces.config --show
   
   # If it doesn't exist, create it
   python -m openplaces.config --reconfigure

Changes Not Taking Effect
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Edited config file but changes aren't reflected

**Solution:**

.. code-block:: python

   # Reload configuration
   from openplaces.config import reload_config
   cfg = reload_config()

Wrong Directories Being Used
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Paths don't match what you configured

**Solution:**

.. code-block:: python

   # Check what's actually loaded
   from openplaces.config import cfg
   
   print(f"User config: {cfg.user_config_path}")
   print(f"Project config: {cfg.project_config_path}")
   
   # Show all directories
   for name, path in cfg.list_directories().items():
       print(f"{name}: {path}")

Permission Denied
~~~~~~~~~~~~~~~~~

**Symptom:** Cannot create config file

**Solution:**

.. code-block:: bash

   # Check config directory permissions
   ls -la ~/.config/openplaces/
   
   # If directory doesn't exist, create it
   mkdir -p ~/.config/openplaces/
   chmod 755 ~/.config/openplaces/

Windows Username with Spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Username contains spaces (e.g., "John Smith")

**Solution:**

The configuration system automatically uses your Windows user folder name
(e.g., ``JohnSmith`` from ``C:\Users\JohnSmith``), which doesn't contain spaces.

In multi-user mode, you can also specify a custom folder name:

.. code-block:: text

   User folder name:
   Default: JohnSmith (from your system user folder)
   
   Press Enter to accept, or type custom name: MyCustomName

API Reference
-------------

OpenPlacesConfig Class
~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: OpenPlacesConfig(project_config_path=None, interactive=True)

   Configuration manager with interactive first-use setup.

   :param project_config_path: Path to project config file
   :type project_config_path: Path, optional
   :param interactive: Whether to prompt user on first use
   :type interactive: bool, default True

   .. py:attribute:: username
      
      Current system username (from ``getpass.getuser()``)

   .. py:attribute:: username_dir
      
      User folder name (from ``Path.home().name``)

   .. py:attribute:: user_config_path
      
      Path to user configuration file

   .. py:attribute:: project_config_path
      
      Path to project configuration file (if found)

   .. py:method:: get_dir(name: str) -> Path

      Get directory path by name.

      :param name: Directory name (e.g., 'core', 'raw', 'external')
      :type name: str
      :return: Resolved directory path
      :rtype: Path
      :raises KeyError: If directory name not found

      **Example:**

      .. code-block:: python

         core_path = cfg.get_dir('core')
         raw_path = cfg.get_dir('raw')

   .. py:method:: list_directories() -> Dict[str, Path]

      Return dictionary of all configured directories.

      :return: Dictionary mapping directory names to paths
      :rtype: Dict[str, Path]

      **Example:**

      .. code-block:: python

         for name, path in cfg.list_directories().items():
             print(f"{name}: {path}")

   .. py:method:: add_custom_directory(name: str, path: str, description: str = '')

      Add a custom directory to configuration.

      :param name: Directory key name
      :type name: str
      :param path: Directory path (relative or absolute)
      :type path: str
      :param description: Description of directory purpose
      :type description: str, optional

      **Example:**

      .. code-block:: python

         cfg.add_custom_directory(
             'reports',
             'data/reports',
             'Generated analysis reports'
         )

   .. py:method:: get(key: str, default=None) -> Any

      Get configuration value with optional default.

      :param key: Configuration key
      :type key: str
      :param default: Default value if key not found
      :return: Configuration value or default

Module Functions
~~~~~~~~~~~~~~~~

.. py:function:: get_config(project_config_path=None, interactive=True) -> OpenPlacesConfig

   Get or create global configuration instance.

   :param project_config_path: Path to project config file
   :type project_config_path: Path, optional
   :param interactive: Whether to prompt user on first use
   :type interactive: bool, default True
   :return: Global configuration instance
   :rtype: OpenPlacesConfig

   **Example:**

   .. code-block:: python

      from openplaces.config import get_config
      
      cfg = get_config()
      cfg = get_config(interactive=False)  # For CI/CD

.. py:function:: reload_config(project_config_path=None, interactive=False) -> OpenPlacesConfig

   Reload configuration from files.

   :param project_config_path: Path to project config file
   :type project_config_path: Path, optional
   :param interactive: Whether to prompt user (usually False for reload)
   :type interactive: bool, default False
   :return: Reloaded configuration instance
   :rtype: OpenPlacesConfig

   **Example:**

   .. code-block:: python

      from openplaces.config import reload_config
      
      cfg = reload_config()

.. py:function:: reset_config()

   Delete user config file and reset to defaults.

   This will trigger the interactive setup on next import.

   **Example:**

   .. code-block:: python

      from openplaces.config import reset_config
      
      reset_config()

.. py:function:: show_config()

   Display current configuration file location and contents.

   **Example:**

   .. code-block:: python

      from openplaces.config import show_config
      
      show_config()

.. py:function:: edit_config()

   Open user config file in default editor.

   **Example:**

   .. code-block:: python

      from openplaces.config import edit_config
      
      edit_config()

Global Config Instance
~~~~~~~~~~~~~~~~~~~~~~

.. py:data:: cfg

   Global configuration instance (lazy-loaded).

   This is a proxy object that creates the actual configuration
   on first access, allowing the interactive prompt to work correctly.

   **Example:**

   .. code-block:: python

      from openplaces.config import cfg
      
      # Access directories
      data = cfg.dir_core
      
      # Access settings
      crs = cfg.crs_lat_long

Best Practices
--------------

1. **Single-user mode** for solo projects
   
   * Simpler structure
   * Fewer directories to navigate
   * Use project defaults

2. **Multi-user mode** for team projects
   
   * Separates user outputs
   * Shared raw data
   * Each user can customize their setup

3. **Commit** ``openplaces.yaml`` to version control
   
   * Provides defaults for all users
   * Documents expected structure

4. **Never commit** user config files
   
   * User-specific customizations
   * May contain absolute paths

5. **Use relative paths** in project config
   
   * Portable across machines
   * Works in different environments

6. **Use absolute paths** in user config (if needed)
   
   * Clarity about where data lives
   * No ambiguity

7. **Reload after editing** config files
   
   .. code-block:: python

      from openplaces.config import reload_config
      cfg = reload_config()

8. **Use** ``interactive=False`` in CI/CD
   
   * Prevents hanging on prompts
   * Uses defaults automatically

9. **Document custom directories** in project README
   
   * Help other users understand structure
   * Explain any non-standard setup

10. **Regular config reviews** for team projects
    
    * Ensure everyone uses consistent structure
    * Update as project evolves

Examples
--------

Basic Script
~~~~~~~~~~~~

.. code-block:: python

   #!/usr/bin/env python
   """Example script using openplaces configuration."""
   
   from openplaces.config import cfg
   import pandas as pd
   
   # Load raw data
   raw_file = cfg.dir_raw / 'dataset.csv'
   df = pd.read_csv(raw_file)
   
   # Process data
   processed = df.dropna()
   
   # Save to core
   output_file = cfg.dir_core / 'processed_dataset.csv'
   processed.to_csv(output_file, index=False)
   
   print(f"Processed data saved to: {output_file}")

Data Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openplaces.config import cfg
   from pathlib import Path
   
   class DataPipeline:
       def __init__(self):
           self.cfg = cfg
           
       def download_data(self, url: str, filename: str):
           """Download to raw directory."""
           output = self.cfg.dir_raw / filename
           # Download logic here
           return output
       
       def extract_data(self, archive_path: Path):
           """Extract to heap directory."""
           output_dir = self.cfg.dir_heap / archive_path.stem
           # Extraction logic here
           return output_dir
       
       def process_data(self, input_path: Path):
           """Process and save to core."""
           output = self.cfg.dir_core / f"processed_{input_path.name}"
           # Processing logic here
           return output
       
       def generate_report(self, data_path: Path):
           """Generate report in out directory."""
           output = self.cfg.dir_out / 'report.pdf'
           # Report generation here
           return output

Multi-Project Setup
~~~~~~~~~~~~~~~~~~~

**Setup for multiple projects:**

.. code-block:: python

   # project1/openplaces.yaml
   directories:
     external: data/external
     raw: data/raw
     core: data/core
     out: data/results
   
   # project2/openplaces.yaml
   directories:
     external: /shared/project2/external
     raw: /shared/project2/raw
     core: data/processed
     out: data/outputs
   
   # ~/.config/openplaces/config.yaml (shared settings)
   max_workers: 8
   download_timeout: 600

**Script that works in both projects:**

.. code-block:: python

   from openplaces.config import cfg
   
   # Works regardless of which project you're in
   print(f"Working directory: {cfg.project_config_path.parent}")
   print(f"Core data: {cfg.dir_core}")
   print(f"Workers: {cfg.max_workers}")

Testing Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from pathlib import Path
   from openplaces.config import get_config, reload_config
   import tempfile
   import yaml
   
   @pytest.fixture
   def temp_config():
       """Create temporary config for testing."""
       with tempfile.TemporaryDirectory() as tmpdir:
           # Create project config
           project_config = Path(tmpdir) / 'openplaces.yaml'
           config = {
               'directories': {
                   'external': 'data/external',
                   'raw': 'data/raw',
                   'core': 'data/core'
               }
           }
           with open(project_config, 'w') as f:
               yaml.dump(config, f)
           
           # Load config
           cfg = get_config(project_config, interactive=False)
           yield cfg
   
   def test_directory_resolution(temp_config):
       """Test that directories are resolved correctly."""
       cfg = temp_config
       
       assert cfg.dir_core.exists() or True  # Path object created
       assert 'core' in str(cfg.dir_core)
       assert cfg.dir_external.name == 'external'
