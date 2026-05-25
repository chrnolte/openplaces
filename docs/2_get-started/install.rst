.. openplaces

.. _install:

Install
=======

.. _as_an_application:

As an application
~~~~~~~~~~~~~~~~~

Install ``openplaces`` as an application with its own environment.

This is the recommended setup for developers and collaborators.

- It allows you to use the most recent functionality 
- It allows you to make :ref:`code contributions <contribute>`.


.. _get_the_repository:

Get the repository
------------------

1. Open your :gui:`Terminal`. 

   .. tab-set::

      .. tab-item:: Windows

         Open :gui:`Command Prompt` or :gui:`Anaconda Prompt`

      .. tab-item:: macOS

         Open :gui:`Terminal`

      .. tab-item:: Linux

         Open :gui:`Terminal`

2. Change to the directory where you want to keep the ``openplaces`` codebase:

   .. code-block:: bash

      # Example: 'code' folder in the user directory
      cd ~/code

3. Clone the ``openplaces`` Github repository:

   .. code-block:: bash

      git clone https://github.com/chrnolte/openplaces.git

4. Change into the repository directory:

   .. code-block:: bash

      cd openplaces


Install the environment
-----------------------

You need to have a version of the `Anaconda package manager <https://www.anaconda.com/docs/getting-started/main>`_ installed.

Either ``conda`` or ``mamba`` need to be executable from your :gui:`Terminal`:

.. tab-set::

   .. tab-item:: Windows

      Use :gui:`Anaconda Prompt`

   .. tab-item:: macOS

      Open :gui:`Terminal`

   .. tab-item:: Linux

      Open :gui:`Terminal`

If you don't have ``conda`` installed yet, we recommend installing `Miniconda <https://www.anaconda.com/docs/getting-started/miniconda/main>`_, a free, miniature installation of Anaconda.

`mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_ is a fast environment solver for ``conda``. It will speed up your installation considerably. If you don't have it installed, run ``conda install mamba -y`` before running the setup script.

Once you have Anaconda, run the setup script:

.. code-block:: bash

   python dev.py setup

The setup script will let you choose the name of your environment.

It will also ask you whether:

- you want to install ``7z``. It allows you to unpack :file:`.zip` files formatted with Windows formats (e.g., the parcel recipe for Virginia's parcels).
- you want the :gui:`QGIS` tool installed in the :gui:`Processing Toolbox`. The tool allows you to import openplaces' :file:`.parquet` tables with associated geometry parquet files as one joined vector layer.
- you want to install a shortcut in your user folder. This lets you type :input:`openplaces` into your :gui:`Terminal` to activate your environment and run Jupyter in your notebooks folder.


Configure the environment
-------------------------

Once the environment is installed, you need to :ref:`configure <configure>` your installation.


Uninstall the environment
-------------------------

To remove your environment, use:

.. code-block:: bash

   python dev.py clean

