.. openplaces

.. _install:

Install
=======

As an application
~~~~~~~~~~~~~~~~~

You can install ``openplaces`` as a standalone application with its own virtual environment.

This is the recommended setup for users who want to use the most recent functionality or plan to :ref:`contribute <contribute>` code (recipes, functions, fixes).

Get the repository
------------------

1. Open :gui:`Terminal` / :gui:`Anaconda Prompt`.

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

You need to have a version of the Anaconda package manager installed.

``conda`` or ``mamba`` needs to be executable from:

- :gui:`Terminal` (OSx, Linux)
- :gui:`Anaconda Prompt` (Windows).

.. note::

   ``mamba`` is a fast solver for ``conda`` and will likely speed up your installation. If you don't have it installed, run ``conda install mamba -y`` before running the setup script.

Run the setup script:

.. code-block:: bash

   python dev.py setup

The setup script will let you choose the name of your environment.

It will also ask you whether

- you need ``7z`` (to unpack some Windows :file:`.zip` files).
- you want the QGIS tool installed (in the Processing Toolbox) that helps you visualize split :file:`.parquet` files (attribute table + geoparquet geometries).
- you want to install a shortcut that lets you type :input:`openplaces` into your terminal / Anaconda Prompt to run Jupyter notebooks in your environment.


Configure the environment
-------------------------

Once the environment is installed, you need to :ref:`configure <configure>` your installation.


Uninstall the environment
-------------------------

To remove your environment, use:

.. code-block:: bash

   python dev.py clean


As a PyPi package
~~~~~~~~~~~~~~~~~

``openplaces`` is being developed to function as a Python package.

.. note::

   During the development stage, the version on PyPi will lag behind the Github repository.

You can 

.. code-block:: bash

   pip install openplaces

Install sub-components of ``openplaces``:

.. code-block:: bash

   # Example extras; adjust to your setup
   pip install "openplaces[dev,docs]"