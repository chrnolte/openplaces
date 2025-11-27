.. openplaces

.. _install:

Install openplaces
==================

As an application
~~~~~~~~~~~~~~~~~

You can install ``openplaces`` as an application with its own virtual environment.

This is the default mode for most users, especially those wanting to contribute to the public repository (e.g., recipes, functions).

Install
-------

You need to have a version of the Anaconda package manager installed (``conda`` or ``mamba`` needs to be accessible from the :gui:`Terminal` (:gui:`Anaconda Prompt` on Windows).

1. Open :gui:`Terminal`.

2. Change to the directory where you want to install the ``openplaces`` codebase:

   .. code-block:: bash

      cd ~/code  # Example: 'code' folder in the user directory

3. Clone the ``openplaces`` Github repository:

   .. code-block:: bash

      git clone https://github.com/chrnolte/openplaces.git 

4. Change into the repository directory:

   .. code-block:: bash

      cd openplaces

5. Run the setup script:

   .. code-block:: bash

      python dev.py setup


Uninstall
---------

To remove your environment, use:

.. code-block:: bash

   python dev.py clean


As a PyPi package
~~~~~~~~~~~~~~~~~

``openplaces`` will ultimately be available as a Python package. However, the version on PyPi lags behind the development on Github.

.. code-block:: bash

   pip install openplaces

You can also choose to install sub-components of ``openplaces``:

.. code-block:: bash

   # Example extras; adjust to your setup
   pip install "openplaces[dev,docs]"