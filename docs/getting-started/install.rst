.. openplaces

.. _install:

Installation
============

As an application
~~~~~~~~~~~~~~~~~

You can install ``openplaces`` as an application with its own virtual environment.

This is the default mode for most users, especially those wanting to contribute to the public repository (e.g., recipes, functions).

Install
-------

You need to have a version of the Anaconda package manager installed.

``conda`` or ``mamba`` needs to be executable in the :gui:`Terminal` (OSx, Linux) or :gui:`Anaconda Prompt` (Windows).

1. Open :gui:`Terminal` / :gui:`Anaconda Prompt`.

2. Change to the directory where you want to install the ``openplaces`` codebase:

   .. code-block:: bash

      # Example: 'code' folder in the user directory
      cd ~/code  

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

``openplaces`` is being developed to function as a Python package.

However, during the development stage, the version on PyPi will lag behind the Github repository.

.. code-block:: bash

   pip install openplaces

Install sub-components of ``openplaces``:

.. code-block:: bash

   # Example extras; adjust to your setup
   pip install "openplaces[dev,docs]"