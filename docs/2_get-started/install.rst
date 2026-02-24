.. openplaces

.. _install:

Install
=======

As an application
~~~~~~~~~~~~~~~~~

You can install ``openplaces`` as an application with its own virtual environment.

This is the default for users who want to use the most recent functionality or plan to contribute recipes or functionality to the public repository.

Install
-------

You need to have a version of the Anaconda package manager installed.

``conda`` or ``mamba`` needs to be executable in

- :gui:`Terminal` (OSx, Linux)
- :gui:`Anaconda Prompt` (Windows).

``mamba`` is a fast solver for ``conda``. If ``conda`` takes too long, run ``conda install mamba`` first.

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

5. Run the setup script:

   .. code-block:: bash

      python dev.py setup

   The setup script will let you choose the name of your environment. It will also ask you whether you need ``7z`` (to unpack some Windows :file:`.zip` files).

6. Once the environment is installed, you need to :ref:`configure <configure>` your installation.


Uninstall
---------

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