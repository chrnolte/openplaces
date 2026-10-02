"""`python -m openplaces.config`: the configuration command line, kept
working after config.py became a package on 2026-09-30."""

from openplaces.config import main

if __name__ == '__main__':
    main()
