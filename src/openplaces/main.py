"""Main module for openplaces package."""


def hello_openplaces():
    """A simple hello function."""
    return "Hello from `openplaces`!"


def get_version():
    """Return the package version."""
    from . import __version__

    return __version__
