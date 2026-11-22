"""The download timeout has a built-in default."""

from openplaces.config.settings import OpenPlacesConfig


def test_download_timeout_has_a_default():
    """A run that finds no project config file still has a timeout."""
    assert OpenPlacesConfig.DEFAULTS['download_timeout'] == 600
