# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

# Recognize Python scripts in '_ext' (e.g. formatting for Github links)
sys.path.insert(0, str(Path('_ext').resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'openplaces'
copyright = '2025, Christoph Nolte'
author = 'Christoph Nolte'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_copybutton',
    'sphinx_design',
    'gh_file',
    'autoapi.extension',
    'sphinx.ext.napoleon',
]

autoapi_dirs = ['../src']
autoapi_type = 'python'
autoapi_options = [
    'members',
    'show-inheritance',
    'show-module-summary',
]
autoapi_keep_files = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_drafts']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = 'openplaces'
html_logo = 'images/openplaces_icon_transparent_130.png'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 4,
    'show_nav_level': 2,
    'navbar_align': 'left',
    'secondary_sidebar_items': ['page-toc'],
    'logo': {
        'text': 'openplaces',
    },
}
html_static_path = ['_static']
html_css_files = ['custom.css']

highlight_language = 'python3'

rst_prolog = """
.. role:: gui
   :class: gui

.. role:: file
   :class: file

.. role:: input(code)
   :class: input
"""
