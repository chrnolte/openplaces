# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess
import sys
from pathlib import Path

# Recognize Python scripts in '_ext' (e.g. formatting for Github links)
sys.path.insert(0, str(Path('_ext').resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'openplaces'
copyright = 'openplaces contributors'
author = 'openplaces contributors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_copybutton',
    'sphinx_design',
    'gh_file',
    'recipe_catalog',
    'recipe_state',
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinxcontrib.mermaid',
]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_drafts']

# -- AutoAPI configuration ---------------------------------------------------

# Exclude untracked files/dirs so autoapi only documents committed code.
# On ReadTheDocs (clean checkout) this returns nothing, so autoapi_ignore stays [].
_repo_root = Path(__file__).parent.parent
try:
    _git = subprocess.run(
        [
            'git',
            'ls-files',
            '--others',
            '--exclude-standard',
            '--directory',
            '--',
            'src/openplaces/',
        ],
        capture_output=True,
        text=True,
        cwd=_repo_root,
    )
    autoapi_ignore = [
        str((_repo_root / _entry.rstrip('/')).resolve())
        for _entry in _git.stdout.splitlines()
    ]
except Exception:
    autoapi_ignore = []

autoapi_dirs = ['../src/openplaces']
autoapi_template_dir = '_templates/autoapi'
autoapi_type = 'python'
autoapi_options = [
    'members',
    'show-inheritance',
    'show-module-summary',
]
autoapi_root = '4_api'

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

.. role:: file(code)
   :class: file

.. role:: input(code)
   :class: input
"""
