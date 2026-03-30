from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

GITHUB_BASE = 'https://github.com/chrnolte/openplaces/blob/main/'


def gh_file_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    ref = GITHUB_BASE + text
    literal = nodes.literal(text, text, classes=['file'])
    link = nodes.reference(rawtext, '', literal, refuri=ref)
    return [link], []


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_role('gh-file', gh_file_role)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
