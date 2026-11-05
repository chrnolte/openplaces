from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.nodes import split_explicit_title
from sphinx.util.typing import ExtensionMetadata

GITHUB_BASE = 'https://github.com/chrnolte/openplaces/blob/main/'


def gh_file_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    # :gh-file:`path` shows the path; :gh-file:`title <path>` shows
    # the title (a recipe id, say) and links to the same file.
    _, title, path = split_explicit_title(text)
    ref = GITHUB_BASE + path
    literal = nodes.literal(title, title, classes=['file'])
    link = nodes.reference(rawtext, '', literal, refuri=ref)
    return [link], []


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_role('gh-file', gh_file_role)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
