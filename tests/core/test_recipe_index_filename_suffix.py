"""The recipe index must recover a filename suffix from a sanitized version.

A version is sanitized on its way into a filename ('4.1' -> '4~1'), so a
base id built from the raw version matches no such recipe and the whole
id reads as having no suffix. Anything deduplicating on
(admin_id, source_id, filename_suffix) then collapses siblings that only
the suffix tells apart.
"""

from openplaces.diagnostics import find_recipes


def test_sanitized_version_still_yields_the_suffix():
    admin = find_recipes('admin', stage='ingest')
    gadm = admin[admin['source_id'] == 'gadm']

    assert not gadm.empty
    assert set(gadm['filename_suffix']) == {'admin1', 'admin2', 'admin3', 'admin4'}
