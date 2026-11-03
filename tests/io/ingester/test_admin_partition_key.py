"""A download partition's key falls back to the admin spine when the
admin recipe guessed for it has no output on this machine.

The SCC builds no US admin ingest, and the last of the six US admin2
recipes by version is the one the guess lands on. The names below are
fabricated.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.ingester import Ingester
from openplaces.io.ingester import partitions as ingester_partitions


def _ingester(monkeypatch, admin_recipe_id=None):
    def fake_get_admin(admin_id, level, recipe=None, columns=None):
        if recipe is not None:
            raise FileNotFoundError(f'No output file of recipe `{recipe}`')
        return pd.DataFrame({columns: ['North Example']}, index=[admin_id])

    monkeypatch.setattr(ingester_partitions, 'get_admin', fake_get_admin)
    monkeypatch.setattr(
        ingester_partitions,
        'find_admin_recipe_id',
        lambda *a, **k: 'XX_admin-unbuilt-2025_admin2',
    )
    ingester = Ingester.__new__(Ingester)
    download_by = {
        'admin_level': 2,
        'admin_key_transform': {'admin2_name': 'remove_spaces'},
    }
    if admin_recipe_id:
        download_by['admin_recipe_id'] = admin_recipe_id
    ingester.recipe = {'admin_id': AdminId('XX'), 'download_by': download_by}
    ingester.download_partition = {'admin_id_to_download': 'XX-NE'}
    return ingester


def test_an_unbuilt_guessed_recipe_reads_the_spine(monkeypatch):
    ingester = _ingester(monkeypatch)

    assert ingester._get_admin_partition_key('admin2_name') == 'NorthExample'


def test_an_unbuilt_declared_recipe_still_raises(monkeypatch):
    ingester = _ingester(monkeypatch, admin_recipe_id='XX_admin-declared-2025_admin2')

    with pytest.raises(FileNotFoundError, match='declared'):
        ingester._get_admin_partition_key('admin2_name')
