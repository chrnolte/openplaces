"""`download_by.partition_key_lookup` names a unit's file through a
sidecar table, and a unit missing from it is a soft skip.

datos.gov.co files each Cundinamarca municipality's cadastre under a
dataset id no admin column carries, so the DANE code is looked up.
The ids and codes below are fabricated.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.ingester import Ingester
from openplaces.io.ingester import partitions as ingester_partitions
from openplaces.io.ingester.partitions import PartitionNotPublished
from openplaces.timing import Timer

_TABLE = pd.DataFrame(
    {'divipola': ['99001', '99002'], 'dataset_id': ['aaaa-1111', 'bbbb-2222']}
)


def _ingester(monkeypatch, code):
    monkeypatch.setattr(ingester_partitions, 'get_recipe_by_id', lambda rid: _TABLE)
    monkeypatch.setattr(
        Ingester, '_get_admin_partition_key', lambda self, placeholder: code
    )
    ingester = Ingester.__new__(Ingester)
    ingester.recipe = {
        'admin_id': AdminId('XX-YY'),
        'download_by': {
            'admin_level': 3,
            'partition_key_lookup': {
                'dataset_id': {
                    'from': 'admin3_id_admin1',
                    'lookup_table': 'XX-YY_parcel-test-2026_datasets',
                    'key': 'divipola',
                    'value': 'dataset_id',
                }
            },
        },
    }
    ingester.timer = Timer('test')
    ingester.verbose = False
    ingester.download_partition = {'admin_id_to_download': 'XX-YY-ZZ'}
    return ingester


def test_a_listed_unit_resolves_to_its_dataset_id(monkeypatch):
    ingester = _ingester(monkeypatch, '99002')

    url = ingester._resolve_placeholders('https://example.org/{dataset_id}.geojson')

    assert url == 'https://example.org/bbbb-2222.geojson'


def test_an_unlisted_unit_is_not_published(monkeypatch):
    ingester = _ingester(monkeypatch, '99003')

    with pytest.raises(PartitionNotPublished, match='99003'):
        ingester._resolve_placeholders('https://example.org/{dataset_id}.geojson')
