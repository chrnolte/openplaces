"""Two resolution defects the review confirmed.

`download_by.partition_key_transformation` is documented for any placeholder
but was only consulted for `admin*` ones, so an entry keyed on `year` was
ignored and the URL was built from the untransformed key.

`_resolve_output_admin_ids` aborted whenever a request resolved to no unit at
the save level, which conflates a renamed identifier with a unit that simply
has no children at that level.
"""

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io import ingester as ingester_module
from openplaces.io.ingester import Ingester, _transform_partition_key
from openplaces.timing import Timer


def _bare(recipe, admin_ids=None):
    ingester = Ingester.__new__(Ingester)
    ingester.recipe = recipe
    ingester.admin_ids = admin_ids or [AdminId(None)]
    ingester.partition_ids = None
    ingester.timer = Timer('test')
    ingester.verbose = False
    ingester._owns_timer = False
    return ingester


def test_a_year_placeholder_is_transformed():
    recipe = {
        'admin_id': AdminId('US'),
        'download_by': {
            'partition': 'year',
            'partition_key_transformation': {
                'year': {'operation': 'substring', 'args': [2, 4]}
            },
        },
    }
    ingester = _bare(recipe)
    ingester.download_partition = {'partition_id_to_download': '2019'}

    resolved = ingester._resolve_placeholders('roll_{year}.zip')

    assert resolved == 'roll_19.zip'


def test_a_short_args_list_names_the_operation():
    with pytest.raises(ValueError, match="'substring' needs 2 argument"):
        _transform_partition_key('2019', {'operation': 'substring', 'args': [2]})


def _spine(monkeypatch, by_level):
    def fake_get_admin(admin_id, level, **kwargs):
        return pd.DataFrame(index=pd.Index(by_level.get(level, []), name='admin_id'))

    monkeypatch.setattr(ingester_module, 'get_admin', fake_get_admin)


def test_a_childless_unit_is_a_quiet_no_op(monkeypatch):
    """A country the spine knows but that carries no level-3 rows."""
    _spine(monkeypatch, {1: ['XA', 'XB'], 3: ['XB-BB-BB']})
    recipe = {'admin_id': AdminId(None), 'save_to': {'admin_level': 3}}
    ingester = _bare(recipe, admin_ids=[AdminId('XA')])

    assert ingester._resolve_output_admin_ids(reprocess=True) == []


def test_an_unknown_unit_still_raises(monkeypatch):
    _spine(monkeypatch, {1: ['XA', 'XB'], 3: ['XB-BB-BB']})
    recipe = {'admin_id': AdminId(None), 'save_to': {'admin_level': 3}}
    ingester = _bare(recipe, admin_ids=[AdminId('XZ')])

    with pytest.raises(ValueError, match='re-mint may have renamed'):
        ingester._resolve_output_admin_ids(reprocess=True)
