"""A missing admin layer is absent coverage only where the level is empty.

`link_geographic_ids` asks for the units at a level under the admin
unit it runs on. A New England town has no admin4 units and so no
admin4 file; that leaves the column null. A unit that does have
admin4 units whose layer was never built must still fail.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openplaces.io.harmonizer import spine as spine_mod


def _missing(*args, **kwargs):
    raise FileNotFoundError('no admin4 file')


def _state():
    return SimpleNamespace(admin_id='US-XX-TWN', spine=None)


def test_a_level_with_no_units_is_absent_coverage(monkeypatch):
    monkeypatch.setattr(spine_mod, 'get_admin', _missing)
    monkeypatch.setattr(spine_mod, 'get_admin_ids', lambda *a, **k: [])
    link = {'admin_level': 4, 'output_column': 'admin4_id'}
    assert spine_mod._resolve_geographic_reference(_state(), link) is None


def test_a_level_with_units_but_no_layer_still_fails(monkeypatch):
    monkeypatch.setattr(spine_mod, 'get_admin', _missing)
    monkeypatch.setattr(spine_mod, 'get_admin_ids', lambda *a, **k: ['US-XX-TWN-AA'])
    link = {'admin_level': 4, 'output_column': 'admin4_id'}
    with pytest.raises(FileNotFoundError):
        spine_mod._resolve_geographic_reference(_state(), link)
