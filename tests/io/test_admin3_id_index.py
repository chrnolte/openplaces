"""Tests for `admin3_id_index_from_admin3_code`'s name fallback.

The fallback exists because a New England town that becomes a city gets a
new Census subdivision code while the spine still holds the old one. It
recovers those rows by stripping the type word from the name, which is
ambiguous in a way the code join is not, so what it recovers has to be
reported accurately and checked for collisions before it becomes an
index. Every unit below is fabricated.
"""

import pandas as pd
import pytest

from openplaces.io import admin


def _spine(rows):
    """A level-3 spine frame as `get_admin` returns one, indexed by id."""
    frame = pd.DataFrame(rows, columns=['admin3_id', 'admin3_id_admin1', 'name'])
    return frame.set_index('admin3_id')


@pytest.fixture
def spine(monkeypatch):
    """One spine unit, reachable by code '0001' or by its stripped name."""
    frame = _spine([('XX-AA-BW', '0001', 'Bridgewater Town')])
    monkeypatch.setattr(admin, 'get_admin', lambda *args, **kwargs: frame)
    return frame


class TestDuplicateGuard:
    def test_two_names_stripping_alike_are_refused(self, spine):
        # Neither code is in the spine, so both rows fall through to the
        # name fallback and both land on the one Bridgewater.
        gdf = pd.DataFrame(
            {
                'admin3_id_admin1': ['9998', '9999'],
                'name': ['Bridgewater', 'Bridgewater town'],
            }
        )
        with pytest.raises(ValueError, match='share'):
            admin.admin3_id_index_from_admin3_code(gdf, 'XX')

    def test_a_recovered_row_may_not_claim_a_matched_unit(self, spine):
        # The first row matches on code; the second recovers by name
        # onto the same unit, which would index one spine unit twice.
        gdf = pd.DataFrame(
            {
                'admin3_id_admin1': ['0001', '9999'],
                'name': ['Bridgewater', 'BRIDGEWATER'],
            }
        )
        with pytest.raises(ValueError, match='share'):
            admin.admin3_id_index_from_admin3_code(gdf, 'XX')

    def test_a_clean_join_still_indexes(self, spine):
        gdf = pd.DataFrame({'admin3_id_admin1': ['0001'], 'name': ['Bridgewater']})
        indexed = admin.admin3_id_index_from_admin3_code(gdf, 'XX')
        assert indexed.index.tolist() == ['XX-AA-BW']


class TestRecoveredWarning:
    def test_the_warning_names_only_the_recovered_rows(self, monkeypatch):
        # `recovered.notna().index` is every unmatched row, so the
        # warning used to name units that did not match at all and that
        # the next block raises on.
        frame = _spine([('XX-AA-BW', '0001', 'Bridgewater Town')])
        monkeypatch.setattr(admin, 'get_admin', lambda *args, **kwargs: frame)
        gdf = pd.DataFrame(
            {
                'admin3_id_admin1': ['9999', '8888'],
                'name': ['Bridgewater', 'Nowhereville'],
            }
        )
        with pytest.warns(UserWarning) as caught:
            with pytest.raises(ValueError, match='no spine match'):
                admin.admin3_id_index_from_admin3_code(gdf, 'XX')
        message = str(caught[0].message)
        assert '1 admin-3 unit(s)' in message
        assert 'Bridgewater' in message
        assert 'Nowhereville' not in message
