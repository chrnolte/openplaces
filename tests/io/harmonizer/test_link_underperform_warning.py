"""The low-match warning covers every column holding a parcel key.

Lake County FL fell from 96.5% matched to nothing without a word,
because the transaction spine had moved its join onto `parcel_link_key`
and the guard recognized only the literal `parcel_id_local`.
"""

import pandas as pd
import pytest

from openplaces.core.schema import Entity
from openplaces.io.harmonizer import HarmonizeState, links


def _state(spine):
    return HarmonizeState(
        recipe={'entity': Entity('parcel')},
        admin_id='XX-YY-ZZ',
        verbose=False,
        timer=None,
        spine=spine,
    )


@pytest.mark.parametrize('mode', ['attributes', 'aggregate'])
@pytest.mark.parametrize('key', ['parcel_id_local', 'parcel_link_key'])
def test_link_by_id_actually_reaches_the_guard(monkeypatch, mode, key):
    """The guard must *run*, not merely be correct when called.

    It was called only from the `attributes` branch, so an
    auto-discovered link (always `aggregate`) went unchecked however
    little it matched, and the count it needed sat behind `verbose`.
    Holmes County FL rebuilt at 6.8% matched in silence.
    """
    spine = pd.DataFrame({key: [f'k{i}' for i in range(100)]})
    ref = pd.DataFrame({'parcel_id_local': ['k0'], 'land_value': [1]})
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref)
    with pytest.warns(UserWarning, match='matched only'):
        links.link_by_id(
            _state(spine),
            'ref',
            mode=mode,
            spine_key=key,
            ref_key='parcel_id_local',
            columns=['land_value'],
        )


@pytest.mark.parametrize('key', ['parcel_id_local', 'parcel_link_key'])
def test_a_join_that_resolves_almost_nothing_warns_on_either_key(key):
    with pytest.warns(UserWarning, match='matched only'):
        links._warn_if_link_underperforms(1, 1000, key, 'ref', 'XX-YY-ZZ')


@pytest.mark.parametrize('key', ['parcel_id_local', 'parcel_link_key'])
def test_a_healthy_join_stays_quiet(key, recwarn):
    links._warn_if_link_underperforms(995, 1000, key, 'ref', 'XX-YY-ZZ')
    assert not [w for w in recwarn if 'matched only' in str(w.message)]


def test_a_gap_filling_pass_is_expected_to_match_a_minority(recwarn):
    # fill_only exists to reach what an earlier pass missed, so a small
    # share is its job. The transaction spine's property link touches
    # only sales of stacked units; warning on it fired for 30 Florida
    # counties and buried the joins that genuinely misfit.
    links._warn_if_link_underperforms(
        1, 1000, 'parcel_id_local', 'ref', 'XX-YY-ZZ', fill_only=True
    )
    assert not [w for w in recwarn if 'matched only' in str(w.message)]


def test_a_key_no_conversion_produces_is_not_second_guessed(recwarn):
    # An address key has no bundled conversion to re-fit, so a thin
    # result there is not evidence of a misfitting rule.
    links._warn_if_link_underperforms(1, 1000, 'address_id_local', 'r', 'XX')
    assert not [w for w in recwarn if 'matched only' in str(w.message)]
