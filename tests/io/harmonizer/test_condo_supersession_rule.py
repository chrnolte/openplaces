"""One supersession rule, shared by the three copies of the links.

Condo-cluster consolidation replaces spine rows and absorbs parcels. The
crosswalk, the overlay and the sidecar on disk each have to drop the
same rows. They did not: the crosswalk filtered by absorbed parcel while
the other two filtered by replaced spine id, so a parcel joining a
cluster whose old footprint was not itself replaced kept its
pre-consolidation overlap beside the new 'condo cluster' link. Only the
sidecar-reload path saw it, and it failed 7 of the 86 CHEER counties.
"""

import pandas as pd

from openplaces.io.harmonizer.links import _superseded_by_consolidation

SPINE_ID = 'footprint_id'


def _index(pairs):
    return pd.MultiIndex.from_tuples(pairs, names=[SPINE_ID, 'parcel_id'])


def test_a_link_to_an_absorbed_parcel_is_superseded():
    """Even when its own footprint was not replaced.

    This is the case the overlay and the sidecar used to keep, and the
    reason a pair appeared twice.
    """
    index = _index([('fp-kept', 'parcel-absorbed'), ('fp-kept', 'parcel-other')])

    mask = _superseded_by_consolidation(
        index,
        SPINE_ID,
        superseded_spine_ids=set(),
        consolidated_pids={'parcel-absorbed'},
    )

    assert list(mask) == [True, False]


def test_a_link_from_a_replaced_spine_row_is_superseded():
    index = _index([('fp-replaced', 'parcel-other'), ('fp-kept', 'parcel-other')])

    mask = _superseded_by_consolidation(
        index,
        SPINE_ID,
        superseded_spine_ids={'fp-replaced'},
        consolidated_pids=set(),
    )

    assert list(mask) == [True, False]


def test_an_untouched_link_survives():
    index = _index([('fp-kept', 'parcel-other')])

    mask = _superseded_by_consolidation(
        index, SPINE_ID, superseded_spine_ids={'fp-x'}, consolidated_pids={'parcel-y'}
    )

    assert list(mask) == [False]


def test_the_same_rule_applies_to_every_copy_of_the_links():
    """The crosswalk, the overlay and the sidecar must agree exactly.

    Each holds a (spine id, parcel id) index over the same links, so the
    same call has to give the same answer for all three; a copy that
    filters more narrowly is what carried the duplicate pair.
    """
    pairs = [
        ('fp-replaced', 'parcel-absorbed'),
        ('fp-kept', 'parcel-absorbed'),
        ('fp-kept', 'parcel-other'),
    ]
    args = (SPINE_ID, {'fp-replaced'}, {'parcel-absorbed'})

    crosswalk = _superseded_by_consolidation(_index(pairs), *args)
    overlay = _superseded_by_consolidation(_index(pairs), *args)
    sidecar = _superseded_by_consolidation(_index(pairs), *args)

    assert list(crosswalk) == list(overlay) == list(sidecar) == [True, True, False]
