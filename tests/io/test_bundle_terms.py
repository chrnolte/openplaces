"""A delivery bundle says what its sources require of whoever receives it.

The engine does not decide whether a bundle is shared -- that is the user's
call -- but the call has to be an informed one. These tests pin what the
notice must get right: shares derived from the data rather than typed, a
share-alike licence named as such, and a source nobody has checked reported
as unchecked rather than quietly omitted.
"""

import pandas as pd
import pytest

from openplaces.io.bundle_terms import bundle_terms, format_notice

RECIPE = 'US_footprint-cheer-2026'

# The measured composition of the shipped NC bundle, 2026-08-18.
_COMPOSITION = ['obm'] * 858 + ['ncdps'] * 69 + ['microsoft'] * 33
_WITH_PARCELS = _COMPOSITION + ['parcel.bladenco'] * 40


@pytest.fixture
def terms():
    return bundle_terms(RECIPE, pd.Series(_WITH_PARCELS))


def test_shares_are_derived_from_the_data_not_declared(terms):
    """A hand-typed share would drift from the bundle; this one cannot."""
    shares = {entry['source_id']: entry['share'] for entry in terms['sources']}

    assert shares['obm'] == pytest.approx(0.858, abs=0.001)
    assert shares['microsoft'] == pytest.approx(0.033, abs=0.001)
    assert sum(v for v in shares.values() if v) == pytest.approx(1.0)


def test_odbl_geometry_is_reported_as_share_alike(terms):
    """The finding that gates publishing the CHEER inventory."""
    assert 'ODbL-1.0' in terms['share_alike']

    # obm 85.8% + microsoft 3.3%, independently of the hand count.
    assert terms['share_alike']['ODbL-1.0'] == pytest.approx(0.891, abs=0.002)


def test_a_dotted_geometry_source_resolves_to_its_source(terms):
    """`parcel.bladenco` is the bladenco source, not an unknown one."""
    ids = {entry['source_id'] for entry in terms['sources']}

    assert 'bladenco' in ids
    assert 'parcel.bladenco' not in ids


def test_an_unchecked_source_is_named_not_omitted():
    """Silence about a source would read as a clearance. It is not one.

    Uses a fabricated source id rather than a real county: which counties
    have been checked changes as the backfill proceeds, and this contract
    is about sources nobody has recorded, not about any particular one.
    """
    composition = _COMPOSITION + ['parcel.no-such-county'] * 40
    terms = bundle_terms(RECIPE, pd.Series(composition))
    unrecorded = {entry['source_id'] for entry in terms['unrecorded']}

    # Reported under the value as it appeared, since no token resolved:
    # an unmatched source is surfaced verbatim rather than dropped.
    assert 'parcel.no-such-county' in unrecorded
    # 'unknown' is a recorded answer -- somebody looked and found no terms.
    assert 'ncdps' not in unrecorded


def test_sources_contributing_no_geometry_are_left_out(terms):
    """NSI and imagery feed the recipe but no polygon inherits their terms."""
    ids = {entry['source_id'] for entry in terms['sources']}

    assert ids == {'obm', 'ncdps', 'microsoft', 'bladenco'}


def test_two_share_alike_licences_are_both_reported():
    """Pitt County parcels are CC-BY-SA-4.0 and the footprints are ODbL.

    Both are share-alike, both are in the cheer-eastern-nc region, and no
    single release satisfies the two at once. The notice has to surface
    that rather than name whichever it happened to see first.
    """
    composition = _COMPOSITION + ['parcel.pittcounty'] * 40
    terms = bundle_terms(RECIPE, pd.Series(composition))

    assert set(terms['share_alike']) == {'ODbL-1.0', 'CC-BY-SA-4.0'}


def test_notice_states_the_obligation_and_who_decides(terms):
    notice = format_notice(RECIPE, terms, 'US-NC')

    assert 'ODbL-1.0' in notice
    assert '89.1%' in notice
    # Names the upstreams that must be credited.
    assert 'obm' in notice and 'microsoft' in notice
    # Says the sharing decision is the distributor's, not the software's.
    assert 'decision for you as the distributor' in notice
    # An unchecked source is named as unchecked rather than omitted.
    unchecked = bundle_terms(
        RECIPE, pd.Series(_COMPOSITION + ['parcel.no-such-county'] * 40)
    )
    assert 'not the same as' in format_notice(RECIPE, unchecked, 'US-NC')


def test_notice_without_geometry_shares_still_lists_sources():
    """An export with no `geometry_source` column still gets a notice."""
    terms = bundle_terms(RECIPE, None)
    notice = format_notice(RECIPE, terms)

    assert 'openplaces' in notice
    assert terms['sources'], 'the dependency walk should still find sources'


def test_the_bundle_declares_a_terms_file_among_its_outputs():
    """The notice ships with the data, so it is one of the bundle paths.

    `flow.dag` declares `delivery_paths().values()` as a delivery job's
    outputs and `unlock_delivery` unlocks them, so being in this dict is
    what makes the notice regenerate with the data rather than go stale.
    """
    from openplaces.io.delivery import delivery_paths

    paths = delivery_paths(RECIPE, region='cheer-eastern-nc')

    assert 'terms' in paths
    assert paths['terms'].suffix == '.txt'
    assert paths['terms'].parent == paths['canonical'].parent


def test_conflicting_share_alike_licences_say_so_in_words():
    """Two share-alike licences read as "dual-license it" unless told.

    The first real run of this notice reported ODbL and CC-BY-SA as two
    independent obligations, which invites exactly the wrong conclusion:
    a share-alike licence governs the whole release, so two of them
    cannot both be honoured and dual licensing does not help. Surfacing
    that is the point of the file.
    """
    composition = _COMPOSITION + ['parcel.pittcounty'] * 40
    notice = format_notice(RECIPE, bundle_terms(RECIPE, pd.Series(composition)))

    assert 'CANNOT ALL BE SATISFIED AT ONCE' in notice
    assert 'Dual licensing does not resolve it' in notice
    # And it names a way forward rather than only the problem.
    assert 'Produced' in notice


def test_a_single_share_alike_licence_raises_no_conflict():
    """The warning must not fire when there is nothing to conflict with."""
    notice = format_notice(RECIPE, bundle_terms(RECIPE, pd.Series(_COMPOSITION)))

    assert 'ODbL-1.0' in notice
    assert 'CANNOT ALL BE SATISFIED' not in notice
