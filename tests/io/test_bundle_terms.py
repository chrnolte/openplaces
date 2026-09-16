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

RECIPE = 'US_footprint-openplaces-2026'

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


def test_two_recipes_sharing_a_source_id_both_report_their_licence():
    """Overture ships two themes under one source id and two licences.

    The addresses theme is permissive, the buildings theme is ODbL-1.0.
    A `geometry_source` value records only the id, so a notice that keeps
    whichever recipe it reached first drops a share-alike obligation from
    a bundle that inherits one.
    """
    composition = _COMPOSITION + ['overture'] * 40
    terms = bundle_terms(RECIPE, pd.Series(composition))

    overture = [e for e in terms['sources'] if e['source_id'] == 'overture']
    licenses = {entry['license'] for entry in overture}

    assert 'ODbL-1.0' in licenses
    assert any(str(text).startswith('mixed-permissive') for text in licenses)

    # The ODbL share now reaches the Overture geometry as well.
    assert terms['share_alike']['ODbL-1.0'] > 0.92

    notice = format_notice(RECIPE, terms)
    # Both entries are named by their recipe, so neither reads as the
    # other's terms.
    assert 'footprint-overture-2026' in notice
    assert 'dwelling-overture-2025' in notice


def test_a_redistribution_restricted_source_is_named_in_the_notice():
    """Edgecombe County parcels are a signed-agreement, no-resale source.

    `redistribution_restricted` was recorded by recipes and read by
    nothing, so the notice for a bundle carrying such a source said
    nothing about it.
    """
    composition = _COMPOSITION + ['parcel.edgecombecounty'] * 40
    terms = bundle_terms(RECIPE, pd.Series(composition))

    assert 'edgecombecounty' in {e['source_id'] for e in terms['restricted']}

    notice = format_notice(RECIPE, terms, 'US-NC')
    assert 'Redistribution restricted' in notice
    assert 'edgecombecounty' in notice.split('Redistribution restricted')[1]


def test_a_bundle_without_restricted_sources_has_no_restriction_section():
    """The section must not appear where nothing recorded a restriction."""
    notice = format_notice(RECIPE, bundle_terms(RECIPE, pd.Series(_COMPOSITION)))

    assert 'Redistribution restricted' not in notice


def test_geometry_of_unrecorded_provenance_is_reported_not_hidden():
    """Shares normalized over non-nulls always summed to 1.0.

    100 attributed rows against 900 nulls then read as "100% of the
    geometry is ODbL-1.0", which overstates what is known about the
    bundle in both directions.
    """
    terms = bundle_terms(RECIPE, pd.Series(['obm'] * 100 + [None] * 900))

    assert terms['unknown_share'] == pytest.approx(0.9)
    assert terms['share_alike']['ODbL-1.0'] == pytest.approx(0.1)

    notice = format_notice(RECIPE, terms)
    assert '90.0%' in notice
    assert 'source not recorded' in notice


def test_private_sharing_reads_as_satisfied_only_where_nothing_restricts_it(terms):
    """Keeping the notice with the data settles attribution, not a contract.

    A signed-agreement, no-resale source is not satisfied by carrying a
    notice, so the sentence that says private sharing generally is must
    not appear over one.
    """
    notice = format_notice(RECIPE, terms, 'US-NC')

    assert 'generally satisfied by keeping this notice' in notice
    assert 'restrict redistribution; see' not in notice
    # Who decides is stated either way.
    assert 'decision for you as the distributor' in notice


def _fabricated_source_terms(monkeypatch, **overrides):
    """Terms for one fabricated source, bypassing the recipe walk.

    No committed recipe records a no-resale clause yet, so the contract
    is pinned on a made-up source rather than on a real county.
    """
    import openplaces.io.bundle_terms as module

    entry = {
        'source_id': 'examplepa',
        'recipe_id': 'US-XX-YY_property-examplepa-2026',
        'license': 'free download; not to be resold without prior consent',
        'terms_url': 'https://example.org/terms',
        'portal_url': None,
        'redistribution_restricted': False,
        'resale_restricted': True,
        **overrides,
    }
    monkeypatch.setattr(
        module, '_source_terms', lambda recipe: {module._terms_key(entry): entry}
    )
    monkeypatch.setattr(module, '_catalog_terms', lambda: {})
    return bundle_terms(RECIPE, None)


def test_a_resale_restricted_source_is_reported_apart_from_redistribution(
    monkeypatch,
):
    """A no-resale clause forbids selling, not sharing.

    Filing it under "Redistribution restricted" would tell the reader
    they may not pass the bundle on at all, which the clause does not say.
    """
    terms = _fabricated_source_terms(monkeypatch)

    assert [e['source_id'] for e in terms['resale_restricted']] == ['examplepa']
    assert terms['restricted'] == []

    notice = format_notice(RECIPE, terms)
    assert 'Resale restricted' in notice
    assert 'examplepa' in notice.split('Resale restricted')[1]
    assert 'Redistribution restricted' not in notice


def test_a_source_without_a_resale_clause_adds_no_resale_section(monkeypatch):
    terms = _fabricated_source_terms(monkeypatch, resale_restricted=None)

    assert terms['resale_restricted'] == []
    assert 'Resale restricted' not in format_notice(RECIPE, terms)


def test_a_restricted_source_replaces_that_sentence_with_a_pointer():
    composition = _COMPOSITION + ['parcel.edgecombecounty'] * 40
    notice = format_notice(RECIPE, bundle_terms(RECIPE, pd.Series(composition)))

    assert 'generally satisfied by keeping this notice' not in notice
    assert 'restrict redistribution; see' in notice
    assert 'decision for you as the distributor' in notice


def test_the_notice_names_the_recipe_roots(terms):
    from openplaces.io.bundle_terms import format_notice
    from openplaces.path import BUNDLED_RECIPES_DIR

    text = format_notice(RECIPE, terms)
    assert 'Recipes' in text
    assert 'openplaces' in text
    assert str(BUNDLED_RECIPES_DIR) in text


# How far a restricted source's data can reach.


def _fake_recipes(monkeypatch, recipes, edges):
    """Stand up a small recipe graph for the source walk."""
    from types import SimpleNamespace

    import openplaces.io.bundle_terms as module

    def get_recipe_by_id(recipe_id):
        return recipes[recipe_id]

    def get_recipe_dependencies(recipe, admin_id=None):
        return [
            SimpleNamespace(upstream_recipe_id=up, resolved=True)
            for up in edges.get(recipe['id'], [])
        ]

    monkeypatch.setattr(module, 'get_recipe_by_id', get_recipe_by_id)
    monkeypatch.setattr(module, 'get_recipe_dependencies', get_recipe_dependencies)
    monkeypatch.setattr(module, '_catalog_terms', lambda: {})


def _restricted_recipe(recipe_id, entity_type, columns):
    from types import SimpleNamespace

    source = SimpleNamespace(
        source_id='closedvendor',
        license='fabricated: no redistribution',
        terms_url='https://example.org/terms',
        portal_url=None,
        redistribution_restricted=True,
        resale_restricted=None,
        usage_requirement=None,
        team_sharing_permitted=None,
    )
    entity = SimpleNamespace(entity_type=entity_type, source=source)
    return {
        'id': recipe_id,
        'admin_id': 'US',
        'entity': entity,
        'columns': columns,
    }


def test_a_source_linked_for_named_columns_reaches_only_those(monkeypatch):
    from openplaces.io.bundle_terms import restricted_inputs

    vendor = _restricted_recipe(
        'US_property-closedvendor-2026',
        'property',
        {'year_built': 'YB', 'county_fips': 'FIPS', 'occupancy_type_raw': 'USE'},
    )
    spine = {
        'id': 'US_footprint-spine-2026',
        'pipeline': [
            {
                'step': 'link_by_id',
                'recipe_id': 'US_property-closedvendor-2026',
                'columns': {'occupancy_type_raw': 'occupancy_type_property_vendor'},
                'count_as': 'n_permits_per_footprint',
            }
        ],
    }
    curate = {'id': 'US_footprint-openplaces-2026', 'entity_recipe': spine['id']}
    _fake_recipes(
        monkeypatch,
        {r['id']: r for r in (vendor, spine, curate)},
        {curate['id']: [spine['id']], spine['id']: [vendor['id']]},
    )

    (entry,) = restricted_inputs(curate)

    assert entry['layer_source'] is False
    assert entry['attributes'] == [
        'n_permits_per_footprint',
        'occupancy_type_property_vendor',
    ]


def test_a_source_read_as_a_layer_reaches_everything_it_maps(monkeypatch):
    from openplaces.io.bundle_terms import restricted_inputs

    roll = _restricted_recipe(
        'US-NC-XX_parcel-closedvendor-2026',
        'parcel',
        {'year_built': 'YB', 'land_value': 'LV'},
    )
    geospine = {
        'id': 'US_parcel-geospine-2026',
        'pipeline': [{'step': 'resolve_spine'}],
    }
    curate = {'id': 'US_parcel-openplaces-2026', 'entity_recipe': geospine['id']}
    _fake_recipes(
        monkeypatch,
        {r['id']: r for r in (roll, geospine, curate)},
        {curate['id']: [geospine['id']], geospine['id']: [roll['id']]},
    )

    (entry,) = restricted_inputs(curate)

    assert entry['layer_source'] is True
    assert entry['attributes'] == ['land_value', 'year_built']
