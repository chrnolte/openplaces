"""A notice names the sources whose values ship, not only the geometry's.

houston-tx's notice named only txgio, the source of its outlines; HCAD,
which supplies Harris County's values through the property spine, was in
no section at all (2026-10-19). Values are counted from the bundle's own
`_source` columns. Fabricated frames, real recipe walk.
"""

import pandas as pd

from openplaces.io.delivery.terms import bundle_terms, format_notice, value_source_cells

RECIPE = 'US_parcel-openplaces-2026'


def test_cells_are_counted_per_specific_token():
    frame = pd.DataFrame(
        {
            'land_value_source': ['examplecad', 'examplecad+imputed', None, 'withheld'],
            'use_group_source': ['parcel', 'statelayer', 'statelayer', None],
            'geometry_source': ['statelayer'] * 4,
            'land_value': [1, 2, 3, 4],
        }
    )
    assert value_source_cells(frame) == {'examplecad': 2, 'statelayer': 2}


def test_a_roll_that_supplies_values_is_named():
    terms = bundle_terms(
        RECIPE,
        pd.Series(['txgio'] * 10),
        admin_ids=['US-TX-HAR'],
        value_cells={'hcad': 8, 'txgio': 3},
    )
    assert {e['source_id'] for e in terms['sources']} == {'txgio'}
    values = {e['source_id']: e for e in terms['values']}
    assert 'hcad' in values and values['hcad']['cells'] == 8
    # The geometry's own source is not listed twice.
    assert 'txgio' not in values
    notice = format_notice(RECIPE, terms)
    assert 'Values from' in notice and 'hcad' in notice


def _catalog_entry(source_id, admin_id, license=None):
    return {
        'source_id': source_id,
        'recipe_id': f'{admin_id}_transaction-{source_id}-2026',
        'license': license,
        'terms_url': None,
        'portal_url': None,
        'redistribution_restricted': None,
        'resale_restricted': None,
        'admin_id': admin_id,
    }


def _fabricated_catalog(monkeypatch, *entries):
    """A catalog of made-up sources and an empty walk, for scope tests."""
    import openplaces.io.delivery.terms as module

    catalog = {}
    for entry in entries:
        catalog.setdefault(entry['source_id'], []).append(entry)
    monkeypatch.setattr(module, '_catalog_terms', lambda: catalog)
    monkeypatch.setattr(
        module, '_source_terms', lambda recipe, admin_ids=None, key=None: {}
    )


def test_a_value_source_without_terms_is_reported_as_unchecked(monkeypatch):
    _fabricated_catalog(monkeypatch, _catalog_entry('examplerecorder', 'US-FL'))
    terms = bundle_terms(
        RECIPE,
        None,
        admin_ids=['US-FL-LA'],
        value_cells={'examplerecorder': 4},
    )
    unchecked = [e['source_id'] for e in terms['unrecorded'] if e.get('cells')]
    assert unchecked == ['examplerecorder']
    assert '4 values' in format_notice(RECIPE, terms)


def test_a_token_naming_another_states_source_is_not_listed(monkeypatch):
    # A step labels rows of every state with its first declared key, so
    # Florida's sales carried a North Carolina county's source id
    # (`address_source`, 2026-10-21). That source's recipes cover none
    # of the bundle's units, so its terms are not the bundle's.
    _fabricated_catalog(
        monkeypatch,
        _catalog_entry('examplerecorder', 'US-FL'),
        _catalog_entry('othercounty', 'US-NC-NHA'),
    )
    terms = bundle_terms(
        RECIPE,
        None,
        admin_ids=['US-FL-LA', 'US-FL-PO'],
        value_cells={'examplerecorder': 4, 'othercounty': 7},
    )
    named = {e['source_id'] for e in [*terms['values'], *terms['unrecorded']]}
    assert named == {'examplerecorder'}
    assert 'othercounty' not in format_notice(RECIPE, terms)


def test_a_county_source_inside_a_state_bundle_is_listed(monkeypatch):
    _fabricated_catalog(monkeypatch, _catalog_entry('othercounty', 'US-NC-NHA'))
    terms = bundle_terms(
        RECIPE, None, admin_ids=['US-NC'], value_cells={'othercounty': 7}
    )
    assert [e['source_id'] for e in terms['values']] == ['othercounty']


def test_a_method_token_is_not_a_source():
    terms = bundle_terms(
        RECIPE,
        pd.Series(['txgio'] * 10),
        admin_ids=['US-TX-HAR'],
        value_cells={'group_statistic': 9, 'rule': 2},
    )
    assert terms['values'] == []


def test_without_value_cells_the_notice_is_unchanged():
    terms = bundle_terms(RECIPE, pd.Series(['txgio'] * 10), admin_ids=['US-TX-HAR'])
    assert terms['values'] == []
    assert 'Values from' not in format_notice(RECIPE, terms)
