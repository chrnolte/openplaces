"""A bundle carries share-alike data only when it is one family.

Maintainer's rule (2026-10-23): share-alike is acceptable as long as it
is internally compatible. ODbL or CC BY-SA, never both, because each
requires the whole release to carry it. Sources, licences and recipes
here are fabricated, so the contract does not move as recipes change.
"""

from __future__ import annotations

import pandas as pd
import pytest

import openplaces.io.delivery as delivery_module
import openplaces.io.delivery.terms as terms_module
from openplaces.io.delivery import (
    ShareAlikeConflictError,
    check_share_alike,
    delivery_paths,
    export_delivery,
)
from openplaces.io.delivery.terms import bundle_terms, share_alike_family
from tests.io.test_delivery import _county, _recipe


def _entry(source_id, license_text):
    return {
        'source_id': source_id,
        'recipe_id': f'US-XX-YY_parcel-{source_id}-2026',
        'license': license_text,
        'terms_url': None,
        'portal_url': None,
        'redistribution_restricted': None,
        'resale_restricted': None,
        'usage_requirement': None,
    }


@pytest.fixture
def catalog(monkeypatch):
    """Make the terms walk see a chosen set of fabricated sources."""

    def use(**licences):
        # A licence string, or a dict of entry fields for a source whose
        # reach (entity, scope, mapped attributes) matters.
        entries = [
            {**_entry(source_id, spec['license']), **spec}
            if isinstance(spec, dict)
            else _entry(source_id, spec)
            for source_id, spec in licences.items()
        ]
        known = {terms_module._terms_key(e): e for e in entries}
        monkeypatch.setattr(terms_module, '_source_terms', lambda *_a, **_kw: known)
        monkeypatch.setattr(terms_module, '_catalog_terms', lambda: {})
        monkeypatch.setattr(delivery_module, 'restricted_inputs', lambda *_a, **_kw: [])

    return use


# Which licences belong together.


@pytest.mark.parametrize(
    ('text', 'family'),
    [
        ('ODbL-1.0', 'ODbL'),
        ('CC-BY-SA-4.0', 'CC BY-SA'),
        ('CC BY-SA 3.0 (fabricated portal)', 'CC BY-SA'),
        ('CC-BY-SA-2.0', 'CC BY-SA'),
        ('CC-BY-SA-1.0', 'CC BY-SA 1.0'),
        ('GPL-3.0-or-later', 'GPL'),
        ('GPL-2.0-or-later', 'GPL'),
        ('GPL-2.0-only', 'GPL-2.0-only'),
        ('OGL-SA', 'OGL-SA'),
        ('CC-BY-4.0', None),
        ('OGL-UK-3.0', None),
        ('public domain', None),
        (None, None),
    ],
)
def test_licences_fall_into_families(text, family):
    assert share_alike_family(text) == family


# The verdict on a bundle's terms.


def test_odbl_alone_passes(catalog):
    catalog(alpha='ODbL-1.0', beta='ODbL-1.0')
    terms = bundle_terms(_recipe(), pd.Series(['alpha'] * 3 + ['beta']))

    assert set(terms['share_alike_families']) == {'ODbL'}
    assert not terms['share_alike_conflict']


def test_two_versions_of_cc_by_sa_pass(catalog):
    """BY-SA 3.0 allows a later version, so both ship under 4.0."""
    catalog(alpha='CC-BY-SA-3.0', beta='CC-BY-SA-4.0')
    terms = bundle_terms(_recipe(), pd.Series(['alpha', 'beta']))

    assert set(terms['share_alike_families']) == {'CC BY-SA'}
    assert len(terms['share_alike_families']['CC BY-SA']) == 2
    assert not terms['share_alike_conflict']


def test_odbl_and_cc_by_sa_conflict(catalog):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    terms = bundle_terms(_recipe(), pd.Series(['alpha'] * 9 + ['beta']))

    assert terms['share_alike_conflict']
    beta = terms['share_alike_families']['CC BY-SA'][0]
    assert beta['source_id'] == 'beta'
    assert beta['share'] == pytest.approx(0.1)


def test_permissive_sources_beside_one_family_pass(catalog):
    catalog(alpha='ODbL-1.0', gamma='CC-BY-4.0', delta='public domain')
    terms = bundle_terms(_recipe(), pd.Series(['alpha', 'gamma', 'delta']))

    assert set(terms['share_alike_families']) == {'ODbL'}
    assert not terms['share_alike_conflict']


def test_cells_a_source_filled_carry_its_licence(catalog):
    """A roll that drew no outline still ships in the values it filled."""
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    terms = bundle_terms(
        _recipe(), pd.Series(['alpha'] * 4), value_cells={'beta': 30, 'alpha': 10}
    )

    assert terms['share_alike_conflict']
    beta = terms['share_alike_families']['CC BY-SA'][0]
    assert beta['cells'] == 30
    assert beta['cell_share'] == pytest.approx(0.75)


def test_a_source_that_left_nothing_in_the_bundle_passes_nothing_on(catalog):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    terms = bundle_terms(_recipe(), pd.Series(['alpha'] * 4), value_cells={})

    assert set(terms['share_alike_families']) == {'ODbL'}
    assert not terms['share_alike_conflict']


# The export and the dry check.


def _two_counties(alpha_source, beta_source):
    _county(
        ['a', 'b'], admin3_id='US-NC-AL', extra={'geometry_source': [alpha_source] * 2}
    )
    _county(
        ['c', 'd'], admin3_id='US-NC-BB', extra={'geometry_source': [beta_source] * 2}
    )


def test_export_refuses_two_families_and_writes_nothing(mock_data_root, catalog):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    _two_counties('alpha', 'beta')

    with pytest.raises(ShareAlikeConflictError) as raised:
        export_delivery(_recipe(), 'US-NC', admin_ids=['US-NC-AL', 'US-NC-BB'])

    message = str(raised.value)
    assert 'alpha (ODbL-1.0): 50.0% of geometry' in message
    assert 'beta (CC-BY-SA-4.0): 50.0% of geometry' in message
    paths = delivery_paths(_recipe(), 'US-NC')
    for path in paths.values():
        assert not path.exists()
        assert not list(path.parent.glob(f'*{path.stem}*'))


def test_export_ships_one_family(mock_data_root, catalog):
    catalog(alpha='CC-BY-SA-3.0', beta='CC-BY-SA-4.0')
    _two_counties('alpha', 'beta')

    paths = export_delivery(_recipe(), 'US-NC', admin_ids=['US-NC-AL', 'US-NC-BB'])

    assert paths['canonical'].exists()


def test_the_dry_check_gives_the_export_verdict_without_writing(
    mock_data_root, catalog
):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    _two_counties('alpha', 'beta')

    report = check_share_alike(_recipe(), admin_id='US-NC')

    assert report['conflict']
    assert set(report['families']) == {'ODbL', 'CC BY-SA'}
    assert 'beta (CC-BY-SA-4.0)' in report['message']
    for path in delivery_paths(_recipe(), 'US-NC').values():
        assert not path.exists()


def test_a_roll_without_sidecars_still_counts_toward_its_family(
    mock_data_root, catalog
):
    """A county roll's values need no `_source` of their own to count.

    Found by the redaction rules instead: inside its scope, a column
    named after an attribute it maps is its value. A sales bundle's
    price column carries no sidecar, and is the roll's all the same.
    """
    catalog(
        alpha='ODbL-1.0',
        beta={
            'license': 'CC-BY-SA-4.0',
            'entity_type': 'parcel',
            'admin_id': 'US-NC-BB',
            'attributes': ['structure_value_per_area'],
            'layer_source': True,
        },
    )
    _two_counties('alpha', 'alpha')

    report = check_share_alike(_recipe(), admin_id='US-NC')

    assert report['conflict']
    beta = report['families']['CC BY-SA'][0]
    assert beta['source_id'] == 'beta'
    # Its county's two structure values, and the two occupancy types
    # whose sidecar names only its layer ('parcel').
    assert beta['cells'] == 4


def test_a_national_source_does_not_claim_columns_by_name(mock_data_root, catalog):
    """Only tokens say what a national layer filled.

    Several national sources share a layer and its attribute names, so
    a column named after what one maps is not evidence of it; a value
    whose sidecar names only its layer still counts.
    """
    catalog(
        alpha={
            'license': 'ODbL-1.0',
            'entity_type': 'footprint',
            'admin_id': None,
            'attributes': ['structure_value_per_area'],
            'layer_source': True,
        },
        beta='CC-BY-SA-4.0',
    )
    _two_counties('beta', 'beta')

    assert not check_share_alike(_recipe(), admin_id='US-NC')['conflict']

    _county(
        ['c', 'd'],
        admin3_id='US-NC-BB',
        extra={
            'geometry_source': ['beta'] * 2,
            'structure_value_per_area_source': ['footprint'] * 2,
        },
    )
    report = check_share_alike(_recipe(), admin_id='US-NC')

    assert report['conflict']
    assert report['families']['ODbL'][0]['cells'] == 2


# Withholding one family, declared per region.


def _withholding(families):
    recipe = _recipe()
    recipe['share']['delivery']['withhold_share_alike'] = families
    return recipe


def _cc_by_sa_outlines_with_odbl_values():
    """CC BY-SA outlines; one value column filled from an ODbL source."""
    for admin3_id, ids in (('US-NC-AL', ['a', 'b']), ('US-NC-BB', ['c', 'd'])):
        _county(
            ids,
            admin3_id=admin3_id,
            extra={
                'geometry_source': ['beta'] * 2,
                'year_built_source': ['alpha'] * 2,
            },
        )


def test_the_same_bundle_conflicts_without_the_declaration(mock_data_root, catalog):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    _cc_by_sa_outlines_with_odbl_values()

    with pytest.raises(ShareAlikeConflictError, match='withhold_share_alike'):
        export_delivery(_recipe(), 'US-NC', admin_ids=['US-NC-AL', 'US-NC-BB'])


def test_withholding_odbl_resolves_the_conflict(mock_data_root, catalog):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    _cc_by_sa_outlines_with_odbl_values()
    recipe = _withholding(['odbl'])

    paths = export_delivery(recipe, 'US-NC', admin_ids=['US-NC-AL', 'US-NC-BB'])

    canonical = pd.read_parquet(paths['canonical'])
    # The CC BY-SA rows all ship; the ODbL values in them do not.
    assert sorted(canonical.index) == ['a', 'b', 'c', 'd']
    assert canonical['year_built'].isna().all()
    assert (canonical['year_built_source'] == 'withheld').all()
    assert canonical['occupancy_type'].notna().all()
    notice = paths['terms'].read_text(encoding='utf-8')
    assert 'Withheld to keep one share-alike licence' in notice
    assert 'licensed under ODbL' in notice
    assert 'alpha: 0 rows left out, 4 values emptied' in notice
    # Its licence is not one the bundle has to carry.
    assert 'licensed ODbL-1.0' not in notice
    assert 'CC-BY-SA-4.0' in notice


def test_withholding_leaves_out_rows_the_family_drew(mock_data_root, catalog):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    _two_counties('alpha', 'beta')
    recipe = _withholding(['ODbL'])

    paths = export_delivery(recipe, 'US-NC', admin_ids=['US-NC-AL', 'US-NC-BB'])

    canonical = pd.read_parquet(paths['canonical'])
    assert sorted(canonical.index) == ['c', 'd']
    assert 'alpha: 2 rows left out' in paths['terms'].read_text(encoding='utf-8')


def test_the_dry_check_reports_what_withholding_leaves(mock_data_root, catalog):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    _cc_by_sa_outlines_with_odbl_values()

    report = check_share_alike(_withholding(['odbl']), admin_id='US-NC')

    assert not report['conflict']
    assert set(report['families']) == {'CC BY-SA'}
    assert report['withhold_share_alike'] == ['ODbL']
    assert report['withheld'] == [
        {
            'source_id': 'alpha',
            'license': 'ODbL-1.0',
            'terms_url': None,
            'share_alike_family': 'ODbL',
            'rows': 0,
            'cells': 4,
        }
    ]
    for path in delivery_paths(_recipe(), 'US-NC').values():
        assert not path.exists()


def test_a_misspelt_family_is_refused():
    with pytest.raises(ValueError, match='not a share-alike family'):
        delivery_module.delivery_regions(_withholding(['odbI']))


def test_withholding_for_an_undeclared_region_is_refused():
    recipe = _recipe()
    recipe['share']['delivery'] = {
        'admin_level': 2,
        'regions': ['northern-wisconsin'],
        'withhold_share_alike': {'nowhere': ['odbl']},
    }
    with pytest.raises(KeyError, match='nowhere'):
        delivery_module.delivery_regions(recipe)


def test_greenville_nc_withholds_odbl_and_nothing_else_does():
    """The declared case (maintainer, 2026-10-23, decision 29(b))."""
    regions = {
        spec['region_id']: spec
        for spec in delivery_module.delivery_regions('US_transaction-openplaces-2026')
    }
    assert regions['greenville-nc']['withhold_share_alike'] == ['ODbL']
    assert regions['greenville-nc']['admin_ids'] == ['US-NC-PIT']
    others = [r for name, r in regions.items() if name != 'greenville-nc']
    assert all(not r['withhold_share_alike'] for r in others)


def test_cheer_eastern_nc_withholds_cc_by_sa_in_public_only():
    """Its team twin keeps Pitt (maintainer, 2026-10-22, option A)."""
    regions = {
        spec['region_id']: spec
        for spec in delivery_module.delivery_regions('US_footprint-openplaces-2026')
    }
    assert regions['cheer-eastern-nc']['withhold_share_alike'] == ['CC BY-SA']
    assert regions['cheer-eastern-nc.team']['withhold_share_alike'] == []


# A team bundle is never published, and share-alike binds only on
# publication (maintainer, 2026-10-23).


@pytest.fixture
def public_and_team(monkeypatch):
    """One region withholding ODbL in public, and its team twin."""
    public = {
        'region_id': 'test-region',
        'admin_level': 2,
        'admin_ids': ['US-NC-AL', 'US-NC-BB'],
        'admin_id': None,
        'audience': 'public',
        'whole_unit': False,
        'withhold_share_alike': ['ODbL'],
    }
    team = {
        **public,
        'region_id': 'test-region.team',
        'audience': 'team',
        'base_region_id': 'test-region',
        'withhold_share_alike': [],
    }
    monkeypatch.setattr(delivery_module, 'delivery_regions', lambda _: [public, team])


def test_a_team_bundle_may_carry_two_families(mock_data_root, catalog, public_and_team):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    _cc_by_sa_outlines_with_odbl_values()

    paths = export_delivery(_recipe(), region='test-region.team')

    canonical = pd.read_parquet(paths['canonical'])
    assert canonical['year_built'].notna().all()
    assert (canonical['year_built_source'] == 'alpha').all()
    notice = paths['terms'].read_text(encoding='utf-8')
    # Named as information, not withheld.
    assert 'THESE CANNOT ALL BE SATISFIED AT ONCE' in notice
    assert 'Withheld to keep one share-alike licence' not in notice


def test_the_public_twin_still_withholds(mock_data_root, catalog, public_and_team):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    _cc_by_sa_outlines_with_odbl_values()

    paths = export_delivery(_recipe(), region='test-region')

    assert pd.read_parquet(paths['canonical'])['year_built'].isna().all()


def test_the_dry_check_reports_a_team_bundles_families_as_information(
    mock_data_root, catalog, public_and_team
):
    catalog(alpha='ODbL-1.0', beta='CC-BY-SA-4.0')
    _cc_by_sa_outlines_with_odbl_values()

    team = check_share_alike(_recipe(), region='test-region.team')
    public = check_share_alike(_recipe(), region='test-region')

    assert team['mixed_families'] and not team['conflict']
    assert team['message'] is None
    assert set(team['families']) == {'ODbL', 'CC BY-SA'}
    assert team['withheld'] == []
    assert not public['mixed_families'] and not public['conflict']


def test_a_team_twin_does_not_inherit_the_declaration():
    recipe = _recipe()
    recipe['share']['delivery'] = {
        'admin_level': 2,
        'regions': ['northern-wisconsin'],
        'team_regions': ['northern-wisconsin'],
        'withhold_share_alike': {'northern-wisconsin': ['odbl']},
    }
    specs = {s['region_id']: s for s in delivery_module.delivery_regions(recipe)}

    assert specs['northern-wisconsin']['withhold_share_alike'] == ['ODbL']
    assert specs['northern-wisconsin.team']['withhold_share_alike'] == []


def test_the_dry_check_reports_members_without_a_curated_file(mock_data_root, catalog):
    catalog(alpha='ODbL-1.0')
    _county(['a', 'b'], admin3_id='US-NC-AL', extra={'geometry_source': ['alpha'] * 2})

    report = check_share_alike(_recipe(), admin_id='US-NC', allow_missing=True)

    assert report['units'] == ['US-NC-AL']
    assert report['missing'] == ['US-NC-BB']
    assert not report['conflict']
