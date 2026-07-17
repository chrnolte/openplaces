"""Tests for the globally-usable admin ID helpers in openplaces.io.admin.

Covers the language-extensible name cleaning, the country-agnostic name-based
ID generation (English and Spanish frames), and that the GADM-reconciliation
index function is parameterized by country rather than hardcoded to the US.
"""

import pandas as pd
import pytest

import openplaces.io.admin as admin
from openplaces.io.admin import clean_geographic_name, generate_admin_ids

# clean_geographic_name


def test_clean_strips_spanish_prefix_and_detects_generic():
    # 'San' honorific stripped; 'Comuna 5' detected as generic word + number.
    clean, digits, suffix, generic = clean_geographic_name('San José')
    assert not clean.startswith('SAN')

    clean, digits, suffix, generic = clean_geographic_name('Comuna 5')
    assert generic == 'comuna'
    assert digits == '5'


def test_clean_detects_english_generic_with_letter_suffix():
    clean, digits, suffix, generic = clean_geographic_name('Ward 3B')
    assert generic == 'ward'
    assert digits == '3'
    assert suffix == 'b'


def test_clean_treats_na_tokens_as_empty():
    assert clean_geographic_name('n/a') == ('', '', '', '')
    assert clean_geographic_name(None) == ('', '', '', '')


def test_clean_token_overrides_are_respected():
    # With a generic-word list that excludes 'ward', it is not detected.
    _, digits, _, generic = clean_geographic_name('Ward 3', generic_words=['comuna'])
    assert generic == ''
    assert digits == '3'


# generate_admin_ids


def _assert_valid_ids(result, parent_prefix):
    ids = result.index.to_series()
    assert ids.notna().all()
    assert not ids.duplicated().any()
    assert ids.str.match(r'^[A-Z0-9\-]+$').all()
    assert ids.str.startswith(parent_prefix).all()


def test_generate_admin_ids_english_frame():
    df = pd.DataFrame(
        {
            'admin3_id': ['US-NC-CE'] * 4,
            'name': ['North East', 'Springfield', 'Wilmington', 'Wrightsville'],
        }
    )
    result = generate_admin_ids(
        df, new_admin_id_col='admin4_id', parent_admin_id_col='admin3_id'
    )
    _assert_valid_ids(result, 'US-NC-CE-')


def test_generate_admin_ids_spanish_frame_folds_accents():
    df = pd.DataFrame(
        {
            'admin2_id': ['CO-AN'] * 4,
            'name': ['San José', 'El Carmen', 'Comuna 5', 'Medellín'],
        }
    )
    result = generate_admin_ids(
        df, new_admin_id_col='admin3_id', parent_admin_id_col='admin2_id'
    )
    _assert_valid_ids(result, 'CO-AN-')
    # 'Comuna 5' uses the Spanish generic word + number → 'C5'.
    assert 'CO-AN-C5' in set(result.index)


def test_generate_admin_ids_default_separator_is_hyphen():
    df = pd.DataFrame({'admin2_id': ['CO-AN'], 'name': ['Medellín']})
    result = generate_admin_ids(
        df, new_admin_id_col='admin3_id', parent_admin_id_col='admin2_id'
    )
    assert result.index[0].startswith('CO-AN-')


# admin3_id_index_from_local (country parameterization)


def test_admin3_index_from_local_uses_country_id(monkeypatch):
    calls = {'get_recipe_countries': [], 'get_admin_countries': []}

    def fake_get_recipe_by_id(recipe_id):
        assert recipe_id == 'CO_admin-dane-2025_admin2'
        return {'admin_id': 'CO', 'entity': 'admin-dane-2025'}

    def fake_get_recipe(country_id, admin_entity, filename=None, **kwargs):
        calls['get_recipe_countries'].append(country_id)
        if filename == 'admin3-names-from-gadm':
            return pd.DataFrame(
                columns=['admin2_id', 'admin3_name_gadm', 'admin3_name_official']
            )
        if filename == 'admin3-ids':
            return pd.DataFrame({'admin3_id_admin1': [], 'admin3_id': []})
        return {'_recipe': admin_entity}  # admin2 recipe object

    def fake_get_admin(*args, level=None, recipe=None, columns=None, **kwargs):
        # positional country_id for the level-3 call
        country_id = args[0] if args else None
        if level == 2:
            return pd.DataFrame(
                {'admin2_id_admin1': ['37']},
                index=pd.Index(['CO-AN'], name='admin2_id'),
            )
        calls['get_admin_countries'].append(country_id)
        gdf = pd.DataFrame(
            {'name': ['Medellin']}, index=pd.Index(['CO-AN-ME'], name='admin3_id')
        )
        return gdf

    monkeypatch.setattr(admin, 'get_recipe_by_id', fake_get_recipe_by_id)
    monkeypatch.setattr(admin, 'get_recipe', fake_get_recipe)
    monkeypatch.setattr(admin, 'get_admin', fake_get_admin)

    local = pd.DataFrame(
        {
            'name': ['Medellin'],
            'name_long': ['Medellin'],
            'admin2_id_admin1': ['37'],
            'admin3_id_admin1': ['37001'],
        }
    )
    result = admin.admin3_id_index_from_local(
        local, admin2_recipe_id='CO_admin-dane-2025_admin2'
    )

    assert 'CO' in calls['get_recipe_countries']
    assert calls['get_admin_countries'] == ['CO']  # never hardcoded 'US'
    assert result.index.name == 'admin3_id'
    assert result.loc['CO-AN-ME', 'name'] == 'Medellin'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
