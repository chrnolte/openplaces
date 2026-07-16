"""Tests for openplaces.recipe.source_id_from_recipe_id."""

from openplaces.recipe import source_id_from_recipe_id


def test_recipe_id_with_admin_prefix():
    assert source_id_from_recipe_id('US_building-nsi-2022') == 'nsi'


def test_recipe_id_without_admin_prefix():
    assert source_id_from_recipe_id('dwelling-overture-2025') == 'overture'


def test_recipe_id_with_filename_suffix_token():
    # The last _-delimited token is parsed, so a crosswalk-style key with an
    # admin prefix and entity token still yields the source id.
    assert source_id_from_recipe_id('US-MA_parcel-massgis-2025') == 'massgis'


def test_irregular_recipe_id_falls_back_to_whole_token():
    assert source_id_from_recipe_id('US_spine') == 'spine'
