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


def test_filename_suffix_is_not_mistaken_for_the_source():
    # A suffixed id ends in the suffix, not in the source token; taking
    # the last token made 'rural' the source id of an igac recipe.
    assert source_id_from_recipe_id('CO_parcel-igac-2026_rural') == 'igac'
    assert source_id_from_recipe_id('US_admin-census-2025_admin3') == 'census'


def test_entity_plus_dataset_recipe_id():
    # The entity token carries no source, so the dataset token answers.
    assert (
        source_id_from_recipe_id('US_footprint_built-n-stories-brails-2026') == 'brails'
    )
