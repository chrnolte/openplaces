"""Which delivery regions get their primary columns imputed."""

from openplaces.io.delivery import imputation_spec

RECIPE = {
    'recipe_id': 'US_footprint-example-2026',
    'share': {
        'columns': ['lat', 'long'],
        'delivery': {
            'admin_level': 2,
            'regions': ['cheer-eastern-nc', 'western-nc'],
            'team_regions': ['cheer-eastern-nc'],
            'impute': {'regions': ['cheer-eastern-nc'], 'version': 'm'},
        },
    },
}


def test_only_listed_regions_are_imputed():
    assert imputation_spec(RECIPE, 'cheer-eastern-nc')['version'] == 'm'
    assert imputation_spec(RECIPE, 'western-nc') is None


def test_a_team_twin_follows_its_base_region():
    assert imputation_spec(RECIPE, 'cheer-eastern-nc.team')['version'] == 'm'


def test_no_block_means_no_imputation():
    recipe = {
        **RECIPE,
        'share': {
            **RECIPE['share'],
            'delivery': {
                k: v for k, v in RECIPE['share']['delivery'].items() if k != 'impute'
            },
        },
    }
    assert imputation_spec(recipe, 'cheer-eastern-nc') is None
