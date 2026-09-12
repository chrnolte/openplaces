"""A recipe that folds its partitions into one `_all` file per unit is
scheduled against that file, the one `get_entities` reads.

`US-FL_property-fldor-2026` and `US-FL_transaction-fldor-2026` declare
`aggregate_by: single_file`; the graph named their plain per-unit path,
so every run re-scheduled the ingest and then failed it for a missing
output.
"""

from openplaces.flow import RecipeDAG
from openplaces.flow.dag import primary_output_path
from openplaces.recipe import get_recipe_by_id


def test_single_file_recipe_names_its_all_file():
    recipe = get_recipe_by_id('US-FL_transaction-fldor-2026')
    assert recipe['aggregate_by']['single_file']
    path = primary_output_path(recipe, admin_id='US-FL-LA')
    assert path.name == 'US-FL-LA_transaction-fldor-2026_all.parquet'


def test_a_plain_recipe_is_unchanged():
    recipe = get_recipe_by_id('US_footprint-fema-2023')
    path = primary_output_path(recipe, admin_id='US-TX-KEY')
    assert path.name == 'US-TX-KEY_footprint-fema-2023.parquet'


def test_the_graph_and_its_consumers_agree_on_the_file():
    dag = RecipeDAG(
        'US_transaction-openplaces-2026', admin_ids=['US-FL-LA'], deliver=False
    )
    produced = dag.output_path('ingest', 'US-FL_transaction-fldor-2026', 'US-FL-LA')
    assert produced.name.endswith('_all.parquet')
    consumed = {
        p.name
        for p in dag.input_paths('harmonize', 'US_transaction-spine-2026', 'US-FL-LA')
    }
    assert produced.name in consumed
