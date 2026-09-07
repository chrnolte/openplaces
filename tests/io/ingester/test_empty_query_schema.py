"""A recipe `query` that empties a partition by design returns the schema the
populated partitions have, and carries the index name whichever way the recipe
names it.

The early return skipped the column prune, so an empty chunk kept the raw
read's own column names. Concatenating it with real chunks then gave the
aggregate a set of stray all-null source columns no populated chunk carries.
The index name was read from one function indexer's kwarg only, so a
`create_index: {method: prefix, name: ...}` recipe lost it and the file landed
on disk with an `__index_level_0__` key.
"""

import pandas as pd

from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _ingester(recipe):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = recipe
    ingester.download_partition = {}
    ingester.processing_chunk = {'admin_id_to_process': 'XX-AA'}
    ingester.timer = Timer('test')
    ingester.verbose = False
    return ingester


def _raw():
    return pd.DataFrame(
        {'CODE': ['a'], 'NAME': ['b'], 'INTERNAL_SEQ': [1], 'LEVEL': ['4']}
    )


def _recipe(**extra):
    recipe = {
        'entity': None,
        'dataset': 'units',
        'columns': {'admin4_id_source': 'CODE', 'name': 'NAME'},
        'query': "LEVEL == '5'",
    }
    recipe.update(extra)
    return recipe


def test_an_emptied_chunk_keeps_only_the_named_columns():
    df = _ingester(_recipe())._preprocess_recipe_data(_raw())

    assert df.empty
    assert list(df.columns) == ['admin4_id_source', 'name']


def test_a_prefix_indexer_still_names_the_index():
    recipe = _recipe(
        create_index={
            'method': 'prefix',
            'prefix': 'XX-',
            'name': 'admin4_id',
            'column': 'admin4_id_source',
        }
    )
    df = _ingester(recipe)._preprocess_recipe_data(_raw())

    assert df.index.name == 'admin4_id'


def test_a_function_indexer_named_by_args_name_is_honored():
    recipe = _recipe(
        create_index={
            'function': 'openplaces.geo.ids.add_openlocationcode_index',
            'args': {'name': 'footprint_id'},
        }
    )
    df = _ingester(recipe)._preprocess_recipe_data(_raw())

    assert df.index.name == 'footprint_id'


def test_keep_unnamed_columns_is_respected():
    """A recipe that deliberately keeps unmapped columns still gets them."""
    recipe = _recipe(keep_unnamed_columns=True)
    df = _ingester(recipe)._preprocess_recipe_data(_raw())

    assert 'INTERNAL_SEQ' in df.columns
