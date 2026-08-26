"""Every recipe YAML must declare `admin_id:` explicitly.

A recipe missing the field was previously reported with an empty `admin_id`
by `find_recipes`, silently dropping it from `_expand_auto_discover`'s
discovery -- a real bug found while building the transaction spine (MA's
masslandrecords was never being discovered, because
US-MA_transaction-masslandrecords-v1.yaml had no `admin_id:` field). Rather
than papering over that in `find_recipes`, every recipe is required to state
its scope explicitly (global recipes use `admin_id: NULL`).
"""

from pathlib import Path

import pandas as pd
import yaml

from openplaces.diagnostics import find_recipes
from openplaces.path import spine_path


def test_all_recipes_declare_admin_id_explicitly():
    recipes_root = Path(__file__).parents[2] / 'src' / 'openplaces' / 'recipes'
    missing = []
    for yaml_path in sorted(recipes_root.rglob('*.yaml')):
        with open(yaml_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if 'entity' not in data and 'dataset' not in data:
            continue
        if 'admin_id' not in data:
            missing.append(yaml_path)
    assert not missing, f'Recipes missing an explicit admin_id: field: {missing}'


def _unit(admin_id):
    """Return (name, national code) for an admin id, from the spine.

    Asserting a literal admin id goes stale on every re-mint - this test
    read `US-NC-NH` until North Carolina's counties widened to three
    characters. The unit's own national code is issued by the Census, not
    by openplaces, so it survives a re-mint and is what a test should
    pin.
    """
    level = admin_id.count('-') + 1
    column = f'admin{level}_id'
    spine = pd.read_csv(
        spine_path(level),
        dtype=str,
        keep_default_na=False,
        usecols=[column, f'{column}_admin1', 'name'],
    ).set_index(column)
    row = spine.loc[admin_id]
    return row['name'], row[f'{column}_admin1']


def test_explicit_admin_id_recipes_resolve():
    df = find_recipes('transaction', stage='ingest')
    masslandrecords = df[df['source_id'] == 'masslandrecords']
    nhcgov = df[df['source_id'] == 'nhcgov']
    widor = df[df['source_id'] == 'widor']
    assert _unit(masslandrecords['admin_id'].iloc[0]) == ('Massachusetts', '25')
    assert _unit(nhcgov['admin_id'].iloc[0]) == ('New Hanover', '37129')
    assert _unit(widor['admin_id'].iloc[0]) == ('Wisconsin', '55')
