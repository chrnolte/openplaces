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

import yaml

from openplaces.diagnostics import find_recipes


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


def test_explicit_admin_id_recipes_resolve():
    df = find_recipes('transaction', stage='ingest')
    masslandrecords = df[df['source_id'] == 'masslandrecords']
    nhcgov = df[df['source_id'] == 'nhcgov']
    widor = df[df['source_id'] == 'widor']
    assert masslandrecords['admin_id'].iloc[0] == 'US-MA'
    assert nhcgov['admin_id'].iloc[0] == 'US-NC-NH'
    assert widor['admin_id'].iloc[0] == 'US-WI'
