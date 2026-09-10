"""The names a connector (spoke) needs are on the public API.

A spoke may read the hub only through `openplaces`'s public module, never
a submodule import. Each export here removed a guess a spoke was making:
the registry's aggregation rule, the provenance-suffix resolver, admin
containment and ancestry, and a recipe's columns without its data.
"""

import pandas as pd
import pytest

import openplaces as op


@pytest.mark.parametrize(
    'name',
    [
        'get_attribute_registry',
        'get_agg_func',
        'resolve_attribute_name',
        'admin_scope_covers',
        'get_admin_ancestor',
        'describe_recipe',
    ],
)
def test_exported(name):
    assert name in op.__all__
    assert callable(getattr(op, name))


def test_registry_exposes_the_aggregation_rule():
    registry = op.get_attribute_registry()
    assert isinstance(registry, pd.DataFrame)
    assert 'aggregation' in registry.columns
    assert op.get_agg_func('land_value') == 'sum'
    assert op.get_agg_func('wetland_share') == 'mean'


def test_ancestry_and_containment_agree():
    assert op.get_admin_ancestor('US-NC-WAR', 2) == 'US-NC'
    assert op.admin_scope_covers('US-NC', 'US-NC-WAR')
    assert not op.admin_scope_covers('US-NC-WA', 'US-NC-WAR')
