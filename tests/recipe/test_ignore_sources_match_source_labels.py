"""Every duplicate-resolution label names a label the source can carry.

A point link's ``resolve_duplicates.ignore_sources`` lists low-rank
sources by the label the linked recipe's source-labels sidecar gives
them at ingest. A relabel in the sidecar (NSI 2026: H became "HIFLD")
that leaves a recipe on the old string turns every duplicate rule off
without an error, so this test fails the build instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import openplaces

RECIPES = Path(openplaces.__file__).parent / 'recipes'


def _steps_with_ignore_sources(node, found):
    if isinstance(node, dict):
        dup = (node.get('thresholds') or {}).get('resolve_duplicates')
        if isinstance(dup, dict) and 'ignore_sources' in dup:
            labels = dup['ignore_sources'] or []
            found.append((node.get('recipe_id'), list(labels)))
        for value in node.values():
            _steps_with_ignore_sources(value, found)
    elif isinstance(node, list):
        for value in node:
            _steps_with_ignore_sources(value, found)


def _cases():
    cases = []
    for path in sorted(RECIPES.rglob('*.yaml')):
        text = path.read_text(encoding='utf-8')
        if 'ignore_sources' not in text:
            continue
        found: list = []
        _steps_with_ignore_sources(yaml.safe_load(text), found)
        for recipe_id, labels in found:
            cases.append(pytest.param(recipe_id, labels, id=f'{path.stem}:{recipe_id}'))
    return cases


CASES = _cases()


def test_some_recipe_resolves_duplicates():
    # Guards the parametrized test below against passing vacuously.
    assert CASES


@pytest.mark.parametrize(('recipe_id', 'labels'), CASES)
def test_ignore_sources_are_sidecar_labels(recipe_id, labels):
    assert recipe_id, 'a resolve_duplicates block needs the step recipe_id'
    sidecars = list(RECIPES.rglob(f'{recipe_id}_source-labels.csv'))
    assert len(sidecars) == 1, f'no single source-labels sidecar for {recipe_id}'
    known = set(pd.read_csv(sidecars[0], dtype=str)['source_label'])
    missing = sorted(set(labels) - known)
    assert not missing, f'{recipe_id}: {missing} not in {sidecars[0].name}'
