"""Where a region's bundle is filed, and that no two bundles share a path.

A named region sits in a folder named after it inside its unit's folder;
only a region declared as its whole unit (`whole_unit_regions`) is filed
as the unit's own bundle (maintainer, 2026-10-23). These tests pin that a
region's path never equals a statewide bundle's or a county's curated
output, for every recipe and every registered region, and that the paths
already shipped do not move.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from openplaces.config import cfg
from openplaces.core.schema import AdminId, Entity
from openplaces.io import delivery as delivery_module
from openplaces.io import readers
from openplaces.io.delivery import (
    delivery_admin_id,
    delivery_paths,
    delivery_regions,
    export_delivery,
)
from openplaces.recipe import get_output_path, get_recipe_by_id

RECIPES = Path(__file__).resolve().parents[2] / 'src' / 'openplaces' / 'recipes'


def _delivering_recipe_ids() -> list[str]:
    """Every recipe in the tree that declares a `share: delivery:` block."""
    found = []
    for path in sorted(RECIPES.rglob('*.yaml')):
        text = path.read_text(encoding='utf-8')
        if 'delivery:' not in text:
            continue
        recipe = yaml.safe_load(text) or {}
        if isinstance(recipe, dict) and (recipe.get('share') or {}).get('delivery'):
            found.append(path.stem)
    return found


DELIVERING = _delivering_recipe_ids()


def test_the_delivering_recipes_are_found():
    assert {
        'US_parcel-openplaces-2026',
        'US_footprint-cheer-2026',
        'US_transaction-openplaces-2026',
    } <= set(DELIVERING)


def _tail(path: Path, n: int) -> str:
    return '/'.join(path.parts[-n:])


def _curated_paths(recipe, members) -> set[Path]:
    """The per-unit curated outputs of *members* the recipe can write."""
    paths = set()
    for member in members:
        try:
            paths.add(get_output_path(recipe, member))
        except ValueError:
            # Outside the recipe's scope or not at its save level.
            continue
    return paths


def _every_registered_region(recipe) -> dict:
    """A copy of *recipe* declaring every registered region in its scope.

    Its own `whole_unit_regions` stay as declared, so the copy asks
    where each region would land if this recipe shipped it.
    """
    registry = readers.get_regions()
    scope = AdminId(str(recipe['admin_id']))
    in_scope = []
    for region_id, rows in registry.groupby('region_id', sort=False):
        anchor = str(rows['region_admin_id'].iat[0]).strip()
        if anchor and scope.is_parent_or_equal_of(AdminId(anchor)):
            in_scope.append(region_id)
    copy = dict(recipe)
    share = dict(recipe['share'])
    block = dict(share['delivery'])
    block['regions'] = in_scope
    block.pop('team_regions', None)
    block['whole_unit_regions'] = [
        r for r in block.get('whole_unit_regions') or [] if r in in_scope
    ]
    share['delivery'] = block
    copy['share'] = share
    return copy


@pytest.mark.parametrize('recipe_id', DELIVERING)
def test_no_region_bundle_shares_a_path(recipe_id):
    """Across every registered region, no two bundles land on one file.

    Nor does a region's bundle land on its unit's whole-unit path (the
    statewide bundle) unless it is that whole unit, nor on a county's
    curated output.
    """
    recipe = get_recipe_by_id(recipe_id)
    every = _every_registered_region(recipe)
    seen: dict[Path, str] = {}
    for spec in delivery_regions(every):
        region = spec['region_id']
        paths = delivery_paths(every, region=region)
        canonical = paths['canonical']
        assert canonical not in seen, (region, seen.get(canonical))
        seen[canonical] = region
        assert canonical not in _curated_paths(recipe, spec['admin_ids']), region
        unit = delivery_admin_id(every, region=region)
        if not spec['whole_unit']:
            # The folder is the region's own, inside its unit's folder.
            assert canonical.parent.name == region
            assert canonical.name.startswith(f'{unit}_')
        for path in paths.values():
            assert path.parent == canonical.parent


@pytest.mark.parametrize('recipe_id', DELIVERING)
def test_declared_regions_and_their_team_twins_have_their_own_paths(recipe_id):
    recipe = get_recipe_by_id(recipe_id)
    specs = delivery_regions(recipe)
    canonicals = [
        delivery_paths(recipe, region=s['region_id'])['canonical'] for s in specs
    ]
    assert len(set(canonicals)) == len(canonicals)
    curated = set()
    for spec in specs:
        curated |= _curated_paths(recipe, spec['admin_ids'])
    assert not curated & set(canonicals)


@pytest.mark.parametrize(
    ('recipe_id', 'region', 'tail'),
    [
        # The statewide bundles keep the state's own path.
        (
            'US_transaction-openplaces-2026',
            'wisconsin-statewide',
            'WI/_all/transaction/openplaces/2026/US-WI_transaction-openplaces-2026.parquet',
        ),
        (
            'US_transaction-openplaces-2026',
            'florida-statewide',
            'FL/_all/transaction/openplaces/2026/US-FL_transaction-openplaces-2026.parquet',
        ),
        # Regions already filed in a folder of their own stay there.
        (
            'US_footprint-cheer-2026',
            'cheer-eastern-nc',
            'NC/_all/footprint/cheer/2026/cheer-eastern-nc/'
            'US-NC_footprint-cheer-2026.parquet',
        ),
        (
            'US_footprint-cheer-2026',
            'cheer-eastern-nc.team',
            'NC/_all/footprint/cheer/2026/cheer-eastern-nc/team/'
            'US-NC_footprint-cheer-2026.parquet',
        ),
        (
            'US_parcel-openplaces-2026',
            'boston-inner-core',
            'MA/_all/parcel/openplaces/2026/boston-inner-core/'
            'US-MA_parcel-openplaces-2026.parquet',
        ),
        (
            'US_parcel-openplaces-2026',
            'nantucket-ma.team',
            'MA/_all/parcel/openplaces/2026/nantucket-ma/team/'
            'US-MA_parcel-openplaces-2026.parquet',
        ),
        # A sub-state region that used to sit flat on its state's path.
        (
            'US_footprint-cheer-2026',
            'cheer-coastal-tx',
            'TX/_all/footprint/cheer/2026/cheer-coastal-tx/'
            'US-TX_footprint-cheer-2026.parquet',
        ),
    ],
)
def test_shipped_paths(recipe_id, region, tail):
    canonical = delivery_paths(recipe_id, region=region)['canonical']
    assert _tail(canonical, tail.count('/') + 1) == tail


def test_the_statewide_bundles_keep_every_file_name():
    paths = delivery_paths('US_transaction-openplaces-2026', region='florida-statewide')
    assert {role: path.name for role, path in paths.items()} == {
        'canonical': 'US-FL_transaction-openplaces-2026.parquet',
        'evidence': 'US-FL_transaction-openplaces-2026_evidence.parquet',
        'terms': 'US-FL_transaction-openplaces-2026_LICENSE.txt',
    }
    assert paths['canonical'].parent.name == '2026'


# Fabricated recipes and regions. The ids are made up (US-NC-AL, US-NC-BB,
# US-NC-CC stand for any three counties); no real record is involved.


def _recipe(**delivery):
    return {
        'recipe_id': 'US_transaction-test-2026',
        'admin_id': AdminId('US'),
        'stage': 'curate',
        'entity': Entity('transaction', 'test', '2026'),
        'process_by': {'admin_level': 3},
        'save_to': {'data_dir': 'core'},
        'share': {
            'columns': ['admin3_id', 'price'],
            'geometry': False,
            'delivery': {'admin_level': 2, **delivery},
        },
    }


@pytest.fixture
def registry(monkeypatch):
    """A fabricated region registry: a state, and three one-county regions."""
    rows = [
        ('test-statewide', 'US-NC', 'US-NC-AL'),
        ('test-statewide', 'US-NC', 'US-NC-BB'),
        ('test-statewide', 'US-NC', 'US-NC-CC'),
        ('test-also-statewide', 'US-NC', 'US-NC-AL'),
        ('test-also-statewide', 'US-NC', 'US-NC-BB'),
        ('test-also-statewide', 'US-NC', 'US-NC-CC'),
        # One county, rolling up to its state, like lakeland-fl.
        ('test-one-county', 'US-NC', 'US-NC-AL'),
        # One county, rolling up to that county itself.
        ('test-county-anchor', 'US-NC-BB', 'US-NC-BB'),
        ('test-other-county', 'US-NC', 'US-NC-CC'),
    ]
    frame = pd.DataFrame(
        [(rid, rid, anchor, member) for rid, anchor, member in rows],
        columns=['region_id', 'name', 'region_admin_id', 'admin_id'],
    )

    def fake(region_id=None):
        if region_id is None:
            return frame
        found = frame[frame['region_id'] == region_id]
        if found.empty:
            raise KeyError(f'Unknown region {region_id!r}')
        return found

    monkeypatch.setattr(readers, 'get_regions', fake)
    return frame


def test_a_one_county_region_is_filed_beside_the_statewide_bundle(registry):
    recipe = _recipe(
        regions=['test-statewide', 'test-one-county'],
        whole_unit_regions=['test-statewide'],
    )
    statewide = delivery_paths(recipe, region='test-statewide')['canonical']
    county = delivery_paths(recipe, region='test-one-county')['canonical']

    assert statewide.name == 'US-NC_transaction-test-2026.parquet'
    assert statewide.parent.name == '2026'
    assert county == statewide.parent / 'test-one-county' / statewide.name
    assert county != get_output_path(recipe, 'US-NC-AL')


def test_a_region_anchored_on_its_county_misses_the_curated_output(registry):
    """Asked at level 2 for a level-3 unit, this used to raise; now it
    files the bundle in the region's folder beside the county's file."""
    recipe = _recipe(regions=['test-county-anchor'])
    canonical = delivery_paths(recipe, region='test-county-anchor')['canonical']
    curated = get_output_path(recipe, 'US-NC-BB')

    assert canonical != curated
    # The bundle is filed in the share bucket, the curated file in the
    # recipe's own; below their roots, the paths line up.
    share = Path(cfg.get_dir('share'))
    core = Path(cfg.get_dir('core'))
    expected = curated.parent / 'test-county-anchor' / curated.name
    assert canonical.relative_to(share) == expected.relative_to(core)


def test_a_region_keeps_its_path_when_a_sibling_is_declared(registry):
    """Declaring a second region on the same state moves neither."""
    alone = delivery_paths(
        _recipe(regions=['test-one-county']), region='test-one-county'
    )
    with_sibling = delivery_paths(
        _recipe(regions=['test-one-county', 'test-other-county']),
        region='test-one-county',
    )
    assert alone == with_sibling
    statewide_alone = delivery_paths(
        _recipe(regions=['test-statewide'], whole_unit_regions=['test-statewide']),
        region='test-statewide',
    )
    statewide_with_sibling = delivery_paths(
        _recipe(
            regions=['test-statewide', 'test-one-county', 'test-other-county'],
            whole_unit_regions=['test-statewide'],
        ),
        region='test-statewide',
    )
    assert statewide_alone == statewide_with_sibling


def test_a_team_twin_sits_in_its_regions_folder(registry):
    recipe = _recipe(regions=['test-one-county'], team_regions=['test-one-county'])
    public = delivery_paths(recipe, region='test-one-county')
    team = delivery_paths(recipe, region='test-one-county.team')
    for role, path in public.items():
        assert team[role] == path.parent / 'team' / path.name
    assert public['canonical'].parent.name == 'test-one-county'


def test_two_whole_unit_regions_on_one_unit_are_refused(registry):
    recipe = _recipe(
        regions=['test-statewide', 'test-also-statewide'],
        whole_unit_regions=['test-statewide', 'test-also-statewide'],
    )
    with pytest.raises(ValueError, match='one unit has one whole-unit bundle'):
        delivery_paths(recipe, region='test-statewide')


def test_an_undeclared_whole_unit_region_is_refused(registry):
    recipe = _recipe(regions=['test-one-county'], whole_unit_regions=['nowhere'])
    with pytest.raises(KeyError, match='nowhere'):
        delivery_regions(recipe)


def test_an_inline_grouping_on_a_county_would_overwrite_its_curated_file():
    recipe = _recipe(admin_ids=['US-NC-AL'], admin_id='US-NC-AL')
    # Filed in the share bucket by default, apart from the curated file.
    assert delivery_paths(recipe)['canonical'] != get_output_path(recipe, 'US-NC-AL')
    # Asked into the curated file's own bucket, it would overwrite it.
    with pytest.raises(ValueError, match='curated output of US-NC-AL'):
        delivery_paths(recipe, output_dir='core')


def test_an_inline_grouping_on_a_state_is_filed_as_the_state():
    recipe = _recipe(admin_ids=['US-NC-AL', 'US-NC-BB'])
    canonical = delivery_paths(recipe)['canonical']
    assert canonical.name == 'US-NC_transaction-test-2026.parquet'
    assert canonical.parent.name == '2026'


def test_a_whole_unit_region_missing_a_county_is_not_exported(
    registry, mock_data_root, monkeypatch
):
    """Filed at the state's path, it would read as the state's bundle."""
    monkeypatch.setattr(
        delivery_module,
        'get_admin',
        lambda *args, **kwargs: pd.DataFrame(
            index=pd.Index(['US-NC-AL', 'US-NC-BB', 'US-NC-CC', 'US-NC-DD'])
        ),
    )
    recipe = _recipe(regions=['test-statewide'], whole_unit_regions=['test-statewide'])
    with pytest.raises(ValueError, match='missing: US-NC-DD'):
        export_delivery(recipe, region='test-statewide', allow_stale=True)
