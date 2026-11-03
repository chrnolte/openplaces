"""Job groups: per-family names, and bundling on the SCC profile only."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import openplaces
from openplaces.core.constants import RECIPE_STAGES
from openplaces.flow.dag import StageNode, group_name

PROFILES = Path(__file__).resolve().parents[2] / 'workflow' / 'profiles'
RECIPES = Path(openplaces.__file__).resolve().parent / 'recipes'


def _profile(name):
    return yaml.safe_load((PROFILES / name / 'config.yaml').read_text('utf-8'))


def test_group_name_is_one_per_family():
    a = StageNode('harmonize', 'US_parcel-spine-2026', 'US-NC-CM')
    b = StageNode('harmonize', 'US_parcel-spine-2026', 'US-MA-SOM')
    assert group_name(a) == group_name(b) == 'harmonize_US_parcel-spine-2026'
    other_stage = StageNode('curate', 'US_parcel-spine-2026', 'US-NC-CM')
    other_recipe = StageNode('harmonize', 'US_building-spine-2026', 'US-NC-CM')
    assert len({group_name(n) for n in (a, other_stage, other_recipe)}) == 3


@pytest.mark.parametrize(
    'node',
    [
        StageNode('deliver', 'US_footprint-cheer-2026', 'US-NC', 'cheer-nc'),
        StageNode('validate', 'US_footprint-cheer-2026', 'US-NC', 'cheer-nc'),
        StageNode('ingest', 'admin-gadm-4.1', None),
    ],
)
def test_jobs_without_a_unit_are_never_grouped(node):
    assert group_name(node) is None


def test_only_the_scc_profile_bundles():
    scc, local = _profile('scc'), _profile('local')
    assert scc['group-components']
    assert all(isinstance(n, int) and n >= 1 for n in scc['group-components'].values())
    # One core per group job: Snakemake then sizes a group by its
    # largest job, not the sum (see the next test).
    assert scc['cores'] == 1
    assert '{rule}' in scc['cluster-generic-submit-cmd']
    assert 'group-components' not in local
    assert 'groups' not in local


def test_scc_jobscript_names_stay_short():
    # Snakemake's default jobscript name holds {name}, which for a group
    # joins up to five per-unit rule names and passed the 255-character
    # file name limit on the SCC (11-03). {rulename} is GROUP there.
    jobname = _profile('scc')['jobname']
    assert '{rulename}' in jobname and '{jobid}' in jobname
    assert '{name}' not in jobname


def test_scc_groups_name_existing_families():
    # A misspelt key bundles nothing, silently: pin each to a recipe.
    for key in _profile('scc')['group-components']:
        stage, _, recipe_id = key.partition('_')
        assert stage in RECIPE_STAGES, key
        assert next(RECIPES.rglob(f'{recipe_id}.yaml'), None), key


def _group_resources(cores):
    from snakemake.resources import GroupResources, Resources

    def job(mem_mb):
        resources = {'_cores': 1, '_nodes': 1, 'mem_mb': mem_mb, 'runtime': 240}
        return SimpleNamespace(resources=resources, pipe_group=None)

    return GroupResources.basic_layered(
        toposorted_jobs=[[job(1000), job(3000), job(2000)]],
        constraints=Resources.from_mapping({'_cores': cores, '_nodes': 200}),
        run_local=False,
        additive_resources=['runtime'],
        sortby=['runtime'],
    )


def test_snakemake_sizes_a_one_core_group_by_its_largest_job():
    # Pins the installed Snakemake's behavior the SCC profile relies on
    # (GroupResources.basic_layered, checked with 9.23.1).
    group = _group_resources(cores=1)
    assert (group['mem_mb'], group['_cores'], group['runtime']) == (3000, 1, 720)
    # Without --cores the jobs form one layer and their requests add up.
    group = _group_resources(cores=None)
    assert (group['mem_mb'], group['_cores']) == (6000, 3)
