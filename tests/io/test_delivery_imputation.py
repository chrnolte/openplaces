"""Which delivery regions get their primary columns imputed."""

import subprocess

import pytest

from openplaces.io.delivery import _code_revision, imputation_spec
from openplaces.io.delivery.terms import bundle_terms, format_notice

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


@pytest.fixture
def no_git(monkeypatch):
    """Run as from a `git archive`: every git call fails."""

    def fail(command, **kwargs):
        return subprocess.CompletedProcess(command, 128, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', fail)


@pytest.mark.parametrize(
    ('stated', 'expected'),
    [
        ('main@2e55b58c', {'branch': 'main', 'commit': '2e55b58c', 'dirty': False}),
        ('2e55b58c', {'branch': None, 'commit': '2e55b58c', 'dirty': False}),
        ('', None),
    ],
)
def test_an_archive_run_takes_its_revision_from_the_job(
    no_git, monkeypatch, stated, expected
):
    monkeypatch.setenv('OPENPLACES_CODE_REVISION', stated)
    assert _code_revision() == expected


def test_the_notice_names_a_commit_without_a_branch():
    terms = bundle_terms('US_footprint-cheer-2026', None)
    terms['imputation'] = {
        'version': 'm',
        'common_year': 2025,
        'filled': {},
        'code': {'branch': None, 'commit': '2e55b58c', 'dirty': False},
    }
    notice = format_notice('US_footprint-cheer-2026', terms)
    assert 'Model code: openplaces commit 2e55b58c.' in notice
