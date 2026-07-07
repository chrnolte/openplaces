"""
Functions to manage the flow of work from prototyping to computing.

- scripts: notebook-to-script conversion and subprocess helpers
- dag: RecipeDAG, the recipe dependency graph behind orchestration
- run_stage: CLI dispatching one (stage, recipe, admin unit) job
- submit: cluster submission wrapper with the inspection gate
"""

from openplaces.flow.scripts import (
    convert_to_script,
    get_caller_path,
    get_caller_path_in_code_directory,
    run_subprocess,
    test_script,
)

__all__ = [
    'convert_to_script',
    'get_caller_path',
    'get_caller_path_in_code_directory',
    'run_subprocess',
    'test_script',
    'RecipeDAG',
]


def __getattr__(name: str):
    # Lazy: RecipeDAG pulls the recipe machinery only when orchestrating
    if name == 'RecipeDAG':
        from openplaces.flow.dag import RecipeDAG

        return RecipeDAG
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
