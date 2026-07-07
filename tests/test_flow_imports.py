"""The notebook import contract of the flow package must keep working."""


def test_notebook_imports():
    from openplaces.flow import convert_to_script, test_script  # noqa: F401


def test_run_subprocess_import():
    from openplaces.flow import run_subprocess  # noqa: F401


def test_caller_path_helpers_import():
    from openplaces.flow import (  # noqa: F401
        get_caller_path,
        get_caller_path_in_code_directory,
    )
