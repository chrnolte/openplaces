#!/usr/bin/env python3
"""Development environment management for openplaces.

This script helps developers set up and manage their local development
environment. End users installing via 'pip install openplaces' don't need this.
"""
import shutil
import subprocess
import sys


# Detect which package manager to use
def get_package_manager():
    """Detect if mamba is available, otherwise use conda."""
    if shutil.which("mamba"):
        return "mamba"
    elif shutil.which("conda"):
        return "conda"
    else:
        print("✗ Error: Neither mamba nor conda found in PATH")
        print("\nPlease install one of the following:")
        print(
            "  - Miniforge (includes mamba): https://github.com/conda-forge/miniforge"
        )
        print("  - Miniconda (conda): https://docs.conda.io/en/latest/miniconda.html")
        sys.exit(1)


PKG_MGR = get_package_manager()
DEFAULT_ENV_NAME = "openplaces"


def get_env_name():
    """Prompt user for environment name."""
    env_name = input(
        f"Environment name (press Enter for '{DEFAULT_ENV_NAME}'): "
    ).strip()
    return env_name if env_name else DEFAULT_ENV_NAME


def run(cmd, check=True):
    """Run command and handle errors."""
    print(f"→ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"✗ Command failed with code {result.returncode}")
        sys.exit(1)
    return result.returncode == 0


def setup():
    """Create conda environment and install package in editable mode."""
    print("This will create a development environment for `openplaces`.\n")
    print(f"Found package manager: {PKG_MGR}.\n")

    env_name = get_env_name()

    print(f"\nUsing package manager: {PKG_MGR}")
    print(f"Creating environment: {env_name}")

    # Create environment with specified name
    print("\nCreating conda environment from environment.yml...")
    if not run(f"{PKG_MGR} env create -f environment.yml -n {env_name} -y"):
        print("✗ Failed to create environment")
        return

    print("\nInstalling openplaces in editable mode...")
    run(f"{PKG_MGR} run -n {env_name} pip install -e . --no-deps")

    print("\nSetting up nbstripout for automatic notebook cleaning...")
    run(f"{PKG_MGR} run -n {env_name} nbstripout --install")

    print("\n✓ Development environment ready!")
    print("\nNext steps:")
    print(f"  1. {PKG_MGR} activate {env_name}")
    print("  2. cd notebooks")
    print("  3. jupyter notebook")
    print("  4. Start coding!")


def update():
    """Update existing environment with latest dependencies."""
    env_name = get_env_name()

    print(f"\nUsing package manager: {PKG_MGR}")
    print(f"Updating environment: {env_name}")

    print("\nUpdating conda environment...")
    run(f"{PKG_MGR} env update -f environment.yml -n {env_name} --prune")

    print("\nReinstalling openplaces...")
    run(f"{PKG_MGR} run -n {env_name} pip install -e . --no-deps")

    print("\nEnsuring nbstripout is configured...")
    run(f"{PKG_MGR} run -n {env_name} nbstripout --install")

    print("\n✓ Environment updated!")


def clean():
    """Remove the development environment."""
    env_name = get_env_name()

    response = input(f"Remove {env_name} environment? [y/N] ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    print(f"\nRemoving environment: {env_name}")
    print("(This may take a minute...)")

    # Capture output to filter the mamba_trash.txt error
    result = subprocess.run(
        f"{PKG_MGR} env remove -n {env_name} -y",
        shell=True,
        capture_output=True,
        text=True,
    )

    # Print stdout (the actual removal progress)
    if result.stdout:
        print(result.stdout)

    # Print stderr but filter out the mamba_trash.txt error
    if result.stderr:
        # Filter out the known harmless error
        lines = result.stderr.split('\n')
        filtered_lines = [
            line
            for line in lines
            if 'mamba_trash.txt' not in line.lower()
            and 'error opening for writing' not in line.lower()
        ]
        if filtered_lines:
            print('\n'.join(filtered_lines))

    # Check if removal was successful
    if (
        result.returncode == 0
        or "Environment removed" in result.stdout
        or "Environment removed" in result.stderr
    ):
        print("\n✓ Environment removed!")
    else:
        print(f"\n✗ Failed to remove environment (exit code: {result.returncode})")


def test():
    """Run the test suite."""
    env_name = get_env_name()

    print(f"\nRunning tests in environment: {env_name}")
    run(f"{PKG_MGR} run -n {env_name} pytest")


def format_code():
    """Format code with ruff."""
    env_name = get_env_name()

    print(f"\nFormatting code in environment: {env_name}")
    print("Formatting Python files...")
    run(f"{PKG_MGR} run -n {env_name} ruff format openplaces/")

    print("Formatting notebooks (if any)...")
    run(f"{PKG_MGR} run -n {env_name} ruff format notebooks/", check=False)

    print("Fixing linting issues...")
    run(f"{PKG_MGR} run -n {env_name} ruff check --fix openplaces/")
    print("\n✓ Code formatted!")


def lint():
    """Check code with ruff."""
    env_name = get_env_name()

    print(f"\nLinting code in environment: {env_name}")
    run(f"{PKG_MGR} run -n {env_name} ruff check openplaces/")


def build():
    """Build package for distribution."""
    env_name = get_env_name()

    print(f"\nBuilding package in environment: {env_name}")
    run(f"{PKG_MGR} run -n {env_name} python -m build")
    print("\n✓ Package built! Check dist/ directory")


def list_envs():
    """List all conda environments."""
    print(f"\nListing all {PKG_MGR} environments:")
    run(f"{PKG_MGR} env list")


def main():
    """Main entry point."""
    commands = {
        "setup": ("Create development environment", setup),
        "update": ("Update development environment", update),
        "clean": ("Remove development environment", clean),
        "test": ("Run tests", test),
        "format": ("Format code with ruff", format_code),
        "lint": ("Check code with ruff", lint),
        "build": ("Build package for distribution", build),
        "list": ("List all conda/mamba environments", list_envs),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Development environment manager (using {PKG_MGR})")
        print("\nUsage: python dev.py <command>")
        print("\nDevelopment commands:")
        for cmd, (desc, _) in commands.items():
            print(f"  {cmd:8} - {desc}")
        sys.exit(1)

    commands[sys.argv[1]][1]()


if __name__ == "__main__":
    main()
