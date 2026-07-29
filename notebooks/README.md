# Jupyter notebook style guide

All Jupyter notebooks in `openplaces` (`notebooks/`) follow a standardized structure.

This structure enables the `openplaces.flow.convert_to_script` tool to reliably parse and translate interactive notebooks into production command-line scripts located in `scripts/`.

---

## The prototyping-to-production workflow

1. **Write the notebook**: Create and develop the notebook interactively under the `notebooks/` directory.
2. **Convert to script**: Execute the cell containing `convert_to_script(commit=True)`. This automatically generates a corresponding Python script in `scripts/` (e.g., `notebooks/02_ingest/parcels/ingest_parcels.ipynb` converts to `scripts/02_ingest/parcels/ingest_parcels.py`).
3. **Verify compliance**: The converter automatically runs `ruff` on the output. You must resolve any linting/formatting errors.
4. **Test the script**: Run `test_script()` to ensure the generated Python script runs successfully in a clean subprocess environment.

---

## Notebook anatomy and required headers

A compliant notebook contains exactly the following structure of markdown and code cells.

### 1. Title block (first cell)
- **Cell type**: Markdown
- **Format**:
  ```markdown
  **`notebook_name`**

  Short description of what this notebook does.
  ```
- **Rule**: Do **NOT** use a markdown heading (like `# Title`) for this block. This is a deliberate choice to keep the notebook environment's table of contents (TOC) flat.
- **Script effect**: During conversion, this title block becomes the module-level docstring `""" ... """` at the top of the generated Python script.

### 2. Configure header
- **Cell type**: Markdown
- **Heading**: `# Configure`
- **Code cells**:
  1. Standard and library imports.
  2. The parser definition block using `argparse.ArgumentParser`.
- **Script effect**: Retained in the Python script.

### 3. Test arguments header
- **Cell type**: Markdown
- **Heading**: `# Test arguments`
- **Code cell**:
  ```python
  ARGS_TEST = (
      "--recipe_id US-NC-NE_parcel-nhcgov-2026 "
      "--admin_ids US-NC-NE "
      "--reprocess "
      "--verbose "
  )

  # Convert argument string to list of strings
  args_list = [x for x in ARGS_TEST.split(" ") if x != ""]

  # Parse list of arguments
  args = parser.parse_args(args_list)

  # Display parsed arguments
  args
  ```
- **Rule**: Keep `ARGS_TEST` as a space-separated string (often split across lines for readability) representing the command-line options you want to run interactively.
- **Script effect**: **CRITICAL:** The script converter replaces **EVERYTHING** between `# Test arguments` and the next top-level heading (`# `) with `args = parser.parse_args()`. Do not place any production logic here. Debug-only cells (like showing recipe parameters using `pretty_print`) are allowed here since they will be stripped from the final script.

### 4. Run / processing header (the third top-level heading)
- **Cell type**: Markdown
- **Heading**: Must be a top-level heading describing the operation (e.g., `# Ingest parcel data`, `# Harmonize footprints`, or `# Run`).
- **Code cells**:
  Instantiate the processing runner (e.g., `Ingester`, `Harmonizer`, `Curator`) and execute it.
- **Script effect**: Retained in the Python script.

### 5. Convert to script header
- **Cell type**: Markdown
- **Content**:
  ```markdown
  ---
  # Convert to script

  *The above line and heading identify the end of the script.*
  ```
- **Code cell**:
  ```python
  from openplaces.flow import convert_to_script

  COMMIT = True

  convert_to_script(commit=COMMIT)
  ```
- **Script effect**: **CRITICAL:** Everything below the `---` horizontal rule and `# Convert to script` markdown cell is completely stripped and excluded from the production Python script.

### 6. Test script header
- **Cell type**: Markdown
- **Heading**: `# Test script`
- **Code cell**:
  ```python
  # from openplaces.flow import test_script

  # test_script(*args_list, committed=COMMIT)
  ```
- **Purpose**: Verifies that the newly generated script runs correctly as a standalone Python process with the test arguments (commented out by default to avoid slow execution on every run).

### 7. Inspect outputs / results header
- **Cell type**: Markdown
- **Heading**: `# Inspect outputs` (or `# Inspect results`)
- **Visual & tabular verification**:
  Include validation helpers to check the generated outputs. Keep these simple and clean:
  - **Tabular inspect**: Transpose a sample of the data to easily read columns:
    ```python
    from openplaces import get_entities

    df = get_entities(args.recipe_id, args.admin_ids, geom=True)
    print(len(df))
    df.sample(5).T
    ```
  - **Visual/map inspect (spatial entities)**:
    ```python
    # Show parcel maps
    ingester.show_ingested_geometries(fill=False)
    ingester.show_random_entity()
    ```
    Or for curated building datasets:
    ```python
    from openplaces.viz import show_building_imagery

    show_building_imagery(
        location=building,
        geodatasets={"footprints": footprints},
        image_recipes=image_recipes,
        admin_id=args.admin_ids[0],
    )
    ```

---

## Full notebook template example

You can use the structure below as a starting template for any new processing pipeline notebook:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**`process_theme`**\n",
    "\n",
    "Description of the notebook pipeline."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Configure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import argparse\n",
    "from openplaces.io.processor import Processor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "parser = argparse.ArgumentParser(description='Process data using a recipe')\n",
    "parser.add_argument('--recipe_id', help='Recipe ID')\n",
    "parser.add_argument('--admin_ids', nargs='*')\n",
    "parser.add_argument('--reprocess', action='store_true')\n",
    "parser.add_argument('--verbose', action='store_true')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Test arguments"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ARGS_TEST = (\n",
    "    '--recipe_id US_theme-source-2026 '\n",
    "    '--admin_ids US-NC-BS '\n",
    "    '--reprocess '\n",
    "    '--verbose '\n",
    ")\n",
    "args_list = [x for x in ARGS_TEST.split(' ') if x]\n",
    "args = parser.parse_args(args_list)\n",
    "args"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Process data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "processor = Processor(args.recipe_id, args.admin_ids, verbose=args.verbose)\n",
    "processor.process(reprocess=args.reprocess)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# Convert to script\n",
    "\n",
    "*The above line and heading identify the end of the script.*\n",
    "\n",
    "*Code below this marker will not be included in the converted `.py` script.*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from openplaces.flow import convert_to_script\n",
    "COMMIT = True\n",
    "convert_to_script(commit=COMMIT)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Test script"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# from openplaces.flow import test_script\n",
    "\n",
    "# test_script(*args_list, committed=COMMIT)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Inspect results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from openplaces import get_entities\n",
    "df = get_entities(args.recipe_id, args.admin_ids[0], geom=True)\n",
    "print(len(df))\n",
    "df.sample(5).T"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
```
