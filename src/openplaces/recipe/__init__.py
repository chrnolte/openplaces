"""
Functions to handle recipes for data ingestion and harmonization:
read and validate recipes, find recipes, build derivatives, get
output paths. One package since 2026-09-30 (until then a single
2,013-line module), one module per concern:

- loading: get_recipe, get_recipe_dict, get_recipe_by_id,
  get_recipe_id, coverage, the stacked-units layer
- tables: retention, additional-layer table recipes, output
  columns, the supplements key
- naming: recipe ids, versions, sources, provenance suffixes
- discovery: find_entity_recipe_id, find_additional_layer_recipes,
  layers and supplemented tables
- patches: patch recipes and their pipeline operations
- dependencies: DepEdge and get_recipe_dependencies
- output: output paths, admin levels, partitions

Every top-level name is re-exported here, private ones included,
because the stages and the tests import them from this package;
a test that patches `openplaces.recipe.<name>` reaches a caller
that imports the name from the package at call time.
"""

from __future__ import annotations

# Names the single module carried in its namespace and other modules
# and tests import or patch from here.
from openplaces.path import (  # noqa: F401
    OpenPlacesReference,
    path,
    recipe_path,
    recipe_roots,
)
from openplaces.recipe.dependencies import (  # noqa: F401
    _MULTI_DISCOVER_STEPS,
    _RECIPE_ID_KEY_REGEX,
    _SPINE_BUILDING_STEPS,
    DepEdge,
    _scan_ingest_recipe_ids,
    get_recipe_dependencies,
)
from openplaces.recipe.discovery import (  # noqa: F401
    _declared_entity_type,
    _recipe_yaml,
    find_additional_layer_recipes,
    find_entity_recipe_id,
    get_layers,
    get_supplemented_table,
    get_supplements_key,
)
from openplaces.recipe.loading import (  # noqa: F401
    STACKED_UNITS_LAYER_KEY,
    _add_stacked_units_layer,
    _cast_dataset,
    _cast_entity,
    coverage_is_complete,
    get_recipe,
    get_recipe_by_id,
    get_recipe_dict,
    get_recipe_id,
    raise_if_coverage_complete,
)
from openplaces.recipe.naming import (  # noqa: F401
    _VERSION_CHUNK_REGEX,
    find_admin_recipe_id,
    find_recipe_id,
    iter_entity_source_versions,
    iter_entity_sources,
    provenance_suffixes,
    resolve_attribute_name,
    source_id_from_recipe_id,
    split_provenance_suffix,
    version_sort_key,
)
from openplaces.recipe.output import (  # noqa: F401
    _year_month_range,
    get_download_admin_level,
    get_output_path,
    get_partition_ids,
    get_process_admin_level,
    get_save_admin_level,
    saves_geometry,
)
from openplaces.recipe.patches import (  # noqa: F401
    _PATCH_OPERATIONS,
    RECIPE_PATCHES_METADATA_KEY,
    _apply_patch_operation,
    apply_recipe_patches,
    find_recipe_patches,
)
from openplaces.recipe.tables import (  # noqa: F401
    _get_save_to,
    build_table_recipe,
    get_recipe_output_columns,
    get_recipe_retention,
    get_table_recipe,
)
