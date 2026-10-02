"""Patch recipes: find_recipe_patches, apply_recipe_patches and
the pipeline operations.
"""

# Looked up through the package at call time, not bound here: the tests
# patch these three names on `openplaces.recipe`, and a function that
# had imported them would keep the originals.
from openplaces import recipe as _recipe
from openplaces.core.schema import (
    AdminId,
)
from openplaces.recipe.loading import (
    get_recipe_id,
)

RECIPE_PATCHES_METADATA_KEY = 'openplaces:recipe_patches'

#: The operations a patch recipe's `pipeline_patch` may list, each keyed
#: by the step it addresses. `step` names the step; `occurrence` (1-based)
#: picks one when the name repeats in the pipeline.
_PATCH_OPERATIONS = (
    'set',
    'extend',
    'insert_before',
    'insert_after',
    'replace',
    'remove',
)


def find_recipe_patches(recipe_id: str, admin_id) -> list[str]:
    """Patch recipes that amend *recipe_id* for *admin_id*, coarse to fine.

    A patch recipe is a recipe file at a finer scope than the recipe it
    names in ``patches:``; it carries a ``pipeline_patch`` and no
    pipeline of its own. This lists the ones whose admin scope covers
    *admin_id*, ordered so that a state's patch applies before a
    county's and the county's last word wins.

    Parameters
    ----------
    recipe_id : str
        The recipe being run (a national curate or harmonize recipe).
    admin_id : str or AdminId
        The unit being processed.
    """
    from openplaces.diagnostics import _recipe_index

    admin_id = AdminId(admin_id) if not isinstance(admin_id, AdminId) else admin_id
    index = _recipe_index()
    if index.empty or 'patches' not in index.columns:
        return []
    rows = index[index['patches'] == recipe_id]
    found = []
    for _, row in rows.iterrows():
        scope = AdminId(row['admin_id']) if row['admin_id'] else AdminId(None)
        if scope.is_parent_or_equal_of(admin_id):
            found.append((scope.get_level(), row['recipe_id']))
    return [rid for _, rid in sorted(found)]


def apply_recipe_patches(recipe: dict, admin_id, patches=None) -> tuple[dict, list]:
    """The recipe as *admin_id* runs it: its pipeline with every patch applied.

    Why patches, and not a county recipe of its own: a county recipe would
    carry its own id, while the output path is built from the admin id
    and the entity, so the two would write the same file and the graph
    would schedule both. A patch keeps one recipe id in the graph, one
    output per unit, and records in the output's footer which patches
    shaped it (``RECIPE_PATCHES_METADATA_KEY``). The national recipe
    stays readable as the generic pipeline and a county's rules are
    found by the county's id
    (plans/core-schema-and-stage-contracts-review.md, section 4).

    Parameters
    ----------
    recipe : dict
        A loaded recipe with a ``pipeline``.
    admin_id : str or AdminId
        The unit being processed; decides which patches apply.
    patches : list of dict, optional
        Patch recipes to apply, in order, in place of the discovered
        ones (tests pass fabricated patches here).

    Returns
    -------
    tuple
        The patched recipe (a deep copy; the input is not touched) and
        the ids of the patches applied, empty when none applied. A
        recipe without a pipeline is returned as is.

    Raises
    ------
    ValueError
        On a patch whose stage or entity type differs from the recipe's,
        an unknown operation, a step name the pipeline does not hold (or
        holds more than once without ``occurrence``), or ``extend`` on a
        parameter that is not a list.
    """
    import copy

    if not recipe.get('pipeline'):
        return recipe, []
    recipe_id = recipe.get('recipe_id')
    if recipe_id is None:
        try:
            recipe_id = get_recipe_id(recipe)
        except ValueError:
            # A bare recipe dict with neither entity nor dataset (the
            # curator's unit tests build these): nothing can name it in
            # `patches:`, so nothing applies.
            return recipe, []
    if patches is None:
        patches = [
            _recipe.get_recipe_by_id(pid)
            for pid in find_recipe_patches(recipe_id, admin_id)
        ]
    if not patches:
        return recipe, []

    patched = copy.deepcopy(recipe)
    pipeline = patched['pipeline']
    applied = []
    for patch in patches:
        patch_id = patch.get('recipe_id', '?')
        if patch.get('stage') != recipe.get('stage'):
            raise ValueError(
                f'{patch_id} (stage {patch.get("stage")!r}) cannot patch '
                f'{recipe_id} (stage {recipe.get("stage")!r}).'
            )
        own = getattr(patch.get('entity'), 'entity_type', None)
        target = getattr(recipe.get('entity'), 'entity_type', None)
        if str(own) != str(target):
            raise ValueError(
                f'{patch_id} ({own}) cannot patch {recipe_id} ({target}): the '
                'entity types differ.'
            )
        if patch.get('pipeline'):
            raise ValueError(
                f'{patch_id} declares both patches: and a pipeline; a patch recipe '
                'amends the pipeline it names and has none of its own.'
            )
        for operation in patch.get('pipeline_patch') or []:
            pipeline = _apply_patch_operation(pipeline, operation, patch_id)
        applied.append(patch_id)
    patched['pipeline'] = pipeline
    return patched, applied


def _apply_patch_operation(pipeline: list, operation: dict, patch_id: str) -> list:
    """One operation of a `pipeline_patch`, applied to a copy of *pipeline*."""
    kinds = [k for k in _PATCH_OPERATIONS if k in operation]
    if len(kinds) != 1:
        raise ValueError(
            f'{patch_id}: a pipeline_patch entry names exactly one of '
            f'{", ".join(_PATCH_OPERATIONS)}, got {sorted(operation)}.'
        )
    kind = kinds[0]
    name = operation[kind]
    positions = [i for i, s in enumerate(pipeline) if s.get('step') == name]
    if not positions:
        raise ValueError(f'{patch_id}: no step {name!r} to {kind} in the pipeline.')
    occurrence = operation.get('occurrence')
    if len(positions) > 1 and occurrence is None:
        raise ValueError(
            f'{patch_id}: {name!r} occurs {len(positions)} times; say which with '
            'occurrence: <1-based index>.'
        )
    index = positions[(occurrence or 1) - 1]
    pipeline = list(pipeline)
    if kind == 'remove':
        del pipeline[index]
    elif kind == 'replace':
        pipeline[index] = dict(operation['step'])
    elif kind == 'insert_before':
        pipeline.insert(index, dict(operation['step']))
    elif kind == 'insert_after':
        pipeline.insert(index + 1, dict(operation['step']))
    elif kind == 'set':
        step = dict(pipeline[index])
        step.update(operation.get('params') or {})
        pipeline[index] = step
    elif kind == 'extend':
        step = dict(pipeline[index])
        key = operation['key']
        current = step.get(key)
        if current is None:
            current = []
        if not isinstance(current, list):
            raise ValueError(
                f'{patch_id}: extend needs a list under {name!r}.{key}, '
                f'found {type(current).__name__}.'
            )
        step[key] = [*current, *(operation.get('values') or [])]
        pipeline[index] = step
    return pipeline
