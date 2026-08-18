"""
Registered enrichment steps for entity attribute evidence.
"""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.enricher import EnrichState, _register
from openplaces.io.enricher.detectors.checkpoint import (
    PredictionCheckpoint,
    local_checkpoint_path,
    prune_local_checkpoints,
)
from openplaces.io.readers import get_admin_ids
from openplaces.io.scrapers.types import ImageSet
from openplaces.recipe import (
    get_output_path,
)


def _image_admin_ids(state: EnrichState, image_save_level: int) -> list[str]:
    """List the admin units whose image metadata should be read.

    Defaults to all children of the processed admin unit at the image
    save level; restricted to `state.image_admin_ids` when a sub-level
    enrichment run was requested.
    """
    if not state.image_admin_ids:
        return get_admin_ids(image_save_level, state.admin_id)
    child_ids: list[str] = []
    for admin_id in state.image_admin_ids:
        admin_id = AdminId(admin_id)
        if admin_id.get_level() < image_save_level:
            child_ids += get_admin_ids(image_save_level, admin_id)
        else:
            child_ids.append(str(admin_id))
    return child_ids


def _build_image_set(state: EnrichState, image_recipe: str | None) -> ImageSet:
    """Fetch imagery for this admin unit's spine entities, in memory.

    Imagery is fetched live for each enrichment run and never persisted.
    Google's Static API policy prohibits "pre-fetching, indexing, storing, or
    caching" of content, so there is no image ingest stage and no cache to
    reuse: the pixels exist only inside this call and the inference that
    follows it. Panorama and place ids, which the policy does permit storing,
    travel with each image as metadata so provenance survives.

    The cost of that is real: every enrichment pass re-fetches and re-pays
    for imagery, so iterating on a classifier is billed per iteration.
    """
    image_recipe_id = image_recipe or state.recipe.get('image_recipe')
    if not image_recipe_id:
        raise ValueError("Image enrichment steps require 'image_recipe'.")

    from openplaces.io.ingester.image_ingester import (
        ImageScraperError,
        fetch_images_in_memory,
    )

    try:
        return fetch_images_in_memory(
            image_recipe_id,
            state.spine,
            verbose=state.verbose,
        )
    except ImageScraperError as exception:
        # A scraper that cannot be initialized (e.g. Street View when its
        # Google Cloud project lacks billing) should skip its own step, not
        # abort a whole batch. Previously this was caught around the image
        # ingest call; the fetch now happens here, so the guard lives here.
        warnings.warn(
            f'{image_recipe_id}: {exception}. Skipping this image step; '
            'its evidence columns will be empty for this admin unit.',
            stacklevel=2,
        )
        return ImageSet()


def _step_checkpoint(
    state: EnrichState,
    column: str,
    save_every: int,
) -> PredictionCheckpoint:
    """Create a local prediction checkpoint for an image step.

    The checkpoint is keyed to the evidence output but lives in the local
    application cache so sync clients cannot lock its frequent atomic writes.
    Former output-side checkpoints are migrated automatically.
    """
    out_path = get_output_path(
        state.recipe, state.admin_id, entity_recipe_id=state.entity_recipe
    )
    legacy_path = out_path.with_name(f'{out_path.stem}_{column}_checkpoint.parquet')
    prune_local_checkpoints()
    checkpoint = PredictionCheckpoint(
        local_checkpoint_path(legacy_path),
        save_every=save_every,
        legacy_paths=[legacy_path, legacy_path.with_suffix('.tmp')],
    )
    state.metadata.setdefault('checkpoints', []).append(checkpoint)
    return checkpoint


def _add_prediction_column(
    state: EnrichState,
    predictions: dict,
    column: str,
) -> EnrichState:
    """Add prediction evidence to state, aligned to the spine index."""
    pred_series = pd.Series(predictions, name=column)
    state.evidence[column] = pred_series.reindex(state.evidence.index)
    # Record which entities were attempted (even when the result is
    # missing), so partial-run outputs merge correctly with existing
    # evidence files.
    state.metadata.setdefault('attempted_keys', set()).update(predictions)
    return state


@_register('classify_roof_shape')
def classify_roof_shape(
    state: EnrichState,
    image_recipe: str | None = None,
    column: str = 'roof_shape_brails',
    model_path: str | None = None,
    **kwargs,
) -> EnrichState:
    """Predict roof-shape evidence from satellite imagery."""
    from openplaces.io.enricher.detectors.roof_shape import (
        RoofShapeClassifier,
    )

    image_set = _build_image_set(state, image_recipe)
    if not image_set.images:
        state.evidence[column] = pd.NA
        return state

    checkpoint = _step_checkpoint(state, column, save_every=500)
    try:
        predictions = RoofShapeClassifier(model_path=model_path).predict(
            image_set, verbose=state.verbose, checkpoint=checkpoint, **kwargs
        )
    finally:
        checkpoint.flush()
    return _add_prediction_column(state, predictions, column)


@_register('classify_occupancy')
def classify_occupancy(
    state: EnrichState,
    image_recipe: str | None = None,
    column: str = 'occupancy_brails',
    model_path: str | None = None,
    **kwargs,
) -> EnrichState:
    """Predict occupancy evidence from street-level imagery."""
    from openplaces.io.enricher.detectors.occupancy import (
        OccupancyClassifier,
    )

    image_set = _build_image_set(state, image_recipe)
    if not image_set.images:
        state.evidence[column] = pd.NA
        return state

    checkpoint = _step_checkpoint(state, column, save_every=500)
    try:
        predictions = OccupancyClassifier(model_path=model_path).predict(
            image_set, verbose=state.verbose, checkpoint=checkpoint, **kwargs
        )
    finally:
        checkpoint.flush()
    return _add_prediction_column(state, predictions, column)


@_register('detect_n_stories', 'detect_nfloors')
def detect_n_stories(
    state: EnrichState,
    image_recipe: str | None = None,
    column: str = 'n_stories_brails',
    model_path: str | None = None,
    **kwargs,
) -> EnrichState:
    """Predict story-count evidence from street-level imagery."""
    from openplaces.io.enricher.detectors.n_stories import NStoriesDetector

    image_set = _build_image_set(state, image_recipe)
    if not image_set.images:
        state.evidence[column] = pd.NA
        return state

    checkpoint = _step_checkpoint(state, column, save_every=50)
    try:
        predictions = NStoriesDetector(model_path=model_path).predict(
            image_set, checkpoint=checkpoint
        )
    finally:
        checkpoint.flush()
    return _add_prediction_column(state, predictions, column)
