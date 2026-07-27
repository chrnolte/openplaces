"""
Shared image classification inference for pretrained detector models.
"""

from __future__ import annotations

import itertools
import os
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from openplaces.io.enricher.detectors.checkpoint import PredictionCheckpoint
from openplaces.io.enricher.detectors.device import get_device
from openplaces.io.scrapers.types import ImageSet

_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
_IMAGE_SIZE = 384
_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)


@cache
def _load_model(model_path: str, device):
    """Deserialize and prepare a classifier model, cached per (path, device).

    Avoids re-running `torch.load` and the device transfer on every
    `predict_classes` call — the same model is reused across every admin
    unit processed in one pipeline run.
    """
    import torch

    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.to(device)
    model.eval()
    return model


def _prepare_image(image: Image.Image, torch: Any) -> Any:
    """Resize and normalize an image for classifier inference."""
    width, height = image.size
    if width < height:
        resized_size = (_IMAGE_SIZE, int(_IMAGE_SIZE * height / width))
    else:
        resized_size = (int(_IMAGE_SIZE * width / height), _IMAGE_SIZE)
    image = image.resize(resized_size, Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.float32, copy=True).transpose(2, 0, 1)
    tensor = torch.from_numpy(array)
    tensor = tensor.div(255)
    mean = tensor.new_tensor(_IMAGE_MEAN).view(-1, 1, 1)
    std = tensor.new_tensor(_IMAGE_STD).view(-1, 1, 1)
    return tensor.sub(mean).div(std)


def predict_classes(
    images: ImageSet,
    model_path: str | Path,
    classes: list[str],
    batch_size: int = 32,
    device: str | None = None,
    verbose: bool = True,
    checkpoint: PredictionCheckpoint | None = None,
) -> dict[Any, str | None]:
    """Predict one class label for each image in an ImageSet.

    Parameters
    ----------
    images
        Image collection with `dir_path` and `images` entries.
    model_path
        Path to a serialized full PyTorch model checkpoint.
    classes
        Class labels in the order used by the model output layer.
    batch_size
        Number of images per forward pass. Within a batch, tensors are
        grouped by shape (the aspect-preserving resize can yield varying
        shapes), so per-image preprocessing is identical to unbatched
        inference.
    device
        Device override (e.g. 'cpu'); defaults to the best available
        device from get_device.
    verbose
        Print the image count and show a progress bar.
    checkpoint
        When given, previously checkpointed predictions are reused,
        new ones are persisted periodically, and an interrupted run
        resumes where it left off.

    Returns
    -------
    dict
        Mapping from image keys to predicted class labels. Missing,
        unsupported, or unreadable image files are mapped to `None`.
    """
    import torch

    device = get_device(device)
    model = _load_model(str(model_path), device)

    predictions: dict[Any, str | None] = {}
    if checkpoint is not None:
        predictions = checkpoint.load()
    items = [
        (key, image) for key, image in images.images.items() if key not in predictions
    ]
    n_total = len(images.images)
    n_done = n_total - len(items)
    if verbose:
        print(f'Classifying {n_total:,} images on {device} ...')
        if n_done:
            print(f'Resuming: {n_done:,} of {n_total:,} already classified')

    def _record(key, value):
        predictions[key] = value
        if checkpoint is not None:
            checkpoint.add(key, value)

    def _load(item):
        """Read and prepare one image off the inference thread."""
        key, image = item
        image_path = Path(images.dir_path) / image.filename
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in _IMAGE_EXTENSIONS
        ):
            return key, None, False
        try:
            with Image.open(image_path) as image_file:
                return key, _prepare_image(image_file.convert('RGB'), torch), False
        except Exception:
            return key, None, True

    n_unreadable = 0
    progress = tqdm(total=n_total, initial=n_done, disable=not verbose)
    # Image decode dominates runtime (~50 ms/image vs ~1 ms/image GPU
    # forward), so loading runs in a thread pool with a bounded prefetch
    # window that keeps at most a few batches of tensors in memory.
    pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1))
    try:
        item_iter = iter(items)
        futures = deque(
            pool.submit(_load, item)
            for item in itertools.islice(item_iter, batch_size * 4)
        )
        by_shape: dict[tuple, list] = {}
        n_in_batch = 0

        def _infer_batch():
            nonlocal by_shape, n_in_batch
            with torch.no_grad():
                for group in by_shape.values():
                    stacked = torch.stack([tensor for _, tensor in group])
                    outputs = model(stacked.to(device))
                    predicted = torch.argmax(outputs.data, 1).cpu()
                    for (key, _), class_index in zip(group, predicted.tolist()):
                        _record(key, classes[class_index])
            progress.update(n_in_batch)
            by_shape = {}
            n_in_batch = 0

        while futures:
            key, tensor, unreadable = futures.popleft().result()
            next_item = next(item_iter, None)
            if next_item is not None:
                futures.append(pool.submit(_load, next_item))
            if tensor is None:
                _record(key, None)
                n_unreadable += unreadable
                progress.update(1)
            else:
                by_shape.setdefault(tuple(tensor.shape), []).append((key, tensor))
                n_in_batch += 1
            if n_in_batch >= batch_size:
                _infer_batch()
        if n_in_batch:
            _infer_batch()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        progress.close()
        if checkpoint is not None:
            checkpoint.flush()

    if n_unreadable:
        warnings.warn(f'{n_unreadable} image file(s) could not be read.')
    return predictions
