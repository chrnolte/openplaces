"""
Story-count detector using the BRAILS++ EfficientDet-D4 model.

Ports the BRAILS++ floor-detector runtime into openplaces,
replacing cv2 with PIL and removing the ``brails`` package import.  All
post-processing (threshold tuning, nested-box removal, stack grouping,
middle-region selection) is reproduced verbatim from the BRAILS++ source.

The EfficientDet inference engine is self-contained in
`efficientdet_lib.infer.Infer`. Model weights are downloaded from Zenodo on
first use and cached under `cfg.models_dir / 'external' / 'brails'`.
"""

from __future__ import annotations

import os
import time
import warnings
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image as PILImage
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.io.enricher.detectors.checkpoint import PredictionCheckpoint
from openplaces.io.enricher.detectors.device import get_device
from openplaces.io.enricher.models import get_model
from openplaces.io.scrapers.types import ImageSet

warnings.filterwarnings('ignore')

_MODEL_URL = 'https://zenodo.org/record/4421613/files/efficientdet-d4_trained.pth'
_MODEL_FILENAME = 'efficientdet-d4_nfloorDetector.pth'


def _legacy_model_path() -> Path | None:
    """Return an existing checkpoint from either former cache location."""
    candidates = [
        Path(cfg.dir_external) / AdminId('US').to_path() / 'models' / _MODEL_FILENAME,
        Path(cfg.dir_external) / 'models' / _MODEL_FILENAME,
    ]
    return next((path for path in candidates if path.exists()), None)


@cache
def _load_infer(model_path: str, use_gpu: bool):
    """Load the EfficientDet inference engine, cached per (path, use_gpu).

    Avoids re-running `load_model` (weight deserialization + device
    transfer) on every `NStoriesDetector.predict` call — the same engine
    is reused across every admin unit processed in one pipeline run.
    """
    from openplaces.io.enricher.detectors.efficientdet_lib.infer import Infer

    gtf_infer = Infer()
    gtf_infer.load_model(model_path, ['floor'], use_gpu=use_gpu)
    return gtf_infer


class NStoriesDetector:
    """Detect the number of stories in buildings from street-view images.

    Uses the BRAILS++ EfficientDet-D4 model (downloaded from Zenodo on first
    use) plus the original BRAILS++ post-processing pipeline. The inference
    engine is bundled with openplaces.

    Parameters
    ----------
    model_path
        Path to the pre-trained `.pth` weights file. When `None`, the
        pretrained model is cached under
        `cfg.models_dir / 'external' / 'brails'`. Existing files in the two
        previous external-data locations are still used.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model_path = Path(model_path) if model_path else None

    def predict(
        self,
        images: ImageSet,
        checkpoint: PredictionCheckpoint | None = None,
    ) -> dict:
        """Predict the number of stories for each image in *images*.

        Parameters
        ----------
        images
            Collection of street-view images to analyse.
        checkpoint
            When given, previously checkpointed predictions are reused,
            new ones are persisted periodically, and an interrupted run
            resumes where it left off.

        Returns
        -------
        dict
            Mapping from the same keys as ``images.images`` to an integer
            story count, or ``None`` when the image file does not exist.
        """
        model_path = get_model(
            _MODEL_URL,
            _MODEL_FILENAME,
            self._model_path or _legacy_model_path(),
            label='story-count detector model',
        )

        # EfficientDet stays on CUDA or CPU; DirectML op coverage is too
        # incomplete for detection models.
        gpu_enabled = get_device().type == 'cuda'

        image_list = [
            os.path.join(images.dir_path, image.filename)
            for image in images.images.values()
        ]
        image_keys = list(images.images.keys())

        # --- helpers (ported verbatim from BRAILS++) ---

        def create_polygon(bounding_box: list) -> Polygon:
            return Polygon(
                [
                    (bounding_box[0], bounding_box[1]),
                    (bounding_box[0], bounding_box[3]),
                    (bounding_box[2], bounding_box[3]),
                    (bounding_box[2], bounding_box[1]),
                ]
            )

        def intersect_polygons(poly1: Polygon, poly2: Polygon) -> float:
            if poly1.intersects(poly2):
                poly_area = poly1.intersection(poly2).area
                if poly1.area != 0 and poly2.area != 0:
                    return poly_area / poly1.area * 100
            return 0.0

        def check_threshold_level(boxes_poly: list) -> tuple[bool, bool]:
            if not boxes_poly:
                return True, False
            false_detect = np.zeros(len(boxes_poly))
            for k in range(len(boxes_poly)):
                overlap_ratio = np.array(
                    [intersect_polygons(p, boxes_poly[k]) for p in boxes_poly],
                    dtype=float,
                )
                false_detections = [
                    idx for idx, val in enumerate(overlap_ratio) if val > 75
                ]
                false_detect[k] = len(false_detections[1:])
            threshold_change = bool(any(false_detect > 2))
            return threshold_change, threshold_change

        def compute_derivative(cent_boxes: np.ndarray) -> np.ndarray:
            n_boxes = cent_boxes.shape[0]
            dy_over_dx = np.zeros((n_boxes, n_boxes)) + 10
            for k in range(n_boxes):
                for m in range(n_boxes):
                    dx = abs(cent_boxes[k, 0] - cent_boxes[m, 0])
                    dy = abs(cent_boxes[k, 1] - cent_boxes[m, 1])
                    if k != m:
                        dy_over_dx[k, m] = dy / dx
            return dy_over_dx

        # --- inference ---

        print('\nDetermining the number of stories for each building...')
        gtf_infer = _load_infer(str(model_path), gpu_enabled)

        start_time = time.time()
        predictions = {}
        if checkpoint is not None:
            predictions = checkpoint.load()
        pending = [
            (key, im_path)
            for key, im_path in zip(image_keys, image_list)
            if key not in predictions
        ]
        n_total = len(image_keys)
        n_done = n_total - len(pending)
        if n_done:
            print(f'Resuming: {n_done:,} of {n_total:,} already detected')

        def _record(key, value):
            predictions[key] = value
            if checkpoint is not None:
                checkpoint.add(key, value)

        temp_dir = TemporaryDirectory(prefix='openplaces-n-stories-')
        tmp_img = Path(temp_dir.name) / 'input.jpg'

        print('\nPerforming story detections...')
        for key, im_path in tqdm(pending, initial=n_done, total=n_total):
            if not os.path.isfile(im_path):
                _record(key, None)
                continue

            # Load + resize with PIL (replaces cv2)
            pil_img = (
                PILImage.open(im_path)
                .convert('RGB')
                .resize((640, 640), PILImage.LANCZOS)
            )
            pil_img.save(str(tmp_img))

            _, _, boxes = gtf_infer.predict(str(tmp_img), threshold=0.2)
            boxes_poly = [create_polygon(b) for b in boxes]

            multiplier = 1
            while check_threshold_level(boxes_poly)[0]:
                if check_threshold_level(boxes_poly)[1]:
                    conf_threshold = 0.2 + multiplier * 0.1
                    if conf_threshold > 1:
                        break
                else:
                    conf_threshold = 0.2 - multiplier * 0.02
                    if conf_threshold == 0:
                        break
                _, _, boxes = gtf_infer.predict(str(tmp_img), threshold=conf_threshold)
                multiplier += 1
                boxes_poly = [create_polygon(b) for b in boxes]

            # Remove nested boxes (>75 % overlap)
            boxes_poly = [create_polygon(b) for b in boxes]
            nested_boxes = np.zeros(10 * len(boxes), dtype=int)
            counter = 0
            for bbox_poly in boxes_poly:
                overlap_ratio = np.array(
                    [intersect_polygons(p, bbox_poly) for p in boxes_poly],
                    dtype=float,
                )
                ind = [idx for idx, val in enumerate(overlap_ratio) if val > 75][1:]
                nested_boxes[counter : counter + len(ind)] = ind
                counter += len(ind)
            nested_boxes = np.unique(nested_boxes[:counter])

            counter = 0
            for box_ind in nested_boxes:
                del boxes[box_ind - counter]
                counter += 1

            n_boxes = len(boxes)
            boxes_poly = []
            boxes_extended_poly = []
            cent_boxes = np.zeros((n_boxes, 2))
            img_h = 640  # after resize

            for k in range(n_boxes):
                bbox = boxes[k]
                temp_poly = create_polygon(bbox)
                boxes_poly.append(temp_poly)
                xc, yc = temp_poly.centroid.xy
                cent_boxes[k, :] = np.array([xc[0], yc[0]])
                boxes_extended_poly.append(
                    create_polygon([0.9 * bbox[0], 0, 1.1 * bbox[2], img_h - 1])
                )

            stacked_ind = []
            for bbox in boxes_extended_poly:
                overlap_ratio = np.array(
                    [intersect_polygons(p, bbox) for p in boxes_extended_poly],
                    dtype=float,
                )
                stacked_ind.append(
                    [idx for idx, val in enumerate(overlap_ratio) if val > 10]
                )

            dy_over_dx = compute_derivative(cent_boxes)
            stacks = np.where(dy_over_dx > 1.3)

            counter = 0
            unique_stacks0 = [[] for _ in range(n_boxes)]
            for k in range(n_boxes):
                while counter < len(stacks[0]) and k == stacks[0][counter]:
                    unique_stacks0[k].append(stacks[1][counter])
                    counter += 1

            unique_stacks0 = [list(x) for x in set(tuple(x) for x in unique_stacks0)]

            if len(unique_stacks0) <= 1:
                n_stories = len(unique_stacks0[0]) if unique_stacks0 else 0
            else:
                lbound = img_h / 5
                ubound = 4 * img_h / 5
                middle_poly = Polygon(
                    [
                        (lbound, 0),
                        (lbound, img_h),
                        (ubound, img_h),
                        (ubound, 0),
                    ]
                )
                overlap_ratio = np.empty(len(unique_stacks0))
                for k in range(len(unique_stacks0)):
                    poly = unary_union([boxes_poly[x] for x in unique_stacks0[k]])
                    overlap_ratio[k] = intersect_polygons(poly, middle_poly)

                ind_keep = np.argsort(-overlap_ratio)[:2]
                stack4address = [
                    ind_keep[k] for k in range(2) if overlap_ratio[ind_keep[k]] > 10
                ]
                if stack4address:
                    n_stories = max(len(unique_stacks0[x]) for x in stack4address)
                else:
                    n_stories = len(unique_stacks0[0])

            _record(key, n_stories)

        temp_dir.cleanup()
        if checkpoint is not None:
            checkpoint.flush()

        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(
            f'\nTotal execution time: {int(hours):02}:{int(minutes):02}:{seconds:05.2f}'
        )

        return predictions
