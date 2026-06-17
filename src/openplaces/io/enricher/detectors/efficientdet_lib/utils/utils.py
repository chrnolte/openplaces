# Author: Zylo117
# cv2 replaced with PIL + torch (no OpenCV dependency)
# torchvision NMS replaced with pure-torch implementation (no torchvision dependency)


import numpy as np
import torch
import torch.nn.functional as F_torch
from PIL import Image as PILImage


def _nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = boxes[order[1:]]
        x1 = rest[:, 0].clamp(min=boxes[i, 0])
        y1 = rest[:, 1].clamp(min=boxes[i, 1])
        x2 = rest[:, 2].clamp(max=boxes[i, 2])
        y2 = rest[:, 3].clamp(max=boxes[i, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = inter / (area_i + area_r - inter)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    offsets = idxs.to(boxes.dtype) * (boxes.max() + 1)
    return _nms(boxes + offsets[:, None], scores, iou_threshold)


def invert_affine(metas: float | list | tuple, preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (
                    new_w / old_w
                )
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (
                    new_h / old_h
                )
    return preds


def _resize_float_array(image: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """Bilinear resize of a float32 HxWxC numpy array (replaces cv2.resize)."""
    t = (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    t = F_torch.interpolate(
        t, size=(new_h, new_w), mode='bilinear', align_corners=False
    )
    return t.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        image = _resize_float_array(image, new_w, new_h)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return (
        canvas,
        new_w,
        new_h,
        old_w,
        old_h,
        padding_w,
        padding_h,
    )


def preprocess(
    *image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)
):
    ori_imgs = []
    for p in image_path:
        # Load with PIL (RGB) and convert to BGR to match cv2.imread channel order
        rgb = np.array(PILImage.open(p).convert('RGB'), dtype=np.float32)
        bgr = rgb[:, :, ::-1].copy()
        ori_imgs.append(bgr)
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [
        aspectaware_resize_padding(img[..., ::-1], max_size, max_size, means=None)
        for img in normalized_imgs
    ]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def postprocess(
    x,
    anchors,
    regression,
    classification,
    regressBoxes,
    clipBoxes,
    threshold,
    iou_threshold,
):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append(
                {
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                }
            )
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(
            1, 0
        )
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(
            transformed_anchors_per,
            scores_per[:, 0],
            classes_,
            iou_threshold=iou_threshold,
        )

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append(
                {
                    'rois': boxes_.cpu().numpy(),
                    'class_ids': classes_.cpu().numpy(),
                    'scores': scores_.cpu().numpy(),
                }
            )
        else:
            out.append(
                {
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                }
            )

    return out
