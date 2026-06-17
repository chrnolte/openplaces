# BRAILS++ story-count inference engine - ported from
# brails.processors.nfloors_detector.lib.infer_detector
# (Yet-Another-EfficientDet-Pytorch)
# Original authors: Zylo117, NHERI-SimCenter / bacetiner

import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

if sys.platform == 'win32':
    # conda-forge pytorch bundles its VC++ runtime DLLs in Library\bin, but
    # that directory is not on Windows' DLL search path by default, so
    # shm.dll fails to find its dependencies (WinError 127).
    _env_lib = os.path.join(sys.prefix, 'Library', 'bin')
    if os.path.isdir(_env_lib):
        os.add_dll_directory(_env_lib)

import torch

from .backbone import EfficientDetBackbone
from .efficientdet.utils import (
    BBoxTransform,
    ClipBoxes,
)
from .utils.utils import (
    invert_affine,
    postprocess,
    preprocess,
)

_COMPOUND_COEF_MAP = {
    'd0': 0,
    'd1': 1,
    'd2': 2,
    'd3': 3,
    'd4': 4,
    'd5': 5,
    'd6': 6,
    'd7': 7,
}
_INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


class Infer:
    """EfficientDet inference wrapper — matches BRAILS++ infer_detector.Infer."""

    def __init__(self, verbose=1):
        self.system_dict = {
            'verbose': verbose,
            'local': {},
            'params': {
                'weights_file': '',
                'obj_list': [],
                'use_cuda': True,
                'threshold': 0.5,
                'iou_threshold': 0.2,
                'force_input_size': None,
                'anchor_ratios': [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                'anchor_scales': [2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
                'use_float16': False,
            },
        }

    def load_model(self, model_path, classes_list, use_gpu=True):
        compound_coef = 0
        for key, val in _COMPOUND_COEF_MAP.items():
            if key in model_path:
                compound_coef = val
                break

        p = self.system_dict['params']
        p['compound_coef'] = compound_coef
        p['weights_file'] = model_path
        p['obj_list'] = classes_list
        p['use_cuda'] = use_gpu and torch.cuda.is_available()

        input_size = (
            _INPUT_SIZES[compound_coef]
            if p['force_input_size'] is None
            else p['force_input_size']
        )
        self.system_dict['local']['input_size'] = input_size

        model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=len(classes_list),
            ratios=p['anchor_ratios'],
            scales=p['anchor_scales'],
        )

        map_loc = torch.device('cpu') if not p['use_cuda'] else None
        state = torch.load(model_path, map_location=map_loc, weights_only=False)
        model.load_state_dict(state)
        model.requires_grad_(False)
        model = model.eval()

        if p['use_cuda']:
            model = model.cuda()
        self.system_dict['local']['model'] = model

    def predict(self, img_path, threshold=0.5):
        p = self.system_dict['params']
        p['threshold'] = threshold
        input_size = self.system_dict['local']['input_size']

        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

        use_cuda = p['use_cuda']
        use_float16 = p['use_float16']

        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        dtype = torch.float16 if use_float16 else torch.float32
        x = x.to(dtype).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = self.system_dict['local'][
                'model'
            ](x)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            out = postprocess(
                x,
                anchors,
                regression,
                classification,
                regressBoxes,
                clipBoxes,
                threshold,
                p['iou_threshold'],
            )

        out = invert_affine(framed_metas, out)
        scores, labels, bboxes = self._extract(out)
        return scores, labels, bboxes

    def _extract(self, preds):
        threshold = self.system_dict['params']['threshold']
        obj_list = self.system_dict['params']['obj_list']
        scores, labels, bboxes = [], [], []
        for i in range(len(preds)):
            if len(preds[i]['rois']) == 0:
                continue
            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(int)
                score = float(preds[i]['scores'][j])
                if score > threshold:
                    scores.append(score)
                    labels.append(obj_list[preds[i]['class_ids'][j]])
                    bboxes.append([x1, y1, x2, y2])
        return scores, labels, bboxes
