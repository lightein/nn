import os
import cv2
import numpy as np
from dnn.datasets import BasePipeline
from dnn.datasets import PIPELINES


@PIPELINES.register_module()
class SRPipeline(BasePipeline):

    def __init__(self, pipeline_cfg):
        super(SRPipeline, self).__init__(pipeline_cfg)

    def __call__(self, results):
        gt_path = results['gt_path']
        gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        gt_h, gt_w = gt_img.shape[:2]

        crop_size = self.pipeline_cfg.get('crop_size', None)
        if crop_size is not None:
            crop_h, crop_w = crop_size[:2]
            x_offset = np.random.randint(0, gt_w - crop_w + 1)
            y_offset = np.random.randint(0, gt_h - crop_h + 1)
            gt_img_ = gt_img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w, ...]
        else:
            gt_img_ = gt_img.copy()

        scale = self.pipeline_cfg['scale']

        lq_img_ = cv2.resize(gt_img_, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)

        results['gt_img'] = gt_img_ / np.float32(255.0)
        results['lq_img'] = lq_img_ / np.float32(255.0)
        results['filename'] = os.path.splitext(os.path.basename(results['gt_path']))[0]

        return results

