import numpy as np
from mmcv.cnn.utils import initialize
from dnn.models import BaseModel
from dnn.models import build_net, build_losses
from dnn.models.builder import MODELS
from dnn.core.evaluation.metrics import psnr
from dnn.core.utils import tensor2img


@MODELS.register_module()
class SRVGGModel(BaseModel):

    def __init__(self, net, loss_cfg=None, train_cfg=None, test_cfg=None, init_cfg=None):
        super(SRVGGModel, self).__init__(init_cfg)
        self.generator = build_net(net)
        # initialize(self.generator, init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cfg = loss_cfg
        self.loss_fn = build_losses(loss_cfg)
        self.init_weights()

    def forward_dummy(self, data):
        self.generator(**data)

    def forward_train(self, data):
        gt = data['gt_img']
        lq = data['lq_img']

        x = self.generator(lq)

        losses = dict()

        if isinstance(self.loss_fn, dict):
            for key, fn in self.loss_fn.items():
                losses[key] = fn(x, gt)
        else:
            losses[self.loss_cfg['type']] = self.loss_fn(x, gt)

        return losses

    def forward_test(self, data):
        gt = data['gt_img']
        lq = data['lq_img']

        out = self.generator(lq)

        # define results list for test.py
        results = []

        # batch_size > 1
        for i in range(gt.size(0)):
            # define dict(eval_result=) for base_dataset
            out_ = tensor2img(out[i, :, :, :], out_type=np.float32)
            gt_ = tensor2img(gt[i, :, :, :], out_type=np.float32)
            result = dict(
                eval_result=dict(psnr=psnr(out_, gt_)),
                data_result=out_,
                # img_metas=data['img_metas'][i]
            )
            results.append(result)

        return results






