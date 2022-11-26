import numpy as np
from mmcv.cnn.utils import initialize
from dnn.models import BaseModel
from dnn.models.builder import MODELS, build_net, build_losses
from dnn.core.evaluation.metrics import psnr
from dnn.core.utils import tensor2img, set_requires_grad


@MODELS.register_module()
class SRVGGModel(BaseModel):
    def __init__(self, net, loss_cfg=None, train_cfg=None, test_cfg=None, init_cfg=None):
        super(SRVGGModel, self).__init__(init_cfg)
        self.generator = build_net(net)
        initialize(self.generator, init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cfg = loss_cfg
        self.loss_fn = build_losses(loss_cfg)
        # self.init_weights()

    def forward_dummy(self, data):
        self.generator(*data)

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
        for i in range(0, gt.size(0)):
            # define dict(eval_result=) for base_dataset
            out_ = tensor2img(out[i, :, :, :], out_type=np.float32)
            gt_ = tensor2img(gt[i, :, :, :], out_type=np.float32)
            lq_ = tensor2img(lq[i, :, :, :], out_type=np.float32)
            psnr_ = psnr(out_, gt_)
            result = dict(
                eval_result=dict(psnr=psnr_),
                data_result=dict(out=out_, gt=gt_, lq=lq_)
            # img_metas=data['img_metas'][i]
            )
            results.append(result)

        return results


@MODELS.register_module()
class SRVGGGanModel(BaseModel):
    def __init__(self,
                 net,
                 loss_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(SRVGGGanModel, self).__init__(init_cfg)
        self.generator = build_net(net['g'])
        initialize(self.generator, init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cfg = loss_cfg
        self.loss_fn = build_losses(loss_cfg)
        self.discriminator = build_net(net['d'])
        initialize(self.discriminator, dict(type='Xavier'))

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get('disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else self.train_cfg.get('disc_init_steps', 0))
        self.step_counter = 0  # counting training steps

    def forward_dummy(self, data):
        """Used for computing network flops.
        See `tools/analysis_tools/get_flops.py`
        """
        out = self.generator(**data)
        return out

    def forward_train(self, data, optimizer):

        lq = data['lq']
        gt = data['gt']

        # generator
        fake_g_output = self.generator(lq)

        losses = dict()
        log_vars = dict()

        # no updates to discriminator parameters.
        set_requires_grad(self.discriminator, False)

        if self.step_counter % self.disc_steps == 0 and self.step_counter >= self.disc_init_steps:
            if isinstance(self.loss_fn, dict):
                for key, fn in self.loss_fn.items():
                    if 'loss_gan' not in key:
                        losses[key] = fn(fake_g_output, gt)

            # gan loss for generator
            fake_g_pred = self.discriminator(fake_g_output)
            losses['loss_gan'] = self.loss_fn['loss_gan'](fake_g_pred, target_is_real=True, is_disc=False)

            # parse loss
            loss_g, log_vars_g = self._parse_losses(losses)
            log_vars.update(log_vars_g)

            # optimize
            optimizer['generator'].zero_grad()
            loss_g.backward()
            optimizer['generator'].step()

        # discriminator
        set_requires_grad(self.discriminator, True)
        # real
        real_d_pred = self.discriminator(gt)
        loss_d_real = self.loss_fn['loss_gan'](real_d_pred, target_is_real=True, is_disc=True)
        loss_d, log_vars_d = self._parse_losses(dict(loss_d_real=loss_d_real))

        # optimize
        optimizer['discriminator'].zero_grad()
        loss_d.backward()
        log_vars.update(log_vars_d)

        # fake
        fake_d_pred = self.discriminator(fake_g_output.detach())
        loss_d_fake = self.loss_fn['loss_gan'](fake_d_pred, target_is_real=False, is_disc=True)
        loss_d, log_vars_d = self._parse_losses(dict(loss_d_fake=loss_d_fake))

        loss_d.backward()
        log_vars.update(log_vars_d)
        optimizer['discriminator'].step()

        self.step_counter += 1

        log_vars.pop('loss')  # remove the unnecessary 'loss'
        outputs = dict(log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def forward_test(self, data):
        lq = data['lq']
        gt = data['gt']
        out = self.generator(lq)
        # define results list for test.py
        results = []

        # batch_size > 1
        for i in range(gt.size(0)):
            # define dict(eval_result=) for base_dataset
            out_ = tensor2img(out[i, :, :, :], out_type=np.float32)
            gt_ = tensor2img(gt[i, :, :, :], out_type=np.float32)
            lq_ = tensor2img(lq[i, :, :, :], out_type=np.float32)
            psnr_ = psnr(out_, gt_)
            result = dict(
                eval_result=dict(psnr=psnr_),
                data_result=dict(out=out_, gt=gt_, lq=lq_)
            )
            results.append(result)

        return results

    def train_step(self, data, optimizer):

        outputs = self(data, return_loss=True, optimizer=optimizer)

        return outputs
