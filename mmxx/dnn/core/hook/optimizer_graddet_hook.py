# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import OptimizerHook as BaseOptimizerHook
from mmcv.runner.hooks import HOOKS
import torch


@HOOKS.register_module()
class OptimizerGradDetHook(BaseOptimizerHook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """

    def __init__(self, grad_clip=None, detect_anomalous_params=False):
        super(OptimizerGradDetHook, self).__init__(grad_clip, detect_anomalous_params)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        with torch.autograd.detect_anomaly():
            runner.outputs['loss'].backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()
