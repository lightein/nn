# Copyright (c) OpenMMLab. All rights reserved.
from .composition_loss import (CharbonnierCompLoss, L1CompositionLoss,
                               MSECompositionLoss)
from .feature_loss import LightCNNFeatureLoss
from .gan_loss import DiscShiftLoss, GANLoss, GaussianBlur, GradientPenaltyLoss
from .gradient_loss import GradientLoss
from .perceptual_loss import (PerceptualLoss, PerceptualVGG,
                              TransferalPerceptualLoss)
from .pixelwise_loss import CharbonnierLoss, L1Loss, MaskedTVLoss, MSELoss
# from .mse_loss import MSELoss, mse_loss
# from .smooth_l1_loss import L1Loss, SmoothL1Loss, l1_loss, smooth_l1_loss
from .utils import (reduce_loss, weight_reduce_loss, weighted_loss,
                    mask_reduce_loss, masked_loss)
