# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np


def psnr(img1, img2, crop_border=0):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse_value = np.mean((img1 - img2)**2)
    if mse_value == 0:
        return float('inf')
    return 20. * np.log10(1.0 / np.sqrt(mse_value))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, crop_border=0):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def end_point_error_map(flow_pred, flow_gt):
    """Calculate end point error map.

    Args:
        flow_pred (ndarray): The predicted optical flow with the
            shape (H, W, 2).
        flow_gt (ndarray): The ground truth of optical flow with the shape
            (H, W, 2).

    Returns:
        ndarray: End point error map with the shape (H , W).
    """
    return np.sqrt(np.sum((flow_pred - flow_gt)**2, axis=-1))


def end_point_error(flow_pred, flow_gt, valid_gt):
    """Calculate end point errors between prediction and ground truth.

    Args:
        flow_pred (list): output list of flow map from flow_estimator
            shape(H, W, 2).
        flow_gt (list): ground truth list of flow map shape(H, W, 2).
        valid_gt (list): the list of valid mask for ground truth with the
            shape (H, W).

    Returns:
        float: end point error for output.
    """
    epe_list = []
    assert len(flow_pred) == len(flow_gt)
    for _flow_pred, _flow_gt, _valid_gt in zip(flow_pred, flow_gt, valid_gt):
        epe_map = end_point_error_map(_flow_pred, _flow_gt)
        val = _valid_gt.reshape(-1) >= 0.5
        epe_list.append(epe_map.reshape(-1)[val])

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)

    return epe


def optical_flow_outliers(flow_pred, flow_gt, valid_gt):
    """Calculate percentage of optical flow outliers for KITTI dataset.

    Args:
        flow_pred (list): output list of flow map from flow_estimator
            shape(H, W, 2).
        flow_gt (list): ground truth list of flow map shape(H, W, 2).
        valid_gt (list): the list of valid mask for ground truth with the
            shape (H, W).

    Returns:
        float: optical flow outliers for output.
    """
    out_list = []
    assert len(flow_pred) == len(flow_gt) == len(valid_gt)
    for _flow_pred, _flow_gt, _valid_gt in zip(flow_pred, flow_gt, valid_gt):
        epe_map = end_point_error_map(_flow_pred, _flow_gt)
        epe = epe_map.reshape(-1)
        mag_map = np.sqrt(np.sum(_flow_gt**2, axis=-1))
        mag = mag_map.reshape(-1) + 1e-6
        val = _valid_gt.reshape(-1) >= 0.5
        # 3.0 and 0.05 is tooken from KITTI devkit
        # Inliers are defined as EPE < 3 pixels or < 5%
        out = ((epe > 3.0) & ((epe / mag) > 0.05)).astype(float)
        out_list.append(out[val])
    out_list = np.concatenate(out_list)
    fl = 100 * np.mean(out_list)

    return fl
