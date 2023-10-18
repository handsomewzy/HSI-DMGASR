import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
from scipy.signal import convolve2d
import torch
from eval_hsi import color_correction


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    # 数据本来就在01之间了，因此要调整min max值，并且不需要后面的了。
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    # tensor = (tensor - min_max[0]) / \
    #     (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.detach().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    # 不需要进行这个操作了，后面的函数都是基于最大值为1设计的。
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    # 针对HSI数据先大概选择三个通道作为RGB图像输出保存
    # img = img[:,:,[3,13,23]]
    # img = np.transpose(img,(1,2,0))
    # img = (img * 255.0).round()
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) #这是针对python存储图片的格式，但是我们是大概选取的RGB通道，不需要换了。
    cv2.imwrite(img_path, img)

def save_img3(img, img_path, mode='RGB'):
    # 针对HSI数据先大概选择三个通道作为RGB图像输出保存
    # img = img[:,:,[70,100,36]] # Chikusei
    # img = img[:,:,[10,30,100]] # PaviaC
    img = img[:,:,[5,15,25]] # Harvard

    # img = np.transpose(img,(1,2,0))
    # img = (img * 255.0).round()
    # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) #这是针对python存储图片的格式，但是我们是大概选取的RGB通道，不需要换了。
    cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=np.array([win_size, win_size]), sigma=1.5)
    window = window.astype(np.float32) / np.sum(np.sum(window))

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)).astype(np.float32) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))



def matlab_style_gauss2D(shape=np.array([11, 11]), sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    siz = (shape - np.array([1, 1])) / 2
    std = sigma
    eps = 2.2204e-16
    x = np.arange(-siz[1], siz[1] + 1, 1)
    y = np.arange(-siz[0], siz[1] + 1, 1)
    m, n = np.meshgrid(x, y)

    h = np.exp(-(m * m + n * n).astype(np.float32) / (2. * sigma * sigma))
    h[h < eps * h.max()] = 0
    sumh = h.sum()

    if sumh != 0:
        h = h.astype(np.float32) / sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)




# 下面三个都是专门用来计算HSI数据的评价指标函数。
def PSNR(pred, gt):
    # 最大的像素值为1，数据范围在01之间
    pred = pred.cpu().detach().numpy().transpose(1,2,0)
    gt = gt.cpu().detach().numpy().transpose(1,2,0)

    # 加入颜色修正
    # pred = color_correction(gt, pred)

    valid = gt - pred
    rmse = math.sqrt(np.mean(valid ** 2))

    if rmse == 0:
        return 100
    psnr = 20 * math.log10(1.0 / rmse)
    return psnr


def SSIM(pred, gt):
    # 最大像素值为1.01之间，c通道在最后。
    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()

    ssim = 0
    for i in range(gt.shape[0]):
        ssim = ssim + compute_ssim(pred[:, :, i], gt[:, :, i])
    return ssim / gt.shape[0]

def SSIM_CHW(pred, gt):
    # 最大像素值为1.01之间，c通道在最后。
    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()

    ssim = 0
    for i in range(gt.shape[0]):
        ssim = ssim + compute_ssim(pred[i, :, :], gt[i, :, :])
    return ssim / gt.shape[0]


def SAM(pred, gt):
    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()

    eps = 2.2204e-16
    pred[np.where(pred == 0)] = eps
    gt[np.where(gt == 0)] = eps

    nom = sum(pred * gt)
    denom1 = sum(pred * pred) ** 0.5
    denom2 = sum(gt * gt) ** 0.5
    sam = np.real(np.arccos(nom.astype(np.float32) / (denom1 * denom2 + eps)))
    sam[np.isnan(sam)] = 0
    sam_sum = np.mean(sam) * 180 / np.pi
    return sam_sum


import numpy as np

def calc_ergas(predicted, true):
    """
    计算relative global error (ERGAS)
    参数：
    predicted：预测结果，numpy数组，形状为（1，波段数，高度，宽度）   
    true：真实结果，numpy数组，形状为（1，波段数，高度，宽度）    
    scale：每个波段的比例因子，数组，长度等于波段数
    返回值：
    ergas：ERGAS值
    """
    # 将预测结果和真实结果拉伸成（样本数，类别数）的形式
    predicted = predicted.cpu().detach().numpy()
    true = true.cpu().detach().numpy()

    predicted = np.reshape(predicted, (-1, predicted.shape[0]))
    true = np.reshape(true, (-1, true.shape[0]))
    scale = np.random.rand(31)

    # 计算样本数和波段数
    n_samples, n_bands = predicted.shape

    # 计算每个类别的像元数目
    n_pixels = np.sum(true, axis=0)

    # 计算每个波段的均值
    mean = np.mean(true, axis=0)

    # 计算每个波段的标准差
    std = np.std(true, axis=0)

    # 计算ERGAS
    ergas = 0
    for i in range(n_bands):
        ergas += np.square((scale[i] / mean[i]) * np.sqrt(np.mean(np.square(predicted[:, i] - true[:, i]))) / std[i]) * (n_pixels[i] / np.sum(n_pixels))
    ergas = 100 / n_bands * np.sqrt(ergas)

    return ergas
