# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   metrics.py
@Time    :   2019/12/4 17:35
@Desc    :
"""
import numpy as np
from scipy.signal import convolve2d
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from HStest import HSTestData
import torch.utils.data as data
from os import listdir
from os.path import join
import scipy.io as scio
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

def compare_ergas(x_true, x_pred, ratio):
    """
    Calculate ERGAS, ERGAS offers a global indication of the quality of fused image.The ideal value is 0.
    :param x_true:
    :param x_pred:
    :param ratio: 上采样系数
    :return:
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    sum_ergas = 0
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / (np.mean(vec_x)**2)
        sum_ergas += tmp
    return (100 / ratio) * np.sqrt(sum_ergas / x_true.shape[0])


def compare_sam(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """
    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred)))
                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi
    return sam_deg


def compare_corr(x_true, x_pred):
    """
    Calculate the cross correlation between x_pred and x_true.
    求对应波段的相关系数，然后取均值
    CC is a spatial measure.
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    x_true = x_true - np.mean(x_true, axis=1).reshape(-1, 1)
    x_pred = x_pred - np.mean(x_pred, axis=1).reshape(-1, 1)
    numerator = np.sum(x_true * x_pred, axis=1).reshape(-1, 1)
    denominator = np.sqrt(np.sum(x_true * x_true, axis=1) * np.sum(x_pred * x_pred, axis=1)).reshape(-1, 1)
    return (numerator / denominator).mean()


def img_2d_mat(x_true, x_pred):
    """
    # 将三维的多光谱图像转为2位矩阵
    :param x_true: (H, W, C)
    :param x_pred: (H, W, C)
    :return: a matrix which shape is (C, H * W)
    """
    h, w, c = x_true.shape
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    x_mat = np.zeros((c, h * w), dtype=np.float32)
    y_mat = np.zeros((c, h * w), dtype=np.float32)
    for i in range(c):
        x_mat[i] = x_true[:, :, i].reshape((1, -1))
        y_mat[i] = x_pred[:, :, i].reshape((1, -1))
    return x_mat, y_mat


def compare_rmse(x_true, x_pred):
    """
    Calculate Root mean squared error
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.shape[0] * x_true.shape[1] * x_true.shape[2]))


def compare_mpsnr(x_true, x_pred, data_range):
    """
    :param x_true: Input image must have three dimension (H, W, C)
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [compare_psnr(image_true=x_true[:, :, k], image_test=x_pred[:, :, k], data_range=data_range)
                  for k in range(channels)]

    return np.mean(total_psnr)


def compare_mssim(x_true, x_pred, data_range):
    """
    :param x_true:
    :param x_pred:
    :param data_range:
    :param multidimension:
    :return:
    """
    mssim = [compare_ssim(im1=x_true[:, :, i], im2=x_pred[:, :, i], data_range=data_range)
            for i in range(x_true.shape[2])]

    return np.mean(mssim)


def compare_sid(x_true, x_pred):
    """
    SID is an information theoretic measure for spectral similarity and discriminability.
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    N = x_true.shape[2]
    err = np.zeros(N)
    for i in range(N):
        err[i] = abs(np.sum(x_pred[:, :, i] * np.log10((x_pred[:, :, i] + 1e-3) / (x_true[:, :, i] + 1e-3))) +
                     np.sum(x_true[:, :, i] * np.log10((x_true[:, :, i] + 1e-3) / (x_pred[:, :, i] + 1e-3))))
    return np.mean(err / (x_true.shape[1] * x_true.shape[0]))


def compare_appsa(x_true, x_pred):
    """
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    nom = np.sum(x_true * x_pred, axis=2)
    denom = np.linalg.norm(x_true, axis=2) * np.linalg.norm(x_pred, axis=2)

    cos = np.where((nom / (denom + 1e-3)) > 1, 1, (nom / (denom + 1e-3)))
    appsa = np.arccos(cos)
    return np.sum(appsa) / (x_true.shape[1] * x_true.shape[0])


def compare_mare(x_true, x_pred):
    """
    :param x_true:
    :param x_pred:
    :return:
    """
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    diff = x_true - x_pred
    abs_diff = np.abs(diff)
    relative_abs_diff = np.divide(abs_diff, x_true + 1)  # added epsilon to avoid division by zero.
    return np.mean(relative_abs_diff)


def img_qi(img1, img2, block_size=8):
    N = block_size ** 2
    sum2_filter = np.ones((block_size, block_size))

    img1_sq = img1 * img1
    img2_sq = img2 * img2
    img12 = img1 * img2

    img1_sum = convolve2d(img1, np.rot90(sum2_filter), mode='valid')
    img2_sum = convolve2d(img2, np.rot90(sum2_filter), mode='valid')
    img1_sq_sum = convolve2d(img1_sq, np.rot90(sum2_filter), mode='valid')
    img2_sq_sum = convolve2d(img2_sq, np.rot90(sum2_filter), mode='valid')
    img12_sum = convolve2d(img12, np.rot90(sum2_filter), mode='valid')

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum * img1_sum + img2_sum * img2_sum
    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul
    quality_map = np.ones(denominator.shape)
    index = (denominator1 == 0) & (img12_sq_sum_mul != 0)
    quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]
    index = (denominator != 0)
    quality_map[index] = numerator[index] / denominator[index]
    return quality_map.mean()


def compare_qave(x_true, x_pred, block_size=8):
    n_bands = x_true.shape[2]
    q_orig = np.zeros(n_bands)
    for idim in range(n_bands):
        q_orig[idim] = img_qi(x_true[:, :, idim], x_pred[:, :, idim], block_size)
    return q_orig.mean()


def quality_assessment(x_true, x_pred, data_range, ratio, multi_dimension=False, block_size=8):
    """
    :param multi_dimension:
    :param ratio:
    :param data_range:
    :param x_true:
    :param x_pred:
    :param block_size
    :return:
    """
    result = {'MPSNR': compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range),
              'MSSIM': compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range),
              'ERGAS': compare_ergas(x_true=x_true, x_pred=x_pred, ratio=ratio),
              'SAM': compare_sam(x_true=x_true, x_pred=x_pred),
              # 'SID': compare_sid(x_true=x_true, x_pred=x_pred),
              'CrossCorrelation': compare_corr(x_true=x_true, x_pred=x_pred),
              'RMSE': compare_rmse(x_true=x_true, x_pred=x_pred),
              # 'APPSA': compare_appsa(x_true=x_true, x_pred=x_pred),
              # 'MARE': compare_mare(x_true=x_true, x_pred=x_pred),
              # "QAVE": compare_qave(x_true=x_true, x_pred=x_pred, block_size=block_size)
              }
    return result

# from scipy import io as sio
# im_out = np.array(sio.loadmat('/home/zhwzhong/PycharmProject/HyperSR/SOAT/HyperSR/SRindices/Chikuse_EDSRViDeCNN_Blocks=9_Feats=256_Loss_H_Real_1_1_X2X2_N5new_BS32_Epo60_epoch_60_Fri_Sep_20_21:38:44_2019.mat')['output'])
# im_gt = np.array(sio.loadmat('/home/zhwzhong/PycharmProject/HyperSR/SOAT/HyperSR/SRindices/Chikusei_test.mat')['gt'])
#
# sum_rmse, sum_sam, sum_psnr, sum_ssim, sum_ergas = [], [], [], [], []
# for i in range(im_gt.shape[0]):
#     print(im_out[i].shape)
#     score = quality_assessment(x_pred=im_out[i], x_true=im_gt[i], data_range=1, ratio=4, multi_dimension=False, block_size=8)
#     sum_rmse.append(score['RMSE'])
#     sum_psnr.append(score['MPSNR'])
#     sum_ssim.append(score['MSSIM'])
#     sum_sam.append(score['SAM'])
#     sum_ergas.append(score['ERGAS'])
#
# print(np.mean(sum_rmse), np.mean(sum_psnr), np.mean(sum_ssim), np.mean(sum_sam))


import numpy as np

def color_correction(lr_input, hr_output, num_channels=31):
    """\n    Perform color correction on the generated HR image to align its mean and variance with those of the LR input.\n    \n    Args:\n    - lr_input: numpy array, shape=(height, width, 3), the LR input image\n    - hr_output: numpy array, shape=(height*scale, width*scale, 3), the generated HR image\n    \n    Returns:\n    - numpy array, shape=(height*scale, width*scale, 3), the color-corrected output image\n    """
    # Calculate mean and standard deviation of each channel in the generated HR image
    hr_mean = np.mean(hr_output, axis=(0, 1))
    hr_std = np.std(hr_output, axis=(0, 1))
    
    # Calculate mean and standard deviation of each channel in the LR input image
    lr_mean = np.mean(lr_input, axis=(0, 1))
    lr_std = np.std(lr_input, axis=(0, 1))
    
    # Perform color correction on each channel
    corrected_output = np.zeros(hr_output.shape, dtype=np.float32)
    for c in range(num_channels):
        corrected_output[:, :, c] = (hr_output[:, :, c] - hr_mean[c]) / hr_std[c] * lr_std[c] + lr_mean[c]
        
    return np.clip(corrected_output, 0.0, 1.0)

def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

class TestsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(TestsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # 把文件先都读取到CPU
        self.img = []
        print("kai shi du qu shu ju ce shi shu ju le")
        for i in range(len(self.image_filenames)):
            print(i)
            mat = scio.loadmat(self.image_filenames[i], verify_compressed_data_integrity=False)
            self.img.append(mat)
        print("gong xi ni !!! shu ju du qu cheng gong le!!!")


    def __getitem__(self, index):
        # mat = scio.loadmat(self.image_filenames[index])
        mat = self.img[index]
        input = mat['LR'].astype(np.float32).transpose(2, 0, 1)
        label = mat['HR'].astype(np.float32).transpose(2, 0, 1)

        input = input[:,:32,:32]
        label = label[:,:128,:128]

        img_HR = torch.from_numpy(label)
        img_LR = torch.from_numpy(input)
        # 这里是上采样了四倍，具体情况改变这个数值。
        # img_LR_1 = img_LR.reshape(1,3,32,32)
        # print(img_LR.shape)
        img_LR_1 = img_LR.reshape(1,102,32,32) # 除了PaviaC数据集，别的都是512，512
        
        img_SR = torch.nn.functional.interpolate(img_LR_1, scale_factor=4, mode='bicubic')

        return {'HR': img_HR, 'SR': img_SR[0], 'LR': img_LR}

    def __len__(self):
        return len(self.image_filenames)

if __name__ == "__main__":
    # pred_list = np.load('./SR3_3_result/Pav_pred_list.npy')
    pred_list = np.load('./Chi4_timetest_list.npy')
    print(len(pred_list))

    result_list = []

    test_num = 64
    channels_3 = 42

    for j in range(test_num):
        indices = [j + i*test_num for i in range(channels_3)]
        print(indices)
        data_list = []
        # for i in indices:
        #     # 直接进行拼接，波段顺序连续
        #     data_list.append(pred_list[i])

        # 每一个里面，存放着间隔的波段,每次循环放一个位置的波段，一共进行三次，复原全部。
        for i in indices:
            # print(pred_list[i][:,:,0][:,:, np.newaxis].shape)
            data_list.append(pred_list[i][:,:,0][:,:, np.newaxis])
        for i in indices:
            data_list.append(pred_list[i][:,:,1][:,:, np.newaxis])
        for i in indices:
            data_list.append(pred_list[i][:,:,2][:,:, np.newaxis])

        # Chikusei 数据集，126通道，把最后俩当作真实的，重复拼接一下。
        data_list.append(pred_list[indices[-1]][:,:,2][:,:, np.newaxis])
        data_list.append(pred_list[indices[-1]][:,:,2][:,:, np.newaxis])

        result = np.concatenate(data_list, axis=-1)
        print(result.shape)
        # result = np.delete(result, -2, axis=-1)
        # result = np.delete(result, -2, axis=-1)
        # print(result.shape)
        result_list.append(result)
    print(len(result_list))


    # 真实数据起初是测试的时候对应的生成，现在更换为真实的样本直接读取。
    # gt_list = np.load('Chi_gt_list.npy')
    # print(len(gt_list))
    # gr_list = []
    # for j in range(4):
    #     indices = [j + i*4 for i in range(42)]
    #     print(indices)
    #     data_list = []
    #     for i in indices:
    #         data_list.append(gt_list[i])
    #     result = np.concatenate(data_list, axis=-1)
    #     # result = np.delete(result, -1, axis=-1)
    #     # result = np.delete(result, -1, axis=-1)
    #     gr_list.append(result)
    # print(len(gr_list))

    # val_set = TestsetFromFolder('../Harvard_4_test/')
    # val_set = TestsetFromFolder('../test/Cave/4/4')
    # val_set = TestsetFromFolder('../test/Foster/4/')
    # val_set = TestsetFromFolder('../test/Chikusei/4/')
    # val_set = TestsetFromFolder('../test/PaviaC/4/')
    val_set = HSTestData(image_dir= '../Chikusei_mat/128test/', n_scale = 4, ch3=False, num_ch=0)
    print(len(val_set))
    gr_list = []
    for mat in val_set:
        gr_list.append(mat['HR'].numpy())


    for idx in range(test_num):
        y = result_list[idx]
        gt = gr_list[idx].transpose(1,2,0)

        # y = y[:128,:128,:]
        # gt = gt[:128,:128,:]
        y = color_correction(gt, y, num_channels=102)

        print(y.shape, gt.shape)
        if idx == 0:
            indices = quality_assessment(gt, y, data_range=1., ratio=4)
        else:
            indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
        # indices = quality_assessment(gt, y, data_range=1., ratio=4)
        print(indices)
    # 平均一下。
    for index in indices:
        indices[index] = indices[index] / (idx+1)
    print("最终的结果平均指标为 {}".format(indices))
    