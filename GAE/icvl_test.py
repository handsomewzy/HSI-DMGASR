#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   测试模型效果
@File    :   test.py
@Time    :   2021/01/14 10:13:19
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2021/01/14 10:13:19
'''

from pathlib import Path
import skimage.measure as measure
import sewar
import spectral
import MCNet
from net import Generator, Discriminator
import torch
from torch.utils.data import DataLoader
from G import OUT_DIR, TEST_DATA_PATH
from icvl_data import LoadData
from utils import *
from common import *
from SSPSR import *
from torch.nn.functional import interpolate
import eval

BATCH_SIZE = 16  # 经过测试，只有和训练的时候，设定一样的BATCH size才行，可能与他自己训练网络中的具体设定有关。
high_sr = 128
low_sr = high_sr / 4
FAKE_HR = torch.zeros([BATCH_SIZE, 31, high_sr, high_sr])
HR = torch.zeros([BATCH_SIZE, 31, high_sr, high_sr])

PSNRs = []
SAMs = []
SSIMs = []


class Bicubic(nn.Module):
    def __init__(self):
        super(Bicubic, self).__init__()

    def forward(self, x):
        x = interpolate(
            x,
            scale_factor=4,  # 这个与具体的超分辨比例有关，这个是全局skip时候，对初始图像进行上采样，一般设置为2 3 4
            mode='bicubic',
            align_corners=True
        )
        return x


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test decice is {}'.format(device))

    # GAN注意力机制网络的验证与读取
    # g_model = Generator(BATCH_SIZE).to(device)
    # state_dict_g = g_model.state_dict()
    #
    # g_weight = OUT_DIR.joinpath('GAN_4_Harvard.pth')
    # g_model = torch.load(g_weight)
    # print(g_model)
    #
    # d_model = Discriminator(BATCH_SIZE).to(device)
    # state_dict_d = d_model.state_dict()

    # d_weight = OUT_DIR.joinpath('icvl_d_model1.pth')
    # d_model = torch.load(d_weight)

    # MCNet的验证与读取
    MCNet = torch.load(OUT_DIR.joinpath('MCNet_4_Harvard.pth'))
    print(MCNet)
    g_model = MCNet

    # SSPSR的验证与读取
    # SSPSR_model = torch.load(OUT_DIR.joinpath('SSPSR_4_Harvard.pth'))
    # print(SSPSR_model)
    # g_model = SSPSR_model

    # 双线性插值，基础baseline网络
    # g_model = Bicubic()

    # for n, p in torch.load(g_weight, map_location=lambda storage, loc: storage).items():
    #     if n in state_dict_g.keys():
    #         state_dict_g[n].copy_(p)
    #     else:
    #         raise KeyError(n)
    #
    # for n, p in torch.load(d_weight, map_location=lambda storage, loc: storage).items():
    #     if n in state_dict_d.keys():
    #         state_dict_d[n].copy_(p)
    #     else:
    #         raise KeyError(n)

    g_model.eval()
    # d_model.eval()

    # 旧的测试集，不用了，新的测试集使用512*512的大小，只取图像的右上部分，与MCNet中的测试结果一致。
    _1, _2, test_paths = get_paths()
    test_data = DataLoader(
        LoadData(test_paths, 'test', fis=high_sr),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )



    count = 0
    for lr, hr in test_data:
        # GAN注意机制的测试，输入的尺寸为5通道的，SSPRS与MCNet输入的尺寸为四通道的，在检测时候注意区分。
        # lr = lr.reshape((lr.shape[0], 1, lr.shape[1], lr.shape[2], lr.shape[3]))
        # hr = hr.reshape((hr.shape[0], 1, hr.shape[1], hr.shape[2], hr.shape[3]))
        lr = lr.to(device)
        hr = hr.to(device)

        # print(lr.shape, hr.shape)

        with torch.no_grad():
            # GAN注意力模型和MCNet模型，输入只有一个，SSPSR输入有俩，其实也是一个。两个操作。
            fake_hr = g_model(lr)  # GAN与MCNet 二线性插值
            # fake_hr = g_model(lr, lr)  # SSPSR


            fake_hr = torch.squeeze(fake_hr)
            hr = torch.squeeze(hr)

            # 将生成的图片也进行归一化处理，不进行归一化！！！！输入的时候已经进行了归一化，现在进行归一化就改变了学习的分布，就变差了！
            # fake_hr = (fake_hr - torch.min(fake_hr)) / (torch.max(fake_hr) - torch.min(fake_hr))

            hr = hr.cpu()
            fake_hr = fake_hr.cpu()
            # print(fake_hr,hr)

            # print(torch.max(fake_hr), torch.max(hr))

            # Sewar库的使用尝试。
            # hr_numpy = hr.numpy()[0]
            # print(hr_numpy.shape)
            # # print(hr_numpy)
            # # print(hr_numpy.dtype)
            # fake_hr_numpy = fake_hr.numpy()[0]
            # # 将数据归一化后，设定最大值为1，将数值限定在零到壹。之前代码值不太对，很抽象。弃用，使用Sewar库的函数
            # print("第{}张图片的各个指标如下！！PSNR：{}，SSIM：{}，SAM：{}".format(count + 1, sewar.psnr(hr_numpy, fake_hr_numpy, MAX=1),
            #                                                      sewar.ssim(hr_numpy, fake_hr_numpy, MAX=1),
            #                                                      sewar.sam(hr_numpy, fake_hr_numpy)))

            total_PSNR = []
            total_ssim = []
            total_sam = []
            for i in range(BATCH_SIZE):
                # print(i)
                hr_numpy = hr.numpy()[i]
                fake_hr_numpy = fake_hr.numpy()[i]

                psnr = eval.PSNR(fake_hr_numpy, hr_numpy)
                sam = eval.SAM(fake_hr_numpy, hr_numpy)
                ssim = eval.SSIM(fake_hr_numpy, hr_numpy)
                # print('img : {} psnr : {:.4f}  sam : {:.4f}'.format(
                #     count + 1, psnr, sam
                # ))

                # 单独这个batch size 的累加
                total_PSNR.append(psnr)
                total_sam.append(sam)
                total_ssim.append(ssim)

            print("第{}个batch size的指标平均值如下！！！ PSNR：{}，SSIM：{}，SAM：{}".format(count, sum(total_PSNR) / len(total_PSNR),
                                                                            sum(total_ssim) / len(total_ssim),
                                                                            sum(total_sam) / len(total_sam)))
            # 整体的计算
            PSNRs.append(sum(total_PSNR) / len(total_PSNR))
            SAMs.append(sum(total_sam) / len(total_sam))
            SSIMs.append(sum(total_ssim) / len(total_ssim))

            # 图像的可视化操作
            # hr_numpy = hr_numpy.transpose(1, 2, 0)
            # fake_hr_numpy = fake_hr_numpy.transpose(1, 2, 0)
            # view1 = spectral.imshow(data=hr_numpy, bands=[30], title="img")
            # # view1 = spectral.imshow(data=hr_numpy, title="img")
            # view2 = spectral.imshow(data=fake_hr_numpy, bands=[30], title="img1")
            # # view2 = spectral.imshow(data=fake_hr_numpy, title="img1")
            # plt.pause(2)

            # 因为bs 设置的关系 算出来的 就是一张图的平均了
            # psnr = eval.PSNR(fake_hr, hr)
            # sam = eval.SAM(fake_hr, hr)
            # print('img : {} psnr : {:.4f}  sam : {:.4f}'.format(
            #     count + 1, psnr, sam
            # ))
            # PSNR += psnr
            # SAMs += sam

        # FAKE_HR[count * BATCH_SIZE:(count + 1) * BATCH_SIZE] = fake_hr
        # HR[count * BATCH_SIZE:(count + 1) * BATCH_SIZE] = hr
        # print(hr.size())

        count += 1
    print("测试集最终的指标平均值如下！！！ PSNR：{}，SSIM：{}，SAM：{}".format(sum(PSNRs) / len(PSNRs),
                                                           sum(SSIMs) / len(SSIMs),
                                                           sum(SAMs) / len(SAMs)))
