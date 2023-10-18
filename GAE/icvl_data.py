#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

#  要 改count
'''
@Useway  :   迭代产生训练数据
@File    :   data.py
@Time    :   2020/12/31 18:08:52
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/12/31 18:08:52
'''

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from torch.nn.functional import interpolate
import h5py
from scipy import io, misc


class LoadData(Dataset):

    def __init__(self, path, label, s=4, channels=31, fis=144):
        # num 31 128 128
        if label == 'train':
            num = 2640
        elif label == 'val':
            num = 80
        else:
            num = 1360
        # ICLV图像格式，光谱维度在第二个维度
        self.HR = torch.zeros([num, channels, fis, fis])

        count = 0
        for i in path:
            print(i)
            # for i in range(1):

            # ICVL数据集的mat格式，需要用h5py进行读取
            # img = h5py.File(i, 'r')['rad']
            # img = h5py.File('./data/4cam_0411-1640-1.mat', 'r')['rad']

            # havard数据集，用io读取，与之前不一样，初始的光谱维度在第三个，得换一下变成第一个
            img = io.loadmat(i)['ref']
            img = img.transpose(2, 0, 1)
            img = np.array(img)

            # 图像归一化操作
            img = np.asarray(img, dtype='float32')
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            # img /= 4095.0
            img = torch.tensor(img)

            print(img.size()[1], img.size()[2])
            for x in range(0, img.size()[1] - fis, fis):
                for y in range(0, img.size()[2] - fis, fis):
                    self.HR[count] = img[:, x:x + fis, y:y + fis]
                    count += 1
            del img

        print('safasfasfsdfds:{}', format(count))
        self.LR = self.down_sample(self.HR)

    def down_sample(self, data, s=4):
        # TODO: 添加高斯噪声(0.01) 并降采样
        # data = data + 0.0000001*torch.randn(*(data.shape))

        data = interpolate(
            data,
            scale_factor=1 / s,
            mode='bicubic',
            align_corners=True
        )

        return data

    def __len__(self):
        return self.HR.shape[0]

    def __getitem__(self, index):
        return self.LR[index], self.HR[index]

# x = LoadData()
# for i,j in x:
#     print(i.shape)
#     print(j.shape)
