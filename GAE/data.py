#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


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


class LoadData(Dataset):

    def __init__(self, path, s=4, fis=144):
        # num 31 512 512 
        self.data = np.load(path)
        self.data = torch.from_numpy(self.data)
        self.data /= (2 ** 16 - 1)
        # print(torch.max(self.data))

        # TODO: 先边缘裁剪 以获取HR
        shape = self.data.shape
        # self.data = self.data[:,:,(s+6):shape[2]-(s+6),(s+6):shape[3]-(s+6)]

        # 取三张
        # 32*3 31 144 144
        self.HR = torch.zeros((shape[0] * 9, 31, 144, 144))

        count = 0
        for i in range(shape[0]):
            for x in range((s + 6), shape[2] - (s + 6) - fis, fis):
                for y in range((s + 6), shape[2] - (s + 6) - fis, fis):
                    self.HR[count] = self.data[i, :, x:x + fis, y:y + fis]
                    count += 1
        # 得到LR图像 num*9 31 36 36 
        # print(count)
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
