import common

import torch.nn as nn
from mimetypes import init
from turtle import forward

from torch import conv2d
import torch
import torch.nn as nn
import common
#from model import common
import torch.nn.functional as F
import math 
import cv2
import os
import datetime
import scipy.io as io

import numpy as np
import torch
import numpy as np
from math import sqrt
import torch.nn as nn
from common import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from icvl_data import LoadData
from utils import SAM, PSNR_GPU, get_paths ,TrainsetFromFolder
import sewar
import MCNet
from pathlib import Path
from torch.nn.functional import interpolate
import torch.distributed as dist
import os
from HStest import HSTestData
from HStrain import HSTrainingData
import torch.nn.functional as func
from SSPSR import HybridLoss

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class EDSR(nn.Module):
    def __init__(self, n_resblocks,n_feats,n_colors,res_scale, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3 
        scale = 2
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        # self.sub_mean = common.MeanShift(1)
        # self.add_mean = common.MeanShift(1, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

# model = EDSR(n_resblocks=16 ,n_feats=64 ,n_colors=102 ,res_scale=1)
# input = torch.randn((1,102,32,32))
# input2 = torch.randn((1,31,128,128))
# out = model(input)
# print((out.shape))

EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4
high_sr = 128
low_sr = high_sr / 4

if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Buliding model
    print('===> Building model')
    load = False
    if load:
        model = torch.load('./weight/EDSR_3_Chi.pth')
        print("模型读取成功, 进行fine tune 训练！！！")
    else:
        model = EDSR(n_resblocks=16 ,n_feats=128 ,n_colors=31 ,res_scale=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)

    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)


    # train_set = TrainsetFromFolder('../Harvard_4_train/') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    # train_set = TrainsetFromFolder('../train/Cave/4/')
    # train_set = TrainsetFromFolder('../train/Chikusei/4/')
    # train_set = TrainsetFromFolder('../train/PaviaC/8/')
    train_set = HSTrainingData(image_dir= '../Harvard_mat/train/', n_scale = 2, augment=True, ch3=False, num_ch=0)
    print(len(train_set))
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)  # 切分训练数据集
    train_loader = DataLoader(dataset=train_set,  batch_size=16, shuffle=True) # 分布式不能进行shuffle

    for epoch in range(100):
        count = 0
        for data in train_loader:
            lr = data['LR'].to(device)
            sr = data['SR'].to(device)
            hr = data['HR'].to(device)

        # for lr,hr in train_loader:
            # lr = lr.to(device)
            # hr = hr.to(device)
            # sr = model_b(lr)
            SR = model(lr)
            # print(SR.shape,lr.shape,hr.shape)


            loss = h_loss(SR, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = count + 1
            print("天哪，这轮训练完成了！第{}个Epoch的第{}轮的损失为：{}".format(epoch, count, loss))

    OUT_DIR = Path('./weight')
    torch.save(model, OUT_DIR.joinpath('EDSR_2_Har.pth'))
