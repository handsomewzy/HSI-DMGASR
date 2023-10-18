#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   光滑图片
@File    :   process.py
@Time    :   2021/02/06 11:49:27
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2021/02/06 11:49:27
'''

from PIL import Image
import numpy as np
import torch
from torch.nn.functional import interpolate
from utils import calc_psnr, SAM

bic_path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/data/bic_icvl_img7.png'
hr_path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/data/hr_icvl_img7.png'
gan_path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/data/process_icvl_img7.png'

paths = [gan_path,bic_path]

hr = Image.open(hr_path)
hr = np.array(hr)
hr = hr.astype(np.float32)
hr = torch.from_numpy(hr)

for path in paths:

    img = Image.open(path)
    img = np.array(img)
    img = img.astype(np.float32)
    img = torch.from_numpy(img)  
    # print((hr-img).sum())
    print((calc_psnr(hr,img),SAM(hr,img)))