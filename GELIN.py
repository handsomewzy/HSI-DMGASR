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
# from icvl_data import LoadData
# from utils import SAM, PSNR_GPU, get_paths ,TrainsetFromFolder
import sewar
# import MCNet
from pathlib import Path
from torch.nn.functional import interpolate
import torch.distributed as dist
import os
from HStest import HSTestData
from HStrain import HSTrainingData
import torch.nn.functional as func

def EzConv(in_channel,out_channel,kernel_size):
    return nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=kernel_size,stride=1,padding=kernel_size//2,bias=True)

class Upsample(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True, conv=EzConv):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsample, self).__init__(*m)
        
class CA(nn.Module):
    '''CA is channel attention'''
    def __init__(self,n_feats,kernel_size=3,bias=True, bn=False, act=nn.ReLU(True),res_scale=1,conv=EzConv,reduction=16):

        super(CA, self).__init__()
        
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.body(x)
        CA = self.conv_du(y)
        CA = torch.mul(y, CA)
        x = CA + x
        return x
                 
class SCconv(nn.Module):
    def __init__(self,n_feats,kernel_size,pooling_r):
        super(SCconv,self).__init__()
        self.half_feats = n_feats//2
        self.f1 = nn.Sequential(
            nn.Conv2d(n_feats//2,n_feats//2,kernel_size,padding=kernel_size//2),
            nn.ReLU(True)
        )
        self.f2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r,stride=pooling_r),
            nn.Conv2d(n_feats//2,n_feats//2,kernel_size,padding=kernel_size//2),
        )
        self.f3 = nn.Conv2d(n_feats//2,n_feats//2,kernel_size,padding=kernel_size//2)
        self.f4 = nn.Sequential(
            nn.Conv2d(n_feats//2,n_feats//2,kernel_size,padding=kernel_size//2),
            nn.ReLU(True)
        )
    
    def forward(self,x):
        x1 = x[:, 0:self.half_feats, :, :]
        x2 = x[:, self.half_feats:, :, :]
        identity_x1 = x1
        out_x1 = torch.sigmoid(torch.add(identity_x1,F.interpolate(self.f2(x1),identity_x1.size()[2:])))
        out_x1 = torch.mul(self.f3(x1),out_x1)
        out_x1 = self.f4(out_x1)
        out_x2 = self.f1(x2)
        out = torch.cat([out_x1,out_x2],dim=1)
        return out

class SSELB(nn.Module):
    def __init__(self,n_feats,kernel_size,pooling_r):
        super(SSELB,self).__init__()
        self.body = nn.Sequential(
            SCconv(n_feats,kernel_size,pooling_r),
            CA(n_feats),
        )

    def forward(self,x):
        res = self.body(x)
        return res + x
        
        
class NGIM(nn.Module):
    def __init__(self,n_feats,scale):
        super(NGIM,self).__init__()
          
        if scale == 4:
            self.TrunkUp = nn.Sequential(
                nn.ConvTranspose2d(n_feats,n_feats,kernel_size=8,stride=4,padding=2),
                nn.PReLU(n_feats)
            )
            self.MultiUp = nn.Sequential(
                nn.Conv2d(n_feats*3,n_feats//2,kernel_size=3,padding=1),
                nn.Conv2d(n_feats//2,n_feats,kernel_size=3,padding=1),
                nn.ConvTranspose2d(n_feats,n_feats,kernel_size=8,stride=4,padding=2),
                nn.PReLU(n_feats)
            )
        elif scale == 8:
            self.TrunkUp = nn.Sequential(
                nn.ConvTranspose2d(n_feats,n_feats,kernel_size=12,stride=8,padding=2),
                nn.PReLU(n_feats)
            )
            self.MultiUp = nn.Sequential(
                nn.Conv2d(n_feats*3,n_feats//2,kernel_size=3,padding=1),
                nn.Conv2d(n_feats//2,n_feats,kernel_size=3,padding=1),
                nn.ConvTranspose2d(n_feats,n_feats,kernel_size=12,stride=8,padding=2),
                nn.PReLU(n_feats)
            )            
        
        self.error_resblock = nn.Sequential(
            nn.Conv2d(n_feats,n_feats,kernel_size=3,padding=1),
        )
    def forward(self,xl,xi,xn):
        
        h1 = self.TrunkUp(xi)
        h2 = self.MultiUp(torch.cat([xl,xi,xn],dim=1))
        e = h2 - h1
        e = self.error_resblock(e)
        h1 = h1 + e
        return h1

class SSELM(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks,pooling_r):
        super(SSELM, self).__init__()
        kernel_size = 3
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size,padding=kernel_size//2)
        body = []
        for i in range(n_blocks):
            body.append(SSELB(n_feats,kernel_size,pooling_r))

        self.body = nn.Sequential(*body)

        #self.recon = nn.Conv2d(n_feats, n_colors, kernel_size=3,padding=kernel_size//2)

    def forward(self, x):
        x = self.head(x)
        
        y = self.body(x) + x

        return y
   
class GELIN(nn.Module):
    def __init__(self,n_feats,n_colors,kernel_size,pooling_r,n_subs, n_ovls,blocks,scale):
        super(GELIN,self).__init__()

        # calculate the group number (the number of branch networks)
        # 向上取整计算组的数量 G 
        self.n_feats = n_feats
        self.n_subs = n_subs
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []        
        self.scale = scale
        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            # 把每一组的开始 idx 与结束 idx 存入 list
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        self.branch = SSELM(n_subs,n_feats,blocks,pooling_r)
        

        self.branch_up = NGIM(n_feats,scale)
        self.branch_recon = nn.Conv2d(n_feats, n_subs, kernel_size=3,padding=kernel_size//2)

    def forward(self,x,lms):
    
        b, c, h, w = x.shape
        m = []
        y = torch.zeros(b, c,  h*self.scale,  w*self.scale).cuda()
        # y = torch.zeros(b, c, h * self.scale, w * self.scale)

        channel_counter = torch.zeros(c).cuda()
        # channel_counter = torch.zeros(c)

        for g in range(self.G):

            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            xi = self.branch(xi)
            m.append(xi)
            
        for g in range(self.G):

            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            if g==0:
                xl = m[self.G-1]
                xi = m[g]
                xn = m[g+1]
            elif g==self.G-1:
                xl = m[g-1]
                xi = m[g]
                xn = m[0]
            else:
                xl = m[g-1]
                xi = m[g]
                xn = m[g+1]  

            xi = self.branch_up(xl,xi,xn)
            xi = self.branch_recon(xi)
            y[:, sta_ind:end_ind, :, :] += xi
            # 用 channel_counter 记录某一个位置被加了几次，然后再除这个数字取平均
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        y = y + lms
        return y

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0
        
class spatial_grad(nn.Module):
    def __init__(self,weight):    
        super(spatial_grad, self).__init__()
        self.get_grad = Get_gradient_nopadding()
        self.fidelity = torch.nn.L1Loss()
        self.weight = weight
    
    def forward(self,y,gt):
        y_grad = self.get_grad(y)
        gt_grad = self.get_grad(gt)
        return self.weight * self.fidelity(y_grad,gt_grad)
        
    
class MixLoss(torch.nn.Module):
    def __init__(self):
        super(MixLoss,self).__init__()
        self.fidelity = torch.nn.L1Loss()
        self.grad_loss = spatial_grad(weight=0.5)
        
    def forward(self,y,gt):
        loss = self.fidelity(y, gt)
        loss_grad = self.grad_loss(y,gt)
        return loss+loss_grad




class HybridLoss(torch.nn.Module):
    def __init__(self, lamd=1e-1, spatial_tv=False, spectral_tv=False):
        super(HybridLoss, self).__init__()
        self.lamd = lamd
        self.use_spatial_TV = spatial_tv
        self.use_spectral_TV = spectral_tv
        self.fidelity = torch.nn.L1Loss()
        self.spatial = TVLoss(weight=1e-3)
        self.spectral = TVLossSpectral(weight=1e-3)

    def forward(self, y, gt):
        loss = self.fidelity(y, gt)
        spatial_TV = 0.0
        spectral_TV = 0.0
        if self.use_spatial_TV:
            spatial_TV = self.spatial(y)
        if self.use_spectral_TV:
            spectral_TV = self.spectral(y)
        total_loss = loss + spatial_TV + spectral_TV
        return total_loss

class Aux_loss(torch.nn.Module):
    def __init__(self):
        super(Aux_loss, self).__init__()
        self.L1_loss = torch.nn.L1Loss()
    def forward(self, y_aux, gt):
        loss = 0.0
        for y in y_aux:
            loss = loss + self.L1_loss(y, gt)
        return loss / len(y_aux)


# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        # h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :]).sum()
        # w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1]).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class TVLossSpectral(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TVLossSpectral, self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        c_x = x.size()[1]
        count_c = self._tensor_size(x[:, 1:, :, :])
        # c_tv = torch.abs((x[:, 1:, :, :] - x[:, :c_x - 1, :, :])).sum()
        c_tv = torch.pow((x[:, 1:, :, :] - x[:, :c_x - 1, :, :]), 2).sum()
        return self.TVLoss_weight * 2 * (c_tv / count_c) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
        
def cal_gradient_c(x):
  c_x = x.size(1)
  g = x[:, 1:, 1:, 1:] - x[:, :c_x - 1, 1:, 1:]
  return g
  
def cal_gradient_x(x):
  c_x = x.size(2)
  g = x[:, 1:, 1:, 1:] - x[:, 1:, :c_x - 1, 1:]
  return g  
  
def cal_gradient_y(x):
  c_x = x.size(3)
  g = x[:, 1:, 1:, 1:] - x[:, 1:, 1:, :c_x - 1]
  return g

def cal_gradient(inp):
  x = cal_gradient_x(inp)
  y = cal_gradient_y(inp)
  c = cal_gradient_c(inp)
  g = torch.sqrt(torch.pow(x, 2) + torch.pow(y,2) + torch.pow(c,2)+1e-6)
  return g

def cal_sam(Itrue, Ifake):
  esp = 1e-6
  # element-wise product
  # torch.sum(dim=1) 沿通道求和
  # [B C H W] * [B C H W] --> [B C H W]  Itrue*Ifake
  # [B 1 H W] InnerPro(keepdim)
  
  InnerPro = torch.sum(Itrue*Ifake,1,keepdim=True)
  #print('InnerPro')
  #print(InnerPro.shape)
  # 沿通道求范数
  # len1  len2  [B 1 H W] (keepdim)
  len1 = torch.norm(Itrue, p=2,dim=1,keepdim=True)
  len2 = torch.norm(Ifake, p=2,dim=1,keepdim=True)
  #print('len1')
  #print(len1.shape)
  
  divisor = len1*len2
  mask = torch.eq(divisor,0)
  divisor = divisor + (mask.float())*esp 
  cosA = torch.sum(InnerPro/divisor,1).clamp(-1+esp, 1-esp)
  #print(cosA.shape)
  sam = torch.acos(cosA)
  sam = torch.mean(sam) / np.pi
  return sam

class HLoss(torch.nn.Module):
    def __init__(self, la1,la2,sam=True, gra=True):
        super(HLoss,self).__init__()
        self.lamd1 = la1
        self.lamd2 = la2
        self.sam = sam
        self.gra = gra
        
        self.fidelity = torch.nn.L1Loss()
        self.gra = torch.nn.L1Loss()
        
    def forward(self, y, gt):
        loss1 = self.fidelity(y, gt)
        loss2 = self.lamd1*cal_sam(y, gt)
        loss3 = self.lamd2*self.gra(cal_gradient(y),cal_gradient(gt))
        loss = loss1+loss2+loss3
        return loss  
        
        


# net = GELIN(n_feats=16,n_colors=31,kernel_size=3,pooling_r=2,n_subs=8,n_ovls=2,blocks=8,scale=4)
# input = torch.randn((1,31,32,32))
# input2 = torch.randn((1,31,128,128))
# out = net(input,input2)
# print((out.shape))


EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-4
high_sr = 128
low_sr = high_sr / 4

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Buliding model
    print('===> Building model')
    load = True
    if load:
        model = torch.load('./weight/GELIN_4_Pav.pth')
        print("模型读取成功, 进行fine tune 训练！！！")
    else:
        model = GELIN(n_feats=256,n_colors=102,kernel_size=3,pooling_r=2,n_subs=8,n_ovls=2,blocks=6,scale=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)




    # train_set = TrainsetFromFolder('../Harvard_4_train/') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    # train_set = TrainsetFromFolder('../train/CAVE/4/')
    # train_set = TrainsetFromFolder('../train/Chikusei/4/')
    # train_set = TrainsetFromFolder('../train/PaviaC/4/')
    train_set = HSTrainingData(image_dir= '../PaviaC_mat/train/', n_scale = 4, augment=True, ch3=False, num_ch=0)
    print(len(train_set))
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)  # 切分训练数据集
    train_loader = DataLoader(dataset=train_set,  batch_size=16, shuffle=True) # 分布式不能进行shuffle

    for epoch in range(EPOCHS):
        count = 0
        for data in train_loader:
            # bs 31 36 36  / bs 31 144 144
            # lr = lr.reshape((lr.shape[0], 1, lr.shape[1], lr.shape[2], lr.shape[3]))
            lr = data['LR'].to(device)
            sr = data['SR'].to(device)
            # hr = hr.reshape((hr.shape[0], 1, hr.shape[1], hr.shape[2], hr.shape[3]))
            hr = data['HR'].to(device)
            # print(lr.shape, hr.shape)

            SR = model(lr,sr)
            # print(SR.shape,lr.shape,hr.shape)
            # 设定对应的损失函数
            loss_func = HLoss(0.3, 0.1)

            loss = loss_func(SR, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = count + 1
            print("天哪，这轮训练完成了！第{}个Epoch的第{}轮的损失为：{}".format(epoch, count, loss))

    OUT_DIR = Path('./weight')
    torch.save(model, OUT_DIR.joinpath('GELIN_4_Pav.pth'))