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

class Bicubic(nn.Module):
    def __init__(self, scale = 4):
        super(Bicubic, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = interpolate(
            x,
            scale_factor=self.scale,  # 这个与具体的超分辨比例有关，这个是全局skip时候，对初始图像进行上采样，一般设置为2 3 4
            mode='bicubic',
            align_corners=True
        )
        return x


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False):
        super(Conv, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size-1)//2
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)

class Conv_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=True):
        super(Conv_ReLU, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size-1)//2
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False):
        super(Conv_BN, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size-1)//2
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, bias=False):
        super(Conv_BN_ReLU, self).__init__()
        if padding == None:
            if stride == 1:
                padding = (kernel_size-1)//2
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Denoise_Block_BN(nn.Module):
    def __init__(self, input_chnl, output_chnl=None,inner_chnl=64, padding=1, num_of_layers=15, groups=1):
        super(Denoise_Block_BN, self).__init__()
        kernel_size = 3
        num_chnl = inner_chnl
        if output_chnl is None:
            output_chnl = input_chnl
        self.conv_input = nn.Sequential(Conv_BN_ReLU(in_channels=input_chnl, out_channels=num_chnl, kernel_size=kernel_size, padding=padding, groups=groups))
        self.conv_layers = self._make_layers(Conv_BN_ReLU,num_chnl=num_chnl, kernel_size=kernel_size, padding=padding, num_of_layers=num_of_layers-2, groups=groups)
        self.conv_out = nn.Sequential(Conv_BN_ReLU(in_channels=num_chnl, out_channels=output_chnl, kernel_size=kernel_size, padding=padding, groups=groups))

    def _make_layers(self, block, num_chnl, kernel_size, padding, num_of_layers, groups=1):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=num_chnl, out_channels=num_chnl, kernel_size=kernel_size, padding=padding, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_out(self.conv_layers(self.conv_input(x)))

class DnCNN(nn.Module):
    def __init__(self, input_chnl, groups=1):
        super(DnCNN, self).__init__()
        kernel_size = 3
        num_chnl = 64
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=input_chnl, out_channels=num_chnl,
                              kernel_size=kernel_size, stride=1, padding=1,
                              groups=1, bias=True),
                              nn.ReLU(inplace=True))
        self.dn_block = self._make_layers(Conv_BN_ReLU, kernel_size, num_chnl, num_of_layers=15, bias=False)
        # self.output = nn.Sequential(nn.Conv2d(in_channels=num_chnl, out_channels=input_chnl,
        #                                       kernel_size=kernel_size, stride=1, padding=1,
        #                                       groups=groups, bias=True),
        #                             nn.BatchNorm2d(input_chnl))
        self.output = nn.Conv2d(in_channels=num_chnl, out_channels=input_chnl,
                                              kernel_size=kernel_size, stride=1, padding=1,
                                              groups=groups, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block,  kernel_size, num_chnl, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=num_chnl, out_channels=num_chnl, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        # x = self.nl(x)
        x = self.dn_block(x)
        return self.output(x) #+ residual

class GDRNN(nn.Module):
    def __init__(self, input_chnl_hsi, group=1):
        super(GDRNN, self).__init__()
        num_chnl = 128
        self.input = nn.Conv2d(in_channels=input_chnl_hsi, out_channels=num_chnl, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.conv1 = nn.Conv2d(in_channels=num_chnl, out_channels=num_chnl, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
        self.conv2 = nn.Conv2d(in_channels=num_chnl, out_channels=num_chnl, kernel_size=3, stride=1, padding=1, bias=False, groups=group)
        self.output = nn.Conv2d(in_channels=num_chnl, out_channels=input_chnl_hsi, kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        # rnd = np.array([j for j in range(self.conv1.out_channels)])
        # for i in range(self.conv1.groups):
        #     rnd[i * (self.conv1.out_channels // self.conv1.groups):(i + 1) * (
        #         self.conv1.out_channels // self.conv1.groups)] = \
        #         np.arange(i, self.conv1.out_channels, self.conv1.groups)
        for _ in range(9):
            out = self.conv1(self.relu(out))
            # out.data = out.data[:, rnd, :, :]
            out = self.conv2(self.relu(out))
            out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out


class myloss_spe(nn.Module):
    def __init__(self, N, lamd=1e-1, mse_lamd=1, epoch=None):
        super(myloss_spe, self).__init__()
        self.N = N
        self.lamd = lamd
        self.mse_lamd = mse_lamd
        self.epoch = epoch
        return

    def forward(self, res, label):
        # print(res.shape,label.shape)
        mse = func.mse_loss(res, label, size_average=False)
        # mse = func.l1_loss(res, label, size_average=False)
        loss = mse / (self.N * 2)
        esp = 1e-12
        H = label.size()[2]
        W = label.size()[3]
        Itrue = label.clone()
        Ifake = res.clone()
        nom = torch.mul(Itrue, Ifake).sum(dim=1)
        denominator = Itrue.norm(p=2, dim=1, keepdim=True).clamp(min=esp) * \
                      Ifake.norm(p=2, dim=1, keepdim=True).clamp(min=esp)
        denominator = denominator.squeeze()
        # sam = -np.pi/2*torch.div(nom, denominator) + np.pi/2
        sam = torch.div(nom, denominator).acos()
        sam[sam != sam] = 0
        sam_sum = torch.sum(sam) / (self.N * H * W)
        if self.epoch is None:
            total_loss = self.mse_lamd * loss + self.lamd * sam_sum
        else:
            norm = self.mse_lamd + self.lamd * 0.1 ** (self.epoch // 10)
            lamd_sam = self.lamd * 0.1 ** (self.epoch // 10)
            total_loss = self.mse_lamd / norm * loss + lamd_sam / norm * sam_sum
        return total_loss


EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4
high_sr = 128
low_sr = high_sr / 4

if __name__ == "__main__":
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Buliding model
    print('===> Building model')
    load = True
    if load:
        model = torch.load('./weight/GDRNN_4_Har.pth')
        print("模型读取成功, 进行fine tune 训练！！！")
    else:
        model = GDRNN(input_chnl_hsi=31)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08)




    # train_set = TrainsetFromFolder('../Harvard_4_train/') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    # train_set = TrainsetFromFolder('../train/Cave/4/')
    # train_set = TrainsetFromFolder('../train/Chikusei/4/')
    # train_set = TrainsetFromFolder('../train/PaviaC/8/')
    train_set = HSTrainingData(image_dir= '../Harvard_mat/train/', n_scale = 4, augment=True, ch3=False, num_ch=0)
    print(len(train_set))
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)  # 切分训练数据集
    train_loader = DataLoader(dataset=train_set,  batch_size=16, shuffle=True) # 分布式不能进行shuffle
    model_b = Bicubic(scale=8)
    for epoch in range(50):
        count = 0
        for data in train_loader:
            lr = data['LR'].to(device)
            hr = data['HR'].to(device)
            lms = data['SR'].to(device)
        # for lr,hr in train_loader:
        #     lr = lr.to(device)
        #     hr = hr.to(device)
        #     lr = model_b(lr)
            # print(lr.shape, hr.shape)

            SR = model(lms)
            # print(SR.shape,lr.shape,hr.shape)
            # 设定对应的损失函数
            loss_func = myloss_spe(hr.data.shape[0])

            loss = loss_func(SR, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = count + 1
            print("天哪，这轮训练完成了！第{}个Epoch的第{}轮的损失为：{}".format(epoch, count, loss))

    OUT_DIR = Path('./weight')
    torch.save(model, OUT_DIR.joinpath('GDRNN_4_Har.pth'))
