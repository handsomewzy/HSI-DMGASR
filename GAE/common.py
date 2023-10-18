import torch.nn as nn
import math
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                # m.append(nn.GroupNorm(n_feats//4,n_feats))
            if i == 0:
                m.append(act)
                # m.append(quantize.Quantization(bit=2, qq_bit=8, finetune=False))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    if dilation == 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), bias=bias)
    elif dilation == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=2, bias=bias, dilation=dilation)

    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=3, bias=bias, dilation=dilation)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
                # m.append(nn.GroupNorm(n_feats//4,n_feats))
            if i == 0:
                m.append(act)
                # m.append(quantize.Quantization(bit=2, qq_bit=8, finetune=False))

        m.append(CALayer(n_feats, 3))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ResBlock_DAQ(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,a_bit=2, w_bit=2, qq_bit=8,finetune=False, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock_DAQ, self).__init__()
        self.a_bit = a_bit 
        self.w_bit = w_bit
        
        self.quant1 = quantize.Quantization(bit=self.a_bit, qq_bit=qq_bit, finetune=finetune)
        self.quant2 = quantize.Quantization(bit=self.a_bit, qq_bit=qq_bit, finetune=finetune)

        # convolution
        if w_bit ==32:
            self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
            self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        else:
            self.conv1 = quantize.Conv2d_Q(n_feats, n_feats, kernel_size, stride=1, padding=1, bias=bias, dilation=1, groups=1, w_bit=w_bit, finetune=finetune)
            self.conv2 = quantize.Conv2d_Q(n_feats, n_feats, kernel_size, stride=1, padding=1, bias=bias, dilation=1, groups=1, w_bit=w_bit, finetune=finetune)

        self.act = act
        self.res_scale = res_scale

    def forward(self, x):
        if self.a_bit!=32:
            out= self.quant1(x)
        else:
            out=x    
        out = self.conv1(out)
        out1 = self.act(out)
        if self.a_bit!=32:
            out1= self.quant2(out1)
        res = self.conv2(out1)
        res = res.mul(self.res_scale)
        res += x
        return res


class ResAttentionBlock_DAQ(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,a_bit=2, w_bit=2, qq_bit=8,finetune=False, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock_DAQ, self).__init__()
        self.a_bit = a_bit 
        self.w_bit = w_bit
        
        self.quant1 = quantize.Quantization(bit=self.a_bit, qq_bit=qq_bit, finetune=finetune)
        self.quant2 = quantize.Quantization(bit=self.a_bit, qq_bit=qq_bit, finetune=finetune)

        # convolution
        if w_bit ==32:
            self.conv1 = conv(n_feats, n_feats, kernel_size=1, bias=bias)
            self.conv2 = conv(n_feats, n_feats, kernel_size=1, bias=bias)
        else:
            self.conv1 = quantize.Conv2d_Q(n_feats, n_feats, kernel_size=1, stride=1, padding=0, bias=bias, dilation=1, groups=1, w_bit=w_bit, finetune=finetune)
            self.conv2 = quantize.Conv2d_Q(n_feats, n_feats, kernel_size=1, stride=1, padding=0, bias=bias, dilation=1, groups=1, w_bit=w_bit, finetune=finetune)

        self.act = act
        self.attn = CALayer(n_feats, 16)
        self.res_scale = res_scale

    def forward(self, x):
        if self.a_bit!=32:
            out= self.quant1(x)
        else:
            out=x    
        # print(out.shape)
        out = self.conv1(out)
        # print(out.shape)
        out1 = self.act(out)
        if self.a_bit!=32:
            out1= self.quant2(out1)
        res = self.conv2(out1)
        res = self.attn(res)
        res = res.mul(self.res_scale)
        res += x
        return res


