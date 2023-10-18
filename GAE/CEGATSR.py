import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from common import *
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.utils.data import DataLoader
import torch.optim as optim


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

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):        # Conv2d input: [B，C，H，W]. W=((w-k+2p)//s)+1
    if dilation == 1:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size-1) // 2, bias=bias)
    elif dilation == 2:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=2, bias=bias, dilation=dilation)

    else:
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=3, bias=bias, dilation=dilation)


def prosessing_conv(in_channels, out_channels, kernel_size, stride, bias=True):      # W=((w-k+2p)//s)+1. [C,H,W]->[C,H/s,W/s]: k-2p=s.
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=bias)
    # s=2,h=w=8;s=3,h=w=6

def transpose_conv(in_channels, out_channels, kernel_size, stride, bias=True):       # [C,H/s,W/s]->[C,H,W]
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1, bias=bias)

    # output = (input-1)*stride + outputpadding - 2*padding + kernelsize
    # 2p-op=k-s
    # s=2,p=1,outp=1,h=w=16, 2*pading-outpadding=1
    # s=3,p=1,outp=0,h=w=16, 2*pading-outpadding=2


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    input: (B,N,C_in)
    output: (B,N,C_out)
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):     # in_features=out_features。
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # definite trainable parameter, W and a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))            # build matrix, size is (input_channel, output_channel)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)    # xavier
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # leakyrelu
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        """
        h: [B,N, in_features]  in_features
        adj: graph adjacent  [N, N], 0 or 1
        """
        # [B_batch,N_nodes,C_channels]
        B, N, C = x.size()
        h = torch.matmul(x, self.W)                                                     # [B,N,C], [B, N, out_features]
        # print("h.shape:",h.shape)       # torch.Size([16, 36, 64])
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N*N, self.out_features), h.repeat(1, N, 1)], dim=2).view(-1, N, N, 2*self.out_features)  # [B, N, N, 2*out_features]
        # print("a_input.shape:",a_input.shape)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))                    # [B, N, N, 1] => [B, N, N], eij∈(0-1)
        zero_vec = -1e12 * torch.ones_like(e)                                           # -endless
        # print("adj is cuda:", adj.is_cuda)     # return false or true
        attention = torch.where(adj>0, e, zero_vec)                                     # [B, N, N]
        # if the element in adj > 0，there is a connection between the 2 nodes，e remains, on the contrary, set mask as negative endless
        attention = F.softmax(attention, dim=2)                                         # [B,N,N]
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h)                                               # [B,N,N]*[B,N,out_features]-> [B,N,out_features], Conversion of node information
        # print("h_prime.shape:", h_prime.shape)  # torch.Size([16, 64, 64])

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, n_heads):
        """
        Dense version of GAT.
        n_heads: multi-head, concat
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # definite multi-head
        self.attentions = [GraphAttentionLayer(in_features, out_features, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(out_features * n_heads, out_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        # print("GAT：x.shape:",x.shape)       # torch.Size([16, 64, 128])
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=2)


class CALayer(nn.Module):               # channel attention mechanism
    def __init__(self, in_channels, reduction_rate=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_rate, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_rate, in_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialResBlock(nn.Module):              # spatial attention block
    def __init__(self, conv, in_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(SpatialResBlock, self).__init__()
        m = []
        for i in range(2):                      # Conv - ReLU - Conv
            m.append(conv(in_feats, in_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class SpectralAttentionResBlock(nn.Module):     # spectral attention block
    def __init__(self, conv, in_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(SpectralAttentionResBlock, self).__init__()
        m = []
        for i in range(2):                      # Conv - ReLU - Conv
            m.append(conv(in_feats, in_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(in_feats, 16))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, up_scale, in_feats, bn=False, act=False, bias=True):
        m = []
        if (up_scale & (up_scale - 1)) == 0:
            for _ in range(int(math.log(up_scale, 2))):
                m.append(conv(in_feats, 4 * in_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(in_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(in_feats))

        elif up_scale == 3:
            m.append(conv(in_feats, 9 * in_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(in_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(in_feats))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
class Pre_ProcessLayer_Graph(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride, bias = True):
        super(Pre_ProcessLayer_Graph, self).__init__()
        self.head = prosessing_conv(in_feats, out_feats, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.head(x)
        # print("conv.shape:", x.shape)
        [B, C, H, W] = x.shape
        y = torch.reshape(x, [B, C, H*W])
        N = H*W
        y = y.permute(0,2,1).contiguous()                # [B,C,N]->[B,N,C]
        adj = torch.zeros(B, N, N).cuda()                # adj:[N, N], 1 or 0
        # adj = torch.zeros(B, N, N)
        k = 9
        for b in range(B):
            dist = cdist(y[b,:,:].cpu().detach().numpy(), y[b,:,:].cpu().detach().numpy(), metric='euclidean')
            # dist = dist + sp.eye(dist.shape[0])
            dist = np.where(dist.argsort(1).argsort(1) <= 6, 1, 0)        # k=9 + itself, all = 10, the largest 10 number is 1, rest is 0. 
            dist = torch.from_numpy(dist).type(torch.FloatTensor)
            dist = torch.unsqueeze(dist, 0)
            adj[b,:,:] = dist
        # y = y.permute(0,2,1).contiguous()       # [B,N,C]->[B,C,N]
        return y, adj


class ProcessLayer_Graph(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, stride, bias = True):
        super(ProcessLayer_Graph, self).__init__()
        self.last = transpose_conv(in_feats, out_feats, kernel_size, stride, bias=bias)

    def forward(self, x):
        y = self.last(x)
        return y


class GCN_Unit(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN_Unit, self).__init__()
        kernel_size = 3
        stride = 2
        n_heads = 2
        dropout = 0.6
        alpha = 0.2
        self.head = Pre_ProcessLayer_Graph(in_feats, out_feats, kernel_size, stride, bias=True)
        self.body = GAT(out_feats, out_feats, dropout, alpha, n_heads)
        # self.body = nn.Conv2d(out_feats, out_feats, kernel_size, stride=1, padding=kernel_size // 2, bias=True)
        self.last = ProcessLayer_Graph(out_feats, out_feats, kernel_size, stride, bias=True)

        self.Act = nn.ReLU()

    def forward(self, x):
        y, adj = self.head(x)       # y.shape = torch.Size([16, 64, 32]), adj.shape = torch.Size([16, 64, 64])
        y = self.body(y, adj)       # y.shape = torch.Size([16, 64, 32])
        # y = self.body(y)       # y.shape = torch.Size([16, 64, 32])
        y = y.permute(0,2,1).contiguous()           # [B,N,C]->[B,C,N]
        [B,C,N] = y.shape
        H = int(math.sqrt(N))
        W = int(math.sqrt(N))
        y = torch.reshape(y,[B,C,H,W])
        # print("reshape later:y.shape:", y.shape)     # torch.Size([16, 64, 8, 8])
        y = self.last(y)        # GCN branch channel is "out_feats".
        # print("transconv:y.shape:", y.shape)     # torch.Size([16, 64, 16, 16])
        # pdb.set_trace()
        return y


class CNN_Unit(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size=3):
        super(CNN_Unit, self).__init__()
        self.point_conv = nn.Conv2d(
            in_channels=in_feats,
            out_channels=out_feats,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_feats,
            out_channels=out_feats,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_feats
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_feats)

    def forward(self, x):
        # y = self.point_conv(self.BN(x)
        y = self.point_conv(x)
        y = self.Act1(y)
        y = self.depth_conv(y)
        y = self.Act2(y)

        y = self.point_conv(y)
        y = self.Act1(y)
        y = self.depth_conv(y)
        y = self.Act2(y)
        return y


class GCN_CNN_Unit(nn.Module):          # GCN_CNN_Unit
    def __init__(self, in_feats, out_feats, up_scale, use_tail=True, conv=default_conv):  # up_scale
        super(GCN_CNN_Unit, self).__init__()
        kernel_size = 3
        self.pre = conv(in_feats, out_feats, kernel_size)
        self.head = GCN_Unit(out_feats, out_feats)
        self.body = CNN_Unit(out_feats, out_feats)
        self.last = conv(out_feats, out_feats, kernel_size)
        self.upsample = Upsampler(conv, up_scale, out_feats)
        self.tail = True
        if use_tail:
            self.tail = conv(out_feats, in_feats, kernel_size)

    def forward(self, x):
        # print("unit in_feats:",x.shape)
        y = self.pre(x)
        GCN_result = self.head(y)
        # print("GCN_result.shape:",GCN_result.shape)         # torch.Size([16, 64, 16, 16])
        CNN_result = self.body(y)
        # print("CNN_result.shape:",CNN_result.shape)         # torch.Size([16, 64, 16, 16])
        # pdb.set_trace()
        # y = torch.cat([GCN_result, CNN_result], dim=1)
        y = GCN_result
        y = self.last(y)
        # print("channel compress:", y.shape)     # torch.Size([16, 64, 16, 16])
        y = self.upsample(y)
        # print("upscale:", y.shape)      # torch.Size([16, 16, 32, 32])
        if self.tail is not None:
            y = self.tail(y)
            # print("reconstruct:",y.shape)    # torch.Size([16, 4, 32, 32])cave
        # pdb.set_trace()
        return y


class SSB(nn.Module):                   # SSB
    def __init__(self, in_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSB, self).__init__()
        self.spa = SpatialResBlock(conv, in_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = SpectralAttentionResBlock(conv, in_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))


class SSPN(nn.Module):
    def __init__(self, in_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()

        kernel_size = 3
        m = []

        for i in range(n_blocks):
            m.append(SSB(in_feats, kernel_size, act=act, res_scale=res_scale))

        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        res += x

        return res


class Spatial_Spectral_Unit(nn.Module): # Spatial_Spectral_Unit
    def __init__(self, in_feats, out_feats, n_blocks, act, res_scale, up_scale, use_tail=False, conv=default_conv):
        super(Spatial_Spectral_Unit, self).__init__()
        kernel_size = 3
        self.head = conv(in_feats, out_feats, kernel_size)
        self.body = SSPN(out_feats, n_blocks, act, res_scale)
        self.upsample = Upsampler(conv, up_scale, out_feats)
        self.tail = None

        if use_tail:
            self.tail = conv(out_feats, in_feats, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)

        return y


class CEGATSR(nn.Module):
    def __init__(self, n_subs, n_ovls, in_feats, n_blocks, out_feats, n_scale, res_scale, use_share=True, conv=default_conv):
        super(CEGATSR, self).__init__()
        kernel_size = 3
        self.shared = use_share
        act = nn.ReLU(True)

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((in_feats - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > in_feats:
                end_ind = in_feats
                sta_ind = in_feats - n_subs
            self.start_idx.append(sta_ind)      # [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120]
            self.end_idx.append(end_ind)        # [8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92, 98, 104, 110, 116, 122, 128]

        if self.shared:
            self.branch = GCN_CNN_Unit(n_subs, out_feats, up_scale=n_scale//2, use_tail=True, conv=default_conv)
            # self.branch = GCN_CNN_Unit(n_subs, out_feats, up_scale=3, use_tail=True, conv=default_conv)
            # self.branch = GCN_CNN_Unit(n_subs, out_feats, use_tail=True, conv=default_conv)
            # up_scale=n_scale//2 means that we upsample the LR input n_scale//2 at the branch network, and then conduct 2 times upsampleing at the global network
        else:
            self.branch = nn.ModuleList
            for i in range(self.G):
                self.branch.append(GCN_CNN_Unit(n_subs, out_feats, up_scale=n_scale//2, use_tail=True, conv=default_conv))
                # self.branch.append(GCN_CNN_Unit(n_subs, out_feats, use_tail=True, conv=default_conv))

        self.trunk = Spatial_Spectral_Unit(in_feats, out_feats, n_blocks, act, res_scale, up_scale=2, use_tail=False, conv=default_conv)
        self.skip_conv = conv(in_feats, out_feats, kernel_size)
        self.final = conv(out_feats, in_feats, kernel_size)
        self.sca = n_scale//2
        # self.sca = 3

    def forward(self, x, lms):
        b, c, h, w = x.shape

        # Initialize intermediate “result”, which is upsampled with n_scale//2 times
        y = torch.zeros(b, c, self.sca * h, self.sca * w).cuda()
        # y = torch.zeros(b, c, self.sca * h, self.sca * w)

        channel_counter = torch.zeros(c).cuda()
        # channel_counter = torch.zeros(c)

        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            if self.shared:
                xi = self.branch(xi)
            else:
                xi = self .branch[g](xi)
                print("xi.shape:", xi.shape)

            y[:, sta_ind:end_ind, :, :] += xi
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        # pdb.set_trace()
        y = self.trunk(y)
        y = y + self.skip_conv(lms)
        y = self.final(y)

        return y

# net = CEGATSR(n_subs=8, n_ovls=2, in_feats=31, n_blocks=3, out_feats=128, n_scale=4, res_scale=0.1, use_share=True, conv=default_conv, ).cuda()
# input = torch.randn((1,31,32,32)).cuda()
# input2 = torch.randn((1,31,128,128)).cuda()
# out = net(input,input2)
# print(out.shape)

EPOCHS = 5
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
        model = torch.load('./weight/CEGAT_3_Chi.pth')
        print("模型读取成功, 进行fine tune 训练！！！")
    else:
        model = CEGATSR(n_subs=8, n_ovls=2, in_feats=128, n_blocks=3, out_feats=64, n_scale=3, res_scale=0.1, use_share=True, conv=default_conv).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)

    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)


    # train_set = TrainsetFromFolder('../Harvard_4_train/') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    # train_set = TrainsetFromFolder('../train/Cave/4/')
    # train_set = TrainsetFromFolder('../train/Chikusei/4/')
    # train_set = TrainsetFromFolder('../train/PaviaC/4/')
    train_set = HSTrainingData(image_dir= '../Chikusei_mat/train/', n_scale = 3, augment=True, ch3=False, num_ch=0)
    print(len(train_set))
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)  # 切分训练数据集
    train_loader = DataLoader(dataset=train_set,  batch_size=8, shuffle=True) # 分布式不能进行shuffle
    model_b = Bicubic()
    for epoch in range(25):
        count = 0
        for data in train_loader:
            lr = data['LR'].to(device)
            sr = data['SR'].to(device)
            hr = data['HR'].to(device)

        # for lr,hr in train_loader:
        #     lr = lr.to(device)
        #     hr = hr.to(device)
        #     sr = model_b(lr)
            SR = model(lr,sr)
            # print(SR.shape,lr.shape,hr.shape)


            loss = h_loss(SR, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = count + 1
            print("天哪，这轮训练完成了！第{}个Epoch的第{}轮的损失为：{}".format(epoch, count, loss))

    OUT_DIR = Path('./weight')
    torch.save(model, OUT_DIR.joinpath('CEGAT_3_Chi.pth'))