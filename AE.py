from common import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# from icvl_data import LoadData
# from utils import SAM_torch, PSNR_GPU, get_paths, TrainsetFromFolder, SAM
import sewar
# import MCNet
from pathlib import Path
from torch.nn.functional import interpolate
import torchvision.models as models
import numpy as np
# from SSPSR import HybridLoss
from GELIN import HLoss
from unet import UNet

import torch.nn.functional as F
import torch.nn as nn
from HStest import HSTestData
from HStrain import HSTrainingData

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Codebook(nn.Module):
    def __init__(self, num_codebook_vectors, latent_dim, beta=0.25):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(33, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 33, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        # 编码
        x = self.encoder(x)
        
        # 改变 z 的大小
        z = x[:, :3, :, :]
        
        # 解码
        x = self.decoder(z)
        return x,z


def random_mask(data, p=0.2):
    """\n    随机将数据中一定比例的像素置为0\n    :param data: 输入数据，尺寸为(bs, 31, 128, 128)\n    :param p: 置0的比例\n    :return: mask后的数据\n    """
    mask = torch.rand(data.size()) > p
    mask = mask.to(data.device)
    return data * mask.float()


class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))

class SSB_DAQ(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSB_DAQ, self).__init__()
        self.spa = ResBlock_DAQ(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock_DAQ(conv, n_feats, 1, act=act, res_scale=res_scale)       

    def forward(self, x):
        return self.spc(self.spa(x))

class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()

        kernel_size = 3
        m = []

        for i in range(n_blocks):
    # ---------------------------------------------------------------------------------------------
    #                                 是否使用DAQ策略的设置！！！！
    # ---------------------------------------------------------------------------------------------
            m.append(SSB(n_feats, kernel_size, act=act, res_scale=res_scale))
            # m.append(SSB_DAQ(n_feats, kernel_size, act=act, res_scale=res_scale))

        self.net = nn.Sequential(*m)

    def forward(self, x):
        res = self.net(x)
        # res += x
        res = res + x

        return res


# a single branch of proposed SSPSR
class BranchUnit(nn.Module):
    def __init__(self, n_colors, n_feats, n_blocks, act, res_scale, up_scale, use_tail=True, conv=default_conv):
        super(BranchUnit, self).__init__()
        kernel_size = 3
        # self.head = conv(n_colors, n_feats, kernel_size)
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size=3, padding=1)
        self.body = SSPN(n_feats, n_blocks, act, res_scale)
        self.upsample = Upsampler(conv, up_scale, n_feats)
        self.tail = None

        if use_tail:
            self.tail = conv(n_feats, n_colors, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)

        return y


class Encoder(nn.Module):
    def __init__(self, input_channel, out_channel,n_feats=128):
        super(Encoder, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        # self.model = nn.Sequential(
        #     SSPN(n_feats=self.input_channel, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel, self.input_channel // 3, 1),
        #     # SSB(n_feats=self.input_channel//3,kernel_size=1,act=nn.LeakyReLU(),res_scale=0.1),
        #     SSPN(n_feats=self.input_channel // 3, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel // 3, self.input_channel // 6, 1),
        #     SSPN(n_feats=self.input_channel // 6, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel // 6, self.out_channel, 1),
        # )

        # self.model2 = nn.Sequential(
        #     SSPN(n_feats=self.input_channel, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel, self.input_channel // 2, 1),
        #     # SSB(n_feats=self.input_channel//3,kernel_size=1,act=nn.LeakyReLU(),res_scale=0.1),
        #     SSPN(n_feats=self.input_channel // 2, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel // 2, self.out_channel, 1),
        # )
        

        self.branch = BranchUnit(input_channel, n_feats=n_feats, n_blocks=3, act=nn.LeakyReLU(), res_scale=0.1,use_tail=False, up_scale=1,conv=default_conv)
        self.final = nn.Conv2d(n_feats, out_channel, kernel_size=3, padding=1) 

    def forward(self, x):
        x = self.branch(x)
        x = self.final(x)
        # x = self.model2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_channel, out_channel, n_feats=128):
        super(Decoder, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        # self.model = nn.Sequential(
        #     SSPN(n_feats=self.input_channel, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel, self.input_channel * 3, 1),
        #     # SSB(n_feats=self.input_channel//3,kernel_size=1,act=nn.LeakyReLU(),res_scale=0.1),
        #     SSPN(n_feats=self.input_channel * 3, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel * 3, self.input_channel * 9, 1),
        #     SSPN(n_feats=self.input_channel  * 9, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel * 9, self.out_channel, 1),
        # )

        # self.model2 = nn.Sequential(
        #     SSPN(n_feats=self.input_channel, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel, self.input_channel * 2, 1),
        #     # SSB(n_feats=self.input_channel//3,kernel_size=1,act=nn.LeakyReLU(),res_scale=0.1),
        #     SSPN(n_feats=self.input_channel * 2, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
        #     nn.Conv2d(self.input_channel * 2, self.out_channel, 1),
        # )

        self.branch = BranchUnit(input_channel, n_feats=n_feats, n_blocks=3, act=nn.LeakyReLU(), res_scale=0.1, up_scale=1,conv=default_conv,use_tail=False)
        self.final = nn.Conv2d(n_feats, out_channel, kernel_size=3, padding=1)

        # self.codebook = Codebook(4096,512) 
        # self.quant_conv = nn.Conv2d(3, 3, 1)
        # self.post_quant_conv = nn.Conv2d(3, 3, 1)


    def forward(self, x):
        # x = self.quant_conv(x)
        # x,_,q_loss = self.codebook(x)
        # x = self.post_quant_conv(x)

        x = self.branch(x)
        x = self.final(x)

        # x = self.model2(x)
        return x

class post_GAE(nn.Module):
    def __init__(self, n_colors):
        self.trunk = BranchUnit(n_colors, n_feats=256, n_blocks=3, act=nn.LeakyReLU(), res_scale=0.1, up_scale=1,conv=default_conv,use_tail=False)
        self.final = nn.Conv2d(256, n_colors, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.trunk(x)
        x = self.final(x)
        return x



class GAE(nn.Module):
    def __init__(self, Encoder, Decoder, n_subs=8, n_ovls=2, n_colors=31, n_feats=128):
        super(GAE, self).__init__()
        self.Encoder = Encoder(n_subs,3,n_feats)
        self.Decoder = Decoder(3,n_subs,n_feats)
        self.device = 'cuda:0'

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []
        self.trunk = BranchUnit(n_colors, n_feats=32, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1, up_scale=1,conv=default_conv,use_tail=False)
        self.final = nn.Conv2d(32, n_colors, kernel_size=3, padding=1)
        
        # self.codebook = Codebook(4096,512)

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)


    def decode(self,x,z_list):
        b, c, h, w = x.shape
        self.device = 'cuda:0'
        channel_counter = torch.zeros(c).to(self.device)
        y = torch.zeros(b, c, h, w).to(self.device)
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            output_i = self.Decoder(z_list[g])
            y[:, sta_ind:end_ind, :, :] += output_i
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        # x = self.quant_conv(y)
        # x,_,__ = self.codebook(x)
        # x = self.post_quant_conv(x)
    # ---------------------------------------------------------------------------------------------
    #                                 是否进行后处理以及不同方式的设置！！！！
    # ---------------------------------------------------------------------------------------------
        y1 = self.trunk(y)
        y1 = self.final(y1)
        # y1 = self.post_unet(y)
        # y1 = self.unet(y1, time=None)
        # y1 = self.after_unet(y1)
        y = y1 +y
        return y

    def encode(self,x):
        b, c, h, w = x.shape
        # channel_counter = torch.zeros(c).cuda()
        self.device='cuda:0'
        y = torch.zeros(b, c, h, w).to(self.device)
        z_list = []
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            # 编码与解码，保存中间的隐藏层变量。
            z = self.Encoder(xi)
            z_list.append(z)
        return z_list

    def forward(self,x):
        b, c, h, w = x.shape
        self.device='cuda:0'
        channel_counter = torch.zeros(c).to(self.device)
        y = torch.zeros(b, c, h, w).to(self.device)
        z_list = []
        q_total = 0.0
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            # 编码与解码，保存中间的隐藏层变量。
            z = self.Encoder(xi)
            # print(z.shape)
            z_list.append(z)
            output_i = self.Decoder(z)
            # q_total += q_loss
            y[:, sta_ind:end_ind, :, :] += output_i
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        # x = self.quant_conv(y)
        # x,_,__ = self.codebook(x)
        # x = self.post_quant_conv(x)
    # ---------------------------------------------------------------------------------------------
    #                                 是否进行后处理以及不同方式的设置！！！！
    # ---------------------------------------------------------------------------------------------
        y1 = self.trunk(y)
        y1 = self.final(y1)
        # y1 = self.post_unet(y)
        # y1 = self.unet(y1, time=None)
        # y1 = self.after_unet(y1)
        y = y1 +y
        return y, z_list
        # return y, z_list, q_total/self.G



class SR_encoder(nn.Module):
    def __init__(self, Encoder, Decoder, n_subs=8, n_ovls=2, n_colors=31, n_feats=128, device='cuda:0'):
        super(SR_encoder, self).__init__()
        self.Encoder = Encoder(n_subs,3,n_feats)
        self.device = device

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

    def forward(self,x):
        b, c, h, w = x.shape
        # device='cuda:0'
        channel_counter = torch.zeros(c).to(self.device)
        y = torch.zeros(b, c, h, w).to(self.device)
        z_list = []
        q_total = 0.0
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = x[:, sta_ind:end_ind, :, :]
            z = self.Encoder(xi)
            # print(z.shape)
            z_list.append(z)

        return z_list


class AE(nn.Module):
    def __init__(self, Encoder, Decoder, in_channels=102, n_feats=128):
        super(AE, self).__init__()
        self.Encoder = Encoder(in_channels,3,n_feats)
        self.Decoder = Decoder(3,in_channels,n_feats)
        self.device = 'cuda:1'
        self.trunk = BranchUnit(in_channels, n_feats=32, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1, up_scale=1,conv=default_conv,use_tail=False)
        self.final = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)

    def decode(self,x):
        y = self.Decoder(x)
        y1 = self.trunk(y)
        y1 = self.final(y1)
        y = y1 + y
        return y

    def encode(self,x):
        x = self.Encoder(x)
        return x

    def forward(self,x):
        x = self.Encoder(x)
        y = self.Decoder(x)
        y1 = self.trunk(y)
        y1 = self.final(y1)
        y = y1 + y
        return y


class AE_duichen(nn.Module):
    def __init__(self, Encoder, Decoder, n_subs=8, n_ovls=2, n_colors=31, n_feats=128):
        super(AE_duichen, self).__init__()
        self.Encoder = Encoder(n_subs,3,n_feats)
        self.Decoder = Decoder(3,n_subs,n_feats)
        self.device = 'cuda:0'

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        self.trunk = BranchUnit(n_colors, n_feats=32, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1, up_scale=1,conv=default_conv,use_tail=False)
        self.final = nn.Conv2d(32, n_colors, kernel_size=3, padding=1)

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)


    def decode(self,x,z_list):
        b, c, h, w = x.shape
        channel_counter = torch.zeros(c).to(self.device)
        y = torch.zeros(b, c, h, w).to(self.device)
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            output_i = self.Decoder(z_list[g])
            y[:, sta_ind:end_ind, :, :] += output_i
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        return y

    def encode(self,x):
        b, c, h, w = x.shape
        # channel_counter = torch.zeros(c).cuda()
        self.device='cuda:0'
        y = torch.zeros(b, c, h, w).to(self.device)
        z_list = []
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]

            xi = x[:, sta_ind:end_ind, :, :]
            # 编码与解码，保存中间的隐藏层变量。
            z = self.Encoder(xi)
            z_list.append(z)
        return z_list

    def forward(self,x):
        b, c, h, w = x.shape
    # ---------------------------------------------------------------------------------------------
    #                                 cuda的设置！！！！
    # ---------------------------------------------------------------------------------------------
        self.device='cuda:0'
        channel_counter = torch.zeros(c).to(self.device)
        y = torch.zeros(b, c, h, w).to(self.device)
        z_list = []
        q_total = 0.0
        for g in range(self.G):
            sta_ind = self.start_idx[g]
            end_ind = self.end_idx[g]
            xi = x[:, sta_ind:end_ind, :, :]
            # 编码与解码，保存中间的隐藏层变量。
            z = self.Encoder(xi)
            # print(z.shape)
            z_list.append(z)
            output_i = self.Decoder(z)
            # q_total += q_loss
            y[:, sta_ind:end_ind, :, :] += output_i
            channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1
        # intermediate “result” is averaged according to their spectral indices
        y = y / channel_counter.unsqueeze(1).unsqueeze(2)
        return y, z_list


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Buliding model
    print('===> Building model')
    load = False
    # if load:
    #     model_E = torch.load('./weight/E_VGGSAM_4_Foster.pth',map_location='cuda:0')
    #     model_D = torch.load('./weight/D_VGGSAM_4_Foster.pth',map_location='cuda:0')
    #     # model_E = torch.load('./weight/E_VGGSAM1_4_Harvard.pth')
    #     # model_D = torch.load('./weight/D_VGGSAM1_4_Harvard.pth')
    #     model_E = model_E.to(device)
    #     model_D = model_D.to(device)
    #     print("模型读取成功, 进行fine tune 训练！！！")
    # else:
    #     # Foster数据集是33通道的，注意更改,Chikusei 128, PaviaC 102
    #     model_E = Encoder(128,3)
    #     model_D = Decoder(3,128)
    #     model_E = model_E.to(device)
    #     model_D = model_D.to(device)
    if load:
        AE_model = torch.load('./weight/GAE_4_Pav.pth')
        AE_model =AE_model.to(device)
        print("模型读取成功, 进行fine tune 训练！！！")
    else:
        AE_model = GAE(Encoder=Encoder, Decoder=Decoder, n_subs=16, n_ovls=4, n_colors=102).to(device)
    
    # AE_model = nn.DataParallel(AE_model,device_ids=[4,5,6,7])

    vgg_model = models.vgg19(pretrained=True)
    vgg_model = vgg_model.to(device)
    # vgg_model = nn.DataParallel(vgg_model,device_ids=[0,1,2,3])
    # x = torch.randn((1, 3,128,128))
    # y = vgg_model(x)
    # print(y.shape)

    # model_E = nn.DataParallel(model_E,device_ids=[0,1,2,3,4,5,6,7])
    # model_D = nn.DataParallel(model_D,device_ids=[0,1,2,3,4,5,6,7])


    criterion = nn.L1Loss().to(device)
    # optimizer_E = optim.Adam(model_E.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_D = optim.Adam(model_D.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08)
    optimizer_AE = optim.Adam(AE_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)

    # optimizer_E = optim.SGD(model_E.parameters(), lr=0.0001)
    # optimizer_D = optim.SGD(model_D.parameters(), lr=0.0001)


    # 加载数据集
    # 重新处理了数据集，重新读取
    # train_set = TrainsetFromFolder('../Harvard_4_train/') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    # train_set = TrainsetFromFolder('../train/Cave/4/') # 注意注意，不同数据集，记得切换通道数量，31，31，33，128,102
    # train_set = TrainsetFromFolder('../train/Foster/4/')
    # train_set = TrainsetFromFolder('../train/Chikusei/4/')
    train_set = TrainsetFromFolder('../train/PaviaC/4/')
    # train_set = HSTrainingData(image_dir= '../PaviaC_mat/train/', n_scale = 4, augment=True, ch3=False, num_ch=0)
    print(len(train_set))
    train_loader = DataLoader(dataset=train_set,  batch_size=8, shuffle=True,num_workers=4)

    for epoch in range(5):
        count = 0
        for data,hr in train_loader:
            # bs 31 36 36  / bs 31 144 144
            # lr = lr.reshape((lr.shape[0], 1, lr.shape[1], lr.shape[2], lr.shape[3]))
            # lr = data['LR'].to(device)
            # hr = hr.reshape((hr.shape[0], 1, hr.shape[1], hr.shape[2], hr.shape[3]))

            # hr = data['HR'].to(device)
            hr = hr.to(device)
            # print(hr.shape,lr.shape)

            # 进行mask操作，让AE学会复原。
            # hr_mask = random_mask(hr,p=0.6)

            hr_rcon, z_list = AE_model(hr)
            z_list = AE_model.encode(hr)
            # print(hr_rcon.shape,len(z_list),z_list[0].shape)

            # 自己设计的损失函数
            sam_loss = SAM_torch(hr_rcon.clone(),hr.clone())
            random_list = torch.randint(0,102,(3,))
            p_loss = criterion(vgg_model(hr_rcon[:,random_list,:,:]), vgg_model(hr[:,random_list,:,:]))
            l1_loss = criterion(hr_rcon,hr)

            # SSPSR loss 实验发现，生成的图片有明显的颜色偏移，使用颜色校正，效果也不好。有可能是训练轮次少了。目前实验结果不好
            # h_loss = HybridLoss(spatial_tv=True, spectral_tv=True).to(device)
            # loss = h_loss(hr_rcon, hr)

            # VGGSAM，目前VGGSAM2版本最好。
            # loss = l1_loss + 1e-3 * p_loss
            loss_func = HLoss(0.3, 0.1)
            loss = loss_func(hr_rcon, hr)
            # loss = loss_func(hr_rcon, hr)+ 1e-3 * p_loss
            # loss = l1_loss + 1e-3 * p_loss + 3e-3 * sam_loss + q_loss
            # print(loss)

            optimizer_AE.zero_grad()
            # optimizer_D.zero_grad()
            # optimizer_E.zero_grad()
            loss.backward(retain_graph=True)
            # print("每个损失对应的梯度 l1_loss.grad = {} , p_loss.grad = {}, sam_loss.grad = {} ".format(l1_loss.grad, p_loss.grad, sam_loss.grad))
            optimizer_AE.step()
            # optimizer_D.step()
            # optimizer_E.step()
            

            count = count + 1
            # print("wow!!! 第{}个Epoch的第{}轮 total_loss = {} , p_loss = {} , l1_loss = {} , sam_loss = {}, q_loss = {} ."
            #     .format(epoch, count, loss, 1e-3 * p_loss, l1_loss , 1e-2 * sam_loss , q_loss))
            print("wow!!! 第{}个Epoch的第{}轮 total_loss = {}" . format(epoch,count,loss))

        OUT_DIR = Path('./weight')
        # torch.save(model_E, OUT_DIR.joinpath('E_VGGSAM_4_Cks.pth'))
        # torch.save(model_D, OUT_DIR.joinpath('D_VGGSAM_4_Cks.pth'))
        torch.save(AE_model, OUT_DIR.joinpath('GAE_8_Pav.pth'))
