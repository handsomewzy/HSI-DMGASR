from common import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from icvl_data import LoadData
from utils import SAM_torch, PSNR_GPU, get_paths, TrainsetFromFolder, SAM
import sewar
import MCNet
from pathlib import Path
from torch.nn.functional import interpolate
import torchvision.models as models
import numpy as np
from SSPSR import HybridLoss
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import os


import torch.nn.functional as F

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


class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()

        kernel_size = 3
        m = []

        for i in range(n_blocks):
            m.append(SSB(n_feats, kernel_size, act=act, res_scale=res_scale))

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
        self.head = conv(n_colors, n_feats, kernel_size)
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
    def __init__(self, input_channel, out_channel):
        super(Encoder, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.model = nn.Sequential(
            SSPN(n_feats=self.input_channel, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
            nn.Conv2d(self.input_channel, self.input_channel // 3, 1),
            # SSB(n_feats=self.input_channel//3,kernel_size=1,act=nn.LeakyReLU(),res_scale=0.1),
            SSPN(n_feats=self.input_channel // 3, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
            nn.Conv2d(self.input_channel // 3, self.input_channel // 6, 1),
            SSPN(n_feats=self.input_channel // 6, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
            nn.Conv2d(self.input_channel // 6, self.out_channel, 1),
        )

    def forward(self, x):
        res = self.model(x)

        return res


class Decoder(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(Decoder, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.model = nn.Sequential(
            SSPN(n_feats=self.input_channel, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
            nn.Conv2d(self.input_channel, self.input_channel * 3, 1),
            # SSB(n_feats=self.input_channel//3,kernel_size=1,act=nn.LeakyReLU(),res_scale=0.1),
            SSPN(n_feats=self.input_channel * 3, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
            nn.Conv2d(self.input_channel * 3, self.input_channel * 9, 1),
            SSPN(n_feats=self.input_channel  * 9, n_blocks=2, act=nn.LeakyReLU(), res_scale=0.1),
            nn.Conv2d(self.input_channel * 9, self.out_channel, 1),
        )

    def forward(self, x):
        res = self.model(x)

        return res


# python -m torch.distributed.launch --nproc_per_node=8 --master_port='29512' --use_env
os.environ['MASTER_ADDR'] = 'localhost'


if __name__ == "__main__":
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])

    # 指定具体的GPU
    torch.cuda.set_device(local_rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"  # 根据gpu的数量来设定，初始gpu为0，这里我的gpu数量为4
    device = torch.device("cuda", local_rank)



    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Buliding model
    print('===> Building model')
    load = True
    if load:
        model_E = torch.load('./weight/E_maskl1_4_CAVE.pth',map_location=device)
        model_D = torch.load('./weight/D_maskl1_4_CAVE.pth',map_location=device)
        # model_E = torch.load('./weight/E_VGGSAM1_4_Harvard.pth')
        # model_D = torch.load('./weight/D_VGGSAM1_4_Harvard.pth')
        # model_E = model_E.to(device)
        # model_D = model_D.to(device)
        model_E = DistributedDataParallel(model_E,device_ids=[local_rank])
        model_D = DistributedDataParallel(model_D,device_ids=[local_rank])
        print("模型读取成功, 进行fine tune 训练！！！")
    else:
        model_E = Encoder(31,3).to(device)
        model_D = Decoder(3,31).to(device)
        model_E = DistributedDataParallel(model_E,device_ids=[local_rank])
        model_D = DistributedDataParallel(model_D,device_ids=[local_rank])
        # model_E = model_E.to(device)
        # model_D = model_D.to(device)


    vgg_model = models.vgg19(pretrained=True)
    vgg_model = vgg_model.cuda()
    # vgg_model = nn.DataParallel(vgg_model,device_ids=[0,1,2,3])
    # x = torch.randn((1, 3,128,128))
    # y = vgg_model(x)
    # print(y.shape)

    # DP并行
    # model_E = nn.DataParallel(model_E,device_ids=[0,1,2,3,4,5,6,7])
    # model_D = nn.DataParallel(model_D,device_ids=[0,1,2,3,4,5,6,7])


    criterion = nn.L1Loss().to(device)
    optimizer_E = optim.Adam(model_E.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D = optim.Adam(model_D.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)


    # 加载数据集
    # 重新处理了数据集，重新读取
    # train_set = TrainsetFromFolder('../Harvard_4_train/') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    train_set = TrainsetFromFolder('../train/Cave/4/4')
    train_sampler = DistributedSampler(train_set)
    if local_rank == 0:
        print(len(train_set))  
    train_loader = DataLoader(dataset=train_set,sampler= train_sampler, batch_size=64, shuffle=False, num_workers=4)

    for epoch in range(50):
        count = 0
        for lr, hr in train_loader:
            # bs 31 36 36  / bs 31 144 144
            # lr = lr.reshape((lr.shape[0], 1, lr.shape[1], lr.shape[2], lr.shape[3]))
            lr = lr.to(device)
            # hr = hr.reshape((hr.shape[0], 1, hr.shape[1], hr.shape[2], hr.shape[3]))
            hr = hr.to(device)
            # print(lr.shape, hr.shape)

            # 进行mask操作，让AE学会复原。
            hr = random_mask(hr,p=0.6)

            # print(hr.device, model_E.device)
            model_E = model_E.to(device)
            z = model_E(hr)
            hr_rcon = model_D(z)

            sam_loss = SAM_torch(hr_rcon.clone(),hr.clone())
            # print(sam_loss)

            # print(SR.shape)
            random_list = torch.randint(0,31,(3,))
            # print(random_list)
            p_loss = criterion(vgg_model(hr_rcon[:,random_list,:,:]), vgg_model(hr[:,random_list,:,:]))
            l1_loss = criterion(hr_rcon,hr)
            
            # x_recon_np = hr_rcon.cpu().detach().numpy()
            # x_recon_np[x_recon_np < 0] = 0
            # x_recon_np[x_recon_np > 1.] = 1.
            # hr_np = hr.cpu().numpy()
            
            # sam_loss = SAM(x_recon_np,hr_np)
            # sam_loss = np.array(sam_loss)
            # sam_loss = torch.from_numpy(sam_loss,requires_grad=True)
            # print(sam_loss)

            # SSPSR loss 实验发现，生成的图片有明显的颜色偏移，使用颜色校正，效果也不好。有可能是训练轮次少了。目前实验结果不好
            # h_loss = HybridLoss(spatial_tv=True, spectral_tv=True).to(device)
            # loss = h_loss(hr_rcon, hr)

            # VGGSAM，目前VGGSAM2版本最好。
            loss = l1_loss
            # loss = l1_loss + 1e-3 * p_loss + 1e-2 * sam_loss
            # print(loss)

            optimizer_D.zero_grad()
            optimizer_E.zero_grad()
            loss.backward(retain_graph=True)
            # print("每个损失对应的梯度 l1_loss.grad = {} , p_loss.grad = {}, sam_loss.grad = {} ".format(l1_loss.grad, p_loss.grad, sam_loss.grad))
            optimizer_D.step()
            optimizer_E.step()

            count = count + 1
            if local_rank == 0:
                print("wow!!! 第{}个Epoch的第{}轮 total_loss = {} , p_loss = {} , l1_loss = {} , sam_loss = {} ."
                    .format(epoch, count, loss, 1e-3 * p_loss, l1_loss , 1e-2 * sam_loss))
            # print("wow!!! 第{}个Epoch的第{}轮 total_loss = {}" . format(epoch,count,loss))

        if local_rank ==0:
            OUT_DIR = Path('./weight')
            torch.save(model_E, OUT_DIR.joinpath('E_maskl1_4_CAVE.pth'))
            torch.save(model_D, OUT_DIR.joinpath('D_maskl1_4_CAVE.pth'))
