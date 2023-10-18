import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.sigmoid(x)
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, padding)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, padding)
        self.attention = AttentionBlock(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        attention = self.attention(x)
        x = x * attention
        x += residual
        x = F.relu(x)
        return x

class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.conv1 = ConvBlock(31, 64, kernel_size=3, padding=1)
        self.residual_attention_block1 = ResidualAttentionBlock(64, 64, kernel_size=3, padding=1)
        self.residual_attention_block2 = ResidualAttentionBlock(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 31, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_attention_block1(x)
        x = self.residual_attention_block2(x)
        x = self.conv2(x)
        x += F.interpolate(x, size=(x.size(-2)*2, x.size(-1)*2), mode='bilinear', align_corners=False)
        return x


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Buliding model
    print('===> Building model')
    finetune_model = AttentionNet()
    finetune_model = finetune_model.to(device)

    # 读取AE模型
    load = False
    if load:
        model_E = torch.load('./weight/E_VGGSAM2_4_Harvard.pth',map_location='cuda:0')
        model_D = torch.load('./weight/D_VGGSAM2_4_Harvard.pth',map_location='cuda:0')
        # model_E = torch.load('./weight/E_VGGSAM1_4_Harvard.pth')
        # model_D = torch.load('./weight/D_VGGSAM1_4_Harvard.pth')
        model_E = model_E.to(device)
        model_D = model_D.to(device)
        print("模型读取成功, 进行fine tune 训练！！！")
    else:
        model_E = Encoder(31,3)
        model_D = Decoder(3,31)
        model_E = model_E.to(device)
        model_D = model_D.to(device)


    model_E = nn.DataParallel(model_E,device_ids=[0,1,2,3,4,5,6,7])
    model_D = nn.DataParallel(model_D,device_ids=[0,1,2,3,4,5,6,7])


    criterion = nn.L1Loss().to(device)
    optimizer_E = optim.Adam(model_E.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D = optim.Adam(model_D.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)


    # 加载数据集
    # 重新处理了数据集，重新读取
    # train_set = TrainsetFromFolder('../Harvard_4_train/') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    train_set = TrainsetFromFolder('../train/Cave/4/4')
    print(len(train_set))
    train_loader = DataLoader(dataset=train_set,  batch_size=64, shuffle=True,num_workers=8)

    for epoch in range(20):
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
            # loss = l1_loss
            loss = l1_loss + 1e-3 * p_loss + 1e-2 * sam_loss
            # print(loss)

            optimizer_D.zero_grad()
            optimizer_E.zero_grad()
            loss.backward(retain_graph=True)
            # print("每个损失对应的梯度 l1_loss.grad = {} , p_loss.grad = {}, sam_loss.grad = {} ".format(l1_loss.grad, p_loss.grad, sam_loss.grad))
            optimizer_D.step()
            optimizer_E.step()

            count = count + 1
            print("wow!!! 第{}个Epoch的第{}轮 total_loss = {} , p_loss = {} , l1_loss = {} , sam_loss = {} ."
                .format(epoch, count, loss, 1e-3 * p_loss, l1_loss , 1e-2 * sam_loss))
            # print("wow!!! 第{}个Epoch的第{}轮 total_loss = {}" . format(epoch,count,loss))

        OUT_DIR = Path('./weight')
        torch.save(model_E, OUT_DIR.joinpath('E_mask_4_CAVE.pth'))
        torch.save(model_D, OUT_DIR.joinpath('D_mask_4_CAVE.pth'))
