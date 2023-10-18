#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   训练GAN网络
@File    :   train.py
@Time    :   2021/01/05 21:17:38
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2021/01/05 21:17:38
'''
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from net import Generator, Discriminator, Spe_loss, TVLoss, ESR_Discriminator
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
from G import *
from data import LoadData
from utils import SAM, PSNR_GPU
from pathlib import Path
import numpy as np
from reading_data import HyperX

EPOCHS = 300
BATCH_SIZE = 16
LR = 1e-3
WARM_UP_EPOCH = 100

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    g_model = Generator(BATCH_SIZE).to(device)
    d_model = Discriminator(BATCH_SIZE).to(device)

    # 专门儿编写的生成器损失函数
    # g_criterion = Loss()
    criterion_spe = Spe_loss().to(device)
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    # criterion_content = nn.L1Loss().to(device)
    criterion_pixel = nn.L1Loss().to(device)

    g_optimizer = optim.Adam(
        g_model.parameters(),
        lr=LR,
        betas=(0.9, 0.999)
    )
    d_optimizer = optim.Adam(
        d_model.parameters(),
        lr=LR,
        betas=(0.9, 0.999)
    )

    best_sorce = {
        'psnr': 0.0,
        'sam': 180.0,
        'epoch': 0,
    }

    data = h5py.File('./data/4cam_0411-1640-1.mat', 'r')
    # print(data.keys())
    # print(data['rad'])
    data_np = np.array(data['rad'])
    # print(data_np.shape)
    data_np = data_np.transpose(1, 2, 0)

    gt = np.zeros((1392, 1300))
    new_dataset = HyperX(data=data_np, gt=gt, patch_size=1200, dataset='256', ignored_labels=[2])
    print(len(new_dataset))
    train_data = DataLoader(new_dataset, batch_size=64, drop_last=True)
    val_data = train_data



    for epoch in range(EPOCHS):
        count = 0
        for lr, hr in train_data:
            # lr = lr.reshape((lr.shape[0], 1, lr.shape[1], lr.shape[2], lr.shape[3]))
            lr = lr.to(device)
            # hr = hr.reshape((hr.shape[0], 1, hr.shape[1], hr.shape[2], hr.shape[3]))
            hr = hr.to(device)
            # shape 变化
            real_labels = torch.ones(BATCH_SIZE, 1).to(device)
            fake_labels = torch.zeros(BATCH_SIZE, 1).to(device)

            # ================================================ #
            #                训练生成器部分                     #
            # ================================================ #

            g_optimizer.zero_grad()

            fake_hr = g_model(lr)
            # l1 loss
            loss_pixel = criterion_pixel(fake_hr, hr)

            # 先训练一会生成器
            if epoch < WARM_UP_EPOCH:
                loss_pixel.backward()
                g_optimizer.step()
                print('warm up epoch {} pixel loss {:.4f}'.format(epoch, loss_pixel.item()))
                continue

            pre_real = d_model(hr)
            pre_fake = d_model(fake_hr)

            # TODO:计算对抗损失
            loss_GAN = criterion_GAN(pre_fake - pre_real.mean(0, keepdim=True), real_labels)

            loss_spe = criterion_spe(fake_hr, hr)

            loss_G = loss_pixel + 1e-3 * loss_GAN + 1e-2 * loss_spe

            loss_G.backward()
            g_optimizer.step()

            # ================================================ #
            #                训练判别器部分                     #
            # ================================================ #

            d_optimizer.zero_grad()

            fake_hr = g_model(lr)

            pre_real = d_model(hr)
            pre_fake = d_model(fake_hr)

            loss_real = criterion_GAN(pre_real - pre_fake.mean(0, keepdim=True), real_labels)
            loss_fake = criterion_GAN(pre_fake - pre_real.mean(0, keepdim=True), fake_labels)
            loss_D = (loss_fake + loss_real) / 2

            loss_D.backward()
            d_optimizer.step()

            print("EPOCH {} step {} G-L1 {:.4f} G-GAN {:.4f} G-SPE {:.4f} \
D-TOTAL {:.4f} D-REAL {:.4f} D-FAKE {:.4f}".format(
                epoch,
                count,
                loss_pixel.item(),
                loss_GAN.item(),
                loss_spe.item(),
                loss_D.item(),
                loss_real.item(),
                loss_fake.item()
            ))

            count += 1

            #  训练完成  开始验证

        g_model.eval()
        d_model.eval()
        val_count = 0
        val_psnr = 0
        val_sam = 0
        for lr, hr in val_data:
            lr = lr.reshape((lr.shape[0], 1, lr.shape[1], lr.shape[2], lr.shape[3]))
            lr = lr.to(device)
            hr = hr.reshape((hr.shape[0], 1, hr.shape[1], hr.shape[2], hr.shape[3]))
            hr = hr.to(device)

            with torch.no_grad():
                fake_hr = g_model(lr)
                fake_hr = torch.squeeze(fake_hr)
                hr = torch.squeeze(hr)

                fake_hr = fake_hr.cpu()
                hr = hr.cpu()

                psnr = PSNR_GPU(hr, fake_hr)
                val_psnr += psnr
                sam = SAM(hr, fake_hr)
                val_sam += sam

                print('val epoch : {} step : {} psnr : {:.4f}  sam : {:.4f}'.format(
                    epoch, val_count + 1, psnr, sam
                ))

                val_count += 1

        print('val averagr psnr : {:.4f} sam : {:.4f}'.format(
            val_psnr / (val_count),
            val_sam / (val_count))
        )

        if val_psnr / (val_count + 1) > best_sorce['psnr']:
            # 以psnr为主  找到最好的 保存下来
            best_sorce['psnr'] = val_psnr / (val_count)

            torch.save(copy.deepcopy(g_model.state_dict()), OUT_DIR.joinpath('g_model.pth'))
            torch.save(copy.deepcopy(d_model.state_dict()), OUT_DIR.joinpath('d_model.pth'))
