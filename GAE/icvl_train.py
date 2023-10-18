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

import torch
import torch.nn as nn
from net import Generator, Discriminator, Spe_loss, TVLoss
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import SAM, PSNR_GPU, get_paths, TrainsetFromFolder
from pathlib import Path


EPOCHS = 1
BATCH_SIZE = 16
LR = 1e-3
high_sr = 128
low_sr = high_sr / 4

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = ('cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # 原始对抗GAN的网络读取 网络中要输入尺寸大小，这是个很愚蠢的网络啊感觉，很不智能
    g_model = Generator(bs=BATCH_SIZE, l=31, h=low_sr, w=low_sr).to(device)
    d_model = Discriminator(bs=BATCH_SIZE, l=31, h=high_sr, w=high_sr).to(device)

    # 专门儿编写的生成器损失函数
    # g_criterion = Loss()
    d_criterion = nn.BCELoss()
    criterion = {
        'l1': nn.L1Loss(),
        'ltv': TVLoss(),
        'ls': Spe_loss(),
        'la': nn.BCELoss(),
    }

    g_optimizer = optim.Adam(
        g_model.parameters(),
        lr=LR
    )
    d_optimizer = optim.SGD(
        d_model.parameters(),
        lr=LR
    )

    sorce = {
        'd_loss': 0.0,
        'g_loss': 0.0,
        'real_sorce': 0.0,
        'fake_sorce': 0.0
    }

    best_sorce = {
        'psnr': 0.0,
        'sam': 180.0,
        'epoch': 0,
    }

    # 重新处理了数据集，重新读取
    train_set = TrainsetFromFolder('../MCNet/4') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    print(len(train_set))
    train_loader = DataLoader(dataset=train_set,  batch_size=16, shuffle=True, drop_last=True)

    for epoch in range(EPOCHS):

        count = 0
        for lr, hr in train_loader:
            lr = lr.reshape((lr.shape[0], 1, lr.shape[1], lr.shape[2], lr.shape[3]))
            lr = lr.to(device)
            hr = hr.reshape((hr.shape[0], 1, hr.shape[1], hr.shape[2], hr.shape[3]))
            hr = hr.to(device)

            real_labels = torch.ones(BATCH_SIZE).to(device)
            fake_labels = torch.zeros(BATCH_SIZE).to(device)

            # ================================================ #
            #                训练判别器部分                     #
            # ================================================ #

            # 计算real标签 也就是hr的损失
            output = d_model(hr)
            d_loss_real = d_criterion(torch.squeeze(output), real_labels)
            # print('real res {}'.format(torch.squeeze(output)))
            real_sorce = output
            sorce['real_sorce'] = real_sorce.mean()

            # 计算fake标签  也就是lr的损失
            fake_hr = g_model(lr)
            output = d_model(fake_hr)
            d_loss_fake = d_criterion(torch.squeeze(output), fake_labels)
            # print('fake res {}'.format(torch.squeeze(output)))
            fake_sorce = output
            sorce['fake_sorce'] = fake_sorce.mean().item()

            # 反向传播 参数更新部分
            d_loss = (d_loss_real + d_loss_fake) / 2
            sorce['d_loss'] = d_loss.item()
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ================================================ #
            #                训练生成器部分                     #
            # ================================================ #

            fake_hr = g_model(lr)
            output = d_model(fake_hr)
            # 损失计算 没有加别的损失，此处只有l1损失
            fake_hr = torch.squeeze(fake_hr)
            hr = torch.squeeze(hr)
            g_loss = criterion['l1'](fake_hr, hr) + 1e-6 * criterion['ltv'](fake_hr) + 1e-2 * criterion['ls'](
                fake_hr, hr) + 1e-3 * d_criterion(torch.squeeze(output), real_labels)

            sorce['g_loss'] = g_loss

            # 反向传播 优化
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            print('EPOCH : {} step : {} \
                    d_loss : {:.4f} g_loss : {:.4f} \
                    real_sorce {:.4f} fake_sorce {:.4f}'.format(
                epoch, count + 1,
                sorce['d_loss'], sorce['g_loss'],
                sorce['real_sorce'], sorce['fake_sorce']
            ))
            count += 1

        # 训练完成，保存模型
        OUT_DIR = Path('./weight')
        torch.save(g_model, OUT_DIR.joinpath('GAN_4_Harvard.pth'))

