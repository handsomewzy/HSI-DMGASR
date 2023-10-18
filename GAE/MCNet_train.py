import torch
import torch.nn as nn
# from torch.nn.parallel import DistributedDataParalled as ddp
from torch.utils.data import DataLoader
import torch.optim as optim
from icvl_data import LoadData
from utils import SAM, PSNR_GPU, get_paths, TrainsetFromFolder
import sewar
import MCNet
from torch.autograd import Variable
from pathlib import Path
from torch.optim.lr_scheduler import MultiStepLR
import os
import torch.distributed as dist
import argparse
from HStest import HSTestData
from HStrain import HSTrainingData
import torch.nn.functional as func

# python -m torch.distributed.launch --nproc_per_node=4 --master_port='29511' --use_env


EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4
high_sr = 128
low_sr = high_sr / 4


if __name__ == "__main__":
    # 初始化
    # dist.init_process_group(backend='nccl')
    # local_rank = int(os.environ["LOCAL_RANK"])
    # print(local_rank)

    # 指定具体的GPU
    # torch.cuda.set_device(local_rank)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"  # 根据gpu的数量来设定，初始gpu为0，这里我的gpu数量为4
    # device = torch.device("cuda", local_rank)


    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device is {}'.format(device))

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Buliding model
    # model = MCNet.MCNet(scale = 3 ,n_colors = 31 ,n_feats = 32).to(device)
    model = torch.load('./weight/MCNet_3_Chi.pth')
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)

    # 初始化ddp模型，用于单机多卡并行运算
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # model = nn.DataParallel(model,device_ids=[4,5,6,7])

    # 旧的数据读取方式，舍弃
    # train_paths, val_paths, _ = get_paths()
    # print(train_paths, val_paths)


    # 加载数据集
    # train_data = DataLoader(
    #     LoadData(train_paths, 'train', fis=high_sr),
    #     # LoadData(),
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # 重新处理了数据集，重新读取
    # 分布式，需要划分数据集,增加sampler模块
    # train_set = TrainsetFromFolder('../Harvard_4_train/') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    # train_set = TrainsetFromFolder('../train/Cave/4/')
    # train_set = TrainsetFromFolder('../train/Chikusei/4/')
    # train_set = TrainsetFromFolder('../train/PaviaC/4/')
    train_set = HSTrainingData(image_dir= '../Chikusei_mat/train/', n_scale = 3, augment=True, ch3=False, num_ch=0)
    print(len(train_set))
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)  # 切分训练数据集
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True) # 分布式不能进行shuffle


    # Setting learning rate
    # scheduler = MultiStepLR(optimizer, milestones=[35, 70, 105, 140, 175], gamma=0.5, last_epoch=-1)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5, last_epoch=-1)

    for epoch in range(50):
        count = 0
        for data in train_loader:
            lr = data['LR'].to(device)
            hr = data['HR'].to(device)
            lms = data['SR'].to(device)

            SR = model(lr)
            # print(SR.shape)

            loss = criterion(SR, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count = count + 1
            print("天哪，这轮训练完成了！第{}个Epoch的第{}轮的损失为：{}".format(epoch, count, loss))
        scheduler.step()

    OUT_DIR = Path('./weight')
    torch.save(model, OUT_DIR.joinpath('MCNet_3_Chi.pth'))
