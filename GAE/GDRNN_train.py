import os
import argparse
import scipy.io as sio
import numpy as np
import time
import random
import torch

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
import torch.nn as nn
from torch.autograd import Variable
from models_GDRRN import GDRRN
from torch.utils.data import DataLoader
import torch.utils.data as data
from pathlib import Path

from utils import TrainsetFromFolder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description="PyTorch GDRRN")
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
# parser.add_argument("--resume", default="model/model_ISSR_epoch_80.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.005, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default: 4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
# parser.add_argument('--dataset', default='data/fusion_trainset.mat', type=str, help='path to general model')
# parser.add_argument('--dataset', default='../HyperDatasets/trainset_of_CNNbasedFusion/train_data/fusion_trainset_CAVE_x32.mat', type=str, help='path to general model')
parser.add_argument('--dataset',
                    default='./generate_training_dataset/generate_trainset_of_GDRRN/train_data/fusion_trainset_Harvard_x4_32/',
                    type=str, help='path to general model')

method_name = 'HSI_SR_GDRRN_Harvard_up4_saml_1e1_g2'
sam_lamd = 0.1
mse_lamd = 1
group = 2
if_control_blc = False

sigma = 25


def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # opt.seed = random.randint(1, 10000)
    opt.seed = 3192
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # 重新处理了数据集，重新读取
    train_set = TrainsetFromFolder('../MCNet/4')  # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    print(len(train_set))
    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, drop_last=True)

    print("===> Building model")
    model = GDRRN(input_chnl_hsi=31, group=group)
    # criterion = nn.MSELoss()
    # criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        # model = torch.nn.DataParallel(model).cuda()
        model = dataparallel(model, 1)  # set the number of parallel GPUs
        # criterion = criterion.cuda()
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    # optimizer = optim.SGD([
    #     {'params': model.parameters()}
    # ], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer = optim.Adam([
        {'params': model.parameters()}
    ], lr=opt.lr, weight_decay=opt.weight_decay)

    print("===> Training")
    lossAarry = np.zeros(opt.nEpochs)
    losspath = 'losses/'
    if not os.path.exists(losspath):
        os.makedirs(losspath)

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        start_time = time.time()
        lossAarry[epoch - 1] = lossAarry[epoch - 1] + train(train_loader, optimizer, model, epoch)
        print(
            "===> Epoch[{}]: Loss={:.5f}, time = {:.4f}".format(epoch, lossAarry[epoch - 1], time.time() - start_time))
        save_checkpoint(model, epoch)

    OUT_DIR = Path('./weight')
    torch.save(model, OUT_DIR.joinpath('GDRNN_4_Harvard.pth'))

    # sio.savemat(losspath + method_name + '_lossArray.mat', {'lossArray': lossAarry})


def train(training_data_loader, optimizer, model, epoch):
    lr = adjust_learning_rate(epoch - 1, opt.step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, low_lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    start_time = time.time()

    model.train()
    lossValue = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        hsi, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        print(hsi.shape)
        if opt.cuda:
            hsi = hsi.cuda()
            label = label.cuda()
        res = model(hsi)

        # loss = criterion(res, label)
        if if_control_blc is True:
            lossfunc = myloss_spe(hsi.data.shape[0], lamd=sam_lamd, mse_lamd=mse_lamd, epoch=epoch - 1)
        else:
            lossfunc = myloss_spe(hsi.data.shape[0], lamd=sam_lamd, mse_lamd=mse_lamd)
        loss = lossfunc.forward(res, label)

        # loss = criterion(res, label)/(input.data.shape[0]*2)

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        lossValue = lossValue + loss.data.item()
        if (iteration + 1) % 50 == 0:
            elapsed_time = time.time() - start_time
            # save_checkpoint(model, iteration)
            print("===> Epoch[{}]: iteration[{}]: Loss={:.5f}, time = {:.4f}".format(epoch, iteration + 1,
                                                                                     # criterion(lres + hres, target).data[0], loss_low.data[0], 0, elapsed_time))
                                                                                     loss.data.item(), elapsed_time))

    elapsed_time = time.time() - start_time
    lossValue = lossValue / (iteration + 1)
    # print("===> Epoch[{}]: Loss={:.5f}, time = {:.4f}".format(epoch, lossValue, elapsed_time))
    return lossValue


class myloss_spe(nn.Module):
    def __init__(self, N, lamd=1e-1, mse_lamd=1, epoch=None):
        super(myloss_spe, self).__init__()
        self.N = N
        self.lamd = lamd
        self.mse_lamd = mse_lamd
        self.epoch = epoch
        return

    def forward(self, res, label):
        print(res.shape,label.shape)
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


def adjust_learning_rate(epoch, step):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    # if epoch < step:
    #     lr = opt.lr #* (0.1 ** (epoch // opt.step))#0.2
    # elif epoch < 3 * step:
    #     lr = opt.lr * 0.1 #* (0.1 ** (epoch // opt.step))#0.2
    # elif epoch < 5 * step:
    #     lr = opt.lr * 0.01  # * (0.1 ** (epoch // opt.step))#0.2
    # else:
    #     lr = opt.lr * 0.001
    lr = opt.lr * (0.1 ** (epoch // opt.step))  # 0.2
    return lr


def save_checkpoint(model, epoch):
    fold = "model_" + method_name + "/"
    model_out_path = fold + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(fold):
        os.makedirs(fold)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0 + ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model


if __name__ == "__main__":
    main()
    exit(0)
