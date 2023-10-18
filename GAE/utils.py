#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


'''
@Useway  :   辅助函数
@File    :   utils.py
@Time    :   2020/12/31 16:26:28
@Author  :   Chen Zhuang 
@Version :   1.0
@Contact :   whut_chenzhuang@163.com
@Time: 2020/12/31 16:26:28
'''

from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from G import OUT_DIR
import scipy.io as io
import h5py
from torch.nn.functional import interpolate
import torch.utils.data as data
import torch
import numpy as np

from os import listdir
from os.path import join
import scipy.io as scio

def data_augmentation(label, mode=0):
    if mode == 0:
        # original
        return label
    elif mode == 1:
        # flip up and down
        return np.flipud(label)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(label)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(label))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(label, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(label, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(label, k=3))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

def save_cave_data():
    # 生成{train val test}.npy 文件

    path = Path('/home/yons/data1/chenzhuang/HSI-SR/DataSet/CAVE')
    data = np.zeros((32, 31, 512, 512))

    for i, p in enumerate(path.iterdir()):
        print(p)
        for j in range(31):

            img_path = p.joinpath(p.parts[-1], p.parts[-1] + '_{:0>2d}.png'.format(j + 1))
            # print(img_path)
            img = Image.open(img_path)
            img = np.array(img)

            # 有特殊shape的图片
            if len(img.shape) != 2:
                data[i][j] = img[:, :, 0]

            if len(img.shape) == 2:
                data[i][j] = img
            # print((i,j))

    print(data[:20].shape, data[20:26].shape, data[26:].shape)
    np.save('train.npy', data[:20])
    np.save('val.npy', data[20:26])
    np.save('test.npy', data[26:])


def calc_psnr(img1, img2):
    return 10. * torch.log10((255.0 ** 2) / torch.mean((img1 - img2) ** 2))


def PSNR_GPU(img1, img2):
    mpsnr = 0
    for l in range(img1.size()[1]):
        mpsnr += 10. * torch.log10(1. / torch.mean((img1[:, l, :, :] - img2[:, l, :, :]) ** 2))

    return mpsnr / img1.size()[1]

    # return 10. * torch.log10((torch.max(img1)**2) / torch.mean((img1 - img2) ** 2))


def SAM(pred, gt):
    # pred = pred.numpy()
    # gt = gt.numpy()
    eps = 2.2204e-16
    pred[np.where(pred == 0)] = eps
    gt[np.where(gt == 0)] = eps

    nom = sum(pred * gt)
    denom1 = sum(pred * pred) ** 0.5
    denom2 = sum(gt * gt) ** 0.5
    sam = np.real(np.arccos(nom.astype(np.float32) / (denom1 * denom2 + eps)))
    sam[np.isnan(sam)] = 0
    sam_sum = np.mean(sam) * 180 / np.pi
    return sam_sum

def SAM_torch(pred, gt):
    eps = torch.tensor(2.2204e-16)

    pred[pred == 0] = eps
    gt[gt == 0] = eps

    nom = torch.sum(pred * gt)
    denom1 = torch.sqrt(torch.sum(pred * pred))
    denom2 = torch.sqrt(torch.sum(gt * gt))

    sam = torch.real(torch.acos(nom / (denom1 * denom2 + eps)))
    sam[torch.isnan(sam)] = 0
    sam_sum = torch.mean(sam) * 180 / torch.tensor(np.pi)
    return sam_sum


def plot():
    path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/icvl_train.log'
    psnr = []
    sam = []

    with open(path, 'r') as f:

        for line in f.readlines():

            line = line.strip()

            if 'val averagr psnr : ' in line:
                line = line.split(' ')
                psnr.append(float(line[-4]))
                sam.append(float(line[-1]))

    epochs = [i for i in range(len(psnr))]

    fib_size = (5, 4)
    fon_size = 12

    plt.figure(figsize=fib_size)
    plt.title('sam of every epoch', fontsize=fon_size)
    plt.xlabel('epoch', fontsize=fon_size)
    plt.ylabel('sam', fontsize=fon_size)
    plt.plot(epochs, sam, 'k.')
    plt.grid(True, linestyle="-.", color="k", linewidth="1.1")
    plt.savefig(OUT_DIR.joinpath('icvl_sam.png'))

    plt.figure(figsize=fib_size)
    plt.title('psnr of every epoch', fontsize=fon_size)
    plt.xlabel('epoch', fontsize=fon_size)
    plt.ylabel('psnr', fontsize=fon_size)
    plt.plot(epochs, psnr, 'k.')
    plt.grid(True, linestyle="-.", color="k", linewidth="1.1")
    plt.savefig(OUT_DIR.joinpath('icvl_psnr.png'))


def save_mat(l=31, w=144, h=144, fis=144):
    path = '/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/weight/icvl_test_fake_hr.pth'
    base_path = Path('/home/yons/data1/chenzhuang/HSI-SR/GAN-HSI-SR/data')
    data = torch.load(path)

    count = 0
    for i in range(8):

        img = torch.zeros([31, 144 * 9, 144 * 8])

        for x in range(0, 1372 - fis, fis):
            for y in range(0, 1174 - fis, fis):
                img[:, x:x + fis, y:y + fis] = data[count]
                count += 1

        img = interpolate(
            img.reshape(1, img.shape[0], img.shape[1], img.shape[2]),
            scale_factor=1,
            mode='bicubic'
        )
        img = torch.squeeze(img)
        img = img.numpy()
        img *= (255)
        img = img.astype(np.uint8)

        im = Image.fromarray(img[27])
        im = im.rotate(180)
        im.save(base_path.joinpath('process_icvl_img{}.png'.format(i)))

        io.savemat(base_path.joinpath('process_icvl_img{}.mat'.format(i)), {'data': img})


# save_mat()

def get_paths():
    PATH = './data/'
    train_paths = []
    val_paths = []
    test_paths = []

    with open(PATH+ 'train_name.txt', 'r') as f:
        for i in f.readlines():
            train_paths.append(PATH + i.strip())

    with open(PATH+ 'val_name.txt', 'r') as f:
        for i in f.readlines():
            val_paths.append(PATH + i.strip())

    with open(PATH+ 'test_name.txt', 'r') as f:
        for i in f.readlines():
            test_paths.append(PATH + i.strip())

    return train_paths, val_paths, test_paths


def save_icvl():
    train_paths, val_paths, test_paths = get_paths()

    train_data = np.zeros((len(train_paths, 31, 144, 144)))
    val_data = np.zeros((len(val_paths, 31, 144, 144)))
    test_data = np.zeros((len(test_paths, 31, 144, 144)))

    for i in range(len(train_paths)):
        img = h5py.File(paths[i], 'r')['rad']
        img = np.array(img)
        img /= 4095.0
    pass


class TrainsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(TrainsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.flag = False
        if self.flag:
            # 把文件先都读取到CPU
            self.img = []
            print("kai shi du qu shu ju xun lian shu ju le")
            for i in range(len(self.image_filenames)):
                if i%1000==0:
                    print(i)
                mat = scio.loadmat(self.image_filenames[i], verify_compressed_data_integrity=False)
                self.img.append(mat)
            print("gong xi ni !!! shu ju du qu cheng gong le!!!")


    def __getitem__(self, index):
        if self.flag:
            mat = self.img[index]
        else:
            mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)
        input = mat['lr'].astype(np.float32)
        label = mat['hr'].astype(np.float32)

        return torch.from_numpy(input), torch.from_numpy(label)

    def __len__(self):
        return len(self.image_filenames)


class ValsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(ValsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index])
        input = mat['LR'].astype(np.float32).transpose(2, 0, 1)
        label = mat['HR'].astype(np.float32).transpose(2, 0, 1)

        return torch.from_numpy(input).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.image_filenames)


