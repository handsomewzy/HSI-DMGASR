import imageio
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import preprocessing
import h5py
from scipy import io, misc
from torch.nn.functional import interpolate

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt

        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']

        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = False
        self.radiation_augmentation = False
        self.mixture_augmentation = False
        self.center_pixel = True
        supervision = 'full'

        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
        # np.random.shuffle(self.indices)

    def down_sample(self, data, s=4):
        # TODO: 添加高斯噪声(0.01) 并降采样
        # data = data + 0.0000001*torch.randn(*(data.shape))

        data = interpolate(
            data,
            scale_factor=1 / s,
            mode='bicubic',
            align_corners=True
        )
        return data

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN

        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        # self.data_HR = data
        # self.data_LR = self.down_sample(data)

        return data, self.down_sample(data)


class LoadData(torch.utils.data.Dataset):

    def __init__(self, path, s=4, fis=144):
        # num 31 512 512
        # self.data = np.load(path)
        self.data = path
        self.data = torch.from_numpy(self.data)
        self.data /= (2 ** 16 - 1)
        # print(torch.max(self.data))

        # TODO: 先边缘裁剪 以获取HR
        shape = self.data.shape
        # print(shape)
        # self.data = self.data[:,:,(s+6):shape[2]-(s+6),(s+6):shape[3]-(s+6)]

        # 取三张
        # 32*3 31 144 144
        self.HR = torch.zeros((shape[0] * 9, 31, 144, 144))

        count = 0
        for i in range(shape[0]):
            for x in range((s + 6), shape[2] - (s + 6) - fis, fis):
                for y in range((s + 6), shape[2] - (s + 6) - fis, fis):
                    self.HR[count] = self.data[i, :, x:x + fis, y:y + fis]
                    count += 1
        # 得到LR图像 num*9 31 36 36
        # print(count)
        self.LR = self.down_sample(self.HR)

    def down_sample(self, data, s=4):
        # TODO: 添加高斯噪声(0.01) 并降采样
        # data = data + 0.0000001*torch.randn(*(data.shape))

        data = interpolate(
            data,
            scale_factor=1 / s,
            mode='bicubic',
            align_corners=True
        )

        return data

    def __len__(self):
        return self.HR.shape[0]

    def __getitem__(self, index):
        return self.LR[index], self.HR[index]




if __name__ == "__main__":
    data1 = io.loadmat("./data/img1.mat")
    print(data1.keys())
    print(data1['ref'])
    data = h5py.File('./data/4cam_0411-1640-1.mat', 'r')
    print(data.keys())
    print(data['rad'])
    data_np = np.array(data['rad'])
    print(data_np.shape)
    data_np = data_np.transpose(1, 2, 0)

    # dataset = LoadData(data_np)

    gt = np.zeros((1392, 1300))
    new_dataset = HyperX(data=data_np, gt=gt, patch_size=1200, dataset='256', ignored_labels=[2])
    print(len(new_dataset))
    dataloader1 = DataLoader(new_dataset, batch_size=64, drop_last=True)
    print(len(dataloader1))
    for i,j in dataloader1:
        print(i.shape)
        print(j.shape)
    # for i, j in new_dataset:
    #     print(i.shape)
    #     print(j.shape)

    # train_data = DataLoader(
    #             new_dataset,
    #             batch_size=64,
    #             shuffle=True,
    #             num_workers=2,
    #             pin_memory=True,
    #             drop_last=True,
    #         )
    # for i,j in train_data:
    #     print(i,j)
    # print(len(train_data))
