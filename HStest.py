import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import cv2
from imsize import imresize
import h5py

class HSTestData(data.Dataset):
    def __init__(self, image_dir, n_scale, num_ch=None, ch3=False):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.n_scale = n_scale
        self.i = num_ch
        self.ch3 = ch3

    def __getitem__(self, index):
        file_index = index
        load_dir = self.image_files[file_index]
        #print(load_dir)
        # data = sio.loadmat(load_dir)
        #data = h5py.File(load_dir,'r')
        # img = np.array(data['block'][...], dtype=np.float32) # chikusei paviac 
        # img = np.array(data['gt'][...], dtype=np.float32) # cave

        # data = sio.loadmat(load_dir)
        # img = np.array(data['ref'][...]), #harvard
        # img = img[0] # 这行是一起的

        img = np.load(load_dir)

        # 512
        gt_size = 64  # fixed
        # Chikusei 归一化
        '''
        DATAMAX = 15133
        DATAMIN = 0
        img = (img - DATAMIN)/(DATAMAX - DATAMIN)
        '''
        img = (img - img.min()) / (img.max() - img.min())

        # gt_size = 128
        gt = img[0:gt_size, 0:gt_size, :]
        ms = imresize(gt,output_shape = (gt_size // self.n_scale, gt_size // self.n_scale))
        lms = imresize(ms,output_shape =(gt_size, gt_size))

        # PC 数据集
        # gt = img[0:219, 0:238, :]
        # ms = imresize(gt,output_shape = (219 // self.n_scale, 238 // self.n_scale))
        # lms = imresize(ms,output_shape =(219, 238))

        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        ms = ms.type(torch.FloatTensor)
        lms = lms.type(torch.FloatTensor)
        gt = gt.type(torch.FloatTensor)
        ms = torch.clamp(ms,0,1)
        lms = torch.clamp(lms,0,1)

        if self.ch3:
            x = 34 # 34,42,10
            # 选取三通道进行测试。
            gt = gt[[self.i, self.i + x, self.i + 2*x],:,:]
            ms = ms[[self.i, self.i + x, self.i + 2*x],:,:]
            lms = lms[[self.i, self.i + x, self.i + 2*x],:,:]
        # return ms, lms, gt
        return {'HR': gt, 'SR': lms, 'LR': ms}

    def __len__(self):
        return len(self.image_files)



# dataset = HSTestData(image_dir= '../PaviaC_mat/', n_scale = 4, num_ch=1,ch3=True)
# print(dataset.image_files)
# print(len(dataset))
# for i in dataset:
#     print(i['HR'].shape,i['LR'].shape,i['SR'].shape)