import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import cv2
import utils
import random
from imsize import imresize
import h5py

class HSTrainingData(data.Dataset):
    def __init__(self, image_dir, n_scale, num_ch=None, augment=None, ch3=False):
        self.image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.augment = augment
        self.n_scale = n_scale
        self.i = num_ch
        self.ch3 = ch3
        if self.augment:
            self.factor = 8
        else:
            self.factor = 1
        
        

    def __getitem__(self, index):
        file_index = index
        aug_num = 0
        if self.augment:
            file_index = index // self.factor
            aug_num = int(index % self.factor)
        load_dir = self.image_files[file_index]
        # print(load_dir)
        
        data = sio.loadmat(load_dir)
        # data = h5py.File(load_dir,'r')
        #print(load_dir)
        img = np.array(data['block'][...], dtype=np.float32) # chikusei paviac 
        # img = np.array(data['gt'][...], dtype=np.float32) # cave

        # img = np.array(data['ref'][...]), #harvard
        # img = img[0] # 这行是一起的

        # print(img.shape)
        
        # Chikusei 归一化
        '''
        DATAMAX = 15133
        DATAMIN = 0
        img = (img - DATAMIN)/(DATAMAX - DATAMIN)
        '''
        # Chikusei 归一化
        img = (img - img.min()) / (img.max() - img.min())
        
        height, width, channels = img.shape
        gt_size = 32 * self.n_scale
        row = random.randint(0, height-gt_size)
        column = random.randint(0, width-gt_size)
        gt = img[row:row+gt_size, column:column+gt_size, :] 
        
        ms = imresize(gt,output_shape=(32,32))
        #sprint(ms.shape)
        lms = imresize(ms,output_shape=(gt_size,gt_size))
        
        ms, lms, gt = utils.data_augmentation(ms, mode=aug_num), utils.data_augmentation(lms, mode=aug_num), \
                        utils.data_augmentation(gt, mode=aug_num)

        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        ms = ms.type(torch.FloatTensor)
        lms = lms.type(torch.FloatTensor)
        gt = gt.type(torch.FloatTensor)
        
        ms = torch.clamp(ms,0,1)
        lms = torch.clamp(lms,0,1)

        if self.ch3:
            # 选取三通道进行测试。
            gt = gt[[self.i, self.i + 34, self.i + 68],:,:]
            ms = ms[[self.i, self.i + 34, self.i + 68],:,:]
            lms = lms[[self.i, self.i + 34, self.i + 68],:,:]

        # return ms, lms, gt
        return {'HR': gt, 'SR': lms, 'LR': ms}

    def __len__(self):
        return len(self.image_files)*self.factor


# dataset = HSTrainingData(image_dir= '../PaviaC_mat/', n_scale = 4, num_ch=1,ch3=True)
# print(dataset.image_files)
# print(len(dataset))
# for i in dataset:
#     print(i['HR'].shape,i['LR'].shape,i['SR'].shape)