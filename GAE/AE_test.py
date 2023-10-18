import os
from os import listdir
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from option import opt
from utils import is_image_file
from MCNet import MCNet
import scipy.io as scio
from eval import PSNR, SSIM, SAM
from SSPSR import *
import metrics as Metrics
from AE import *
from eval_hsi import color_correction, quality_assessment
from HStest import HSTestData
import torch.utils.data as data
from os.path import join

class TestsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(TestsetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # 把文件先都读取到CPU
        self.img = []
        print("kai shi du qu shu ju ce shi shu ju le")
        for i in range(len(self.image_filenames)):
            print(i)
            mat = scio.loadmat(self.image_filenames[i], verify_compressed_data_integrity=False)
            self.img.append(mat)
        print("gong xi ni !!! shu ju du qu cheng gong le!!!")


    def __getitem__(self, index):
        # mat = scio.loadmat(self.image_filenames[index])
        mat = self.img[index]
        input = mat['LR'].astype(np.float32).transpose(2, 0, 1)
        label = mat['HR'].astype(np.float32).transpose(2, 0, 1)

        input = input[:,:16,:16]
        label = label[:,:128,:128]

        img_HR = torch.from_numpy(label)
        img_LR = torch.from_numpy(input)
        # 这里是上采样了四倍，具体情况改变这个数值。
        # img_LR_1 = img_LR.reshape(1,3,32,32)
        print(img_LR.shape)
        img_LR_1 = img_LR.reshape(1,128,16,16) # 除了PaviaC数据集，别的都是512，512
        
        img_SR = torch.nn.functional.interpolate(img_LR_1, scale_factor=8, mode='bicubic')

        return {'HR': img_HR, 'SR': img_SR[0], 'LR': img_LR}

    def __len__(self):
        return len(self.image_filenames)


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)
# input_path = '../Harvard_4_test/'
# input_path = '../test/Cave/4/'
# input_path = '../test/Foster/4/'
# input_path = '../test/Chikusei/4/'
# input_path = '../test/PaviaC/4/'
out_path = './result'


# val_set = TestsetFromFolder('../Harvard_8_test/')
# val_set = TestsetFromFolder('../test/Cave/4/')
# val_set = TestsetFromFolder('../test/Foster/4/')
# val_set = TestsetFromFolder('../test/Chikusei/4/')
# val_set = TestsetFromFolder('../test/PaviaC/8/')

val_set = HSTestData(image_dir= '../PaviaC_mat/128test/', n_scale = 4)
val_loader = DataLoader(dataset=val_set,  batch_size=1, shuffle=False)

# 读取模型
# model = torch.load('./weight_64_32_total/GAE_4_Pav.pth',map_location='cpu')
model = torch.load('./weightGAE/GAE_4_Pav80.pth',map_location='cpu')
# model = torch.load('./duichen_GAE/GAE_4_Pav.pth',map_location='cpu')
model = model.to(device)
print(model)


# images_name = [x for x in listdir(input_path) if is_image_file(x)]

avg_psnr = 0.0
avg_ssim = 0.0
avg_sam = 0.0
# val_set = HSTestData(image_dir= '../PaviaC_mat/test/', n_scale = 4, ch3=False, num_ch=10)
# val_loader = DataLoader(dataset=val_set,  batch_size=1, shuffle=True)

for index,data in enumerate(val_loader):                                                               
    lr = data['LR'].to(device)
    lms = data['SR'].to(device)
    gt = data['HR'].to(device)

# for index in range(len(images_name)):

    # print(input_path)
    # mat = scio.loadmat(input_path + images_name[index])
    # # print(mat['HR'].shape)
    # hyperHR = mat['HR'].transpose(2, 0, 1).astype(np.float32)
    # gt = Variable(torch.from_numpy(hyperHR).float(), volatile=True).contiguous().view(1, -1, hyperHR.shape[1],
                                                                                            # hyperHR.shape[2])
    if opt.cuda:
        gt = gt.to(device)
    # gt = gt[:,:,:256,:256]
    print(gt.shape)
    # z = model_E(input)
    # x_recon = model_D(z)

    # x_recon,z_list = model(gt)
    # print(model.encode(lms)[0].shape)
    z_list = model.encode(gt)
    print(len(z_list), z_list[0].shape, gt.shape)
    x_recon = model.decode(gt, z_list)
    # x_recon = model.decode(z_list)
    print(x_recon.shape)

    # del model,model_E
    result_path = './result/Chi_AE_4'
    os.makedirs(result_path, exist_ok=True)

    # 对前后的图像进行指标测试
    # x_recon_np = x_recon[0].cpu().detach().numpy().transpose(1,2,0)
    # x_recon_np[x_recon_np < 0] = 0
    # x_recon_np[x_recon_np > 1.] = 1.
    # x_recon_np = color_correction(mat['HR'], x_recon_np)



    # 颜色修正 HWC,注意这个位置
    y = x_recon[0].cpu().detach().numpy().transpose(1, 2, 0)
    gt = gt[0].cpu().detach().numpy().transpose(1, 2, 0)
    y[y < 0] = 0
    y[y > 1.] = 1.
    print(y.shape, gt.shape)
    # y = color_correction(gt, y)
    if index == 0:
        indices = quality_assessment(gt, y, data_range=1., ratio=4)
    else:
        indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
    print(indices)
    
    # eval_psnr = PSNR(x_recon_np, mat['HR'])
    # eval_ssim = SSIM(x_recon_np, mat['HR'])
    # eval_sam = SAM(x_recon_np, mat['HR'])
    # print("第 {} 张图片的 PSNR = {} , SSIM = {} , SAM = {}  ".format(index,eval_psnr,eval_ssim,eval_sam))

    # avg_psnr += eval_psnr
    # avg_ssim += eval_ssim
    # avg_sam += eval_sam

    # hr_img,lr_img = gt, lr[0].cpu().detach().numpy().transpose(1, 2, 0)
    # hr_img,lr_img = mat['HR'], mat['LR']
    hr_img,lr_img = gt, lr[0].cpu().detach().numpy().transpose(1, 2, 0)
    hr_img,lr_img = (hr_img * 255.0).round(), (lr_img * 255.0).round()
    recon_img = Metrics.tensor2img(x_recon)
    # z_img = Metrics.tensor2img(z_list[2])
    # print(z_img.shape)

    Metrics.save_img(
            hr_img, '{}/{}_hr.png'.format(result_path,  index))
    Metrics.save_img(
            recon_img, '{}/{}_recon.png'.format(result_path,  index))
    # for i in range(len(z_list)):
    #     z_img = Metrics.tensor2img(z_list[i])
    #     Metrics.save_img(z_img, '{}/{}_{}_z.png'.format(result_path,  index, i))

    # Metrics.save_img3(
            # z_img, '{}/{}_z.png'.format(result_path,  index))
    print("zhe lun hao la!!!")

else:
    # 平均一下。
    for index in indices:
        indices[index] = indices[index] / len(val_loader)
    print("最终的结果平均指标为 {}".format(indices))
    # print("测试集的平均指标为 PSNR = {} , SSIM = {} , SAM = {}  ".format(avg_psnr/len(images_name),avg_ssim/len(images_name),avg_sam/len(images_name)))

        
