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
from AE import Encoder, Decoder
from eval_hsi import color_correction, quality_assessment

def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# input_path = '../Harvard_4_test/'
input_path = '../test/Cave/4/4/'
out_path = './result'
dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
# 读取模型
# model_E = torch.load('./weight/E_VGGSAM2_4_Harvard.pth',map_location='cuda:0')
# model_D = torch.load('./weight/D_VGGSAM2_4_Harvard.pth',map_location='cuda:0')

model_E = torch.load('./weight/E_maskl1_4_CAVE.pth',map_location='cuda:0')
model_D = torch.load('./weight/D_maskl1_4_CAVE.pth',map_location='cuda:0')

model_E = model_E.to(device)
model_D = model_D.to(device)

images_name = [x for x in listdir(input_path) if is_image_file(x)]

avg_psnr = 0.0
avg_ssim = 0.0
avg_sam = 0.0
for index in range(len(images_name)):

    mat = scio.loadmat(input_path + images_name[index])
    hyperHR = mat['HR'].transpose(2, 0, 1).astype(np.float32)
    input = Variable(torch.from_numpy(hyperHR).float(), volatile=True).contiguous().view(1, -1, hyperHR.shape[1],
                                                                                            hyperHR.shape[2])
    if opt.cuda:
        input = input.to(device)
    
    z = model_E(input)
    x_recon = model_D(z)

#     result_path = './result/Harvard_AE_test_4'
    result_path = './result/CAVE_AE_test_4'
    os.makedirs(result_path, exist_ok=True)

    # 对前后的图像进行指标测试
    x_recon_np = x_recon[0].cpu().detach().numpy().transpose(1,2,0)
    x_recon_np[x_recon_np < 0] = 0
    x_recon_np[x_recon_np > 1.] = 1.
    x_recon_np = color_correction(mat['HR'], x_recon_np)



    # 颜色修正 HWC,注意这个位置
#     y = x_recon[0].cpu().detach().numpy().transpose(1, 2, 0)
#     gt = mat['HR']
#     y = color_correction(gt, y)
#     if index == 0:
#         indices = quality_assessment(gt, y, data_range=1., ratio=4)
#     else:
#         indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
#     print(indices)
    
    eval_psnr = PSNR(x_recon_np, mat['HR'])
    eval_ssim = SSIM(x_recon_np, mat['HR'])
    eval_sam = SAM(x_recon_np, mat['HR'])
    print("第 {} 张图片的 PSNR = {} , SSIM = {} , SAM = {}  ".format(index,eval_psnr,eval_ssim,eval_sam))

    avg_psnr += eval_psnr
    avg_ssim += eval_ssim
    avg_sam += eval_sam

    hr_img,lr_img = mat['HR'],mat['LR']
    hr_img,lr_img = (hr_img * 255.0).round(), (lr_img * 255.0).round()
    recon_img = Metrics.tensor2img(x_recon)
    z_img = Metrics.tensor2img(z)

    Metrics.save_img(
            hr_img, '{}/{}_hr.png'.format(result_path,  index))
    Metrics.save_img(
            recon_img, '{}/{}_recon.png'.format(result_path,  index))
    Metrics.save_img3(
            z_img, '{}/{}_z.png'.format(result_path,  index))
    print("zhe lun hao la!!!")

else:
    print("测试集的平均指标为 PSNR = {} , SSIM = {} , SAM = {}  ".format(avg_psnr/len(images_name),avg_ssim/len(images_name),avg_sam/len(images_name)))

        
