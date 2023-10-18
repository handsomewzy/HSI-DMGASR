import os
from os import listdir
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from option import opt
from utils import is_image_file, ValsetFromFolder
from MCNet import MCNet
import scipy.io as scio
from eval import PSNR, SSIM, SAM
from SSPSR import *
import metrics as Metrics
from eval_hsi import quality_assessment
from HStest import HSTestData
from models_GDRRN import *
from GELIN import *
from CEGATSR import *
from EDSR import *
import torch.utils.data as data
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image

# python -m torch.distributed.launch --nproc_per_node=4 --master_port='29511' --use_env
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

        input = input[:,:50,:50]
        label = label[:,:200,:200]

        img_HR = torch.from_numpy(label)
        img_LR = torch.from_numpy(input)
        # 这里是上采样了四倍，具体情况改变这个数值。
        # img_LR_1 = img_LR.reshape(1,3,32,32)
        print(img_LR.shape)
        img_LR_1 = img_LR.reshape(1,102,50,50) # 除了PaviaC数据集，别的都是512，512
        
        img_SR = torch.nn.functional.interpolate(img_LR_1, scale_factor=8, mode='bicubic')

        return {'HR': img_HR, 'SR': img_SR[0], 'LR': img_LR}

    def __len__(self):
        return len(self.image_filenames)

def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp


class Bicubic(nn.Module):
    def __init__(self):
        super(Bicubic, self).__init__()

    def forward(self, x):
        x = interpolate(
            x,
            scale_factor=2,  # 这个与具体的超分辨比例有关，这个是全局skip时候，对初始图像进行上采样，一般设置为2 3 4
            mode='bicubic',
            align_corners=True
        )
        return x


def main():
    # input_path = '../Harvard_4_test/'
    # input_path = '../test/Cave/4/'
    # input_path = '../test/Foster/4/'
    input_path = '../test/Chikusei/4/'
    # input_path = '../test/PaviaC/4/'
    out_path = './result'

    PSNRs = []
    SSIMs = []
    SAMs = []

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print(opt.cuda)

    # if opt.cuda:
    #     print("=> use gpu id: '{}'".format(opt.gpus))
    #     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    #     if not torch.cuda.is_available():
    #         raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    # dist.init_process_group(backend='nccl')
    # local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # 加载模型并读取
    # SSPSR模型读取
    # model = torch.load('./weight/SSPSR_4_Harvard.pth')
    # print(model)

    # MCNet模型读取
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_name = 'EDSR'
    dataset_name = 'Har'
    model = torch.load('./weight/{}_3_{}.pth'.format(model_name, dataset_name))
    model = model.to(device)  
    print(model_name, dataset_name)


    # 双线性插值，基础baseline网络
    # model = Bicubic()
    # model = model.to(device)


    
    # model.load_state_dict(checkpoint["model"])
    # print(model)
    images_name = [x for x in listdir(input_path) if is_image_file(x)]

    # ------------------------------------------------------------------------------
    #                                   4倍数据集，八倍数据集，注意区分
    # ------------------------------------------------------------------------------
    # val_set = TestsetFromFolder('../Harvard_4_test/')
    # val_set = TestsetFromFolder('../test/Cave/4/')
    # val_set = TestsetFromFolder('../test/Foster/4/')
    # val_set = TestsetFromFolder('../test/Chikusei/8/')
    # val_set = TestsetFromFolder('../test/PaviaC/4/')
    # val_set = HSTestData(image_dir= '../Chikusei_mat/128test/', n_scale = 4)
    val_set = HSTestData(image_dir= '../Harvard_mat/test/', n_scale = 3)
    print(len(val_set))
    val_loader = DataLoader(dataset=val_set,  batch_size=1, shuffle=False)
    # model_b = Bicubic().to(device)

    for index,data in enumerate(val_loader):                                                               
        lr = data['LR'].to(device)
        lms = data['SR'].to(device)
        gt = data['HR'].to(device)
        # ------------------------------------------------------------------------------
        #                                   不同模型不同输入格式
        # ------------------------------------------------------------------------------
        # output = model(lr, lr) # SSPSR
        # output = model(lr, lms) # GELIN CEGAT
        output = model(lr) # EDSR MCNet
        # output = lms
        # output = model(lms)  # GDRRN
        save_img = False
        result_path = './result/{}_{}_3'.format(dataset_name, model_name)
        os.makedirs(result_path, exist_ok=True)

        y = output[0].cpu().detach().numpy().transpose(1, 2, 0)
        gt = gt[0].cpu().detach().numpy().transpose(1, 2, 0)
        y[y < 0] = 0
        y[y > 1.] = 1.
        print(y.shape, gt.shape)
        # y = color_correction(gt, y)
        if index == 0:
            indices = quality_assessment(gt, y, data_range=1., ratio=4)
            np.save('{}/{}_{}_hr.npy'.format(result_path,  model_name, dataset_name), gt)
            np.save('{}/{}_{}_sr.npy'.format(result_path,  model_name, dataset_name), y)
        else:
            indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
        print(indices)


        if save_img:
            hr_img,lr_img = gt, lr[0].cpu().detach().numpy().transpose(1, 2, 0)
            # hr_img, lr_img = data['HR'][0].numpy().transpose(1, 2, 0), data['LR'][0].numpy().transpose(1, 2, 0)
            hr_img,lr_img = (hr_img * 255.0).round(), (lr_img * 255.0).round()
            sr_img = Metrics.tensor2img(output)
            err_img = np.abs(hr_img - sr_img) / 255.0

            print(hr_img.shape)

            Metrics.save_img(
                hr_img, '{}/{}_hr.png'.format(result_path,  index))
            Metrics.save_img(
                sr_img, '{}/{}_sr.png'.format(result_path,  index))
            Metrics.save_img(
                lr_img, '{}/{}_lr.png'.format(result_path,  index))
            # Metrics.save_img(
            #     err_img, '{}/{}_err.png'.format(result_path,  index))

            # hr_img = Image.open('{}/{}_hr.png'.format(result_path,  index)).convert('L')
            # sr_img = Image.open('{}/{}_sr.png'.format(result_path,  index)).convert('L')
            # err_img = np.abs(np.array(hr_img) - np.array(sr_img)) / 255.0
            # 将误差图像转换为彩色图像并保存
            plt.imshow(err_img[:,:,[15]], cmap='jet')
            plt.axis('off')
            plt.savefig('{}/{}_err.png'.format(result_path,  index))  # 保存图像

    # print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(PSNRs), np.mean(SSIMs),
    #                                                                            np.mean(SAMs)))
    # 平均一下。
    for index in indices:
        indices[index] = indices[index] / len(val_loader)
    print("最终的结果平均指标为 {}".format(indices))


if __name__ == "__main__":
    main()
