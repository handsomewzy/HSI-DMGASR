import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import scipy.io as scio
from torch.utils.data import DataLoader
from AE import *
from SSPSR import *
from eval_hsi import quality_assessment, color_correction
from HStest import HSTestData
from HStrain import HSTrainingData
from GELIN import HLoss
import torch.optim as optim
import torch.nn as nn
from unet import UNet
import matplotlib.pyplot as plt
from PIL import Image
import time

def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])

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
    # ---------------------------------------------------------------------------------------------
    #                                       不同数据集尺寸设置！！！！
    # ---------------------------------------------------------------------------------------------
        input = input[:,:32,:32]
        label = label[:,:128,:128]

        img_HR = torch.from_numpy(label)
        img_LR = torch.from_numpy(input)
        # 这里是上采样了四倍，具体情况改变这个数值。
        # img_LR_1 = img_LR.reshape(1,3,32,32)
        print(img_LR.shape)
        # img_LR_1 = img_LR.reshape(1,102,32,32) # 除了PaviaC数据集，别的都是512，512
        img_LR_1 = torch.unsqueeze(img_LR, 0)
        
        img_SR = torch.nn.functional.interpolate(img_LR_1, scale_factor=4, mode='bicubic')

        return {'HR': img_HR, 'SR': img_SR[0], 'LR': img_LR}

    def __len__(self):
        return len(self.image_filenames)


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


    def __getitem__(self, index): # CHW
        if self.flag:
            mat = self.img[index]
        else:
            mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)

        input = mat['lr'].astype(np.float32)
        label = mat['hr'].astype(np.float32)
    # ---------------------------------------------------------------------------------------------
    #                                       不同数据集尺寸设置！！！！
    # ---------------------------------------------------------------------------------------------
        # 选取三通道进行测试。
        # input = input[[3,13,23],:,:]
        # label = label[[3,13,23],:,:]

        img_HR = torch.from_numpy(label)
        img_LR = torch.from_numpy(input)
        # 这里是上采样了四倍，具体情况改变这个数值。上采样要多一个通道，四维度的，注意这里的通道数
        # print(img_LR.shape)
        # img_LR_1 = img_LR.reshape(1,102,16,16)
        img_LR_1 = torch.unsqueeze(img_LR, 0)
        img_SR = torch.nn.functional.interpolate(img_LR_1, scale_factor=4, mode='bicubic')

        return {'HR': img_HR, 'SR': img_SR[0], 'LR': img_LR}

    def __len__(self):
        return len(self.image_filenames)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    # ---------------------------------------------------------------------------------------------
    #                                       训练还是验证设置！！！！
    # ---------------------------------------------------------------------------------------------
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None
    
    # ---------------------------------------------------------------------------------------------
    #                                       数据集设置！！！！
    # ---------------------------------------------------------------------------------------------

    # train_set = TrainsetFromFolder('../Harvard_4_train/') # 数据集有两个，第一个是input，人为制造的LR样本，第二个是label，HR样本，注意顺序
    # train_set = TrainsetFromFolder('../train/Cave/4/')
    # train_set = TrainsetFromFolder('../train/Foster/4/')
    # train_set = TrainsetFromFolder('../train/Chikusei/4/')
    # train_set = TrainsetFromFolder('../train/PaviaC/4/')
    train_set = HSTrainingData(image_dir= '../PaviaC_mat/train/', n_scale = 4, augment=True, ch3=False, num_ch=0)
    print(len(train_set))

    train_loader = DataLoader(dataset=train_set,  batch_size=4, shuffle=True) # 分布式不能进行shuffle

    # val_set = TestsetFromFolder('../Harvard_4_test/')
    # val_set = TestsetFromFolder('../test/Cave/4/')
    # val_set = TestsetFromFolder('../test/Foster/4/')
    # val_set = TestsetFromFolder('../test/Chikusei/4/')
    # val_set = TestsetFromFolder('../test/PaviaC/4/')
    val_set = HSTestData(image_dir= '../PaviaC_mat/128test/', n_scale = 4)
    print(len(val_set))
    
    val_loader = DataLoader(dataset=val_set,  batch_size=1, shuffle=False)

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            # 首先对数据进行降维，读取AE模型，然后再进行扩散训练
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            # 读取模型
            # model_GAE_SR = torch.load('./GAE_4_Har.pth',map_location='cuda:0')
            # 其中一个编码器的配置文件，loss与优化器
            # optimizer_AE = optim.Adam(model_GAE_SR.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
            # criterion = nn.L1Loss().to(device)

            for _, train_data in enumerate(train_loader):
                train_data['HR'] = train_data['HR'].to(device)
                train_data['SR'] = train_data['SR'].to(device)
    # ---------------------------------------------------------------------------------------------
    #                                       GAE的模型读取设置！！！！
    # ---------------------------------------------------------------------------------------------
                # model_GAE = torch.load('./duichen_GAE/GAE_4_Pav.pth',map_location='cuda:0')
                model_GAE = torch.load('./weight_64_32_total/GAE_4_Pav.pth',map_location='cuda:0')
                # model_GAE = torch.load('./AEduichen_4_Pav.pth',map_location='cuda:0')
                # model_GAE = torch.load('./GAE_4_Pav124.pth',map_location='cuda:0')
                model_GAE = model_GAE.to(device)
                # model_E = model_E.to(device)
                zHR_list = model_GAE.encode(train_data['HR'])
                zSR_list = model_GAE.encode(train_data['SR'])
                # print(len(zHR_list),zHR_list[0].shape)

                # 优化SRGAE的参数
                # for i in range(len(zHR_list)):
                #     if i==0:
                #         loss_GAE = criterion(zHR_list[i].clone(),zSR_list[i].clone())
                #     else:
                #         loss_GAE += criterion(zHR_list[i].clone(),zSR_list[i].clone())

                # GAE把一张31通道的图像，中间隐藏层映射为几张3通道的图像，遍历进行训练。
                for i in range(len(zHR_list)):
                    train_data['HR'] = zHR_list[i]
                    train_data['SR'] = zSR_list[i]
                    # print(train_data['HR'].shape. train_data['SR'].shape)
                    diffusion.feed_data(train_data)
                    diffusion.optimize_parameters()

                # optimizer_AE.zero_grad()
                # loss_GAE.backward(retain_graph=True)
                # optimizer_AE.step()

                print("第{}轮训练完成了，给你点反应好吧。没卡，在训练。loss_GAE =".format(current_step))
                current_step += 1
                if current_step > n_iter:
                    break

                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    # 释放无关变量
                    del train_data, zHR_list, zSR_list
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    total_time = 0.0
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        # 提前把SR上采样的图片存储一下，不然后面存取的时候总是出现问题。
                        row_data = val_data['SR']
                        # row_data = row_data.to(device)
                        # print(row_data['SR'].shape)

                        # 相应的，验证的时候要先映射，然后结束之后查看的时候再还原。
                        # val_data['HR'] = val_data['HR'].to(device)
                        val_data['SR'] = val_data['SR'].to(device)


                        zSRval_list = model_GAE.encode(val_data['SR'])
                        new_list = []
                        start_time = time.time()
                        for i in range(len(zSRval_list)):
                            val_data['SR'] = zSRval_list[i]
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=False)
                            visuals = diffusion.get_current_visuals()
                            visuals['SR'] = visuals['SR'].to(device)
                            visuals['SR'] = torch.unsqueeze(visuals['SR'], 0)
                            # print(visuals['SR'].shape) # 三维的，后续解码要再增加一个维度，但是之前四倍为什么不需要？？？
                            new_list.append(visuals['SR'])
                        # new_list = new_list.to(device)
                        # 注意注意，此处出现问题，之前四倍的时候，new——list里面存的是三维，例如（3，128，128），是不包含batch size维度的
                        # 但是新的八倍依旧这么设置就出错了。不知道为什么啊？？？
                        # print(new_list[0].shape, row_data.shape)
                        visuals['SR'] = model_GAE.decode(row_data, new_list)
                        end_time = time.time()
                        print(visuals['SR'].shape)
                        elapsed_time = end_time - start_time
                        print(f"函数运行时间为：{elapsed_time} 秒")
                        total_time = total_time + elapsed_time

                        # 输入计算时候需要是HWC的形式。全新的计算，从GELIN里面拿的，之前的从MCNet，感觉不好。
                        y = visuals['SR'][-1].cpu().detach().numpy().transpose(1, 2, 0)
                        gt = visuals['HR'][0].cpu().detach().numpy().transpose(1, 2, 0)
                        # gt = y
                        print(y.shape,gt.shape)

                        # ---------------------------------------------------------------------------------------------
                        #                                 颜色修正的设置，注意不同数据集的通道
                        # ---------------------------------------------------------------------------------------------
                        if idx == 1:
                            indices = quality_assessment(gt, y, data_range=1., ratio=4)
                            # np.save('{}/{}_{}_hr.npy'.format(result_path,  current_step, idx), gt)
                            # np.save('{}/{}_{}_sr.npy'.format(result_path,  current_step, idx), y)
                        else:
                            indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
                        print('不进行颜色修正的结果为 {}'.format(indices))


                        y = color_correction(gt, y, num_channels=102)

                        if idx == 1:
                            indices_cc = quality_assessment(gt, y, data_range=1., ratio=4)
                            np.save('{}/{}_{}_hr.npy'.format(result_path,  current_step, idx), gt)
                            np.save('{}/{}_{}_sr.npy'.format(result_path,  current_step, idx), y)
                        else:
                            indices_cc = sum_dict(indices_cc, quality_assessment(gt, y, data_range=1., ratio=4))
                        print('进行后颜色修正的结果为 {}'.format(indices_cc))

                        # eval_psnr = Metrics.PSNR(SR_img, visuals['HR'][0])
                        # print(eval_psnr)

                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
                        fake_img = Metrics.tensor2img(row_data)
                        # print(sr_img.shape, hr_img)
                        err_img = np.abs(y - gt)

                        # generation
                        Metrics.save_img3(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img3(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img3(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img3(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        
                        # 将误差图像转换为彩色图像并保存
                        plt.imshow(err_img[:,:,[15]], cmap='jet')
                        plt.axis('off')
                        plt.savefig('{}/{}_{}_err.png'.format(result_path, current_step, idx))  # 保存图像

                        # avg_psnr += eval_psnr
                        # print("前{}张图片的PSNR大小为 {} !!".format(idx,avg_psnr))

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )
                    
                    # 平均一下。
                    for index in indices:
                        indices[index] = indices[index] / idx
                    for index in indices_cc:
                        indices_cc[index] = indices_cc[index] / idx
                    print("最终的结果平均指标为 未修正 {}".format(indices))
                    print("最终的结果平均指标为 修正 {}".format(indices_cc))
                    print("模型总共的推理时间为 {}, 平均每一个数据的推理时间为 {}".format(total_time ,total_time / idx))

                    # avg_psnr = avg_psnr / idx

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # indices: {}, indices_cc: {} , total_time: {}, ave_time: {}'.format(indices, indices_cc, total_time ,total_time / idx))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> indices: {} , indices_cc: {},total_time: {}, ave_time: {}'.format(
                        current_epoch, current_step, indices, indices_cc,total_time ,total_time / idx))
                    # tensorboard logger
                    # tb_logger.add_scalar('indices', indices, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    # gen_path = os.path.join(opt['path']['checkpoint'], 'I{}_E{}_GAESR.pth'.format(current_step, current_epoch))
                    # torch.save(model_GAE_SR, opt_path)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_sam = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            # 首先对数据进行降维，读取AE模型，然后再进行扩散训练
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # 读取模型
    # ---------------------------------------------------------------------------------------------
    #                                       GAE的模型读取设置！！！！
    # ---------------------------------------------------------------------------------------------
            model_GAE = torch.load('./weight_64_32_total/GAE_4_Pav.pth',map_location='cuda:0')
            # model_GAE = torch.load('./GAE_8_Pav.pth',map_location='cuda:0')
            # model_E = torch.load('./Enc_4_Pav.pth',map_location='cuda:0')
            model_GAE = model_GAE.to(device)
            # model_E = model_E.to(device)

            # 提前把SR上采样的图片存储一下，不然后面存取的时候总是出现问题。
            row_data = val_data['SR']
            row_data = row_data.to(device)
            clean_data = val_data['HR']
            val_data['SR'] = val_data['SR'].to(device)

            zSRval_list = model_GAE.encode(val_data['SR'])
            new_list = []
            for i in range(len(zSRval_list)):
                val_data['SR'] = zSRval_list[i]
                diffusion.feed_data(val_data)
                diffusion.test(continous=False)
                visuals = diffusion.get_current_visuals()
                visuals['SR'] = visuals['SR'].to(device)
                visuals['SR'] = torch.unsqueeze(visuals['SR'], 0)
                new_list.append(visuals['SR'])
            # new_list = new_list.to(device)
            visuals['SR'] = model_GAE.decode(row_data, new_list)


            # HSI数据的评价指标,需要先转化为numpy进行计算。
            # eval_psnr = Metrics.PSNR(visuals['HR'][0], clean_data[0])
            # eval_ssim = Metrics.SSIM(visuals['HR'][0], clean_data[0])
            # eval_sam = Metrics.SAM(visuals['HR'][0], clean_data[0])
            visuals['SR'][-1][visuals['SR'][-1] < 0] = 0
            visuals['SR'][-1][visuals['SR'][-1] > 1] = 1.

            # visuals['HR'] = visuals['HR'].to(device)
            # visuals['HR'],_ = model_GAE(visuals['HR'])

            print(visuals['SR'][-1].shape, visuals['HR'][0].shape)

            # 输入计算时候需要是HWC的形式。全新的计算，从GELIN里面拿的，之前的从MCNet，感觉不好。
            y = visuals['SR'][-1].cpu().detach().numpy().transpose(1, 2, 0)
            gt = visuals['HR'][0].cpu().detach().numpy().transpose(1, 2, 0)

            # 颜色修正
            # y = color_correction(gt, y, num_channels=31)

            if idx == 1:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
                np.save('{}/{}_{}_hr.npy'.format(result_path,  current_step, idx), gt)
                np.save('{}/{}_{}_sr.npy'.format(result_path,  current_step, idx), y)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            # indices = quality_assessment(gt, y, data_range=1., ratio=4)
            print(indices)

            # eval_psnr = Metrics.PSNR(visuals['SR'][-1], visuals['HR'][0])
            # eval_psnr = Metrics.PSNR(y, gt)
            # eval_ssim = Metrics.SSIM_CHW(visuals['SR'][-1], visuals['HR'][0])
            # eval_sam = Metrics.SAM(visuals['SR'][-1], visuals['HR'][0])
            # eval_ergas = Metrics.calc_ergas(visuals['SR'][-1], visuals['HR'][0])
            # print(eval_psnr)
            # print("第 {} 张图片的 PSNR = {} , SSIM = {} , SAM = {}  ".format(idx,eval_psnr,eval_ssim,eval_sam))

            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
            fake_img = Metrics.tensor2img(row_data)
            err_img = np.abs(y - gt)

            

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img3(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img3(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img3(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img3(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img3(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img3(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
            

            # hr_img = Image.open('{}/{}_{}_hr.png'.format(result_path, current_step, idx)).convert('L')
            # sr_img = Image.open('{}/{}_{}_sr.png'.format(result_path, current_step, idx)).convert('L')
            # err_img = np.abs(np.array(hr_img) - np.array(sr_img)) / 255.0
            # 将误差图像转换为彩色图像并保存
            plt.imshow(err_img[:,:,[15]], cmap='jet')
            plt.axis('off')
            plt.savefig('{}/{}_{}_err.png'.format(result_path, current_step, idx))  # 保存图像

            # generation 这是基于PNG，三通道的评价函数计算方式。
            # eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            # eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            # avg_psnr += eval_psnr
            # avg_ssim += eval_ssim
            # avg_sam += eval_sam

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        # 平均一下。
        for index in indices:
            indices[index] = indices[index] / idx
        print("最终的结果平均指标为 {}".format(indices))

        # avg_psnr = avg_psnr / idx
        # avg_ssim = avg_ssim / idx
        # avg_sam = avg_sam / idx

        # log
        logger.info('# Validation # 各项指标为: {}'.format(indices))
        # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        # logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        # logger.info('# Validation # SAM: {:.4e}'.format(avg_sam))
        logger_val = logging.getLogger('val')  # validation logger
        # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
        #     current_epoch, current_step, avg_psnr, avg_ssim))
        logger_val.info('# Validation # 各项指标为: {}'.format(indices))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
