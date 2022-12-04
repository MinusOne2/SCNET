##################测试的主程序
import cv2
import os
import argparse
# -*- coding: utf-8 -*-
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.SCNET import SCNET
from utils import *
import matplotlib.pyplot as plt
from main_dataset import  Dataset
from torch.utils.data import DataLoader
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    #"font.size": 80,
#     "mathtext.fontset":'stix',
}
rcParams.update(config)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #########GPU计算
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#######################################################参数设置
parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17)
parser.add_argument("--logdir", type=str, default="logs")
parser.add_argument("--batchSize", type=int, default=1)
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    # t1 = time.clock()

    net = SCNET(channels=1, num_of_layers=opt.num_of_layers) #########载入网络
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net113.pth'))) #加载参数
    model.eval()  #测试模式
    # load data info
    print('Loading data info ...\n')
    dataset_test = Dataset(train=False)  #测试数据
    loader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=opt.batchSize, shuffle=False)

    # files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    # files_source.sort()

        # process data
    start = time.time()
    psnr_avg=0
    for i, data in enumerate(loader_test):  #####匹配数据
        Img, label ,name_Img, name_label= data    #Imgsize(1,1,100,100) label(1,1,100,100)
        ISource = torch.Tensor(Img)
        ISource = Variable(ISource.cuda())
        with torch.no_grad():  # this can save much memory
            # Out1 = torch.clamp(ISource - model(ISource), 0., 1.)
            Out = ISource - model(ISource)
    # t2 = time.clock()
    # print(t2 - t1)
        zero = torch.zeros_like(ISource)
        ones = torch.ones_like(ISource)
        Out  = torch.where(Out < 0.1, zero, Out)
        Out  = torch.where(Out > 1,ones,Out)

        psnr_test = batch_PSNR(Out, label, 1.)
        psnr_avg+=psnr_test
        print(psnr_test)

        ISource = torch.squeeze(ISource, 0).cpu()
        label = torch.squeeze(label,0)
        Out= torch.squeeze(Out, 0).cpu()
        # zero = torch.zeros_like(ISource)
        # ones = torch.ones_like(ISource)
        # Out  = torch.where(Out < 0.1, zero, Out)
        # Out  = torch.where(Out > 1,ones,Out)
        if i>=0:  ###############################测试结果展示
            plt.subplot(1,3,1)
            plt.imshow(ISource.permute(1,2,0),cmap="gray")
            plt.title('ORI',fontsize=30)
            plt.subplot(1,3,2)
            plt.imshow(label.permute(1,2,0),cmap="gray")
            plt.title('CLEAN-60%',fontsize=30)
            plt.subplot(1,3,3)
            plt.imshow(Out.permute(1,2,0),cmap="gray")
            plt.title('SCNET',fontsize=30)
            plt.show()

    end=time.time()
    print('平均：'+str(psnr_avg/i))
    print(end-start)

########################################################################
if __name__ == "__main__":
    main()
