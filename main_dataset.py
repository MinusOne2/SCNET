###################处理数据的子程序，主要将图像数据处理为原图像和对应的CLEAN标签图像
import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
import matplotlib.pyplot as plt

def normalize(data):
    return data/255.

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        #####################训练和测试状态下，数据地址
        if self.train:
            self.data_path = r'C:\Users\oywt\Project\SCNET\data\CLEANDATA\train\ORI/'###不要有中文路径
        else:
            self.data_path = r'C:\Users\oywt\Project\SCNET\data\CLEANDATA\test\ORI/'#noise_10dm/'

        data = []
        label = []
        labelname=[]
        dataname=[]
        filesource=os.listdir(self.data_path)  ##打开数据夹
        filesource.sort(key=lambda x:int(x.split('.')[0])) #按数字大小排序

        # filesource = filesource[-400:] # minidataset for expr
        for filename in filesource:  #遍历数据夹
            id = int(filename.split('.')[0])
            img = cv2.imread(os.path.join(self.data_path, filename))  # w h c
            img = normalize(np.float32(img[:, :, 0]))   # w h
            img = np.expand_dims(img, 0)    # 1 w h
            if id % 2 == 0: ####偶数为标签图
                label.append(img)
                labelname.append(filename)
            else:   ####奇数为ISAR原图
                data.append(img)
                dataname.append(filename)
        self.data = np.array(data)
        self.label = np.array(label)
        self.datalname=dataname
        self.labelname=labelname
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index): ####将原图和标签做一一对应
        data = self.data[index]
        label = self.label[index]
        dataname=self.datalname[index]
        labelname=self.labelname[index]
        return torch.Tensor(data), torch.Tensor(label),dataname,labelname
