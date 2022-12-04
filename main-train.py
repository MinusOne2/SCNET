###############训练用的主程序
#######载入库
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.SCNET import SCNET
from models.FFDNet import FFDNet
# from dataset import prepare_data, Dataset
from main_dataset import Dataset
from utils import *
import matplotlib.pyplot as plt
  
###########使用GPU训练
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
###############参数设置
parser = argparse.ArgumentParser(description="FFDNet")
parser.add_argument("--batchSize", type=int, default=1, help="batchsize")
parser.add_argument("--num_of_layers", type=int, default=2, help="网络层数")
parser.add_argument("--epochs", type=int, default=30, help="训练迭代次数")
parser.add_argument("--milestone", type=int, default=300, help="达到miletone时，学习率下降")
parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
parser.add_argument("--outf", type=str, default="logs", help='log files路径')
#parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
#parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
#parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
opt = parser.parse_args()
################主程序

path = './result/trans/'

def show_fp(fp, fp_name, epoch):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fp_img = fp[0,...].max(dim=0)[0].detach().cpu()

    sns.heatmap(fp_img, cmap='rainbow', cbar=True)
    plt.title(str(epoch))
    plt.savefig(path + fp_name + str(epoch)+'.png')
    plt.close()
    # plt.title(fp_name)
    # plt.show()


def show_img(fp, fp_name, epoch):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fp_img = fp[0,...].max(dim=0)[0].detach().cpu()

    plt.title(str(epoch))
    plt.imshow(fp_img, cmap='gray')
    plt.savefig(path + fp_name + str(epoch)+'.png')
    plt.close()
    # plt.title(fp_name)
    # plt.show()

def main():
    # Load dataset
    print('Loading dataset ...\n')#打印数据库
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = SCNET(channels=1, num_of_layers=opt.num_of_layers)#######载入网络
    # net = FFDNet()
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False) #########损失函数MSE
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net510.pth'))) ####载入训练好的网络参数
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    s='logs/cnn'+str(opt.epochs)

    writer = SummaryWriter(s)#opt.outf
    step = 0
    # noiseL_B=[0,55] # ingnored when opt.mode=='S'
    current_lr = opt.lr
    for epoch in range(opt.epochs):  ########开始迭代
        avg_loss=0
        cnt=0

        if (epoch+1) % opt.milestone==0: #####降低学习率条件
            current_lr/= 2.

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):

            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train, img_label ,name_train,name_label= data  #####匹配数据
#########显示旁瓣特征图像###########
            # plt.subplot(1, 3, 1)
            # plt.imshow(img_train.squeeze(0).permute(1,2,0), cmap='gray')
            # print(name_train)
            # plt.subplot(1, 3, 2)
            # plt.imshow(img_label.squeeze(0).permute(1,2,0), cmap='gray')
            # print(name_label)
            # zero = torch.zeros_like(img_train)
            # noise=img_train-img_label
            # noise = torch.where(noise < 0.01, zero, noise)
            # plt.subplot(1, 3, 3)
            # plt.imshow(noise.squeeze(0).permute(1, 2, 0), cmap='gray')
            # print(noise)
            # plt.show()
#########调试用###################

            img_train, img_label = Variable(img_train.cuda()), Variable(img_label.cuda())
            noise=img_train-img_label  #######噪声（旁瓣）图
            noise = Variable(noise.cuda())  #放入GPU

            out_train, x1, x2 = model(img_train)
            loss_l2 = criterion(out_train, noise) / (img_train.size()[0]*2)

            out_train_clean = torch.clamp(img_train-out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train_clean, img_label, 1.)       #psnr计算
            # loss_psnr = 10-torch.log2(psnr_train)

            loss = loss_l2
            avg_loss+=loss.item()
            cnt+=1
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # print("[epoch %d][%d/%d] loss: %.4f loss_l2: %.4f loss_psnr: %.4f PSNR_train: %.4f" %
            #     (epoch+1, i+1, len(loader_train), loss.item(), loss_l2.item(), loss_psnr, psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
        if epoch % 1 == 0:      #######显示训练情况工具
        # Log the scalar values
            writer.add_scalar('loss', avg_loss/cnt, epoch)
            writer.add_scalar('PSNR on training data', psnr_train, epoch)

            show_fp(x1, 'xcnn_fp_', epoch)
            show_fp(x2, 'xtrans_fp_', epoch)
            show_img(out_train, 'noise_', epoch)
            show_img(out_train_clean, 'clean_sar_', epoch)

        step += 1

        ## the end of each epoch
        # model.eval()
        # validate
        psnr_val = 0
        # plt.figure(figsize=(20, 20))
        # for k in range(len(dataset_val)):
        #
        #
        #     plt.subplot(4, 3, k + 1)
        #
        #     plt.imshow(dataset_val[k].permute(1, 2, 0), cmap='gray')
        #     plt.axis('off')
        # plt.show()
        # for k in range(len(dataset_val)):
        #     img_val = torch.unsqueeze(dataset_val[k], 0)  #加维度
        #     #img_val=dataset_val[k]
        #     noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
        #     imgn_val = img_val + noise
        #
        #     img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
        #     out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
        #     psnr_val += batch_PSNR(out_val, img_val, 1.)
        # psnr_val /= len(dataset_val)
        # print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # # log the images
        # out_train = torch.clamp(img_train-model(img_train), 0., 1.)
        # Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        # Imgn = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        # Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        # writer.add_image('noisy image', Imgn, epoch)
        # writer.add_image('reconstructed image', Irecon, epoch)
        # # save model

        # torch.save(model.state_dict(), os.path.join(opt.outf, 'net511.pth'))

if __name__ == "__main__":
    main()