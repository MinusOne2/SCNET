# ####################################  文件重命名
#
# import os
# import shutil #复制文件的库
#
# time=[0,1,2]
# path = r"F:\Code\DnCNN-PyTorch-master\data\CLEANDATA\train\ORI"
# for i in time:
#     if i==0:
#         newpath = r"F:\Code\DnCNN-PyTorch-master\data\CLEANDATA\train\noise_7dm"
#     elif i==1:
#         newpath = r"F:\Code\DnCNN-PyTorch-master\data\CLEANDATA\train\noise_10dm"
#     else:
#         newpath = r"F:\Code\DnCNN-PyTorch-master\data\CLEANDATA\train\noise_14dm"
#
#     filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
#     print(filelist) #文件夹中所有文件名
#     for file in filelist:
#         filename=file.split('.')
#         old=int(filename[0])
#         if old%2==0:
#             new=newpath+os.sep+str(old)+'.png'   #292,584,876,1168
#             old=path+os.sep+str(old)+'.png'
#             os.remove(new)
#             shutil.copy(old,new)
#         # shutil.copy(old, new)  # 复制后命名
#
# ########################################################################

# ####################################  重命名到新文件夹
# import os
# import shutil #复制文件的库
#
# path=r"F:\复旦\雷达成像\周江浩毕业对接材料\3.程序清单\第四章程序\classify\dataset\train\A380"
# newpath=r"F:\复旦\雷达成像\数据库\pic\clean\train"
# filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
# filelist=sorted(filelist,key=lambda x:int(x[:-4]))
# print(filelist) #文件夹中所有文件名
# i=0
# for file in filelist:
#     filename=file.split('.')
#     old=int(filename[0])
# # if (old%2==0):
#     # i+=1
#     # new=newpath+os.sep+str(i)+'.png'
#     new=old//2+1
#     old=path+os.sep+str(old)+'.png'
#     new=path+os.sep+str(new)+'.png'
#     os.rename(old,new)  # 复制后命名
    # new = newpath + os.sep + str(old+1168) + '.png'
    # old=path+os.sep+str(old)+'.png'
    # shutil.copy(old,new)
#
# ########################################################################

# ###########################################################################
# 随机化并移动文件
import os
import shutil #复制文件的库
import random
import string

path=r"F:\复旦\雷达成像\周江浩毕业对接材料\3.程序清单\第四章程序\classify\dataset\train\mission"
newpath=r"F:\复旦\雷达成像\周江浩毕业对接材料\3.程序清单\第四章程序\classify\dataset\test\mission"
filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
filelist=sorted(filelist,key=lambda x:int(x[:-4]))
print(filelist) #文件夹中所有文件名

i=0
list=[]
while i<50:
    nums=random.randint(1,146)
    if nums not in list:
        list.append(nums)
        testfile=path+os.sep+str(nums)+'.png'
        shutil.move(testfile,newpath)
    else:
        i-=1
    i+=1

# #############################################################
# import os
# import shutil #复制文件的库
# import random
# import string

# testpath=r"F:\Code\DnCNN-PyTorch-master\data\CLEANDATA\test\noise_14dm"
# trainpath=r"F:\Code\DnCNN-PyTorch-master\data\CLEANDATA\train\noise_14dm"
# filelist_test = os.listdir(testpath)  # 该文件夹下所有的文件（包括文件夹）
# filelist_train = os.listdir(trainpath)
# filelist_test=sorted(filelist_test,key=lambda x:int(x[:-4]))
# filelist_train=sorted(filelist_train,key=lambda x:int(x[:-4]))
# #print(filelist) #文件夹中所有文件名
# test=[]
# for file in filelist_test:
#     filename=file.split('.')
#     testname=int(filename[0])
#     test.append(testname)
# print(test)
# for file in filelist_train:
#     filename_t=file.split('.')
#     trainname=int(filename_t[0])
#     if trainname in test:
#         fileremove=trainpath+os.sep+file
#         os.remove(fileremove)