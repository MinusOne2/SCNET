# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['Times New Roman']


names = ['0','25','50','75','100','125','150','175','200','225','250','275','300','325','350','375', '400','425', '450','475', '500']
x = range(len(names))
y = [46.14,12.128,8.119,4.205, 2.479,1.568 ,1.179,1.1293, 0.8693,0.715,0.7529,0.644,0.4493,0.5209,0.4848,0.4512,0.4389,0.4276,0.404,0.4076,0.3797]
# y1=[0.86,0.85,0.853,0.849,0.83]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
plt.ylim(0, 10)  # 限定纵轴的范围
plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'Loss')
# plt.plot(x, y1, marker='*', ms=10,label=u'y=x^3曲线图')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Eoch") #X轴标签
plt.ylabel("Loss") #Y轴标签
plt.title("Loss-Epoch") #标题
plt.show()

###############################
names = ['0','25','50','75','100','125','150','175','200','225','250','275','300','325','350','375', '400','425', '450','475', '500']
x = range(len(names))
y = [24.7,28.6,27.8,30.49,34.16, 35.23,36.1 ,35.31,37.11, 35.72,38.84,39.27,41.67,40.99,40.99,40.54,41.44,41.87,41.74,42.09,42.26]
# y1=[0.86,0.85,0.853,0.849,0.83]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
plt.ylim(20, 50)  # 限定纵轴的范围
plt.plot(x, y, marker='o', mec='r', mfc='w',label=u'PSNR')
# plt.plot(x, y1, marker='*', ms=10,label=u'y=x^3曲线图')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Eoch") #X轴标签
plt.ylabel("PSNR") #Y轴标签
plt.title("PSNR-Epoch") #标题
plt.show()