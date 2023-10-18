import h5py
# import imgvision as iv
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import spectral

# 导入高光谱图像
# HSI = np.load('Hyperspectral_Image2.npy')
for x in range(0, 1040  - 512, 512):
    print(x)
    for y in range(0, 1392  - 512, 512):
        print(x,y)
HSI = io.loadmat("./data/img1.mat")['ref']

img = HSI
img = np.asarray(img, dtype='float32')
img = (img - np.min(img)) / (np.max(img) - np.min(img))

print(img)
# print(HSI.shape)
# print(HSI)


img = h5py.File('./data/4cam_0411-1640-1.mat', 'r')['rad']
img = np.array(img)
img = img.transpose(1, 2, 0)
print(img.shape)

view1 = spectral.imshow(data=HSI, bands=[30, 27, 11], title="img")
# spectral.view_cube(HSI, bands=[30, 27, 11])
view2 = spectral.imshow(data=img, bands=[30, 27, 11], title="img")

# view3 = spectral.imshow(data=HSI, bands=[30, 27, 11])
# view3.set_display_mode("overlay")
# view3.class_alpha = 0.3  # 设置类别透明度为0.3

plt.pause(600)

# (225,246,87)  该光谱图像是 空间维度225×246，光谱维度87（370nm~800nm 间隔5nm）

# 光谱图像的RGB显示
# 创建转换器   illuminant='D50'表示D50下显示。支持光源包括A/B/C/D50~75，以及自定义光源
# band 为波段参数，如370~800nm.间隔5nm 即 band = np.arange(370,805,5)； 若370~702nm.间隔4nm 即 band = np.arange(370,706,4)
# convertor = iv.spectra(illuminant='D50', band=np.arange(370, 805, 5))

# Havard数据集，400nm-700nm，间隔为10nm
# convertor = iv.spectra(illuminant='D50', band=np.arange(400, 700, 10))
#
# # RGB图像转换 若展示sRGB则space='srgb'；若展示XYZ图像，则space='xyz'；若展示NikonD700相机的显示结果，则space='nkd700'. 注意nkd700下的RGB未经过Gamma校正。
# Image = convertor.space(HSI, space='srgb')
# # 图像显示
# plt.imshow(Image)
# plt.show()
