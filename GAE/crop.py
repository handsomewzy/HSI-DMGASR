# import scipy.io
# import numpy as np
# import os

# # 读取MATLAB数据
# data = scipy.io.loadmat('../Chikusei_data_block16')['block']

# # 计算数据的尺寸
# height, width = data.shape[:2]

# # 计算切割后的块数
# block_size = 128
# num_rows = height // block_size
# num_cols = width // block_size

# # image_dir = '/mnt/workspace/workgroup/zhaoyang.wzy/Chikusei_mat/test/crop'
# # image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
# # print(image_files)
# # for x in image_files:
# #     print(x)
# #     data = np.load(x)
# #     print(data.shape)

# # 切割数据
# for i in range(num_rows):
#     for j in range(num_cols):
#         # 计算当前块的起始和结束位置
#         row_start = i * block_size
#         row_end = row_start + block_size
#         col_start = j * block_size
#         col_end = col_start + block_size
#         # 提取当前块的数据
#         block = data[row_start:row_end, col_start:col_end]
#         # 将块保存到本地
#         # block_filename = f'./128test/block14_{i}_{j}.npy'
#         np.save('/mnt/workspace/workgroup/zhaoyang.wzy/Chikusei_mat/test/128test/block16_{}_{}'.format(i,j), block)




import scipy.io
import numpy as np
import os

# 读取MATLAB数据
image_dir = '/mnt/workspace/workgroup/zhaoyang.wzy/Harvard_mat/test'
image_files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
print(image_files)
for i in range(len(image_files)):
    data = scipy.io.loadmat(image_files[i])['ref']
    print(data.shape,i)
    # 计算数据的尺寸
    height, width = data.shape[:2]

    # 提取图像的四个角和中心
    block_size = 512
    top_left = data[:block_size, :block_size]
    top_right = data[:block_size, width - block_size:]
    bottom_left = data[height - block_size:, :block_size]
    bottom_right = data[height - block_size:, width - block_size:]
    center_row_start = (height // 2) - (block_size // 2)
    center_row_end = center_row_start + block_size
    center_col_start = (width // 2) - (block_size // 2)
    center_col_end = center_col_start + block_size
    center = data[center_row_start:center_row_end, center_col_start:center_col_end]

    # 将块保存到本地
    np.save('/mnt/workspace/workgroup/zhaoyang.wzy/Harvard_mat/128test/{}_top_left.npy'.format(i), top_left)
    np.save('/mnt/workspace/workgroup/zhaoyang.wzy/Harvard_mat/128test/{}_top_right.npy'.format(i), top_right)
    np.save('/mnt/workspace/workgroup/zhaoyang.wzy/Harvard_mat/128test/{}_bottom_left.npy'.format(i), bottom_left)
    np.save('/mnt/workspace/workgroup/zhaoyang.wzy/Harvard_mat/128test/{}_bottom_right.npy'.format(i), bottom_right)
    np.save('/mnt/workspace/workgroup/zhaoyang.wzy/Harvard_mat/128test/{}_center.npy'.format(i), center)
