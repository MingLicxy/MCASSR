import numpy as np
from PIL import Image

# 假设你的 .bin 文件存储的是 256x256 的灰度图像
width, height = 256, 256
filename_bin = '/home/caoxinyu/All-in-One/MRI/train/HQ/IXI002-Guys-0828-T2_0.bin'
filename_png = '/home/caoxinyu/All-in-One/MRI/visual/IXI002-Guys-0828-T2_0.png'

# 从 .bin 文件读取数据
data = np.fromfile(filename_bin, dtype=np.uint8)  # 假设每个像素为 8 位无符号整数
data = data.reshape((height, width))  # 根据图像的宽高调整形状

# 将数据转换为图像
image = Image.fromarray(data)

# 保存为 .png 文件
image.save(filename_png)



