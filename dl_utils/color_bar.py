import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

# 定义自定义颜色列表
#colors = ["#050086", "#16ECD0", "#DBEB11", "#E82A09", "#850004"]  #热图
colors = ["#0D0887", "#BD3786", "#F0F921"]

# 创建自定义颜色映射
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# 创建一个示例数据集
data = np.random.rand(10, 10)

# 使用自定义颜色映射创建热图
fig, ax = plt.subplots()
cax = ax.imshow(data, cmap=custom_cmap)
#cax = ax.imshow(data, cmap='coolwarm')

# 添加颜色条
cbar = fig.colorbar(cax)








# 设置标题和标签
ax.set_title('Custom Color Heatmap Example')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# 保存图像到指定路径
save_path = '/home/caoxinyu/Arbitrary-scale/liif-main/demo/atten_visual/heatmap.png'  # 替换为你的路径
plt.savefig(save_path, bbox_inches='tight')

# 显示图像
plt.show()