import numpy as np
import matplotlib.pyplot as plt

# 高斯函数的参数
mean = 0
std_dev = 6
x = np.linspace(-10, 10, 1000)  # 生成x值
y = np.exp(-(x - mean)**2 / (2 * std_dev**2))  # 高斯函数

# 创建图形和坐标轴
fig, ax = plt.subplots()

# 区域划分：定义不同的x坐标值，来划分不同的覆盖区域
x1 = x[(x >= -10) & (x < -1.5)]
y1 = y[(x >= -10) & (x < -1.5)]
x2 = x[(x >= -1.5) & (x < 1.5)]
y2 = y[(x >= -1.5) & (x < 1.5)]
x4 = x[(x >= 1.5) & (x <= 10)]
y4 = y[(x >= 1.5) & (x <= 10)]

# 填充不同区域的颜色 (使用6DADD1颜色)
ax.fill_between(x1, y1, color='#36A9E1', alpha=0.8)  # 第一部分：6DADD1
ax.fill_between(x2, y2, color='#EB5B25', alpha=0.8)  # 第二部分：6DADD1
ax.fill_between(x4, y4, color='#36A9E1', alpha=0.8)  # 第四部分：6DADD1

# 绘制高斯曲线，控制颜色和粗细
ax.plot(x, y, color='#5C6572', linewidth=15)

# 去除坐标系的顶部和右侧
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 只保留x轴
ax.spines['left'].set_color('none')
ax.yaxis.set_ticks_position('none')

# 去除所有坐标轴上的刻度和数值
ax.set_xticks([])  # 去除x轴的刻度
ax.set_yticks([])  # 去除y轴的刻度

# 设置x轴标签和标题
ax.set_xlabel('X轴')
ax.set_title('高斯曲线')

# 不显示图例
ax.legend().set_visible(False)

# 保存图像，设置透明背景
save_path = '/home/caoxinyu/Arbitrary-scale/liif-main/results/gaussian_curve_3.png'
plt.savefig(save_path, bbox_inches='tight', transparent=True)

# 显示图像
plt.show()

# 返回保存路径
print(f"图像已保存到: {save_path}")




