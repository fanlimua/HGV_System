import pandas as pd
import numpy as np

# —————— 1. 读取数据 ——————
# 将 'your_file.xlsx' 换成你的文件名，sheet_name 根据实际情况修改
df = pd.read_excel('your_file.xlsx', sheet_name='Sheet1')

# 提取坐标
x = df['Position_X'].to_numpy()
y = df['Position_Y'].to_numpy()

# —————— 2. 圆拟合（Kasa 方法） ——————
# 构造线性方程 A·[D, E, F]^T = b
A = np.column_stack([x, y, np.ones_like(x)])
b_vec = -(x**2 + y**2)

# 求解最小二乘：得到 D, E, F
D, E, F = np.linalg.lstsq(A, b_vec, rcond=None)[0]

# 由 D, E, F 计算圆心 (a, b0) 和半径 r
a = -D / 2
b0 = -E / 2
r = np.sqrt(a**2 + b0**2 - F)

# —————— 3. 误差计算 ——————
# 每个点到拟合圆心的距离
distances = np.sqrt((x - a)**2 + (y - b0)**2)
# 与拟合半径的绝对误差
errors = np.abs(distances - r)

# 最大误差
max_error = errors.max()
# 平均误差
mean_error = errors.mean()
# 误差标准差
std_error = errors.std()
# 圆度指标（最大距离与最小距离之差）
roundness = distances.max() - distances.min()

# —————— 4. 输出结果 ——————
print(f'Fitted Circle Center:    ({a:.4f}, {b0:.4f})')
print(f'Fitted Circle Radius:    {r:.4f}')
print(f'Max Error:    {max_error:.4f}')
print(f'Average Error:    {mean_error:.4f}')
print(f'Standard Deviation:    {std_error:.4f}')
print(f'Roundness:    {roundness:.4f}')
