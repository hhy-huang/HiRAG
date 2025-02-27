import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statistics

# 设置Seaborn样式
sns.set(style="whitegrid")

# 示例数据
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y_values = [[90.31, 60.29, 69.97], 
            [99.51, 98.72, 87.53], 
            [99.74, 98.48, 99.17], 
            [99.74, 98.18, 99.17], 
            [99.72, 97.78, 99.05], 
            [99.69, 97.78, 98.72],
            [99.67, 97.78, 98.72], 
            [99.67, 97.22, 98.72], 
            [99.64, 97.22, 98.48], 
            [99.64, 97.22, 98.18]]
y = [statistics.mean(x) for x in y_values]

# 创建图形
plt.figure(figsize=(5, 3))

# 绘制折线图
sns.lineplot(x=x, y=y, marker='o', label='Sparsity')

# 准备填充区域的数据
y_min = [min(val) for val in y_values]
y_max = [max(val) for val in y_values]
# 计算每一层的三个数据的变化率
change_rates = []
for i in range(0, len(y_values)-1):
    prev_values = y_values[i]
    curr_values = y_values[i+1]
    rates = [(curr - prev) / prev * 100 for prev, curr in zip(prev_values, curr_values)]
    change_rates.append(rates)

# 计算变化率的平均值、最小值和最大值
change_rates_mean = [statistics.mean(rates) for rates in change_rates]
change_rates_min = [min(rates) for rates in change_rates]
change_rates_max = [max(rates) for rates in change_rates]
print(change_rates_mean[1])
# 绘制变化率的折线图
sns.lineplot(x=x[1:], y=change_rates_mean, marker='o', color='#BD86A4', label='Change Rate')

# 填充变化率的最大值和最小值区域
plt.fill_between(x[1:], change_rates_min, change_rates_max, alpha=0.3, color='gray')

# 填充区域
plt.fill_between(x, y_min, y_max, alpha=0.3, color='gray')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.axhline(y=98, color='r', linestyle='--', linewidth=1, label='y=98')
# yticks = plt.gca().get_yticks()
# 在 y=99 位置添加文本注释，与 y 轴刻度对齐
# plt.text(0, 98, '98', color='r', verticalalignment='top', horizontalalignment='center')# 设置标题和标签
# plt.title('Cluster Sparsity vs. Layer')
plt.xlabel('Layer', fontsize=19)
plt.ylabel('Cluster Sparsity (%)', fontsize=19)
plt.legend(title='', fontsize=15)
plt.subplots_adjust(left=0.18, right=0.97, top=0.98, bottom=0.23)

# 显示图形
plt.savefig('./test.png')