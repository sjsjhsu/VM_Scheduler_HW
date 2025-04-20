import json
import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # 设置后端为 TkAgg
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 处理虚拟机原始数据，包括去除热点，滤波，傅里叶变换

with open(r'D:\PyCharm Projects\data_analysis\Hotspot\Hotspot\vm503.json', 'r') as f:
    data = json.load(f)

# 取出每五分钟的CPU利用率数据
vm_util_all = data['vm_util']  # 假设数据是一个嵌套列表，每个元素代表一个时间点的CPU利用率
vm_util = vm_util_all[0]  # 取出第一个虚拟机的CPU利用率数据
host_vcpus = data['host_vcpus']  # 假设每个虚拟机对应的vCPUs数目
# 假设时间戳是从 'time_start' 开始，每五分钟一采样
time_start = pd.to_datetime(data['time_start'])
time_index = pd.date_range(start=time_start, periods=len(vm_util), freq='5min')  # 每5分钟一个时间点
print(len(vm_util))
print(time_index)
# threshold_factor = 2
# time_series = pd.Series(vm_util, index=time_index)
# time_series = time_series[::3]

# avg_value = vm_util[0]
# max_value = vm_util[0]
# for idx, utilization in enumerate(vm_util):
#     # 如果当前值超过2倍的最大值，则认为是热点
#     if utilization > avg_value * threshold_factor:
#         vm_util[idx] = max_value  # 将当前点的值设置为当前最大值
#     else:
#         # 更新最大值
#         max_value = max(max_value, utilization)
#         avg_value = (avg_value * idx + utilization) / (idx + 1)  # 更新平均值
# print(max_value)

# 将数据转换为pandas的Series
time_series_cleaned = pd.Series(vm_util, index=time_index)

# 3. 去抖动：使用Savitzky-Golay滤波平滑数据，窗口大小为11
smoothed_series = savgol_filter(time_series_cleaned, window_length=21, polyorder=2)

# 调整图形大小和布局
plt.figure(figsize=(12, 8))

# 原始数据图
plt.subplot(2, 1, 1)
plt.plot(time_series_cleaned, color='black')
plt.xlabel('Time', fontsize=12)  # 为横轴添加标签
plt.ylabel('CPU Util', fontsize=12)  # 为纵轴添加标签
plt.xticks([])  # 去掉横轴刻度
plt.yticks([])  # 去掉纵轴刻度

# 重建数据图
plt.subplot(2, 1, 2)
plt.plot(np.real(smoothed_series), color='black')
plt.xlabel('Time', fontsize=12)  # 为横轴添加标签
plt.ylabel('CPU Util', fontsize=12)  # 为纵轴添加标签
plt.xticks([])  # 去掉横轴刻度
plt.yticks([])  # 去掉纵轴刻度

# 调整子图间距
plt.subplots_adjust(hspace=0.4)
# 显示图形
plt.show()

N = len(smoothed_series)  # 数据长度
fft_values = np.fft.fft(smoothed_series)  # 傅里叶变换

# 2. 计算频率分量
frequencies = np.fft.fftfreq(N)
T = 1 / frequencies
# 3. 获取幅度和相位
amplitude = np.abs(fft_values)  # 幅度
phase = np.angle(fft_values)  # 相位

positive_freq_indices = np.where(T > 0)  # 只选取正频率
sorted_indices = np.argsort(amplitude[positive_freq_indices])[::-1]
sorted_amplitudes = amplitude[sorted_indices]
sorted_phases = phase[sorted_indices]
sorted_T = T[sorted_indices]
sorted_frequencies = frequencies[sorted_indices]

# 4. 选择主频率成分，通常选择低频部分
# 你可以通过查看 amplitude 来选择主频成分
# 例如，这里我们假设只保留前 10 个频率成分
num_components = 10
sorted_main_frequencies = sorted_frequencies[:num_components]
sorted_main_T = sorted_T[:num_components]
sorted_main_amplitudes = sorted_amplitudes[:num_components]
sorted_main_phases = sorted_phases[:num_components]

# 5. 重构信号（只使用前10个频率分量）
reconstructed_signal = np.zeros_like(smoothed_series, dtype=complex)
for i in range(num_components):
    reconstructed_signal += sorted_main_amplitudes[i] * np.exp(1j * sorted_main_phases[i]) * np.exp(
        2j * np.pi * sorted_main_frequencies[i] * np.arange(N))

# 6. 可视化结果
# 调整图形大小和布局
plt.figure(figsize=(6, 6))

# 原始数据图
plt.subplot(2, 1, 1)
plt.plot(smoothed_series, color='black')
plt.xlabel('Time', fontsize=12)  # 为横轴添加标签
plt.ylabel('CPU Util', fontsize=12)  # 为纵轴添加标签
plt.xticks([])  # 去掉横轴刻度
plt.yticks([])  # 去掉纵轴刻度

# 重建数据图
plt.subplot(2, 1, 2)
plt.plot(np.real(reconstructed_signal), color='black')
plt.xlabel('Time', fontsize=12)  # 为横轴添加标签
plt.ylabel('CPU Util', fontsize=12)  # 为纵轴添加标签
plt.xticks([])  # 去掉横轴刻度
plt.yticks([])  # 去掉纵轴刻度

# 调整子图间距
plt.subplots_adjust(hspace=0.4)

# 显示图形
plt.show()
