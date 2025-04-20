import json
import matplotlib
import numpy as np

matplotlib.use('TkAgg')  # 设置后端为 TkAgg
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

with open(r'D:\PyCharm Projects\data_analysis\Hotspot\Hotspot\vm592.json', 'r') as f:
    data = json.load(f)

vm_util_all = data['vm_util']  # 假设数据是一个嵌套列表，每个元素代表一个时间点的CPU利用率
vm_util = vm_util_all[0]  # 取出第一个虚拟机的CPU利用率数据
host_vcpus = data['host_vcpus']  # 假设每个虚拟机对应的vCPUs数目
# 假设时间戳是从 'time_start' 开始，每五分钟一采样
time_start = pd.to_datetime(data['time_start'])
time_index = pd.date_range(start=time_start, periods=len(vm_util), freq='5min')  # 每5分钟一个时间点
# threshold_factor = 2
# time_series = pd.Series(vm_util, index=time_index)
# #三个点取一次
# # time_series = time_series[::3]
#
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

time_series_cleaned = pd.Series(vm_util, index=time_index)

# 3. 去抖动：使用Savitzky-Golay滤波平滑数据
smoothed_series = savgol_filter(time_series_cleaned, window_length=41, polyorder=2)

# 绘图部分
plt.figure(figsize=(6, 6))
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

num_components = 20
sorted_main_frequencies = sorted_frequencies[:num_components]
sorted_main_T = sorted_T[:num_components]
sorted_main_amplitudes = sorted_amplitudes[:num_components]
sorted_main_phases = sorted_phases[:num_components]

# print(sorted_main_T)
# print(sorted_main_amplitudes)
# print(sorted_main_phases)

reconstructed_signal = np.zeros_like(smoothed_series, dtype=complex)
for i in range(num_components):
    reconstructed_signal += sorted_main_amplitudes[i] * np.exp(1j * sorted_main_phases[i]) * np.exp(
        2j * np.pi * sorted_main_frequencies[i] * np.arange(N))

signal_first = sorted_main_amplitudes[1] * np.exp(1j * sorted_main_phases[1]) * np.exp(
        2j * np.pi * sorted_main_frequencies[1] * np.arange(N))
signal_second = sorted_main_amplitudes[2] * np.exp(1j * sorted_main_phases[2]) * np.exp(
        2j * np.pi * sorted_main_frequencies[2] * np.arange(N))
signal_third = sorted_main_amplitudes[3] * np.exp(1j * sorted_main_phases[3]) * np.exp(
        2j * np.pi * sorted_main_frequencies[3] * np.arange(N))
signal_fourth = sorted_main_amplitudes[4] * np.exp(1j * sorted_main_phases[4]) * np.exp(
        2j * np.pi * sorted_main_frequencies[4] * np.arange(N))
signal_twentieth = sorted_main_amplitudes[10] * np.exp(1j * sorted_main_phases[10]) * np.exp(
        2j * np.pi * sorted_main_frequencies[10] * np.arange(N))

fft_real = np.real(reconstructed_signal)
fft_first = np.real(signal_first)
fft_second = np.real(signal_second)
fft_third = np.real(signal_third)
fft_fourth = np.real(signal_fourth)
fft_twentieth = np.real(signal_twentieth)
ymin = np.min(fft_first)
ymax = np.max(fft_first)

plt.figure(figsize=(8, 6))
plt.subplot(3, 1, 1)
plt.plot(fft_first, color='black')
plt.ylim(ymin, ymax)
plt.xlabel('Time', fontsize=12)  # 为横轴添加标签
plt.ylabel('CPU Util', fontsize=12)  # 为纵轴添加标签
plt.xticks([])  # 去掉横轴刻度
plt.yticks([])  # 去掉纵轴刻度

plt.subplot(3, 1, 2)
plt.plot(fft_second, color='black')
plt.ylim(ymin, ymax)
plt.xlabel('Time', fontsize=12)  # 为横轴添加标签
plt.ylabel('CPU Util', fontsize=12)  # 为纵轴添加标签
plt.xticks([])  # 去掉横轴刻度
plt.yticks([])  # 去掉纵轴刻度

plt.subplot(3, 1, 3)
plt.plot(fft_twentieth, color='black')
plt.ylim(ymin, ymax)
plt.xlabel('Time', fontsize=12)  # 为横轴添加标签
plt.ylabel('CPU Util', fontsize=12)  # 为纵轴添加标签
plt.xticks([])  # 去掉横轴刻度
plt.yticks([])  # 去掉纵轴刻度
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()

plt.figure(figsize=(6, 3))
plt.subplot(1, 1, 1)
plt.plot(np.real(reconstructed_signal), color='black')
plt.xlabel('Time', fontsize=12)  # 为横轴添加标签
plt.ylabel('CPU Util', fontsize=12)  # 为纵轴添加标签
plt.xticks([])  # 去掉横轴刻度
plt.yticks([])  # 去掉纵轴刻度
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()
