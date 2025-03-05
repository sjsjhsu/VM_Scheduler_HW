import json
import matplotlib
import numpy as np
matplotlib.use('TkAgg')  # 设置后端为 TkAgg
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 处理虚拟机原始数据，包括去除热点，滤波，傅里叶变换

with open(r'D:\PyCharm Projects\data_analysis\Hotspot\Hotspot\vm1.json', 'r') as f:
    data = json.load(f)

# 取出每五分钟的CPU利用率数据
vm_util_all = data['vm_util']  # 假设数据是一个嵌套列表，每个元素代表一个时间点的CPU利用率
vm_util = vm_util_all[0]  # 取出第一个虚拟机的CPU利用率数据
host_vcpus = data['host_vcpus']  # 假设每个虚拟机对应的vCPUs数目
# 假设时间戳是从 'time_start' 开始，每五分钟一采样
time_start = pd.to_datetime(data['time_start'])
time_index = pd.date_range(start=time_start, periods=len(vm_util), freq='5min')  # 每5分钟一个时间点
threshold_factor = 2
time_series = pd.Series(vm_util, index=time_index)

avg_value = vm_util[0]
max_value = vm_util[0]
for idx, utilization in enumerate(vm_util):
    # 如果当前值超过2倍的最大值，则认为是热点
    if utilization > avg_value * threshold_factor:
        vm_util[idx] = max_value  # 将当前点的值设置为当前最大值
    else:
        # 更新最大值
        max_value = max(max_value, utilization)
        avg_value = (avg_value * idx + utilization) / (idx + 1)  # 更新平均值
# print(max_value)

# 将数据转换为pandas的Series
time_series_cleaned = pd.Series(vm_util, index=time_index)

# 3. 去抖动：使用Savitzky-Golay滤波平滑数据，窗口大小为11
smoothed_series = savgol_filter(time_series_cleaned, window_length=11, polyorder=2)

# 绘图部分
plt.figure(figsize=(15, 6))

# 画去除热点后的图
plt.subplot(2, 1, 1)
plt.plot(time_series, label='Original Data', alpha=0.6)
plt.plot(time_series_cleaned, label='Data After Hotspot Removal', color='orange', linewidth=2)
plt.legend(loc='best')
plt.title('Data After Hotspot Removal')

# 画去抖动后的图
plt.subplot(2, 1, 2)
plt.plot(time_series_cleaned, label='Data After Hotspot Removal', color='orange', alpha=0.6)
plt.plot(time_index, smoothed_series, label='Smoothed Data (Savitzky-Golay)', color='green', linewidth=2)
plt.legend(loc='best')
plt.title('Data After Smoothing')

plt.tight_layout()
plt.show()

N = len(smoothed_series)  # 数据长度
fft_values = np.fft.fft(smoothed_series)  # 傅里叶变换

# 2. 计算频率分量
frequencies = np.fft.fftfreq(N)

# 3. 获取幅度和相位
amplitude = np.abs(fft_values)  # 幅度
phase = np.angle(fft_values)  # 相位

# 4. 选择主频率成分，通常选择低频部分
# 你可以通过查看 amplitude 来选择主频成分
# 例如，这里我们假设只保留前 10 个频率成分
num_components = 40
main_frequencies = frequencies[:num_components]
main_T = 1 / main_frequencies
main_amplitudes = amplitude[:num_components]
main_phases = phase[:num_components]


sorted_indices = np.argsort(main_amplitudes)[::-1]
sorted_main_T = main_T[sorted_indices]
sorted_main_amplitudes = main_amplitudes[sorted_indices]
sorted_main_phases = main_phases[sorted_indices]
sorted_main_frequencies = 1 / sorted_main_T

# 5. 重构信号（只使用前10个频率分量）
reconstructed_signal = np.zeros_like(smoothed_series, dtype=complex)
for i in range(num_components):
    reconstructed_signal += sorted_main_amplitudes[i] * np.exp(1j * sorted_main_phases[i]) * np.exp(
        2j * np.pi * sorted_main_frequencies[i] * np.arange(N))

# 6. 可视化结果
plt.figure(figsize=(12, 6))

# print(np.real(reconstructed_signal))
fft_real = np.real(reconstructed_signal)
# 原始数据与拟合数据进行比较
plt.subplot(2, 1, 1)
plt.plot(smoothed_series, label="Original Data")
plt.title("Original Data")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.real(reconstructed_signal), label="Reconstructed from FFT")
plt.title("Reconstructed Data from FFT")
plt.legend()

plt.tight_layout()
plt.show()