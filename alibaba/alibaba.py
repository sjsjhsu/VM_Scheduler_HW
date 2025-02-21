import json

import matplotlib
import numpy as np
from scipy.optimize import curve_fit
from statsmodels.tsa.stl._stl import STL

matplotlib.use('TkAgg')  # 设置后端为 TkAgg

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 假设数据存储在vm_util.json文件中
with open(r'D:\PyCharm Projects\data_analysis\new_alibaba_json\m_50.json', 'r') as f:
    data = json.load(f)

# 取出每五分钟的CPU利用率数据
vm_util_all = data['vm_util']  # 假设数据是一个嵌套列表，每个元素代表一个时间点的CPU利用率
vm_util = vm_util_all[0]  # 取出第一个虚拟机的CPU利用率数据
host_vcpus = data['host_vcpus']  # 假设每个虚拟机对应的vCPUs数目
# 假设时间戳是从 'time_start' 开始
time_start = pd.to_datetime(data['time_start'])
time_index = pd.date_range(start=time_start, periods=len(vm_util), freq='12s')
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
smoothed_series = savgol_filter(time_series_cleaned, window_length=40, polyorder=2)
print(smoothed_series)
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

# -------------------- 傅里叶变换 --------------------
def fft_fit(data):
    N = len(data)
    freqs = np.fft.fftfreq(N)
    fft_values = np.fft.fft(data)

    # 选择主要的频率分量（通常是低频成分）
    half_N = N // 2
    fft_values[half_N:] = 0  # 只保留正频率部分

    # 逆FFT恢复信号
    fitted_data = np.fft.ifft(fft_values)

    return fitted_data.real


# -------------------- 三角函数拟合 --------------------
def sine_wave(t, A, omega, phi, C):
    return A * np.sin(omega * t + phi) + C


def triangle_fit(data):
    time = np.arange(len(data))

    # 使用curve_fit进行拟合，初始猜测参数
    p0 = [np.max(data) - np.min(data), 2 * np.pi / len(data), 0, np.mean(data)]
    popt, _ = curve_fit(sine_wave, time, data, p0=p0, maxfev=5000)

    # 使用拟合的参数生成拟合数据
    fitted_data = sine_wave(time, *popt)

    return fitted_data


# -------------------- STL 分解 --------------------
def stl_decomposition(data):
    # 转换为时间序列，假设周期为一天（或者根据需要选择周期）
    time_index = pd.date_range(start="2022-01-01", periods=len(data), freq="D")
    series = pd.Series(data, index=time_index)

    # 使用STL分解，周期设置为1440（即一天的分钟数）
    stl = STL(series)
    result = stl.fit()

    trend_component = result.trend
    seasonal_component = result.seasonal
    residual_component = result.resid

    return trend_component, seasonal_component, residual_component


# fft_fitted = fft_fit(smoothed_series)

# 2. 三角函数拟合
# triangle_fitted = triangle_fit(smoothed_series)

# 3. STL分解
trend, seasonal, residual = stl_decomposition(smoothed_series)

# 绘制结果
# plt.figure(figsize=(12, 10))
#
# plt.subplot(2, 1, 1)
# plt.plot(smoothed_series, label='Original Data')
# plt.title('Original Data')
# plt.legend(loc='best')

# plt.subplot(3, 1, 2)
# plt.plot(fft_fitted, label='FFT Fitted Data', color='orange')
# plt.title('FFT Fitted Data')
# plt.legend(loc='best')

# plt.subplot(4, 1, 3)
# plt.plot(triangle_fitted, label='Triangle Fitted Data', color='green')
# plt.title('Triangle Function Fitted Data')
# plt.legend(loc='best')

# plt.subplot(2, 1, 2)
# plt.plot(trend, label='Trend Component', color='blue')
# plt.plot(seasonal, label='Seasonal Component', color='red')
# plt.plot(residual, label='Residual Component', color='purple')
# plt.title('STL Decomposition')
# plt.legend(loc='best')
#
# plt.tight_layout()
# plt.show()


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

# 打印前几个频率成分的信息
print("\nMain Frequencies (first 40):")
print(main_frequencies)  # 打印主频率成分
print("\nMain Periods (first 40):")
print(main_T)  # 打印主周期成分的周期
print("\nMain Amplitudes (first 40):")
print(main_amplitudes)  # 打印主频率成分的幅度
print("\nMain Phases (first 40):")
print(main_phases)  # 打印主频率成分的相位
sorted_indices = np.argsort(main_amplitudes)[::-1]
sorted_main_T = main_T[sorted_indices]
sorted_main_amplitudes = main_amplitudes[sorted_indices]

# 打印排序后的结果
print("排序后的周期 (main_T):", sorted_main_T)
print("排序后的振幅 (main_amplitudes):", sorted_main_amplitudes)

# 5. 重构信号（只使用前10个频率分量）
reconstructed_signal = np.zeros_like(smoothed_series, dtype=complex)
for i in range(num_components):
    reconstructed_signal += main_amplitudes[i] * np.exp(1j * main_phases[i]) * np.exp(
        2j * np.pi * main_frequencies[i] * np.arange(N))

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

# data = fft_real

# # 1. 定义三角函数模型
# def sine_wave(t, A, f, phi, C):
#     """三角函数模型，正弦波加上常数偏移"""
#     return A * np.sin(2 * np.pi * f * t + phi) + C
#
#
# # 2. 将数据分割为若干个周期，每个周期288个数据点
# period_length = 2888
# num_periods = len(data) // period_length  # 计算周期数
# # num_periods = int(num_periods / 20)
# # 绘制每个周期的拟合结果
# plt.figure(figsize=(16, 12))
# for i in range(num_periods):
#     # 取出第i个周期的数据
#     period_data = data[i * period_length: (i + 1) * period_length]
#
#     triangle_fitted = triangle_fit(period_data)
#
#     # 2.5. 绘制原始数据和拟合结果
#     plt.subplot(num_periods, 1, i + 1)
#     plt.plot(period_data, label='Original Data', color='blue')
#     plt.plot(triangle_fitted, label='Fitted Curve', color='red', linestyle='--')
#     plt.legend(loc='best')
#     plt.title(f'Period {i + 1}')
#
# plt.tight_layout()
# plt.show()
