import os
import json

import matplotlib
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 设置后端为 TkAgg


# 加载虚拟机的 CPU 利用率数据
def load_vm_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["vm_util"][0]  # 返回 CPU 利用率数据


# 计算两台虚拟机的 CPU 利用率总和
def compute_combined_util(vm1_data, vm2_data):
    # 截取最小长度的数据
    min_length = min(len(vm1_data), len(vm2_data))
    vm1_data = vm1_data[:min_length]
    vm2_data = vm2_data[:min_length]
    return [u1 + u2 for u1, u2 in zip(vm1_data, vm2_data)]


def evaluate_one_pair(vm1, vm2, data_folder):
    # 加载虚拟机数据
    vm1_file = os.path.join(data_folder, vm1)
    vm2_file = os.path.join(data_folder, vm2)
    vm1_data = load_vm_data(vm1_file)
    vm2_data = load_vm_data(vm2_file)

    # 时间序列索引（假设时间间隔为 5 分钟）
    time_start = "2022-09-14 16:35:00"  # 替换为实际的起始时间
    min_length = min(len(vm1_data), len(vm2_data))  # 计算最小长度
    time_index = pd.date_range(start=time_start, periods=min_length, freq="5min")
    threshold_factor = 2

    # 计算 CPU 利用率总和
    combined_util = compute_combined_util(vm1_data, vm2_data)

    avg_value = combined_util[0]
    max_value = combined_util[0]
    for idx, utilization in enumerate(combined_util):
        # 如果当前值超过2倍的最大值，则认为是热点
        if utilization > avg_value * threshold_factor:
            combined_util[idx] = max_value  # 将当前点的值设置为当前最大值
        else:
            # 更新最大值
            max_value = max(max_value, utilization)
            avg_value = (avg_value * idx + utilization) / (idx + 1)  # 更新平均值

    # 将数据转换为pandas的Series
    time_series_cleaned = pd.Series(combined_util, index=time_index)

    # 3. 去抖动：使用Savitzky-Golay滤波平滑数据，窗口大小为11
    smoothed_series = savgol_filter(time_series_cleaned, window_length=40, polyorder=2)

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

    # 5. 重构信号（只使用前10个频率分量）
    reconstructed_signal = np.zeros_like(smoothed_series, dtype=complex)
    for i in range(num_components):
        reconstructed_signal += main_amplitudes[i] * np.exp(1j * main_phases[i]) * np.exp(
            2j * np.pi * main_frequencies[i] * np.arange(N))

    # 6. 可视化结果
    plt.figure(figsize=(12, 6))

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


evaluate_one_pair("vm1.json", "vm1202.json", "Hotspot/Hotspot")
