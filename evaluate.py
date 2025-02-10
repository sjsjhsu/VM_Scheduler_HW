import os
import json
import matplotlib
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def load_vm_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["vm_util"][0]  # 返回 CPU 利用率数据


# 将两台虚拟机的 CPU 利用率相加
def compute_combined_util(vm1_data, vm2_data):
    # 截取最小长度的数据
    min_length = min(len(vm1_data), len(vm2_data))
    vm1_data = vm1_data[:min_length]
    vm2_data = vm2_data[:min_length]
    return [u1 + u2 for u1, u2 in zip(vm1_data, vm2_data)]


def process_vm_data(vm_data):
    time_start = "2022-09-14 16:35:00"
    time_index = pd.date_range(start=time_start, periods=len(vm_data), freq='5min')  # 每5分钟一个时间点
    threshold_factor = 2

    avg_value = vm_data[0]
    max_value = vm_data[0]
    for idx, utilization in enumerate(vm_data):
        # 如果当前值超过2倍的最大值，则认为是热点
        if utilization > avg_value * threshold_factor:
            vm_data[idx] = max_value  # 将当前点的值设置为当前最大值
        else:
            # 更新最大值
            max_value = max(max_value, utilization)
            avg_value = (avg_value * idx + utilization) / (idx + 1)  # 更新平均值

    # 将数据转换为pandas的Series
    time_series_cleaned = pd.Series(vm_data, index=time_index)

    # 3. 去抖动：使用Savitzky-Golay滤波平滑数据
    smoothed_series = savgol_filter(time_series_cleaned, window_length=40, polyorder=2)

    return smoothed_series


# 计算失真率
def compute_distortion(original, reconstructed):
    return np.linalg.norm(original - reconstructed) / np.linalg.norm(original)


# 可视化
def visualize_results(original, reconstructed, title, distortion_rate):
    plt.figure(figsize=(12, 6))
    plt.plot(original, label="Original (Method 1)", alpha=0.6)
    plt.plot(reconstructed, label="Reconstructed (Method 2)", alpha=0.8, linestyle="--")
    plt.title(f"{title}\nDistortion Rate: {distortion_rate:.4f}")
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("CPU Utilization")
    plt.grid()
    plt.show()


# 计算波动程度的函数
def calculate_relative_variability(signal):
    mean_value = np.mean(signal)  # 均值
    std_dev = np.std(signal)  # 标准差
    rms = np.sqrt(np.mean(signal ** 2))  # 均方根值
    range_val = np.max(signal) - np.min(signal)  # 范围（最大值 - 最小值）

    # 计算变异系数（相对波动率）
    if mean_value != 0:  # 避免除以零
        cv = std_dev / mean_value
    else:
        cv = float('inf')  # 如果均值为零，则返回无穷大，表示无法计算相对波动率

    return std_dev, rms, range_val, cv


def combine_one_pair_original(vm1, vm2, data_folder):
    # CPU 原始利用率直接相加
    vm1_file = os.path.join(data_folder, vm1)
    vm2_file = os.path.join(data_folder, vm2)
    vm1_data = load_vm_data(vm1_file)
    vm2_data = load_vm_data(vm2_file)

    # 计算 CPU 利用率总和
    combined_util = compute_combined_util(vm1_data, vm2_data)
    smoothed_series_vm1 = process_vm_data(vm1_data)
    smoothed_series_vm2 = process_vm_data(vm2_data)
    smoothed_series = process_vm_data(combined_util)

    N = len(smoothed_series)  # 数据长度
    fft_values = np.fft.fft(smoothed_series)  # 傅里叶变换

    # 计算频率分量
    frequencies = np.fft.fftfreq(N)
    # 获取幅度和相位
    amplitude = np.abs(fft_values)  # 幅度
    phase = np.angle(fft_values)  # 相位

    # 4. 选择主频率成分，通常选择低频部分
    # 你可以通过查看 amplitude 来选择主频成分
    num_components = 40
    main_frequencies = frequencies[:num_components]
    main_amplitudes = amplitude[:num_components]
    main_phases = phase[:num_components]

    # 重构信号
    reconstructed_signal = np.zeros_like(smoothed_series, dtype=complex)
    for i in range(num_components):
        reconstructed_signal += main_amplitudes[i] * np.exp(1j * main_phases[i]) * np.exp(
            2j * np.pi * main_frequencies[i] * np.arange(N))

    N1 = len(smoothed_series_vm1)
    fft_values1 = np.fft.fft(smoothed_series_vm1)
    # 计算频率分量
    frequencies1 = np.fft.fftfreq(N1)
    # 获取幅度和相位
    amplitude1 = np.abs(fft_values1)  # 幅度
    phase1 = np.angle(fft_values1)  # 相位

    main_frequencies1 = frequencies1[:num_components]
    main_amplitudes1 = amplitude1[:num_components]
    main_phases1 = phase1[:num_components]
    # 重构信号
    reconstructed_signal1 = np.zeros_like(smoothed_series_vm1, dtype=complex)
    for i in range(num_components):
        reconstructed_signal1 += main_amplitudes1[i] * np.exp(1j * main_phases1[i]) * np.exp(
            2j * np.pi * main_frequencies1[i] * np.arange(N1))

    N2 = len(smoothed_series_vm2)
    fft_values2 = np.fft.fft(smoothed_series_vm2)
    # 计算频率分量
    frequencies2 = np.fft.fftfreq(N2)
    # 获取幅度和相位
    amplitude2 = np.abs(fft_values2)  # 幅度
    phase2 = np.angle(fft_values2)  # 相位

    main_frequencies2 = frequencies2[:num_components]
    main_amplitudes2 = amplitude2[:num_components]
    main_phases2 = phase2[:num_components]
    # 重构信号
    reconstructed_signal2 = np.zeros_like(smoothed_series_vm2, dtype=complex)
    for i in range(num_components):
        reconstructed_signal2 += main_amplitudes2[i] * np.exp(1j * main_phases2[i]) * np.exp(
            2j * np.pi * main_frequencies2[i] * np.arange(N2))

    # 计算虚拟机1、虚拟机2以及组合信号的波动程度
    std_vm1, rms_vm1, range_vm1, cv_vm1 = calculate_relative_variability(np.real(reconstructed_signal1))
    std_vm2, rms_vm2, range_vm2, cv_vm2 = calculate_relative_variability(np.real(reconstructed_signal2))
    std_combined, rms_combined, range_combined, cv_combined = calculate_relative_variability(
        np.real(reconstructed_signal))

    plt.figure(figsize=(12, 6))

    # VM1数据
    plt.subplot(3, 1, 1)
    plt.plot(np.real(reconstructed_signal1), label="VM1 Data")
    plt.title("VM1 Data")
    plt.legend()
    plt.text(0.95, 0.95, f"CV: {cv_vm1:.4f}", ha='right', va='top', transform=plt.gca().transAxes, fontsize=12,
             color='blue')

    # VM2数据
    plt.subplot(3, 1, 2)
    plt.plot(np.real(reconstructed_signal2), label="VM2 Data")
    plt.title("VM2 Data")
    plt.legend()
    plt.text(0.95, 0.95, f"CV: {cv_vm2:.4f}", ha='right', va='top', transform=plt.gca().transAxes, fontsize=12,
             color='blue')

    # 组合数据
    plt.subplot(3, 1, 3)
    plt.plot(np.real(reconstructed_signal), label="Combine Data")
    plt.title("Combine Data")
    plt.legend()
    plt.text(0.95, 0.95, f"CV: {cv_combined:.4f}", ha='right', va='top', transform=plt.gca().transAxes, fontsize=12,
             color='blue')

    plt.tight_layout()
    plt.show()


# 评估失真率
def evaluate_distortion(vm1, vm2, data_folder):
    # 加载虚拟机数据
    vm1_file = os.path.join(data_folder, vm1)
    vm2_file = os.path.join(data_folder, vm2)
    vm1_data = load_vm_data(vm1_file)
    vm2_data = load_vm_data(vm2_file)

    # 方法一 CPU 原始利用率直接相加
    combined_util1 = compute_combined_util(vm1_data, vm2_data)
    smoothed_series = process_vm_data(combined_util1)
    fft_values = np.fft.fft(smoothed_series)  # 傅里叶变换
    reconstructed_series1 = np.fft.ifft(fft_values).real

    # 方法二 先对两个数据分别处理，然后再相加
    min_length = min(len(vm1_data), len(vm2_data))
    smoothed_series1 = process_vm_data(vm1_data[:min_length])
    fft_values1 = np.fft.fft(smoothed_series1)
    smoothed_series2 = process_vm_data(vm2_data[:min_length])
    fft_values2 = np.fft.fft(smoothed_series2)
    # 合成
    combined_fft = fft_values1 + fft_values2
    reconstructed_series2 = np.fft.ifft(combined_fft).real

    # 失真率
    distortion_rate = compute_distortion(np.array(reconstructed_series1), np.array(reconstructed_series2))

    visualize_results(reconstructed_series1, reconstructed_series2, f"Comparison of Methods for {vm1} and {vm2}",
                      distortion_rate)


combine_one_pair_original("vm32.json", "vm655.json", "Hotspot/Hotspot")

# evaluate_distortion("vm1.json", "vm1202.json", "Hotspot/Hotspot")
