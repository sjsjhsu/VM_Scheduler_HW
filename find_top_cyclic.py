import pandas as pd

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def load_vm_util(vm_id):
    vm_file = f"Hotspot/Hotspot/{vm_id}.json"
    try:
        return load_json(vm_file)["vm_util"][0]
    except FileNotFoundError:
        print(f"文件 {vm_file} 未找到")
        return []


# 去除热点数据：将超过阈值的点设置为当前最大值
def remove_hotspots(vm_util, threshold_factor=2):
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
    return vm_util


# 处理数据：去除热点，使用 Savitzky-Golay 滤波平滑数据
def process_vm_data(vm_util):
    time_start = "2022-09-14 16:35:00"
    time_index = pd.date_range(start=time_start, periods=len(vm_util), freq='5min')  # 每5分钟一个时间点
    # 去除热点
    vm_util_cleaned = pd.Series(remove_hotspots(vm_util), index=time_index)

    # 使用 Savitzky-Golay 滤波平滑数据
    smoothed_series = savgol_filter(vm_util_cleaned, window_length=40, polyorder=2)
    return smoothed_series


# 执行傅里叶变换，计算频率分量
def fft_transform(smoothed_series):
    N = len(smoothed_series)  # 数据长度
    fft_values = np.fft.fft(smoothed_series)  # 傅里叶变换

    # 计算频率分量
    frequencies = np.fft.fftfreq(N)

    # 获取幅度和相位
    amplitude = np.abs(fft_values)  # 幅度
    phase = np.angle(fft_values)  # 相位

    num_components = 40
    main_frequencies = frequencies[:num_components]
    main_T = 1 / main_frequencies
    main_amplitudes = amplitude[:num_components]
    main_phases = phase[:num_components]

    reconstructed_signal = np.zeros_like(smoothed_series, dtype=complex)
    for i in range(num_components):
        reconstructed_signal += main_amplitudes[i] * np.exp(1j * main_phases[i]) * np.exp(
            2j * np.pi * main_frequencies[i] * np.arange(N))

    return reconstructed_signal


# 重建信号：仅使用前 40 个频率成分
def reconstruct_signal(frequencies, amplitude, phase, num_components=40):
    N = len(frequencies)
    reconstructed_signal = np.zeros_like(amplitude, dtype=complex)
    for i in range(num_components):
        reconstructed_signal += amplitude[i] * np.exp(1j * phase[i]) * np.exp(
            2j * np.pi * frequencies[i] * np.arange(N))

    return np.real(reconstructed_signal)


# 绘制原始信号和重建信号的对比图并保存结果
def plot_signals(smoothed_series, reconstructed_signal, vm_id):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(smoothed_series, label="Original Data")
    plt.title("Original Data")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.real(reconstructed_signal), label="Reconstructed from FFT")
    plt.title("Reconstructed Data from FFT")
    plt.legend()

    # 保存图像
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{vm_id}_fft_reconstruction.png")
    plt.close()


def main():
    for vm_num in range(1, 1260):  # 遍历 vm1 到 vm1259
        vm_id = f"vm{vm_num}"
        print(f"Processing {vm_id}...")

        vm_util = load_vm_util(vm_id)

        if not vm_util:
            continue  # 如果没有数据，跳过

        # 处理数据
        smoothed_series = process_vm_data(vm_util)

        # 执行傅里叶变换
        reconstructed_signal = fft_transform(smoothed_series)

        # 绘制并保存结果
        plot_signals(smoothed_series, reconstructed_signal, vm_id)


if __name__ == "__main__":
    main()

