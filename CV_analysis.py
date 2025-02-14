import os
import json

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def load_vm_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["vm_util"][0]  # 返回 CPU 利用率数据


# 计算CV值
def calculate_cv(data):
    mean_val = np.mean(data)
    if mean_val == 0:  # 防止除以零
        return float('inf')
    std_dev = np.std(data)
    return std_dev / mean_val


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


# 计算两个虚拟机合成后的CPU利用率
def compute_combined_util(vm1_data, vm2_data):
    min_length = min(len(vm1_data), len(vm2_data))  # 截取相同长度的数据
    vm1_data = vm1_data[:min_length]
    vm2_data = vm2_data[:min_length]
    return [v1 + v2 for v1, v2 in zip(vm1_data, vm2_data)]


# 处理配对数据
def process_pairings(pairing_file, data_folder, output_file):
    # 读取配对结果文件
    with open(pairing_file, "r") as f:
        pairings = json.load(f)

    results = []

    for pair in pairings:
        vm1_file = os.path.join(data_folder, pair["vm1"])
        vm2_file = os.path.join(data_folder, pair["vm2"])

        # 加载两个虚拟机的数据
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
        amplitude = np.abs(fft_values)  # 幅度
        phase = np.angle(fft_values)  # 相位

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
        reconstructed_signal1 = np.zeros_like(smoothed_series_vm1, dtype=complex)
        for i in range(num_components):
            reconstructed_signal1 += main_amplitudes1[i] * np.exp(1j * main_phases1[i]) * np.exp(
                2j * np.pi * main_frequencies1[i] * np.arange(N1))

        N2 = len(smoothed_series_vm2)
        fft_values2 = np.fft.fft(smoothed_series_vm2)
        # 计算频率分量
        frequencies2 = np.fft.fftfreq(N2)
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

        # 计算单个虚拟机的CV值
        cv_vm1 = calculate_cv(np.real(reconstructed_signal1))
        cv_vm2 = calculate_cv(np.real(reconstructed_signal2))

        # 计算合成后的CPU利用率和CV值
        cv_combined = calculate_cv(np.real(reconstructed_signal))

        # 计算CV变化率
        avg_original_cv = (cv_vm1 + cv_vm2) / 2  # 原始两个虚拟机的平均CV
        if avg_original_cv != 0:  # 防止除以零
            cv_change_rate = ((cv_combined - avg_original_cv) / avg_original_cv) * 100
        else:
            cv_change_rate = float('inf')  # 如果原始CV为零，变化率设为无穷大

        # 存储结果
        results.append({
            "vm1": pair["vm1"],
            "vm2": pair["vm2"],
            "score": pair["score"],
            "cv_vm1": cv_vm1,
            "cv_vm2": cv_vm2,
            "cv_combined": cv_combined,
            "cv_change_rate": cv_change_rate  # 变化率百分比
        })

    # 保存到新的JSON文件
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def process_cv_distribution(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    cv_changes = []

    # 遍历所有配对数据
    for pair in data:
        # 获取 CV 变化率
        cv_change_rate = pair["cv_change_rate"]

        # 限制变化率在 [-100, 100] 范围内
        if cv_change_rate > 100:
            cv_change_rate = 100
        elif cv_change_rate < -100:
            cv_change_rate = -100

        cv_changes.append(cv_change_rate)

    cv_changes = np.array(cv_changes)

    # 统计分布
    distribution = {
        "total_pairs": int(len(cv_changes)),
        "increase_0_10": int(np.sum((cv_changes > 0) & (cv_changes <= 10))),
        "increase_10_50": int(np.sum((cv_changes > 10) & (cv_changes <= 50))),
        "increase_50_100": int(np.sum((cv_changes > 50) & (cv_changes <= 100))),
        "decrease_0_10": int(np.sum((cv_changes < 0) & (cv_changes >= -10))),
        "decrease_10_50": int(np.sum((cv_changes < -10) & (cv_changes >= -50))),
        "decrease_50_100": int(np.sum((cv_changes < -50) & (cv_changes >= -100))),
        "no_change": int(np.sum(cv_changes == 0))
    }

    # 保存统计结果到文件
    with open(output_file, "w") as f:
        json.dump(distribution, f, indent=4)

    print(f"CV 变化分布统计已保存到 {output_file}")


def plot_cv_change_distribution(file_path):
    try:
        # 从文件读取数据
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"无法读取文件: {e}")
        return

    # 提取数据
    categories = [
        "Increase 0-10%",
        "Increase 10-50%",
        "Increase 50-100%",
        "Decrease 0-10%",
        "Decrease 10-50%",
        "Decrease 50-100%",
        "No Change"
    ]

    values = [
        data.get("increase_0_10", 0),
        data.get("increase_10_50", 0),
        data.get("increase_50_100", 0),
        data.get("decrease_0_10", 0),
        data.get("decrease_10_50", 0),
        data.get("decrease_50_100", 0),
        data.get("no_change", 0)
    ]

    # 创建条形图
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=['green', 'green', 'green', 'red', 'red', 'red', 'blue'])

    # 添加标题和标签
    plt.title("CV Change Distribution", fontsize=16)
    plt.ylabel("Number of VM Pairs", fontsize=12)
    plt.xlabel("CV Change Categories", fontsize=12)

    # 显示数值在条形图上
    for i, v in enumerate(values):
        plt.text(i, v + 5, str(v), ha='center', fontsize=10)

    # 调整显示
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    # 显示图形
    plt.show()


def plot_score_vs_cv_change(file_path):
    try:
        # 从文件读取数据
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"无法读取文件: {e}")
        return

    if not isinstance(data, list):
        print("Error: 数据格式不正确，应为包含字典的列表")
        return

    # 提取数据
    scores = []
    cv_changes = []

    for pair in data:
        scores.append(pair["score"])
        # 限制 CV 变化率在 [-100, 100]
        cv_change_rate = pair["cv_change_rate"]
        if cv_change_rate > 100:
            cv_change_rate = 100
        elif cv_change_rate < -100:
            cv_change_rate = -100
        cv_changes.append(cv_change_rate)

    # 创建平面直角坐标系上的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(scores, cv_changes, alpha=0.7, color='blue', edgecolors='black')

    # 添加辅助线
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, label='No Change')  # 横轴参考线
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Score = 0')  # 纵轴参考线

    # 添加网格线
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # 添加标题和标签
    plt.title("Score vs CV Change Rate", fontsize=16)
    plt.xlabel("Score", fontsize=12)
    plt.ylabel("CV Change Rate (%)", fontsize=12)
    plt.legend()

    # 调整坐标轴范围（可选）
    plt.xlim(min(scores) - 1, max(scores) + 1)
    plt.ylim(-110, 110)  # CV变化率限制在 [-110, 110]，便于观察

    # 显示图形
    plt.tight_layout()
    plt.show()


# pairing_file = "vm_pairs_scores.json"
# data_folder = "./Hotspot/Hotspot"
# output_file = "vm_pairs_cv_analysis.json"
# 统计所有配对的虚拟机的cv值和变化率
# process_pairings(pairing_file, data_folder, output_file)

# input_file = "vm_pairs_cv_analysis.json"
# output_file2 = "cv_change_distribution.json"
# 统计cv变化率的分布
# process_cv_distribution(input_file, output_file2)
# plot_cv_change_distribution(output_file2)

# 得分和cv变化的散点图
file_path = "json/vm_pairs_cv_analysis.json"
plot_score_vs_cv_change(file_path)
