import numpy as np

import json
from collections import Counter


# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# 统计周期出现的次数
def count_periods(vm_data):
    # 提取所有虚拟机的周期数据
    all_periods = []

    for vm in vm_data:
        all_periods.extend(vm['top_periods'])

    # 过滤掉无穷大值
    finite_periods = [p for p in all_periods if p != float('inf')]

    # 统计每个周期出现的次数
    period_counts = Counter(finite_periods)

    return period_counts


# 统计幅度（第二个值）
def count_amplitude_ranges(vm_data, step=20):
    # 提取所有虚拟机的幅度数据（第二个值）
    all_amplitudes = []

    for vm in vm_data:
        all_amplitudes.append(vm['top_amplitudes'][1])  # 获取第二个幅度值

    # 定义幅度范围的边界
    min_amplitude = min(all_amplitudes)
    max_amplitude = max(all_amplitudes)

    # 计算范围
    bins = np.arange(min_amplitude, max_amplitude + step, step)

    # 统计每个幅度范围内的数量
    amplitude_ranges = np.digitize(all_amplitudes, bins)

    # 统计每个范围内的幅度数量
    range_counts = Counter(amplitude_ranges)

    # 创建一个字典来表示每个范围的幅度数量
    range_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
    amplitude_range_counts = {range_labels[i]: range_counts[i + 1] for i in range(len(range_labels))}

    return amplitude_range_counts


# 将结果保存到 JSON 文件
def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


# 主程序
def main():
    # 统计周期
    # input_file = 'json/vm_analysis_results.json'
    # output_file = 'json/period_counts.json'
    # vm_data = load_json(input_file)
    # period_counts = count_periods(vm_data)
    # save_json(period_counts, output_file)

    # 统计幅度
    input_file = 'json/vm_analysis_results.json'
    output_file = 'json/amplitude_counts.json'
    vm_data = load_json(input_file)
    amplitude_counts = count_amplitude_ranges(vm_data)
    save_json(amplitude_counts, output_file)


if __name__ == '__main__':
    main()
