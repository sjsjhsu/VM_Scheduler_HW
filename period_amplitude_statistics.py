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
        all_periods.append(vm['top_periods'][1])
        # all_periods.append(vm['top_periods'][2])

    # print(len(all_periods))

    # 过滤掉无穷大值
    finite_periods = [p for p in all_periods if p != float('inf')]

    # 统计每个周期出现的次数
    period_counts = Counter(finite_periods)
    # print(len(finite_periods))

    return period_counts


def count_amplitude_ranges(vm_data, step=100):
    # 提取所有虚拟机的幅度数据（第二个值）
    all_amplitudes = []

    for vm in vm_data:
        # all_amplitudes.extend(vm['top_amplitudes'])
        all_amplitudes.append(vm['top_amplitudes'][1])  # 获取第二个幅度值
        # all_amplitudes.append(vm['top_amplitudes'][2])
        # all_amplitudes.append(vm['top_amplitudes'][3])

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


def count_phase_ranges(vm_data, step=0.2):
    all_phases = []

    for vm in vm_data:
        # all_amplitudes.extend(vm['top_amplitudes'])
        all_phases.append(vm['top_phases'][1])  # 获取第二个幅度值
        # all_phases.append(vm['top_phases'][2])
        # all_phases.append(vm['top_phases'][3])

    # 定义幅度范围的边界
    min_phase = min(all_phases)
    max_phase = max(all_phases)

    # 计算范围
    bins = np.arange(min_phase, max_phase + step, step)

    # 统计每个幅度范围内的数量
    phase_ranges = np.digitize(all_phases, bins)

    # 统计每个范围内的幅度数量
    range_counts = Counter(phase_ranges)

    # 创建一个字典来表示每个范围的幅度数量
    range_labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]
    phase_range_counts = {range_labels[i]: range_counts[i + 1] for i in range(len(range_labels))}

    return phase_range_counts


# 将结果保存到 JSON 文件
def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


# 主程序
def main():
    # 统计周期
    # input_file = 'json/vm_analysis_results3.json'
    # output_file = 'json/period_counts.json'
    # vm_data = load_json(input_file)
    # period_counts = count_periods(vm_data)
    # save_json(period_counts, output_file)

    # 统计幅度
    # input_file = 'json/vm_analysis_results3.json'
    # output_file = 'json/amplitude_counts.json'
    # vm_data = load_json(input_file)
    # amplitude_counts = count_amplitude_ranges(vm_data)
    # save_json(amplitude_counts, output_file)

    # 统计相位
    input_file = 'json/vm_analysis_results3.json'
    output_file = 'json/phase_counts.json'
    vm_data = load_json(input_file)
    amplitude_counts = count_phase_ranges(vm_data)
    save_json(amplitude_counts, output_file)
    # 打开并读取 JSON 文件
    with open('json/period_counts.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 输出读取的数据
    # print(data)

    # 如果需要计算所有 value 的总和
    total_sum = 0
    for key, value in data.items():
        try:
            # 将 key 转换为浮点数，再转换为整数
            num = float(key)
            if num % 288 == 0 and num != 2304:  # 检查是否为 144 的整数倍
                total_sum += value
        except ValueError:
            continue

    print(total_sum)


if __name__ == '__main__':
    main()
