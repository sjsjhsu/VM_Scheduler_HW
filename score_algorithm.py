# 在vm_analysis处理过原始数据后，对处理后的虚拟机数据进行配对打分，并将结果保存
# 根据周期筛选
# 根据幅度
# 根据相位计算相似度
import numpy as np
import json
from collections import defaultdict

with open("json/vm_analysis_results.json", "r") as f:
    vm_data = json.load(f)

# 提取虚拟机的特征
vms = []
for vm in vm_data:
    vms.append({
        "vm_id": vm["vm_id"],
        "periods": vm["top_periods"],
        "amplitudes": vm["top_amplitudes"],
        "phases": vm["top_phases"],
    })


# 周期分组
def group_by_period(vms, epsilon_T=50):
    """
    按照周期对虚拟机进行分组
    :param vms: 虚拟机列表
    :param epsilon_T: 周期分组的阈值
    :return: 按周期分组的虚拟机
    """
    groups = defaultdict(list)
    for vm in vms:
        # 使用虚拟机的第一个非直流分量的主周期作为分组依据
        main_period = vm["periods"][1]
        group_key = round(main_period / epsilon_T)
        groups[group_key].append(vm)
    return groups


# 配对分数计算函数
def calculate_score(vm1, vm2, w1=1.0, w2=5.0, w3=3.0):
    """
    计算两个虚拟机的配对得分
    :param vm1: 第一个虚拟机的特征字典，包括 "periods"、"phases"、"amplitudes"
    :param vm2: 第二个虚拟机的特征字典，包括 "periods"、"phases"、"amplitudes"
    :param w1: 周期匹配的权重
    :param w2: 相位互补的权重
    :param w3: 幅度匹配的权重
    :return: 配对得分（数值越大越适合配对）
    """

    periods1 = vm1["periods"]  # 虚拟机1的周期列表
    periods2 = vm2["periods"]  # 虚拟机2的周期列表
    phases1 = vm1["phases"]  # 虚拟机1的相位列表
    phases2 = vm2["phases"]  # 虚拟机2的相位列表
    amplitudes1 = vm1["amplitudes"]  # 虚拟机1的幅度列表
    amplitudes2 = vm2["amplitudes"]  # 虚拟机2的幅度列表

    # 1. 计算周期差异
    freq_diff = 0.0
    for i in range(1, len(periods1)):  # 从索引1开始（跳过第一个周期，即直流分量,inf）
        period1 = periods1[i]
        period2 = periods2[i]
        max_period = max(period1, period2)
        if max_period > 0:
            freq_diff += abs(period1 - period2) / max_period
    freq_score = -freq_diff  # 周期差越小，得分越高

    # 2. 计算相位匹配
    phase_score = 0.0
    for i in range(1, len(phases1)):
        phase1 = phases1[i]
        phase2 = phases2[i]
        phase_diff = abs(phase1 - phase2)  # 相位差
        phase_score += np.cos(phase_diff - np.pi)

    # 3. 计算幅度差异
    norm_amp_diff = 0.0
    for i in range(1, len(amplitudes1)):
        amplitude1 = amplitudes1[i]
        amplitude2 = amplitudes2[i]
        max_amplitude = max(amplitude1, amplitude2)
        if max_amplitude > 0:
            norm_amp_diff += abs(amplitude1 - amplitude2) / max_amplitude
    amp_score = -norm_amp_diff  # 幅度差越小，得分越高

    total_score = w1 * freq_score + w2 * phase_score + w3 * amp_score

    return total_score


# 按周期分组
epsilon_T = 50
groups = group_by_period(vms, epsilon_T=epsilon_T)

# 分组内配对
matched = set()
pairs = []

for group_key, group in groups.items():
    n = len(group)
    if n < 2:  # 组内虚拟机不足两台
        continue
    score_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score_matrix[i, j] = calculate_score(group[i], group[j])

    # 匹配组内虚拟机
    group_matched = set()
    for i in range(n):
        if i in group_matched:
            continue
        best_j = None
        best_score = float('-inf')
        for j in range(i + 1, n):
            if j not in group_matched and score_matrix[i, j] > best_score:
                best_score = score_matrix[i, j]
                best_j = j
        if best_j is not None:
            pairs.append((group[i]["vm_id"], group[best_j]["vm_id"], best_score))
            group_matched.add(i)
            group_matched.add(best_j)

# 按得分由高到低排序
sorted_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)
output_file = "json/vm_pairs_scores.json"
output_data = [
    {"vm1": pair[0], "vm2": pair[1], "score": round(pair[2], 3)} for pair in sorted_pairs
]

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)