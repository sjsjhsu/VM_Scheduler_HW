import numpy as np
import matplotlib
import json
import score_algorithm
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')  # 设置后端为 TkAgg

with open("json/filter_vm_analysis_results.json", "r") as f:
    vm_data = json.load(f)

vms = []
for vm in vm_data:
    vms.append({
        "vm_id": vm["vm_id"],
        "index": vm["index"],
        "periods": vm["top_periods"],
        "amplitudes": vm["top_amplitudes"],
        "phases": vm["top_phases"],
    })


def calAffinity(vm1, vm2):
    amplitude1 = vm1["amplitudes"][1]
    amplitude2 = vm2["amplitudes"][1]
    phase1 = vm1["phases"][1]
    phase2 = vm2["phases"][1]
    R = np.sqrt(amplitude1 ** 2 + amplitude2 ** 2 + 2 * amplitude1 * amplitude2 * np.cos(phase1 - phase2))
    if np.cos(phase1 - phase2) < 0:
        return (amplitude1 + amplitude2) / R
    else:
        return 0


def draw_affinity_matrix(vm_data_list):
    vm_periods_day1 = []
    for i in range(0, len(vm_data_list)):
        if vm_data_list[i]["periods"][1] == 288:
            vm_periods_day1.append(vm_data_list[i])

    # 选择前五个 VM
    top5_vms = vm_periods_day1[:6]

    # 初始化 Affinity 矩阵
    affinity_matrix = np.zeros((6, 6))

    # 计算 Affinity
    for i in range(6):
        for j in range(6):
            if i != j:
                affinity_matrix[i, j] = calAffinity(top5_vms[i], top5_vms[j])
            else:
                affinity_matrix[i, j] = 1  # 自己和自己 affinity 为 1

    # 画图
    fig, ax = plt.subplots()
    im = ax.imshow(affinity_matrix, cmap='viridis')

    # 设置坐标轴标签
    vm_ids = [vm["vm_id"] for vm in top5_vms]
    ax.set_xticks(np.arange(len(vm_ids)))
    ax.set_yticks(np.arange(len(vm_ids)))
    ax.set_xticklabels(vm_ids)
    ax.set_yticklabels(vm_ids)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 添加数值标签
    for i in range(len(vm_ids)):
        for j in range(len(vm_ids)):
            text = ax.text(j, i, f"{affinity_matrix[i, j]:.2f}", ha="center", va="center", color="w")

    # 添加标题和颜色条
    ax.set_title("Affinity Matrix of Top 6 VMs (with period 288)")
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()


def calculate_score(vm1, vm2):
    phases1 = vm1["phases"]  # 虚拟机1的相位列表
    phases2 = vm2["phases"]  # 虚拟机2的相位列表

    if np.cos(phases1[1] - phases2[1]) > 0:
        return float('-inf')
    return calAffinity(vm1, vm2)


# 按周期分组
epsilon_T = 288
groups = score_algorithm.group_by_period(vms, epsilon_T=epsilon_T)

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
            pairs.append(
                (group[i]["vm_id"], group[i]["index"], group[best_j]["vm_id"], group[best_j]["index"], best_score))
            group_matched.add(i)
            group_matched.add(best_j)

# 按得分由高到低排序
sorted_pairs = sorted(pairs, key=lambda x: x[4], reverse=True)
output_file = "json/vm_pairs_scores_third.json"
output_data = [
    {"vm1": pair[0], "index1": pair[1], "vm2": pair[2], "index2": pair[3], "score": round(pair[4], 3)} for pair in
    sorted_pairs
]

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
