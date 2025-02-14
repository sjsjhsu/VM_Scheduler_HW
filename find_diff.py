import numpy as np

import json


# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# 读取虚拟机的 CPU 利用率数据
def load_vm_util(vm_id):
    vm_file = f"Hotspot/Hotspot/{vm_id}"  # 假设虚拟机数据文件存储在该路径
    try:
        return load_json(vm_file)["vm_util"]
    except FileNotFoundError:
        print(f"文件 {vm_file} 未找到")
        return []


# 处理 CPU 利用率数据，确保它们的长度一致（截断为较短的长度）
def align_vm_util(vm_util1, vm_util2):
    # 找出两组数据的最小长度
    min_length = min(len(vm_util1), len(vm_util2))

    # 截断较长的数据
    vm_util1 = vm_util1[:min_length]
    vm_util2 = vm_util2[:min_length]

    return vm_util1, vm_util2


# 查找虚拟机在两个算法中的匹配
def find_diff_pairs(file1, file2):
    # 加载两个算法的匹配结果
    algorithm1_pairs = load_json(file1)
    algorithm2_pairs = load_json(file2)

    # 将配对结果转化为字典形式，键是 vm1，值是匹配的 vm2
    algorithm1_dict = {}
    algorithm2_dict = {}

    for pair in algorithm1_pairs:
        algorithm1_dict[pair["vm1"]] = pair["vm2"]

    for pair in algorithm2_pairs:
        algorithm2_dict[pair["vm1"]] = pair["vm2"]

    diff_pairs = []

    # 遍历从 vm1 到 vm1259 的虚拟机
    for vm_id in range(1, 1260):
        vm_id_str = f"vm{vm_id}.json"  # 获取虚拟机的字符串表示，如 "vm1.json"

        vm1_match_algo1 = algorithm1_dict.get(vm_id_str)
        vm1_match_algo2 = algorithm2_dict.get(vm_id_str)

        # 如果在算法1中有匹配，但算法2中没有，或者两个算法中的匹配虚拟机不同
        if vm1_match_algo1 and not vm1_match_algo2:
            diff_pairs.append({
                "algorithm1": {"vm1": vm_id_str, "vm2": vm1_match_algo1},
                "algorithm2": None
            })
        elif vm1_match_algo2 and not vm1_match_algo1:
            diff_pairs.append({
                "algorithm1": None,
                "algorithm2": {"vm1": vm_id_str, "vm2": vm1_match_algo2}
            })
        elif vm1_match_algo1 != vm1_match_algo2:
            diff_pairs.append({
                "algorithm1": {"vm1": vm_id_str, "vm2": vm1_match_algo1},
                "algorithm2": {"vm1": vm_id_str, "vm2": vm1_match_algo2}
            })

    return diff_pairs


# 保存结果到 JSON 文件
def save_diff(diff, output_file):
    with open(output_file, 'w') as f:
        json.dump(diff, f, indent=4)


# 计算两个虚拟机的 CPU 利用率相加后的统计指标
def calculate_combined_stats(vm_util1, vm_util2):
    # 先对数据进行对齐（截断）
    vm_util1, vm_util2 = align_vm_util(vm_util1, vm_util2)

    # 将两个虚拟机的 CPU 利用率相加
    combined_util = np.add(vm_util1, vm_util2)

    # 计算相加后的统计指标
    avg_combined = np.mean(combined_util)
    p95_combined = np.percentile(combined_util, 95)
    max_combined = np.max(combined_util)

    return avg_combined, p95_combined, max_combined


# 从 diff.json 文件中获取每个虚拟机的匹配情况
def analyze_diff_pairs(diff_file):
    diff_data = load_json(diff_file)

    results = []
    for pair in diff_data:
        # 获取算法一和算法二中的虚拟机ID
        vm1_algo1 = pair["algorithm1"]["vm1"] if pair["algorithm1"] else None
        vm2_algo1 = pair["algorithm1"]["vm2"] if pair["algorithm1"] else None
        vm1_algo2 = pair["algorithm2"]["vm1"] if pair["algorithm2"] else None
        vm2_algo2 = pair["algorithm2"]["vm2"] if pair["algorithm2"] else None

        if vm1_algo1 and vm2_algo1 and vm1_algo2 and vm2_algo2:
            # 获取两个算法中匹配的虚拟机的 CPU 利用率数据
            vm1_util_algo1 = load_vm_util(vm1_algo1)
            vm2_util_algo1 = load_vm_util(vm2_algo1)
            vm1_util_algo2 = load_vm_util(vm1_algo2)
            vm2_util_algo2 = load_vm_util(vm2_algo2)

            # 计算算法一中虚拟机匹配的 CPU 利用率相加后的统计指标
            avg_combined_algo1, p95_combined_algo1, max_combined_algo1 = calculate_combined_stats(vm1_util_algo1,
                                                                                                  vm2_util_algo1)

            # 计算算法二中虚拟机匹配的 CPU 利用率相加后的统计指标
            avg_combined_algo2, p95_combined_algo2, max_combined_algo2 = calculate_combined_stats(vm1_util_algo2,
                                                                                                  vm2_util_algo2)

            results.append({
                "algorithm1": {"vm1": vm1_algo1, "vm2": vm2_algo1},
                "algorithm2": {"vm1": vm1_algo2, "vm2": vm2_algo2},
                "algorithm1_stats": {"avg": avg_combined_algo1, "p95": p95_combined_algo1, "max": max_combined_algo1},
                "algorithm2_stats": {"avg": avg_combined_algo2, "p95": p95_combined_algo2, "max": max_combined_algo2}
            })

    return results


# 主程序
def main():
    # 找不同匹配结果
    file1 = 'json/vm_pairs_scores.json'
    file2 = 'json/vm_pairs_scores_second.json'
    output_file = 'json/diff.json'
    diff = find_diff_pairs(file1, file2)
    save_diff(diff, output_file)

    # 分析
    diff_file = 'json/diff.json'  # diff.json 文件路径
    results = analyze_diff_pairs(diff_file)

    # 保存结果到 JSON 文件
    output_file = "json/analyzed_diff_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
