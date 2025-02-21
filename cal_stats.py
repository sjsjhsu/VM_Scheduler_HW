import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# 计算合成后的P95利用率相对于合成前两个虚拟机的P95利用率之和的降低比例
def calculate_reduction_ratio(combined_stats, vm1_stats, vm2_stats):
    p95_combined = combined_stats.get('p95', 0)
    p95_vm1 = vm1_stats.get('p95', 0)
    p95_vm2 = vm2_stats.get('p95', 0)

    # 合成前两个虚拟机P95利用率的和
    p95_sum_before = p95_vm1 + p95_vm2

    # 计算下降比例
    if p95_sum_before > 0:
        reduction_ratio = (p95_sum_before - p95_combined) / p95_sum_before * 100
        return reduction_ratio
    return 0  # 如果合成前的P95之和为0，返回0


# 计算合成后的最大利用率相较于合成前两个虚拟机最大利用率中较大的那个增加了多少
def calculate_increase_ratio(combined_stats, vm1_stats, vm2_stats):
    max_combined = combined_stats.get('max', 0)
    max_vm1 = vm1_stats.get('max', 0)
    max_vm2 = vm2_stats.get('max', 0)

    # 合成前两个虚拟机最大利用率中较大的那个
    max_before = max(max_vm1, max_vm2)

    # 计算增加量
    if max_before > 0:
        increase_ratio = (max_combined - max_before) / max_before * 100
        return increase_ratio
    return 0  # 如果合成前的最大利用率为0，返回0


def main():
    # 加载数据
    file_path = 'json/analyzed_stats_results.json'
    data = load_json(file_path)

    reduction_ratios = []

    for pair in data:
        # 获取合成后的P95和两个虚拟机的P95
        combined_stats = pair.get('combined_stats', {})
        vm1_stats = pair.get('vm1_stats', {})
        vm2_stats = pair.get('vm2_stats', {})

        # 计算降低比例
        reduction_ratio = calculate_reduction_ratio(combined_stats, vm1_stats, vm2_stats)

        reduction_ratios.append(reduction_ratio)

    avg_reduction = sum(reduction_ratios) / len(reduction_ratios) if reduction_ratios else 0
    print(f"共配对了{len(reduction_ratios)}组虚拟机，P95利用率的平均下降率: {avg_reduction:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.hist(reduction_ratios, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Reduction Ratios')
    plt.xlabel('Reduction Ratio (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main2():
    file_path = 'json/analyzed_stats_results.json'
    data = load_json(file_path)

    increase_ratios = []

    for pair in data:
        # 获取合成后的最大利用率和两个虚拟机的最大利用率
        combined_stats = pair.get('combined_stats', {})
        vm1_stats = pair.get('vm1_stats', {})
        vm2_stats = pair.get('vm2_stats', {})

        # 计算增加量
        increase_ratio = calculate_increase_ratio(combined_stats, vm1_stats, vm2_stats)

        increase_ratios.append(increase_ratio)

    avg_increase = sum(increase_ratios) / len(increase_ratios) if increase_ratios else 0
    print(f"最高利用率平均增加率: {avg_increase:.2f}")

    plt.figure(figsize=(10, 6))
    plt.hist(increase_ratios, bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Increase in Max Utilization')
    plt.xlabel('Increase in Max Utilization')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main2()
