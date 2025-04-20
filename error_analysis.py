# 分析公式计算的误差，以判断可信性
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
# Matplotlib中设置字体-黑体，解决Matplotlib中文乱码问题
plt.rcParams['axes.unicode_minus'] = False

# 生成信号
def generate_signal(A1, f1, C1, A2, f2, C2, t):
    """
    生成两个不同频率的正弦波信号
    """
    f1_signal = A1 * np.sin(2 * np.pi * f1 * t + C1)
    f2_signal = A2 * np.sin(2 * np.pi * f2 * t + C2)
    return f1_signal, f2_signal


def simple_case():
    # 设置参数
    A1, f1, C1 = 1, 0.0035, 0  # 第一个正弦波的振幅、频率和相位
    A2, f2, C2 = 1, 0.0070, np.pi / 4  # 第二个正弦波的振幅、频率和相位
    t = np.linspace(0, 1, 1000)  # 时间轴，1秒，1000个数据点

    # 生成信号
    signal1, signal2 = generate_signal(A1, f1, C1, A2, f2, C2, t)

    # 按照公式计算合成信号的振幅 R 和相位 φ
    R, phi = calculate_amplitude_and_phase(A1, A2, C1, C2)

    # 计算合成信号（考虑 R 和 φ）
    combined_signal = R * np.sin(2 * np.pi * f1 * t + phi)

    # 计算误差
    error_simplified = np.linalg.norm(signal1 + signal2 - combined_signal) / np.linalg.norm(signal1 + signal2)

    # 计算绝对误差
    absolute_error = np.linalg.norm(signal1 + signal2 - combined_signal)

    # 计算相位误差
    phase_error = np.abs(np.angle(signal1 + signal2) - np.angle(combined_signal))

    # 输出误差
    print(f"Normalized Error (simplified formula vs direct addition): {error_simplified:.4f}")
    print(f"Absolute Error: {absolute_error:.4f}")
    print(f"Phase Error (in radians): {np.mean(phase_error):.4f}")

    # 绘制图形
    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, signal1 + signal2, label="Combined Signal (Direct Addition)")
    plt.title("Combined Signal from f1 and f2")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, combined_signal, label="Combined Signal (R and φ)", linestyle="--")
    plt.title("Combined Signal with R and φ")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, (signal1 + signal2) - combined_signal, label="Error (Direct - Combined)", color='red')
    plt.title("Error between Direct and Combined")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 读取周期统计文件
def load_period_counts(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# 生成测试用例
def generate_test_cases(period_counts):
    test_cases = []

    for period, count in period_counts.items():

        period = float(period)
        
        # 计算频率，频率是周期的倒数
        frequency = 1 / period

        test_case = {
            "period": period,
            "frequency": frequency,
            "amplitude": np.random.uniform(100, 2000),  # 随机振幅
            "phase": np.random.uniform(-np.pi, np.pi)  # 随机相位
        }
        test_cases.append(test_case)

    return test_cases


# 计算振幅 R 和相位 φ
def calculate_amplitude_and_phase(A1, A2, C1, C2):
    R = np.sqrt(A1 ** 2 + A2 ** 2 + 2 * A1 * A2 * np.cos(C1 - C2))  # 计算振幅
    phi = np.arctan2(A1 * np.sin(C1) + A2 * np.sin(C2), A1 * np.cos(C1) + A2 * np.cos(C2))  # 计算相位
    return R, phi


# 执行合成并计算误差
def analyze_test_cases(test_cases):
    results = []

    for i in range(len(test_cases)):
        for j in range(i + 1, len(test_cases)):
            # 提取两个测试用例
            test_case_1 = test_cases[i]
            test_case_2 = test_cases[j]

            # 提取测试用例的振幅，频率，相位
            A1, f1, C1 = test_case_1['amplitude'], test_case_1['frequency'], test_case_1['phase']
            A2, f2, C2 = test_case_2['amplitude'], test_case_2['frequency'], test_case_2['phase']

            # 按照公式计算合成信号的振幅 R 和相位 φ
            R, phi = calculate_amplitude_and_phase(A1, A2, C1, C2)

            # 合成信号与直接相加的误差计算
            signal_1 = A1 * np.sin(2 * np.pi * f1 * np.linspace(0, 1, 1000) + C1)
            signal_2 = A2 * np.sin(2 * np.pi * f2 * np.linspace(0, 1, 1000) + C2)

            # 计算合成信号（通过振幅 R 和相位 φ）
            combined_signal = R * np.sin(2 * np.pi * f1 * np.linspace(0, 1, 1000) + phi)

            # 直接相加信号
            direct_sum = signal_1 + signal_2

            # 计算误差
            error = np.linalg.norm(direct_sum - combined_signal) / np.linalg.norm(direct_sum)
            absolute_error = np.linalg.norm(direct_sum - combined_signal)
            phase_error = np.abs(np.angle(direct_sum) - np.angle(combined_signal))

            result = {
                "test_case_1": test_case_1,
                "test_case_2": test_case_2,
                "calculated_amplitude_R": R,
                "calculated_phase_phi": phi,
                "error": error,
                "absolute_error": absolute_error,
                "phase_error": np.mean(phase_error)
            }
            results.append(result)

    return results


# 保存结果函数
def save_results(results, output_file):
    def convert_ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError("Unsupported type")

    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=convert_ndarray_to_list)
        print(f"Results saved to {output_file}")
    except TypeError as e:
        print(f"Error: {e}")


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


# 统计 error 字段的分布
def plot_error_distribution(file_path):
    data = load_json(file_path)

    errors = []
    for entry in data:
        if 'error' in entry:
            errors.append(entry['error'])

    bins = np.arange(0, max(errors) + 0.1, 0.1)  # 以0.1为步长的分段区间

    # 统计每个分段的数量
    counts, bin_edges = np.histogram(errors, bins=bins)

    print("Error Value Distribution:")
    for i in range(len(counts)):
        print(f"Range {bin_edges[i]:.1f} - {bin_edges[i + 1]:.1f}: {counts[i]}")

    # 4. 绘制误差分布的柱状图
    plt.figure(figsize=(5, 3), facecolor='white')
    plt.hist(errors, bins=30, color='black', alpha=1)
    plt.xlabel('误差', fontsize=10, color='black')
    plt.ylabel('频率', fontsize=10, color='black')
    plt.xticks(fontsize=12, color='black')
    plt.yticks(fontsize=12, color='black')

    # 显示图表
    plt.tight_layout()
    plt.show()


def main():
    input_file = 'json/period_counts.json'
    output_file = 'json/error_analysis_results.json'

    period_counts = load_period_counts(input_file)

    test_cases = generate_test_cases(period_counts)

    results = analyze_test_cases(test_cases)

    save_results(results, output_file)
    print(f"测试结果已保存到 {output_file}")


if __name__ == '__main__':
    main()
    # simple_case()
    plot_error_distribution('json/error_analysis_results.json')
