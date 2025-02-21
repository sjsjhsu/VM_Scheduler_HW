import os
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# 用傅里叶变换处理所有虚拟机原始数据，保留每个虚拟机幅度最大的前四个分量的周期，幅度，相位

input_folder = "./Hotspot/Hotspot/"
output_json = "./json/vm_analysis_results2.json"


# 处理单个虚拟机数据
def process_vm_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 提取虚拟机的 CPU 利用率和时间相关信息
    vm_util_all = data['vm_util']
    vm_util = vm_util_all[0]  # 取出 CPU 利用率数据
    time_start = pd.to_datetime(data['time_start'])
    time_index = pd.date_range(start=time_start, periods=len(vm_util), freq='5min')
    threshold_factor = 2

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

    # 将数据转换为pandas的Series
    time_series_cleaned = pd.Series(vm_util, index=time_index)

    # 去抖动：平滑处理
    smoothed_series = savgol_filter(time_series_cleaned, window_length=40, polyorder=2)

    # 计算傅里叶变换
    N = len(smoothed_series)
    fft_values = np.fft.fft(smoothed_series)
    frequencies = np.fft.fftfreq(N)
    amplitude = np.abs(fft_values)  # 幅度
    phase = np.angle(fft_values)  # 相位

    # 选择主要频率成分
    num_components = 40
    main_frequencies = frequencies[:num_components]
    main_T = 1 / main_frequencies
    main_amplitudes = amplitude[:num_components]
    main_phases = phase[:num_components]

    # 对幅度排序，找出最重要的频率成分
    sorted_indices = np.argsort(main_amplitudes)[::-1]
    top_periods = main_T[sorted_indices[:4]]  # 最重要的三个周期
    top_amplitudes = main_amplitudes[sorted_indices[:4]]  # 最重要的三个幅度
    top_phases = main_phases[sorted_indices[:4]]  # 最重要的三个相位

    # 如果有有效结果，返回前三个的周期、幅度、相位
    if len(top_periods) > 0:
        return top_periods.tolist(), top_amplitudes.tolist(), top_phases.tolist()
    else:
        return None, None, None  # 如果数据无效，返回 None


# 遍历文件夹中的所有 JSON 文件
results = []
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):  # 检查是否为 JSON 文件
        file_path = os.path.join(input_folder, filename)
        print(f"正在处理文件: {filename}")
        periods, amplitudes, phases = process_vm_file(file_path)
        if periods is not None:
            results.append({
                "vm_id": filename,  # 虚拟机 ID
                "top_periods": periods,  # 前四个周期
                "top_amplitudes": amplitudes,  # 前四个幅度
                "top_phases": phases  # 前四个相位
            })

# with open('top.txt', 'r') as f:
#     vm_ids = f.readlines()  # 读取所有虚拟机 ID，每行一个
#     vm_ids = [vm_id.strip() for vm_id in vm_ids]  # 去除多余的空格和换行符
#
#     for vm_id in vm_ids:
#         # 构造虚拟机 JSON 文件路径
#         json_file = f"vm{vm_id}.json"
#         file_path = os.path.join(input_folder, json_file)
#
#         if os.path.exists(file_path):  # 确保文件存在
#             print(f"正在处理文件: {json_file}")
#             periods, amplitudes, phases = process_vm_file(file_path)
#
#             if periods is not None:
#                 results.append({
#                     "vm_id": vm_id,  # 虚拟机 ID
#                     "top_periods": periods,  # 前四个周期
#                     "top_amplitudes": amplitudes,  # 前四个幅度
#                     "top_phases": phases  # 前四个相位
#                 })
#         else:
#             print(f"文件 {json_file} 未找到，跳过该虚拟机。")

# 保存结果到 JSON 文件
with open(output_json, "w") as jsonfile:
    json.dump(results, jsonfile, indent=4)
print(f"分析结果已保存到 {output_json}")
