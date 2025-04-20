import os
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# 用傅里叶变换处理所有虚拟机原始数据，保留每个虚拟机幅度最大的前四个分量的周期，幅度，相位

input_folder = "./Hotspot/Hotspot/"
output_json = "./json/vm_analysis_results3.json"


# 处理单个虚拟机数据
def process_vm_file(file_path, i=0):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 提取虚拟机的 CPU 利用率和时间相关信息
    vm_util_all = data['vm_util']
    vm_util = vm_util_all[i]  # 取出 CPU 利用率数据
    time_start = pd.to_datetime(data['time_start'])
    time_index = pd.date_range(start=time_start, periods=len(vm_util), freq='5min')
    time_series_cleaned = pd.Series(vm_util, index=time_index)

    # 去抖动：平滑处理
    smoothed_series = savgol_filter(time_series_cleaned, window_length=41, polyorder=2)

    # 计算傅里叶变换
    N = len(smoothed_series)
    fft_values = np.fft.fft(smoothed_series)
    frequencies = np.fft.fftfreq(N)
    amplitude = np.abs(fft_values)  # 幅度
    phase = np.angle(fft_values)  # 相位
    T = 1 / frequencies

    # 对幅度排序，找出最重要的频率成分
    positive_freq_indices = np.where(T > 0)  # 只选取正频率
    sorted_indices = np.argsort(amplitude[positive_freq_indices])[::-1]
    top_periods = T[sorted_indices[:4]]  # 最重要的三个周期
    top_amplitudes = amplitude[sorted_indices[:4]]  # 最重要的三个幅度
    top_phases = phase[sorted_indices[:4]]  # 最重要的三个相位

    # 如果有有效结果，返回前三个的周期、幅度、相位
    if len(top_periods) > 0:
        return top_periods.tolist(), top_amplitudes.tolist(), top_phases.tolist()
    else:
        return None, None, None  # 如果数据无效，返回 None


# # 遍历文件夹中的所有 JSON 文件
# results = []
# for filename in os.listdir(input_folder):
#     if filename.endswith(".json"):  # 检查是否为 JSON 文件
#         file_path = os.path.join(input_folder, filename)
#         print(f"正在处理文件: {filename}")
#         for i in range(10):
#             periods, amplitudes, phases = process_vm_file(file_path, i)
#             if periods is not None:
#                 results.append({
#                     "vm_id": filename,  # 虚拟机 ID
#                     "index": i,
#                     "top_periods": periods,  # 前四个周期
#                     "top_amplitudes": amplitudes,  # 前四个幅度
#                     "top_phases": phases  # 前四个相位
#                 })
#
# # 保存结果到 JSON 文件
# with open(output_json, "w") as jsonfile:
#     json.dump(results, jsonfile, indent=4)


with open("json/data.json", "w") as jsonfile:
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):  # 检查是否为 JSON 文件
            file_path = os.path.join(input_folder, filename)
            print(f"正在处理文件: {filename}")
            for i in range(10):
                periods, amplitudes, phases = process_vm_file(file_path, i)
                if periods is not None:
                    result = {
                        "vm_id": filename + "_" + str(i),  # 虚拟机 ID
                        "first_period": periods[1],
                        "first_amplitude": amplitudes[1],
                        "second_period": periods[2],
                        "second_amplitude": amplitudes[2],
                        "third_period": periods[3],
                        "third_amplitude": amplitudes[3],
                    }
                # 每次写入一个虚拟机的结果
                jsonfile.write(json.dumps(result, ensure_ascii=False) + "\n")
