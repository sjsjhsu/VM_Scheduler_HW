# import json
#
# # 指定需要统计的 JSON 文件路径
# json_file = 'alibaba_json/m_1.json'  # 替换为您的 JSON 文件路径
#
# # 读取 JSON 文件并统计
# with open(json_file, 'r') as file:
#     data = json.load(file)  # 加载 JSON 数据
#     vm_util = data.get("vm_util", [])  # 获取 "vm_util" 的内容
#     count = len(vm_util)  # 统计元素数量
#
# print(f"JSON 文件 '{json_file}' 中共有 {count} 组时间戳和 CPU 利用率数据。")


import json
import os

# 输入和输出文件路径
input_folder = 'alibaba_json/'  # 原始 JSON 文件所在文件夹路径
output_folder = 'new_alibaba_json/'  # 新的 JSON 文件输出文件夹路径

# 固定参数
host_vcpus = 72
time_start = "2022-09-14 16:45:00"
time_end = "2022-09-22 16:40:00"

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历原始 JSON 文件并进行转换
for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 读取原始 JSON 文件
        with open(input_path, 'r') as infile:
            data = json.load(infile)

        # 提取 vm_util 的 CPU 利用率列表
        cpu_utils = [entry["cpu_util"] / 100 for entry in data.get("vm_util", [])]

        # 创建新格式的 JSON 数据（注意 vm_util 是嵌套的列表）
        new_data = {
            "host_vcpus": host_vcpus,
            "time_start": time_start,
            "time_end": time_end,
            "vm_util": [cpu_utils]  # 嵌套列表
        }

        # 写入新格式的 JSON 文件
        with open(output_path, 'w') as outfile:
            json.dump(new_data, outfile, indent=4)

print(f"所有文件已转换完成！新文件保存在 {output_folder}")

