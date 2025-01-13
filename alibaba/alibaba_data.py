import csv
import json
import os

# 输入和输出文件路径
input_csv = 'machine_usage.csv'  # 替换为您的CSV文件路径
output_folder = 'alibaba_json/'  # 输出JSON文件的文件夹路径

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 存储文件句柄以便逐行写入
file_handles = {}

# 读取CSV文件
with open(input_csv, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 3:  # 确保行有足够的数据列
            continue

        vm_id = row[0]  # 虚拟机ID
        timestamp = row[1]  # 时间戳
        cpu_util = row[2]  # CPU利用率

        # 检查CPU利用率是否为空或非数字
        if not cpu_util.isdigit():
            continue  # 跳过无效行

        # 准备当前虚拟机的文件句柄
        if vm_id not in file_handles:
            output_path = os.path.join(output_folder, f"{vm_id}.json")
            file_handles[vm_id] = open(output_path, 'w')
            file_handles[vm_id].write('{"vm_util": [')  # 初始化 JSON 文件

        # 写入数据到对应虚拟机的 JSON 文件
        file_handles[vm_id].write(json.dumps({"timestamp": int(timestamp), "cpu_util": int(cpu_util)}))
        file_handles[vm_id].write(",")  # 每条记录后加逗号

# 关闭所有文件句柄并修正 JSON 格式
for vm_id, handle in file_handles.items():
    handle.seek(handle.tell() - 1, os.SEEK_SET)  # 移动文件指针，去掉最后的逗号
    handle.write("]}")  # 结束 JSON 文件
    handle.close()

print(f"转换完成！JSON 文件已保存到 {output_folder}")

