import os
import gzip
import csv
import json

# 工作目录和文件路径
input_folder = "./azure_data/"
output_json = "filtered_vm_data4.json"  # 输出的 JSON 文件路径

# 固定参数
host_vcpus = 72
time_start = "2018-09-14 16:45:00"
time_end = "2018-09-22 16:40:00"


# 获取第一个文件第一行的虚拟机 ID
def get_target_vm_id():
    first_file = os.path.join(input_folder, "azure_data-file-1-of-195.csv.gz")
    if not os.path.exists(first_file):
        print(f"第一个文件 {first_file} 不存在，请检查路径。")
        exit()
    print("正在读取第一个文件的虚拟机 ID...")
    with gzip.open(first_file, "rt") as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i == 3:
                return row[1]  # 返回第一行的虚拟机 ID
            i = i + 1


# 提取目标虚拟机的 CPU 利用率
def process_files(target_vm_id):
    cpu_utilization = []
    for i in range(1, 196):
        file_path = os.path.join(input_folder, f"azure_data-file-{i}-of-195.csv.gz")
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，跳过...")
            continue
        print(f"正在处理文件 {file_path}...")
        with gzip.open(file_path, "rt") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    vm_id = row[1]  # 第二列是虚拟机 ID
                    cpu_util = float(row[4])  # 第五列是 CPU 利用率
                    if vm_id == target_vm_id:
                        cpu_utilization.append(round(cpu_util, 3))  # 保留三位小数
                except (IndexError, ValueError):
                    continue  # 跳过无效行
    return cpu_utilization


# 保存为 JSON 文件
def save_to_json(cpu_utilization):
    data = {
        "host_vcpus": host_vcpus,
        "time_start": time_start,
        "time_end": time_end,
        "vm_util": [cpu_utilization]  # 嵌套列表
    }
    with open(output_json, "w") as jsonfile:
        json.dump(data, jsonfile, indent=4)
    print(f"转换完成！JSON 文件已保存为 {output_json}")


# 主函数
def main():
    print("正在获取目标虚拟机 ID...")
    target_vm_id = get_target_vm_id()
    print(f"目标虚拟机 ID: {target_vm_id}")
    print("正在提取目标虚拟机的 CPU 利用率数据...")
    cpu_utilization = process_files(target_vm_id)
    print(f"提取完成，共找到 {len(cpu_utilization)} 条记录。")
    print("正在保存为 JSON 文件...")
    save_to_json(cpu_utilization)


if __name__ == "__main__":
    main()
