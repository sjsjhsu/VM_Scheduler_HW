import os
import gzip
import csv
import json
import requests

# 下载文件的基础 URL
base_url = "https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-"

# 固定参数
host_vcpus = 72
time_start = "2018-09-14 16:45:00"
time_end = "2018-09-22 16:40:00"

# 工作目录
download_folder = "./azure_data/"
output_json = "filtered_vm_data.json"

# 确保下载目录存在
if not os.path.exists(download_folder):
    os.makedirs(download_folder)


# 获取第一个文件第一行的虚拟机 ID
def get_target_vm_id():
    file_url = f"{base_url}1-of-195.csv.gz"
    local_file = os.path.join(download_folder, "azure_data-file-1-of-195.csv.gz")
    if not os.path.exists(local_file):
        print(f"正在下载文件 {file_url}...")
        download_file(file_url, local_file)
    print("正在解压文件...")
    with gzip.open(local_file, "rt") as f:
        reader = csv.reader(f)
        for row in reader:
            return row[1]  # 返回第一行的虚拟机 ID


# 下载文件函数
def download_file(url, local_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
    else:
        print(f"文件 {url} 下载失败，状态码: {response.status_code}")


# 提取目标虚拟机的 CPU 利用率
def process_all_files(target_vm_id):
    cpu_utilization = []
    for i in range(1, 196):  # 遍历所有 195 个文件
        file_url = f"{base_url}{i}-of-195.csv.gz"
        local_file = os.path.join(download_folder, f"azure_data-file-{i}-of-195.csv.gz")
        if not os.path.exists(local_file):
            print(f"正在下载文件 {file_url}...")
            download_file(file_url, local_file)
        print(f"正在处理文件 {local_file}...")
        with gzip.open(local_file, "rt") as f:
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
    cpu_utilization = process_all_files(target_vm_id)
    print(f"提取完成，共找到 {len(cpu_utilization)} 条记录。")
    print("正在保存为 JSON 文件...")
    save_to_json(cpu_utilization)


if __name__ == "__main__":
    for i in range(1, 196):  # 遍历所有 195 个文件
        file_url = f"{base_url}{i}-of-195.csv.gz"
        local_file = os.path.join(download_folder, f"azure_data-file-{i}-of-195.csv.gz")
        if not os.path.exists(local_file):
            print(f"正在下载文件 {file_url}...")
            download_file(file_url, local_file)
