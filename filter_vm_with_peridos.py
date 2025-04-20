import os
import shutil

import json
import numpy as np


def filter_vms(data):
    filtered_vms = []

    for vm in data:
        # 忽略第一个分量（直流分量），只考虑交流分量
        amplitudes = vm['top_amplitudes'][1:4]
        periods = vm['top_periods'][1:4]
        phases = vm['top_phases'][1:4]

        # if any(period > 288 for period in periods):
        #     continue
        if periods[0] in [288, 144, 96, 72] and periods[1] in [288, 144, 96, 72]:
            if periods[2] > 288 and amplitudes[2] > 0.6 * amplitudes[0]:
                continue
            filtered_vms.append(vm)
            continue

        if periods[0] > 288:
            continue
        if periods[1] > 288 and periods[2] > 288 and amplitudes[1] > 0.3 * amplitudes[0] and amplitudes[2] > 0.3 * amplitudes[0]:
            continue
        # if periods[1] > 288 and amplitudes[1] > 0.6 * amplitudes[0]:
        #     continue
        # if periods[2] > 288 and amplitudes[2] > 0.6 * amplitudes[0]:
        #     continue

            # 检查是否存在两个分量的合成幅值小于另一个分量的110%，且该分量的周期是288或144
        for i in range(3):
            # 获取另外两个分量的幅度和相位
            other_amplitudes = [amplitudes[j] for j in range(3) if j != i]
            other_phases = [phases[j] for j in range(3) if j != i]

            # 计算两个分量的合成幅值
            amplitude1, amplitude2 = other_amplitudes
            phase1, phase2 = other_phases
            R = np.sqrt(amplitude1 ** 2 + amplitude2 ** 2 + 2 * amplitude1 * amplitude2 * np.cos(phase1 - phase2))

            # 检查条件
            if R < 1.1 * amplitudes[i] and periods[i] in [144, 288]:
                filtered_vms.append(vm)
                break  # 满足条件即可，无需重复检查

    return filtered_vms


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):  # 只处理 PNG 文件
                file_path = os.path.join(folder_path, filename)
                try:
                    os.unlink(file_path)  # 删除文件
                except Exception as e:
                    print(f"无法删除 {file_path}: {e}")
    else:
        os.makedirs(folder_path)  # 如果文件夹不存在，则创建它


def copy_filtered_vms_images(filtered_vms, image_folder, output_folder):
    clear_folder(output_folder)

    for vm_id in filtered_vms:
        # 将后缀名从.json改为.png
        png_filename = vm_id.replace(".json", ".png")
        source_image_path = os.path.join(image_folder, png_filename)
        dest_image_path = os.path.join(output_folder, png_filename)

        # 如果图片存在，则复制到输出文件夹
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, dest_image_path)


# 读取JSON文件
with open('json/vm_analysis_results3.json', 'r') as file:
    data = json.load(file)

# 筛选符合条件的虚拟机
filtered_vms = filter_vms(data)

# if "vm108.json" in filtered_vms:
#     print("该虚拟机满足条件。")
# else:
#     print("该虚拟机不满足条件。")

# 输出结果
# print(filtered_vms)
print(len(filtered_vms))
with open('json/filter_vm_analysis_results.json', "w") as jsonfile:
    json.dump(filtered_vms, jsonfile, indent=4)

# image_folder = "results"  # 虚拟机图片所在的文件夹
# output_folder = "periods_results"  # 输出文件夹，用于保存筛选后的图片
#
# copy_filtered_vms_images(filtered_vms, image_folder, output_folder)
