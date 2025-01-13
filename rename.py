import os


def rename_vm_files(folder_path):
    """
    将存储虚拟机CPU利用率的文件按照 'vm1.json', 'vm2.json' 等递增顺序重命名。

    :param folder_path: 包含虚拟机文件的文件夹路径
    """
    # 获取文件夹中所有文件的列表
    file_list = os.listdir(folder_path)

    # 过滤出 .json 文件（如果不是 .json 文件，可以改成其他后缀名）
    json_files = [file for file in file_list if file.endswith(".json")]

    # 对文件名进行排序，确保重命名的顺序一致
    json_files.sort()

    # 遍历文件并重新命名
    for i, old_name in enumerate(json_files):
        # 新文件名，例如 'vm1.json', 'vm2.json', ...
        new_name = f"vm{i + 1}.json"

        # 构建完整的文件路径
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {old_name} -> {new_name}")

    print("所有文件已重命名完成！")


# 示例调用
folder_path = "./Hotspot/Hotspot"  # 替换为你的文件夹路径
rename_vm_files(folder_path)
