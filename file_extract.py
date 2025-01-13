import gzip
import shutil

# 指定输入和输出文件路径
input_file = r'/azure_data\vm_cpu_readings-file-60-of-195.csv.gz'  # 替换为您的 .csv.gz 文件路径
output_file = 'azure_data-file-60-of-195.csv'  # 替换为解压后的 .csv 文件路径

# 解压 .csv.gz 文件
with gzip.open(input_file, 'rb') as f_in:
    with open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"文件解压完成，解压后的文件为: {output_file}")
