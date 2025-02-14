"""
对于两个信号 A1*sin(C1+Bx) + A2*sin(C2+Bx)相加（频率相同）
根据合成公式计算
"""
import numpy as np
import json
from collections import defaultdict

with open("json/vm_analysis_results.json", "r") as f:
    vm_data = json.load(f)

vms = []
for vm in vm_data:
    vms.append({
        "vm_id": vm["vm_id"],
        "periods": vm["top_periods"],
        "amplitudes": vm["top_amplitudes"],
        "phases": vm["top_phases"],
    })


def group_by_amplitude(vms):
    """
    按照幅度对虚拟机进行分组
    :param vms: 虚拟机列表
    :return: 按幅度分组的虚拟机
    """
    groups = defaultdict(list)

    # 定义幅度范围
    for vm in vms:
        amplitude = vm["top_amplitudes"][1]  # 获取幅度字段中的第二个值

        if amplitude < 10:
            group_key = "0-10"
        elif 10 <= amplitude < 60:
            group_key = "10-60"
        elif 60 <= amplitude < 120:
            group_key = "60-120"
        elif 120 <= amplitude < 200:
            group_key = "120-200"
        else:
            group_key = "200+"

        groups[group_key].append(vm)

    return groups


def calculate_score(vm1, vm2):
    """
    计算两个虚拟机的配对得分
    :param vm1: 第一个虚拟机的特征字典，包括 "periods"、"phases"、"amplitudes"
    :param vm2: 第二个虚拟机的特征字典，包括 "periods"、"phases"、"amplitudes"
    :return: 配对得分（数值越大越适合配对）
    """

    periods1 = vm1["periods"]  # 虚拟机1的周期列表
    periods2 = vm2["periods"]  # 虚拟机2的周期列表
    phases1 = vm1["phases"]  # 虚拟机1的相位列表
    phases2 = vm2["phases"]  # 虚拟机2的相位列表
    amplitudes1 = vm1["amplitudes"]  # 虚拟机1的幅度列表
    amplitudes2 = vm2["amplitudes"]  # 虚拟机2的幅度列表
