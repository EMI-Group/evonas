import pandas as pd
import matplotlib.pyplot as plt
import os

target_dir = '/data/code_yzh/DistillNAS/runs/search_nyu/run22_search_random'
# Ensure output directory exists
output_dir = os.path.join(target_dir, 'show_evo_pics/')
os.makedirs(output_dir, exist_ok=True)

# Load external CSV file
input_csv = os.path.join(target_dir, 'pop.csv')
df = pd.read_csv(input_csv)
max_gen = df['gen'].max()


from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

def calc_hv(ref_pt, F, normalized=True):
    # 非支配前沿
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    nd_F = F[front, :]
    ref_point = 1.01 * np.array(ref_pt)
    hv = HV(ref_point=ref_point)(nd_F)
    if normalized:
        hv = hv / np.prod(ref_point)
    return hv


# 只需选择两个或三个目标，HV通常支持2/3维
objectives = ['abs_rel', 'latency', 'macs']  # 或 ['abs_rel', 'latency', 'macs']，看你目标维数

# 归一化目标值
max_vals = {'abs_rel':0.5, 'latency':50.0, 'macs':50.0}
df_normalized = df.copy()
for obj in objectives:
    df_normalized[obj] = df[obj] / max_vals[obj]  # 归一化

# 选定参考点（以所有数据最大值*1.01）
# ref_pt = [df[obj].max() for obj in objectives]
# ref_pt=[np.float64(0.1776), np.float64(33.4427), np.float64(37.6922)]
# ref_pt=[np.float64(0.3039), np.float64(34.4057), np.float64(37.7665)]
ref_pt=[1.0,1.0,1.0]
print(f'ref_point: {ref_pt}')

grouped = df_normalized.groupby('gen')
gens = sorted(grouped.groups.keys())
hv_list = []
prev_front = None

for g in gens:
    curr_F = grouped.get_group(g)[objectives].values
    if prev_front is None:
        merged_F = curr_F
    else:
        merged_F = np.vstack([curr_F, prev_front])
    # F = grouped.get_group(g)[objectives].values
    hv = calc_hv(ref_pt, merged_F, normalized=True)
    hv_list.append(hv)
    # 必须：每轮更新历史前沿
    # front = NonDominatedSorting().do(merged_F, only_non_dominated_front=True)
    # prev_front = merged_F[front, :]


print(f'{target_dir} hv: {hv_list[-1]}')

# 绘制 HV 收敛曲线
plt.figure(figsize=(8, 5))
plt.plot(gens, hv_list, marker='o', label='Hypervolume')
plt.xlabel('Generation')
plt.ylabel('Normalized Hypervolume')
plt.title('HV Convergence Curve')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_hv_curve.png'))
plt.close()