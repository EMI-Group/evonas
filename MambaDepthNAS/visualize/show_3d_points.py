import pandas as pd
import numpy as np
import os
import argparse

# === Configuration ===
# target_dir = '/data/code_yzh/DistillNAS/runs/search_nyu/run0_search_nyu'
parser = argparse.ArgumentParser(description="Generate PLY from CSV")
parser.add_argument('--dir', type=str, help='Path to target directory')
parser.add_argument('--all', action='store_true', help='')
args = parser.parse_args()
target_dir = args.dir
ALL = args.all  # Set to True to use all data points, False for final generation only

input_csv = os.path.join(target_dir, 'pop.csv')
output_dir = os.path.join(target_dir, 'show_evo_pics')
os.makedirs(output_dir, exist_ok=True)

# Extract last part of target_dir for naming
target_name = os.path.basename(target_dir.rstrip('/'))
suffix = 'all_nds' if ALL else 'final_gen'
ply_output_path = os.path.join(output_dir, f'{target_name}_{suffix}.ply')

# === Load Data ===
df = pd.read_csv(input_csv)

# === Objectives to include in PLY ===
objectives = ['abs_rel', 'latency', 'macs']

# === Normalization ===
max_vals = {'abs_rel':0.3, 'latency':50.0, 'macs':50.0}
df_normalized = df.copy()
for obj in objectives:
    df_normalized[obj] = df[obj] / max_vals[obj]

# === Select Data Points ===


if ALL:
    selected_df = df_normalized
else:
    max_gen = df['gen'].max()
    selected_df = df_normalized[df_normalized['gen'] == max_gen]

points = selected_df[objectives].values.astype(np.float32)

if ALL:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    front = NonDominatedSorting().do(points, only_non_dominated_front=True)
    points = points[front, :]

# === Write PLY File ===
with open(ply_output_path, 'w') as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {points.shape[0]}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for point in points:
        f.write(f"{point[0]} {point[1]} {point[2]}\n")

print(f"PLY file saved to: {ply_output_path}")
