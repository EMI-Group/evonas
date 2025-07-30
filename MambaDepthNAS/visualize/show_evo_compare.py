import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm

# ------------------------------------------------------------------
# 1. 路径准备
# ------------------------------------------------------------------
target_dir_1 = '/data/code_yzh/DistillNAS/runs/search_nyu/run4_search_nyu_p50e200'
target_dir_2 = '/data/code_yzh/DistillNAS/runs/search_nyu/run5_search_nyu_pbit025'

out_dir = os.path.join(target_dir_1, 'show_evo_pics')
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------
# 2. 读取数据
# ------------------------------------------------------------------
df1 = pd.read_csv(os.path.join(target_dir_1, 'pop.csv'))
df2 = pd.read_csv(os.path.join(target_dir_2, 'pop.csv'))

# 统一坐标范围
xmin = min(df1['latency'].min(), df2['latency'].min())
xmax = max(df1['latency'].max(), df2['latency'].max())
ymin = 0.08                                 # AbsRel >= 0
ymax = max(df1['abs_rel'].max(), df2['abs_rel'].max())

# 给两个种群分别建代数 → 颜色 映射
def color_by_gen(df, cmap):
    gens = sorted(df['gen'].unique())
    color_map = {g: cmap(i / (len(gens) - 1)) for i, g in enumerate(gens)}
    return df['gen'].map(color_map)

# ------------------------------------------------------------------
# 3. 绘制
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Pop-1
axes[0].scatter(df1['latency'], df1['abs_rel'],
                s=10, c=color_by_gen(df1, cm.viridis_r), marker='o')
axes[0].set_title('Pop-1  (p_bit=0.50)')
axes[0].set_xlabel('Latency'); axes[0].set_ylabel('AbsRel')
axes[0].grid(alpha=.3)

# Pop-2
axes[1].scatter(df2['latency'], df2['abs_rel'],
                s=10, c=color_by_gen(df2, cm.plasma_r), marker='o')
axes[1].set_title('Pop-2  (p_bit=0.25)')
axes[1].set_xlabel('Latency')
axes[1].grid(alpha=.3)

for ax in axes:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

plt.suptitle('Latency vs AbsRel – Two Populations (matching axes)', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'lat_vs_absrel_side_by_side.png'), dpi=300)
plt.close()
