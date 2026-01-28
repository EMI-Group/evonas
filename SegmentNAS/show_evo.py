import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

GIF = False
HV = False

target_dir = '/root/workspace/yzh/code/paper3_code/runs/ade20k/01_search/run_search'
# Ensure output directory exists
output_dir = os.path.join(target_dir, 'show_evo_pics/')
os.makedirs(output_dir, exist_ok=True)

# Load external CSV file
input_csv = os.path.join(target_dir, 'pop.csv')
df = pd.read_csv(input_csv)
max_gen = df['gen'].max()
# max_gen = 100

# Define generations to plot (e.g., every 5 generations from 1 to 50)
gen_steps = list(range(1, max_gen + 1, 14))
if gen_steps[-1] != max_gen:
    gen_steps.append(max_gen)

cmap = plt.cm.viridis_r
colors = [cmap(i / (len(gen_steps) - 1)) for i in range(len(gen_steps))]

# Plot 1: Latency vs loss
plt.figure(figsize=(8, 6))
for i, gen in enumerate(gen_steps):
    data = df[df['gen'] == gen]
    plt.scatter(data['latency'], data['loss'], label=f'Gen {gen}', color=colors[i])
plt.xlabel('Latency')
plt.ylabel('loss')
plt.title('Latency vs loss (Multiple Generations)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'latency_vs_loss_multi_gen.png'))
plt.close()

# Plot 2: MACs vs loss
plt.figure(figsize=(8, 6))
for i, gen in enumerate(gen_steps):
    data = df[df['gen'] == gen]
    plt.scatter(data['macs'], data['loss'], label=f'Gen {gen}', color=colors[i])
plt.xlabel('MACs')
plt.ylabel('loss')
plt.title('MACs vs loss (Multiple Generations)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'macs_vs_loss_multi_gen.png'))
plt.close()


# Plot 3: MACs vs Latency
plt.figure(figsize=(8, 6))
for i, gen in enumerate(gen_steps):
    data = df[df['gen'] == gen]
    plt.scatter(data['macs'], data['latency'], label=f'Gen {gen}', color=colors[i])
plt.xlabel('MACs')
plt.ylabel('Latency')
plt.title('MACs vs Latency (Multiple Generations)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'macs_vs_latency_multi_gen.png'))
plt.close()

# Plot 4: 3D Plot - loss vs Latency vs MACs
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i, gen in enumerate(gen_steps):
    data = df[df['gen'] == gen]
    ax.scatter(data['latency'], data['macs'], data['loss'], label=f'Gen {gen}', color=colors[i])
ax.set_xlabel('Latency')
ax.set_ylabel('MACs')
ax.set_zlabel('loss')
ax.set_title('3D Plot: loss vs Latency vs MACs (Multiple Generations)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3d_loss_latency_macs_multi_gen.png'))
plt.close()



if GIF:
    ### gif 1
    # Generate animation of 3D plot across generations with fixed axis limits
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Compute global axis limits
    x_min, x_max = df['latency'].min(), df['latency'].max()
    y_min, y_max = df['macs'].min(), df['macs'].max()
    z_min, z_max = df['loss'].min(), df['loss'].max()

    def update(frame):
        ax.clear()
        current_gen = frame + 1
        data = df[df['gen'] == current_gen]
        ax.scatter(data['latency'], data['macs'], data['loss'], color='blue')
        ax.set_xlabel('Latency')
        ax.set_ylabel('MACs')
        ax.set_zlabel('loss')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(f'3D Evolution (loss): Gen {current_gen}')


    ani = animation.FuncAnimation(fig, update, frames=max_gen, interval=200)
    ani.save(os.path.join(output_dir, '3d_evolution.gif'), writer='pillow')
    plt.close()

    ### gif 2
    # 2D Animation: Latency vs AbsRel across generations
    fig2d, ax2d = plt.subplots(figsize=(8, 6))

    # Global axis limits
    x_min, x_max = df['latency'].min(), df['latency'].max()
    y_min, y_max = df['loss'].min(), df['loss'].max()

    def update_2d(frame):
        ax2d.clear()
        current_gen = frame + 1
        data = df[df['gen'] == current_gen]
        ax2d.scatter(data['latency'], data['loss'], color='blue')
        ax2d.set_xlim(x_min, x_max)
        ax2d.set_ylim(y_min, y_max)
        ax2d.set_xlabel('Latency')
        ax2d.set_ylabel('loss')
        ax2d.set_title(f'Latency vs loss - Generation {current_gen}')
        ax2d.grid(True)

    max_gen = df['gen'].max()
    ani_2d = animation.FuncAnimation(fig2d, update_2d, frames=max_gen, interval=200)
    ani_2d.save(os.path.join(output_dir, 'latency_vs_loss_evolution.gif'), writer='pillow')
    plt.close()

### singe object
# Grouped by generation
grouped = df.groupby('gen')
gens = sorted(grouped.groups.keys())

# loss：max & mean
loss_max = [grouped.get_group(g)['loss'].max() for g in gens]
loss_mean = [grouped.get_group(g)['loss'].mean() for g in gens]

latency_min = [grouped.get_group(g)['latency'].min() for g in gens]
latency_mean = [grouped.get_group(g)['latency'].mean() for g in gens]

macs_min = [grouped.get_group(g)['macs'].min() for g in gens]
macs_mean = [grouped.get_group(g)['macs'].mean() for g in gens]

# Plot loss: max + mean
plt.figure(figsize=(8, 5))
plt.plot(gens, loss_max, marker='o', label='Max loss')
plt.plot(gens, loss_mean, marker='x', linestyle='--', label='Mean loss')
plt.xlabel('Generation')
plt.ylabel('loss')
plt.title('Convergence - loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_loss_max_mean.png'))
plt.close()

# Plot Latency: min + mean
plt.figure(figsize=(8, 5))
plt.plot(gens, latency_min, marker='s', label='Min Latency', color='orange')
plt.plot(gens, latency_mean, marker='x', linestyle='--', label='Mean Latency', color='red')
plt.xlabel('Generation')
plt.ylabel('Latency')
plt.title('Convergence - Latency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_latency_min_mean.png'))
plt.close()

# Plot MACs: min + mean
plt.figure(figsize=(8, 5))
plt.plot(gens, macs_min, marker='^', label='Min MACs', color='green')
plt.plot(gens, macs_mean, marker='x', linestyle='--', label='Mean MACs', color='blue')
plt.xlabel('Generation')
plt.ylabel('MACs')
plt.title('Convergence - MACs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'convergence_macs_min_mean.png'))
plt.close()


if HV:
    from pymoo.indicators.hv import HV
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    import numpy as np

    # 把“越大越好”的 loss 变成“越小越好”的优化目标
    # 假设 df['loss'] 在 [0, 1] 或 [0, 100] 都没关系，乘个 -1 即可
    df['loss'] = -df['loss']

    def calc_hv(ref_pt, F, normalized=True):
        # 非支配前沿
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        nd_F = F[front, :]
        # 参考点略比最差情况更差一点
        ref_point = 1.01 * np.array(ref_pt)
        hv = HV(ref_point=ref_point)(nd_F)
        if normalized:
            hv = hv / np.prod(ref_point)
        return hv

    # 三个目标：注意这里用的是 loss_obj（负号）
    objectives = ['loss', 'latency', 'macs']

    # 参考点使用所有数据的最大值（loss_obj 越大越差，对应 loss 越小）
    ref_pt = [df[obj].max() for obj in objectives]

    grouped = df.groupby('gen')
    gens = sorted(grouped.groups.keys())
    hv_list = []
    prev_front = None

    for g in gens:
        curr_F = grouped.get_group(g)[objectives].values
        front_i_idx = NonDominatedSorting().do(curr_F, only_non_dominated_front=True)
        front_i = curr_F[front_i_idx, :]

        if prev_front is None:
            merged_F = front_i
        else:
            merged_F = np.vstack([front_i, prev_front])

        hv = calc_hv(ref_pt, merged_F, normalized=True)
        hv_list.append(hv)

        # 更新历史前沿
        front = NonDominatedSorting().do(merged_F, only_non_dominated_front=True)
        prev_front = merged_F[front, :]

    # 绘制 HV 收敛曲线
    plt.figure(figsize=(8, 5))
    plt.plot(gens, hv_list, marker='o', label='Hypervolume')
    plt.xlabel('Generation')
    plt.ylabel('Normalized Hypervolume')
    plt.title('HV Convergence Curve (loss, Latency, MACs)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_hv_curve_loss_latency_macs.png'))
    plt.close()