import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

# Ensure output directory exists
output_dir = '/data/code_yzh/DistillNAS/runs/search_kitti/run0_search_kitti/show_evo_pics/'
os.makedirs(output_dir, exist_ok=True)

# Load external CSV file
input_csv = '/data/code_yzh/DistillNAS/runs/search_kitti/run0_search_kitti/pop.csv'
df = pd.read_csv(input_csv)

# Define generations to plot (e.g., every 5 generations from 1 to 50)
gen_steps = range(1, 51, 7)
cmap = plt.cm.viridis_r
colors = [cmap(i / (len(gen_steps) - 1)) for i in range(len(gen_steps))]

# Plot 1: Latency vs AbsRel
plt.figure(figsize=(8, 6))
for i, gen in enumerate(gen_steps):
    data = df[df['gen'] == gen]
    plt.scatter(data['latency'], data['abs_rel'], label=f'Gen {gen}', color=colors[i])
plt.xlabel('Latency')
plt.ylabel('Abs Relative Error')
plt.title('Latency vs AbsRel (Multiple Generations)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'latency_vs_absrel_multi_gen.png'))
plt.close()

# Plot 2: MACs vs AbsRel
plt.figure(figsize=(8, 6))
for i, gen in enumerate(gen_steps):
    data = df[df['gen'] == gen]
    plt.scatter(data['macs'], data['abs_rel'], label=f'Gen {gen}', color=colors[i])
plt.xlabel('MACs')
plt.ylabel('Abs Relative Error')
plt.title('MACs vs AbsRel (Multiple Generations)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'macs_vs_absrel_multi_gen.png'))
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

# Plot 4: 3D Plot - AbsRel vs Latency vs MACs
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i, gen in enumerate(gen_steps):
    data = df[df['gen'] == gen]
    ax.scatter(data['latency'], data['macs'], data['abs_rel'], label=f'Gen {gen}', color=colors[i])
ax.set_xlabel('Latency')
ax.set_ylabel('MACs')
ax.set_zlabel('Abs Relative Error')
ax.set_title('3D Plot: AbsRel vs Latency vs MACs (Multiple Generations)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '3d_absrel_latency_macs_multi_gen.png'))
plt.close()

### gif 1
# Generate animation of 3D plot across generations with fixed axis limits
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Compute global axis limits
x_min, x_max = df['latency'].min(), df['latency'].max()
y_min, y_max = df['macs'].min(), df['macs'].max()
z_min, z_max = df['abs_rel'].min(), df['abs_rel'].max()

def update(frame):
    ax.clear()
    current_gen = frame + 1
    data = df[df['gen'] == current_gen]
    ax.scatter(data['latency'], data['macs'], data['abs_rel'], color='blue')
    ax.set_xlabel('Latency')
    ax.set_ylabel('MACs')
    ax.set_zlabel('Abs Relative Error')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_title(f'3D Evolution: Gen {current_gen}')

ani = animation.FuncAnimation(fig, update, frames=50, interval=300)
ani.save(os.path.join(output_dir, '3d_evolution.gif'), writer='pillow')
plt.close()

### gif 2
# 2D Animation: Latency vs AbsRel across generations
fig2d, ax2d = plt.subplots(figsize=(8, 6))

# Global axis limits
x_min, x_max = df['latency'].min(), df['latency'].max()
y_min, y_max = df['abs_rel'].min(), df['abs_rel'].max()

def update_2d(frame):
    ax2d.clear()
    current_gen = frame + 1
    data = df[df['gen'] == current_gen]
    ax2d.scatter(data['latency'], data['abs_rel'], color='blue')
    ax2d.set_xlim(x_min, x_max)
    ax2d.set_ylim(y_min, y_max)
    ax2d.set_xlabel('Latency')
    ax2d.set_ylabel('Abs Relative Error')
    ax2d.set_title(f'Latency vs AbsRel - Generation {current_gen}')
    ax2d.grid(True)

max_gen = df['gen'].max()
ani_2d = animation.FuncAnimation(fig2d, update_2d, frames=max_gen, interval=300)
ani_2d.save(os.path.join(output_dir, 'latency_vs_absrel_evolution.gif'), writer='pillow')
plt.close()