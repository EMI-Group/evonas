import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 11              # 全局默认字号
mpl.rcParams["axes.titlesize"] = 14         # 标题
mpl.rcParams["axes.labelsize"] = 13         # x/y 轴标签
mpl.rcParams["xtick.labelsize"] = 11        # x 轴刻度
mpl.rcParams["ytick.labelsize"] = 11        # y 轴刻度
mpl.rcParams["legend.fontsize"] = 11        # 图例
mpl.rcParams["figure.titlesize"] = 15       # suptitle（整张图大标题）

def make_demo_csv(path="demo.csv"):
    # 列顺序：Model, Metric, FLOPs, Params, FPS
    demo = pd.DataFrame({
        "Model":  ["EvoXNAS-N1", "EvoXNAS-N2", "EvoXNAS-N3", "NeWCRFs_Swin-L",   "IEBins_Swin-T",   "iDisc_Swin-T", "AdaBins_E-B5"],
        "Abs Rel":[0.095,         0.089,         0.085,         0.095,     0.108,     0.109,    0.103],   # 例如 AbsRel（越小越好）
        "MACs":   [20.6,          27.1,           33.9,         281.1,      378.2,      218.8,  187.77],    # 例如 G MACs
        "Params": [18.6,          24.1,           30.7,          270,      90.7,        39,     78.3],    # 例如 M params
        "FPS":    [53.0,          37.7,           31.9,          18.6,        19.9,     12.6,   25],
    })
    demo.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] demo csv saved to: {os.path.abspath(path)}")


def scatter_tradeoff(
    csv_path,
    out_path="scatter_tradeoff.png",
    metric_name="Abs Rel",
    flops_name="MACs",
    params_name="Params",
    fps_name="FPS",
    model_name="Model",
    x_log=False,          # FLOPs 跨度大时可设 True
    y_reverse=True,       # AbsRel 等越小越好时，建议 True
    annotate=True         # 标注模型名
):
    df = pd.read_csv(csv_path)
    need_cols = [model_name, metric_name, flops_name, params_name, fps_name]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"CSV missing column: {c}, got columns={list(df.columns)}")

    x = df[flops_name].astype(float).to_numpy()
    y = df[metric_name].astype(float).to_numpy()
    p = df[params_name].astype(float).to_numpy()
    fps = df[fps_name].astype(float).to_numpy()
    labels = df[model_name].astype(str).to_numpy()

    # Params -> 点面积：用 sqrt 缩放更接近“视觉线性”
    p_norm = (p - p.min()) / (p.max() - p.min() + 1e-12)
    sizes = 80 + 1200 * np.sqrt(p_norm)   # 你可以按审稿版式再调

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)

    sc = ax.scatter(
        x, y,
        s=sizes,
        c=fps,
        alpha=0.85,
    )

    # 颜色条
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(fps_name)

    ax.set_xlabel(r"$\mathbf{\leftarrow}$ " + flops_name + " (G)")
    ax.set_ylabel(metric_name + r" $\mathbf{\rightarrow}$")

    if x_log:
        ax.set_xscale("log")

    if y_reverse:
        ax.invert_yaxis()

    # 标注模型名
    if annotate:
        # 轻微偏移，避免文字压住点
        for xi, yi, name, pi, s in zip(x, y, labels, p, sizes):
            r_pt = np.sqrt(float(s) / np.pi)  # 圆点半径（points）
            dy = r_pt + 0  # 自适应上移距离（points）
            dx = 0
            if name == 'NeWCRFs_Swin-L':
                dy = -2*dy
            if name == 'BinsFormer_Swin-L':
                dx = 3*r_pt
                dy = 0

            text = f"{name}\n({pi:.1f}M)"
            ax.annotate(
                text, (xi, yi),
                xytext=(dx, dy), textcoords="offset points",
                ha="center", va="bottom",
                multialignment="center",
                fontsize=11, fontfamily="Times New Roman"
            )
    ax.margins(x=0.13, y=0.13)  # x/y 轴都留 13% 空白
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    # ax.set_title("Accuracy vs MACs (size=Params, color=FPS)")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[OK] figure saved to: {os.path.abspath(out_path)}")
    plt.show()


if __name__ == "__main__":
    # make_demo_csv("demo.csv")
    scatter_tradeoff(
        csv_path="demo.csv",
        out_path="tradeoff_demo.png",
        x_log=False,        # 需要可改 True
        y_reverse=True,     # AbsRel 这类指标建议 True
        annotate=True
    )
