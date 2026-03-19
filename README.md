<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/evox_brand_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/evox_brand_dark.svg">
    <img alt="EvoX Logo" height="128" width="500px" src="./assets/evox_brand_dark.svg">
  </picture>
  </a>
  <br>
</h1>
</div>

---

## Introduction

Modern computer vision tasks require a delicate balance between predictive accuracy and real-time efficiency, but the substantial inference cost of large vision models (LVMs) notably restricts their deployment on resource-constrained edge devices. EvoNAS addresses this by introducing a highly efficient, multi-objective evolutionary architecture search framework. To overcome the severe representation collapse and ranking inconsistency typical in conventional weight-sharing paradigms, EvoNAS utilizes a hybrid Vision State Space and Vision Transformer (VSS-ViT) supernet optimized via the Progressive Supernet Training (PST) strategy. This is further stabilized by a novel Cross-Architecture Dual-Domain Knowledge Distillation (CA-DDKD) approach, which aligns features in both spatial and frequency domains using DCT constraints to lock in high-frequency geometric priors. Evaluated through a hardware-isolated Distributed Multi-Model Parallel Evaluation (DMMPE) engine that eliminates computational noise, the resulting EvoNets establish Pareto-optimal trade-offs, demonstrating robust generalizability from 2D dense prediction to high-fidelity 3D rendering tasks like 3D Gaussian Splatting.

---

## Key Features

- 🔬 **Hybrid VSS-ViT Search Space**: Combines linear-time Vision State Space (VSS) modules for local geometric feature capture with Vision Transformer (ViT) modules for global semantic reasoning.
- ⚡ **Progressive Supernet Training (PST)**: A curriculum-learning strategy that expands from maximum-capacity configurations to compact variants, ensuring a smooth fitness landscape and stable supernet convergence.
- 🧠 **CA-DDKD Strategy**: Cross-Architecture Dual-Domain Knowledge Distillation using DCT constraints to mitigate representation collapse and preserve high-frequency geometric priors across both spatial and frequency domains.
- 🚀 **DMMPE Framework**: A hardware-isolated distributed evaluation engine with GPU resource pooling and asynchronous scheduling, eliminating latency jitter during parallel architecture evaluation.
- 🌐 **Universal Geometric Transferability**: Generalizes across COCO, ADE20K, KITTI/NYU v2, and 3D Gaussian Splatting without task-specific design changes.

---

## Results

> 🔬 This project accompanies a paper currently under review. Quantitative results and pre-trained model weights will be released upon acceptance.

---

## Installation

> [!WARNING]
> Mamba SSM requires a prebuilt CUDA wheel that must match your exact Python, CUDA, and PyTorch versions. Download the appropriate `.whl` from the [Mamba releases page](https://github.com/state-spaces/mamba/releases) before proceeding.

```bash
# 1. Create environment
conda create -n EvoNAS python=3.10 -y && conda activate EvoNAS

# 2. Install PyTorch (CUDA 11.8)
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Install Mamba SSM (replace with your downloaded wheel path)
pip install /path/to/mamba_ssm-2.2.4+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 4. Install Spatial-Mamba kernel (source: https://github.com/EdwardChasel/Spatial-Mamba)
cd kernels/selective_scan && pip install .
cd kernels/dwconv2d && pip install .

# 5. Install remaining dependencies
pip install timm==0.4.12 fvcore tensorboardX mmcv==2.2.0 \
            numpy==2.0.1 scipy==1.15.2 pymoo==0.6.1.3 \
            ptflops==0.7.4 pandas Cython==3.0.12

# 6. Task-specific toolkits (install as needed)
pip install mmdet==3.3.0           # Object Detection
pip install mmsegmentation==1.2.2 ftfy==6.3.1  # Semantic Segmentation
```

---

## Data Preparation

Download the datasets and organize them as follows. Pre-defined train/test split files are provided in `data_splits/`.

```
data/
├── NYU_Depth_V2/
│   ├── sync/              ← training RGB-D frames
│   └── test/              ← test images
├── KITTI/
│   ├── raw/               ← raw KITTI sequences
│   └── depth/             ← ground-truth depth maps
├── coco/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
└── ADEChallengeData2016/  ← ADE20K
    ├── images/
    └── annotations/
```

> [!NOTE]
> Update `--data_path` and `--gt_path` in the relevant config files under `configs/` to point to your local data directories before running any scripts.

---

## Quickstart

> [!NOTE]
> Stage 1 requires the ImageNet-1k pretrained supernet weights (`vssd_supernet_imagenet_1k.pth`). **Download link coming soon.** Place the file in the project root before fine-tuning.

EvoNAS follows a four-stage pipeline: **(1)** pretrain the supernet on ImageNet-1k, **(2)** fine-tune it on the target dataset using the PST strategy, **(3)** run the multi-objective evolutionary search, and **(4)** retrain the discovered subnet.

**KITTI (example)**

```bash
sh scripts/whole_run_kitti.sh                                   # PST fine-tuning (all 8 steps)
python MambaDepthNAS/search.py configs/search/search_kitti.txt  # evolutionary search
python MambaDepthNAS/retrain.py configs/retrain_kitti.txt       # retrain
```

<details>
<summary>Commands for all supported tasks</summary>

**NYU Depth v2**
```bash
sh scripts/whole_run_nyu.sh
python MambaDepthNAS/search.py configs/search/search_nyu.txt
python MambaDepthNAS/retrain.py configs/retrain_nyu.txt
```

**COCO Object Detection**
```bash
sh scripts/detection/supernet_steptrain.sh
python DetectionNAS/search.py DetectionNAS/configs/01_search/search_coco.txt
python DetectionNAS/retrain.py DetectionNAS/configs/02_retrain/retrain_supernet_base.txt
```

**ADE20K Semantic Segmentation**
```bash
sh scripts/segment/supernet_steptrain_ade20k.sh
python SegmentNAS/search.py SegmentNAS/configs/01_search/search_ade20k.txt
python SegmentNAS/retrain.py SegmentNAS/configs/02_retrain/retrain_supernet_base.txt
```

</details>

---

## Project Structure

```
EvoNAS/
├── MambaDepthNAS/          # Monocular depth estimation module
│   ├── train.py            #   PST supernet fine-tuning
│   ├── search.py           #   NSGA-II/III evolutionary search
│   ├── retrain.py          #   Subnet retraining
│   ├── networks/           #   VSS-ViT encoder + decoder variants
│   └── distillation/       #   CA-DDKD (spatial + frequency KD)
├── DetectionNAS/           # Object detection (MMDetection)
├── SegmentNAS/             # Semantic segmentation (MMSegmentation)
├── configs/                # PST, search, and retrain configs
├── scripts/                # End-to-end shell scripts per dataset
├── data_splits/            # Official train/test split file lists
└── tools/                  # Visualization (evolution curve, HV, det/seg)
```

---

## Acknowledgements

EvoNAS builds on a strong ecosystem of open-source tools. We are grateful to the teams behind [PyTorch](https://pytorch.org/), [Mamba SSM](https://github.com/state-spaces/mamba), [Spatial-Mamba](https://github.com/EdwardChasel/Spatial-Mamba), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [pymoo](https://github.com/anyoptimization/pymoo), and [timm](https://github.com/huggingface/pytorch-image-models) for making this work possible.

---

## License

EvoNAS 遵循 **GNU 通用公共许可证 3.0 (GPL-3.0)** 进行授权。完整的条款和条件请参阅 [LICENSE](./LICENSE) 文件。

---

<div align="center">
<sub>⭐ If you find this project helpful, please consider giving it a star.</sub>
</div>
