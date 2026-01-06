# EvoXNAS

Using **Once-for-All Progressive Shrinking**

------

## Requirements

```
conda create -n EvoXNAS python=3.10
conda activate EvoXNAS

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia

# download from https://github.com/state-spaces/mamba/releases
pip install [/**/mamba_ssm-2.2.4+cu11torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl]

pip install timm==0.4.12
pip install fvcore
pip install tensorboardX
pip install mmcv==2.2.0

pip install Cython==3.0.12
# donwload from https://github.com/EdwardChasel/Spatial-Mamba
cd kernels/selective_scan && pip install .

pip install numpy==2.0.1
pip install scipy==1.15.2

pip install pymoo==0.6.1.3
pip install ptflops==0.7.4
pip install pandas

# for object detection task
pip install mmdet==3.3.0

# for semantic segmentation task
pip install mmsegmentation==1.2.2
pip install ftfy==6.3.1
```

##  Update Logs

2025/4/11 update SuperNet(Mamba) and Search Space

2025/4/15 update NAS Search(Optimize d1 and params)

2025/6/27 update SuperNet Fine-tuning(Training strategy, Only sample one layer per stage, ...)

2025/7/5 update SuperNet Fine-tuning(Training strategy, Open Depth space, Code encode-decode, Teacher model fixed)

2025/7/7 update SuperNet Fine-tuning(Add CAP and DFM of scaleKD)

2025/7/8 update SuperNet Fine-tuning(Change supernet depth to [8,8,8,8], Pretrained weights and sample code)

2025/7/9 update SuperNet Fine-tuning(final)(Depth 01bit code, Script of whole run)

2025/7/10 update SuperNet Fine-tuning(final)(Recover supernet depth to [2,4,8,4], Add run_script on nyu, Solve some bugs)

2025/7/15 update NAS Search(Optimize Abs_rel, Latency and MACs, Special corssover and mutation for mixed code, Check any stage is all zeros, Accelerate the verification process)

2025/7/17 update NAS Search(Subprocess to test latency, Sovle GPU memory bug v1 by `save_history=False`)

2025/7/18 update NAS Search(Fixed GPU memory bug v2 by `group=None` , Accelerate evaluation by `persistent_workers=True`, Centralized processing of latency testing)

2025/7/20 update Retrain(Only train by supernet)

2025/7/23 update tools(Calculate HV, Visualized evolution curve)

2025/7/31 update NAS Search(Parallel model validation, Mutil-GPUs latency test, Fix process hanging issue)

2025/8/13 update Retrain(Support variable width)

2025/8/14 update Retrain(Support mixed precision training)

2025/8/19 update Retrain(Support other encoders for compaper, e.g. ConvNeXt, EfficientNet, SwinTransformer, MambaVision, MLLA)

2025/8/21 update SuperNet Fine-tuning(Support mixed precision training)

2025/8/25 update SuperNet Fine-tuning(Support init weights by remapping pretrained weights)

2025/9/2 update NAS Search(Change search space code order)

2025/9/26 update Retrain(Support new decoders e.g. NewCRFs,iDisc,VMamba and solved weight BUG)

2025/9/30 update Retrain(Support neck module choices e.g. PPM, SP and None)

2025/10/24 update Object Detection support (supernet training, NAS search, retraining all compatible)

2025/10/29 update Semantic Segmentation support (supernet training, NAS search, retraining all compatible)

2025/11/1 update SuperNet Fine-tuning (support training supernet by architecture pool)

2025/11/7 update Semantic Segmentation tearcher network (r101-upernet replaced DepthAnything)

2025/11/9 update SuperNet Training (fixed bug of architecture pool)

2025/12/15 update SuperNet Training (for Detection and Segment, e.g. SyncBN, teacher model, hyper param)

2025/12/22 update SuperNet Training (for Detection and Segment, mamba-based FPN)

2025/12/23 update SuperNet Training (update tearcher model for Detection ———— mask2former, mIoU=83.65)

2026/1/6 update SuperNet Training (Segmentation training by iter)

------

##  Search Space

```
MLP_RATIO:   [0.5, 1.0, 2.0, 3.0, 3.5, 4.0]
D_STATE:     [16, 32, 48, 64] 
SSD_EXPAND:  [0.5, 1, 2, 3, 4]
```

------

##  Train Piplines

```
configs/prog_shrink
├── supernet_train_kitti_0_maxnet.txt   # Step 0: max network
├── supernet_train_kitti_1_state_1.txt  # Step 1: relax D_STATE=[48, 64]
├── supernet_train_kitti_2_state_2.txt  # Step 2: relax D_STATE=[16, 32, 48, 64]
├── supernet_train_kitti_3_mlp_1.txt    # Step 3: relax MLP_RATIO=[3.0, 3.5, 4.0]
├── supernet_train_kitti_4_mlp_2.txt    # Step 4: relax MLP_RATIO=[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]
├── supernet_train_kitti_5_ssd_1.txt    # Step 5: relax SSD_EXPAND=[2, 3, 4]
├── supernet_train_kitti_6_ssd_2.txt    # Step 6: relax SSD_EXPAND=[0.5, 1, 2, 3, 4]
├── supernet_train_kitti_7_depth.txt    # Step 7: relax free Depth
```

#### start run

```
sh whole_run_kitti.sh
```

------

## TODO
- [x] Search Space (VSSD)
    - [x] add choose (MLP_RATIO, D_STATE, SSD_EXPAND)
- [x] Pretrain Training strategy on IN-1k
- [x] Loading pretrained weight (encoder, decoder, ema)
- [x] SuperNet Fine-tuning
    - [x] encoder-decoder lr, warmup lr, optimizer
    - [ ] ~~ema, Gradient Accumulation, amp~~
    - [x] ~~Only sample one layer per stage~~
    - [x] ~~One batch train multiple SubNet~~
    - [ ] ~~Update SuperNet by Fedavg~~
    - [ ] ~~Parallel train SubNet~~
    - [x] Knowledge Distillation (bug fixed)
    - [x] add new kd strategy (part1 and part2 of scaleKD)
    - [x] ~~Change supernet depth to [8,8,8,8] and pretrained weights~~
    - [x] add mixed precision (AMP)
- [x] NAS Search (pymoo, param_cal)
    - [x] add depth choose ~~or only one layer~~
    - [ ] ~~recompute the Params and FLOP of subnet~~
    - [x] optimization extra objectives (Latency and MAC)
    - [x] crossover and mutation separate mixed code
    - [x] open batch size > 1 for val
    - [x] use subprocess to test latency
    - [x] save_history set False to solve abnormal increase in GPU memory (add EvolutionLogger to replace)
    - [x] parallel evaluation(25% speed up or 60% speed up without persistent_workers)
    - [x] add GIFs images that display the evolutionary trajectory
    - [x] add method NSGA-III
    - [x] mutil-GPU latency test(the number of GPUs speed up)
- [x] Retrain
    - [x] build selected network by searched arch code
    - [ ] ~~load and map weight from supernet~~
    - [x] fixed dis_modules_s4 weight loading(not helpful)
    - [x] add mixed precision (AMP)
    - [x] add other encoders (CNN,ViT,Mamba)
- [x] Extended to Object Detection task
- [x] Extended to Semantic Segmentation task
------

## Pipline
1. pretrain supernet on ImageNet-1k (only Once);
2. fine-tune supernet on object datasets(e.g. KITTI);
3. NAS search by evolution;
4. retrain the searched network;