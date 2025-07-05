# Mamba_DistillNAS

Using **Once-for-All Progressive Shrinking**

------

##  Update Logs

2025/4/11 update SuperNet(VSSD) and Search Space

2025/4/15 update NAS Search(optimize d1 and params)

2025/6/27 update SuperNet Fine-tuning(Training strategy, Only sample one layer per stage, ...)

2025/7/5 update SuperNet Fine-tuning(Training strategy, Open Depth space, Code encode-decode, Teacher model fixed)

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
configs/
├── supernet_train_kitti_0_maxnet.txt      # Step 0: max network
├── supernet_train_kitti_1_mlp_1.txt       # Step 1: relax MLP_RATIO=[3.0, 3.5, 4.0]
├── supernet_train_kitti_2_mlp_2.txt       # Step 1: relax MLP_RATIO=[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]
├── supernet_train_kitti_3_state_1.txt     # Step 2: relax D_STATE=[48, 64]
├── supernet_train_kitti_4_state_2.txt     # Step 3: relax D_STATE=[16, 32, 48, 64]
├── supernet_train_kitti_5_ssdExpand_1.txt # Step 4: relax SSD_EXPAND=[2, 3, 4]
├── supernet_train_kitti_6_ssdExpand_2.txt # Step 5: relax SSD_EXPAND=[0.5, 1, 2, 3, 4]
```

------

## TODO
- [x] Search Space (VSSD)
    - [x] add choose (MLP_RATIO, D_STATE, SSD_EXPAND)
- [x] Pretrain Training strategy on IN-1k
- [x] Loading pretrained weight (encoder, decoder, ema)
- [x] SuperNet Fine-tuning
    - [x] encoder-decoder lr, warmup lr, optimizer
    - [ ] ema, Gradient Accumulation, amp
    - [x] Only sample one layer per stage
    - [x] One batch train multiple SubNet
    - [ ] Update SuperNet by Fedavg
    - [ ] Parallel train SubNet
    - [ ] Knowledge Distillation (bug)
- [x] NAS Search (pymoo, param_cal)
    - [ ] add depth choose or only one layer
- [ ] Retrain

------

## pipline
1. pretrain supernet on ImageNet-1k (only Once);
2. fine-tune supernet on object datasets(e.g. KITTI);
3. NAS search by evolution;
4. retrain the searched network;