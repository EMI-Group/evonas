# Mamba_DistillNAS

Using **Once-for-All Progressive Shrinking**。

------

##  Update Logs

2025/4/11 update SuperNet(VSSD) and Search Space

------

##  Search Space

```
MLP_RATIO:   [0.5, 1.0, 2.0, 3.0, 3.5, 4.0]
D_STATE:     [16, 32, 48, 64] 
SSD_EXPAND:  [0.5, 1, 2, 3, 4]
```

------

##  Train Piplines

configs/
├── supernet_train_kitti_0_maxnet.txt      # Step 0: 最大模型预训练
├── supernet_train_kitti_1_mlp_1.txt       # Step 1: 解锁 MLP_RATIO=[3.0, 3.5, 4.0]
├── supernet_train_kitti_2_mlp_2.txt       # Step 1: 解锁 MLP_RATIO=[0.5, 1.0, 2.0, 3.0, 3.5, 4.0]
├── supernet_train_kitti_3_state_1.txt     # Step 2: 解锁 D_STATE=[48, 64]
├── supernet_train_kitti_4_state_2.txt     # Step 3: 解锁 D_STATE=[16, 32, 48, 64]
├── supernet_train_kitti_6_ssdExpand_1.txt # Step 4: 解锁 SSD_EXPAND=[2, 3, 4]
├── supernet_train_kitti_6_ssdExpand_2.txt # Step 5: 解锁 SSD_EXPAND=[0.5, 1, 2, 3, 4]

------

## TODO
- [x] Search Space (VSSD)
    - [x] add choose (MLP_RATIO, D_STATE, SSD_EXPAND)
- [ ] Training strategy (encoder-decoder lr, warmup lr, ema, Gradient Accumulation, amp)
- [ ] Loading pretrained weight (encoder, decoder, ema)
- [ ] SuperNet Training (Knowledge Distillation)
- [x] NAS Search
- [ ] Retrain