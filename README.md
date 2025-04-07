# paper3_code

2025/3/16 update MambaVision-T + MambaVision for encoder-decoder model with new skip connection

2025/3/26 update VSSD-T + MambaVision for encoder-decoder model

2025/3/28 update VSSD-T + NeWCRFs for encoder-decoder model

|           Model           | Param(M) |   d1   |
| :-----------------------: | :------: | :----: |
|    Swin-T+MambaVision     |    64    | 0.9664 |
| MambaVision-T+MambaVision |    62    | 0.9630 |
|      MILA+MambaVision     |    47    | 0.9675 |
|    VSSD-T+MambaVision     |    46    | 0.9673 |
|   *VSSD-T+NeWCRFs(c/2)*   |    45    | 0.9675 |

# TODO
- [ ] Search Space (VSSD)
    - [ ] add choose (mlp_ratio, dim)
- [ ] SuperNet Training (Knowledge Distillation)
- [ ] NAS Search
- [ ] Retrain