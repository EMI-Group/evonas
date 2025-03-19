# paper3_code

2025/3/16 update MambaVision-T + MambaVision for encoder-decoder model with new skip connection

MambaVision-T(c=[80,160,320,640]) + MambaVision(d=[1,1,1,1]) + 新skip

|           Model           | Param(M) |   d1   |
| :-----------------------: | :------: | :----: |
| MambaVision-T+MambaVision |   61.5   | 0.9630 |
|    Swin-T+MambaVision     |    88    | 0.9664 |

# TODO
- [ ] Search Space (mamba)
- [ ] SuperNet Training (Knowledge Distillation)
- [ ] NAS Search
- [ ] Retrain